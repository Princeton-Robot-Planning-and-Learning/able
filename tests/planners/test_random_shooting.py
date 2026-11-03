"""Tests for planners/random_shooting.py."""

import dataclasses

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmTask,
    TwoLinkArmModel,
    make_domain,
)
from able.models.exact_match_cache import ExactMatchCache
from able.planners.random_shooting import (
    RandomShootingParams,
    RandomShootingPlanner,
    get_plan_features,
)
from able.structs import Budget, PlanningFailure

TRUE_LIMITS = np.array([-2.0, 2.5, -2.2, 2.4])
GOAL = np.array([0.8, 1.2])


def _known_belief(limits: np.ndarray) -> IntervalBelief:
    return IntervalBelief(np.stack([limits, limits], axis=1))


def _execute(model, policy, task, limits) -> tuple[np.ndarray, float]:
    """Run the policy for one episode and return the final state and total cost."""
    env = SimulatedEnvironment(model, limits, np.random.default_rng(0))
    state = env.reset(task)
    total_cost = 0.0
    for _ in range(model.horizon):
        result = env.step(policy.get_action(state))
        state = result.state
        total_cost += result.cost
    return state, total_cost


def test_plan_reaches_goal_with_known_limits():
    """Tests that a plan gets close to a reachable goal when the limits are known, and
    that executing it incurs the cost the planner expected."""
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    task = ArmTask(initial_state=np.zeros(2), goal=GOAL)
    belief = _known_belief(TRUE_LIMITS)
    params = planner.get_default_params()
    policy = planner.plan(task, belief, params, Budget(), np.random.default_rng(0))
    stats = policy.get_planning_stats()
    assert stats["used_warm_start"] == 0.0
    final_state, total_cost = _execute(model, policy, task, TRUE_LIMITS)
    assert np.isclose(total_cost, stats["expected_cost"])
    initial_distance = np.linalg.norm(model.forward_kinematics(np.zeros(2)) - GOAL)
    final_distance = np.linalg.norm(model.forward_kinematics(final_state) - GOAL)
    assert initial_distance > 1.5
    assert final_distance < 0.25


def test_warm_start_reduces_samples():
    """Tests that planning from a cached plan needs fewer samples than planning from
    scratch."""
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    task = ArmTask(initial_state=np.zeros(2), goal=GOAL)
    belief = _known_belief(TRUE_LIMITS)
    params = planner.get_default_params()

    scratch_samples = []
    for seed in range(5):
        policy = planner.plan(
            task, belief, params, Budget(), np.random.default_rng(seed)
        )
        scratch_samples.append(policy.get_planning_stats()["num_samples"])
    assert all(n > params.batch_size for n in scratch_samples)

    policy = planner.plan(task, belief, params, Budget(), np.random.default_rng(0))
    examples = planner.get_plan_examples(task, belief, policy)
    warm_params = dataclasses.replace(params, model=params.model.fit(examples))

    policy = planner.plan(
        task, belief, warm_params, Budget(), np.random.default_rng(10)
    )
    stats = policy.get_planning_stats()
    assert stats["used_warm_start"] == 1.0
    # One batch: the cached plan plus one perturbed copy per batch slot.
    assert stats["num_samples"] == 1 + params.batch_size
    assert stats["num_samples"] < min(scratch_samples)


def test_plan_under_uncertainty():
    """Tests planning under the prior belief."""
    domain = make_domain(seed=0)
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    policy = planner.plan(
        task,
        IntervalBelief(JOINT_LIMIT_PRIOR),
        planner.get_default_params(),
        Budget(),
        np.random.default_rng(0),
    )
    for _ in range(model.horizon):
        action = policy.get_action(np.zeros(2))
        assert action.shape == (2,)
        assert np.all(np.abs(action) <= model.max_joint_displacement)
    with pytest.raises(PlanningFailure):
        policy.get_action(np.zeros(2))


def test_budget():
    """Tests that the planner respects the iteration budget."""
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    task = ArmTask(initial_state=np.zeros(2), goal=np.array([1.9, 0.0]))
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    params = planner.get_default_params()
    rng = np.random.default_rng(0)
    policy = planner.plan(task, belief, params, Budget(max_iterations=1), rng)
    assert policy.get_planning_stats()["num_samples"] == params.batch_size
    with pytest.raises(PlanningFailure):
        planner.plan(task, belief, params, Budget(max_iterations=0), rng)


def test_get_plan_features():
    """Tests that nearby goals share a cache key and distant goals do not."""
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    task = ArmTask(initial_state=np.zeros(2), goal=np.array([0.81, 1.19]))
    nearby = ArmTask(initial_state=np.zeros(2), goal=np.array([0.79, 1.21]))
    distant = ArmTask(initial_state=np.zeros(2), goal=np.array([0.5, 1.2]))
    features = get_plan_features(task, belief)
    assert features.shape == (2 + 4 + 4,)
    assert np.array_equal(features[:2], task.goal)
    assert np.allclose(features[2:6], JOINT_LIMIT_PRIOR.mean(axis=1))
    cache = ExactMatchCache(key_resolution=0.1)
    key = cache.get_key(features)
    assert cache.get_key(get_plan_features(nearby, belief)) == key
    assert cache.get_key(get_plan_features(distant, belief)) != key
    narrower = IntervalBelief(JOINT_LIMIT_PRIOR / 2)
    assert cache.get_key(get_plan_features(task, narrower)) != key


def test_get_plan_examples():
    """Tests that a plan becomes one row that the model proposes back."""
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    task = ArmTask(initial_state=np.zeros(2), goal=GOAL)
    belief = _known_belief(TRUE_LIMITS)
    params = planner.get_default_params()
    policy = planner.plan(task, belief, params, Budget(), np.random.default_rng(0))
    examples = planner.get_plan_examples(task, belief, policy)
    assert len(examples) == 1
    assert np.array_equal(examples.x[0], get_plan_features(task, belief))
    assert np.array_equal(examples.y[0], np.array(policy.actions).ravel())
    assert examples.costs[0] == policy.get_planning_stats()["expected_cost"]
    proposals = params.model.fit(examples).sample(
        examples.x[0], params.num_proposals, np.random.default_rng(0)
    )
    assert np.array_equal(proposals, examples.y)


def test_default_params():
    """Tests that the planner returns the default parameters it was given."""
    model = TwoLinkArmModel()
    default = RandomShootingPlanner(model).get_default_params()
    assert default.batch_size == RandomShootingParams().batch_size
    assert isinstance(default.model, ExactMatchCache) and not default.model.entries
    params = RandomShootingParams(batch_size=8)
    assert RandomShootingPlanner(model, params).get_default_params() is params
