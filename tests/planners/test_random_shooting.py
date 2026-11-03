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
from able.planners.random_shooting import (
    RandomShootingParams,
    RandomShootingPlanner,
    add_warm_start,
    get_warm_start_key,
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


def test_default_params():
    """Tests that the planner starts from the given parameters, or from
    RandomShootingParams() if none are given."""
    model = TwoLinkArmModel()
    defaults = RandomShootingPlanner(model).get_default_params()
    assert defaults.key_resolution == RandomShootingParams().key_resolution
    assert not defaults.warm_starts
    params = RandomShootingParams(key_resolution=0.2, threshold=0.04)
    assert RandomShootingPlanner(model, params).get_default_params() is params


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
    actions = np.array([policy.get_action(np.zeros(2)) for _ in range(model.horizon)])
    warm_starts: dict = {}
    key = get_warm_start_key(task, belief, params.key_resolution)
    add_warm_start(
        warm_starts, key, policy.get_planning_stats()["expected_cost"], actions
    )
    warm_params = dataclasses.replace(params, warm_starts=warm_starts)

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


def test_warm_start_helpers():
    """Tests get_warm_start_key() and add_warm_start()."""
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    task = ArmTask(initial_state=np.zeros(2), goal=np.array([0.81, 1.19]))
    nearby = ArmTask(initial_state=np.zeros(2), goal=np.array([0.79, 1.21]))
    distant = ArmTask(initial_state=np.zeros(2), goal=np.array([0.5, 1.2]))
    key = get_warm_start_key(task, belief, 0.1)
    assert get_warm_start_key(nearby, belief, 0.1) == key
    assert get_warm_start_key(distant, belief, 0.1) != key

    warm_starts: dict = {}
    actions = np.zeros((15, 2))
    for cost in (0.5, 0.1, 0.4, 0.3, 0.2):
        add_warm_start(warm_starts, key, cost, actions, max_per_key=4)
    assert [cost for cost, _ in warm_starts[key]] == [0.1, 0.2, 0.3, 0.4]
    actions[0, 0] = 1.0
    assert warm_starts[key][0][1][0, 0] == 0.0
