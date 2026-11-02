"""Tests for learners/dreaming.py."""

import numpy as np
import pytest

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.cycle import CycleBudgets, run_cycles
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import ArmTask, TwoLinkArmModel, make_domain
from able.info_gatherers.joint_limits import WidestGapGatherer
from able.learners.base import Learner, NullLearner
from able.learners.dreaming import DreamingLearner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.robot import Robot
from able.structs import Budget


def _make_robot(
    task_distribution: FiniteTaskDistribution[ArmTask],
    initial_belief: IntervalBelief,
    dreaming: bool,
) -> Robot:
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    updater = JointLimitBeliefUpdater()
    gatherer = WidestGapGatherer()
    learner: Learner[np.ndarray, RandomShootingParams]
    if dreaming:
        learner = DreamingLearner(
            model,
            task_distribution,
            planner,
            updater,
            gatherer,
            gathering_budget=Budget(max_iterations=1),
            execution_budget=Budget(max_iterations=1),
            planning_budget=Budget(),
        )
    else:
        learner = NullLearner()
    return Robot(planner, learner, updater, gatherer, initial_belief)


def _run(robot: Robot, domain, num_dreams: int, num_cycles: int, seed: int) -> list:
    """Run cycles and return the evaluation planning statistics."""
    assert isinstance(domain.task_distribution, FiniteTaskDistribution)
    budgets = CycleBudgets(
        sleep=Budget(max_iterations=num_dreams),
        information_gathering=Budget(max_iterations=1),
        execution=Budget(max_iterations=1),
        planning=Budget(),
    )
    results = run_cycles(
        robot,
        domain.real_env,
        domain.task_distribution,
        list(domain.task_distribution.get_tasks()),
        num_cycles,
        budgets,
        np.random.default_rng(seed),
    )
    return [e.planning_stats for r in results for e in r.evaluations]


def test_learn_adds_warm_starts():
    """Tests that dreaming adds warm starts without changing its inputs."""
    domain = make_domain(seed=0)
    assert isinstance(domain.task_distribution, FiniteTaskDistribution)
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    updater = JointLimitBeliefUpdater()
    learner = DreamingLearner(
        model,
        domain.task_distribution,
        planner,
        updater,
        WidestGapGatherer(),
        gathering_budget=Budget(max_iterations=1),
        execution_budget=Budget(max_iterations=1),
        planning_budget=Budget(),
        max_warm_starts_per_key=2,
    )
    params = planner.get_default_params()
    belief = domain.initial_belief
    new_params = learner.learn(
        belief, params, Budget(max_iterations=3), np.random.default_rng(0)
    )
    assert not params.warm_starts
    assert new_params.warm_starts
    assert all(1 <= len(entries) <= 2 for entries in new_params.warm_starts.values())
    assert new_params.batch_size == params.batch_size
    assert belief == domain.initial_belief

    with pytest.raises(AssertionError):
        learner.learn(belief, params, Budget(), np.random.default_rng(0))


def test_known_limits_always_hit():
    """Tests that dreamed plans are used when the next belief is predictable.

    With known limits, every imagined next belief equals the real one, so every
    evaluation should start from a cached plan.
    """
    true_limits = np.array([-1.2, 1.4, -2.0, 0.9])
    domain = make_domain(seed=0, true_limits=true_limits)
    assert isinstance(domain.task_distribution, FiniteTaskDistribution)
    known = IntervalBelief(np.stack([true_limits, true_limits], axis=1))
    dream_stats = _run(
        _make_robot(domain.task_distribution, known, dreaming=True),
        domain,
        num_dreams=1,
        num_cycles=2,
        seed=0,
    )
    null_stats = _run(
        _make_robot(domain.task_distribution, known, dreaming=False),
        domain,
        num_dreams=1,
        num_cycles=2,
        seed=0,
    )
    assert all(stats["used_warm_start"] == 1.0 for stats in dream_stats)
    assert all(stats["used_warm_start"] == 0.0 for stats in null_stats)
    assert np.mean([s["num_samples"] for s in dream_stats]) < np.mean(
        [s["num_samples"] for s in null_stats]
    )


def test_dreaming_reduces_planning_effort():
    """Tests that dreaming reduces evaluation planning effort under uncertainty."""
    dream_samples, null_samples, hits = [], [], []
    for seed in range(3):
        domain = make_domain(seed)
        assert isinstance(domain.task_distribution, FiniteTaskDistribution)
        for dreaming in (True, False):
            robot = _make_robot(
                domain.task_distribution, domain.initial_belief, dreaming
            )
            stats = _run(robot, domain, num_dreams=8, num_cycles=4, seed=seed)
            samples = [s["num_samples"] for s in stats]
            if dreaming:
                dream_samples.extend(samples)
                hits.extend(s["used_warm_start"] for s in stats)
            else:
                null_samples.extend(samples)
    assert np.mean(hits) > 0.5
    assert np.mean(dream_samples) < 0.7 * np.mean(null_samples)
