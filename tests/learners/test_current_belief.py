"""Tests for learners/current_belief.py."""

import numpy as np

from able.approaches import (
    make_current_belief_robot,
    make_dreaming_robot,
    make_scratch_robot,
)
from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.cycle import CycleBudgets, run_cycles
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmTask,
    TwoLinkArmModel,
    make_domain,
)
from able.info_gatherers.base import NullInformationGatherer
from able.learners.current_belief import CurrentBeliefLearner
from able.models.exact_match_cache import ExactMatchCache
from able.planners.random_shooting import RandomShootingPlanner, get_plan_features
from able.robot import Robot
from able.structs import Budget

BUDGETS = CycleBudgets(
    sleep=Budget(max_iterations=8),
    information_gathering=Budget(max_iterations=0),
    execution=Budget(max_iterations=1),
    planning=Budget(),
)


def _get_stats(robot, domain, num_cycles: int, seed: int) -> list[dict[str, float]]:
    """Run cycles and return the planning statistics of every evaluation."""
    task_distribution = domain.task_distribution
    assert isinstance(task_distribution, FiniteTaskDistribution)
    results = run_cycles(
        robot,
        domain.real_env,
        task_distribution,
        list(task_distribution.get_tasks()),
        num_cycles,
        BUDGETS,
        np.random.default_rng(seed),
    )
    return [e.planning_stats for r in results for e in r.evaluations]


def test_caches_plans_for_current_belief():
    """Tests that the cached plans are keyed on the current belief."""
    tasks = [
        ArmTask(initial_state=np.zeros(2), goal=np.array(goal))
        for goal in [(1.0, 1.0), (0.5, 1.5)]
    ]
    planner = RandomShootingPlanner(TwoLinkArmModel())
    learner = CurrentBeliefLearner(FiniteTaskDistribution(tasks), planner, Budget())
    params = planner.get_default_params()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    new_params = learner.learn(
        belief, params, Budget(max_iterations=2), np.random.default_rng(0)
    )
    cache = new_params.model
    assert isinstance(cache, ExactMatchCache)
    assert set(cache.entries) == {
        cache.get_key(get_plan_features(task, belief)) for task in tasks
    }


def test_known_limits_always_hit():
    """Tests that every evaluation uses a cached plan when the belief cannot change."""
    true_limits = np.array([-1.2, 1.4, -2.0, 0.9])
    domain = make_domain(seed=0, true_limits=true_limits)
    task_distribution = domain.task_distribution
    assert isinstance(task_distribution, FiniteTaskDistribution)
    planner = RandomShootingPlanner(TwoLinkArmModel())
    robot = Robot(
        planner=planner,
        learner=CurrentBeliefLearner(task_distribution, planner, Budget()),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=IntervalBelief(np.stack([true_limits, true_limits], axis=1)),
    )
    stats = _get_stats(robot, domain, num_cycles=2, seed=0)
    assert all(s["used_warm_start"] == 1.0 for s in stats)


def test_anticipation_beats_current_belief():
    """Tests that dreaming hits the cache more often than training on the current
    belief, which in turn plans with fewer samples than planning from scratch."""
    hits: dict[str, list[float]] = {"current_belief": [], "dreaming": []}
    samples: dict[str, list[float]] = {"scratch": [], "current_belief": []}
    for seed in range(3):
        domain = make_domain(seed)
        robots = {
            "scratch": make_scratch_robot(domain),
            "current_belief": make_current_belief_robot(domain, BUDGETS.planning),
            "dreaming": make_dreaming_robot(
                domain,
                BUDGETS.information_gathering,
                BUDGETS.execution,
                BUDGETS.planning,
            ),
        }
        for name, robot in robots.items():
            stats = _get_stats(robot, domain, num_cycles=4, seed=seed)
            if name in hits:
                hits[name].extend(s["used_warm_start"] for s in stats)
            if name in samples:
                samples[name].extend(s["num_samples"] for s in stats)
    assert 0 < np.mean(hits["current_belief"]) < np.mean(hits["dreaming"])
    assert np.mean(samples["current_belief"]) < np.mean(samples["scratch"])
