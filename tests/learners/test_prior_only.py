"""Tests for learners/prior_only.py."""

import numpy as np

from able.approaches import make_prior_only_robot
from able.beliefs.interval import IntervalBelief
from able.cycle import CycleBudgets, run_cycles
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmTask,
    TwoLinkArmModel,
    make_domain,
)
from able.learners.prior_only import PriorOnlyLearner
from able.planners.random_shooting import RandomShootingPlanner, get_warm_start_key
from able.structs import Budget


def test_trains_once_under_prior():
    """Tests that the first call caches plans under the prior, whatever belief it is
    given, and later calls change nothing."""
    tasks = [
        ArmTask(initial_state=np.zeros(2), goal=np.array(goal))
        for goal in [(1.0, 1.0), (0.5, 1.5)]
    ]
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    planner = RandomShootingPlanner(TwoLinkArmModel())
    learner = PriorOnlyLearner(prior, FiniteTaskDistribution(tasks), planner, Budget())
    params = planner.get_default_params()
    narrower = IntervalBelief(JOINT_LIMIT_PRIOR / 2)
    rng = np.random.default_rng(0)

    trained = learner.learn(narrower, params, Budget(max_iterations=2), rng)
    assert set(trained.warm_starts) == {
        get_warm_start_key(task, prior, params.key_resolution) for task in tasks
    }
    assert learner.learn(narrower, trained, Budget(max_iterations=2), rng) is trained


def test_cached_plans_only_help_under_the_prior():
    """Tests that an evaluation uses a cached plan exactly when the belief still has the
    prior's key."""
    domain = make_domain(seed=0, edge_probability=1.0)
    prior = domain.initial_belief
    assert isinstance(prior, IntervalBelief)
    task_distribution = domain.task_distribution
    assert isinstance(task_distribution, FiniteTaskDistribution)
    robot = make_prior_only_robot(domain, Budget())
    rng = np.random.default_rng(0)
    eval_tasks = [task_distribution.sample(rng) for _ in range(4)]
    budgets = CycleBudgets(
        sleep=Budget(max_iterations=2),
        information_gathering=Budget(max_iterations=0),
        execution=Budget(max_iterations=1),
        planning=Budget(),
    )
    results = run_cycles(
        robot, domain.real_env, task_distribution, eval_tasks, 3, budgets, rng
    )
    resolution = robot.planner_params.key_resolution
    num_hits = 0
    for result in results:
        belief = result.belief
        assert isinstance(belief, IntervalBelief)
        for evaluation in result.evaluations:
            key = get_warm_start_key(evaluation.task, belief, resolution)
            prior_key = get_warm_start_key(evaluation.task, prior, resolution)
            hit = evaluation.planning_stats["used_warm_start"] == 1.0
            assert hit == (key == prior_key)
            num_hits += hit
    # With every goal at a limit, the first execution changes the belief.
    assert num_hits == 0
