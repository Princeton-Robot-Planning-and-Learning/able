"""Tests for cycle.py."""

import numpy as np

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.cycle import CycleBudgets, evaluate, run_cycles
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, TwoLinkArmModel, make_domain
from able.info_gatherers.task_execution import TaskExecutionGatherer
from able.learners.base import NullLearner
from able.planners.random_shooting import RandomShootingPlanner
from able.robot import Robot
from able.structs import Budget

TRUE_LIMITS = np.array([-1.2, 1.4, -2.0, 0.9])


def _make_robot(domain):
    planner = RandomShootingPlanner(TwoLinkArmModel())
    return Robot(
        planner=planner,
        learner=NullLearner(),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=TaskExecutionGatherer(
            planner, domain.task_distribution, Budget()
        ),
        initial_belief=domain.initial_belief,
    )


def test_evaluate():
    """Tests that evaluation records the outcome and leaves the robot unchanged."""
    domain = make_domain(seed=0, true_limits=TRUE_LIMITS)
    robot = _make_robot(domain)
    rng = np.random.default_rng(0)
    belief, params = robot.belief, robot.planner_params
    task = domain.task_distribution.sample(rng)

    result = evaluate(robot, task, domain.real_env, Budget(), rng)
    assert result.failure is None
    assert result.task is task
    assert result.trajectory is not None
    assert len(result.trajectory) == TwoLinkArmModel().horizon
    assert np.isclose(result.total_cost, result.trajectory.total_cost)
    assert result.planning_wall_time > 0
    assert result.planning_stats["num_samples"] > 0
    assert robot.belief is belief
    assert robot.planner_params is params

    result = evaluate(robot, task, domain.real_env, Budget(max_iterations=0), rng)
    assert result.failure is not None
    assert result.trajectory is None
    assert result.total_cost == float("inf")


def test_run_cycles():
    """Tests several cycles end to end with a learner that does nothing."""
    domain = make_domain(seed=0, edge_probability=1.0, true_limits=TRUE_LIMITS)
    robot = _make_robot(domain)
    rng = np.random.default_rng(0)
    eval_tasks = [domain.task_distribution.sample(rng) for _ in range(4)]
    budgets = CycleBudgets(
        sleep=Budget(),
        information_gathering=Budget(max_iterations=1),
        planning=Budget(),
    )
    results = run_cycles(
        robot, domain.real_env, eval_tasks, num_cycles=3, budgets=budgets, rng=rng
    )
    assert [result.cycle for result in results] == [0, 1, 2]
    previous_width = IntervalBelief(JOINT_LIMIT_PRIOR).get_total_width()
    for result in results:
        assert len(result.gathered_data) == 1
        assert [e.task for e in result.evaluations] == eval_tasks
        assert all(e.failure is None for e in result.evaluations)
        belief = result.belief
        assert isinstance(belief, IntervalBelief)
        assert belief.contains(TRUE_LIMITS)
        assert belief.get_total_width() <= previous_width
        previous_width = belief.get_total_width()
    assert results[-1].belief is robot.belief
