"""Tests for cycle.py."""

import dataclasses

import numpy as np
import pytest

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.cycle import CycleBudgets, evaluate, execute, run_cycles
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, TwoLinkArmModel, make_domain
from able.info_gatherers.base import NullInformationGatherer
from able.learners.base import Learner, NullLearner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.recording.base import Stage
from able.robot import Robot
from able.structs import Budget
from tests.recording.event_recorder import EventRecorder

TRUE_LIMITS = np.array([-1.2, 1.4, -2.0, 0.9])


class _OneBatchLearner(Learner[np.ndarray, RandomShootingParams]):
    """Limits the planner to one batch of samples."""

    def learn(
        self,
        belief: Belief[np.ndarray],
        params: RandomShootingParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> RandomShootingParams:
        """Return parameters with one batch."""
        return dataclasses.replace(params, max_batches=1)


def _make_robot(domain, learner=None):
    return Robot(
        planner=RandomShootingPlanner(TwoLinkArmModel()),
        learner=NullLearner() if learner is None else learner,
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=domain.initial_belief,
    )


def _budgets(num_executions: int | None) -> CycleBudgets:
    return CycleBudgets(
        sleep=Budget(),
        information_gathering=Budget(max_iterations=0),
        execution=Budget(max_iterations=num_executions),
        planning=Budget(),
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
    assert result.success == domain.real_env.check_goal(
        task, result.trajectory.states[-1]
    )
    assert result.planning_wall_time > 0
    assert result.planning_stats["num_samples"] > 0
    assert robot.belief is belief
    assert robot.planner_params is params

    result = evaluate(robot, task, domain.real_env, Budget(max_iterations=0), rng)
    assert result.failure is not None
    assert not result.success
    assert result.trajectory is None
    assert result.total_cost == float("inf")


def test_execute():
    """Tests that execution updates the belief during the episode."""
    domain = make_domain(seed=0, edge_probability=1.0, true_limits=TRUE_LIMITS)
    robot = _make_robot(domain)
    rng = np.random.default_rng(0)
    params = robot.planner_params
    task = domain.task_distribution.sample(rng)

    result = execute(robot, task, domain.real_env, Budget(), rng)
    assert result.failure is None
    assert result.trajectory is not None
    belief = robot.belief
    assert isinstance(belief, IntervalBelief)
    assert belief.contains(TRUE_LIMITS)
    assert (
        belief.get_total_width() < IntervalBelief(JOINT_LIMIT_PRIOR).get_total_width()
    )
    assert robot.planner_params is params

    # A planning failure happens before any step, so the belief is unchanged.
    result = execute(robot, task, domain.real_env, Budget(max_iterations=0), rng)
    assert result.failure is not None
    assert robot.belief is belief


def test_run_cycles():
    """Tests several cycles end to end with a learner that does nothing."""
    domain = make_domain(seed=0, edge_probability=1.0, true_limits=TRUE_LIMITS)
    robot = _make_robot(domain)
    rng = np.random.default_rng(0)
    eval_tasks = [domain.task_distribution.sample(rng) for _ in range(4)]
    results = run_cycles(
        robot,
        domain.real_env,
        domain.task_distribution,
        eval_tasks,
        num_cycles=3,
        budgets=_budgets(num_executions=2),
        rng=rng,
    )
    assert [result.cycle for result in results] == [0, 1, 2]
    previous_width = IntervalBelief(JOINT_LIMIT_PRIOR).get_total_width()
    for result in results:
        assert not result.gathered_data
        assert len(result.executions) == 2
        assert all(e.failure is None for e in result.executions)
        assert [e.task for e in result.evaluations] == eval_tasks
        assert all(e.failure is None for e in result.evaluations)
        belief = result.belief
        assert isinstance(belief, IntervalBelief)
        assert belief.contains(TRUE_LIMITS)
        assert belief.get_total_width() <= previous_width
        previous_width = belief.get_total_width()
    assert previous_width < IntervalBelief(JOINT_LIMIT_PRIOR).get_total_width()
    assert results[-1].belief is robot.belief

    with pytest.raises(AssertionError):
        run_cycles(
            robot,
            domain.real_env,
            domain.task_distribution,
            eval_tasks,
            num_cycles=1,
            budgets=_budgets(num_executions=None),
            rng=rng,
        )


def test_executed_tasks_do_not_depend_on_planning():
    """Tests that robots that consume different amounts of randomness while planning
    execute the same tasks."""
    domain = make_domain(seed=0)
    task_sequences = []
    for learner in (NullLearner(), _OneBatchLearner()):
        results = run_cycles(
            _make_robot(domain, learner),
            domain.real_env,
            domain.task_distribution,
            eval_tasks=[],
            num_cycles=3,
            budgets=_budgets(num_executions=2),
            rng=np.random.default_rng(0),
        )
        task_sequences.append([id(e.task) for r in results for e in r.executions])
    assert len(task_sequences[0]) == 6
    assert task_sequences[0] == task_sequences[1]


def test_run_cycles_records_every_stage():
    """Tests the events a recorder sees in one cycle."""
    domain = make_domain(seed=0, true_limits=TRUE_LIMITS)
    robot = _make_robot(domain)
    eval_tasks = [domain.task_distribution.sample(np.random.default_rng(0))] * 2
    recorder = EventRecorder()
    results = run_cycles(
        robot,
        domain.real_env,
        domain.task_distribution,
        eval_tasks,
        num_cycles=1,
        budgets=_budgets(num_executions=1),
        rng=np.random.default_rng(0),
        recorder=recorder,
    )
    horizon = TwoLinkArmModel().horizon
    episode = ["start_episode"] + ["record_step"] * horizon + ["end_episode"]
    assert recorder.get_names() == (
        ["start_stage", "end_stage"]
        + ["start_stage", "record_queries", "end_stage"]
        + ["start_stage", *episode, "end_stage"]
        + ["start_stage", *episode, *episode, "end_stage"]
    )
    stages = [e[2] for e in recorder.events if e[0] == "start_stage"]
    assert stages == list(Stage)
    labels = [e[1] for e in recorder.events if e[0] == "start_episode"]
    assert labels == [
        "Executed task 1",
        "Evaluation task 1 of 2",
        "Evaluation task 2 of 2",
    ]
    outcomes = [e[1] for e in recorder.events if e[0] == "end_episode"]
    assert all("planning time" in outcome for outcome in outcomes)

    # Steps during execution see the updated belief, and the last one matches the
    # belief the robot was evaluated under.
    first_step = recorder.get_names().index("start_episode") + 1
    execution_steps = recorder.events[first_step : first_step + horizon]
    assert execution_steps[-1][1] is results[0].belief
