"""Tests for robot.py."""

import dataclasses

import numpy as np

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JointQuery, TwoLinkArmModel, make_domain
from able.info_gatherers.base import NullInformationGatherer
from able.learners.base import Learner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.robot import Robot
from able.structs import Budget, QueryResponse, Task, Trajectory


class _BatchSizeLearner(Learner[np.ndarray, RandomShootingParams]):
    """Increments the batch size every time it learns."""

    def learn(
        self,
        belief: Belief[np.ndarray],
        params: RandomShootingParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> RandomShootingParams:
        """Return parameters with a larger batch size."""
        return dataclasses.replace(params, batch_size=params.batch_size + 1)


class _ObservingLearner(_BatchSizeLearner):
    """Records every execution it observes."""

    def __init__(self) -> None:
        self.observed: list[tuple[Task, Trajectory, Belief[np.ndarray]]] = []

    def observe_execution(
        self, task: Task, trajectory: Trajectory, belief: Belief[np.ndarray]
    ) -> None:
        self.observed.append((task, trajectory, belief))


def test_robot():
    """Tests that each robot method changes only what it should."""
    domain = make_domain(seed=0, edge_probability=1.0)
    planner = RandomShootingPlanner(TwoLinkArmModel())
    robot = Robot(
        planner=planner,
        learner=_BatchSizeLearner(),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=domain.initial_belief,
    )
    rng = np.random.default_rng(0)
    assert robot.belief is domain.initial_belief
    initial_params = robot.planner_params
    assert initial_params.batch_size == RandomShootingParams().batch_size

    task = domain.task_distribution.sample(rng)
    robot.plan(task, Budget(), rng)
    assert robot.belief is domain.initial_belief
    assert robot.planner_params is initial_params

    robot.sleep(Budget(), rng)
    assert robot.belief is domain.initial_belief
    assert robot.planner_params.batch_size == initial_params.batch_size + 1

    params = robot.planner_params
    assert not robot.gather_information(domain.real_env, Budget(), rng)
    assert robot.belief is domain.initial_belief
    # The joint-limit updater ignores query responses.
    robot.update_belief_from_queries(
        [QueryResponse(JointQuery(joint=0, angle=1.0), True)]
    )
    assert robot.belief is domain.initial_belief

    # Moving joint 0 to 0.6 shows that its upper limit is at least 0.6.
    robot.update_belief_from_transition(
        np.zeros(2), np.array([0.6, 0.0]), np.array([0.6, 0.0])
    )
    belief = robot.belief
    assert isinstance(belief, IntervalBelief)
    assert belief.intervals[1, 0] == 0.6
    assert robot.planner_params is params


def test_robot_observe_execution():
    """Tests that the robot passes executions to its learner with its current
    belief, without changing its belief or parameters."""
    domain = make_domain(seed=0)
    learner = _ObservingLearner()
    robot = Robot(
        planner=RandomShootingPlanner(TwoLinkArmModel()),
        learner=learner,
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=domain.initial_belief,
    )
    params = robot.planner_params
    task = domain.task_distribution.sample(np.random.default_rng(0))
    trajectory = Trajectory(states=(np.zeros(2),), actions=(), costs=())
    robot.observe_execution(task, trajectory)
    assert len(learner.observed) == 1
    observed_task, observed_trajectory, observed_belief = learner.observed[0]
    assert observed_task is task
    assert observed_trajectory is trajectory
    assert observed_belief is robot.belief is domain.initial_belief
    assert robot.planner_params is params
