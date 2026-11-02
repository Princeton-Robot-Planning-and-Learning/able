"""Tests for robot.py."""

import dataclasses

import numpy as np

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.base import Belief
from able.envs.two_link_arm import TwoLinkArmModel, make_domain
from able.info_gatherers.joint_limits import WidestGapGatherer
from able.learners.base import Learner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.robot import Robot
from able.structs import Budget


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


def test_robot():
    """Tests that each robot method changes only what it should."""
    domain = make_domain(seed=0, edge_probability=1.0)
    planner = RandomShootingPlanner(TwoLinkArmModel())
    robot = Robot(
        planner=planner,
        learner=_BatchSizeLearner(),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=WidestGapGatherer(),
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
    data = robot.gather_information(domain.real_env, Budget(max_iterations=2), rng)
    assert len(data) == 2
    assert robot.belief is domain.initial_belief
    robot.update_belief(data)
    assert robot.belief != domain.initial_belief
    assert robot.planner_params is params
