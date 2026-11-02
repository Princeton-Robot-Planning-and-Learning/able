"""Approaches for the two-link arm.

An approach is a choice of planner, learner, belief updater, and information
gatherer. Each function here builds a robot for an arm domain.
"""

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import ArmDomain, ArmTask, JointQuery, TwoLinkArmModel
from able.info_gatherers.base import NullInformationGatherer
from able.learners.base import NullLearner
from able.learners.dreaming import DreamingLearner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.robot import Robot
from able.structs import Budget

ArmArray = NDArray[np.float64]
ArmRobot = Robot[
    ArmArray, ArmArray, ArmTask, ArmArray, RandomShootingParams, JointQuery, bool
]


def make_scratch_robot(domain: ArmDomain) -> ArmRobot:
    """A robot that plans from scratch with random shooting and never learns."""
    env_model = domain.env_model
    assert isinstance(env_model, TwoLinkArmModel)
    return Robot(
        planner=RandomShootingPlanner(env_model),
        learner=NullLearner(),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=domain.initial_belief,
    )


def make_dreaming_robot(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    max_warm_starts_per_key: int = 4,
) -> ArmRobot:
    """A robot that plans with random shooting and dreams while it sleeps.

    The budgets describe what the robot does between sleeping and evaluation, so
    they should match the cycle budgets for information gathering, task
    execution, and planning.
    """
    env_model = domain.env_model
    task_distribution = domain.task_distribution
    assert isinstance(env_model, TwoLinkArmModel)
    assert isinstance(task_distribution, FiniteTaskDistribution)
    planner = RandomShootingPlanner(env_model)
    belief_updater = JointLimitBeliefUpdater()
    info_gatherer: NullInformationGatherer = NullInformationGatherer()
    learner = DreamingLearner(
        env_model,
        task_distribution,
        planner,
        belief_updater,
        info_gatherer,
        gathering_budget=gathering_budget,
        execution_budget=execution_budget,
        planning_budget=planning_budget,
        max_warm_starts_per_key=max_warm_starts_per_key,
    )
    return Robot(
        planner=planner,
        learner=learner,
        belief_updater=belief_updater,
        info_gatherer=info_gatherer,
        initial_belief=domain.initial_belief,
    )
