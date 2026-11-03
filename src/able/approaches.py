"""Approaches for the two-link arm and the limb environment.

An approach is a choice of planner, learner, belief updater, and information
gatherer. Each function here builds a robot for a domain. The recorder, if given,
sees what the robot imagines while it sleeps.
"""

import dataclasses

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.belief_updaters.range_of_motion import RangeOfMotionBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.envs.limb.tasks import LimbDomain
from able.envs.two_link_arm import ArmDomain, ArmTask, JointQuery, TwoLinkArmModel
from able.info_gatherers.base import NullInformationGatherer
from able.info_gatherers.push_to_limit import PushToLimitGatherer
from able.learners.base import NullLearner
from able.learners.belief_sources import (
    CurrentBeliefSource,
    ImaginedNextBeliefSource,
    PriorSource,
)
from able.learners.model_learner import ModelLearner
from able.models.exact_match_cache import ExactMatchCache
from able.planners.mpc import LimbMPCPlanner, MPCParams
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.recording.base import CycleRecorder
from able.robot import Robot
from able.structs import Budget

ArmArray = NDArray[np.float64]
ArmRobot = Robot[
    ArmArray, ArmArray, ArmTask, ArmArray, RandomShootingParams, JointQuery, bool
]
LimbRobot = Robot[
    LimbState, ArmArray, LimbTask, ArmArray, MPCParams, LimbQuery, QueryOutcome
]


def _make_caching_planner(
    env_model: TwoLinkArmModel, max_warm_starts_per_key: int
) -> RandomShootingPlanner:
    """Random shooting with warm starts from a cache of the cheapest plans per
    discretized (goal, belief) key."""
    params = RandomShootingParams(
        num_proposals=max_warm_starts_per_key,
        model=ExactMatchCache(max_per_key=max_warm_starts_per_key),
    )
    return RandomShootingPlanner(env_model, params)


def make_scratch_robot(
    domain: ArmDomain,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
) -> ArmRobot:
    """A robot that plans from scratch with random shooting and never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    env_model = domain.env_model
    assert isinstance(env_model, TwoLinkArmModel)
    return Robot(
        planner=RandomShootingPlanner(env_model),
        learner=NullLearner(),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=domain.initial_belief,
    )


def make_current_belief_robot(
    domain: ArmDomain,
    planning_budget: Budget,
    max_warm_starts_per_key: int = 4,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
) -> ArmRobot:
    """A robot that plans with random shooting and, while it sleeps, caches plans
    for every task under its current belief.

    It never anticipates belief updates, so it is a baseline for the dreaming
    robot.
    """
    del recorder  # unused because the robot imagines nothing while it sleeps
    env_model = domain.env_model
    task_distribution = domain.task_distribution
    assert isinstance(env_model, TwoLinkArmModel)
    assert isinstance(task_distribution, FiniteTaskDistribution)
    planner = _make_caching_planner(env_model, max_warm_starts_per_key)
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner, task_distribution, CurrentBeliefSource(), planning_budget
        ),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=domain.initial_belief,
    )


def make_prior_only_robot(
    domain: ArmDomain,
    planning_budget: Budget,
    max_warm_starts_per_key: int = 4,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
) -> ArmRobot:
    """A robot that plans with random shooting and caches plans for every task
    under the prior in its first sleep, then never learns again."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    env_model = domain.env_model
    task_distribution = domain.task_distribution
    prior = domain.initial_belief
    assert isinstance(env_model, TwoLinkArmModel)
    assert isinstance(task_distribution, FiniteTaskDistribution)
    assert isinstance(prior, IntervalBelief)
    planner = _make_caching_planner(env_model, max_warm_starts_per_key)
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            task_distribution,
            PriorSource(prior),
            planning_budget,
            max_sleeps=1,
        ),
        belief_updater=JointLimitBeliefUpdater(),
        info_gatherer=NullInformationGatherer(),
        initial_belief=prior,
    )


def make_dreaming_robot(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    max_warm_starts_per_key: int = 4,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
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
    planner = _make_caching_planner(env_model, max_warm_starts_per_key)
    belief_updater = JointLimitBeliefUpdater()
    info_gatherer: NullInformationGatherer = NullInformationGatherer()
    belief_source: ImaginedNextBeliefSource = ImaginedNextBeliefSource(
        env_model,
        task_distribution,
        planner,
        belief_updater,
        info_gatherer,
        gathering_budget=gathering_budget,
        execution_budget=execution_budget,
        planning_budget=planning_budget,
        recorder=recorder,
    )
    learner = ModelLearner(planner, task_distribution, belief_source, planning_budget)
    return Robot(
        planner=planner,
        learner=learner,
        belief_updater=belief_updater,
        info_gatherer=info_gatherer,
        initial_belief=domain.initial_belief,
    )


def make_limb_scratch_robot(
    domain: LimbDomain,
    num_rollouts: int = 32,
    num_particles: int = 32,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A robot that plans with MPC, asks the person to push to their limit, and
    never learns.

    MPC scores num_rollouts candidates per step and constrains the limb to the
    range that half of num_particles belief samples allow.
    """
    del recorder  # unused because the robot imagines nothing while it sleeps
    env_model = domain.env_model
    assert isinstance(env_model, LimbModel)
    default = MPCParams()
    sampling = dataclasses.replace(default.config.sampling, num_rollouts=num_rollouts)
    params = MPCParams(
        config=dataclasses.replace(default.config, sampling=sampling),
        num_particles=num_particles,
    )
    belief_updater = RangeOfMotionBeliefUpdater(env_model.variant)
    return Robot(
        planner=LimbMPCPlanner(env_model, params),
        learner=NullLearner(),
        belief_updater=belief_updater,
        info_gatherer=PushToLimitGatherer(env_model.variant, belief_updater),
        initial_belief=domain.initial_belief,
    )
