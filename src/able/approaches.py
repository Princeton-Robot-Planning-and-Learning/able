"""Approaches for the two-link arm and the limb environment.

An approach is a choice of planner, learner, belief updater, and information
gatherer. Each function here builds a robot for a domain. The recorder, if given,
sees what the robot imagines while it sleeps.
"""

import dataclasses

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.belief_updaters.range_of_motion import RangeOfMotionBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.friction.model import FrictionModel, FrictionState, FrictionTask
from able.envs.friction.query import FrictionQuery, PushRecord
from able.envs.friction.tasks import FrictionDomain
from able.envs.friction.world import Push
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.envs.limb.tasks import LimbDomain
from able.envs.two_link_arm import ArmDomain, ArmTask, JointQuery, TwoLinkArmModel
from able.info_gatherers.base import NullInformationGatherer
from able.info_gatherers.probe_floor import ProbeFloorGatherer
from able.info_gatherers.push_to_limit import PushToLimitGatherer
from able.learners.base import NullLearner
from able.learners.belief_sources import (
    CurrentBeliefSource,
    ImaginedNextBeliefSource,
    PriorSource,
)
from able.learners.model_learner import ModelLearner
from able.models.base import ConditionalModel
from able.models.exact_match_cache import ExactMatchCache
from able.planners.mpc import LimbMPCPlanner, MPCParams
from able.planners.pomcpow import (
    FrictionPOMCPOWPlanner,
    POMCPOWConfig,
    POMCPOWParams,
)
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
FrictionRobot = Robot[
    FrictionState,
    Push,
    FrictionTask,
    ArmArray,
    POMCPOWParams,
    FrictionQuery,
    PushRecord,
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


def _make_limb_parts(
    domain: LimbDomain,
    num_rollouts: int,
    num_particles: int,
    model: ConditionalModel | None = None,
) -> tuple[LimbModel, LimbMPCPlanner, RangeOfMotionBeliefUpdater, PushToLimitGatherer]:
    """The limb robot's model of the environment, planner, belief updater, and
    gatherer. MPC scores num_rollouts candidates per step, constrains the limb to
    the range that half of num_particles belief samples allow, and is seeded by the
    model, if given."""
    env_model = domain.env_model
    assert isinstance(env_model, LimbModel)
    default = MPCParams()
    sampling = dataclasses.replace(default.config.sampling, num_rollouts=num_rollouts)
    params = MPCParams(
        config=dataclasses.replace(default.config, sampling=sampling),
        num_particles=num_particles,
        model=default.model if model is None else model,
    )
    belief_updater = RangeOfMotionBeliefUpdater(env_model.variant)
    return (
        env_model,
        LimbMPCPlanner(env_model, params),
        belief_updater,
        PushToLimitGatherer(env_model.variant, belief_updater),
    )


def make_limb_scratch_robot(
    domain: LimbDomain,
    num_rollouts: int = 32,
    num_particles: int = 32,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A robot that plans with MPC, asks the person to push to their limit, and
    never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles
    )
    return Robot(
        planner=planner,
        learner=NullLearner(),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_limb_current_belief_robot(
    domain: LimbDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    num_rollouts: int = 32,
    num_particles: int = 32,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model, while it sleeps, learns plans for sampled tasks
    under its current belief. It never anticipates belief updates."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, model
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            domain.task_distribution,
            CurrentBeliefSource(),
            planning_budget,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_limb_prior_only_robot(
    domain: LimbDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    num_rollouts: int = 32,
    num_particles: int = 32,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model learns plans for sampled tasks under the prior in
    its first sleep, and never learns again."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, model
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            domain.task_distribution,
            PriorSource(domain.initial_belief),
            planning_budget,
            refit="per_sleep",
            max_sleeps=1,
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_limb_dreaming_robot(
    domain: LimbDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    num_rollouts: int = 32,
    num_particles: int = 32,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot that dreams while it sleeps: its model learns plans for sampled
    tasks under the beliefs it imagines holding at evaluation.

    The budgets should match the cycle budgets for information gathering, task
    execution, and planning.
    """
    env_model, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, model
    )
    belief_source: ImaginedNextBeliefSource = ImaginedNextBeliefSource(
        env_model,
        domain.task_distribution,
        planner,
        belief_updater,
        gatherer,
        gathering_budget=gathering_budget,
        execution_budget=execution_budget,
        planning_budget=planning_budget,
        recorder=recorder,
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            domain.task_distribution,
            belief_source,
            planning_budget,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def _make_friction_parts(
    domain: FrictionDomain, num_sims: int, max_depth: int
) -> tuple[
    FrictionModel, FrictionPOMCPOWPlanner, FrictionBeliefUpdater, ProbeFloorGatherer
]:
    """The friction robot's model of the environment, planner, belief updater, and
    gatherer. POMCPOW runs num_sims simulations per push, max_depth pushes deep."""
    env_model = domain.env_model
    assert isinstance(env_model, FrictionModel)
    params = POMCPOWParams(POMCPOWConfig(num_sims=num_sims, max_depth=max_depth))
    belief_updater = FrictionBeliefUpdater(env_model.world)
    return (
        env_model,
        FrictionPOMCPOWPlanner(env_model, belief_updater, default_params=params),
        belief_updater,
        ProbeFloorGatherer(env_model.world, belief_updater),
    )


def make_friction_scratch_robot(
    domain: FrictionDomain,
    num_sims: int = 1000,
    max_depth: int = 8,
    recorder: CycleRecorder[FrictionState, FrictionTask, ArmArray] | None = None,
) -> FrictionRobot:
    """A robot that plans each push with POMCPOW, probes the least certain floor,
    and never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_friction_parts(
        domain, num_sims, max_depth
    )
    return Robot(
        planner=planner,
        learner=NullLearner(),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )
