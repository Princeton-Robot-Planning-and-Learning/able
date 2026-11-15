"""Approaches for the two-link arm and the limb environment.

An approach is a choice of planner, learner, belief updater, and information
gatherer. Each function here builds a robot for a domain. The recorder, if given,
sees what the robot imagines while it sleeps.
"""

import dataclasses
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.belief_updaters.human_reach import ReachBeliefUpdater
from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.belief_updaters.range_of_motion import RangeOfMotionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.friction.model import FrictionModel, FrictionState, FrictionTask
from able.envs.friction.query import FrictionQuery, PushRecord
from able.envs.friction.tasks import FrictionDomain
from able.envs.friction.world import Push
from able.envs.human_reach.model import ReachAction, ReachModel, ReachState, ReachTask
from able.envs.human_reach.query import ReachQuery, ReachResponse
from able.envs.human_reach.tasks import ReachDomain
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.envs.limb.tasks import LimbDomain
from able.envs.two_link_arm import ArmDomain, ArmTask, JointQuery, TwoLinkArmModel
from able.info_gatherers.base import NullInformationGatherer
from able.info_gatherers.probe_floor import ProbeFloorGatherer
from able.info_gatherers.push_to_limit import PushToLimitGatherer
from able.info_gatherers.reach_query import ReachQueryGatherer
from able.learners.base import NullLearner
from able.learners.belief_features import make_moment_features
from able.learners.belief_sources import (
    BeliefCoverageSource,
    CurrentBeliefSource,
    ImaginedNextBeliefSource,
    PriorSource,
)
from able.learners.model_learner import ModelLearner
from able.learners.rollout_model_learner import RolloutModelLearner
from able.models.base import ConditionalModel
from able.models.exact_match_cache import ExactMatchCache
from able.models.kernel_regression import KernelRegressor
from able.planners.mpc import LimbMPCPlanner, MPCParams
from able.planners.pddlstream import PDDLStreamParams, ReachPDDLStreamPlanner
from able.planners.pomcpow import (
    EarlyStopConfig,
    FrictionPOMCPOWPlanner,
    POMCPOWConfig,
    POMCPOWParams,
    ProposalConfig,
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
ReachRobot = Robot[
    ReachState,
    ReachAction,
    ReachTask,
    ArmArray,
    PDDLStreamParams,
    ReachQuery,
    ReachResponse,
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


def _make_arm_dreaming_parts(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    max_warm_starts_per_key: int,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None,
) -> tuple[
    FiniteTaskDistribution[ArmTask],
    RandomShootingPlanner,
    JointLimitBeliefUpdater,
    NullInformationGatherer,
    ImaginedNextBeliefSource,
]:
    """The task distribution, caching planner, belief updater, and gatherer of
    an arm robot that dreams, and the belief source that dreams with them."""
    env_model = domain.env_model
    task_distribution = domain.task_distribution
    assert isinstance(env_model, TwoLinkArmModel)
    assert isinstance(task_distribution, FiniteTaskDistribution)
    planner = _make_caching_planner(env_model, max_warm_starts_per_key)
    belief_updater = JointLimitBeliefUpdater()
    info_gatherer: NullInformationGatherer = NullInformationGatherer()
    dreamer: ImaginedNextBeliefSource = ImaginedNextBeliefSource(
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
    return task_distribution, planner, belief_updater, info_gatherer, dreamer


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
    task_distribution, planner, belief_updater, info_gatherer, dreamer = (
        _make_arm_dreaming_parts(
            domain,
            gathering_budget,
            execution_budget,
            planning_budget,
            max_warm_starts_per_key,
            recorder,
        )
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(planner, task_distribution, dreamer, planning_budget),
        belief_updater=belief_updater,
        info_gatherer=info_gatherer,
        initial_belief=domain.initial_belief,
    )


def get_cache_key_features(
    resolution: float,
) -> Callable[[Belief[ArmArray]], ArmArray]:
    """A function from an interval belief to its part of the plan cache's keys
    at the resolution, as features.

    Plans for the same task under beliefs with the same features share a key, so
    they are cached together.
    """

    def get_features(belief: Belief[ArmArray]) -> ArmArray:
        assert isinstance(belief, IntervalBelief), "Cache keys need IntervalBelief"
        return np.array(belief.get_discretized_key(resolution), dtype=np.float64)

    return get_features


def make_belief_coverage_robot(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    num_candidates: int = 4,
    max_warm_starts_per_key: int = 4,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
) -> ArmRobot:
    """A robot that plans with random shooting and, while it sleeps, caches
    plans for dreamed beliefs unlike those it has already trained under.

    Each training belief is the farthest of num_candidates dreams. The distance
    between two beliefs is the largest difference between their parts of the
    plan cache's keys, so a dream that shares a key with a trained belief is at
    distance zero. The budgets should match the cycle budgets for information
    gathering, task execution, and planning.
    """
    task_distribution, planner, belief_updater, info_gatherer, dreamer = (
        _make_arm_dreaming_parts(
            domain,
            gathering_budget,
            execution_budget,
            planning_budget,
            max_warm_starts_per_key,
            recorder,
        )
    )
    cache = planner.get_default_params().model
    assert isinstance(cache, ExactMatchCache)
    belief_source: BeliefCoverageSource = BeliefCoverageSource(
        dreamer, get_cache_key_features(cache.key_resolution), num_candidates
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner, task_distribution, belief_source, planning_budget
        ),
        belief_updater=belief_updater,
        info_gatherer=info_gatherer,
        initial_belief=domain.initial_belief,
    )


def _make_dreamer(
    domain: Any,
    env_model: Any,
    planner: Any,
    belief_updater: Any,
    gatherer: Any,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    recorder: CycleRecorder[Any, Any, ArmArray] | None,
) -> ImaginedNextBeliefSource:
    """A dreamer that imagines the robot's gathering and execution until its next
    evaluation, with the robot's own parts."""
    return ImaginedNextBeliefSource(
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


def _make_coverage_source(
    dreamer: ImaginedNextBeliefSource, prior: Belief[ArmArray], num_candidates: int
) -> BeliefCoverageSource:
    """Each training belief is the farthest of num_candidates dreams, in each
    parameter's mean and standard deviation in units of the prior's."""
    return BeliefCoverageSource(dreamer, make_moment_features(prior), num_candidates)


def _make_limb_parts(
    domain: LimbDomain,
    num_rollouts: int,
    num_particles: int,
    model: ConditionalModel | None = None,
    seed_tolerance: float = 0.0,
) -> tuple[LimbModel, LimbMPCPlanner, RangeOfMotionBeliefUpdater, PushToLimitGatherer]:
    """The limb robot's model of the environment, planner, belief updater, and
    gatherer. MPC scores num_rollouts candidates per step, constrains the limb to
    the range that half of num_particles belief samples allow, and is seeded by the
    model, if given. A step the model proposes is taken if it costs at most
    1 + seed_tolerance times the best directed candidate."""
    env_model = domain.env_model
    assert isinstance(env_model, LimbModel)
    default = MPCParams()
    sampling = dataclasses.replace(default.config.sampling, num_rollouts=num_rollouts)
    params = MPCParams(
        config=dataclasses.replace(default.config, sampling=sampling),
        num_particles=num_particles,
        model=default.model if model is None else model,
        seed=dataclasses.replace(default.seed, tolerance=seed_tolerance),
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
    seed_tolerance: float = 0.0,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model, while it sleeps, learns plans for sampled tasks
    under its current belief. It never anticipates belief updates."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, model, seed_tolerance
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
    seed_tolerance: float = 0.0,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model learns plans for sampled tasks under the prior in
    its first sleep, and never learns again."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, model, seed_tolerance
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
    seed_tolerance: float = 0.0,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot that dreams while it sleeps: its model learns plans for sampled
    tasks under the beliefs it imagines holding at evaluation.

    The budgets should match the cycle budgets for information gathering, task
    execution, and planning.
    """
    env_model, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, model, seed_tolerance
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


def make_limb_belief_coverage_robot(
    domain: LimbDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_candidates: int = 4,
    num_training_tasks: int = 1,
    num_rollouts: int = 32,
    num_particles: int = 32,
    seed_tolerance: float = 0.0,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model learns, while it sleeps, under dreamed beliefs
    unlike those it has already trained under: each training belief is the
    farthest of num_candidates dreams. The model is refit once per sleep, as for
    dreaming."""
    env_model, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, model, seed_tolerance
    )
    dreamer = _make_dreamer(
        domain,
        env_model,
        planner,
        belief_updater,
        gatherer,
        gathering_budget,
        execution_budget,
        planning_budget,
        recorder,
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            domain.task_distribution,
            _make_coverage_source(dreamer, domain.initial_belief, num_candidates),
            planning_budget,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def _make_friction_parts(
    domain: FrictionDomain,
    num_sims: int,
    max_depth: int,
    model: ConditionalModel | None = None,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
) -> tuple[
    FrictionModel, FrictionPOMCPOWPlanner, FrictionBeliefUpdater, ProbeFloorGatherer
]:
    """The friction robot's model of the environment, planner, belief updater, and
    gatherer. POMCPOW runs up to num_sims simulations per push, max_depth pushes
    deep, and stops early by the early_stop rule, if given. Training tasks planned
    while sleeping run training_num_sims simulations per push, if given, instead.
    The model, if given, proposes pushes as the proposal config says, and the value
    model, if given, corrects the leaf values."""
    env_model = domain.env_model
    assert isinstance(env_model, FrictionModel)
    config = POMCPOWConfig(
        num_sims=num_sims, max_depth=max_depth, early_stop=early_stop
    )
    params = POMCPOWParams(
        config, training_num_sims=training_num_sims, value_model=value_model
    )
    if model is not None:
        params = dataclasses.replace(params, model=model)
    if proposal is not None:
        params = dataclasses.replace(params, proposal=proposal)
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
    early_stop: EarlyStopConfig | None = None,
    recorder: CycleRecorder[FrictionState, FrictionTask, ArmArray] | None = None,
) -> FrictionRobot:
    """A robot that plans each push with POMCPOW, probes the least certain floor,
    and never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_friction_parts(
        domain, num_sims, max_depth, early_stop=early_stop
    )
    return Robot(
        planner=planner,
        learner=NullLearner(),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_friction_current_belief_robot(
    domain: FrictionDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    num_sims: int = 1000,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    recorder: CycleRecorder[FrictionState, FrictionTask, ArmArray] | None = None,
) -> FrictionRobot:
    """A friction robot whose model, while it sleeps, learns the pushes of solved
    rollouts under its current belief. It never anticipates belief updates."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    env_model, planner, belief_updater, gatherer = _make_friction_parts(
        domain,
        num_sims,
        max_depth,
        model,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
    )
    return Robot(
        planner=planner,
        learner=RolloutModelLearner(
            planner,
            domain.task_distribution,
            CurrentBeliefSource(),
            planning_budget,
            env_model,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_friction_prior_only_robot(
    domain: FrictionDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    num_sims: int = 1000,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    recorder: CycleRecorder[FrictionState, FrictionTask, ArmArray] | None = None,
) -> FrictionRobot:
    """A friction robot whose model learns the pushes of solved rollouts under the
    prior in its first sleep, and never learns again."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    env_model, planner, belief_updater, gatherer = _make_friction_parts(
        domain,
        num_sims,
        max_depth,
        model,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
    )
    return Robot(
        planner=planner,
        learner=RolloutModelLearner(
            planner,
            domain.task_distribution,
            PriorSource(domain.initial_belief),
            planning_budget,
            env_model,
            refit="per_sleep",
            max_sleeps=1,
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_friction_dreaming_robot(
    domain: FrictionDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    num_sims: int = 1000,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    recorder: CycleRecorder[FrictionState, FrictionTask, ArmArray] | None = None,
) -> FrictionRobot:
    """A friction robot that dreams while it sleeps: its model learns the pushes of
    solved rollouts under the beliefs it imagines holding at evaluation.

    The budgets should match the cycle budgets for information gathering, task
    execution, and planning.
    """
    env_model, planner, belief_updater, gatherer = _make_friction_parts(
        domain,
        num_sims,
        max_depth,
        model,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
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
        learner=RolloutModelLearner(
            planner,
            domain.task_distribution,
            belief_source,
            planning_budget,
            env_model,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_friction_belief_coverage_robot(
    domain: FrictionDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_candidates: int = 4,
    num_training_tasks: int = 1,
    num_sims: int = 1000,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    recorder: CycleRecorder[FrictionState, FrictionTask, ArmArray] | None = None,
) -> FrictionRobot:
    """A friction robot whose model learns, while it sleeps, under dreamed beliefs
    unlike those it has already trained under: each training belief is the
    farthest of num_candidates dreams. The model is refit once per sleep, as for
    dreaming."""
    env_model, planner, belief_updater, gatherer = _make_friction_parts(
        domain,
        num_sims,
        max_depth,
        model,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
    )
    dreamer = _make_dreamer(
        domain,
        env_model,
        planner,
        belief_updater,
        gatherer,
        gathering_budget,
        execution_budget,
        planning_budget,
        recorder,
    )
    return Robot(
        planner=planner,
        learner=RolloutModelLearner(
            planner,
            domain.task_distribution,
            _make_coverage_source(dreamer, domain.initial_belief, num_candidates),
            planning_budget,
            env_model,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def _make_reach_parts(
    domain: ReachDomain,
    max_time: float,
    num_particles: int,
    delta: float,
    model: ConditionalModel | None = None,
    max_memory_gb: float | None = None,
    guided_budget_fraction: float = 1.0,
    placement: str = "continuous",
) -> tuple[ReachModel, ReachPDDLStreamPlanner, ReachBeliefUpdater, ReachQueryGatherer]:
    """The human reach robot's model of the environment, planner, belief updater,
    and gatherer. PDDLStream plans for up to max_time seconds, and a solve fails if
    it grows the process's memory by more than max_memory_gb, if set. A try the
    model guides gets at most guided_budget_fraction of the time limit and of the
    planning budget's stream calls. A pose is usable if at least 1 - delta of
    num_particles belief particles reach it.
    Placements come from the slot grid or anywhere on a surface (placement, "slots"
    or "continuous"), and the model, if given, proposes them."""
    env_model = domain.env_model
    assert isinstance(env_model, ReachModel)
    params = PDDLStreamParams(
        max_time=max_time,
        num_particles=num_particles,
        delta=delta,
        max_memory_gb=max_memory_gb,
        guided_budget_fraction=guided_budget_fraction,
        placement=placement,
    )
    if model is not None:
        params = dataclasses.replace(params, model=model)
    belief_updater = ReachBeliefUpdater(env_model.sim.shoulder)
    return (
        env_model,
        ReachPDDLStreamPlanner(env_model, params),
        belief_updater,
        ReachQueryGatherer(belief_updater, env_model.config),
    )


def make_human_reach_scratch_robot(
    domain: ReachDomain,
    max_time: float = 300.0,
    num_particles: int = 256,
    delta: float = 0.02,
    max_memory_gb: float | None = None,
    placement: str = "continuous",
    recorder: CycleRecorder[ReachState, ReachTask, ArmArray] | None = None,
) -> ReachRobot:
    """A robot that plans with PDDLStream, asks the person to reach for the gripper,
    and never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_reach_parts(
        domain,
        max_time,
        num_particles,
        delta,
        max_memory_gb=max_memory_gb,
        placement=placement,
    )
    return Robot(
        planner=planner,
        learner=NullLearner(),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_human_reach_current_belief_robot(
    domain: ReachDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    max_time: float = 300.0,
    num_particles: int = 256,
    delta: float = 0.02,
    max_memory_gb: float | None = None,
    guided_budget_fraction: float = 1.0,
    placement: str = "continuous",
    recorder: CycleRecorder[ReachState, ReachTask, ArmArray] | None = None,
) -> ReachRobot:
    """A human reach robot whose model, while it sleeps, learns plans for sampled
    scenes under its current belief. It never anticipates belief updates."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_reach_parts(
        domain,
        max_time,
        num_particles,
        delta,
        model,
        max_memory_gb,
        guided_budget_fraction,
        placement,
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


def make_human_reach_prior_only_robot(
    domain: ReachDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    max_time: float = 300.0,
    num_particles: int = 256,
    delta: float = 0.02,
    max_memory_gb: float | None = None,
    guided_budget_fraction: float = 1.0,
    placement: str = "continuous",
    recorder: CycleRecorder[ReachState, ReachTask, ArmArray] | None = None,
) -> ReachRobot:
    """A human reach robot whose model learns plans for sampled scenes under the
    prior in its first sleep, and never learns again."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_reach_parts(
        domain,
        max_time,
        num_particles,
        delta,
        model,
        max_memory_gb,
        guided_budget_fraction,
        placement,
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


def make_human_reach_dreaming_robot(
    domain: ReachDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 1,
    max_time: float = 300.0,
    num_particles: int = 256,
    delta: float = 0.02,
    max_memory_gb: float | None = None,
    guided_budget_fraction: float = 1.0,
    placement: str = "continuous",
    recorder: CycleRecorder[ReachState, ReachTask, ArmArray] | None = None,
) -> ReachRobot:
    """A human reach robot that dreams while it sleeps: its model learns plans for
    sampled scenes under the beliefs it imagines holding at evaluation.

    The budgets should match the cycle budgets for information gathering, task
    execution, and planning.
    """
    env_model, planner, belief_updater, gatherer = _make_reach_parts(
        domain,
        max_time,
        num_particles,
        delta,
        model,
        max_memory_gb,
        guided_budget_fraction,
        placement,
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


def make_human_reach_belief_coverage_robot(
    domain: ReachDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_candidates: int = 4,
    num_training_tasks: int = 1,
    max_time: float = 300.0,
    num_particles: int = 256,
    delta: float = 0.02,
    max_memory_gb: float | None = None,
    guided_budget_fraction: float = 1.0,
    placement: str = "continuous",
    recorder: CycleRecorder[ReachState, ReachTask, ArmArray] | None = None,
) -> ReachRobot:
    """A human reach robot whose model learns, while it sleeps, under dreamed beliefs
    unlike those it has already trained under: each training belief is the
    farthest of num_candidates dreams. The model is refit once per sleep, as for
    dreaming."""
    env_model, planner, belief_updater, gatherer = _make_reach_parts(
        domain,
        max_time,
        num_particles,
        delta,
        model,
        max_memory_gb,
        guided_budget_fraction,
        placement,
    )
    dreamer = _make_dreamer(
        domain,
        env_model,
        planner,
        belief_updater,
        gatherer,
        gathering_budget,
        execution_budget,
        planning_budget,
        recorder,
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            domain.task_distribution,
            _make_coverage_source(dreamer, domain.initial_belief, num_candidates),
            planning_budget,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        ),
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )
