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
from able.belief_updaters.sites import MultiSiteUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.friction.model import FrictionModel, FrictionState, FrictionTask
from able.envs.friction.pomdp import PuckPushingModel
from able.envs.friction.query import FrictionQuery, PushRecord
from able.envs.friction.sites import (
    FrictionSiteDomain,
    count_route_floors,
    make_route_usage,
)
from able.envs.friction.tasks import FrictionDomain, make_prior_belief
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, Push
from able.envs.human_reach.model import ReachAction, ReachModel, ReachState, ReachTask
from able.envs.human_reach.query import ReachQuery, ReachResponse
from able.envs.human_reach.tasks import ReachDomain
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.envs.limb.tasks import LimbDomain
from able.envs.multi_person_arm import (
    MultiPersonArmModel,
    MultiPersonDomain,
    MultiPersonLimitUpdater,
    RoutineTaskDistribution,
    ScheduleTaskDistribution,
    get_person_plan_features,
)
from able.envs.multi_site import (
    ImaginedDay,
    MultiSiteModel,
    SiteRoutineTaskDistribution,
)
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmDomain,
    ArmTask,
    JointQuery,
    TwoLinkArmModel,
)
from able.info_gatherers.base import NullInformationGatherer
from able.info_gatherers.bisect_limits import BisectLimitGatherer
from able.info_gatherers.probe_floor import ProbeFloorGatherer
from able.info_gatherers.push_to_limit import PushToLimitGatherer
from able.info_gatherers.reach_query import ReachQueryGatherer
from able.info_gatherers.sites import MultiSiteGatherer
from able.learners.allocation import (
    AllocationModel,
    AllocationSource,
    get_cliff_gain,
    get_diminishing_gain,
    get_ramp_gain,
    get_saturating_gain,
    make_moment_transfer,
)
from able.learners.base import NullLearner
from able.learners.belief_features import make_moment_features
from able.learners.belief_sources import (
    BeliefCoverageSource,
    CurrentBeliefSource,
    ImaginedNextBeliefSource,
    PriorSource,
)
from able.learners.model_learner import ModelLearner
from able.learners.person_focus import (
    PersonFocusSource,
    PersonModelLearner,
    get_same_person_transfer,
    make_person_features,
    make_person_key_features,
    make_person_moment_transfer,
    make_person_revision_features,
    make_person_routine_features,
    make_routine_transfer,
    make_same_person_transfer,
)
from able.learners.rollout_model_learner import RolloutModelLearner
from able.learners.site_focus import (
    SiteFocusSource,
    SiteNullLearner,
    SiteRolloutModelLearner,
    make_site_key_features,
    make_site_revision_features,
    make_site_usage_features,
    make_usage_transfer,
)
from able.models.base import ConditionalModel
from able.models.exact_match_cache import ExactMatchCache
from able.models.kernel_regression import KernelRegressor
from able.planners.mpc import LimbMPCPlanner, MPCParams, SeedConfig
from able.planners.pddlstream import PDDLStreamParams, ReachPDDLStreamPlanner
from able.planners.pomcpow import (
    EarlyStopConfig,
    FrictionPOMCPOWPlanner,
    POMCPOWConfig,
    POMCPOWParams,
    ProposalConfig,
)
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.planners.sites import MultiSitePlanner
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


# The goal's entries in a plan cache key, which a nearest-key lookup must match.
_GOAL_KEY_DIMS = 2


def _make_caching_planner(
    env_model: TwoLinkArmModel, max_warm_starts_per_key: int, max_key_distance: int = 0
) -> RandomShootingPlanner:
    """Random shooting with warm starts from a cache of the cheapest plans per
    discretized (goal, belief) key. With max_key_distance above 0, a belief whose
    key has no plans for the goal uses the plans of the nearest key for the same
    goal, if each belief entry is within max_key_distance steps of resolution."""
    params = RandomShootingParams(
        num_proposals=max_warm_starts_per_key,
        model=ExactMatchCache(
            max_per_key=max_warm_starts_per_key,
            max_key_distance=max_key_distance,
            num_exact_dims=_GOAL_KEY_DIMS,
        ),
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
    max_key_distance: int = 0,
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
    planner = _make_caching_planner(
        env_model, max_warm_starts_per_key, max_key_distance
    )
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
    max_key_distance: int = 0,
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
    planner = _make_caching_planner(
        env_model, max_warm_starts_per_key, max_key_distance
    )
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
    max_key_distance: int = 0,
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
    planner = _make_caching_planner(
        env_model, max_warm_starts_per_key, max_key_distance
    )
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
    max_key_distance: int = 0,
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
            max_key_distance,
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
    max_key_distance: int = 0,
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
            max_key_distance,
        )
    )
    cache = planner.get_default_params().model
    assert isinstance(cache, ExactMatchCache)
    assert isinstance(cache.key_resolution, float), "Need one resolution"
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


def make_key_distance_transfer(
    transfer_by_distance: list[float],
) -> Callable[[ArmArray, ArmArray], ArmArray]:
    """Transfer between beliefs from the largest difference between their cache
    keys' belief entries: 1 at distance 0, transfer_by_distance[d - 1] at distance
    d, and 0 beyond the list."""
    values = np.array([1.0, *transfer_by_distance])

    def transfer(a: ArmArray, b: ArmArray) -> ArmArray:
        distances = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)
        index = np.minimum(np.round(distances).astype(int), len(values))
        return np.append(values, 0.0)[index]

    return transfer


def make_allocation_robot(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    num_units: int,
    rule: str,
    dreams_per_unit: int = 4,
    max_warm_starts_per_key: int = 4,
    max_key_distance: int = 0,
    transfer_by_distance: list[float] | None = None,
    first_unit_gain: float = 0.9,
    extra_unit_decay: float = 0.7,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
) -> ArmRobot:
    """A robot that plans with random shooting and, while it sleeps, splits its
    num_units training beliefs among dreamed beliefs by an allocation rule.

    A training belief caches plans for every task under its key, so one unit
    makes a key fast, and later units add cheaper plans. The rule predicts that
    k units at a key buy 1 - (1 - first_unit_gain) * extra_unit_decay^(k - 1)
    of what a key can gain. With max_key_distance above 0,
    the cache lends a key's plans to nearby keys, and the rule predicts that a
    trained key helps a key at distance d by transfer_by_distance[d - 1], or fully
    if transfer_by_distance is None. The budgets should match the cycle budgets,
    and num_units should match the sleep budget's iterations.
    """
    task_distribution, planner, belief_updater, info_gatherer, dreamer = (
        _make_arm_dreaming_parts(
            domain,
            gathering_budget,
            execution_budget,
            planning_budget,
            max_warm_starts_per_key,
            recorder,
            max_key_distance,
        )
    )
    cache = planner.get_default_params().model
    assert isinstance(cache, ExactMatchCache)
    if transfer_by_distance is None:
        transfer_by_distance = [1.0] * max_key_distance
    assert (
        len(transfer_by_distance) == max_key_distance
    ), "Need a transfer for each key distance the cache lends plans across"
    assert isinstance(cache.key_resolution, float), "Need one resolution"
    model = AllocationModel(
        get_features=get_cache_key_features(cache.key_resolution),
        transfer=make_key_distance_transfer(list(transfer_by_distance)),
        gain=get_diminishing_gain(first_unit_gain, extra_unit_decay),
        combine="max",
    )
    belief_source: AllocationSource = AllocationSource(
        dreamer, model, rule, num_units, dreams_per_unit
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


def _make_sampled_arm_parts(
    domain: ArmDomain, model: ConditionalModel, num_proposals: int
) -> tuple[
    TwoLinkArmModel, RandomShootingPlanner, JointLimitBeliefUpdater, BisectLimitGatherer
]:
    """The model, random shooting planner with warm starts from the model, belief
    updater, and bisecting gatherer of a robot on the sampled arm."""
    env_model = domain.env_model
    assert isinstance(env_model, TwoLinkArmModel)
    params = RandomShootingParams(num_proposals=num_proposals, model=model)
    belief_updater = JointLimitBeliefUpdater()
    return (
        env_model,
        RandomShootingPlanner(env_model, params),
        belief_updater,
        BisectLimitGatherer(belief_updater),
    )


def _make_sampled_arm_robot(
    domain: ArmDomain,
    model: ConditionalModel,
    num_proposals: int,
    num_training_tasks: int,
    planning_budget: Budget,
    make_source: Callable[..., Any] | None,
    max_sleeps: int | None = None,
    gathering_budget: Budget | None = None,
    execution_budget: Budget | None = None,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    training_iterations: int | None = None,
) -> ArmRobot:
    """A sampled-arm robot whose learner trains under make_source(dreamer), or
    never learns if make_source is None. The model is refit once per sleep.

    Dreams plan with planning_budget, as the robot does while awake. Training
    plans get training_iterations iterations if given, since planning during
    sleep need not be as fast, and planning_budget otherwise."""
    env_model, planner, belief_updater, gatherer = _make_sampled_arm_parts(
        domain, model, num_proposals
    )
    learner: Any = NullLearner()
    if make_source is not None:
        dreamer = None
        if gathering_budget is not None and execution_budget is not None:
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
        learner = ModelLearner(
            planner,
            domain.task_distribution,
            make_source(dreamer),
            (
                planning_budget
                if training_iterations is None
                else Budget(max_iterations=training_iterations)
            ),
            max_sleeps=max_sleeps,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
        )
    return Robot(
        planner=planner,
        learner=learner,
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_sampled_arm_scratch_robot(
    domain: ArmDomain,
    model: ConditionalModel,
    num_proposals: int = 4,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
) -> ArmRobot:
    """A sampled-arm robot that plans from scratch and never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    return _make_sampled_arm_robot(domain, model, num_proposals, 1, Budget(), None)


def make_sampled_arm_current_belief_robot(
    domain: ArmDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    training_iterations: int | None = None,
) -> ArmRobot:
    """A sampled-arm robot that trains its model under its current belief."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    return _make_sampled_arm_robot(
        domain,
        model,
        num_proposals,
        num_training_tasks,
        planning_budget,
        lambda _: CurrentBeliefSource(),
        training_iterations=training_iterations,
    )


def make_sampled_arm_prior_only_robot(
    domain: ArmDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    training_iterations: int | None = None,
) -> ArmRobot:
    """A sampled-arm robot that trains its model under the prior in its first
    sleep, then never learns again."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    return _make_sampled_arm_robot(
        domain,
        model,
        num_proposals,
        num_training_tasks,
        planning_budget,
        lambda _: PriorSource(domain.initial_belief),
        max_sleeps=1,
        training_iterations=training_iterations,
    )


def make_sampled_arm_dreaming_robot(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    training_iterations: int | None = None,
) -> ArmRobot:
    """A sampled-arm robot that trains its model under the beliefs it dreams of
    holding at evaluation. The budgets should match the cycle budgets."""
    return _make_sampled_arm_robot(
        domain,
        model,
        num_proposals,
        num_training_tasks,
        planning_budget,
        lambda dreamer: dreamer,
        gathering_budget=gathering_budget,
        execution_budget=execution_budget,
        recorder=recorder,
        training_iterations=training_iterations,
    )


def make_sampled_arm_belief_coverage_robot(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_candidates: int = 4,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    training_iterations: int | None = None,
) -> ArmRobot:
    """A sampled-arm robot that trains under the farthest of num_candidates dreams
    from the beliefs it has trained under, by moment features. The budgets should
    match the cycle budgets."""
    return _make_sampled_arm_robot(
        domain,
        model,
        num_proposals,
        num_training_tasks,
        planning_budget,
        lambda dreamer: _make_coverage_source(
            dreamer, domain.initial_belief, num_candidates
        ),
        gathering_budget=gathering_budget,
        execution_budget=execution_budget,
        recorder=recorder,
        training_iterations=training_iterations,
    )


def make_sampled_arm_allocation_robot(
    domain: ArmDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_units: int,
    rule: str,
    dreams_per_unit: int = 4,
    transfer_length: float = 2.2,
    narrowing_rate: float = 0.2,
    gain_scale: float = 11.0,
    gain_power: float = 1.5,
    run_length: float | None = None,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    training_iterations: int | None = None,
) -> ArmRobot:
    """A sampled-arm robot that splits each sleep's num_units training beliefs
    among dreamed beliefs by an allocation rule.

    The rule predicts that k units at a belief buy 1 - (1 + k / gain_scale)^
    (-gain_power) of what planning there can gain, and that they buy a fraction
    of that at other beliefs, falling with the distance between the beliefs'
    means and with how much narrower the training belief is (see
    make_moment_transfer), as measured on this arm. A belief benefits from its
    best training belief.

    If run_length is set, the rule instead predicts a plateau, as when a tight
    planning budget makes the model's proposals useless until it has trained
    enough near a belief: a belief gains nothing until it has run_length
    effective units, which add over training beliefs weighted by transfer.

    The budgets should match the cycle budgets, and num_units should match the
    sleep budget's iterations.
    """
    transfer = make_moment_transfer(transfer_length, narrowing_rate)
    allocation_model = AllocationModel(
        get_features=make_moment_features(domain.initial_belief),
        transfer=transfer,
        gain=get_saturating_gain(gain_scale, gain_power),
        combine="max",
    )
    if run_length is not None:
        allocation_model = AllocationModel(
            get_features=make_moment_features(domain.initial_belief),
            transfer=transfer,
            gain=get_cliff_gain(run_length),
            combine="sum",
            run_length=run_length,
        )
    return _make_sampled_arm_robot(
        domain,
        model,
        num_proposals,
        num_training_tasks,
        planning_budget,
        lambda dreamer: AllocationSource(
            dreamer, allocation_model, rule, num_units, dreams_per_unit
        ),
        gathering_budget=gathering_budget,
        execution_budget=execution_budget,
        recorder=recorder,
        training_iterations=training_iterations,
    )


def _make_multi_person_robot(
    domain: MultiPersonDomain,
    model: ConditionalModel,
    planning_budget: Budget,
    make_source: Callable[[ImaginedNextBeliefSource, NDArray[np.float64]], Any] | None,
    gathering_budget: Budget | None = None,
    execution_budget: Budget | None = None,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    training_iterations: int | None = None,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    keep_only_last_sleep: bool = False,
) -> ArmRobot:
    """A robot for several people that plans with random shooting, warm-started
    from a model of plans whose input is the goal and the person's intervals.

    Its learner trains under make_source(dreamer, schedule), or never learns if
    make_source is None. Each unit of sleep trains for the person its belief
    names. Dreams plan with planning_budget, as the robot does while awake, and
    training plans get training_iterations iterations if given. The model is
    refit once per sleep, on that sleep's plans only if keep_only_last_sleep.
    """
    env_model = domain.env_model
    task_distribution = domain.task_distribution
    assert isinstance(env_model, MultiPersonArmModel)
    assert isinstance(
        task_distribution, (ScheduleTaskDistribution, RoutineTaskDistribution)
    )
    params = RandomShootingParams(num_proposals=num_proposals, model=model)
    planner = RandomShootingPlanner(
        env_model, params, get_features=get_person_plan_features
    )
    belief_updater = MultiPersonLimitUpdater()
    gatherer: NullInformationGatherer = NullInformationGatherer()
    learner: Any = NullLearner()
    if make_source is not None:
        dreamer = ImaginedNextBeliefSource(
            env_model,
            task_distribution,
            planner,
            belief_updater,
            gatherer,
            gathering_budget=Budget() if gathering_budget is None else gathering_budget,
            execution_budget=(
                Budget(max_iterations=1)
                if execution_budget is None
                else execution_budget
            ),
            planning_budget=planning_budget,
            recorder=recorder,
        )
        learner = PersonModelLearner(
            planner,
            task_distribution,
            make_source(dreamer, task_distribution.schedule),
            (
                planning_budget
                if training_iterations is None
                else Budget(max_iterations=training_iterations)
            ),
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
            keep_only_last_sleep=keep_only_last_sleep,
        )
    return Robot(
        planner=planner,
        learner=learner,
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )


def make_multi_person_scratch_robot(
    domain: MultiPersonDomain,
    model: ConditionalModel,
    num_proposals: int = 4,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
) -> ArmRobot:
    """A robot for several people that plans from scratch and never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    return _make_multi_person_robot(
        domain, model, Budget(), None, num_proposals=num_proposals
    )


def make_multi_person_current_belief_robot(
    domain: MultiPersonDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    training_iterations: int | None = None,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    keep_only_last_sleep: bool = False,
) -> ArmRobot:
    """A robot for several people that trains under its current belief, each unit
    for a person drawn with their share of the schedule."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    return _make_multi_person_robot(
        domain,
        model,
        planning_budget,
        lambda _, schedule: PersonFocusSource(CurrentBeliefSource(), schedule),
        num_proposals=num_proposals,
        num_training_tasks=num_training_tasks,
        training_iterations=training_iterations,
        keep_only_last_sleep=keep_only_last_sleep,
    )


def make_multi_person_dreaming_robot(
    domain: MultiPersonDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    training_iterations: int | None = None,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    keep_only_last_sleep: bool = False,
) -> ArmRobot:
    """A robot for several people that trains under the beliefs it dreams of
    holding at evaluation, each unit for a person drawn with their share of the
    schedule. The budgets should match the cycle budgets."""
    return _make_multi_person_robot(
        domain,
        model,
        planning_budget,
        PersonFocusSource,
        gathering_budget,
        execution_budget,
        num_proposals,
        num_training_tasks,
        training_iterations,
        recorder,
        keep_only_last_sleep=keep_only_last_sleep,
    )


def make_multi_person_belief_coverage_robot(
    domain: MultiPersonDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_candidates: int = 4,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    training_iterations: int | None = None,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    keep_only_last_sleep: bool = False,
) -> ArmRobot:
    """A robot for several people that trains under the farthest of
    num_candidates dreamed (person, belief) pairs from those it has trained
    under, by the person's index and limits' moments, and by how many times
    their routine has been revised if there are routines. The budgets should
    match the cycle budgets."""
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    features = make_person_features(prior)
    routines = domain.task_distribution
    if isinstance(routines, RoutineTaskDistribution):
        features = make_person_revision_features(prior, routines.get_num_revisions)
    return _make_multi_person_robot(
        domain,
        model,
        planning_budget,
        lambda dreamer, schedule: BeliefCoverageSource(
            PersonFocusSource(dreamer, schedule),
            features,
            num_candidates,
            remember_chosen=not keep_only_last_sleep,
        ),
        gathering_budget,
        execution_budget,
        num_proposals,
        num_training_tasks,
        training_iterations,
        recorder,
        keep_only_last_sleep=keep_only_last_sleep,
    )


def make_multi_person_allocation_robot(
    domain: MultiPersonDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_units: int,
    rule: str,
    run_length: float,
    dreams_per_unit: int = 4,
    transfer_length: float = 2.2,
    narrowing_rate: float = 0.2,
    max_key_distance: int | None = None,
    num_proposals: int = 4,
    num_training_tasks: int = 8,
    training_iterations: int | None = None,
    recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    keep_only_last_sleep: bool = False,
) -> ArmRobot:
    """A robot for several people that splits each sleep's num_units units among
    dreamed (person, belief) pairs by an allocation rule.

    The rule predicts a plateau: planning for a person gains nothing until the
    model has run_length effective units for them. Units for one person count
    toward another by the transfer between their beliefs, whoever they are:
    from their moments (see make_moment_transfer), and only between people on
    the same routine if there are routines. If max_key_distance is set, the
    transfer is instead full when the people are on the same routine and their
    plan cache keys' belief entries are within max_key_distance steps, as for a
    cache that lends plans for the same goal that far. deliberate_practice
    counts units only toward the person they trained for, by the same transfer.
    The budgets should match the cycle budgets, and num_units should match the
    sleep budget's iterations.
    """
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    moment_transfer = make_moment_transfer(transfer_length, narrowing_rate)
    routines = domain.task_distribution
    get_features = make_person_features(prior)
    transfer = make_person_moment_transfer(moment_transfer)
    independent_transfer: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ] = get_same_person_transfer
    if max_key_distance is None and isinstance(routines, RoutineTaskDistribution):
        get_features = make_person_routine_features(prior, routines.get_routine_id)
        transfer = make_routine_transfer(moment_transfer)
        independent_transfer = make_same_person_transfer(transfer)
    if max_key_distance is not None:
        cache = model
        assert isinstance(cache, ExactMatchCache), "Key distances need a plan cache"
        assert isinstance(routines, RoutineTaskDistribution), "Need routines"
        assert isinstance(cache.key_resolution, float), "Need one resolution"
        get_features = make_person_key_features(
            cache.key_resolution, routines.get_routine_id
        )
        transfer = make_routine_transfer(
            make_key_distance_transfer([1.0] * max_key_distance)
        )
        independent_transfer = make_same_person_transfer(transfer)
    allocation_model = AllocationModel(
        get_features=get_features,
        transfer=transfer,
        gain=get_cliff_gain(run_length),
        combine="sum",
        run_length=run_length,
        independent_transfer=independent_transfer,
        include_current_belief=False,
    )
    return _make_multi_person_robot(
        domain,
        model,
        planning_budget,
        lambda dreamer, schedule: AllocationSource(
            PersonFocusSource(dreamer, schedule),
            allocation_model,
            rule,
            num_units,
            dreams_per_unit,
            remember_training=not keep_only_last_sleep,
        ),
        gathering_budget,
        execution_budget,
        num_proposals,
        num_training_tasks,
        training_iterations,
        recorder,
        keep_only_last_sleep=keep_only_last_sleep,
    )


def _make_limb_parts(
    domain: LimbDomain,
    num_rollouts: int,
    num_particles: int,
    confidence: float,
    model: ConditionalModel | None = None,
    seed: SeedConfig = SeedConfig(),
) -> tuple[LimbModel, LimbMPCPlanner, RangeOfMotionBeliefUpdater, PushToLimitGatherer]:
    """The limb robot's model of the environment, planner, belief updater, and
    gatherer. MPC scores num_rollouts candidates per step, constrains the limb to
    the range that `confidence` of num_particles belief samples allow, and is
    seeded by the model, if given, as the seed config says."""
    env_model = domain.env_model
    assert isinstance(env_model, LimbModel)
    default = MPCParams()
    sampling = dataclasses.replace(default.config.sampling, num_rollouts=num_rollouts)
    params = MPCParams(
        config=dataclasses.replace(default.config, sampling=sampling),
        num_particles=num_particles,
        confidence=confidence,
        model=default.model if model is None else model,
        seed=seed,
    )
    belief_updater = RangeOfMotionBeliefUpdater(
        env_model.variant,
        env_model.query_model,
        env_model.settings.person,
        env_model.torque_noise,
    )
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
    confidence: float = 0.5,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A robot that plans with MPC, asks the person to push to their limit, and
    never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain, num_rollouts, num_particles, confidence
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
    confidence: float = 0.5,
    seed_tolerance: float = 0.0,
    num_proposals: int = 8,
    seed_commit_steps: int = 1,
    training_budget: Budget | None = None,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model, while it sleeps, learns plans for sampled tasks
    under its current belief. It never anticipates belief updates. Training plans
    get training_budget, if given, instead of planning_budget."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain,
        num_rollouts,
        num_particles,
        confidence,
        model,
        SeedConfig(num_proposals, seed_tolerance, seed_commit_steps),
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            domain.task_distribution,
            CurrentBeliefSource(),
            planning_budget if training_budget is None else training_budget,
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
    confidence: float = 0.5,
    seed_tolerance: float = 0.0,
    num_proposals: int = 8,
    seed_commit_steps: int = 1,
    training_budget: Budget | None = None,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model learns plans for sampled tasks under the prior in
    its first sleep, and never learns again. Training plans get training_budget,
    if given, instead of planning_budget."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    _, planner, belief_updater, gatherer = _make_limb_parts(
        domain,
        num_rollouts,
        num_particles,
        confidence,
        model,
        SeedConfig(num_proposals, seed_tolerance, seed_commit_steps),
    )
    return Robot(
        planner=planner,
        learner=ModelLearner(
            planner,
            domain.task_distribution,
            PriorSource(domain.initial_belief),
            planning_budget if training_budget is None else training_budget,
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
    confidence: float = 0.5,
    seed_tolerance: float = 0.0,
    num_proposals: int = 8,
    seed_commit_steps: int = 1,
    training_budget: Budget | None = None,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot that dreams while it sleeps: its model learns plans for sampled
    tasks under the beliefs it imagines holding at evaluation.

    The budgets should match the cycle budgets for information gathering, task
    execution, and planning. Training plans get training_budget, if given,
    instead of planning_budget.
    """
    env_model, planner, belief_updater, gatherer = _make_limb_parts(
        domain,
        num_rollouts,
        num_particles,
        confidence,
        model,
        SeedConfig(num_proposals, seed_tolerance, seed_commit_steps),
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
            planning_budget if training_budget is None else training_budget,
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
    confidence: float = 0.5,
    seed_tolerance: float = 0.0,
    num_proposals: int = 8,
    seed_commit_steps: int = 1,
    training_budget: Budget | None = None,
    recorder: CycleRecorder[LimbState, LimbTask, ArmArray] | None = None,
) -> LimbRobot:
    """A limb robot whose model learns, while it sleeps, under dreamed beliefs
    unlike those it has already trained under: each training belief is the
    farthest of num_candidates dreams. The model is refit once per sleep, as for
    dreaming. Training plans get training_budget, if given, instead of
    planning_budget."""
    env_model, planner, belief_updater, gatherer = _make_limb_parts(
        domain,
        num_rollouts,
        num_particles,
        confidence,
        model,
        SeedConfig(num_proposals, seed_tolerance, seed_commit_steps),
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
            planning_budget if training_budget is None else training_budget,
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
    return _make_friction_parts_for_model(
        env_model,
        num_sims,
        max_depth,
        model,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
    )


def _make_friction_parts_for_model(
    env_model: FrictionModel,
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
    """_make_friction_parts() for the friction model of a domain."""
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
    belief_updater = ReachBeliefUpdater(
        env_model.sim.shoulder, delivery_scale=env_model.delivery_scale
    )
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
    delta: float = 0.1,
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
    delta: float = 0.1,
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
    delta: float = 0.1,
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
    delta: float = 0.1,
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
    delta: float = 0.1,
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


FrictionSiteRobot = Robot[Any, Any, Any, Any, Any, Any, Any]


def make_friction_site_cache(
    num_floors: int,
    position_resolution: float = 0.1,
    mean_resolution: float = 0.01,
    std_resolution: float = 0.005,
    max_per_key: int = 4,
    max_key_distance: int = 6,
) -> ExactMatchCache:
    """A cache of pushes for the friction planner, keyed by its push features
    (see get_push_features): the puck and goal at position_resolution and the
    floor under the puck must match, and the belief's mean and standard
    deviation of each floor's friction, at mean_resolution and std_resolution,
    may differ by up to max_key_distance steps."""
    resolution = (
        [position_resolution] * 4
        + [1.0] * num_floors
        + [mean_resolution] * num_floors
        + [std_resolution] * num_floors
    )
    return ExactMatchCache(
        key_resolution=tuple(resolution),
        max_per_key=max_per_key,
        max_key_distance=max_key_distance,
        num_exact_dims=4 + num_floors,
    )


def _make_friction_site_robot(
    domain: FrictionSiteDomain,
    model: ConditionalModel | None,
    planning_budget: Budget,
    make_source: Callable[[NDArray[np.float64], Any], Any] | None,
    num_sims: int,
    max_depth: int,
    early_stop: EarlyStopConfig | None,
    proposal: ProposalConfig | None,
    training_num_sims: int | None,
    value_model: KernelRegressor | None,
    num_training_tasks: int,
    keep_only_last_sleep: bool,
    dream_budgets: tuple[Budget, Budget] | None = None,
    recorder: CycleRecorder[Any, Any, Any] | None = None,
) -> FrictionSiteRobot:
    """A robot for several friction sites that plans each site's task with
    POMCPOW under the site's belief, and probes the day's site's floors, weighted
    by how many of its routine tasks cross each. Its learner rolls out the routine
    of the site each unit of sleep names, under make_source(schedule, dreamer), or
    never learns if make_source is None. With dream_budgets (gathering,
    execution), the dreamer imagines a day at a site drawn with the schedule;
    otherwise it is None. The model is refit once per sleep, on that sleep's
    rollouts only if keep_only_last_sleep.
    """
    site_model = domain.env_model
    routines = domain.task_distribution
    assert isinstance(site_model, MultiSiteModel)
    assert isinstance(site_model.inner, FrictionModel)
    assert isinstance(routines, SiteRoutineTaskDistribution)
    _, inner_planner, inner_updater, _ = _make_friction_parts_for_model(
        site_model.inner,
        num_sims,
        max_depth,
        model,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
    )
    planner = MultiSitePlanner(inner_planner)
    updater = MultiSiteUpdater(inner_updater)
    world = site_model.inner.world
    paths = PuckPushingModel(world)

    def get_gatherer(site: int) -> ProbeFloorGatherer:
        routine = [task.inner for task in routines.get_routine(site)]
        weights = count_route_floors(paths, routine, world.puck_map.num_surfaces)
        return ProbeFloorGatherer(world, inner_updater, floor_weights=weights)

    dreamer = None
    if dream_budgets is not None:
        day = ImaginedDay(routines)
        dreamer = ImaginedNextBeliefSource(
            site_model,
            day,
            planner,
            updater,
            MultiSiteGatherer(get_gatherer, day.draw_site),
            gathering_budget=dream_budgets[0],
            execution_budget=dream_budgets[1],
            planning_budget=planning_budget,
            recorder=recorder,
        )
    learner: Any = SiteNullLearner()
    if make_source is not None:
        learner = SiteRolloutModelLearner(
            planner,
            routines,
            make_source(routines.schedule, dreamer),
            planning_budget,
            site_model,
            refit="per_sleep",
            num_training_tasks=num_training_tasks,
            keep_only_last_sleep=keep_only_last_sleep,
        )
    return Robot(
        planner=planner,
        learner=learner,
        belief_updater=updater,
        info_gatherer=MultiSiteGatherer(get_gatherer, lambda _: routines.site),
        initial_belief=domain.initial_belief,
    )


def _focus_dreams(dreamer: Any, schedule: NDArray[np.float64]) -> SiteFocusSource:
    """Dreams focused on sites drawn with the schedule. The schedule is given as
    a function so that the dreams are made in this process, where revised
    routines stay current."""
    return SiteFocusSource(dreamer, lambda: schedule)


def make_friction_site_scratch_robot(
    domain: FrictionSiteDomain,
    num_sims: int = 100,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    recorder: CycleRecorder[Any, Any, Any] | None = None,
) -> FrictionSiteRobot:
    """A robot for several friction sites that never learns."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    return _make_friction_site_robot(
        domain,
        None,
        Budget(),
        None,
        num_sims,
        max_depth,
        early_stop,
        None,
        None,
        None,
        1,
        False,
    )


def make_friction_site_current_belief_robot(
    domain: FrictionSiteDomain,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 2,
    num_sims: int = 100,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    keep_only_last_sleep: bool = False,
    recorder: CycleRecorder[Any, Any, Any] | None = None,
) -> FrictionSiteRobot:
    """A robot for several friction sites whose units each train for a site
    drawn by its share of the schedule, under the current belief."""
    del recorder  # unused because the robot imagines nothing while it sleeps
    return _make_friction_site_robot(
        domain,
        model,
        planning_budget,
        lambda schedule, _: SiteFocusSource(CurrentBeliefSource(), schedule),
        num_sims,
        max_depth,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
        num_training_tasks,
        keep_only_last_sleep,
    )


def make_friction_site_dreaming_robot(
    domain: FrictionSiteDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_training_tasks: int = 2,
    num_sims: int = 100,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    keep_only_last_sleep: bool = False,
    recorder: CycleRecorder[Any, Any, Any] | None = None,
) -> FrictionSiteRobot:
    """A robot for several friction sites whose units each train for a site
    drawn by its share of the schedule, under the belief it dreams of holding
    after a day at a site drawn the same way. The budgets should match the cycle
    budgets."""
    return _make_friction_site_robot(
        domain,
        model,
        planning_budget,
        lambda schedule, dreamer: _focus_dreams(dreamer, schedule),
        num_sims,
        max_depth,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
        num_training_tasks,
        keep_only_last_sleep,
        (gathering_budget, execution_budget),
        recorder,
    )


def make_friction_site_belief_coverage_robot(
    domain: FrictionSiteDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_candidates: int = 4,
    num_training_tasks: int = 2,
    num_sims: int = 100,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    keep_only_last_sleep: bool = False,
    recorder: CycleRecorder[Any, Any, Any] | None = None,
) -> FrictionSiteRobot:
    """A robot for several friction sites whose units each train for the
    farthest of num_candidates (site, belief) pairs from those it has trained
    for, by the site's index, the moments of its belief, and how many times its
    routine has been revised. Each candidate is a dream of the belief after a
    day at a site drawn with the schedule, focused on a site drawn the same way.
    The budgets should match the cycle budgets."""
    site_model, routines = domain.env_model, domain.task_distribution
    assert isinstance(site_model, MultiSiteModel)
    assert isinstance(site_model.inner, FrictionModel)
    assert isinstance(routines, SiteRoutineTaskDistribution)
    num_floors = site_model.inner.world.puck_map.num_surfaces
    prior = make_prior_belief(
        dataclasses.replace(DEFAULT_FRICTION_PRIOR, num_surfaces=num_floors),
        500,
        np.random.default_rng(0),
    )
    features = make_site_revision_features(prior, routines.get_num_revisions)
    return _make_friction_site_robot(
        domain,
        model,
        planning_budget,
        lambda schedule, dreamer: BeliefCoverageSource(
            _focus_dreams(dreamer, schedule),
            features,
            num_candidates,
            remember_chosen=not keep_only_last_sleep,
        ),
        num_sims,
        max_depth,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
        num_training_tasks,
        keep_only_last_sleep,
        (gathering_budget, execution_budget),
        recorder,
    )


def get_ramp_units(
    ramp: tuple[float, float],
    ramp_passes: tuple[float, float] | None,
    routine_size: int,
    num_training_tasks: int,
) -> tuple[float, float]:
    """The effective units where a ramp gain starts and ends: ramp, or with
    ramp_passes, those numbers of passes over a routine of routine_size tasks
    when each unit plans num_training_tasks of them."""
    if ramp_passes is None:
        return ramp
    units_per_pass = routine_size / num_training_tasks
    return ramp_passes[0] * units_per_pass, ramp_passes[1] * units_per_pass


def make_friction_site_allocation_robot(
    domain: FrictionSiteDomain,
    gathering_budget: Budget,
    execution_budget: Budget,
    planning_budget: Budget,
    model: ConditionalModel,
    num_units: int,
    rule: str,
    run_length: float,
    dreams_per_unit: int = 4,
    transfer_length: float = 2.2,
    narrowing_rate: float = 0.2,
    ramp_start: float = 1.0,
    ramp_passes: tuple[float, float] | None = (0.2, 0.8),
    shared_floor_transfer: float = 0.55,
    other_transfer: float = 0.18,
    num_training_tasks: int = 2,
    num_sims: int = 100,
    max_depth: int = 8,
    early_stop: EarlyStopConfig | None = None,
    proposal: ProposalConfig | None = None,
    training_num_sims: int | None = None,
    value_model: KernelRegressor | None = None,
    keep_only_last_sleep: bool = False,
    recorder: CycleRecorder[Any, Any, Any] | None = None,
) -> FrictionSiteRobot:
    """A robot for several friction sites that splits each sleep's num_units
    units among (site, belief) pairs, drawn by the schedule, by an allocation
    rule.

    With a cache from make_friction_site_cache(), the rule predicts a plateau:
    a site gains nothing until it has run_length effective units, and units
    count fully toward sites on the same routine whose beliefs are within the
    cache's lending distance. With another model, a site's gain rises linearly
    from ramp_start effective units to run_length, or, with ramp_passes, from
    and to those numbers of passes over a routine, each routine_size /
    num_training_tasks units, so longer routines need more units. A unit counts
    fully toward sites on the same routine, shared_floor_transfer toward
    routines whose paths cross the same floors, and other_transfer toward the
    rest, scaled by the cosine similarity of the routines' floor shares and
    times the transfer between the beliefs' moments (see make_moment_transfer).
    deliberate_practice counts units only toward the site they trained for, by
    the same transfer. The candidates are dreams of the belief after a day at a
    site drawn with the schedule, each focused on a site drawn the same way.
    num_units should match the sleep budget's iterations, and the budgets the
    cycle budgets.
    """
    site_model, routines = domain.env_model, domain.task_distribution
    assert isinstance(site_model, MultiSiteModel)
    assert isinstance(site_model.inner, FrictionModel)
    assert isinstance(routines, SiteRoutineTaskDistribution)
    num_floors = site_model.inner.world.puck_map.num_surfaces
    if isinstance(model, ExactMatchCache):
        belief_resolution = np.atleast_1d(model.key_resolution)[-2 * num_floors :]
        get_features = make_site_key_features(
            belief_resolution, routines.get_routine_id
        )
        transfer = make_routine_transfer(
            make_key_distance_transfer([1.0] * model.max_key_distance)
        )
    else:
        prior = make_prior_belief(
            dataclasses.replace(DEFAULT_FRICTION_PRIOR, num_surfaces=num_floors),
            500,
            np.random.default_rng(0),
        )
        get_features = make_site_usage_features(
            prior,
            routines.get_routine_id,
            make_route_usage(site_model.inner.world, routines),
        )
        transfer = make_usage_transfer(
            num_floors,
            make_moment_transfer(transfer_length, narrowing_rate),
            shared_floor_transfer,
            other_transfer,
        )
    cliff = isinstance(model, ExactMatchCache)
    allocation_model = AllocationModel(
        get_features=get_features,
        transfer=transfer,
        gain=(
            get_cliff_gain(run_length)
            if cliff
            else get_ramp_gain(
                *get_ramp_units(
                    (ramp_start, run_length),
                    ramp_passes,
                    len(routines.get_routine(0)),
                    num_training_tasks,
                )
            )
        ),
        combine="sum",
        run_length=run_length if cliff else None,
        independent_transfer=make_same_person_transfer(transfer),
        include_current_belief=False,
    )
    return _make_friction_site_robot(
        domain,
        model,
        planning_budget,
        lambda schedule, dreamer: AllocationSource(
            _focus_dreams(dreamer, schedule),
            allocation_model,
            rule,
            num_units,
            dreams_per_unit,
            remember_training=not keep_only_last_sleep,
        ),
        num_sims,
        max_depth,
        early_stop,
        proposal,
        training_num_sims,
        value_model,
        num_training_tasks,
        keep_only_last_sleep,
        (gathering_budget, execution_budget),
        recorder,
    )
