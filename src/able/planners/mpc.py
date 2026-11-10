"""Predictive-sampling MPC for the limb environment.

Predictive sampling chooses controls one step at a time: it samples candidates
around the shifted previous plan, scores them with rollouts, and commits the start
of the best one. The limb planner samples limits from the belief, moves a goal
beyond them to just inside, and runs predictive sampling in its own copy of the
simulator, with rollouts in a second copy. The physics does not depend on the
person, so the controls MPC commits are replayed open loop. The planner replays them
first to check that the plan reaches the goal and breaks no constraint.
"""

import dataclasses
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic

import numpy as np
from kinder.envs.dynamic3d.limb_utils import NUM_ROBOT_JOINTS
from numpy.typing import NDArray

from able.belief_updaters.range_of_motion import FLOOR_STEP
from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.constraints import (
    arm_in_collision,
    exceeds_robot_torque_limits,
    get_robot_motion_violation,
    limb_in_collision,
)
from able.envs.limb.human import LimbVariant
from able.envs.limb.model import STOP, LimbModel, LimbState, LimbTask
from able.envs.limb.sim import CoupledState, LimbSim
from able.models.base import ConditionalModel, Dataset
from able.models.flow_matching import FlowMatchingModel
from able.planners.base import TrainablePlanner
from able.policies.base import OpenLoopPolicy, Policy
from able.structs import Budget, PlanningFailure, PlanningTimeout, StateType

# How far inside the believed limits a retargeted goal goes (rad), per attempt.
RETARGET_MARGIN = 0.05


class NoFeasibleControls(PlanningFailure):
    """Raised when every candidate breaks a constraint."""


@dataclass(frozen=True)
class PredictiveSamplingConfig:
    """Settings of predictive sampling.

    A candidate is num_control_points control points, interpolated linearly to one
    control per step of the horizon. The noise scale shrinks with the distance to
    the goal, from noise_scale at noise_reference_error down to min_noise_ratio of
    it.
    """

    num_rollouts: int = 32
    horizon: int = 12
    num_control_points: int = 4
    commit_steps: int = 1
    noise_scale: float = 0.12
    noise_reference_error: float = 1.2
    min_noise_ratio: float = 0.3


@dataclass(frozen=True)
class SeedConfig:
    """How predictive sampling uses a seeder.

    Up to num_rollouts proposals (the first is the seed) are scored before the full
    batch. The first one that reaches the goal, or costs at most (1 + tolerance)
    times the best of the problem's extra candidates, is committed.
    """

    num_rollouts: int = 8
    tolerance: float = 0.0


# Proposes up to num per-step control sequences, each (horizon, dim), for a state
# and step.
Seeder = Callable[[Any, int, int], list[NDArray[np.float64]]]


def get_control_basis(horizon: int, num_control_points: int) -> NDArray[np.float64]:
    """The (horizon, num_control_points) matrix that interpolates control points to
    per-step controls."""
    source = np.linspace(0.0, 1.0, num_control_points)
    target = np.linspace(0.0, 1.0, horizon)
    identity = np.eye(num_control_points)
    return np.stack(
        [np.interp(target, source, identity[k]) for k in range(num_control_points)],
        axis=1,
    )


class RolloutProblem(ABC, Generic[StateType]):
    """What predictive sampling optimizes: rollouts of controls from a state."""

    @abstractmethod
    def get_state(self) -> StateType:
        """The current state, to roll out from."""

    @abstractmethod
    def get_error(self) -> float:
        """The current distance to the goal."""

    @abstractmethod
    def rollout(
        self, state: StateType, controls: NDArray[np.float64]
    ) -> tuple[bool, float]:
        """Roll per-step controls with shape (horizon, dim) out from the state.

        Returns whether the rollout broke a constraint, and its cost.
        """

    def get_extra_candidates(
        self, state: StateType, noise_scale: float, rng: np.random.Generator
    ) -> list[NDArray[np.float64]]:
        """Problem-specific candidates as control points, such as controls that
        point at the goal. The default has none."""
        del state, noise_scale, rng  # unused by default
        return []

    def is_near_goal(self) -> bool:
        """Whether the last rollout ended close enough to count as reaching the
        goal. The default is never."""
        return False


class PredictiveSampling(Generic[StateType]):
    """Chooses controls one step at a time by sampling and scoring candidates.

    The candidates are the previous plan shifted by the committed steps, zero
    controls, the problem's extra candidates, and noisy copies of the shifted plan,
    all clipped to [lower, upper]. Rejected rollouts lose to any accepted one.

    With a seeder, each step first scores the problem's extra candidates (a bar
    that does not depend on the seeder) and then the proposals, and commits a
    proposal that the seed config accepts. Otherwise it fills the batch with the
    usual candidates and commits the best of everything scored. The first
    proposal of the first step also seeds the plan that later steps shift.
    """

    def __init__(
        self,
        config: PredictiveSamplingConfig,
        problem: RolloutProblem[StateType],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        rng: np.random.Generator,
        seeder: Seeder | None = None,
        seed_config: SeedConfig = SeedConfig(),
    ) -> None:
        self._config = config
        self._problem = problem
        self._lower, self._upper = lower, upper
        self._rng = rng
        self._basis = get_control_basis(config.horizon, config.num_control_points)
        self._control_indices = np.round(
            np.linspace(0, config.horizon - 1, config.num_control_points)
        ).astype(int)
        self._num_commit = int(np.clip(config.commit_steps, 1, config.horizon))
        self._nominal = np.zeros((config.num_control_points, len(lower)))
        self._seeder = seeder
        self._seed_config = seed_config
        self._step_index = 0
        self.num_rollouts = 0
        self.num_seeded_steps = 0
        self.num_full_steps = 0

    def expand(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Control points to per-step controls."""
        return self._basis @ points

    def project(self, controls: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-step controls to the control points that fit them best."""
        return np.asarray(np.linalg.lstsq(self._basis, controls, rcond=None)[0])

    def step(self) -> NDArray[np.float64]:
        """The controls to apply next, with shape (commit_steps, dim).

        Raises NoFeasibleControls if every candidate breaks a constraint.
        """
        state = self._problem.get_state()
        error = self._problem.get_error()
        proposals = self._get_proposals(state)
        if proposals and self._step_index == 0:
            self._nominal = self._clip(self.project(proposals[0]))
        # Each entry: control points, per-step controls, and (rejected, cost).
        scored: list[tuple[Any, NDArray[np.float64], tuple[bool, float]]] = []
        if proposals:
            scale = self._get_noise_scale(error)
            for extra in self._problem.get_extra_candidates(state, scale, self._rng):
                points = self._clip(extra)
                controls = self.expand(points)
                scored.append((points, controls, self._rollout(state, controls)))
            costs = [cost for _, _, (rejected, cost) in scored if not rejected]
            threshold = min(costs, default=np.inf) * (1 + self._seed_config.tolerance)
            for controls in proposals:
                result = self._rollout(state, controls)
                scored.append((self.project(controls), controls, result))
                rejected, cost = result
                if not rejected and (self._problem.is_near_goal() or cost <= threshold):
                    self.num_seeded_steps += 1
                    return self._commit(scored[-1][0], controls)
        self.num_full_steps += 1
        num_left = self._config.num_rollouts - len(scored)
        for points in self._sample_candidates(state, error, num_left, not proposals):
            controls = self.expand(points)
            scored.append((points, controls, self._rollout(state, controls)))
        points, controls, (rejected, _) = min(scored, key=lambda entry: entry[2])
        if rejected:
            raise NoFeasibleControls("Every MPC candidate breaks a constraint")
        return self._commit(points, controls)

    def _get_proposals(self, state: StateType) -> list[NDArray[np.float64]]:
        if self._seeder is None or self._seed_config.num_rollouts < 1:
            return []
        proposals = self._seeder(
            state, self._step_index, self._seed_config.num_rollouts
        )
        return [np.clip(p, self._lower, self._upper) for p in proposals]

    def _rollout(
        self, state: StateType, controls: NDArray[np.float64]
    ) -> tuple[bool, float]:
        self.num_rollouts += 1
        return self._problem.rollout(state, controls)

    def _commit(
        self, points: NDArray[np.float64], controls: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self._nominal = self._clip(points)
        self._step_index += self._num_commit
        return controls[: self._num_commit]

    def _get_noise_scale(self, error: float) -> float:
        config = self._config
        ratio = np.clip(error / config.noise_reference_error, config.min_noise_ratio, 1)
        return config.noise_scale * float(ratio)

    def _clip(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(points, self._lower, self._upper)

    def _sample_candidates(
        self, state: StateType, error: float, num_rollouts: int, with_extra: bool
    ) -> list[NDArray[np.float64]]:
        scale = self._get_noise_scale(error)
        # Shift the previous plan by the committed steps, extending its last control
        # with a noisy ramp.
        expanded = self.expand(self._nominal)
        tail = self._rng.normal(expanded[-1], scale)
        ramp = np.linspace(expanded[-1], tail, self._num_commit + 1)[1:]
        shifted = np.vstack([expanded[self._num_commit :], ramp])
        nominal = shifted[self._control_indices]
        candidates = [self._clip(nominal), np.zeros_like(nominal)]
        if with_extra:
            extra = self._problem.get_extra_candidates(state, scale, self._rng)
            candidates.extend(self._clip(c) for c in extra)
        num_noisy = max(num_rollouts - len(candidates), 0)
        noise = self._rng.normal(scale=scale, size=(num_noisy,) + nominal.shape)
        candidates.extend(self._clip(nominal + sample) for sample in noise)
        return candidates


@dataclass(frozen=True)
class LimbMPCConfig:
    """MPC settings, from limb-manipulation.

    Rollout costs weight the distance to the goal, the limb's speed near the goal
    and above max_velocity, and the limb's and robot's speeds. A rollout that breaks
    a constraint is rejected.
    """

    sampling: PredictiveSamplingConfig = PredictiveSamplingConfig()
    directed_gains: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    goal_reaching_weight: float = 100.0
    terminal_goal_weight: float = 100.0
    velocity_penalty_weight: float = 0.5
    velocity_penalty_threshold: float = 0.05
    max_velocity: float = 2.0
    max_velocity_penalty: float = 300.0
    velocity_regularization_weight: float = 1.0
    robot_velocity_regularization_weight: float = 1.0
    violation_cost: float = 1e6
    collision_cost: float = 1e4
    # MPC stops this far inside the goal tolerance, so the replay can drift.
    replay_slack: float = 0.05
    divergence_patience: int = 200
    divergence_tolerance: float = 1e-5


@dataclass(frozen=True, eq=False)
class MPCParams:
    """Parameters of the limb MPC planner.

    Planning treats a configuration as inside the person's range if at least
    `confidence` of `num_particles` samples from the belief allow it. The model
    proposes the next horizon of controls from get_mpc_features() and seeds MPC; an
    untrained model proposes nothing.
    """

    config: LimbMPCConfig = field(default_factory=LimbMPCConfig)
    num_particles: int = 32
    confidence: float = 0.5
    model: ConditionalModel = field(default_factory=FlowMatchingModel)
    seed: SeedConfig = SeedConfig()


def get_mpc_features(
    state: CoupledState,
    goal: NDArray[np.float64],
    belief: RangeOfMotionBelief,
    step: int,
    previous_control: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The model's input at an MPC step: limb and robot positions, goal, belief
    mean and standard deviation, step, limb and robot velocities, last control.

    The belief's moments are those of its truncated Gaussians, so a floor raised by
    the limb's motion changes them. The untruncated parameters change only when the
    person signals a limit or answers a query.
    """
    return np.concatenate(
        [
            state.limb_positions,
            state.robot_positions[:NUM_ROBOT_JOINTS],
            goal,
            belief.truncated_mean,
            belief.truncated_stddev,
            [float(step)],
            state.limb_velocities,
            state.robot_velocities[:NUM_ROBOT_JOINTS],
            previous_control,
        ]
    )


def get_control_window(
    controls: list[NDArray[np.float64]], step: int, horizon: int
) -> NDArray[np.float64]:
    """The model's label at an MPC step: the next horizon of controls, padded with
    the last one."""
    window = controls[step : step + horizon]
    return np.array(window + [controls[-1]] * (horizon - len(window)))


class LimbMPCPolicy(OpenLoopPolicy[LimbState, NDArray[np.float64]]):
    """An open-loop plan that remembers how MPC made it: the target goal, the MPC
    horizon, the state before and control of each MPC step, and the belief the plan
    was made under, with its floors raised to the start's reach."""

    def __init__(
        self,
        actions: list[NDArray[np.float64]],
        stats: dict[str, float],
        target: NDArray[np.float64],
        horizon: int,
        states: list[CoupledState],
        controls: list[NDArray[np.float64]],
        belief: RangeOfMotionBelief,
    ) -> None:
        super().__init__(actions, stats)
        self.target = target
        self.horizon = horizon
        self.states = tuple(states)
        self.controls = tuple(controls)
        self.belief = belief


@dataclass(frozen=True, eq=False)
class ParticleLimits:
    """A chance constraint on the limb's configuration, from belief samples."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    particles: NDArray[np.float64]
    confidence: float

    @classmethod
    def from_particles(
        cls, variant: LimbVariant, particles: NDArray[np.float64], confidence: float
    ) -> "ParticleLimits":
        """The limits of each particle (theta sample)."""
        limits = [variant.get_joint_limits(theta) for theta in particles]
        return cls(
            lower=np.stack([lower for lower, _ in limits]),
            upper=np.stack([upper for _, upper in limits]),
            particles=particles,
            confidence=confidence,
        )

    def __call__(self, positions: NDArray[np.float64]) -> bool:
        """Whether enough particles allow the configuration."""
        inside = np.all((self.lower <= positions) & (positions <= self.upper), axis=1)
        return bool(inside.mean() >= self.confidence)

    def get_quantile_theta(self) -> NDArray[np.float64]:
        """The scale of each motion that `confidence` of the particles reach."""
        return np.quantile(self.particles, 1.0 - self.confidence, axis=0)


def retarget(
    goal: NDArray[np.float64], limits: ParticleLimits, variant: LimbVariant
) -> tuple[NDArray[np.float64], bool]:
    """The goal moved just inside the believed limits, if it lies beyond them, and
    whether it moved."""
    if limits(goal):
        return goal, False
    lower, upper = variant.get_joint_limits(limits.get_quantile_theta())
    beyond = [j for j in variant.active_joints if not lower[j] <= goal[j] <= upper[j]]
    margin, moved = RETARGET_MARGIN, goal
    for _ in range(10):
        moved = goal.copy()
        for j in beyond or variant.active_joints:
            low = min(lower[j] + margin, upper[j])
            high = max(upper[j] - margin, lower[j])
            moved[j] = np.clip(goal[j], low, high)
        if limits(moved):
            break
        margin += RETARGET_MARGIN
    return moved, True


def _with_start_reach(
    belief: RangeOfMotionBelief, start: LimbState
) -> RangeOfMotionBelief:
    """The belief with each motion's floor raised to the start's reach, by the
    belief updater's rule."""
    for motion, reach in enumerate(start.max_reach):
        if reach > belief.lower[motion] + FLOOR_STEP:
            belief = belief.with_floor(motion, reach)
    return belief


class _Checker:
    """Checks the constraints after a simulator step, under the believed limits."""

    def __init__(
        self,
        model: LimbModel,
        limits: ParticleLimits,
        resting_penetration: dict[int, float],
    ) -> None:
        self._model = model
        self._limits = limits
        self._resting = resting_penetration

    def step(self, sim: LimbSim, correction: NDArray[np.float64]) -> str:
        """Step once under the correction and return the broken constraint, if
        any."""
        commanded = sim.get_commanded_torque(correction)
        if exceeds_robot_torque_limits(sim, commanded):
            return "robot torque limits"
        if self._model.torque_limits.are_exceeded(sim.advance(commanded)):
            return "human torque limits"
        violation = get_robot_motion_violation(sim, self._model.max_robot_velocity)
        if violation:
            return violation
        if not self._limits(np.array(sim.env.limb.get_joint_positions())):
            return "believed range of motion"
        if arm_in_collision(sim) or limb_in_collision(sim, self._resting):
            return "collision"
        return ""


class _LimbRolloutProblem(RolloutProblem[CoupledState]):
    """Rollouts of torque corrections in a private copy of the simulator."""

    def __init__(
        self,
        world: LimbSim,
        rollout_world: LimbSim,
        goal: NDArray[np.float64],
        config: LimbMPCConfig,
        action_repeat: int,
        checker: _Checker,
        reach: float,
    ) -> None:
        self._world = world
        self._rollout_world = rollout_world
        self._goal = goal
        self._config = config
        self._action_repeat = action_repeat
        self._checker = checker
        self._reach = reach
        self._last_error = np.inf

    def get_state(self) -> CoupledState:
        return self._world.capture()

    def get_error(self) -> float:
        positions = self._world.env.limb.get_joint_positions()
        return self._world.get_limb_error(positions, self._goal)

    def rollout(
        self, state: CoupledState, controls: NDArray[np.float64]
    ) -> tuple[bool, float]:
        sim, config = self._rollout_world, self._config
        sim.restore(state, regrasp=False)
        self._last_error = np.inf
        goal_cost = velocity_cost = regularization = robot_regularization = 0.0
        squared_error = 0.0
        for control in controls:
            for _ in range(self._action_repeat):
                broken = self._checker.step(sim, control)
                if broken:
                    cost = config.collision_cost if broken == "collision" else 0.0
                    return True, cost + config.violation_cost
            velocities = np.asarray(sim.env.limb.get_joint_velocities())
            positions = sim.env.limb.get_joint_positions()
            squared_error = sim.get_limb_error(positions, self._goal) ** 2
            goal_cost += squared_error
            velocity_cost += self._get_velocity_penalty(velocities, squared_error)
            regularization += float(np.sum(velocities**2))
            robot_velocities = sim.env.robot.arm.get_joint_velocities()
            robot_regularization += float(
                np.sum(np.square(robot_velocities[:NUM_ROBOT_JOINTS]))
            )
        cost = (
            config.goal_reaching_weight * goal_cost
            + config.terminal_goal_weight * squared_error
            + config.velocity_penalty_weight * velocity_cost
            + config.velocity_regularization_weight * regularization
            + config.robot_velocity_regularization_weight * robot_regularization
        )
        self._last_error = float(np.sqrt(squared_error))
        return False, cost

    def is_near_goal(self) -> bool:
        return self._last_error < self._reach

    def _get_velocity_penalty(
        self, velocities: NDArray[np.float64], squared_error: float
    ) -> float:
        """Damp the limb near the goal and cap its speed, so MPC does not
        overshoot."""
        config = self._config
        speed = float(np.linalg.norm(velocities))
        penalty = 0.0
        if squared_error < config.velocity_penalty_threshold:
            decay = np.exp(-squared_error / config.velocity_penalty_threshold)
            penalty += speed * float(decay)
        if speed > config.max_velocity:
            penalty += config.max_velocity_penalty
        return penalty

    def get_extra_candidates(
        self, state: CoupledState, noise_scale: float, rng: np.random.Generator
    ) -> list[NDArray[np.float64]]:
        """Torques that push the limb toward the goal, each with a noisy twin."""
        num_points = self._config.sampling.num_control_points
        candidates = []
        for gain in self._config.directed_gains:
            torque = self._world.get_limb_directed_torque(self._goal, gain)
            points = np.tile(torque, (num_points, 1))
            noise = rng.normal(scale=noise_scale, size=points.shape)
            candidates.extend([points, points + noise])
        return candidates


class LimbMPCPlanner(
    TrainablePlanner[
        LimbState, NDArray[np.float64], LimbTask, NDArray[np.float64], MPCParams
    ]
):
    """Plans torque corrections for a limb task with predictive-sampling MPC.

    One iteration of the budget is one MPC rollout, which caps the planner's work
    the same way on any machine. MPC starts no step once the plan has used the
    budget's rollouts or time, so the last step can go over by less than one
    step's rollouts. The policy is the committed controls up to the step where the
    replay reaches the task's goal, which ends the task, then STOP to the horizon,
    so nothing moves after a plan that stops short of the goal (for example, at a
    retargeted goal). The model in the parameters seeds each MPC step; if planning
    with it fails other than by running out of budget, the planner plans again
    without it, and both attempts share the budget and the statistics. They are
    the number of rollouts, MPC steps, and seeded steps, whether it fell back,
    whether the goal was retargeted, and the replayed distance to the goal; a
    failed plan carries them up to the failure. The default parameters, if given,
    replace MPCParams().
    """

    def __init__(
        self, model: LimbModel, default_params: MPCParams | None = None
    ) -> None:
        self._model = model
        self._default_params = MPCParams() if default_params is None else default_params
        self._world: LimbSim | None = None
        self._rollout_world: LimbSim | None = None

    def get_default_params(self) -> MPCParams:
        return self._default_params

    def get_model(self, params: MPCParams) -> ConditionalModel:
        return params.model

    def set_model(self, params: MPCParams, model: ConditionalModel) -> MPCParams:
        return dataclasses.replace(params, model=model)

    def get_plan_examples(
        self,
        task: LimbTask,
        belief: Belief[NDArray[np.float64]],
        policy: Policy[LimbState, NDArray[np.float64]],
    ) -> Dataset:
        """One row per MPC step of the plan: the features MPC's seeder saw, and
        the next horizon of controls.

        The features use the belief the plan was made under, which plan() raised
        to the start's reach, so they match what the seeder saw. The belief
        argument is unused.
        """
        del belief  # unused: the plan records the belief it was made under
        assert isinstance(policy, LimbMPCPolicy), "Need a plan from plan()"
        controls, horizon = list(policy.controls), policy.horizon
        x, y = [], []
        for step, state in enumerate(policy.states):
            previous = controls[step - 1] if step else np.zeros(NUM_ROBOT_JOINTS)
            x.append(
                get_mpc_features(state, policy.target, policy.belief, step, previous)
            )
            y.append(get_control_window(controls, step, horizon).ravel())
        return Dataset(
            x=np.array(x),
            y=np.array(y),
            costs=np.zeros(len(x)),
            groups=np.zeros(len(x), dtype=np.int64),
        )

    def close(self) -> None:
        """Disconnect the planner's simulators."""
        for sim in (self._world, self._rollout_world):
            if sim is not None:
                sim.close()
        self._world = self._rollout_world = None

    def _get_worlds(self, start: CoupledState) -> tuple[LimbSim, LimbSim]:
        if self._world is None:
            self._world = LimbSim(self._model.variant, self._model.settings)
            self._world.restore(start)
            self._rollout_world = self._world.clone()
        assert self._rollout_world is not None
        self._world.restore(start)
        self._rollout_world.restore(start)
        return self._world, self._rollout_world

    def plan(
        self,
        task: LimbTask,
        belief: Belief[NDArray[np.float64]],
        params: MPCParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> LimbMPCPolicy:
        """Find torque corrections that reach the goal under the believed limits.

        The limb starts at the task's initial configuration, so the person's range
        covers it: the believed limits first rise to the start's reach, as the
        belief updater raises them after every step. Otherwise a start beyond what
        half of the belief's samples allow would break the believed range of motion
        before any control.

        Raises PlanningTimeout if the plan runs out of the budget's rollouts or
        time, and PlanningFailure if MPC breaks a constraint, diverges, runs out of
        steps, or the replay misses the goal, with and without the model.
        """
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        start_time = time.perf_counter()
        belief = _with_start_reach(belief, task.initial_state)
        particles = np.stack([belief.sample(rng) for _ in range(params.num_particles)])
        limits = ParticleLimits.from_particles(
            self._model.variant, particles, params.confidence
        )
        goal, retargeted = retarget(np.asarray(task.goal), limits, self._model.variant)
        if not limits(goal):
            raise PlanningFailure("The goal is outside the believed range of motion")
        stats = {
            "num_rollouts": 0.0,
            "num_mpc_steps": 0.0,
            "num_seeded_steps": 0.0,
            "fell_back": 0.0,
            "retargeted": float(retargeted),
        }
        attempt = _Attempt(task, goal, limits, belief, params, budget, start_time, rng)
        try:
            return self._plan_with_fallback(attempt, stats)
        except PlanningFailure as failure:
            # Each attempt counts its rollouts as it stops, so the statistics are
            # complete only now.
            failure.stats = dict(stats)
            raise

    def _plan_with_fallback(
        self, attempt: "_Attempt", stats: dict[str, float]
    ) -> LimbMPCPolicy:
        """Plan with the model, and again without it if that fails for a reason
        other than the shared budget running out."""
        try:
            return self._plan_once(attempt, stats, use_model=True)
        except PlanningTimeout:
            raise
        except PlanningFailure:
            if not attempt.model_proposed:
                raise
        stats["fell_back"] = 1.0
        return self._plan_once(attempt, stats, use_model=False)

    def _plan_once(
        self, attempt: "_Attempt", stats: dict[str, float], use_model: bool
    ) -> LimbMPCPolicy:
        model, config = self._model, attempt.params.config
        start = attempt.task.initial_state
        world, rollout_world = self._get_worlds(start.coupled)
        checker = _Checker(model, attempt.limits, dict(start.resting_penetration))
        threshold = max(
            world.goal_tolerance - config.replay_slack, world.goal_tolerance / 2
        )
        problem = _LimbRolloutProblem(
            world,
            rollout_world,
            attempt.goal,
            config,
            model.action_repeat,
            checker,
            threshold,
        )
        controls: list[NDArray[np.float64]] = []
        states: list[CoupledState] = []

        def seeder(state: CoupledState, step: int, num: int) -> list:
            previous = controls[-1] if controls else np.zeros(NUM_ROBOT_JOINTS)
            x = get_mpc_features(state, attempt.goal, attempt.belief, step, previous)
            draws = attempt.params.model.sample(x, num, attempt.rng)
            attempt.model_proposed |= len(draws) > 0
            horizon = config.sampling.horizon
            return list(draws.reshape(len(draws), horizon, NUM_ROBOT_JOINTS))

        lower, upper = world.robot_torque_limits
        mpc = PredictiveSampling(
            config.sampling,
            problem,
            lower,
            upper,
            attempt.rng,
            seeder if use_model else None,
            attempt.params.seed,
        )
        error = best_error = problem.get_error()
        num_stale = 0
        try:
            while error >= threshold:
                if len(controls) >= model.horizon:
                    raise PlanningFailure(
                        f"Ran out of steps at an error of {error:.3f}"
                    )
                if num_stale >= config.divergence_patience:
                    raise PlanningFailure(f"Diverged at an error of {best_error:.3f}")
                elapsed = time.perf_counter() - attempt.start_time
                max_wall_time = attempt.budget.max_wall_time
                if max_wall_time is not None and elapsed >= max_wall_time:
                    raise PlanningTimeout("Ran out of planning budget time")
                # Both attempts' rollouts, since they share the budget.
                num_rollouts = int(stats["num_rollouts"]) + mpc.num_rollouts
                if attempt.budget.is_exhausted(elapsed, num_rollouts):
                    raise PlanningTimeout("Ran out of planning budget rollouts")
                state = world.capture()
                try:
                    committed = mpc.step()
                except NoFeasibleControls as failure:
                    raise PlanningFailure(str(failure)) from failure
                stats["num_mpc_steps"] += 1
                for control in committed:
                    for _ in range(model.action_repeat):
                        broken = checker.step(world, control)
                        if broken:
                            raise PlanningFailure(f"MPC broke a constraint: {broken}")
                    states.append(state)
                    controls.append(control)
                    state = world.capture()
                error = problem.get_error()
                if best_error - error > config.divergence_tolerance:
                    num_stale = 0
                else:
                    num_stale += 1
                best_error = min(best_error, error)
        finally:
            stats["num_rollouts"] += mpc.num_rollouts
            stats["num_seeded_steps"] += mpc.num_seeded_steps

        actions = self._replay(world, start, controls, attempt.task.goal, checker)
        replay_error = world.get_limb_error(
            world.env.limb.get_joint_positions(), attempt.goal
        )
        if replay_error >= world.goal_tolerance:
            raise PlanningFailure(f"The replay ended {replay_error:.3f} from the goal")
        padding = [STOP] * (model.horizon - len(actions))
        return LimbMPCPolicy(
            actions + padding,
            {**stats, "replay_error": replay_error},
            attempt.goal,
            config.sampling.horizon,
            states,
            controls,
            attempt.belief,
        )

    def _replay(
        self,
        world: LimbSim,
        start: LimbState,
        controls: list[NDArray[np.float64]],
        task_goal: NDArray[np.float64],
        checker: _Checker,
    ) -> list[NDArray[np.float64]]:
        """Replay the controls from the start until the limb reaches the task's
        goal, which ends the task, and return the controls replayed.

        Raises PlanningFailure if the replay breaks a constraint.
        """
        world.restore(start.coupled)
        for step, control in enumerate(controls):
            for _ in range(self._model.action_repeat):
                broken = checker.step(world, control)
                if broken:
                    raise PlanningFailure(f"The replay broke a constraint: {broken}")
            positions = world.env.limb.get_joint_positions()
            if world.get_limb_error(positions, task_goal) < world.goal_tolerance:
                return controls[: step + 1]
        return controls


@dataclass
class _Attempt:
    """What both of a plan's attempts share."""

    task: LimbTask
    goal: NDArray[np.float64]
    limits: ParticleLimits
    belief: RangeOfMotionBelief
    params: MPCParams
    budget: Budget
    start_time: float
    rng: np.random.Generator
    model_proposed: bool = False
