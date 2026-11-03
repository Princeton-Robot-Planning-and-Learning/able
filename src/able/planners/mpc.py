"""Predictive-sampling MPC for the limb environment.

Predictive sampling chooses controls one step at a time: it samples candidates
around the shifted previous plan, scores them with rollouts, and commits the start
of the best one. The limb planner samples limits from the belief, moves a goal
beyond them to just inside, and runs predictive sampling in its own copy of the
simulator, with rollouts in a second copy. The physics does not depend on the
person, so the controls MPC commits are replayed open loop. The planner replays them
first to check that the plan reaches the goal and breaks no constraint.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic

import numpy as np
from kinder.envs.dynamic3d.limb_utils import NUM_ROBOT_JOINTS
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.envs.limb.constraints import (
    arm_in_collision,
    exceeds_robot_torque_limits,
    get_robot_motion_violation,
    limb_in_collision,
)
from able.envs.limb.human import LimbVariant
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.sim import CoupledState, LimbSim
from able.planners.base import Planner
from able.policies.base import OpenLoopPolicy
from able.structs import Budget, PlanningFailure, StateType

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


class PredictiveSampling(Generic[StateType]):
    """Chooses controls one step at a time by sampling and scoring candidates.

    The candidates are the previous plan shifted by the committed steps, zero
    controls, the problem's extra candidates, and noisy copies of the shifted plan,
    all clipped to [lower, upper]. Rejected rollouts lose to any accepted one.
    """

    def __init__(
        self,
        config: PredictiveSamplingConfig,
        problem: RolloutProblem[StateType],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        rng: np.random.Generator,
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
        self.num_rollouts = 0

    def expand(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Control points to per-step controls."""
        return self._basis @ points

    def step(self) -> NDArray[np.float64]:
        """The controls to apply next, with shape (commit_steps, dim).

        Raises NoFeasibleControls if every candidate breaks a constraint.
        """
        state = self._problem.get_state()
        candidates = self._sample_candidates(state, self._problem.get_error())
        scores = [self._problem.rollout(state, self.expand(c)) for c in candidates]
        self.num_rollouts += len(candidates)
        best = min(range(len(scores)), key=scores.__getitem__)
        if scores[best][0]:
            raise NoFeasibleControls("Every MPC candidate breaks a constraint")
        self._nominal = candidates[best]
        return self.expand(candidates[best])[: self._num_commit]

    def _get_noise_scale(self, error: float) -> float:
        config = self._config
        ratio = np.clip(error / config.noise_reference_error, config.min_noise_ratio, 1)
        return config.noise_scale * float(ratio)

    def _clip(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(points, self._lower, self._upper)

    def _sample_candidates(
        self, state: StateType, error: float
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
        candidates.extend(
            self._clip(c)
            for c in self._problem.get_extra_candidates(state, scale, self._rng)
        )
        num_noisy = max(self._config.num_rollouts - len(candidates), 0)
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
    # The gain of the goal-directed torque that holds the limb after MPC stops.
    hold_gain: float = 2.0
    divergence_patience: int = 200
    divergence_tolerance: float = 1e-5


@dataclass(frozen=True, eq=False)
class MPCParams:
    """Parameters of the limb MPC planner.

    Planning treats a configuration as inside the person's range if at least
    `confidence` of `num_particles` samples from the belief allow it.
    """

    config: LimbMPCConfig = field(default_factory=LimbMPCConfig)
    num_particles: int = 32
    confidence: float = 0.5


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
    ) -> None:
        self._world = world
        self._rollout_world = rollout_world
        self._goal = goal
        self._config = config
        self._action_repeat = action_repeat
        self._checker = checker

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
        return False, cost

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
    Planner[LimbState, NDArray[np.float64], LimbTask, NDArray[np.float64], MPCParams]
):
    """Plans torque corrections for a limb task with predictive-sampling MPC.

    One iteration of the budget is one MPC step. The policy is the committed
    controls, then goal-directed torques that hold the limb at the goal to the end
    of the horizon (holding with no correction lets the limb sag). The planner
    computes the holding torques while it replays the plan. Its planning statistics
    are the number of rollouts, the number of MPC steps, whether the goal was
    retargeted, and the replayed distance to the goal.
    """

    def __init__(self, model: LimbModel) -> None:
        self._model = model
        self._world: LimbSim | None = None
        self._rollout_world: LimbSim | None = None

    def get_default_params(self) -> MPCParams:
        return MPCParams()

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
    ) -> OpenLoopPolicy[LimbState, NDArray[np.float64]]:
        """Find torque corrections that reach the goal under the believed limits.

        Raises PlanningFailure if MPC breaks a constraint, diverges, runs out of
        steps or budget, or the replay misses the goal.
        """
        start_time = time.perf_counter()
        model, config = self._model, params.config
        particles = np.stack([belief.sample(rng) for _ in range(params.num_particles)])
        limits = ParticleLimits.from_particles(
            model.variant, particles, params.confidence
        )
        goal, retargeted = retarget(np.asarray(task.goal), limits, model.variant)
        if not limits(goal):
            raise PlanningFailure("The goal is outside the believed range of motion")
        start = task.initial_state
        world, rollout_world = self._get_worlds(start.coupled)
        checker = _Checker(model, limits, dict(start.resting_penetration))
        problem = _LimbRolloutProblem(
            world, rollout_world, goal, config, model.action_repeat, checker
        )
        lower, upper = world.robot_torque_limits
        mpc = PredictiveSampling(config.sampling, problem, lower, upper, rng)
        threshold = max(
            world.goal_tolerance - config.replay_slack, world.goal_tolerance / 2
        )

        controls: list[NDArray[np.float64]] = []
        error = best_error = problem.get_error()
        num_steps = num_stale = 0
        while error >= threshold:
            if len(controls) >= model.horizon:
                raise PlanningFailure(f"Ran out of steps at an error of {error:.3f}")
            if num_stale >= config.divergence_patience:
                raise PlanningFailure(f"Diverged at an error of {best_error:.3f}")
            if budget.is_exhausted(time.perf_counter() - start_time, num_steps):
                raise PlanningFailure("Ran out of planning budget")
            try:
                committed = mpc.step()
            except NoFeasibleControls as failure:
                raise PlanningFailure(str(failure)) from failure
            num_steps += 1
            for control in committed:
                for _ in range(model.action_repeat):
                    broken = checker.step(world, control)
                    if broken:
                        raise PlanningFailure(f"MPC broke a constraint: {broken}")
                controls.append(control)
            error = problem.get_error()
            if best_error - error > config.divergence_tolerance:
                num_stale = 0
            else:
                num_stale += 1
            best_error = min(best_error, error)

        actions = self._replay(world, start, controls, goal, config.hold_gain, checker)
        replay_error = world.get_limb_error(world.env.limb.get_joint_positions(), goal)
        if replay_error >= world.goal_tolerance:
            raise PlanningFailure(f"The replay ended {replay_error:.3f} from the goal")
        stats = {
            "num_rollouts": float(mpc.num_rollouts),
            "num_mpc_steps": float(num_steps),
            "retargeted": float(retargeted),
            "replay_error": replay_error,
        }
        return OpenLoopPolicy(actions, stats)

    def _replay(
        self,
        world: LimbSim,
        start: LimbState,
        controls: list[NDArray[np.float64]],
        goal: NDArray[np.float64],
        hold_gain: float,
        checker: _Checker,
    ) -> list[NDArray[np.float64]]:
        """Replay the controls from the start, then hold the limb at the goal to the
        end of the horizon, and return all of the actions.

        Raises PlanningFailure if the replay breaks a constraint.
        """
        world.restore(start.coupled)
        actions = []
        for step in range(self._model.horizon):
            if step < len(controls):
                action = controls[step]
            else:
                action = world.get_limb_directed_torque(goal, hold_gain)
            for _ in range(self._model.action_repeat):
                broken = checker.step(world, action)
                if broken:
                    raise PlanningFailure(f"The replay broke a constraint: {broken}")
            actions.append(action)
        return actions
