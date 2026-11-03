"""Predictive-sampling MPC over control points.

Each step samples candidate control sequences around the shifted previous plan,
scores them by rolling them out, and commits the start of the best one. Problems
supply the rollouts, so the same MPC works for any simulator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

import numpy as np
from numpy.typing import NDArray

from able.structs import PlanningFailure, StateType


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
