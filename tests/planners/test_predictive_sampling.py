"""Tests for planners/predictive_sampling.py."""

import numpy as np
import pytest

from able.planners.predictive_sampling import (
    NoFeasibleControls,
    PredictiveSampling,
    PredictiveSamplingConfig,
    RolloutProblem,
    get_control_basis,
)

GOAL = np.array([1.0, -0.5])
DT = 0.1


def _simulate(state: np.ndarray, controls: np.ndarray) -> list[np.ndarray]:
    """A 2D point mass: the state is (position, velocity), controls accelerate it."""
    states = []
    for control in controls:
        velocity = state[2:] + DT * control
        state = np.concatenate([state[:2] + DT * velocity, velocity])
        states.append(state)
    return states


class _PointMass(RolloutProblem[np.ndarray]):
    """Reach the goal; rollouts that leave x <= max_x are rejected."""

    def __init__(self, max_x: float = np.inf, extra: list | None = None) -> None:
        self.state = np.zeros(4)
        self.max_x = max_x
        self.extra = [] if extra is None else extra

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_error(self) -> float:
        return float(np.linalg.norm(self.state[:2] - GOAL))

    def rollout(self, state: np.ndarray, controls: np.ndarray) -> tuple[bool, float]:
        states = _simulate(state, controls)
        if any(s[0] > self.max_x for s in states):
            return True, np.inf
        return False, sum(float(np.sum((s[:2] - GOAL) ** 2)) for s in states)

    def get_extra_candidates(self, state, noise_scale, rng) -> list:
        return list(self.extra)


def test_control_basis():
    """Tests that the basis interpolates control points linearly."""
    basis = get_control_basis(horizon=5, num_control_points=3)
    assert basis.shape == (5, 3)
    assert np.allclose(basis.sum(axis=1), 1.0)
    points = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
    assert np.allclose(basis @ points, [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]])


def test_reaches_the_goal():
    """Tests that committing the best controls each step drives the point mass to
    the goal."""
    config = PredictiveSamplingConfig(noise_scale=2.0, noise_reference_error=1.0)
    problem = _PointMass()
    mpc = PredictiveSampling(
        config, problem, np.full(2, -3.0), np.full(2, 3.0), np.random.default_rng(0)
    )
    for _ in range(100):
        controls = mpc.step()
        assert controls.shape == (1, 2)
        assert np.all(np.abs(controls) <= 3.0)
        problem.state = _simulate(problem.state, controls)[-1]
    assert problem.get_error() < 0.1
    assert mpc.num_rollouts == 100 * config.num_rollouts


def test_constraints():
    """Tests that rejected candidates lose and that extra candidates are scored."""
    lower, upper = np.full(2, -3.0), np.full(2, 3.0)
    config = PredictiveSamplingConfig(num_rollouts=4)
    # Moving toward the wall at x = 0, only braking hard avoids it.
    start = np.array([-0.05, 0.0, 0.3, 0.0])
    brake = np.tile([-3.0, 0.0], (4, 1))
    problem = _PointMass(max_x=0.0, extra=[brake])
    problem.state = start
    mpc = PredictiveSampling(config, problem, lower, upper, np.random.default_rng(0))
    assert np.allclose(mpc.step(), [[-3.0, 0.0]])

    problem = _PointMass(max_x=0.0)
    problem.state = start
    mpc = PredictiveSampling(config, problem, lower, upper, np.random.default_rng(0))
    with pytest.raises(NoFeasibleControls):
        mpc.step()
