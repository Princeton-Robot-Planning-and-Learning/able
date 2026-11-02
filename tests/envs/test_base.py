"""Tests for envs/base.py."""

import numpy as np
import pytest
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.envs.base import Domain, FiniteTaskDistribution
from able.envs.simulated import SimulatedEnvironment
from tests.envs.wall_model import WallModel, WallTask


class _PointBelief(Belief[NDArray[np.float64]]):
    """A belief that puts all of its mass on one value."""

    def __init__(self, value: NDArray[np.float64]) -> None:
        self._value = value

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return the single value."""
        del rng  # unused because the belief is deterministic
        return self._value


def test_finite_task_distribution():
    """Tests for FiniteTaskDistribution."""
    tasks = [WallTask(initial_state=np.zeros(1), goal=g) for g in (1.0, 2.0, 3.0)]
    dist = FiniteTaskDistribution(tasks)
    assert dist.get_tasks() == tuple(tasks)
    rng = np.random.default_rng(0)
    samples = [dist.sample(rng) for _ in range(100)]
    assert {id(t) for t in samples} == {id(t) for t in tasks}
    with pytest.raises(AssertionError):
        FiniteTaskDistribution([])


def test_batch_rollout_model():
    """Tests the BatchRolloutModel interface and its default cost lower bound."""
    model = WallModel(horizon=3)
    task = WallTask(initial_state=np.zeros(1), goal=2.0)
    rng = np.random.default_rng(0)
    actions = np.array([[[1.0], [1.0], [1.0]], [[0.5], [0.5], [0.5]]])  # (K=2, T=3, 1)
    walls = np.array([[10.0], [1.0], [0.25]])  # (P=3, 1)
    states = model.rollout_batch(task.initial_state, actions, walls, rng)
    assert states.shape == (2, 3, 4, 1)
    assert np.allclose(states[0, :, -1, 0], [3.0, 1.0, 0.25])
    assert np.allclose(states[1, :, -1, 0], [1.5, 1.0, 0.25])
    costs = model.get_cost_batch(task, states, actions, walls)
    assert costs.shape == (2, 3)
    assert np.isclose(costs[0, 0], 0.3 + 1.0)
    assert np.isclose(costs[1, 2], 0.15 + 1.75)
    lower_bound = model.get_cost_lower_bound(task, walls)
    assert lower_bound.shape == (3,)
    assert np.all(lower_bound == 0.0)


def test_domain():
    """Tests for Domain."""
    model = WallModel()
    task = WallTask(initial_state=np.zeros(1), goal=2.0)
    true_wall = np.array([1.5])
    domain = Domain(
        env_model=model,
        task_distribution=FiniteTaskDistribution([task]),
        initial_belief=_PointBelief(np.array([5.0])),
        real_env=SimulatedEnvironment(model, true_wall, np.random.default_rng(0)),
    )
    assert domain.real_env.reset(task) is task.initial_state
