"""Tests for envs/simulated.py."""

import numpy as np
import pytest

from able.envs.simulated import SimulatedEnvironment
from tests.envs.wall_model import ACTION_COST, WallModel, WallTask


def test_simulated_environment():
    """Tests for SimulatedEnvironment."""
    model = WallModel(horizon=3)
    env = SimulatedEnvironment(model, np.array([1.5]), np.random.default_rng(0))
    task = WallTask(initial_state=np.zeros(1), goal=2.0)

    with pytest.raises(AssertionError):
        env.step(np.ones(1))

    state = env.reset(task)
    assert np.allclose(state, [0.0])

    result = env.step(np.ones(1))
    assert np.allclose(result.state, [1.0])

    # Queries are answered with the hidden wall position and leave the episode alone.
    assert env.ask(1.5)
    assert not env.ask(1.6)
    assert np.isclose(result.cost, ACTION_COST)
    assert not result.terminated

    # The wall stops the point at 1.5.
    result = env.step(np.ones(1))
    assert np.allclose(result.state, [1.5])
    assert not result.terminated

    # The last step adds the terminal cost of the final state.
    result = env.step(np.zeros(1))
    assert np.allclose(result.state, [1.5])
    assert np.isclose(result.cost, 0.5)
    assert result.terminated

    with pytest.raises(AssertionError):
        env.step(np.ones(1))

    # Resetting starts a new episode.
    state = env.reset(task)
    assert np.allclose(state, [0.0])
    assert not env.step(np.ones(1)).terminated
