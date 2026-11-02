"""Tests for episodes.py."""

import numpy as np

from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.policies.base import OpenLoopPolicy
from tests.envs.wall_model import ACTION_COST, WallModel, WallTask


def test_run_episode():
    """Tests run_episode()."""
    model = WallModel(horizon=3)
    env = SimulatedEnvironment(model, np.array([1.5]), np.random.default_rng(0))
    task = WallTask(initial_state=np.zeros(1), goal=2.0)
    actions = [np.ones(1), np.ones(1), np.zeros(1)]

    trajectory = run_episode(env, task, OpenLoopPolicy(actions))
    assert len(trajectory) == 3
    assert np.allclose(np.array(trajectory.states).ravel(), [0.0, 1.0, 1.5, 1.5])
    assert np.isclose(trajectory.total_cost, 2 * ACTION_COST + 0.5)

    trajectory = run_episode(env, task, OpenLoopPolicy(actions), max_steps=2)
    assert len(trajectory) == 2
    assert np.allclose(np.array(trajectory.states).ravel(), [0.0, 1.0, 1.5])
