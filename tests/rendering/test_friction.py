"""Tests for rendering/friction.py."""

import numpy as np

from able.envs.friction.model import FrictionModel
from able.envs.friction.tasks import FrictionTaskDistribution, make_prior_belief
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, Push
from able.rendering.base import Snapshot
from able.rendering.friction import FrictionRenderer


def test_friction_renderer():
    """Tests frames outside an episode, after a push, and in dreams."""
    model = FrictionModel()
    renderer = FrictionRenderer(model, width=640, height=288)
    rng = np.random.default_rng(0)
    belief = make_prior_belief(DEFAULT_FRICTION_PRIOR, 50, rng)
    frictions = np.array([0.03, 0.08, 0.15])
    empty = renderer.render(Snapshot("Sleep", belief, frictions))
    assert empty.shape == (288, 640, 3) and empty.dtype == np.uint8

    task = FrictionTaskDistribution(model).sample(rng)
    state = model.sample_next_state(
        task.initial_state, Push(np.pi / 2, 0.5), frictions, rng
    )
    states = (task.initial_state, state)
    frame = renderer.render(
        Snapshot("Task", belief, frictions, task, states, hypothesis=frictions + 0.01)
    )
    assert frame.shape == (288, 640, 3)
    assert not np.array_equal(frame, empty)
    model.close()
