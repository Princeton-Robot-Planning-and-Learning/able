"""Tests for rendering/friction.py."""

import dataclasses

import numpy as np
import pytest

from able.envs.friction.model import FrictionModel
from able.envs.friction.tasks import FrictionTaskDistribution, make_prior_belief
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, Push
from able.rendering.base import Snapshot
from able.rendering.friction import FrictionRenderer


@pytest.mark.parametrize("num_floors", [3, 8])
def test_friction_renderer(num_floors):
    """Tests frames outside an episode, after a push, and in dreams."""
    model = FrictionModel(num_floors=num_floors)
    renderer = FrictionRenderer(model, width=640, height=288)
    rng = np.random.default_rng(0)
    prior = dataclasses.replace(DEFAULT_FRICTION_PRIOR, num_surfaces=num_floors)
    belief = make_prior_belief(prior, 50, rng)
    frictions = prior.sample(rng)
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
