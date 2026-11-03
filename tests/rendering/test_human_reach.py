"""Tests for rendering/human_reach.py."""

import numpy as np

from able.envs.human_reach.model import ReachModel, ReachTask
from able.envs.human_reach.tasks import make_prior_belief
from able.rendering.base import Snapshot
from able.rendering.human_reach import HumanReachRenderer


def test_human_reach_renderer():
    """Tests frames outside an episode, before and after a plan, and in dreams."""
    model = ReachModel(num_requested=2, num_clutter=2)
    renderer = HumanReachRenderer(model, width=640, height=288)
    rng = np.random.default_rng(0)
    belief = make_prior_belief(model.config, 50, rng)
    theta = np.array([0.7, 0.6, 0.65])
    empty = renderer.render(Snapshot("Sleep", belief, theta))
    assert empty.shape == (288, 640, 3) and empty.dtype == np.uint8
    state = model.get_initial_state(0)
    task = ReachTask(state, tuple(state.scene.requested_objects))
    after = model.sample_next_state(state, (), theta, rng)
    frame = renderer.render(
        Snapshot("Task", belief, theta, task, (state, after), hypothesis=theta - 0.05)
    )
    assert frame.shape == (288, 640, 3) and not np.array_equal(frame, empty)
    renderer.close()
    model.close()
