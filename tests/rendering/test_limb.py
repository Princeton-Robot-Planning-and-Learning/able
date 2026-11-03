"""Tests for rendering/limb.py."""

import numpy as np

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import get_variant
from able.envs.limb.model import LimbModel, LimbTask
from able.rendering.base import Snapshot
from able.rendering.limb import LimbRenderer


def test_limb_renderer():
    """Tests frames inside and outside an episode, and in dreams."""
    model = LimbModel(get_variant("bed-left-arm"))
    renderer = LimbRenderer(model, width=640, height=288)
    try:
        belief = RangeOfMotionBelief.from_prior(0.9, 0.15, 0.7, 1.2, 3).with_floor(
            0, 0.8
        )
        theta = np.array([0.85, 1.0, 0.95])
        empty = renderer.render(Snapshot("Sleep", belief, theta))
        assert empty.shape == (288, 640, 3) and empty.dtype == np.uint8

        start = model.sim.capture()
        goal = np.array(start.limb_positions) + 0.1
        state = model.get_initial_state(start, goal)
        task = LimbTask(initial_state=state, goal=goal)
        frame = renderer.render(
            Snapshot("Task", belief, theta, task, (state,), hypothesis=theta - 0.1)
        )
        assert frame.shape == (288, 640, 3)
        assert not np.array_equal(frame, empty)
    finally:
        renderer.close()
        model.close()
