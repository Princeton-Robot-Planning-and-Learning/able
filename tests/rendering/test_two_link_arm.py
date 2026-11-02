"""Tests for rendering/two_link_arm.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, ArmTask, TwoLinkArmModel
from able.rendering.base import Snapshot
from able.rendering.two_link_arm import TwoLinkArmRenderer

TRUE_LIMITS = np.array([-1.2, 1.4, -2.0, 0.9])


def test_render():
    """Tests that frames have the requested size and change with the belief."""
    model = TwoLinkArmModel()
    renderer = TwoLinkArmRenderer(model, width=640, height=320)
    task = ArmTask(initial_state=np.zeros(2), goal=np.array([1.0, 1.0]))
    states = (np.zeros(2), np.array([0.3, 0.2]), np.array([0.6, 0.4]))
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    narrow = IntervalBelief(np.stack([TRUE_LIMITS - 0.1, TRUE_LIMITS], axis=1))

    image = renderer.render(
        Snapshot(
            "Cycle 0 · Sleep\nDream 1",
            prior,
            TRUE_LIMITS,
            task=task,
            states=states,
            hypothesis=JOINT_LIMIT_PRIOR.mean(axis=1),
        )
    )
    assert image.shape == (320, 640, 3)
    assert image.dtype == np.uint8

    # Frames outside episodes have no arm or goal.
    without_episode = renderer.render(Snapshot("Queries", prior, TRUE_LIMITS))
    assert without_episode.shape == (320, 640, 3)
    narrowed = renderer.render(Snapshot("Queries", narrow, TRUE_LIMITS))
    assert not np.array_equal(without_episode, narrowed)

    with pytest.raises(AssertionError):
        renderer.render(Snapshot("Bad", object(), TRUE_LIMITS))  # type: ignore[arg-type]
