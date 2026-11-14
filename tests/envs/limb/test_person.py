"""Tests for envs/limb/person.py."""

import numpy as np
import pytest

from able.envs.limb.person import PersonModel

LIMITS = (np.array([-1.0, -0.5]), np.array([1.0, 0.5]))


def test_walls():
    """Tests that the walls act only past the limits, and push back inside."""
    person = PersonModel()
    assert np.array_equal(
        person.get_wall_torque(np.array([0.9, -0.4]), LIMITS), np.zeros(2)
    )
    torque = person.get_wall_torque(np.array([1.1, -0.6]), LIMITS)
    assert torque == pytest.approx([-40.0, 40.0])


def test_torque():
    """Tests that the person holds up the limb, damps, adds effort and walls, and stays
    within their strength."""
    person = PersonModel(damping=2.0)
    gravity = np.array([3.0, -1.0])
    inside = person.get_torque(
        np.zeros(2), np.array([1.0, 0.0]), gravity, LIMITS, "human-left-arm"
    )
    assert inside == pytest.approx([1.0, -1.0])
    assert np.array_equal(
        person.get_torque(np.zeros(2), np.zeros(2), gravity, None, "human-left-arm"),
        gravity,
    )
    effort = person.get_torque(
        np.zeros(2),
        np.zeros(2),
        gravity,
        LIMITS,
        "human-left-arm",
        effort=np.array([1.0, 0.0]),
    )
    assert effort == pytest.approx([4.0, -1.0])
    strong = person.get_torque(
        np.array([2.0, 0.0]), np.zeros(2), np.zeros(2), LIMITS, "human-left-arm"
    )
    assert strong[0] == -person.arm_max_torque
    leg = person.get_torque(
        np.array([2.0, 0.0]), np.zeros(2), np.zeros(2), LIMITS, "human-left-leg"
    )
    assert leg[0] == -person.leg_max_torque
    assert np.array_equal(
        person.get_support_torque(np.array([80.0]), "human-left-arm"), [50.0]
    )
