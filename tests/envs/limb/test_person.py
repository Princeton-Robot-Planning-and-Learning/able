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


def test_band():
    """Tests that the band resists only within its width inside the limits, damps
    there, and stays within its torque limit."""
    person = PersonModel(band_width=0.15, band_stiffness=30.0, band_damping=1.0)
    still = np.zeros(2)
    assert np.array_equal(
        person.get_band_torque(np.array([0.8, -0.3]), np.array([1.0, 1.0]), LIMITS),
        np.zeros(2),
    )
    torque = person.get_band_torque(np.array([0.95, -0.45]), still, LIMITS)
    assert torque == pytest.approx([-3.0, 3.0])
    moving = person.get_band_torque(np.array([0.95, -0.45]), np.ones(2), LIMITS)
    assert moving == pytest.approx([-4.0, 2.0])
    assert person.get_band_torque(np.array([3.0, 0.0]), still, LIMITS)[0] == -15.0
    with_band = person.get_torque(
        np.array([0.95, 0.0]), still, still, LIMITS, "human-left-arm", band=True
    )
    assert with_band == pytest.approx([-3.0, 0.0])
    assert np.array_equal(
        person.get_torque(
            np.array([0.95, 0.0]), still, still, LIMITS, "human-left-arm"
        ),
        still,
    )


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
