"""Tests for envs/limb/human.py."""

import numpy as np
import pytest
from kinder.envs.dynamic3d.limbs import DEFAULT_RANGE_OF_MOTION, get_human_joint_limits

from able.envs.limb.human import VARIANTS, RangeOfMotionPrior, get_variant


def test_theta_scales_the_active_bounds():
    """Tests that theta scales the bound of the joint each motion governs, and
    nothing else."""
    variant = get_variant("human-left-arm")
    defaults = DEFAULT_RANGE_OF_MOTION.as_dict()
    theta = np.array([0.9, 0.8, 0.7])
    lower, upper = variant.get_joint_limits(theta)
    for i, motion in enumerate(variant.active_motions):
        joint, sign = variant.active_joints[i], variant.active_signs[i]
        bound = upper[joint] if sign > 0 else -lower[joint]
        assert bound == pytest.approx(np.deg2rad(defaults[motion] * theta[i]))
    assert variant.get_bounds(theta) == pytest.approx(
        [
            upper[j] if s > 0 else lower[j]
            for j, s in zip(variant.active_joints, variant.active_signs)
        ]
    )
    default_lower, default_upper = variant.get_joint_limits(np.ones(3))
    for joint in variant.inactive_joints:
        assert lower[joint] == default_lower[joint]
        assert upper[joint] == default_upper[joint]


@pytest.mark.parametrize("name", VARIANTS)
def test_every_variant_bounds_the_joints_it_claims(name):
    """Tests that scaling one active motion moves exactly the bound the variant
    records."""
    variant = get_variant(name)
    base_lower, base_upper = get_human_joint_limits(variant.limb_name)
    for motion, joint, sign in zip(
        variant.active_motions, variant.active_joints, variant.active_signs
    ):
        rom = DEFAULT_RANGE_OF_MOTION.scaled({motion: 0.5})
        lower, upper = get_human_joint_limits(variant.limb_name, rom)
        assert np.isclose(lower[joint], base_lower[joint]) == (sign > 0)
        assert np.isclose(upper[joint], base_upper[joint]) == (sign < 0)
        others = [j for j in range(len(lower)) if j != joint]
        assert np.allclose(lower[others], base_lower[others])
        assert np.allclose(upper[others], base_upper[others])


def test_variants():
    """Tests that the limb decides the active motions and unknown names fail."""
    assert len(VARIANTS) == 16
    leg = get_variant("bed-right-leg")
    assert leg.limb_name == "human-right-leg"
    assert leg.active_motions == ("hip_flexion", "hip_abduction", "knee_flexion")
    assert leg.inactive_joints == (2, 4, 5)
    assert get_variant("isolated-left-arm").active_signs[0] == +1
    with pytest.raises(ValueError, match="Unknown variant"):
        get_variant("sofa-third-arm")


def test_prior():
    """Tests that prior draws stay in range and center near the mean."""
    prior = RangeOfMotionPrior()
    draws = np.stack([prior.sample(3, np.random.default_rng(s)) for s in range(300)])
    assert draws.shape == (300, 3)
    assert np.all((prior.low <= draws) & (draws <= prior.high))
    assert abs(draws.mean() - prior.mean) < 0.03
