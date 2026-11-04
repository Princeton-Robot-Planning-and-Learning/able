"""Tests for learners/belief_features.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.beliefs.particles import ParticleBelief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.learners.belief_features import get_moments, make_moment_features


def _particles(center: float, spread: float) -> ParticleBelief:
    rng = np.random.default_rng(0)
    particles = center + spread * rng.standard_normal((500, 2))
    return ParticleBelief(particles, [-10.0, -10.0], [10.0, 10.0])


def test_moments_of_each_belief():
    """Particle beliefs give the particles' moments, and range-of-motion beliefs
    give the truncated Gaussians' moments."""
    belief = _particles(1.0, 0.5)
    mean, std = get_moments(belief)
    assert np.allclose(mean, belief.particles.mean(axis=0))
    assert np.allclose(std, belief.particles.std(axis=0))
    rom = RangeOfMotionBelief.from_prior(1.0, 0.2, 0.5, 1.5, num_motions=3)
    mean, std = get_moments(rom)
    assert mean.shape == std.shape == (3,)
    assert np.allclose(mean, 1.0) and np.all(std < 0.2)
    with pytest.raises(TypeError):
        get_moments(IntervalBelief([[0.0, 1.0], [0.0, 1.0]]))


def test_moment_features_are_in_prior_units():
    """The prior's standard deviations are 1, and a shift of one prior standard
    deviation moves the mean's features by 1."""
    prior = _particles(0.0, 2.0)
    get_features = make_moment_features(prior)
    features = get_features(prior)
    assert features.shape == (4,)
    assert np.allclose(features[2:], 1.0)
    shifted = ParticleBelief(prior.particles + 2.0, prior.low, prior.high)
    assert np.allclose(get_features(shifted)[:2] - features[:2], 1.0, atol=0.05)
