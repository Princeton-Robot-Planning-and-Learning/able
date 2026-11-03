"""Tests for beliefs/particles.py."""

import numpy as np

from able.beliefs.particles import ParticleBelief


def _belief(seed: int = 0) -> ParticleBelief:
    rng = np.random.default_rng(0)
    return ParticleBelief(rng.uniform(0, 1, (500, 2)), [0, 0], [1, 1], seed=seed)


def test_sample_mean_and_std():
    """Samples are particles, and the moments are the particles' moments."""
    belief = _belief()
    sample = belief.sample(np.random.default_rng(1))
    assert any(np.array_equal(sample, p) for p in belief.particles)
    assert np.allclose(belief.mean, belief.particles.mean(0))
    assert np.allclose(belief.std, belief.particles.std(0))
    assert not belief.particles.flags.writeable


def test_resample_follows_the_weights():
    """Resampling keeps likely particles, within the bounds, and is deterministic."""
    belief = _belief()
    log_weights = -0.5 * ((belief.particles[:, 0] - 0.2) / 0.05) ** 2
    new = belief.resample(log_weights)
    assert abs(new.mean[0] - 0.2) < 0.05 and new.std[0] < belief.std[0]
    assert np.all((new.particles >= 0) & (new.particles <= 1))
    assert np.array_equal(new.particles, belief.resample(log_weights).particles)
    # The next resampling uses a new seed.
    assert not np.array_equal(
        new.resample(np.zeros(500)).particles, belief.resample(np.zeros(500)).particles
    )


def test_resample_without_finite_weights_keeps_the_belief():
    """If nothing explains the data, the belief is unchanged."""
    belief = _belief()
    assert belief.resample(np.full(500, -np.inf)) is belief
