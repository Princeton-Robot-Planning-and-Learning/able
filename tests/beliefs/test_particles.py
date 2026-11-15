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


def test_resample_noise_follows_the_spread():
    """Without evidence, resampling widens a narrow belief by its bandwidth's factor,
    not by noise on the scale of the bounds, and moves every particle."""
    rng = np.random.default_rng(0)
    narrow = ParticleBelief(rng.normal(0.5, 0.01, (500, 2)), [0, 0], [1, 1])
    new = narrow.resample(np.zeros(500))
    assert np.allclose(new.std, narrow.std * np.sqrt(1.01), rtol=0.05)
    assert np.allclose(new.mean, narrow.mean, atol=0.002)
    assert len(np.unique(new.particles[:, 0])) == 500


def test_resample_keeps_narrowing_with_evidence():
    """Repeated measurements keep narrowing the belief, about as Bayes' rule does, and
    its mean stays within a few of its standard deviations of the truth."""
    belief = _belief()
    rng = np.random.default_rng(1)
    for _ in range(100):
        measurement = 0.3 + rng.normal(0.0, 0.05)
        belief = belief.resample(
            -0.5 * ((belief.particles[:, 0] - measurement) / 0.05) ** 2
        )
    # Bayes' rule gives 0.05 / sqrt(100) = 0.005; fading evidence leaves a bit
    # more. The old fixed noise of 0.02 stopped near 0.035.
    assert 0.004 < belief.std[0] < 0.009
    assert abs(belief.mean[0] - 0.3) < 3 * belief.std[0]


def test_resample_onto_one_particle_keeps_some_spread():
    """If one particle takes all the weight, the noise's floor still spreads the
    particles around it."""
    belief = _belief()
    log_weights = np.full(500, -np.inf)
    log_weights[7] = 0.0
    new = belief.resample(log_weights)
    assert np.allclose(new.mean, belief.particles[7], atol=0.001)
    assert np.allclose(new.std, 0.001, rtol=0.2)


def test_resample_without_finite_weights_keeps_the_belief():
    """If nothing explains the data, the belief is unchanged."""
    belief = _belief()
    assert belief.resample(np.full(500, -np.inf)) is belief
