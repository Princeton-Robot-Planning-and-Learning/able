"""Tests for beliefs/range_of_motion.py."""

import numpy as np
import pytest

from able.beliefs.range_of_motion import RangeOfMotionBelief


def _prior() -> RangeOfMotionBelief:
    return RangeOfMotionBelief.from_prior(0.9, 0.15, 0.7, 1.2, num_motions=3)


def test_samples_stay_in_bounds():
    """Tests that samples stay within [lower, high] and center near the mean."""
    belief = _prior()
    rng = np.random.default_rng(0)
    draws = np.stack([belief.sample(rng) for _ in range(2000)])
    assert draws.shape == (2000, 3)
    assert np.all((0.7 <= draws) & (draws <= 1.2))
    assert np.allclose(draws.mean(axis=0), belief.truncated_mean, atol=0.02)


def test_measurements_move_toward_the_truth_and_narrow():
    """Tests that repeated noisy measurements converge to the truth."""
    truth = np.array([0.85, 1.05, 0.75])
    belief = _prior()
    rng = np.random.default_rng(0)
    for _ in range(20):
        for motion, scale in enumerate(truth):
            belief = belief.with_measurement(motion, scale + rng.normal(0, 0.05), 0.05)
    assert np.all(np.abs(belief.mean - truth) < 0.03)
    assert np.all(belief.stddev < 0.15)
    exact = _prior().with_measurement(0, 0.83, noise_stddev=1e-6)
    assert exact.mean[0] == pytest.approx(0.83, abs=1e-6)
    assert exact.mean[1:] == pytest.approx([0.9, 0.9])


def test_a_floor_truncates_and_narrows():
    """Tests that reaching a scale safely rules out every narrower person."""
    belief = _prior()
    raised = belief.with_floor(1, 1.0)
    assert np.allclose(raised.lower, [0.7, 1.0, 0.7])
    assert raised.truncated_stddev[1] < belief.truncated_stddev[1]
    assert raised.truncated_mean[1] > belief.truncated_mean[1]
    draws = np.stack([raised.sample(np.random.default_rng(i)) for i in range(200)])
    assert draws[:, 1].min() >= 1.0
    # A lower reach keeps the floor, and the floor stays below the top.
    assert raised.with_floor(1, 0.8) == raised
    assert raised.with_floor(1, 5.0).lower[1] < 1.2
    # A measurement keeps the floor.
    assert np.array_equal(raised.with_measurement(0, 1.0, 0.1).floor, raised.floor)


def test_equality():
    """Tests that beliefs compare and hash by value and are never modified."""
    belief = _prior()
    assert belief == _prior() and hash(belief) == hash(_prior())
    raised = belief.with_floor(0, 0.9)
    assert raised != belief
    assert np.allclose(belief.floor, 0.7)
