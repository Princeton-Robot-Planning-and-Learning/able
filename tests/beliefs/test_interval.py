"""Tests for beliefs/interval.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief


def test_interval_belief_construction():
    """Tests that IntervalBelief validates and copies its intervals."""
    with pytest.raises(AssertionError):
        IntervalBelief([0.0, 1.0])
    with pytest.raises(AssertionError):
        IntervalBelief([[1.0, 0.0]])
    source = np.array([[0.0, 1.0], [2.0, 3.0]])
    belief = IntervalBelief(source)
    source[0, 0] = -5.0
    assert belief.intervals[0, 0] == 0.0
    with pytest.raises(ValueError):
        belief.intervals[0, 0] = -5.0


def test_interval_belief_sample():
    """Tests IntervalBelief.sample()."""
    belief = IntervalBelief([[0.0, 1.0], [2.0, 2.0], [-3.0, -1.0]])
    rng = np.random.default_rng(0)
    samples = np.array([belief.sample(rng) for _ in range(200)])
    assert samples.shape == (200, 3)
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 1.0)
    assert np.all(samples[:, 1] == 2.0)
    assert np.all(samples[:, 2] >= -3.0) and np.all(samples[:, 2] <= -1.0)
    assert all(belief.contains(s) for s in samples)


def test_interval_belief_summaries():
    """Tests the width, containment, and key summaries of IntervalBelief."""
    belief = IntervalBelief([[0.0, 1.0], [2.0, 2.0]])
    assert belief.get_total_width() == 1.0
    assert belief.contains(np.array([0.5, 2.0]))
    assert not belief.contains(np.array([0.5, 2.1]))
    assert not belief.contains(np.array([-0.1, 2.0]))

    nearby = IntervalBelief([[0.01, 1.01], [2.0, 2.0]])
    distant = IntervalBelief([[0.0, 0.5], [2.0, 2.0]])
    key = belief.get_discretized_key(resolution=0.1)
    assert key == (5, 20, 10, 0)
    assert nearby.get_discretized_key(resolution=0.1) == key
    assert distant.get_discretized_key(resolution=0.1) != key


def test_interval_belief_equality():
    """Tests equality and hashing of IntervalBelief."""
    belief = IntervalBelief([[0.0, 1.0], [2.0, 3.0]])
    same = IntervalBelief([[0.0, 1.0], [2.0, 3.0]])
    different = IntervalBelief([[0.0, 1.0], [2.0, 2.5]])
    assert belief == same
    assert hash(belief) == hash(same)
    assert belief != different
    assert len({belief, same, different}) == 2
