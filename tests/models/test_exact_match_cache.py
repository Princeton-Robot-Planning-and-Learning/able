"""Tests for models/exact_match_cache.py."""

import numpy as np

from able.models.base import Dataset
from able.models.exact_match_cache import ExactMatchCache


def _dataset(x: list[list[float]], costs: list[float]) -> Dataset:
    num_rows = len(x)
    y = np.arange(2 * num_rows, dtype=float).reshape(num_rows, 2)
    return Dataset(np.array(x), y, np.array(costs), np.arange(num_rows))


def test_exact_match_cache():
    """Tests that the cache keeps the cheapest outputs for each rounded input."""
    cache = ExactMatchCache(key_resolution=0.1, max_per_key=2)
    rng = np.random.default_rng(0)
    assert cache.sample(np.zeros(2), 4, rng).shape == (0, 0)

    # The first three inputs round to the same key, and the last one does not.
    dataset = _dataset(
        [[0.0, 1.0], [0.01, 1.02], [-0.02, 0.97], [0.5, 1.0]], [0.3, 0.1, 0.3, 0.2]
    )
    fitted = cache.fit(dataset)
    assert not cache.entries
    assert len(fitted.entries) == 2
    key = fitted.get_key(np.array([0.0, 1.0]))
    # The cheapest row, then the earlier of the two rows that tie.
    assert [cost for cost, _ in fitted.entries[key]] == [0.1, 0.3]
    assert np.array_equal(fitted.entries[key][1][1], dataset.y[0])

    proposals = fitted.sample(np.array([0.02, 1.01]), 4, rng)
    assert np.array_equal(proposals, dataset.y[[1, 0]])
    assert np.array_equal(fitted.sample(np.array([0.0, 1.0]), 1, rng), dataset.y[[1]])
    assert np.array_equal(fitted.sample(np.array([0.5, 1.0]), 4, rng), dataset.y[[3]])
    assert fitted.sample(np.array([0.3, 1.0]), 4, rng).shape == (0, 0)


def test_exact_match_cache_ignores_rng():
    """Tests that proposals do not draw from the rng."""
    fitted = ExactMatchCache().fit(_dataset([[0.0]], [1.0]))
    rng = np.random.default_rng(0)
    fitted.sample(np.zeros(1), 4, rng)
    assert rng.uniform() == np.random.default_rng(0).uniform()
