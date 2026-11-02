"""Tests for models/neural.py."""

import numpy as np

from able.models.base import Dataset
from able.models.neural import MLPRegressor


def _linear_dataset(num_rows: int, seed: int = 0) -> Dataset:
    """Outputs (2x + 1, -x) for inputs x in [-1, 1], one group per row."""
    x = np.random.default_rng(seed).uniform(-1, 1, size=(num_rows, 1))
    y = np.concatenate([2 * x + 1, -x], axis=1)
    return Dataset(x, y, np.zeros(num_rows), np.arange(num_rows))


def test_mlp_regressor():
    """Tests that the MLP learns a linear map and keeps the best held-out weights."""
    model = MLPRegressor(width=32, depth=2, num_steps=400, batch_size=64)
    rng = np.random.default_rng(0)
    assert model.sample(np.zeros(1), 4, rng).shape == (0, 0)

    fitted = model.fit(_linear_dataset(120))
    assert model.fitted is None
    assert fitted.fitted is not None
    assert 0 < fitted.fitted.best_step <= 400
    assert fitted.fitted.validation_loss < 0.01
    proposal = fitted.sample(np.array([0.5]), 4, rng)
    assert proposal.shape == (1, 2)
    assert np.allclose(proposal[0], [2.0, -0.5], atol=0.1)


def test_mlp_regressor_is_reproducible():
    """Tests that the same seed and data give the same model."""
    model = MLPRegressor(width=16, depth=1, num_steps=50)
    dataset = _linear_dataset(30)
    first = model.fit(dataset).sample(np.array([0.2]), 1, np.random.default_rng(0))
    second = model.fit(dataset).sample(np.array([0.2]), 1, np.random.default_rng(1))
    assert np.array_equal(first, second)
    other = MLPRegressor(width=16, depth=1, num_steps=50, seed=1).fit(dataset)
    assert not np.array_equal(
        first, other.sample(np.array([0.2]), 1, np.random.default_rng(0))
    )


def test_mlp_regressor_without_holdout():
    """Tests training when there are too few groups to hold any out."""
    dataset = _linear_dataset(10)
    dataset = Dataset(dataset.x, dataset.y, dataset.costs, np.zeros(10, np.int64))
    fitted = MLPRegressor(width=16, depth=1, num_steps=20).fit(dataset)
    assert fitted.fitted is not None
    assert np.isnan(fitted.fitted.validation_loss)
    assert fitted.fitted.best_step == 20
