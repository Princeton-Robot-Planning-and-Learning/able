"""Tests for models/kernel_regression.py."""

import numpy as np
import pytest

from able.models.base import Dataset
from able.models.kernel_regression import KernelRegressor


def _rows(x, y) -> Dataset:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return Dataset(x, y, np.zeros(len(x)), np.arange(len(x)))


def test_predictions_shrink_toward_the_mean_far_from_the_data():
    """Near many rows the prediction is their mean; far from all rows it is the
    mean of every output; the weight says how much came from nearby rows."""
    x = [[0.0, 0.0]] * 50 + [[10.0, 10.0]] * 50
    y = [[1.0]] * 50 + [[3.0]] * 50
    model = KernelRegressor(bandwidths=(1.0, 1.0), prior_weight=1.0).fit(_rows(x, y))
    near, weight = model.predict(np.zeros(2))
    assert near[0] == pytest.approx((50 * 1.0 + 2.0) / 51)
    assert weight == pytest.approx(50 / 51)
    far, weight = model.predict(np.array([5.0, -20.0]))
    assert far[0] == pytest.approx(2.0) and weight == pytest.approx(0.0)


def test_bandwidths_scale_each_column():
    """A row one bandwidth away weighs exp(-1/2)."""
    model = KernelRegressor(bandwidths=(2.0, 0.5), prior_weight=1.0)
    model = model.fit(_rows([[0.0, 0.0]], [[4.0]]))
    _, weight = model.predict(np.array([2.0, 0.0]))
    assert weight == pytest.approx(np.exp(-0.5) / (np.exp(-0.5) + 1.0))
    _, weight = model.predict(np.array([0.0, 0.5]))
    assert weight == pytest.approx(np.exp(-0.5) / (np.exp(-0.5) + 1.0))
    with pytest.raises(AssertionError):
        KernelRegressor(bandwidths=(1.0,)).fit(_rows([[0.0, 0.0]], [[4.0]]))


def test_untrained_and_sampled_predictions():
    """An untrained model predicts nothing; a trained one samples its
    prediction once."""
    rng = np.random.default_rng(0)
    model = KernelRegressor(bandwidths=(1.0,))
    assert model.sample(np.zeros(1), 3, rng).shape[0] == 0
    assert model.predict(np.zeros(1))[1] == 0.0
    trained = model.fit(_rows([[0.0]], [[2.0, -1.0]]))
    assert model.x is None and trained.bandwidths == (1.0,)
    draws = trained.sample(np.zeros(1), 3, rng)
    assert draws.shape == (1, 2) and np.allclose(draws, [[2.0, -1.0]])
