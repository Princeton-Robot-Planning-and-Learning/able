"""Tests for models/flow_matching.py."""

import numpy as np
import torch

from able.models.base import Dataset
from able.models.flow_matching import FlowMatchingModel, embed_time
from able.models.neural import MLPRegressor


def _two_mode_dataset(num_rows: int) -> Dataset:
    """For an input x in [0, 1], the output is x + 1 or -(x + 1) with equal
    probability."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(num_rows, 1))
    signs = rng.choice([-1.0, 1.0], size=(num_rows, 1))
    return Dataset(x, signs * (x + 1), np.zeros(num_rows), np.arange(num_rows))


def test_embed_time():
    """Tests the shape and range of the time features."""
    features = embed_time(torch.tensor([0.0, 0.5, 1.0]), 8)
    assert features.shape == (3, 8)
    assert torch.all(features.abs() <= 1.0)


def test_flow_matching_covers_both_modes():
    """Tests that draws cover both modes, where regression predicts their mean."""
    dataset = _two_mode_dataset(400)
    options = {"width": 128, "depth": 2, "num_steps": 3000, "batch_size": 128}
    flow = FlowMatchingModel(**options).fit(dataset)
    draws = flow.sample(np.array([0.5]), 200, np.random.default_rng(0))
    assert draws.shape == (200, 1)
    assert np.mean(np.abs(np.abs(draws) - 1.5) < 0.3) > 0.8
    assert 0.25 < np.mean(draws > 0) < 0.75

    regression = MLPRegressor(**options).fit(dataset)
    mean = regression.sample(np.array([0.5]), 200, np.random.default_rng(0))
    assert mean.shape == (1, 1)
    assert abs(mean[0, 0]) < 0.5


def test_flow_matching_is_reproducible():
    """Tests that draws depend only on the seed, the data, and the rng."""
    dataset = _two_mode_dataset(50)
    model = FlowMatchingModel(width=16, depth=1, num_steps=30)
    first = model.fit(dataset).sample(np.array([0.3]), 5, np.random.default_rng(0))
    second = model.fit(dataset).sample(np.array([0.3]), 5, np.random.default_rng(0))
    assert np.array_equal(first, second)
    third = model.fit(dataset).sample(np.array([0.3]), 5, np.random.default_rng(1))
    assert not np.array_equal(first, third)
