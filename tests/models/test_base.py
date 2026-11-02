"""Tests for models/base.py."""

import numpy as np

from able.models.base import Dataset, split_groups


def _dataset(num_rows: int, groups: list[int]) -> Dataset:
    return Dataset(
        x=np.arange(num_rows, dtype=float).reshape(-1, 1),
        y=np.zeros((num_rows, 2)),
        costs=np.ones(num_rows),
        groups=np.array(groups),
    )


def test_dataset():
    """Tests that concatenation keeps rows in order and groups distinct."""
    empty = Dataset.empty(1, 2)
    assert len(empty) == 0
    assert empty.x.shape == (0, 1) and empty.y.shape == (0, 2)

    first = _dataset(3, [5, 5, 9])
    second = _dataset(2, [5, 7])
    combined = Dataset.concatenate([empty, first, second])
    assert len(combined) == 5
    assert list(combined.x[:, 0]) == [0, 1, 2, 0, 1]
    assert list(combined.groups) == [0, 0, 1, 2, 3]
    assert len(Dataset.concatenate([])) == 0


def test_split_groups():
    """Tests that whole groups are held out, and nothing with too few groups."""
    groups = np.repeat(np.arange(12), 3)
    held = split_groups(groups, 1 / 6, 6, np.random.default_rng(0))
    assert held.sum() == 2 * 3
    for group in np.unique(groups):
        assert len(set(held[groups == group])) == 1
    assert not split_groups(groups[:15], 1 / 6, 6, np.random.default_rng(0)).any()
