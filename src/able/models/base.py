"""Base class for conditional models, and the data they are trained on."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Dataset:
    """Training rows for a conditional model of y given x.

    The inputs x have shape (N, D) and the outputs y have shape (N, E). Each row
    also has a cost, where lower is better, and a group. Rows in one group, such as
    the steps of one plan, are held out together when a model validates on held-out
    data.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    costs: NDArray[np.float64]
    groups: NDArray[np.int64]

    def __post_init__(self) -> None:
        assert self.x.ndim == 2 and self.y.ndim == 2, "Need 2D inputs and outputs"
        num_rows = len(self.x)
        assert len(self.y) == len(self.costs) == len(self.groups) == num_rows

    @classmethod
    def empty(cls, x_dim: int = 0, y_dim: int = 0) -> "Dataset":
        """A dataset with no rows."""
        return cls(
            x=np.zeros((0, x_dim)),
            y=np.zeros((0, y_dim)),
            costs=np.zeros(0),
            groups=np.zeros(0, dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.x)

    @staticmethod
    def concatenate(datasets: Sequence["Dataset"]) -> "Dataset":
        """All rows of the datasets, in order.

        Groups are renumbered so that groups from different datasets stay
        distinct.
        """
        nonempty = [d for d in datasets if len(d) > 0]
        if not nonempty:
            return Dataset.empty()
        groups = []
        offset = 0
        for dataset in nonempty:
            _, renumbered = np.unique(dataset.groups, return_inverse=True)
            groups.append(renumbered.astype(np.int64) + offset)
            offset += int(renumbered.max()) + 1
        return Dataset(
            x=np.concatenate([d.x for d in nonempty]),
            y=np.concatenate([d.y for d in nonempty]),
            costs=np.concatenate([d.costs for d in nonempty]),
            groups=np.concatenate(groups),
        )


def split_groups(
    groups: NDArray[np.int64],
    holdout_fraction: float,
    min_groups: int,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Choose whole groups to hold out for validation.

    Returns a mask over the rows that is True for held-out rows. Nothing is held
    out when there are fewer than min_groups groups.
    """
    unique = np.unique(groups)
    num_held = int(len(unique) * holdout_fraction)
    if len(unique) < min_groups or num_held == 0:
        return np.zeros(len(groups), dtype=bool)
    held = rng.choice(unique, size=num_held, replace=False)
    return np.isin(groups, held)


class ConditionalModel(ABC):
    """A model of outputs given inputs, trained on a dataset.

    Planners use models to propose outputs, such as action sequences, for an
    input that describes a planning problem. Models are immutable: fit() returns a
    new model and leaves this one unchanged, so planner parameters that hold a model
    never change after they are created.
    """

    @abstractmethod
    def fit(self, dataset: Dataset) -> "ConditionalModel":
        """Return a new model trained on the dataset."""

    @abstractmethod
    def sample(
        self, x: NDArray[np.float64], num_samples: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Propose outputs for the input x with shape (D,).

        Returns an array with shape (n, E) for some n <= num_samples. It has no rows
        if the model has nothing to propose, for example before any training.
        """
