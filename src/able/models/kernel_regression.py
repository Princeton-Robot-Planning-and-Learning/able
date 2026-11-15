"""Kernel regression that falls back to the mean output where it has little data."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from able.models.base import ConditionalModel, Dataset


@dataclass(frozen=True, eq=False)
class KernelRegressor(ConditionalModel):
    """Predicts the outputs near an input as a kernel-weighted mean of the training
    outputs, shrunk toward the mean of all training outputs where the training
    inputs are sparse.

    Each training row weighs exp(-d^2 / 2), where d is its distance to the input
    with every column scaled by its bandwidth. The prior weight acts as that many
    extra rows at the input whose outputs are the mean of all training outputs.
    Far from the data the prediction is that mean, and with plenty of close rows it
    is their mean.
    """

    bandwidths: tuple[float, ...] = ()
    prior_weight: float = 1.0
    x: NDArray[np.float64] | None = None
    y: NDArray[np.float64] | None = None

    def fit(self, dataset: Dataset) -> "KernelRegressor":
        """Keep the dataset's rows."""
        assert len(dataset) > 0, "Need at least one row"
        bandwidths = tuple(float(b) for b in self.bandwidths)
        assert len(bandwidths) == dataset.x.shape[1], "Need one bandwidth a column"
        return replace(
            self, bandwidths=bandwidths, x=dataset.x.copy(), y=dataset.y.copy()
        )

    def predict(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """The outputs for the input x with shape (D,), and the fraction of their
        weight that comes from the training rows rather than the prior. Untrained
        models predict no outputs with no weight."""
        if self.x is None or self.y is None:
            return np.zeros(0), 0.0
        scaled = (self.x - x) / np.asarray(self.bandwidths)
        weights = np.exp(-0.5 * np.einsum("ij,ij->i", scaled, scaled))
        total = float(weights.sum())
        prior = self.prior_weight * self.y.mean(axis=0)
        outputs = (weights @ self.y + prior) / (total + self.prior_weight)
        return outputs, total / (total + self.prior_weight)

    def sample(
        self, x: NDArray[np.float64], num_samples: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """The prediction for x as one row, or no rows before training."""
        del rng  # unused because the prediction is deterministic
        if self.x is None or num_samples < 1:
            return np.zeros((0, 0))
        return self.predict(x)[0][None]
