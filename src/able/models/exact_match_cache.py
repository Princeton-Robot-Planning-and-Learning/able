"""A cache that proposes the cheapest outputs seen for an input, after rounding."""

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from able.models.base import ConditionalModel, Dataset

# The cached outputs for one key: (cost, output) pairs, cheapest first.
CacheEntries = tuple[tuple[float, NDArray[np.float64]], ...]


@dataclass(frozen=True, eq=False)
class ExactMatchCache(ConditionalModel):
    """Remembers the cheapest outputs for each input, rounded to a resolution.

    Inputs that round to the same key share their outputs, and inputs with any
    other key get no proposals. The cache does not generalize, so it suits small
    finite sets of planning problems.
    """

    key_resolution: float = 0.1
    max_per_key: int = 4
    entries: Mapping[tuple[int, ...], CacheEntries] = field(default_factory=dict)

    def get_key(self, x: NDArray[np.float64]) -> tuple[int, ...]:
        """The key that the input rounds to."""
        return tuple(int(v) for v in np.round(x / self.key_resolution))

    def fit(self, dataset: Dataset) -> "ExactMatchCache":
        """Cache the cheapest max_per_key outputs of the rows for each key.

        Rows are added in order, and among rows with equal costs the earlier ones
        are kept.
        """
        entries: dict[tuple[int, ...], CacheEntries] = {}
        for x, y, cost in zip(dataset.x, dataset.y, dataset.costs):
            key = self.get_key(x)
            candidates = list(entries.get(key, ())) + [(float(cost), y.copy())]
            candidates.sort(key=lambda entry: entry[0])
            entries[key] = tuple(candidates[: self.max_per_key])
        return dataclasses.replace(self, entries=entries)

    def sample(
        self, x: NDArray[np.float64], num_samples: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """The cheapest cached outputs for the input's key, up to num_samples.

        The rng is not used.
        """
        del rng  # unused because the proposals are the cached outputs
        cached = self.entries.get(self.get_key(x), ())[:num_samples]
        if not cached:
            return np.zeros((0, 0))
        return np.stack([y for _, y in cached])
