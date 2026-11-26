"""A cache that proposes the cheapest outputs seen for an input, after rounding."""

import dataclasses
import functools
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

    Inputs that round to the same key share their outputs. The resolution is one
    number for every input entry, or one per entry. By default inputs
    with any other key get no proposals, so the cache does not generalize and
    suits small finite sets of planning problems.

    With max_key_distance above 0, an input whose key has no outputs gets the
    outputs of the nearest cached key whose first num_exact_dims entries equal
    its own and whose other entries each differ from its own by at most
    max_key_distance. Distance is the largest difference between entries, or with
    sum_distance, the sum of their differences, with ties going to the smaller sum
    of squared differences and then to the key cached first.
    """

    key_resolution: float | tuple[float, ...] = 0.1
    max_per_key: int = 4
    entries: Mapping[tuple[int, ...], CacheEntries] = field(default_factory=dict)
    max_key_distance: int = 0
    num_exact_dims: int = 0
    sum_distance: bool = False

    def get_key(self, x: NDArray[np.float64]) -> tuple[int, ...]:
        """The key that the input rounds to."""
        resolution = np.asarray(list(np.atleast_1d(self.key_resolution)), dtype=float)
        return tuple(int(v) for v in np.round(x / resolution))

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
        key = self.get_key(x)
        cached = self.entries.get(key, ())
        if not cached and self.max_key_distance > 0:
            nearest = self.get_nearest_key(key)
            if nearest is not None:
                cached = self.entries[nearest]
        cached = cached[:num_samples]
        if not cached:
            return np.zeros((0, 0))
        return np.stack([y for _, y in cached])

    def get_nearest_key(self, key: tuple[int, ...]) -> tuple[int, ...] | None:
        """The nearest cached key within max_key_distance whose first
        num_exact_dims entries equal the key's, or None if there is none."""
        exact = key[: self.num_exact_dims]
        keys = self._keys_by_exact_part.get(exact)
        if keys is None:
            return None
        rest = np.array(key[self.num_exact_dims :])
        differences = np.abs(keys[:, self.num_exact_dims :] - rest)
        if self.sum_distance:
            distances = differences.sum(axis=1)
        else:
            distances = differences.max(axis=1, initial=0)
        within = np.flatnonzero(distances <= self.max_key_distance)
        if len(within) == 0:
            return None
        order = np.lexsort(((differences[within] ** 2).sum(axis=1), distances[within]))
        return tuple(int(v) for v in keys[within[order[0]]])

    @functools.cached_property
    def _keys_by_exact_part(self) -> dict[tuple[int, ...], NDArray[np.int64]]:
        grouped: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for key in self.entries:
            grouped.setdefault(key[: self.num_exact_dims], []).append(key)
        return {part: np.array(keys) for part, keys in grouped.items()}
