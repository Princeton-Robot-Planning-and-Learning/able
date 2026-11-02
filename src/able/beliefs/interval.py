"""A uniform belief over a box of hidden parameter values."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.beliefs.base import Belief


class IntervalBelief(Belief[NDArray[np.float64]]):
    """Each hidden parameter is uniformly distributed over its own interval.

    The intervals have shape (D, 2), one [low, high] row per parameter. A
    parameter is known exactly once its interval has zero width.
    """

    def __init__(self, intervals: ArrayLike) -> None:
        intervals = np.array(intervals, dtype=np.float64)
        assert intervals.ndim == 2 and intervals.shape[1] == 2, "Need shape (D, 2)"
        assert np.all(intervals[:, 0] <= intervals[:, 1]), "Need low <= high"
        intervals.setflags(write=False)
        self._intervals = intervals

    @property
    def intervals(self) -> NDArray[np.float64]:
        """The read-only (D, 2) array of intervals."""
        return self._intervals

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw each parameter uniformly from its interval."""
        return rng.uniform(self._intervals[:, 0], self._intervals[:, 1])

    def get_total_width(self) -> float:
        """The sum of the interval widths."""
        return float(np.sum(self._intervals[:, 1] - self._intervals[:, 0]))

    def contains(self, hidden_params: NDArray[np.float64]) -> bool:
        """Check whether every parameter lies within its interval."""
        return bool(
            np.all(self._intervals[:, 0] <= hidden_params)
            and np.all(hidden_params <= self._intervals[:, 1])
        )

    def get_discretized_key(self, resolution: float) -> tuple[int, ...]:
        """A hashable summary of the belief: the midpoint and width of each interval,
        rounded to the resolution."""
        midpoints = self._intervals.mean(axis=1)
        widths = self._intervals[:, 1] - self._intervals[:, 0]
        summary = np.concatenate([midpoints, widths]) / resolution
        return tuple(int(v) for v in np.round(summary))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalBelief):
            return NotImplemented
        return np.array_equal(self._intervals, other.intervals)

    def __hash__(self) -> int:
        return hash(self._intervals.tobytes())

    def __repr__(self) -> str:
        return f"IntervalBelief({self._intervals.tolist()})"
