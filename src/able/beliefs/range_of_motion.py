"""A belief over range-of-motion scales: one truncated Gaussian per motion."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import truncnorm

from able.beliefs.base import Belief

# How far below the top of the range a floor may rise, so the belief keeps mass.
_FLOOR_MARGIN = 1e-3


@dataclass(frozen=True, eq=False)
class RangeOfMotionBelief(Belief[NDArray[np.float64]]):
    """Each motion's scale is Gaussian, truncated to [lower, high].

    Measurements update the untruncated Gaussian in closed form. The lower bound is
    the population's low or a floor: a scale the person was seen to reach safely,
    so their limit lies beyond it. The population's low and high are per motion.
    """

    mean: NDArray[np.float64]
    stddev: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]
    floor: NDArray[np.float64]

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.mean, self.stddev, self.low, self.high)}
        assert shapes == {self.floor.shape}, "Need one value per motion"
        assert np.all(self.stddev > 0.0), "Need positive standard deviations"
        assert np.all(self.low < self.high)

    @classmethod
    def from_prior(
        cls,
        mean: ArrayLike,
        stddev: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        num_motions: int,
    ) -> "RangeOfMotionBelief":
        """The prior's truncated Gaussian for each motion, with no floor. Each
        parameter is one value for every motion or one per motion."""

        def per_motion(value: ArrayLike) -> NDArray[np.float64]:
            array = np.asarray(value, dtype=np.float64)
            return np.broadcast_to(array, (num_motions,)).copy()

        return cls(
            mean=per_motion(mean),
            stddev=per_motion(stddev),
            low=per_motion(low),
            high=per_motion(high),
            floor=per_motion(low),
        )

    @property
    def lower(self) -> NDArray[np.float64]:
        """The lower bound of each motion's scale."""
        return np.maximum(self.low, self.floor)

    def _truncated(self) -> Any:
        lower = (self.lower - self.mean) / self.stddev
        upper = (self.high - self.mean) / self.stddev
        return truncnorm(lower, upper, loc=self.mean, scale=self.stddev)

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw each scale from its truncated Gaussian."""
        return np.asarray(self._truncated().rvs(random_state=rng), dtype=np.float64)

    @property
    def truncated_mean(self) -> NDArray[np.float64]:
        """The mean of each truncated Gaussian."""
        return np.asarray(self._truncated().mean(), dtype=np.float64)

    @property
    def truncated_stddev(self) -> NDArray[np.float64]:
        """The standard deviation of each truncated Gaussian."""
        return np.asarray(self._truncated().std(), dtype=np.float64)

    def with_floor(self, motion: int, scale: float) -> "RangeOfMotionBelief":
        """The belief after seeing the person reach the scale of a motion safely."""
        floor = self.floor.copy()
        floor[motion] = min(
            max(floor[motion], scale), self.high[motion] - _FLOOR_MARGIN
        )
        return dataclasses.replace(self, floor=floor)

    def with_measurement(
        self, motion: int, scale: float, noise_stddev: float
    ) -> "RangeOfMotionBelief":
        """The belief after measuring a motion's scale with Gaussian noise."""
        prior_var = self.stddev[motion] ** 2
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / noise_stddev**2)
        posterior_mean = posterior_var * (
            self.mean[motion] / prior_var + scale / noise_stddev**2
        )
        mean, stddev = self.mean.copy(), self.stddev.copy()
        mean[motion], stddev[motion] = posterior_mean, np.sqrt(posterior_var)
        return dataclasses.replace(self, mean=mean, stddev=stddev)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RangeOfMotionBelief):
            return NotImplemented
        return (
            np.array_equal(self.mean, other.mean)
            and np.array_equal(self.stddev, other.stddev)
            and np.array_equal(self.floor, other.floor)
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __hash__(self) -> int:
        return hash((self.mean.tobytes(), self.stddev.tobytes(), self.floor.tobytes()))
