"""A belief made of equally weighted particles of the hidden parameters."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.beliefs.base import Belief


class ParticleBelief(Belief[NDArray[np.float64]]):
    """Equally weighted particles with shape (N, D), within per-parameter bounds.

    Resampling draws from the belief's own seed, so belief updaters stay
    deterministic and need no random stream.
    """

    def __init__(
        self,
        particles: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        jitter: float = 0.02,
        seed: int = 0,
    ) -> None:
        particles = np.array(particles, dtype=np.float64)
        assert particles.ndim == 2 and len(particles) > 0, "Need shape (N, D)"
        particles.setflags(write=False)
        self._particles = particles
        self._low = np.array(low, dtype=np.float64)
        self._high = np.array(high, dtype=np.float64)
        self._jitter = jitter  # resampling noise, as a fraction of high - low
        self._seed = seed

    @property
    def particles(self) -> NDArray[np.float64]:
        """The read-only (N, D) array of particles."""
        return self._particles

    @property
    def low(self) -> NDArray[np.float64]:
        """The lowest possible value of each parameter."""
        return self._low

    @property
    def high(self) -> NDArray[np.float64]:
        """The highest possible value of each parameter."""
        return self._high

    @property
    def mean(self) -> NDArray[np.float64]:
        """The mean of the particles."""
        return self._particles.mean(axis=0)

    @property
    def std(self) -> NDArray[np.float64]:
        """The standard deviation of the particles."""
        return self._particles.std(axis=0)

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw one particle."""
        return self._particles[rng.integers(len(self._particles))].copy()

    def resample(self, log_weights: NDArray[np.float64]) -> "ParticleBelief":
        """Systematic resampling by the weights, plus jitter.

        Returns this belief if no particle has a finite weight.
        """
        log_weights = np.asarray(log_weights, dtype=np.float64)
        finite = np.isfinite(log_weights)
        if not finite.any():
            return self
        weights = np.where(finite, np.exp(log_weights - log_weights[finite].max()), 0)
        weights /= weights.sum()
        rng = np.random.default_rng(self._seed)
        n = len(self._particles)
        positions = (rng.random() + np.arange(n)) / n
        index = np.searchsorted(np.cumsum(weights), positions).clip(max=n - 1)
        span = self._high - self._low
        noise = rng.normal(0.0, self._jitter, self._particles.shape) * span
        new = np.clip(self._particles[index] + noise, self._low, self._high)
        return ParticleBelief(new, self._low, self._high, self._jitter, self._seed + 1)

    def __repr__(self) -> str:
        return (
            f"ParticleBelief({len(self._particles)} particles, "
            f"mean={self.mean.round(4).tolist()}, std={self.std.round(4).tolist()})"
        )
