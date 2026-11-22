"""A belief made of equally weighted particles of the hidden parameters."""

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.beliefs.base import Belief


class ParticleBelief(Belief[NDArray[np.float64]]):
    """Equally weighted particles with shape (N, D), within per-parameter bounds.

    Resampling draws from the belief's own seed, so belief updaters stay
    deterministic and need no random stream.

    Resampling adds Gaussian noise to the particles, so that they do not collapse
    onto a few values. Each parameter's noise is bandwidth times its standard
    deviation under the weights, and at least min_jitter times high - low, so the
    noise shrinks as the belief narrows. It widens the belief by a factor of
    sqrt(1 + bandwidth**2) per resampling, so evidence fades over about
    1 / bandwidth**2 resamplings. This keeps the belief from growing more confident
    than an imperfect likelihood warrants.

    With independent false, the parameters are correlated across particles, so
    evidence about some of them resamples them all.
    """

    def __init__(
        self,
        particles: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        bandwidth: float = 0.1,
        min_jitter: float = 0.001,
        seed: int = 0,
        independent: bool = True,
    ) -> None:
        particles = np.array(particles, dtype=np.float64)
        assert particles.ndim == 2 and len(particles) > 0, "Need shape (N, D)"
        particles.setflags(write=False)
        self._particles = particles
        self._low = np.array(low, dtype=np.float64)
        self._high = np.array(high, dtype=np.float64)
        self._bandwidth = bandwidth
        self._min_jitter = min_jitter
        self._seed = seed
        self._independent = independent

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

    def resample(
        self, log_weights: NDArray[np.float64], dims: Sequence[int] | None = None
    ) -> "ParticleBelief":
        """Systematic resampling by the weights, plus noise.

        If dims is given, the weights depend on those parameters alone, and only they
        are resampled. The others keep their particles, as if independent of them, so
        that resampling for unrelated evidence does not narrow them. A belief whose
        parameters are not independent resamples them all.

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
        cols = slice(None) if dims is None or not self._independent else list(dims)
        old = self._particles[:, cols]
        low, high = self._low[cols], self._high[cols]
        mean = weights @ old
        std = np.sqrt(weights @ (old - mean) ** 2)
        scale = np.maximum(self._bandwidth * std, self._min_jitter * (high - low))
        noise = rng.normal(0.0, 1.0, old.shape) * scale
        new = self._particles.copy()
        new[:, cols] = np.clip(old[index] + noise, low, high)
        return ParticleBelief(
            new,
            self._low,
            self._high,
            self._bandwidth,
            self._min_jitter,
            self._seed + 1,
            self._independent,
        )

    def reweight(
        self,
        log_likelihood: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        dims: Sequence[int] | None = None,
        min_ess: float = 0.5,
        max_steps: int = 20,
    ) -> "ParticleBelief":
        """Resamples by the likelihood in steps, each by the largest power of it that
        keeps the effective sample size at least min_ess of the particles.

        Each step's noise moves the particles before the next, so a sharp likelihood
        narrows the belief gradually instead of onto the few particles nearest its peak.
        dims is as in resample.
        """
        belief, remaining = self, 1.0
        for step in range(max_steps):
            log_weights = log_likelihood(belief.particles)
            power = remaining
            if step < max_steps - 1:
                power = _tempering_power(log_weights, remaining, min_ess)
            belief = belief.resample(power * log_weights, dims)
            remaining -= power
            if remaining <= 0.0:
                break
        return belief

    def __repr__(self) -> str:
        return (
            f"ParticleBelief({len(self._particles)} particles, "
            f"mean={self.mean.round(4).tolist()}, std={self.std.round(4).tolist()})"
        )


def _tempering_power(
    log_weights: NDArray[np.float64], most: float, min_ess: float
) -> float:
    """The largest power up to most at which the weights keep an effective sample size
    of min_ess of the particles, or most if no power does."""
    finite = log_weights[np.isfinite(log_weights)]
    if len(finite) == 0:
        return most
    centered = finite - finite.max()

    def ess(power: float) -> float:
        weights = np.exp(power * centered)
        return float(weights.sum() ** 2 / (weights**2).sum() / len(log_weights))

    if ess(most) >= min_ess or ess(1e-9 * most) < min_ess:
        return most
    low, high = 1e-9 * most, most
    for _ in range(40):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if ess(mid) >= min_ess else (low, mid)
    return low
