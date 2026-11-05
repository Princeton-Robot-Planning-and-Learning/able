"""Features of beliefs for choosing training beliefs in environments without a plan
cache: each hidden parameter's mean and standard deviation, in units of the prior's
standard deviation."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.beliefs.range_of_motion import RangeOfMotionBelief

BeliefFeatures = Callable[[Belief[Any]], NDArray[np.float64]]


def get_moments(
    belief: Belief[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The mean and standard deviation of each hidden parameter under the belief."""
    if isinstance(belief, ParticleBelief):
        return belief.mean, belief.std
    if isinstance(belief, RangeOfMotionBelief):
        return belief.truncated_mean, belief.truncated_stddev
    raise TypeError(f"No moments for {type(belief).__name__}")


def make_moment_features(prior: Belief[Any]) -> BeliefFeatures:
    """A function from a belief to each parameter's mean and standard deviation,
    divided by the parameter's standard deviation under the prior.

    A distance of 1 between features is then one prior standard deviation of a
    parameter's mean or of its uncertainty.
    """
    _, prior_std = get_moments(prior)
    scale = np.where(prior_std > 0.0, prior_std, 1.0)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        mean, std = get_moments(belief)
        return np.concatenate([mean / scale, std / scale])

    return get_features
