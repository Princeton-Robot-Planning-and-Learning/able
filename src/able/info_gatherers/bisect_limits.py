"""Gathering information about the arm's joint limits by bisecting them."""

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JointQuery
from able.info_gatherers.base import SequentialGatherer


class BisectLimitGatherer(SequentialGatherer[NDArray[np.float64], JointQuery, bool]):
    """Asks whether a joint reaches the midpoint of the widest limit interval.

    Candidates ask at each fraction of every limit's interval that is wider than
    min_width. Gathering stops once every limit is known within min_width.
    """

    def __init__(
        self,
        belief_updater: JointLimitBeliefUpdater,
        fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
        min_width: float = 1e-3,
    ) -> None:
        super().__init__(belief_updater)
        self._fractions = fractions
        self._min_width = min_width

    def choose(
        self,
        belief: Belief[NDArray[np.float64]],
        rng: np.random.Generator | None = None,
    ) -> JointQuery | None:
        """The midpoint of the widest limit interval, or None if all are known."""
        del rng  # unused because the choice is deterministic
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        widths = belief.intervals[:, 1] - belief.intervals[:, 0]
        limit = int(np.argmax(widths))
        if widths[limit] <= self._min_width:
            return None
        return _ask_at(belief, limit, 0.5)

    def get_candidates(
        self,
        belief: Belief[NDArray[np.float64]],
        rng: np.random.Generator | None = None,
    ) -> list[JointQuery]:
        """Each fraction of every limit interval not yet known."""
        del rng  # unused because the candidates are fixed
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        widths = belief.intervals[:, 1] - belief.intervals[:, 0]
        return [
            _ask_at(belief, limit, fraction)
            for limit in np.flatnonzero(widths > self._min_width)
            for fraction in self._fractions
        ]


def _ask_at(belief: IntervalBelief, limit: int, fraction: float) -> JointQuery:
    low, high = belief.intervals[limit]
    return JointQuery(joint=limit // 2, angle=float(low + fraction * (high - low)))
