"""Exact belief updates for the joint limits of the two-link arm."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JointQuery
from able.structs import QueryResponse


class JointLimitBeliefUpdater(BeliefUpdater[NDArray[np.float64], JointQuery, bool]):
    """Narrows the interval belief over the joint limits [lo1, hi1, lo2, hi2].

    If a joint can move to the queried angle, its lower limit is at most the angle
    and its upper limit is at least the angle. If it cannot, the angle is either
    below the lower limit or above the upper limit. The belief decides which: the
    angle is above the upper limit if it is at least the smallest possible upper
    limit, and below the lower limit otherwise. This requires every possible lower
    limit of a joint to be at most every possible upper limit, which holds for the
    prior and is preserved by every update.
    """

    def update(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[QueryResponse[JointQuery, bool]],
    ) -> IntervalBelief:
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        intervals = belief.intervals.copy()
        for pair in data:
            lower = intervals[2 * pair.query.joint]
            upper = intervals[2 * pair.query.joint + 1]
            assert (
                lower[1] <= upper[0]
            ), "Every possible lower limit must be at most every possible upper limit"
            angle = pair.query.angle
            if pair.response:
                lower[1] = min(lower[1], angle)
                upper[0] = max(upper[0], angle)
            elif angle >= upper[0]:
                upper[1] = min(upper[1], angle)
            else:
                lower[0] = max(lower[0], angle)
        assert np.all(
            intervals[:, 0] <= intervals[:, 1]
        ), "The responses contradict the belief"
        return IntervalBelief(intervals)
