"""Exact belief updates for the joint limits of the two-link arm."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JointQuery
from able.structs import QueryResponse


class JointLimitBeliefUpdater(
    BeliefUpdater[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], JointQuery, bool
    ]
):
    """Narrows the interval belief over the joint limits [lo1, hi1, lo2, hi2] from
    the arm's motion.

    For each step and joint, the updater compares the commanded joint angle (state
    plus action) with the observed one. Every observed angle lies within the
    limits, so it bounds the lower limit from above and the upper limit from
    below. If the observed angle differs from the commanded one, the motion was
    clipped and the limit on that side equals the observed angle. A query answered
    yes bounds both limits of its joint by the angle; one answered no puts the
    limit on the angle's side (the joint's range always contains 0) past it.
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self._tolerance = tolerance

    def update_from_transition(
        self,
        belief: Belief[NDArray[np.float64]],
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        next_state: NDArray[np.float64],
    ) -> IntervalBelief:
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        intervals = belief.intervals.copy()
        for joint in range(2):
            lo_idx, hi_idx = 2 * joint, 2 * joint + 1
            commanded = state[joint] + action[joint]
            observed = next_state[joint]
            intervals[lo_idx, 1] = min(intervals[lo_idx, 1], observed)
            intervals[hi_idx, 0] = max(intervals[hi_idx, 0], observed)
            if abs(observed - commanded) > self._tolerance:
                clipped_idx = lo_idx if commanded < observed else hi_idx
                intervals[clipped_idx] = [observed, observed]
        return IntervalBelief(intervals)

    def update_from_queries(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[QueryResponse[JointQuery, bool]],
    ) -> Belief[NDArray[np.float64]]:
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        intervals = belief.intervals.copy()
        for datum in data:
            lo_idx, hi_idx = 2 * datum.query.joint, 2 * datum.query.joint + 1
            angle = datum.query.angle
            if datum.response:
                intervals[lo_idx, 1] = min(intervals[lo_idx, 1], angle)
                intervals[hi_idx, 0] = max(intervals[hi_idx, 0], angle)
            elif angle < 0.0:
                intervals[lo_idx, 0] = max(intervals[lo_idx, 0], angle)
            else:
                intervals[hi_idx, 1] = min(intervals[hi_idx, 1], angle)
            for idx in (lo_idx, hi_idx):
                intervals[idx, 0] = min(intervals[idx, 0], intervals[idx, 1])
        return IntervalBelief(intervals)
