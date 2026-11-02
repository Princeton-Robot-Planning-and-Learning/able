"""Exact belief updates for the joint limits of the two-link arm."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.structs import Trajectory


class JointLimitBeliefUpdater(
    BeliefUpdater[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """Narrows the interval belief over the joint limits [lo1, hi1, lo2, hi2].

    For each step and joint, the updater compares the commanded joint angle (state
    plus action) with the observed one. Every observed angle lies within the
    limits, so it bounds the lower limit from above and the upper limit from
    below. If the observed angle differs from the commanded one, the motion was
    clipped and the limit on that side equals the observed angle.
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self._tolerance = tolerance

    def update(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[Trajectory[NDArray[np.float64], NDArray[np.float64]]],
    ) -> IntervalBelief:
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        intervals = belief.intervals.copy()
        for trajectory in data:
            for t, action in enumerate(trajectory.actions):
                state = trajectory.states[t]
                next_state = trajectory.states[t + 1]
                for joint in range(2):
                    self._update_joint(
                        intervals,
                        joint,
                        commanded=state[joint] + action[joint],
                        observed=next_state[joint],
                    )
        return IntervalBelief(intervals)

    def _update_joint(
        self,
        intervals: NDArray[np.float64],
        joint: int,
        commanded: float,
        observed: float,
    ) -> None:
        lo_idx, hi_idx = 2 * joint, 2 * joint + 1
        intervals[lo_idx, 1] = min(intervals[lo_idx, 1], observed)
        intervals[hi_idx, 0] = max(intervals[hi_idx, 0], observed)
        if abs(observed - commanded) > self._tolerance:
            clipped_idx = lo_idx if commanded < observed else hi_idx
            intervals[clipped_idx] = [observed, observed]
