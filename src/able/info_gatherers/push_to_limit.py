"""Gathering information about the range of motion by asking the person to push."""

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.range_of_motion import RangeOfMotionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import LimbVariant
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.info_gatherers.base import SequentialGatherer


class PushToLimitGatherer(
    SequentialGatherer[NDArray[np.float64], LimbQuery, QueryOutcome]
):
    """Asks about the least certain motion, with a target near its limit.

    The target is the motion's normal range, or its floor if that is higher, plus
    uniform noise within probe_band (rad). Each query is chosen under the belief
    updated with the earlier responses. One iteration of the budget is one query.
    """

    def __init__(
        self,
        variant: LimbVariant,
        belief_updater: RangeOfMotionBeliefUpdater,
        probe_band: float = 0.3,
    ) -> None:
        super().__init__(belief_updater)
        self._magnitudes = variant.default_magnitudes
        self._probe_band = probe_band

    def choose(
        self, belief: Belief[NDArray[np.float64]], rng: np.random.Generator
    ) -> LimbQuery:
        """The next query under the belief."""
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        motion = int(np.argmax(belief.stddev))
        noise = rng.uniform(-self._probe_band, self._probe_band)
        return LimbQuery(motion, self._get_limit(belief, motion) + noise)

    def get_candidates(
        self, belief: Belief[NDArray[np.float64]], rng: np.random.Generator
    ) -> list[LimbQuery]:
        """Every motion, with targets at the ends and middle of the probe band."""
        del rng  # unused because the targets are fixed
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        offsets = np.linspace(-self._probe_band, self._probe_band, 3)
        return [
            LimbQuery(motion, self._get_limit(belief, motion) + float(offset))
            for motion in range(len(self._magnitudes))
            for offset in offsets
        ]

    def _get_limit(self, belief: RangeOfMotionBelief, motion: int) -> float:
        return float(self._magnitudes[motion] * max(1.0, float(belief.lower[motion])))
