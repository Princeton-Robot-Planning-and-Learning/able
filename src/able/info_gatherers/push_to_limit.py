"""Gathering information about the range of motion by asking the person to push."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.range_of_motion import RangeOfMotionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.base import Environment
from able.envs.limb.human import LimbVariant
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.info_gatherers.base import InformationGatherer
from able.structs import Budget, QueryResponse


class PushToLimitGatherer(
    InformationGatherer[NDArray[np.float64], LimbQuery, QueryOutcome]
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
        self._magnitudes = variant.default_magnitudes
        self._belief_updater = belief_updater
        self._probe_band = probe_band

    def choose(
        self, belief: RangeOfMotionBelief, rng: np.random.Generator
    ) -> LimbQuery:
        """The next query under the belief."""
        motion = int(np.argmax(belief.stddev))
        limit = self._magnitudes[motion] * max(1.0, float(belief.lower[motion]))
        noise = rng.uniform(-self._probe_band, self._probe_band)
        return LimbQuery(motion, float(limit + noise))

    def gather(
        self,
        belief: Belief[NDArray[np.float64]],
        env: Environment[Any, Any, Any, LimbQuery, QueryOutcome],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[LimbQuery, QueryOutcome]]:
        """Ask queries until the budget runs out."""
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Information gathering needs a finite budget"
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        data: list[QueryResponse[LimbQuery, QueryOutcome]] = []
        start_time = time.perf_counter()
        while not budget.is_exhausted(time.perf_counter() - start_time, len(data)):
            query = self.choose(belief, rng)
            datum = QueryResponse(query, env.ask(query))
            data.append(datum)
            belief = self._belief_updater.update_from_queries(belief, [datum])
        return data
