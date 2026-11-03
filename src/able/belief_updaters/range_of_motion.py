"""Belief updates for the person's range of motion."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import LimbVariant
from able.envs.limb.model import LimbState
from able.envs.limb.query import LimbQuery, QueryModel, QueryOutcome
from able.structs import QueryResponse

# How much a floor must rise, as a scale, to count; repeated motions reach the same
# angle up to numerical noise.
FLOOR_STEP = 1e-3


class RangeOfMotionBeliefUpdater(
    BeliefUpdater[
        LimbState, NDArray[np.float64], NDArray[np.float64], LimbQuery, QueryOutcome
    ]
):
    """Updates the belief from query responses and from the limb's motion.

    Responses: where a settled person stopped measures their limit, short of it by
    the query model's mean short fall and with its noise. Reaching the target shows
    the limit lies beyond it, which raises the floor. A push that never settled or
    hit the speed cap shows nothing.

    Motion: when the person signals that the limb passed their limit, the signalled
    angle measures the limit. Until then, the farthest the limb has gone raises the
    floor.
    """

    def __init__(
        self, variant: LimbVariant, query_model: QueryModel = QueryModel()
    ) -> None:
        self._magnitudes = variant.default_magnitudes
        self._query_model = query_model

    def update_from_transition(
        self,
        belief: Belief[NDArray[np.float64]],
        state: LimbState,
        action: NDArray[np.float64],
        next_state: LimbState,
    ) -> RangeOfMotionBelief:
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        noise = self._query_model.noise_stddev
        for motion, magnitude in enumerate(self._magnitudes):
            signalled = next_state.signalled[motion]
            if signalled is None:
                reach = next_state.max_reach[motion]
                if reach > belief.lower[motion] + FLOOR_STEP:
                    belief = belief.with_floor(motion, reach)
            elif state.signalled[motion] is None:
                belief = belief.with_measurement(
                    motion, signalled / magnitude, noise / magnitude
                )
        return belief

    def update_from_queries(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[QueryResponse[LimbQuery, QueryOutcome]],
    ) -> RangeOfMotionBelief:
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        model = self._query_model
        for datum in data:
            outcome = datum.response
            magnitude = self._magnitudes[outcome.motion]
            if outcome.peak_velocity >= model.max_velocity:
                continue
            if outcome.reached_target:
                scale = (outcome.answer + model.short_fall) / magnitude
                belief = belief.with_floor(outcome.motion, scale)
            elif outcome.settled:
                belief = belief.with_measurement(
                    outcome.motion,
                    (outcome.answer + model.short_fall) / magnitude,
                    model.noise_stddev / magnitude,
                )
        return belief
