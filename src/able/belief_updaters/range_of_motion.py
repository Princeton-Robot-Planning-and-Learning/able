"""Belief updates for the person's range of motion."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import LimbVariant
from able.envs.limb.model import LimbState
from able.envs.limb.person import PersonModel
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
    floor, and so does the resistance sensed there. Resistance of at least
    num_stddevs times the sensor noise puts the farthest angle that many N m into
    the band, which measures the limit; less resistance raises the floor to where
    the band starts, less the noise's num_stddevs. Each task counts only its
    farthest angle, so a farther one replaces the measurement.
    """

    def __init__(
        self,
        variant: LimbVariant,
        query_model: QueryModel = QueryModel(),
        person: PersonModel = PersonModel(),
        torque_noise: float = 1.0,
        num_stddevs: float = 2.0,
    ) -> None:
        assert torque_noise > 0.0, "Need noisy torque sensing"
        self._magnitudes = variant.default_magnitudes
        self._query_model = query_model
        self._person = person
        self._torque_noise = torque_noise
        self._num_stddevs = num_stddevs

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
                belief = self._update_from_resistance(
                    belief, motion, state.resistance, next_state.resistance
                )
            elif state.signalled[motion] is None:
                belief = belief.with_measurement(
                    motion, signalled / magnitude, noise / magnitude
                )
        return belief

    def _update_from_resistance(
        self,
        belief: RangeOfMotionBelief,
        motion: int,
        before: tuple[tuple[float, float, float] | None, ...],
        after: tuple[tuple[float, float, float] | None, ...],
    ) -> RangeOfMotionBelief:
        old = before[motion] if before else None
        new = after[motion] if after else None
        if new is None or new == old:
            return belief
        magnitude = self._magnitudes[motion]
        threshold = self._num_stddevs * self._torque_noise
        stiffness = self._person.band_stiffness
        noise = self._torque_noise / stiffness / magnitude
        if old is not None and old[1] >= threshold:
            belief = belief.with_measurement(
                motion, self._get_limit(*old) / magnitude, noise, weight=-1.0
            )
        angle, torque, speed = new
        if torque >= threshold:
            return belief.with_measurement(
                motion, self._get_limit(angle, torque, speed) / magnitude, noise
            )
        limit = self._get_limit(angle, torque + threshold, speed)
        floor = min(limit, angle + self._person.band_width) / magnitude
        if floor > belief.lower[motion] + FLOOR_STEP:
            belief = belief.with_floor(motion, floor)
        return belief

    def _get_limit(self, angle: float, torque: float, speed: float) -> float:
        """The limit where the band would resist with the torque at the angle and
        speed toward the limit."""
        person = self._person
        depth = (torque - person.band_damping * speed) / person.band_stiffness
        return angle + person.band_width - depth

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
