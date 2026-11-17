"""Gathering information about the floors' friction with probe pushes."""

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.friction.query import FrictionQuery, PushRecord, probe_queries
from able.envs.friction.world import PuckWorld
from able.info_gatherers.base import SequentialGatherer


class ProbeFloorGatherer(
    SequentialGatherer[NDArray[np.float64], FrictionQuery, PushRecord]
):
    """Probes the floor the belief is least sure of.

    Each floor has one probe pose, searched for the belief's lowest possible
    friction. Each query is chosen under the belief updated with the earlier
    responses, and gathering stops early once every floor's standard deviation is
    within min_std (about a tenth of the prior's). One iteration of the budget is
    one query.
    """

    def __init__(
        self,
        world: PuckWorld,
        belief_updater: FrictionBeliefUpdater,
        speed: float = 0.8,
        min_std: float = 0.005,
    ) -> None:
        super().__init__(belief_updater)
        self._world = world
        self._speed = speed
        self._min_std = min_std

    def choose(
        self,
        belief: Belief[NDArray[np.float64]],
        rng: np.random.Generator | None = None,
    ) -> FrictionQuery | None:
        """The probe of the least certain floor, or None if every floor is known."""
        del rng  # unused because the probes are fixed
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        std = belief.std
        floor = int(np.argmax(std))
        if std[floor] <= self._min_std:
            return None
        return probe_queries(self._world, belief.low, self._speed)[floor]

    def get_candidates(
        self,
        belief: Belief[NDArray[np.float64]],
        rng: np.random.Generator | None = None,
    ) -> list[FrictionQuery]:
        """The probe of every floor not yet known."""
        del rng  # unused because the probes are fixed
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        probes = probe_queries(self._world, belief.low, self._speed)
        return [probes[f] for f in np.flatnonzero(belief.std > self._min_std)]
