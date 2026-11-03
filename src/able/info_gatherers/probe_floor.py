"""Gathering information about the floors' friction with probe pushes."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.base import Environment
from able.envs.friction.query import FrictionQuery, PushRecord, probe_queries
from able.envs.friction.world import PuckWorld
from able.info_gatherers.base import InformationGatherer
from able.structs import Budget, QueryResponse


class ProbeFloorGatherer(
    InformationGatherer[NDArray[np.float64], FrictionQuery, PushRecord]
):
    """Probes the floor the belief is least sure of.

    Each floor has one probe pose, searched for the belief's lowest possible
    friction. Each query is chosen under the belief updated with the earlier
    responses, and gathering stops early once every floor's standard deviation is
    within min_std (about the resampling jitter). One iteration of the budget is one
    query.
    """

    def __init__(
        self,
        world: PuckWorld,
        belief_updater: FrictionBeliefUpdater,
        speed: float = 0.8,
        min_std: float = 0.005,
    ) -> None:
        self._world = world
        self._belief_updater = belief_updater
        self._speed = speed
        self._min_std = min_std

    def choose(self, belief: ParticleBelief) -> FrictionQuery | None:
        """The probe of the least certain floor, or None if every floor is known."""
        std = belief.std
        floor = int(np.argmax(std))
        if std[floor] <= self._min_std:
            return None
        return probe_queries(self._world, belief.low, self._speed)[floor]

    def gather(
        self,
        belief: Belief[NDArray[np.float64]],
        env: Environment[Any, Any, Any, FrictionQuery, PushRecord],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[FrictionQuery, PushRecord]]:
        """Ask queries until the budget runs out or every floor is known."""
        del rng  # unused because the probes are fixed
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Information gathering needs a finite budget"
        data: list[QueryResponse[FrictionQuery, PushRecord]] = []
        start_time = time.perf_counter()
        while not budget.is_exhausted(time.perf_counter() - start_time, len(data)):
            assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
            query = self.choose(belief)
            if query is None:
                break
            datum = QueryResponse(query, env.ask(query))
            data.append(datum)
            belief = self._belief_updater.update_from_queries(belief, [datum])
        return data
