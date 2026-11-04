"""Gathering information about the person's reach by presenting the gripper."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.human_reach import ReachBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.base import Environment
from able.envs.human_reach.query import (
    QuerySimulator,
    ReachQuery,
    ReachResponse,
    bedside_directions,
    reach_distance,
)
from able.envs.human_reach.scene import BedsideStocking3DEnvConfig
from able.info_gatherers.base import InformationGatherer
from able.structs import Budget, QueryResponse


class ReachQueryGatherer(
    InformationGatherer[NDArray[np.float64], ReachQuery, ReachResponse]
):
    """Asks along the direction where the belief disagrees most about the reach.

    Directions point from the shoulder to the bedside slots. For each direction,
    the candidate distances nearest the belief's num_points quantiles of P(reached)
    form a query, and the direction whose answers are most uncertain (summed
    binary entropy) wins. Gathering stops once no query is uncertain by min_entropy.
    Whether the robot can present a point is checked by IK in a simulator of the
    gatherer's own and cached, since it never changes. Each query is chosen under
    the belief updated with the earlier responses. One iteration of the budget is
    one query.
    """

    def __init__(
        self,
        belief_updater: ReachBeliefUpdater,
        config: BedsideStocking3DEnvConfig = BedsideStocking3DEnvConfig(),
        num_points: int = 4,
        grid_step: float = 0.025,
        min_entropy: float = 0.1,
    ) -> None:
        self._belief_updater = belief_updater
        self._config = config
        self._num_points = num_points
        self._grid_step = grid_step
        self._min_entropy = min_entropy
        self._presenter = QuerySimulator(config)
        self._directions: NDArray[np.float64] | None = None
        self._known: dict[tuple[int, int], bool] = {}

    @property
    def directions(self) -> NDArray[np.float64]:
        """Unit vectors from the shoulder to every bedside slot."""
        if self._directions is None:
            shoulder = self._presenter.sim.shoulder
            self._directions = bedside_directions(self._config, shoulder)
        return self._directions

    @property
    def distances(self) -> NDArray[np.float64]:
        """Candidate distances along every direction, spanning the prior's reach."""
        lb = np.array(self._config.reach_theta_lb)
        ub = np.array(self._config.reach_theta_ub)
        r = reach_distance(self.directions, np.stack([lb, ub])[:, None])
        return np.arange(r[0].min(), r[1].max() + 1e-9, self._grid_step)

    def close(self) -> None:
        """Close the gatherer's simulator."""
        self._presenter.close()

    def presentable(self, k: int, j: int) -> bool:
        """Whether the robot can present distance j along direction k (cached)."""
        if (k, j) not in self._known:
            sim = self._presenter.sim
            point = sim.shoulder + self.distances[j] * self.directions[k]
            self._known[k, j] = self._presenter.can_present(point)
        return self._known[k, j]

    def choose(self, belief: ParticleBelief) -> ReachQuery | None:
        """The query whose answers the belief is least sure of, or None.

        Distances not yet checked count as presentable until the winner's are
        checked.
        """
        dist = self.distances
        levels = np.arange(1, self._num_points + 1) / (self._num_points + 1)
        r = reach_distance(self.directions[:, None], belief.particles[None])
        p = (r[..., None] > dist).mean(1)  # P(reached), per direction and distance
        while True:
            best, score = None, self._min_entropy
            for k in range(len(self.directions)):
                idx = np.array(
                    [j for j in range(len(dist)) if self._known.get((k, j), True)], int
                )
                if not len(idx):
                    continue
                picked = idx[np.unique(np.abs(p[k, idx][:, None] - levels).argmin(0))]
                q = np.clip(p[k, picked], 1e-9, 1 - 1e-9)
                h = float(-(q * np.log(q) + (1 - q) * np.log(1 - q)).sum())
                if h > score:
                    best, score = (k, picked), h
            if best is None:
                return None
            k, picked = best
            unknown = [j for j in picked if (k, j) not in self._known]
            if not unknown:
                distances = tuple(float(dist[j]) for j in picked)
                return ReachQuery(self.directions[k].copy(), distances)
            for j in unknown:
                self.presentable(k, j)

    def gather(
        self,
        belief: Belief[NDArray[np.float64]],
        env: Environment[Any, Any, Any, ReachQuery, ReachResponse],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[ReachQuery, ReachResponse]]:
        """Ask queries until the budget runs out or no query is uncertain."""
        del rng  # unused because the choice is deterministic
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Information gathering needs a finite budget"
        data: list[QueryResponse[ReachQuery, ReachResponse]] = []
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
