"""Gathering information with queries drawn at random from the current belief's
range, for every environment."""

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.beliefs.particles import ParticleBelief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.base import Environment
from able.envs.friction.query import _SIGMAS, FrictionQuery, probe_queries
from able.envs.friction.world import PuckWorld
from able.envs.human_reach.query import (
    QuerySimulator,
    ReachQuery,
    bedside_directions,
    reach_distance,
)
from able.envs.human_reach.scene import BedsideStocking3DEnvConfig
from able.envs.limb.query import LimbQuery
from able.envs.two_link_arm import JointQuery
from able.info_gatherers.base import InformationGatherer
from able.structs import Budget, HiddenParamsType, QueryResponse, QueryType


def _fractions(num: int) -> NDArray[np.float64]:
    return np.arange(1, num + 1) / (num + 1)


class QuerySpace(ABC, Generic[QueryType]):
    """An environment's queries within the current belief's range."""

    @abstractmethod
    def get_queries(self, belief: Belief[Any]) -> list[QueryType]:
        """A fixed set of queries at set points of the belief's range."""

    @abstractmethod
    def sample(self, belief: Belief[Any], rng: np.random.Generator) -> QueryType | None:
        """A query drawn uniformly from the belief's range, or None if none is left."""

    def sample_finite(
        self, belief: Belief[Any], rng: np.random.Generator
    ) -> QueryType | None:
        """A query drawn uniformly from get_queries(), or None if it is empty."""
        queries = self.get_queries(belief)
        return queries[int(rng.integers(len(queries)))] if queries else None


class QueryDistribution(ABC, Generic[QueryType]):
    """Draws a query under a belief."""

    def __init__(self, space: QuerySpace[QueryType]) -> None:
        self.space = space

    @abstractmethod
    def sample(self, belief: Belief[Any], rng: np.random.Generator) -> QueryType | None:
        """A random query, or None if none is left."""


class FiniteQueryDistribution(QueryDistribution[QueryType]):
    """Uniform over the space's fixed set of queries."""

    def sample(self, belief: Belief[Any], rng: np.random.Generator) -> QueryType | None:
        return self.space.sample_finite(belief, rng)


class InfiniteQueryDistribution(QueryDistribution[QueryType]):
    """Uniform over the space's continuous range."""

    def sample(self, belief: Belief[Any], rng: np.random.Generator) -> QueryType | None:
        return self.space.sample(belief, rng)


class RandomGatherer(InformationGatherer[HiddenParamsType, QueryType, Any]):
    """Asks random queries until the budget runs out, each drawn under the belief
    updated with the earlier responses. One iteration of the budget is one query."""

    def __init__(
        self,
        distribution: QueryDistribution[QueryType],
        belief_updater: BeliefUpdater[Any, Any, HiddenParamsType, QueryType, Any],
    ) -> None:
        self._distribution = distribution
        self._belief_updater = belief_updater

    def gather(
        self,
        belief: Belief[HiddenParamsType],
        env: Environment[Any, Any, Any, QueryType, Any],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[QueryType, Any]]:
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Information gathering needs a finite budget"
        data: list[QueryResponse[QueryType, Any]] = []
        start_time = time.perf_counter()
        while not budget.is_exhausted(time.perf_counter() - start_time, len(data)):
            query = self._distribution.sample(belief, rng)
            if query is None:
                break
            datum = QueryResponse(query, env.ask(query))
            data.append(datum)
            belief = self._belief_updater.update_from_queries(belief, [datum])
        return data


class ArmQueries(QuerySpace[JointQuery]):
    """Angles within each joint limit's interval [lo1, hi1, lo2, hi2]."""

    def __init__(self, num_points: int = 3, min_width: float = 1e-3) -> None:
        self._num_points = num_points
        self._min_width = min_width

    def _open_limits(self, belief: Belief[Any]) -> tuple[NDArray[np.float64], Any]:
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        intervals = belief.intervals
        widths = intervals[:, 1] - intervals[:, 0]
        return intervals, np.flatnonzero(widths > self._min_width)

    def get_queries(self, belief: Belief[Any]) -> list[JointQuery]:
        intervals, limits = self._open_limits(belief)
        queries = []
        for i in limits:
            lo, hi = intervals[i]
            for f in _fractions(self._num_points):
                queries.append(JointQuery(int(i) // 2, float(lo + f * (hi - lo))))
        return queries

    def sample(
        self, belief: Belief[Any], rng: np.random.Generator
    ) -> JointQuery | None:
        intervals, limits = self._open_limits(belief)
        if not len(limits):
            return None
        i = int(rng.choice(limits))
        return JointQuery(i // 2, float(rng.uniform(*intervals[i])))


class LimbQueries(QuerySpace[LimbQuery]):
    """Targets between each motion's magnitude times the belief's lower and upper
    scale."""

    def __init__(self, magnitudes: NDArray[np.float64], num_points: int = 5) -> None:
        self._magnitudes = np.asarray(magnitudes, dtype=np.float64)
        self._num_points = num_points

    def _ranges(self, belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        return np.stack([belief.lower, belief.high], -1) * self._magnitudes[:, None]

    def get_queries(self, belief: Belief[Any]) -> list[LimbQuery]:
        return [
            LimbQuery(motion, float(lo + f * (hi - lo)))
            for motion, (lo, hi) in enumerate(self._ranges(belief))
            for f in _fractions(self._num_points)
        ]

    def sample(self, belief: Belief[Any], rng: np.random.Generator) -> LimbQuery:
        ranges = self._ranges(belief)
        motion = int(rng.integers(len(ranges)))
        return LimbQuery(motion, float(rng.uniform(*ranges[motion])))


class ReachQueries(QuerySpace[ReachQuery]):
    """Points along the directions from the shoulder to the bedside slots, within
    the shortest and longest reach the belief's particles allow along each, where
    the robot can present its gripper.

    Presentable points come from a grid of distances, checked by IK once per
    direction when first needed and cached.
    """

    def __init__(
        self,
        config: BedsideStocking3DEnvConfig = BedsideStocking3DEnvConfig(),
        num_points: int = 4,
        grid_step: float = 0.025,
    ) -> None:
        self._config = config
        self._num_points = num_points
        self._grid_step = grid_step
        self._presenter = QuerySimulator(config)
        self._directions: NDArray[np.float64] | None = None
        self._presentable: dict[int, NDArray[np.bool_]] = {}

    @property
    def directions(self) -> NDArray[np.float64]:
        """Unit vectors from the shoulder to every bedside slot."""
        if self._directions is None:
            shoulder = self._presenter.sim.shoulder
            self._directions = bedside_directions(self._config, shoulder)
        return self._directions

    @property
    def distances(self) -> NDArray[np.float64]:
        """Grid distances spanning the prior's reach along every direction."""
        lb = np.array(self._config.reach_theta_lb)
        ub = np.array(self._config.reach_theta_ub)
        r = reach_distance(self.directions, np.stack([lb, ub])[:, None])
        return np.arange(r[0].min(), r[1].max() + 1e-9, self._grid_step)

    def close(self) -> None:
        """Close the IK simulator."""
        self._presenter.close()

    def _cells(self, belief: Belief[Any], k: int) -> NDArray[np.float64]:
        """Presentable grid distances along direction k within the belief's span."""
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        dist = self.distances
        if k not in self._presentable:
            shoulder = self._presenter.sim.shoulder
            self._presentable[k] = np.array(
                [
                    self._presenter.can_present(shoulder + d * self.directions[k])
                    for d in dist
                ]
            )
        r = reach_distance(self.directions[k], belief.particles)
        inside = (dist >= r.min()) & (dist <= r.max())
        return dist[inside & self._presentable[k]]

    def _query(self, k: int, distances: NDArray[np.float64]) -> ReachQuery:
        return ReachQuery(self.directions[k].copy(), tuple(float(d) for d in distances))

    def _fixed_query(self, belief: Belief[Any], k: int) -> ReachQuery | None:
        cells = self._cells(belief, k)
        if not len(cells):
            return None
        picks = np.unique((_fractions(self._num_points) * len(cells)).astype(int))
        return self._query(k, cells[picks])

    def get_queries(self, belief: Belief[Any]) -> list[ReachQuery]:
        queries = [self._fixed_query(belief, k) for k in range(len(self.directions))]
        return [q for q in queries if q is not None]

    def sample_finite(
        self, belief: Belief[Any], rng: np.random.Generator
    ) -> ReachQuery | None:
        # One fixed query per direction, so a uniform direction is a uniform query.
        for k in rng.permutation(len(self.directions)):
            query = self._fixed_query(belief, int(k))
            if query is not None:
                return query
        return None

    def sample(
        self, belief: Belief[Any], rng: np.random.Generator
    ) -> ReachQuery | None:
        for k in rng.permutation(len(self.directions)):
            cells = self._cells(belief, int(k))
            if len(cells):
                picks = rng.choice(cells, self._num_points)
                half = self._grid_step / 2
                distances = np.sort(picks + rng.uniform(-half, half, self._num_points))
                return self._query(int(k), distances)
        return None


class FrictionQueries(QuerySpace[FrictionQuery]):
    """Each floor's probe pose, at speeds up to the probe's, which stays on its
    clear run at the belief's lowest friction."""

    def __init__(self, world: PuckWorld, num_speeds: int = 3) -> None:
        self._world = world
        self._num_speeds = num_speeds

    def _probes(self, belief: Belief[Any]) -> Sequence[FrictionQuery]:
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        return probe_queries(self._world, belief.low)

    def get_queries(self, belief: Belief[Any]) -> list[FrictionQuery]:
        min_speed = self._world.physics.min_speed
        return [
            FrictionQuery(p.floor, p.puck, p.angle, float(s))
            for p in self._probes(belief)
            for s in np.linspace(min_speed, p.speed, self._num_speeds + 1)[1:]
        ]

    def sample(self, belief: Belief[Any], rng: np.random.Generator) -> FrictionQuery:
        probes = self._probes(belief)
        probe = probes[int(rng.integers(len(probes)))]
        speed = float(rng.uniform(self._world.physics.min_speed, probe.speed))
        # The probe's run is clear within this heading cone.
        spread = _SIGMAS * self._world.physics.angle_noise
        angle = probe.angle + float(rng.uniform(-spread, spread))
        if not self._world.push_feasible(np.array(probe.puck), angle):
            angle = probe.angle
        return FrictionQuery(probe.floor, probe.puck, angle, speed)
