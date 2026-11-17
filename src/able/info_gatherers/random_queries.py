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
        queries = self.space.get_queries(belief)
        return queries[int(rng.integers(len(queries)))] if queries else None


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
    """Points along directions from the shoulder, between the shortest and longest
    reach the belief's particles allow along each.

    The fixed set uses the directions to the bedside slots; samples mix two of
    them at random. Points the robot cannot present are skipped when asked.
    """

    def __init__(
        self,
        config: BedsideStocking3DEnvConfig = BedsideStocking3DEnvConfig(),
        num_points: int = 4,
    ) -> None:
        self._config = config
        self._num_points = num_points
        self._directions: NDArray[np.float64] | None = None

    @property
    def directions(self) -> NDArray[np.float64]:
        """Unit vectors from the shoulder to every bedside slot."""
        if self._directions is None:
            presenter = QuerySimulator(self._config)
            self._directions = bedside_directions(self._config, presenter.sim.shoulder)
            presenter.close()
        return self._directions

    def _span(self, belief: Belief[Any], direction: NDArray) -> tuple[float, float]:
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        r = reach_distance(direction, belief.particles)
        return float(r.min()), float(r.max())

    def get_queries(self, belief: Belief[Any]) -> list[ReachQuery]:
        queries = []
        for direction in self.directions:
            lo, hi = self._span(belief, direction)
            distances = tuple(
                float(lo + f * (hi - lo)) for f in _fractions(self._num_points)
            )
            queries.append(ReachQuery(direction.copy(), distances))
        return queries

    def sample(self, belief: Belief[Any], rng: np.random.Generator) -> ReachQuery:
        a, b = self.directions[rng.choice(len(self.directions), 2)]
        direction = a + rng.uniform() * (b - a)
        direction /= np.linalg.norm(direction)
        lo, hi = self._span(belief, direction)
        distances = np.sort(rng.uniform(lo, hi, self._num_points))
        return ReachQuery(direction, tuple(float(d) for d in distances))


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
