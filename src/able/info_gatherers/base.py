"""Base class for information gatherers."""

import time
from abc import ABC, abstractmethod
from typing import Any, Generic

import numpy as np

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.envs.base import Environment
from able.structs import (
    Budget,
    HiddenParamsType,
    QueryResponse,
    QueryType,
    ResponseType,
)


class InformationGatherer(ABC, Generic[HiddenParamsType, QueryType, ResponseType]):
    """Asks an environment queries to learn about its hidden parameters.

    Gatherers only ask queries and never run episodes. They keep no state between
    calls and work with any environment, so learners can run them in simulated
    environments built from sampled hidden parameters to imagine the responses the
    robot would receive. The robot updates its belief once, with all of the
    responses, after gathering ends.
    """

    @abstractmethod
    def gather(
        self,
        belief: Belief[HiddenParamsType],
        env: Environment[Any, Any, Any, QueryType, ResponseType],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[QueryType, ResponseType]]:
        """Ask queries within the budget and return each one with its response."""


class NullInformationGatherer(
    InformationGatherer[HiddenParamsType, QueryType, ResponseType]
):
    """Never asks a query."""

    def gather(
        self,
        belief: Belief[HiddenParamsType],
        env: Environment[Any, Any, Any, QueryType, ResponseType],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[QueryType, ResponseType]]:
        """Return no data."""
        return []


class SequentialGatherer(
    InformationGatherer[HiddenParamsType, QueryType, ResponseType]
):
    """Asks one query at a time, each chosen under the belief updated with the
    earlier responses, until the budget runs out or no query is worth asking. One
    iteration of the budget is one query."""

    def __init__(
        self, belief_updater: BeliefUpdater[Any, Any, HiddenParamsType, Any, Any]
    ) -> None:
        self._belief_updater = belief_updater

    @property
    def belief_updater(self) -> BeliefUpdater[Any, Any, HiddenParamsType, Any, Any]:
        """The updater that folds each response into the belief."""
        return self._belief_updater

    @abstractmethod
    def choose(
        self, belief: Belief[HiddenParamsType], rng: np.random.Generator
    ) -> QueryType | None:
        """The next query under the belief, or None to stop."""

    def get_candidates(
        self, belief: Belief[HiddenParamsType], rng: np.random.Generator
    ) -> list[QueryType]:
        """The queries the robot could ask next. By default only choose()'s."""
        query = self.choose(belief, rng)
        return [] if query is None else [query]

    def gather(
        self,
        belief: Belief[HiddenParamsType],
        env: Environment[Any, Any, Any, QueryType, ResponseType],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[QueryType, ResponseType]]:
        """Ask queries until the budget runs out or choose() returns None."""
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Information gathering needs a finite budget"
        data: list[QueryResponse[QueryType, ResponseType]] = []
        start_time = time.perf_counter()
        while not budget.is_exhausted(time.perf_counter() - start_time, len(data)):
            query = self.choose(belief, rng)
            if query is None:
                break
            datum = QueryResponse(query, env.ask(query))
            data.append(datum)
            belief = self._belief_updater.update_from_queries(belief, [datum])
        return data
