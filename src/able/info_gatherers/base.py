"""Base class for information gatherers."""

from abc import ABC, abstractmethod
from typing import Any, Generic

import numpy as np

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
