"""Base class for belief updaters."""

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from able.beliefs.base import Belief
from able.structs import HiddenParamsType, QueryResponse, QueryType, ResponseType


class BeliefUpdater(ABC, Generic[HiddenParamsType, QueryType, ResponseType]):
    """Computes a new belief from the responses to queries.

    Updaters do not modify their inputs or keep state, so learners can apply them
    to imagined responses without affecting the robot's actual belief.
    """

    @abstractmethod
    def update(
        self,
        belief: Belief[HiddenParamsType],
        data: Sequence[QueryResponse[QueryType, ResponseType]],
    ) -> Belief[HiddenParamsType]:
        """Return the belief after observing the responses to the queries."""
