"""Base class for belief updaters."""

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from able.beliefs.base import Belief
from able.structs import (
    ActType,
    HiddenParamsType,
    QueryResponse,
    QueryType,
    ResponseType,
    StateType,
)


class BeliefUpdater(
    ABC, Generic[StateType, ActType, HiddenParamsType, QueryType, ResponseType]
):
    """Computes a new belief from what the robot observes.

    The robot observes two kinds of data. While it executes a task, it sees the
    state that results from each action. When it asks queries, it sees the
    responses. Updaters do not modify their inputs or keep state, so learners can
    apply them to imagined data without affecting the robot's actual belief.
    """

    @abstractmethod
    def update_from_transition(
        self,
        belief: Belief[HiddenParamsType],
        state: StateType,
        action: ActType,
        next_state: StateType,
    ) -> Belief[HiddenParamsType]:
        """Return the belief after observing one step of task execution."""

    @abstractmethod
    def update_from_queries(
        self,
        belief: Belief[HiddenParamsType],
        data: Sequence[QueryResponse[QueryType, ResponseType]],
    ) -> Belief[HiddenParamsType]:
        """Return the belief after observing the responses to the queries."""
