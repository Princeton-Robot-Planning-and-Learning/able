"""Base class for policies, and a policy that executes a fixed plan."""

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from able.structs import ActType, PlanningFailure, StateType


class Policy(ABC, Generic[StateType, ActType]):
    """A closed-loop policy for one task, built by a planner for a fixed belief.

    A policy is used for one episode and then discarded.
    """

    @abstractmethod
    def get_action(self, state: StateType) -> ActType:
        """Choose an action for the state.

        Raises PlanningFailure if no action can be found, for example when online
        replanning runs out of budget.
        """

    def get_planning_stats(self) -> dict[str, float]:
        """Planner-specific statistics about the planning done for this policy.

        The statistics cover the initial planning and any replanning done in
        get_action() so far. The default has no statistics.
        """
        return {}


class OpenLoopPolicy(Policy[StateType, ActType]):
    """Executes a fixed sequence of actions without looking at the states."""

    def __init__(
        self,
        actions: Sequence[ActType],
        planning_stats: dict[str, float] | None = None,
    ) -> None:
        self._actions = tuple(actions)
        self._planning_stats = dict(planning_stats) if planning_stats else {}
        self._num_steps = 0

    def get_action(self, state: StateType) -> ActType:
        """Return the next action in the sequence."""
        del state  # unused because the policy is open loop
        if self._num_steps >= len(self._actions):
            raise PlanningFailure("The open-loop plan has no actions left")
        action = self._actions[self._num_steps]
        self._num_steps += 1
        return action

    def get_planning_stats(self) -> dict[str, float]:
        return dict(self._planning_stats)
