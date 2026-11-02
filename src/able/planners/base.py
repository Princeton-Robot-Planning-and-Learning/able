"""Base class for planners."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from able.beliefs.base import Belief
from able.policies.base import Policy
from able.structs import (
    ActType,
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    StateType,
    TaskType,
)


class Planner(
    ABC, Generic[StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType]
):
    """Produces a policy for a task under a belief about the hidden parameters.

    The planner parameters control how the planner searches, and learners improve
    planning by changing them. Planners keep no state between calls to plan(), so
    the parameters are the only way that learning affects planning.
    """

    @abstractmethod
    def get_default_params(self) -> PlannerParamsType:
        """The parameters to use before any learning."""

    @abstractmethod
    def plan(
        self,
        task: TaskType,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        budget: Budget,
        rng: np.random.Generator,
    ) -> Policy[StateType, ActType]:
        """Find a policy for the task.

        The budget covers all planning for the task, including any replanning the
        policy does while it runs. Raises PlanningFailure if no policy is found
        within the budget.
        """
