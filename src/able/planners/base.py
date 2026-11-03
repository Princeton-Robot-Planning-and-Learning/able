"""Base classes for planners, including planners that learn a model."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from able.beliefs.base import Belief
from able.models.base import ConditionalModel, Dataset
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


class TrainablePlanner(
    Planner[StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType]
):
    """A planner whose parameters hold a conditional model that guides planning.

    The planner decides what the model's inputs and outputs are. It turns plans into
    training rows, and learners fit the model to those rows and put it back into
    the parameters.
    """

    @abstractmethod
    def get_model(self, params: PlannerParamsType) -> ConditionalModel:
        """The model in the parameters."""

    @abstractmethod
    def set_model(
        self, params: PlannerParamsType, model: ConditionalModel
    ) -> PlannerParamsType:
        """New parameters with the model replaced."""

    @abstractmethod
    def get_plan_examples(
        self,
        task: TaskType,
        belief: Belief[HiddenParamsType],
        policy: Policy[StateType, ActType],
    ) -> Dataset:
        """Training rows from a policy that plan() returned for the task under the
        belief."""
