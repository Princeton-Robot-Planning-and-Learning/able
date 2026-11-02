"""Base class for learners, and a learner that does nothing."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from able.beliefs.base import Belief
from able.structs import Budget, HiddenParamsType, PlannerParamsType


class Learner(ABC, Generic[HiddenParamsType, PlannerParamsType]):
    """Improves the planner parameters while the robot sleeps.

    Learners have no access to the real environment. They may keep state across
    cycles, such as data accumulated in earlier sleeps.
    """

    @abstractmethod
    def learn(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        budget: Budget,
        rng: np.random.Generator,
    ) -> PlannerParamsType:
        """Return new planner parameters given the current belief and parameters."""


class NullLearner(Learner[HiddenParamsType, PlannerParamsType]):
    """Never changes the planner parameters."""

    def learn(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        budget: Budget,
        rng: np.random.Generator,
    ) -> PlannerParamsType:
        """Return the parameters unchanged."""
        return params
