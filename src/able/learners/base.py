"""Base class for learners, and a learner that does nothing."""

from abc import ABC, abstractmethod
from typing import Any, Generic

import numpy as np

from able.beliefs.base import Belief
from able.structs import Budget, HiddenParamsType, PlannerParamsType, Task, Trajectory


class Learner(ABC, Generic[HiddenParamsType, PlannerParamsType]):
    """Improves the planner parameters while the robot sleeps.

    Learners have no access to the real environment. They may keep state across
    cycles, such as data accumulated in earlier sleeps or from the tasks the robot
    executed.
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

    def observe_execution(
        self,
        task: Task[Any, Any],
        trajectory: Trajectory[Any, Any],
        belief: Belief[HiddenParamsType],
    ) -> None:
        """Observe a task the robot executed, with its belief after the execution.

        Learners can keep what they observe and use it in later sleeps. Success is
        not observed, because the robot cannot observe it directly. The default
        ignores the execution.
        """
        del task, trajectory, belief  # unused by default


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
