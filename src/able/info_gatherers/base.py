"""Base class for information gatherers."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from able.beliefs.base import Belief
from able.envs.base import Environment
from able.structs import (
    ActType,
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    StateType,
    TaskType,
    Trajectory,
)


class InformationGatherer(
    ABC, Generic[StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType]
):
    """Interacts with an environment to collect data about the hidden parameters.

    Gatherers keep no state between calls and work with any environment, so
    learners can run them in simulated environments built from sampled hidden
    parameters to imagine what the robot would observe.
    """

    @abstractmethod
    def gather(
        self,
        belief: Belief[HiddenParamsType],
        planner_params: PlannerParamsType,
        env: Environment[StateType, ActType, TaskType],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[Trajectory[StateType, ActType]]:
        """Interact with the environment and return the trajectories observed.

        The planner parameters are the robot's current ones, for gatherers that
        plan with the robot's planner.
        """
