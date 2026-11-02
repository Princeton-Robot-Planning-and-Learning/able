"""The robot: its components, belief, and planner parameters."""

from typing import Generic, Sequence

import numpy as np

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.envs.base import Environment
from able.info_gatherers.base import InformationGatherer
from able.learners.base import Learner
from able.planners.base import Planner
from able.policies.base import Policy
from able.structs import (
    ActType,
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    StateType,
    TaskType,
    Trajectory,
)


class Robot(Generic[StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType]):
    """Combines a planner, learner, belief updater, and information gatherer.

    The robot's state is its current belief and planner parameters. Only sleep()
    changes the parameters and only update_belief() changes the belief. An
    approach is a choice of the four components.
    """

    def __init__(
        self,
        planner: Planner[
            StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType
        ],
        learner: Learner[HiddenParamsType, PlannerParamsType],
        belief_updater: BeliefUpdater[StateType, ActType, HiddenParamsType],
        info_gatherer: InformationGatherer[
            StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType
        ],
        initial_belief: Belief[HiddenParamsType],
    ) -> None:
        self._planner = planner
        self._learner = learner
        self._belief_updater = belief_updater
        self._info_gatherer = info_gatherer
        self._belief = initial_belief
        self._planner_params = planner.get_default_params()

    @property
    def belief(self) -> Belief[HiddenParamsType]:
        """The robot's current belief about the hidden parameters."""
        return self._belief

    @property
    def planner_params(self) -> PlannerParamsType:
        """The robot's current planner parameters."""
        return self._planner_params

    def sleep(self, budget: Budget, rng: np.random.Generator) -> None:
        """Run the learner to update the planner parameters."""
        self._planner_params = self._learner.learn(
            self._belief, self._planner_params, budget, rng
        )

    def plan(
        self, task: TaskType, budget: Budget, rng: np.random.Generator
    ) -> Policy[StateType, ActType]:
        """Plan for the task under the current belief and planner parameters."""
        return self._planner.plan(task, self._belief, self._planner_params, budget, rng)

    def gather_information(
        self,
        env: Environment[StateType, ActType, TaskType],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[Trajectory[StateType, ActType]]:
        """Run the information gatherer in the environment."""
        return self._info_gatherer.gather(
            self._belief, self._planner_params, env, budget, rng
        )

    def update_belief(self, data: Sequence[Trajectory[StateType, ActType]]) -> None:
        """Update the belief with gathered data."""
        self._belief = self._belief_updater.update(self._belief, data)
