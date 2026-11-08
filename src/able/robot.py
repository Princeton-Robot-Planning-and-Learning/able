"""The robot: its components, belief, and planner parameters."""

from typing import Generic, Sequence

import numpy as np

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.envs.base import Environment
from able.info_gatherers.base import InformationGatherer
from able.learners.base import Learner
from able.parallel import WorkerPool
from able.planners.base import Planner
from able.policies.base import Policy
from able.structs import (
    ActType,
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    QueryResponse,
    QueryType,
    ResponseType,
    StateType,
    TaskType,
)


class Robot(
    Generic[
        StateType,
        ActType,
        TaskType,
        HiddenParamsType,
        PlannerParamsType,
        QueryType,
        ResponseType,
    ]
):
    """Combines a planner, learner, belief updater, and information gatherer.

    The robot's state is its current belief and planner parameters. Only sleep()
    changes the parameters, and only the two update_belief methods change the
    belief. An approach is a choice of the four components.
    """

    def __init__(
        self,
        planner: Planner[
            StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType
        ],
        learner: Learner[HiddenParamsType, PlannerParamsType],
        belief_updater: BeliefUpdater[
            StateType, ActType, HiddenParamsType, QueryType, ResponseType
        ],
        info_gatherer: InformationGatherer[HiddenParamsType, QueryType, ResponseType],
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

    @property
    def planner(
        self,
    ) -> Planner[StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType]:
        """The robot's planner, for worker processes that plan as the robot does."""
        return self._planner

    @property
    def learner(self) -> Learner[HiddenParamsType, PlannerParamsType]:
        """The robot's learner, for worker processes that train as the robot does."""
        return self._learner

    def sleep(
        self, budget: Budget, rng: np.random.Generator, pool: WorkerPool | None = None
    ) -> None:
        """Run the learner to update the planner parameters, with the pool's workers
        if given."""
        self._planner_params = self._learner.learn(
            self._belief, self._planner_params, budget, rng, pool
        )

    def plan(
        self, task: TaskType, budget: Budget, rng: np.random.Generator
    ) -> Policy[StateType, ActType]:
        """Plan for the task under the current belief and planner parameters."""
        return self._planner.plan(task, self._belief, self._planner_params, budget, rng)

    def gather_information(
        self,
        env: Environment[StateType, ActType, TaskType, QueryType, ResponseType],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[QueryType, ResponseType]]:
        """Ask the environment queries with the information gatherer."""
        return self._info_gatherer.gather(self._belief, env, budget, rng)

    def update_belief_from_queries(
        self, data: Sequence[QueryResponse[QueryType, ResponseType]]
    ) -> None:
        """Update the belief with the responses to queries."""
        self._belief = self._belief_updater.update_from_queries(self._belief, data)

    def update_belief_from_transition(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> None:
        """Update the belief with one step of task execution."""
        self._belief = self._belief_updater.update_from_transition(
            self._belief, state, action, next_state
        )
