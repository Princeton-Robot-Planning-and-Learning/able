"""A model learner for planners whose policies keep planning while they run."""

from typing import Any, Generic

import numpy as np

from able.beliefs.base import Belief
from able.envs.base import EnvironmentModel, TaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.learners.belief_sources import BeliefSource
from able.learners.model_learner import ModelLearner
from able.models.base import Dataset
from able.planners.base import TrainablePlanner
from able.structs import (
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    PlanningFailure,
)


class RolloutModelLearner(
    ModelLearner[HiddenParamsType, PlannerParamsType],
    Generic[HiddenParamsType, PlannerParamsType],
):
    """A model learner that runs each training plan for one episode before taking
    its rows.

    A closed-loop policy plans while it runs, so plan() alone holds only its first
    decision. Each training task's policy therefore runs in an environment
    simulated with hidden parameters sampled from the training belief, and only
    episodes that complete the task give training rows, which describe the
    training belief. The learner still never sees the real environment.
    """

    def __init__(
        self,
        planner: TrainablePlanner[Any, Any, Any, HiddenParamsType, PlannerParamsType],
        task_distribution: TaskDistribution[Any],
        belief_source: BeliefSource[HiddenParamsType, PlannerParamsType],
        planning_budget: Budget,
        env_model: EnvironmentModel[Any, Any, Any, HiddenParamsType, Any, Any],
        refit: str = "per_iteration",
        max_sleeps: int | None = None,
        num_training_tasks: int = 8,
    ) -> None:
        super().__init__(
            planner,
            task_distribution,
            belief_source,
            planning_budget,
            refit,
            max_sleeps,
            num_training_tasks,
        )
        self._env_model = env_model

    def plan_training_tasks(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Dataset:
        """Run the plans for the training tasks in simulated environments and
        return the rows of the ones that complete their task."""
        examples = []
        for task in self._get_training_tasks(rng):
            env = SimulatedEnvironment(self._env_model, belief.sample(rng), rng)
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
                trajectory = run_episode(env, task, policy)
            except PlanningFailure:
                continue
            if env.check_goal(task, trajectory.states[-1]):
                examples.append(self._planner.get_plan_examples(task, belief, policy))
        return Dataset.concatenate(examples)
