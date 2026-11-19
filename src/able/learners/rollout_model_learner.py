"""A model learner for planners whose policies keep planning while they run."""

from typing import Any, Generic, Sequence

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
    simulated with hidden parameters sampled from the training belief, and the
    planner turns the episode into training rows, which describe the training
    belief. By default only episodes that complete the task give rows. The learner
    still never sees the real environment.
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
        keep_only_last_sleep: bool = False,
    ) -> None:
        super().__init__(
            planner,
            task_distribution,
            belief_source,
            planning_budget,
            refit,
            max_sleeps,
            num_training_tasks,
            keep_only_last_sleep,
        )
        self._env_model = env_model

    def draw_training_tasks(
        self, belief: Belief[HiddenParamsType], rng: np.random.Generator
    ) -> list[Any]:
        """The training tasks under the belief, each paired with hidden
        parameters sampled from it, which its rollout runs with. Tasks that
        depend on the hidden parameters are drawn given them."""
        if not self._task_distribution.depends_on_hidden_params:
            tasks = super().draw_training_tasks(belief, rng)
            return [(task, belief.sample(rng)) for task in tasks]
        pairs = []
        for _ in range(self._num_training_tasks):
            hidden_params = belief.sample(rng)
            task = self._task_distribution.given(hidden_params).sample(rng)
            pairs.append((task, hidden_params))
        return pairs

    def plan_tasks(
        self,
        tasks: Sequence[Any],
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Dataset:
        """Run the plans for the tasks, each a pair of a task and the hidden
        parameters its rollout runs with (see draw_training_tasks()), made with
        the planner's training parameters under the belief, in simulated
        environments, and return the planner's rows for each episode."""
        params = self._planner.get_training_params(params)
        examples = []
        for task, hidden_params in tasks:
            env = SimulatedEnvironment(self._env_model, hidden_params, rng)
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
                trajectory = run_episode(env, task, policy)
            except PlanningFailure:
                continue
            solved = env.check_goal(task, trajectory.states[-1])
            examples.append(
                self._planner.get_rollout_examples(
                    task, belief, policy, trajectory, solved
                )
            )
        return Dataset.concatenate(examples)
