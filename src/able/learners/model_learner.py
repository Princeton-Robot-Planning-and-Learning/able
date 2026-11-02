"""A learner that trains a planner's model on plans for chosen beliefs."""

import time
from typing import Any, Generic

import numpy as np

from able.beliefs.base import Belief
from able.envs.base import FiniteTaskDistribution
from able.learners.base import Learner
from able.learners.belief_sources import BeliefSource
from able.models.base import Dataset
from able.planners.base import TrainablePlanner
from able.structs import (
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    PlanningFailure,
    Task,
    Trajectory,
)

REFIT_TIMINGS = ("per_iteration", "per_sleep")


class ModelLearner(
    Learner[HiddenParamsType, PlannerParamsType],
    Generic[HiddenParamsType, PlannerParamsType],
):
    """Trains the planner's model on plans for every task under beliefs from a
    belief source.

    Each iteration of a sleep gets a belief from the belief source, plans for every
    task in the task distribution under it, and adds the plans to the training
    data. The data accumulates across sleeps. One iteration of the budget is one
    training belief.

    The model is fit to all of the data either after every iteration, so later
    iterations plan with what earlier ones learned, or once at the end of each
    sleep. Fitting always trains on all of the data, so the timing only changes
    which model each iteration plans with.

    If learn_from_executions is set, the training data also includes the rows the
    planner makes from the episodes the robot executes. After max_sleeps sleeps, if
    set, the learner stops changing the parameters.
    """

    def __init__(
        self,
        planner: TrainablePlanner[Any, Any, Any, HiddenParamsType, PlannerParamsType],
        task_distribution: FiniteTaskDistribution[Any],
        belief_source: BeliefSource[HiddenParamsType, PlannerParamsType],
        planning_budget: Budget,
        refit: str = "per_iteration",
        learn_from_executions: bool = False,
        max_sleeps: int | None = None,
    ) -> None:
        assert refit in REFIT_TIMINGS, f"refit must be one of {REFIT_TIMINGS}"
        self._planner = planner
        self._task_distribution = task_distribution
        self._belief_source = belief_source
        self._planning_budget = planning_budget
        self._refit = refit
        self._learn_from_executions = learn_from_executions
        self._max_sleeps = max_sleeps
        self._dataset = Dataset.empty()
        self._num_sleeps = 0

    @property
    def dataset(self) -> Dataset:
        """All of the training data so far."""
        return self._dataset

    def learn(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        budget: Budget,
        rng: np.random.Generator,
    ) -> PlannerParamsType:
        """Train until the budget runs out and return parameters with the new
        model."""
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Learning needs a finite budget"
        if self._max_sleeps is not None and self._num_sleeps >= self._max_sleeps:
            return params
        self._num_sleeps += 1
        start_time = time.perf_counter()
        num_iterations = 0
        while not budget.is_exhausted(time.perf_counter() - start_time, num_iterations):
            num_iterations += 1
            training_belief = self._belief_source.get_training_belief(
                num_iterations, belief, params, rng
            )
            examples = self._plan_every_task(training_belief, params, rng)
            self._dataset = Dataset.concatenate([self._dataset, examples])
            if self._refit == "per_iteration":
                params = self._fit(params)
        if self._refit == "per_sleep":
            params = self._fit(params)
        return params

    def observe_execution(
        self,
        task: Task[Any, Any],
        trajectory: Trajectory[Any, Any],
        belief: Belief[HiddenParamsType],
    ) -> None:
        """Add the planner's rows for the execution, if learning from executions.

        The rows are used from the next fit on.
        """
        if not self._learn_from_executions:
            return
        examples = self._planner.get_execution_examples(task, belief, trajectory)
        self._dataset = Dataset.concatenate([self._dataset, examples])

    def _fit(self, params: PlannerParamsType) -> PlannerParamsType:
        if len(self._dataset) == 0:
            return params
        model = self._planner.get_model(params)
        return self._planner.set_model(params, model.fit(self._dataset))

    def _plan_every_task(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Dataset:
        """Plan for every task under the belief and return the planner's rows."""
        examples = []
        for task in self._task_distribution.get_tasks():
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
            except PlanningFailure:
                continue
            examples.append(self._planner.get_plan_examples(task, belief, policy))
        return Dataset.concatenate(examples)
