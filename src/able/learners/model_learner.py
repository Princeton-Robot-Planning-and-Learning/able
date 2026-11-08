"""A learner that trains a planner's model on plans for chosen beliefs."""

import time
from typing import Any, Generic

import numpy as np

from able.beliefs.base import Belief
from able.envs.base import FiniteTaskDistribution, TaskDistribution
from able.learners.base import Learner
from able.learners.belief_sources import BeliefSource
from able.models.base import Dataset
from able.parallel import WorkerContext, WorkerPool
from able.planners.base import TrainablePlanner
from able.structs import (
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    PlanningFailure,
)

REFIT_TIMINGS = ("per_iteration", "per_sleep")


class ModelLearner(
    Learner[HiddenParamsType, PlannerParamsType],
    Generic[HiddenParamsType, PlannerParamsType],
):
    """Trains the planner's model on plans for training tasks under beliefs from a
    belief source.

    Each iteration of a sleep gets a belief from the belief source, plans for the
    training tasks under it, and adds the plans to the training data. The data
    accumulates across sleeps. One iteration of the budget is one training belief.
    The training tasks of a finite task distribution are all of its tasks. Other
    distributions draw num_training_tasks new tasks in each iteration.

    The model is fit to all of the data either after every iteration, so later
    iterations plan with what earlier ones learned, or once at the end of each
    sleep. Fitting always trains on all of the data, so the timing only changes
    which model each iteration plans with.

    The learner never sees the real environment: all of its data comes from
    planning. After max_sleeps sleeps, if set, the learner stops changing the
    parameters.

    Given a pool, a sleep that refits once at its end and has a fixed number of
    iterations runs in parallel: the pool's workers make every draw of the belief
    source at once, this process chooses each iteration's training belief in
    order, and the workers then plan every iteration's training tasks at once.
    Each draw and each iteration's training tasks get their own random generator,
    spawned from the sleep's, so the result does not depend on the number of
    workers.
    """

    def __init__(
        self,
        planner: TrainablePlanner[Any, Any, Any, HiddenParamsType, PlannerParamsType],
        task_distribution: TaskDistribution[Any],
        belief_source: BeliefSource[HiddenParamsType, PlannerParamsType],
        planning_budget: Budget,
        refit: str = "per_iteration",
        max_sleeps: int | None = None,
        num_training_tasks: int = 8,
    ) -> None:
        assert refit in REFIT_TIMINGS, f"refit must be one of {REFIT_TIMINGS}"
        assert num_training_tasks > 0, "Need at least one training task"
        self._num_training_tasks = num_training_tasks
        self._planner = planner
        self._task_distribution = task_distribution
        self._belief_source = belief_source
        self._planning_budget = planning_budget
        self._refit = refit
        self._max_sleeps = max_sleeps
        self._dataset = Dataset.empty()
        self._num_sleeps = 0

    @property
    def dataset(self) -> Dataset:
        """All of the training data so far."""
        return self._dataset

    @property
    def belief_source(self) -> BeliefSource[HiddenParamsType, PlannerParamsType]:
        """The source of the beliefs the learner trains under."""
        return self._belief_source

    def learn(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        budget: Budget,
        rng: np.random.Generator,
        pool: WorkerPool | None = None,
    ) -> PlannerParamsType:
        """Train until the budget runs out and return parameters with the new
        model."""
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Learning needs a finite budget"
        if self._max_sleeps is not None and self._num_sleeps >= self._max_sleeps:
            return params
        self._num_sleeps += 1
        if (
            pool is not None
            and self._refit == "per_sleep"
            and budget.max_iterations is not None
            and budget.max_wall_time is None
        ):
            return self._learn_in_parallel(
                belief, params, budget.max_iterations, rng, pool
            )
        start_time = time.perf_counter()
        num_iterations = 0
        while not budget.is_exhausted(time.perf_counter() - start_time, num_iterations):
            num_iterations += 1
            training_belief = self._belief_source.get_training_belief(
                num_iterations, belief, params, rng
            )
            examples = self.plan_training_tasks(training_belief, params, rng)
            self._dataset = Dataset.concatenate([self._dataset, examples])
            if self._refit == "per_iteration":
                params = self._fit(params)
        if self._refit == "per_sleep":
            params = self._fit(params)
        return params

    def _learn_in_parallel(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        num_iterations: int,
        rng: np.random.Generator,
        pool: WorkerPool,
    ) -> PlannerParamsType:
        """One sleep of num_iterations iterations, with the draws and the training
        plans made in the pool's workers."""
        num_draws = self._belief_source.num_draws
        draw_rngs = rng.spawn(num_iterations * num_draws)
        draws = pool.map(
            _draw_in_worker,
            [(i + 1, belief, params, r) for i, r in enumerate(draw_rngs)],
        )
        training_beliefs = [
            self._belief_source.choose(
                iteration + 1,
                draws[iteration * num_draws : (iteration + 1) * num_draws],
            )
            for iteration in range(num_iterations)
        ]
        plan_rngs = rng.spawn(num_iterations)
        examples = pool.map(
            _plan_training_tasks_in_worker,
            [(b, params, r) for b, r in zip(training_beliefs, plan_rngs)],
        )
        self._dataset = Dataset.concatenate([self._dataset, *examples])
        return self._fit(params)

    def _fit(self, params: PlannerParamsType) -> PlannerParamsType:
        if len(self._dataset) == 0:
            return params
        model = self._planner.get_model(params)
        return self._planner.set_model(params, model.fit(self._dataset))

    def _get_training_tasks(self, rng: np.random.Generator) -> list[Any]:
        if isinstance(self._task_distribution, FiniteTaskDistribution):
            return list(self._task_distribution.get_tasks())
        return [
            self._task_distribution.sample(rng) for _ in range(self._num_training_tasks)
        ]

    def plan_training_tasks(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Dataset:
        """Plan for the training tasks under the belief and return the planner's
        rows."""
        examples = []
        for task in self._get_training_tasks(rng):
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
            except PlanningFailure:
                continue
            examples.append(self._planner.get_plan_examples(task, belief, policy))
        return Dataset.concatenate(examples)


def _draw_in_worker(
    context: WorkerContext,
    index: int,
    belief: Belief[Any],
    params: Any,
    rng: np.random.Generator,
) -> Belief[Any]:
    """A worker's draw from its own copy of the learner's belief source."""
    learner = context.learner
    assert isinstance(learner, ModelLearner), "The worker's learner trains a model"
    return learner.belief_source.draw(index, belief, params, rng)


def _plan_training_tasks_in_worker(
    context: WorkerContext, belief: Belief[Any], params: Any, rng: np.random.Generator
) -> Dataset:
    """A worker's training plans under one training belief, with its own copy of
    the learner."""
    learner = context.learner
    assert isinstance(learner, ModelLearner), "The worker's learner trains a model"
    return learner.plan_training_tasks(belief, params, rng)
