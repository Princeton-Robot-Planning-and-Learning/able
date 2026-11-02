"""Learning warm starts for random shooting by training its model on plans for
chosen beliefs."""

import dataclasses
import time
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import ArmTask
from able.learners.base import Learner
from able.models.base import Dataset
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.structs import Budget, PlanningFailure

ArmArray = NDArray[np.float64]


class PlanCachingLearner(Learner[ArmArray, RandomShootingParams]):
    """Trains the planner's model on plans for every task under a belief chosen in
    each iteration.

    Each iteration chooses a belief to train under, plans for every task in the
    task distribution under that belief, and adds the plans to its training data.
    It then fits the planner's model to all of the data, including data from
    earlier sleeps. Subclasses decide which belief each iteration trains under.
    Each iteration plans with the model fit by earlier iterations, and one
    iteration of the budget is one chosen belief.
    """

    def __init__(
        self,
        task_distribution: FiniteTaskDistribution[ArmTask],
        planner: RandomShootingPlanner,
        planning_budget: Budget,
    ) -> None:
        self._task_distribution = task_distribution
        self._planner = planner
        self._planning_budget = planning_budget
        self._dataset = Dataset.empty()

    def learn(
        self,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> RandomShootingParams:
        """Train until the budget runs out and return parameters with the new
        model."""
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Learning needs a finite budget"
        start_time = time.perf_counter()
        num_iterations = 0
        while not budget.is_exhausted(time.perf_counter() - start_time, num_iterations):
            num_iterations += 1
            training_belief = self._get_training_belief(
                num_iterations, belief, params, rng
            )
            examples = self._plan_every_task(training_belief, params, rng)
            self._dataset = Dataset.concatenate([self._dataset, examples])
            if len(self._dataset) > 0:
                params = dataclasses.replace(
                    params, model=params.model.fit(self._dataset)
                )
        return params

    @abstractmethod
    def _get_training_belief(
        self,
        iteration: int,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        rng: np.random.Generator,
    ) -> IntervalBelief:
        """The belief to plan under in an iteration, numbered from 1, given the
        robot's current belief."""

    def _plan_every_task(
        self,
        belief: IntervalBelief,
        params: RandomShootingParams,
        rng: np.random.Generator,
    ) -> Dataset:
        """Plan for every task under the belief and return the plans as training
        rows, one group per plan."""
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
