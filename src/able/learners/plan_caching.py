"""Learning warm starts for random shooting by caching plans for chosen beliefs."""

import dataclasses
import time
from abc import abstractmethod
from typing import Hashable

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import ArmTask
from able.learners.base import Learner
from able.planners.random_shooting import (
    RandomShootingParams,
    RandomShootingPlanner,
    WarmStart,
    add_warm_start,
    get_warm_start_key,
)
from able.structs import Budget, PlanningFailure

ArmArray = NDArray[np.float64]


class PlanCachingLearner(Learner[ArmArray, RandomShootingParams]):
    """Caches plans for every task under a belief chosen in each iteration.

    Each iteration chooses a belief to train under, plans for every task in the
    task distribution under that belief, and adds the plans to the warm starts.
    Subclasses decide which belief each iteration trains under. Each iteration
    plans with the warm starts cached by earlier iterations, and one iteration of
    the budget is one chosen belief.
    """

    def __init__(
        self,
        task_distribution: FiniteTaskDistribution[ArmTask],
        planner: RandomShootingPlanner,
        planning_budget: Budget,
        max_warm_starts_per_key: int = 4,
    ) -> None:
        self._task_distribution = task_distribution
        self._planner = planner
        self._planning_budget = planning_budget
        self._max_warm_starts_per_key = max_warm_starts_per_key

    def learn(
        self,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> RandomShootingParams:
        """Train until the budget runs out and return parameters with the new
        warm starts."""
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Learning needs a finite budget"
        warm_starts: dict[Hashable, tuple[WarmStart, ...]] = dict(params.warm_starts)
        start_time = time.perf_counter()
        num_iterations = 0
        while not budget.is_exhausted(time.perf_counter() - start_time, num_iterations):
            num_iterations += 1
            iteration_params = dataclasses.replace(
                params, warm_starts=dict(warm_starts)
            )
            training_belief = self._get_training_belief(
                num_iterations, belief, iteration_params, rng
            )
            self._cache_plans(training_belief, iteration_params, warm_starts, rng)
        return dataclasses.replace(params, warm_starts=warm_starts)

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

    def _cache_plans(
        self,
        belief: IntervalBelief,
        params: RandomShootingParams,
        warm_starts: dict[Hashable, tuple[WarmStart, ...]],
        rng: np.random.Generator,
    ) -> None:
        """Plan for every task under the belief and add the plans to the warm
        starts in place."""
        for task in self._task_distribution.get_tasks():
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
            except PlanningFailure:
                continue
            add_warm_start(
                warm_starts,
                get_warm_start_key(task, belief, params.key_resolution),
                policy.get_planning_stats()["expected_cost"],
                np.array(policy.actions),
                self._max_warm_starts_per_key,
            )
