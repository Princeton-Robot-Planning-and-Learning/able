"""Learning warm starts for random shooting under the prior, once."""

import numpy as np

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import ArmTask
from able.learners.plan_caching import ArmArray, PlanCachingLearner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.structs import Budget


class PriorOnlyLearner(PlanCachingLearner):
    """Caches plans for every task under the prior, in the first sleep only.

    This is a baseline that trains once, before the robot has any evidence about
    the hidden parameters, and never again. The first sleep uses its full budget
    to cache plans under the prior, and later sleeps leave the planner parameters
    unchanged. Cached plans are keyed on the belief, so they are only used while
    the robot's belief still matches the prior.
    """

    def __init__(
        self,
        prior: IntervalBelief,
        task_distribution: FiniteTaskDistribution[ArmTask],
        planner: RandomShootingPlanner,
        planning_budget: Budget,
        max_warm_starts_per_key: int = 4,
    ) -> None:
        super().__init__(
            task_distribution, planner, planning_budget, max_warm_starts_per_key
        )
        self._prior = prior
        self._has_trained = False

    def learn(
        self,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> RandomShootingParams:
        """Cache plans under the prior the first time, and do nothing after."""
        if self._has_trained:
            return params
        self._has_trained = True
        return super().learn(belief, params, budget, rng)

    def _get_training_belief(
        self,
        iteration: int,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        rng: np.random.Generator,
    ) -> IntervalBelief:
        del iteration, belief, params, rng  # unused because training uses the prior
        return self._prior
