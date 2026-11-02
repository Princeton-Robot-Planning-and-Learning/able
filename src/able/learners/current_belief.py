"""Learning warm starts for random shooting under the current belief only."""

import numpy as np

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.learners.plan_caching import ArmArray, PlanCachingLearner
from able.planners.random_shooting import RandomShootingParams


class CurrentBeliefLearner(PlanCachingLearner):
    """Caches plans for every task under the robot's current belief.

    This is a baseline for DreamingLearner that never anticipates belief updates.
    Each iteration plans for every task under the current belief, starting from
    the plans that earlier iterations cached, so the budget refines the plans for
    one belief instead of covering several imagined ones. The cached plans help
    in evaluation only when the belief has not changed since the robot slept.
    """

    def _get_training_belief(
        self,
        iteration: int,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        rng: np.random.Generator,
    ) -> IntervalBelief:
        del iteration, params, rng  # unused because the belief is always the same
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        return belief
