"""Tests for learners/plan_caching.py."""

import numpy as np
import pytest

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, ArmTask, TwoLinkArmModel
from able.learners.plan_caching import ArmArray, PlanCachingLearner
from able.models.exact_match_cache import ExactMatchCache
from able.planners.random_shooting import (
    RandomShootingParams,
    RandomShootingPlanner,
    get_plan_features,
)
from able.structs import Budget

FIXED_BELIEF = IntervalBelief(
    np.stack([JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 0] + 0.5], axis=1)
)


class _FixedBeliefLearner(PlanCachingLearner):
    """Trains under the same belief in every iteration and records the iterations."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.iterations: list[int] = []

    def _get_training_belief(
        self,
        iteration: int,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        rng: np.random.Generator,
    ) -> IntervalBelief:
        self.iterations.append(iteration)
        return FIXED_BELIEF


def test_plan_caching_learner():
    """Tests that each iteration caches a plan for every task under the chosen belief,
    without changing the inputs."""
    tasks = [
        ArmTask(initial_state=np.zeros(2), goal=np.array(goal))
        for goal in [(1.0, 1.0), (0.5, 1.5), (-1.0, 1.2)]
    ]
    cache = ExactMatchCache(max_per_key=2)
    planner = RandomShootingPlanner(
        TwoLinkArmModel(), RandomShootingParams(num_proposals=2, model=cache)
    )
    learner = _FixedBeliefLearner(FiniteTaskDistribution(tasks), planner, Budget())
    params = planner.get_default_params()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    new_params = learner.learn(
        belief, params, Budget(max_iterations=3), np.random.default_rng(0)
    )
    assert learner.iterations == [1, 2, 3]
    assert params.model is cache and not cache.entries
    trained = new_params.model
    assert isinstance(trained, ExactMatchCache)
    assert set(trained.entries) == {
        cache.get_key(get_plan_features(task, FIXED_BELIEF)) for task in tasks
    }
    # Three iterations each add one plan per task, and at most two are kept.
    assert all(len(entries) == 2 for entries in trained.entries.values())
    for entries in trained.entries.values():
        assert entries[0][0] <= entries[1][0]

    # The next sleep keeps the plans from this one.
    newer_params = learner.learn(
        belief, new_params, Budget(max_iterations=1), np.random.default_rng(1)
    )
    assert learner.iterations == [1, 2, 3, 1]
    assert isinstance(newer_params.model, ExactMatchCache)
    assert set(newer_params.model.entries) == set(trained.entries)

    with pytest.raises(AssertionError):
        learner.learn(belief, params, Budget(), np.random.default_rng(0))
