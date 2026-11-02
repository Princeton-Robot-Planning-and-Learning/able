"""Tests for learners/plan_caching.py."""

import numpy as np
import pytest

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, ArmTask, TwoLinkArmModel
from able.learners.plan_caching import ArmArray, PlanCachingLearner
from able.planners.random_shooting import (
    RandomShootingParams,
    RandomShootingPlanner,
    get_warm_start_key,
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
    planner = RandomShootingPlanner(TwoLinkArmModel())
    learner = _FixedBeliefLearner(
        FiniteTaskDistribution(tasks), planner, Budget(), max_warm_starts_per_key=2
    )
    params = planner.get_default_params()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    new_params = learner.learn(
        belief, params, Budget(max_iterations=3), np.random.default_rng(0)
    )
    assert learner.iterations == [1, 2, 3]
    assert not params.warm_starts
    assert set(new_params.warm_starts) == {
        get_warm_start_key(task, FIXED_BELIEF, params.key_resolution) for task in tasks
    }
    # Three iterations each add one plan per task, and at most two are kept.
    assert all(len(entries) == 2 for entries in new_params.warm_starts.values())
    for entries in new_params.warm_starts.values():
        assert entries[0][0] <= entries[1][0]

    with pytest.raises(AssertionError):
        learner.learn(belief, params, Budget(), np.random.default_rng(0))
