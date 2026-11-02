"""Tests for learners/base.py."""

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR
from able.learners.base import NullLearner
from able.planners.random_shooting import RandomShootingParams
from able.structs import Budget, Task, Trajectory


def test_null_learner():
    """Tests that NullLearner returns the parameters unchanged."""
    params = RandomShootingParams()
    learner: NullLearner = NullLearner()
    new_params = learner.learn(
        IntervalBelief(JOINT_LIMIT_PRIOR), params, Budget(), np.random.default_rng(0)
    )
    assert new_params is params


def test_null_learner_ignores_executions():
    """Tests that observing an execution leaves what NullLearner learns unchanged."""
    params = RandomShootingParams()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    learner: NullLearner = NullLearner()
    task = Task(initial_state=np.zeros(2), goal=np.ones(2))
    trajectory = Trajectory(states=(np.zeros(2),), actions=(), costs=())
    learner.observe_execution(task, trajectory, belief)
    assert learner.learn(belief, params, Budget(), np.random.default_rng(0)) is params
