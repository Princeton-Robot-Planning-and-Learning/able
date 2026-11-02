"""Tests for learners/base.py."""

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR
from able.learners.base import NullLearner
from able.planners.random_shooting import RandomShootingParams
from able.structs import Budget


def test_null_learner():
    """Tests that NullLearner returns the parameters unchanged."""
    params = RandomShootingParams()
    learner: NullLearner = NullLearner()
    new_params = learner.learn(
        IntervalBelief(JOINT_LIMIT_PRIOR), params, Budget(), np.random.default_rng(0)
    )
    assert new_params is params
