"""Tests for info_gatherers/base.py."""

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.envs.simulated import SimulatedEnvironment
from able.info_gatherers.base import NullInformationGatherer
from able.structs import Budget
from tests.envs.wall_model import WallModel


def test_null_information_gatherer():
    """Tests that NullInformationGatherer asks nothing."""
    rng = np.random.default_rng(0)
    env = SimulatedEnvironment(WallModel(), np.array([1.5]), rng)
    gatherer: NullInformationGatherer = NullInformationGatherer()
    data = gatherer.gather(
        IntervalBelief([[0.0, 3.0]]), env, Budget(max_iterations=5), rng
    )
    assert not data
