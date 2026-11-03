"""Tests for envs/limb/query.py."""

import numpy as np
import pytest

from able.envs.limb.human import get_variant
from able.envs.limb.query import LimbQuery, QueryModel, QuerySimulator


def test_query_model():
    """Tests the effort, the restoring torque, and the stopping point."""
    model = QueryModel()
    rng = np.random.default_rng(0)
    efforts = np.array([model.sample_effort(rng) for _ in range(2000)])
    assert 0.0 < efforts.min() and efforts.max() < model.stiffness * model.zone_width
    assert 0.0 < model.short_fall < model.zone_width
    assert model.noise_stddev == pytest.approx(0.05)

    limit = 2.0
    assert model.get_restoring_torque(limit - model.zone_width - 0.1, limit) == 0.0
    inside = model.get_restoring_torque(limit - model.zone_width / 2, limit)
    assert inside < 0.0
    assert model.get_restoring_torque(limit + 0.05, limit) < inside
    with pytest.raises(ValueError):
        QueryModel(zone_width=0.1, stiffness=10.0, effort_mean=5.0)


def test_queries_stop_short_of_the_limit():
    """Tests that the person stops inside their range, unless the target comes
    first."""
    variant = get_variant("human-left-arm")
    theta = np.array([0.9, 1.0, 0.8])
    limits = np.abs(variant.get_bounds(theta))
    simulator = QuerySimulator(variant)
    rng = np.random.default_rng(0)
    try:
        for motion in range(3):
            outcome = simulator.ask(theta, LimbQuery(motion, limits[motion] + 0.5), rng)
            assert not outcome.reached_target and outcome.settled
            short_fall = limits[motion] - outcome.answer
            assert 0.0 < short_fall < 2 * QueryModel().zone_width
        outcome = simulator.ask(theta, LimbQuery(0, 0.5), rng)
        assert outcome.reached_target
        assert outcome.answer == pytest.approx(0.5, abs=0.05)
    finally:
        simulator.close()
