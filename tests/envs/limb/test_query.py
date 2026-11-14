"""Tests for envs/limb/query.py."""

import numpy as np
import pytest

from able.envs.limb.human import get_variant
from able.envs.limb.query import LimbQuery, QueryModel, QuerySimulator


def test_query_model():
    """Tests the short fall's distribution and its bounds."""
    model = QueryModel()
    rng = np.random.default_rng(0)
    draws = np.array([model.sample_short_fall(rng) for _ in range(2000)])
    assert 0.0 < draws.min() and draws.max() < model.max_short_fall
    assert np.mean(draws) == pytest.approx(model.short_fall, abs=0.01)
    assert model.noise_stddev == pytest.approx(0.05)
    with pytest.raises(ValueError):
        QueryModel(short_fall_mean=0.4)


@pytest.mark.parametrize("name", ["human-left-arm", "bed-right-leg"])
def test_queries_stop_short_of_the_limit(name):
    """Tests that the person stops their short fall inside their range, unless the
    target comes first."""
    variant = get_variant(name)
    theta = np.array([0.9, 1.0, 0.8])
    limits = np.abs(variant.get_bounds(theta))
    simulator = QuerySimulator(variant)
    model = simulator.model
    rng = np.random.default_rng(0)
    try:
        for motion in range(3):
            outcome = simulator.ask(theta, LimbQuery(motion, limits[motion] + 0.5), rng)
            assert not outcome.reached_target and outcome.settled
            assert outcome.peak_velocity < model.max_velocity
            short_fall = limits[motion] - outcome.answer
            assert short_fall == pytest.approx(outcome.short_fall, abs=0.01)
            assert 0.0 < short_fall < model.max_short_fall + 0.01
        outcome = simulator.ask(theta, LimbQuery(0, 0.5), rng)
        assert outcome.reached_target and outcome.settled
        assert outcome.answer == pytest.approx(0.5, abs=0.01)
    finally:
        simulator.close()
