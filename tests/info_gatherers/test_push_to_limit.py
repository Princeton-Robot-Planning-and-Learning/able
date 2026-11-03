"""Tests for info_gatherers/push_to_limit.py."""

import numpy as np
import pytest

from able.belief_updaters.range_of_motion import RangeOfMotionBeliefUpdater
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import RangeOfMotionPrior, get_variant
from able.envs.limb.model import LimbModel
from able.envs.simulated import SimulatedEnvironment
from able.info_gatherers.push_to_limit import PushToLimitGatherer
from able.structs import Budget

VARIANT = get_variant("human-left-arm")


def _prior() -> RangeOfMotionBelief:
    prior = RangeOfMotionPrior()
    return RangeOfMotionBelief.from_prior(
        prior.mean, prior.stddev, prior.low, prior.high, 3
    )


def _gatherer(probe_band: float = 0.3) -> PushToLimitGatherer:
    return PushToLimitGatherer(VARIANT, RangeOfMotionBeliefUpdater(VARIANT), probe_band)


def test_choose():
    """Tests that targets lie near the normal range, or above a higher floor."""
    gatherer = _gatherer()
    belief = _prior()
    rng = np.random.default_rng(0)
    for _ in range(50):
        query = gatherer.choose(belief, rng)
        limit = VARIANT.default_magnitudes[query.motion]
        assert limit - 0.3 <= query.target <= limit + 0.3
    exact = _gatherer(probe_band=0.0)
    query = exact.choose(belief, rng)
    raised = belief.with_floor(query.motion, 1.1)
    assert exact.choose(raised, rng).motion == query.motion
    assert exact.choose(raised, rng).target == pytest.approx(1.1 * query.target)


def test_gather_narrows_the_belief_toward_the_person():
    """Tests that queries narrow the belief and never exclude the person."""
    theta = np.array([0.85, 1.05, 0.95])
    rng = np.random.default_rng(7)
    model = LimbModel(VARIANT)
    env = SimulatedEnvironment(model, theta, rng)
    updater = RangeOfMotionBeliefUpdater(VARIANT)
    belief = _prior()
    try:
        data = _gatherer().gather(belief, env, Budget(max_iterations=6), rng)
    finally:
        model.close()
    assert len(data) == 6
    after = updater.update_from_queries(belief, data)
    assert np.all(after.stddev <= belief.stddev)
    assert after.stddev.sum() < belief.stddev.sum()
    assert np.all(after.lower <= theta)
    assert (
        np.abs(after.truncated_mean - theta).sum()
        < np.abs(belief.truncated_mean - theta).sum()
    )
    with pytest.raises(AssertionError):
        _gatherer().gather(belief, env, Budget(), rng)
