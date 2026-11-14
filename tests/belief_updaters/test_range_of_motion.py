"""Tests for belief_updaters/range_of_motion.py."""

import numpy as np
import pytest
from pybullet_helpers.geometry import SE2Pose

from able.belief_updaters.range_of_motion import RangeOfMotionBeliefUpdater
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import RangeOfMotionPrior, get_variant
from able.envs.limb.model import LimbState
from able.envs.limb.query import LimbQuery, QueryModel, QueryOutcome
from able.envs.limb.sim import CoupledState
from able.structs import QueryResponse

VARIANT = get_variant("human-left-arm")
MAGNITUDES = VARIANT.default_magnitudes
MODEL = QueryModel()


def _prior() -> RangeOfMotionBelief:
    prior = RangeOfMotionPrior()
    return RangeOfMotionBelief.from_prior(
        prior.mean, prior.stddev, prior.low, prior.high, 3
    )


def _state(max_reach, signalled=(None, None, None)) -> LimbState:
    coupled = CoupledState(SE2Pose(0.0, 0.0, 0.0), (), (), (0.0,) * 6, (), None)
    return LimbState(coupled, np.zeros(6), tuple(max_reach), tuple(signalled), ())


def _response(motion, answer, reached_target=False, settled=True, peak=0.5):
    outcome = QueryOutcome(motion, 2.0, answer, reached_target, 4.0, settled, peak)
    return QueryResponse(LimbQuery(motion, 2.0), outcome)


def test_update_from_queries():
    """Tests that settled answers measure, reaching the target raises the floor,
    and other answers show nothing."""
    updater = RangeOfMotionBeliefUpdater(VARIANT, MODEL)
    belief = _prior()
    after = updater.update_from_queries(belief, [_response(0, 0.8 * MAGNITUDES[0])])
    assert after.stddev[0] < belief.stddev[0]
    assert np.array_equal(after.stddev[1:], belief.stddev[1:])
    expected = (0.8 * MAGNITUDES[0] + MODEL.short_fall) / MAGNITUDES[0]
    exact = RangeOfMotionBeliefUpdater(VARIANT, QueryModel(short_fall_stddev=1e-6))
    exact_after = exact.update_from_queries(belief, [_response(0, 0.8 * MAGNITUDES[0])])
    assert exact_after.mean[0] == pytest.approx(expected, abs=1e-6)

    censored = updater.update_from_queries(
        belief, [_response(1, MAGNITUDES[1], reached_target=True)]
    )
    assert censored.lower[1] == pytest.approx(1.0 + MODEL.short_fall / MAGNITUDES[1])
    assert np.array_equal(censored.mean, belief.mean)

    ignored = [
        _response(0, 0.8 * MAGNITUDES[0], settled=False),
        _response(0, 0.8 * MAGNITUDES[0], peak=MODEL.max_velocity),
    ]
    assert updater.update_from_queries(belief, ignored) == belief


def test_update_from_transition():
    """Tests that reach raises the floor and a signal measures the limit once."""
    updater = RangeOfMotionBeliefUpdater(VARIANT, QueryModel(short_fall_stddev=1e-6))
    belief = _prior()
    start = _state([0.2, 0.0, 0.0])
    inside = _state([0.75, 0.0, 0.0])
    after = updater.update_from_transition(belief, start, np.zeros(7), inside)
    assert after.lower[0] == pytest.approx(0.75)
    assert np.array_equal(after.mean, belief.mean)
    # Reaching the same angle again changes nothing.
    again = _state([0.7505, 0.0, 0.0])
    assert updater.update_from_transition(after, inside, np.zeros(7), again) == after

    signalled = _state([0.82, 0.0, 0.0], (0.81 * MAGNITUDES[0], None, None))
    crossed = updater.update_from_transition(belief, start, np.zeros(7), signalled)
    assert crossed.mean[0] == pytest.approx(0.81, abs=1e-3)
    assert crossed.stddev[0] < 1e-3
    assert np.array_equal(crossed.floor, belief.floor)
    # A signal that was already seen is not counted again.
    later = updater.update_from_transition(crossed, signalled, np.zeros(7), signalled)
    assert later == crossed


def test_random_motion_never_excludes_the_person():
    """Tests that floors from motion inside the person's range stay below it."""
    updater = RangeOfMotionBeliefUpdater(VARIANT, MODEL)
    theta = np.array([0.85, 1.05, 0.95])
    rng = np.random.default_rng(0)
    belief = _prior()
    state = _state([0.0, 0.0, 0.0])
    for _ in range(100):
        reach = np.maximum(state.max_reach, rng.uniform(0.0, theta))
        next_state = _state(reach)
        belief = updater.update_from_transition(belief, state, np.zeros(7), next_state)
        state = next_state
    assert np.all(belief.lower <= theta)
    assert np.all(belief.lower > RangeOfMotionPrior().low)
