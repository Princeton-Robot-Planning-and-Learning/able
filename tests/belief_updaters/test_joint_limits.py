"""Tests for belief_updaters/joint_limits.py."""

import numpy as np
import pytest

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, JointQuery, TwoLinkArmModel
from able.structs import QueryResponse


def _ask(joint: int, angle: float, response: bool) -> QueryResponse[JointQuery, bool]:
    return QueryResponse(JointQuery(joint=joint, angle=angle), response)


def test_yes_narrows_both_limits():
    """Tests that a reachable angle bounds the lower limit from above and the upper
    limit from below."""
    updater = JointLimitBeliefUpdater()
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    belief = updater.update(prior, [_ask(0, 1.0, True), _ask(1, -1.0, True)])
    assert isinstance(belief, IntervalBelief)
    assert np.allclose(belief.intervals[0], JOINT_LIMIT_PRIOR[0])
    assert np.allclose(belief.intervals[1], [1.0, np.pi])
    assert np.allclose(belief.intervals[2], [-np.pi, -1.0])
    assert np.allclose(belief.intervals[3], JOINT_LIMIT_PRIOR[3])
    # The prior is unchanged.
    assert prior == IntervalBelief(JOINT_LIMIT_PRIOR)


def test_no_bounds_the_limit_on_the_side_of_the_angle():
    """Tests that an unreachable angle bounds the limit between it and zero."""
    updater = JointLimitBeliefUpdater()
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    belief = updater.update(prior, [_ask(0, 2.0, False), _ask(1, -2.0, False)])
    assert np.allclose(belief.intervals[0], JOINT_LIMIT_PRIOR[0])
    assert np.allclose(belief.intervals[1], [0.5, 2.0])
    assert np.allclose(belief.intervals[2], [-2.0, -0.5])
    assert np.allclose(belief.intervals[3], JOINT_LIMIT_PRIOR[3])


def test_bisection_pins_limit():
    """Tests that repeated yes and no answers shrink an interval around the true
    limit."""
    updater = JointLimitBeliefUpdater()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    true_upper = 1.234
    for _ in range(20):
        angle = float(belief.intervals[1].mean())
        belief = updater.update(belief, [_ask(0, angle, angle <= true_upper)])
    low, high = belief.intervals[1]
    assert low <= true_upper <= high
    assert high - low < 1e-5


def test_belief_always_contains_true_limits():
    """Tests that answers to random queries never rule out the true limits and that the
    belief only shrinks."""
    updater = JointLimitBeliefUpdater()
    model = TwoLinkArmModel()
    rng = np.random.default_rng(0)
    for _ in range(10):
        true_limits = rng.uniform(JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 1])
        belief = IntervalBelief(JOINT_LIMIT_PRIOR)
        for _ in range(5):
            queries = [
                JointQuery(joint=int(rng.integers(2)), angle=rng.uniform(-np.pi, np.pi))
                for _ in range(3)
            ]
            data = [
                QueryResponse(q, model.sample_response(q, true_limits, rng))
                for q in queries
            ]
            new_belief = updater.update(belief, data)
            assert new_belief.contains(true_limits)
            assert new_belief.get_total_width() <= belief.get_total_width() + 1e-12
            belief = new_belief


def test_rejects_invalid_inputs():
    """Tests that the updater rejects responses that contradict the belief, beliefs
    where a no would be ambiguous, and other kinds of belief."""
    updater = JointLimitBeliefUpdater()
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    # Every joint can move to 0.
    with pytest.raises(AssertionError):
        updater.update(prior, [_ask(0, 0.0, False)])
    # An angle of 0.5 might be below the lower limit or above the upper limit.
    overlapping = IntervalBelief([[-1.0, 1.0], [0.0, 2.0], [-1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(AssertionError):
        updater.update(overlapping, [_ask(0, 0.5, False)])
    with pytest.raises(AssertionError):
        updater.update(object(), [])  # type: ignore[arg-type]
