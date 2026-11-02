"""Tests for belief_updaters/joint_limits.py."""

import numpy as np
import pytest

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, JointQuery, TwoLinkArmModel
from able.structs import QueryResponse


def _execute(
    belief: Belief[np.ndarray], true_limits: np.ndarray, actions: np.ndarray
) -> Belief[np.ndarray]:
    """Execute the actions from q = 0 on an arm with the true limits, updating the
    belief after every step."""
    model = TwoLinkArmModel()
    updater = JointLimitBeliefUpdater()
    rng = np.random.default_rng(0)
    state = np.zeros(2)
    for action in actions:
        next_state = model.sample_next_state(state, action, true_limits, rng)
        belief = updater.update_from_transition(belief, state, action, next_state)
        state = next_state
    return belief


def test_clipped_step_pins_limit():
    """Tests that hitting a limit makes it known exactly."""
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    true_limits = np.array([-0.7, 2.0, -1.5, 0.8])
    # Joint 1 moves down until it hits -0.7; joint 2 moves up without hitting 0.8.
    belief = _execute(prior, true_limits, np.array([[-0.3, 0.2]] * 3))
    assert isinstance(belief, IntervalBelief)
    assert np.allclose(belief.intervals[0], [-0.7, -0.7])
    assert np.allclose(belief.intervals[1], JOINT_LIMIT_PRIOR[1])
    assert np.allclose(belief.intervals[2], JOINT_LIMIT_PRIOR[2])
    # Joint 2 reached 0.6, so its upper limit is at least 0.6.
    assert np.allclose(belief.intervals[3], [0.6, np.pi])
    assert belief.contains(true_limits)
    # The prior is unchanged.
    assert prior == IntervalBelief(JOINT_LIMIT_PRIOR)


def test_unclipped_steps_narrow_intervals():
    """Tests that moving freely rules out limits the arm moved past."""
    true_limits = np.array([-2.0, 2.0, -2.0, 2.0])
    belief = _execute(
        IntervalBelief(JOINT_LIMIT_PRIOR), true_limits, np.array([[0.3, -0.3]] * 3)
    )
    belief = _execute(belief, true_limits, np.array([[-0.25, 0.25]] * 4))
    assert isinstance(belief, IntervalBelief)
    assert np.allclose(belief.intervals[0], [-np.pi, -1.0])
    assert np.allclose(belief.intervals[1], [0.9, np.pi])
    assert np.allclose(belief.intervals[2], [-np.pi, -0.9])
    assert np.allclose(belief.intervals[3], [1.0, np.pi])


def test_belief_always_contains_true_limits():
    """Tests that random motion never rules out the true limits and that the belief only
    shrinks."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        true_limits = rng.uniform(JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 1])
        belief = IntervalBelief(JOINT_LIMIT_PRIOR)
        for _ in range(5):
            base = rng.uniform(-0.3, 0.3, size=(1, 2))
            actions = np.clip(base + rng.normal(0, 0.05, size=(15, 2)), -0.3, 0.3)
            new_belief = _execute(belief, true_limits, actions)
            assert isinstance(new_belief, IntervalBelief)
            assert new_belief.contains(true_limits)
            assert new_belief.get_total_width() <= belief.get_total_width() + 1e-12
            belief = new_belief


def test_queries_have_no_effect():
    """Tests that responses to joint queries leave the belief unchanged."""
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    data = [
        QueryResponse(JointQuery(joint=0, angle=1.0), True),
        QueryResponse(JointQuery(joint=1, angle=-3.0), False),
    ]
    assert JointLimitBeliefUpdater().update_from_queries(prior, data) is prior


def test_requires_interval_belief():
    """Tests that the updater rejects other kinds of belief."""
    with pytest.raises(AssertionError):
        JointLimitBeliefUpdater().update_from_transition(
            object(), np.zeros(2), np.zeros(2), np.zeros(2)  # type: ignore[arg-type]
        )
