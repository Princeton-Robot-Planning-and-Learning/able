"""Tests for belief_updaters/joint_limits.py."""

import numpy as np
import pytest

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, ArmTask, TwoLinkArmModel
from able.structs import Trajectory

TASK = ArmTask(initial_state=np.zeros(2), goal=np.array([1.0, 1.0]))


def _run(true_limits: np.ndarray, actions: np.ndarray) -> Trajectory:
    """Execute the actions on an arm with the true limits."""
    model = TwoLinkArmModel(horizon=len(actions))
    env = SimulatedEnvironment(model, true_limits, np.random.default_rng(0))
    states = [env.reset(TASK)]
    costs = []
    for action in actions:
        result = env.step(action)
        states.append(result.state)
        costs.append(result.cost)
    return Trajectory(tuple(states), tuple(actions), tuple(costs))


def test_clipped_step_pins_limit():
    """Tests that hitting a limit makes it known exactly."""
    updater = JointLimitBeliefUpdater()
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    true_limits = np.array([-0.7, 2.0, -1.5, 0.8])
    # Joint 1 moves down until it hits -0.7; joint 2 moves up without hitting 0.8.
    actions = np.array([[-0.3, 0.2]] * 3)
    belief = updater.update(prior, [_run(true_limits, actions)])
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
    updater = JointLimitBeliefUpdater()
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    true_limits = np.array([-2.0, 2.0, -2.0, 2.0])
    up = _run(true_limits, np.array([[0.3, -0.3]] * 3))
    down = _run(true_limits, np.array([[-0.25, 0.25]] * 4))
    belief = updater.update(prior, [up, down])
    assert np.allclose(belief.intervals[0], [-np.pi, -1.0])
    assert np.allclose(belief.intervals[1], [0.9, np.pi])
    assert np.allclose(belief.intervals[2], [-np.pi, -0.9])
    assert np.allclose(belief.intervals[3], [1.0, np.pi])


def test_belief_always_contains_true_limits():
    """Tests that random motion never rules out the true limits and that the belief only
    shrinks."""
    updater = JointLimitBeliefUpdater()
    rng = np.random.default_rng(0)
    for _ in range(10):
        true_limits = rng.uniform(JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 1])
        belief = IntervalBelief(JOINT_LIMIT_PRIOR)
        for _ in range(5):
            base = rng.uniform(-0.3, 0.3, size=(1, 2))
            actions = np.clip(base + rng.normal(0, 0.05, size=(15, 2)), -0.3, 0.3)
            new_belief = updater.update(belief, [_run(true_limits, actions)])
            assert new_belief.contains(true_limits)
            assert new_belief.get_total_width() <= belief.get_total_width() + 1e-12
            belief = new_belief


def test_requires_interval_belief():
    """Tests that the updater rejects other kinds of belief."""
    with pytest.raises(AssertionError):
        JointLimitBeliefUpdater().update(object(), [])  # type: ignore[arg-type]
