"""Tests for envs/limb/model.py."""

import numpy as np
import pytest

from able.envs.limb.human import get_variant
from able.envs.limb.model import STOP, LimbModel, LimbTask
from able.envs.limb.query import LimbQuery, QueryOutcome

THETA = np.array([1.0, 1.0, 1.0])


@pytest.fixture(name="model", scope="module")
def _model():
    # The bed scene's own grasp is clear of the person and holds still.
    model = LimbModel(get_variant("bed-left-arm"), horizon=5)
    yield model
    model.close()


@pytest.fixture(name="start", scope="module")
def _start(model):
    return model.sim.capture()


def _make_task(model: LimbModel, start, elbow_offset: float) -> LimbTask:
    """A task whose goal bends the elbow by the offset from the start."""
    goal = np.array(start.limb_positions)
    goal[3] += elbow_offset
    return LimbTask(initial_state=model.get_initial_state(start, goal), goal=goal)


def test_initial_state(model, start):
    """Tests that a task starts with the goal, the limb's resting contact, and
    nothing seen."""
    task = _make_task(model, start, 0.5)
    state = task.initial_state
    assert np.array_equal(state.goal, task.goal)
    assert not state.reached and not state.failure and not state.violated
    assert state.signalled == (None, None, None)
    assert state.resting_penetration
    assert np.allclose(state.max_reach, model.get_reach(state.limb_positions))
    assert _make_task(model, start, 0.0).initial_state.reached


def test_steps_are_deterministic(model, start):
    """Tests that stepping from a state gives the same result, also after the
    simulator has moved on."""
    rng = np.random.default_rng(0)
    state = _make_task(model, start, 0.5).initial_state
    action = np.array([1.0, -1.0, 0.5, 0.0, 0.2, 0.0, 0.0])
    first = model.sample_next_state(state, action, THETA, rng)
    model.sample_next_state(first, action, THETA, rng)
    second = model.sample_next_state(state, action, THETA, rng)
    assert np.allclose(first.limb_positions, second.limb_positions)
    assert not np.allclose(first.limb_positions, state.limb_positions)


def test_range_changes_the_physics_only_past_it(model, start):
    """Tests that two people whose ranges cover the motion move the same way."""
    rng = np.random.default_rng(0)
    state = _make_task(model, start, 0.5).initial_state
    action = np.array([1.0, -1.0, 0.5, 0.0, 0.2, 0.0, 0.0])
    narrow, wide = THETA, 1.15 * THETA
    first = model.sample_next_state(state, action, narrow, rng)
    second = model.sample_next_state(state, action, wide, rng)
    assert not first.violated
    assert np.allclose(first.limb_positions, second.limb_positions)


def test_reaching_the_goal_ends_the_task(model, start):
    """Tests that the state stays fixed once the limb reaches the goal, which
    completes the task."""
    rng = np.random.default_rng(0)
    task = _make_task(model, start, 0.15)
    state = task.initial_state
    for _ in range(50):
        if state.reached:
            break
        action = model.sim.get_limb_directed_torque(task.goal, gain=2.0)
        next_state = model.sample_next_state(state, action, THETA, rng)
        assert model.get_cost(task, state, action, next_state, THETA) == 0.0
        state = next_state
    assert state.reached
    assert model.sample_next_state(state, np.full(7, 5.0), THETA, rng) is state
    assert model.get_terminal_cost(task, state, THETA) < model.sim.goal_tolerance
    assert model.check_goal(task, state, THETA)


def test_passing_the_limit(model, start):
    """Tests that a person with a narrow range signals the limit, which fails the
    task."""
    rng = np.random.default_rng(0)
    task = _make_task(model, start, 0.5)
    state = task.initial_state
    reach = np.array(state.max_reach)
    # Put each limit just inside the limb's current reach, where it is positive.
    narrow = np.where(reach > 0.05, reach - 0.05, 1.0)
    next_state = model.sample_next_state(state, np.zeros(7), narrow, rng)
    assert next_state.violated
    assert not model.check_goal(task, next_state, narrow)
    for motion, angle in enumerate(next_state.signalled):
        assert (angle is not None) == (reach[motion] > 0.05)


def test_broken_constraints_freeze_the_state(model, start):
    """Tests that a torque past the robot's limits fails and stops the motion."""
    rng = np.random.default_rng(0)
    task = _make_task(model, start, 0.5)
    failed = model.sample_next_state(task.initial_state, np.full(7, 1e4), THETA, rng)
    assert failed.failure == "robot torque limits" and not failed.reached
    assert model.sample_next_state(failed, np.zeros(7), THETA, rng) is failed
    assert not model.check_goal(task, failed, THETA)


def test_stopping_before_the_goal(model, start):
    """Tests that stopping fails the task and freezes the state without new
    evidence."""
    rng = np.random.default_rng(0)
    task = _make_task(model, start, 0.5)
    moved = model.sample_next_state(task.initial_state, np.full(7, 0.2), THETA, rng)
    assert not moved.failure and not moved.reached
    stopped = model.sample_next_state(moved, STOP, THETA, rng)
    assert stopped.failure == "stopped before the goal"
    assert stopped.coupled is moved.coupled
    assert stopped.max_reach == moved.max_reach
    assert stopped.signalled == moved.signalled
    assert model.sample_next_state(stopped, np.full(7, 1.0), THETA, rng) is stopped
    assert not model.check_goal(task, stopped, THETA)


def test_sample_response(model):
    """Tests that queries are answered for the person with the hidden parameters."""
    response = model.sample_response(LimbQuery(0, 0.3), THETA, np.random.default_rng(0))
    assert isinstance(response, QueryOutcome)
    assert response.reached_target
