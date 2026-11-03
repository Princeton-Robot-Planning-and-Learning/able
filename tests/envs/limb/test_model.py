"""Tests for envs/limb/model.py."""

import numpy as np
import pytest

from able.envs.limb.human import get_variant
from able.envs.limb.model import LimbModel, LimbTask
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


def test_initial_state(model, start):
    """Tests that a task starts with the limb's resting contact and nothing seen."""
    state = model.get_initial_state(start)
    assert state.signalled == (None, None, None)
    assert not state.failure and not state.violated
    assert state.resting_penetration
    assert np.allclose(state.max_reach, model.get_reach(state.limb_positions))


def test_steps_are_deterministic(model, start):
    """Tests that stepping from a state gives the same result, also after the
    simulator has moved on."""
    rng = np.random.default_rng(0)
    state = model.get_initial_state(start)
    action = np.array([1.0, -1.0, 0.5, 0.0, 0.2, 0.0, 0.0])
    first = model.sample_next_state(state, action, THETA, rng)
    model.sample_next_state(first, action, THETA, rng)
    second = model.sample_next_state(state, action, THETA, rng)
    assert np.allclose(first.limb_positions, second.limb_positions)
    assert not np.allclose(first.limb_positions, state.limb_positions)


def test_holding_still_reaches_a_goal_at_the_start(model, start):
    """Tests costs and the goal check for a task whose goal is the start."""
    rng = np.random.default_rng(0)
    state = model.get_initial_state(start)
    task = LimbTask(initial_state=state, goal=state.limb_positions)
    for _ in range(model.horizon):
        next_state = model.sample_next_state(state, np.zeros(7), THETA, rng)
        assert model.get_cost(task, state, np.zeros(7), next_state, THETA) == 0.0
        state = next_state
    assert model.get_terminal_cost(task, state, THETA) < 0.01
    assert model.check_goal(task, state, THETA)


def test_passing_the_limit(model, start):
    """Tests that a person with a narrow range signals the limit, which fails the
    task."""
    rng = np.random.default_rng(0)
    state = model.get_initial_state(start)
    task = LimbTask(initial_state=state, goal=state.limb_positions)
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
    state = model.get_initial_state(start)
    failed = model.sample_next_state(state, np.full(7, 1e4), THETA, rng)
    assert failed.failure == "robot torque limits"
    assert model.sample_next_state(failed, np.zeros(7), THETA, rng) is failed
    task = LimbTask(initial_state=state, goal=state.limb_positions)
    assert not model.check_goal(task, failed, THETA)


def test_sample_response(model):
    """Tests that queries are answered for the person with the hidden parameters."""
    response = model.sample_response(LimbQuery(0, 0.3), THETA, np.random.default_rng(0))
    assert isinstance(response, QueryOutcome)
    assert response.reached_target
