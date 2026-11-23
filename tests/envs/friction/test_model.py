"""Tests for envs/friction/model.py."""

import math
from dataclasses import replace

import numpy as np
import pytest

from able.envs.friction.model import (
    NO_PUSH,
    FrictionModel,
    FrictionState,
    PuckRewards,
)
from able.envs.friction.query import FrictionQuery
from able.envs.friction.world import THREE_ROUTES, Push
from able.structs import Task

FRICTIONS = np.array([0.03, 0.07, 0.15])


@pytest.fixture(name="model", scope="module")
def _model():
    model = FrictionModel()
    yield model
    model.close()


def _state_at(puck: tuple[float, float]) -> FrictionState:
    robot = (1.2, 1.0)
    return FrictionState(puck, robot, puck, (*robot, 0.0), THREE_ROUTES.goal)


def _task(model: FrictionModel) -> Task:
    goal = model.world.puck_map.goal
    return Task(model.get_initial_state(), (goal[0], goal[1]))


def test_a_push_moves_the_puck_and_the_robot(model):
    """A feasible push records the measurements and costs the push and the drive."""
    state = model.get_initial_state()
    task = _task(model)
    assert state.puck == model.world.puck_map.puck_start and model.horizon == 15
    push = Push(math.pi / 2, 0.5, 0.0)
    next_state = model.sample_next_state(
        state, push, FRICTIONS, np.random.default_rng(0)
    )
    record = next_state.last_push
    assert record is not None and record.feasible and record.puck == state.puck
    assert next_state.pushes == 1 and not next_state.ended
    assert next_state.robot != state.robot
    assert np.linalg.norm(np.subtract(next_state.puck, next_state.true_puck)) < 0.1
    # The task's start is known; the next start is the measured stop.
    assert not state.puck_measured and not record.start_measured
    assert next_state.puck_measured
    after = model.sample_next_state(
        next_state, Push(-math.pi / 2, 0.3, 0.0), FRICTIONS, np.random.default_rng(1)
    )
    assert after.last_push is not None and after.last_push.start_measured
    cost = model.get_cost(task, state, push, next_state, FRICTIONS)
    assert cost == pytest.approx(1.0 + 0.5 * record.drive_distance)
    assert not model.check_goal(task, next_state, FRICTIONS)


def test_a_puck_stopped_in_the_slot_is_shed(model):
    """A puck that stops where the robot cannot push it rests, as the robot sees,
    at the nearest spot it can push from, while the record keeps the stop."""
    state = model.get_initial_state()
    push = Push(0.0, 0.9)
    next_state = model.sample_next_state(
        state, push, FRICTIONS, np.random.default_rng(0)
    )
    record = next_state.last_push
    assert record is not None and record.stop[0] > 2.1  # inside the slot
    assert next_state.puck == next_state.true_puck and next_state.puck[0] < 2.0
    assert not next_state.puck_measured
    rest = np.array(next_state.puck)
    assert np.array_equal(model.world.shed(rest, state.goal), rest)


def test_a_hole_ends_the_task(model):
    """Passing over a hole costs the penalty, and the state stays fixed after."""
    state = _state_at((1.6, 1.1))
    task = _task(model)
    frictions = np.array([0.03, 0.07, 0.05])
    rng = np.random.default_rng(0)
    push = Push(0.0, 1.0)
    next_state = model.sample_next_state(state, push, frictions, rng)
    assert next_state.outcome == "hole" and next_state.ended
    cost = model.get_cost(task, state, push, next_state, frictions)
    assert cost >= PuckRewards().hole_penalty
    after = model.sample_next_state(next_state, NO_PUSH, frictions, rng)
    assert after is next_state
    assert model.get_cost(task, next_state, NO_PUSH, after, frictions) == 0.0
    assert not model.check_goal(task, after, frictions)


def test_the_goal_ends_the_task(model):
    """A puck that stops in the state's goal completes the task, and the same push
    toward the map's goal does not end a task whose goal is elsewhere."""
    state = _state_at((3.9, 1.8))
    task = _task(model)
    speed = math.sqrt(2 * FRICTIONS[1] * 9.81 * 0.1)  # stops 0.1 m past release
    push = Push(math.pi / 2, speed)
    next_state = model.sample_next_state(
        state, push, FRICTIONS, np.random.default_rng(0)
    )
    assert next_state.outcome == "goal" and model.check_goal(
        task, next_state, FRICTIONS
    )
    assert model.get_cost(task, state, push, next_state, FRICTIONS) < 0
    elsewhere = replace(state, goal=(1.2, 3.4))
    next_state = model.sample_next_state(
        elsewhere, push, FRICTIONS, np.random.default_rng(0)
    )
    assert not next_state.ended and next_state.goal == elsewhere.goal
    record = next_state.last_push
    assert record is not None and not record.goal_reached
    # The state's own goal ends it.
    in_goal = replace(state, goal=(3.9, 2.0))
    next_state = model.sample_next_state(
        in_goal, push, FRICTIONS, np.random.default_rng(0)
    )
    assert next_state.outcome == "goal"


def test_an_infeasible_push_costs_a_push(model):
    """A push the robot cannot make leaves the puck and the robot."""
    state = _state_at((0.55, 2.2))
    push = Push(0.0, 1.0)
    next_state = model.sample_next_state(
        state, push, FRICTIONS, np.random.default_rng(0)
    )
    assert next_state.last_push is not None and not next_state.last_push.feasible
    assert next_state.robot == state.robot and next_state.pushes == 1
    assert model.get_cost(_task(model), state, push, next_state, FRICTIONS) == 1.0


def test_probes_run_in_their_own_simulator():
    """A query never builds or touches the episode's simulator."""
    model = FrictionModel()
    query = FrictionQuery(0, (0.95, 2.85), math.radians(20), 0.8)
    record = model.sample_response(query, FRICTIONS, np.random.default_rng(0))
    assert record.feasible and record.puck == query.puck
    assert model._sim is None  # pylint: disable=protected-access
    model.close()


def test_a_layout_brings_its_holes():
    """A push in a state with a layout slides through the layout's holes, which
    the map without it lacks, and the next state keeps the layout."""
    model = FrictionModel(num_holes=0, goal_radius=0.2, num_floors=4, maze="corridor1")
    frictions = np.full(4, 0.05)
    push = Push(0.0, 1.0)
    outcomes = {}
    for layout in (None, "o:1,12|h:2.80,1.40"):
        state = model.get_initial_state((1.6, 1.4), (0.8, 1.4), (5.4, 2.2), layout)
        next_state = model.sample_next_state(
            state, push, frictions, np.random.default_rng(0)
        )
        assert next_state.layout == layout
        outcomes[layout] = next_state.outcome
    model.close()
    assert outcomes == {None: None, "o:1,12|h:2.80,1.40": "hole"}
