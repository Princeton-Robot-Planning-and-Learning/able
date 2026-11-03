"""Tests for envs/human_reach/model.py."""

import numpy as np
import pytest

from able.envs.human_reach.model import ReachModel, ReachTask
from able.envs.human_reach.problem import plan_bedside
from able.envs.human_reach.query import ReachQuery, bedside_directions
from able.envs.human_reach.scene import ObjectCentricBedsideStocking3DEnv

THETA = np.array([0.7, 0.6, 0.65])


@pytest.fixture(name="model", scope="module")
def _model():
    model = ReachModel(num_requested=2, num_clutter=0)
    yield model
    model.close()


def _task(model: ReachModel, seed: int = 0) -> ReachTask:
    state = model.get_initial_state(seed)
    return ReachTask(state, tuple(state.scene.requested_objects))


def test_an_empty_plan_delivers_nothing(model):
    """Without a plan, the items stay out of reach on the far furniture, and the
    state stays fixed."""
    task = _task(model)
    state = task.initial_state
    assert model.horizon == 1 and not state.executed
    assert model.get_terminal_cost(task, state, THETA) == 2.0
    rng = np.random.default_rng(0)
    after = model.sample_next_state(state, (), THETA, rng)
    assert after.executed and after.num_unreached == 2
    assert [(name, reached) for name, _, reached in after.deliveries] == [
        ("req0", False),
        ("req1", False),
    ]
    assert not model.check_goal(task, after, THETA)
    assert model.get_cost(task, state, (), after, THETA) == 0.0
    assert model.sample_next_state(after, (), THETA, rng) is after


def test_a_plan_delivers_the_items(model, tmp_path, monkeypatch):
    """A plan found with the true reach delivers both items within it."""
    monkeypatch.chdir(tmp_path)
    task = _task(model)
    planner_sim = ObjectCentricBedsideStocking3DEnv(2, 0, allow_state_access=True)
    planner_sim.reset(options={"init_state": task.initial_state.scene})
    plan = plan_bedside(
        planner_sim, task.initial_state.scene, theta_particles=THETA[None], seed=0
    )
    planner_sim.close()
    assert plan is not None
    after = model.sample_next_state(
        task.initial_state, tuple(plan), THETA, np.random.default_rng(0)
    )
    assert [(name, reached) for name, _, reached in after.deliveries] == [
        ("req0", True),
        ("req1", True),
    ]
    assert model.check_goal(task, after, THETA)
    assert model.get_terminal_cost(task, after, THETA) == 0.0


def test_queries_run_in_their_own_simulator(model):
    """A query never touches the model's simulator."""
    sim = model.sim
    base = sim.robot.get_base()
    d = bedside_directions(model.config, sim.shoulder)[0]
    response = model.sample_response(
        ReachQuery(d, (0.6,)), THETA, np.random.default_rng(0)
    )
    assert len(response) == 1 and sim.robot.get_base() == base
