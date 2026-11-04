"""Tests for envs/human_reach/model.py."""

import numpy as np
import pytest

from able.envs.human_reach.model import NO_ACTION, ReachModel, ReachTask
from able.envs.human_reach.query import ReachQuery, bedside_directions
from able.envs.human_reach.scene import ObjectCentricBedsideStocking3DEnv
from able.planners.pddlstream import plan_bedside

THETA = np.array([0.7, 0.6, 0.65])


@pytest.fixture(name="model", scope="module")
def _model():
    model = ReachModel(num_requested=2, num_clutter=0)
    yield model
    model.close()


def _task(model: ReachModel, seed: int = 0) -> ReachTask:
    state = model.get_initial_state(seed)
    return ReachTask(state, tuple(state.scene.requested_objects))


def test_no_action_changes_nothing(model):
    """NO_ACTION leaves the state, which has nothing delivered yet."""
    task = _task(model)
    state = task.initial_state
    assert model.horizon == 40 and state.num_actions == 0
    assert model.get_terminal_cost(task, state, THETA) == 2.0
    after = model.sample_next_state(state, NO_ACTION, THETA, np.random.default_rng(0))
    assert after is state and not model.check_goal(task, after, THETA)
    assert model.get_cost(task, state, NO_ACTION, after, THETA) == 0.0


def test_actions_deliver_the_items(model, tmp_path, monkeypatch):
    """A plan found with the true reach, executed one action at a time, delivers
    both items within reach; a fresh simulator continues a state with a held
    item."""
    monkeypatch.chdir(tmp_path)
    task = _task(model)
    planner_sim = ObjectCentricBedsideStocking3DEnv(2, 0, allow_state_access=True)
    planner_sim.reset(options={"init_state": task.initial_state.scene})
    plan = plan_bedside(
        planner_sim, task.initial_state.scene, theta_particles=THETA[None], seed=0
    )
    planner_sim.close()
    assert plan is not None
    rng = np.random.default_rng(0)
    state, deliveries = task.initial_state, []
    first_pick = next(i for i, (name, _) in enumerate(plan) if name == "pick")
    for i, action in enumerate(plan):
        if i == first_pick + 1:
            # Continue in a fresh simulator while holding the item.
            fresh = ReachModel(num_requested=2, num_clutter=0)
            state = fresh.sample_next_state(state, action, THETA, rng)
            fresh.close()
        else:
            state = model.sample_next_state(state, action, THETA, rng)
        if state.delivery is not None:
            deliveries.append((state.delivery[0], state.delivery[2]))
    assert state.num_actions == len(plan)
    assert sorted(deliveries) == [("req0", True), ("req1", True)]
    assert model.check_goal(task, state, THETA)
    assert model.get_terminal_cost(task, state, THETA) == 0.0


def test_queries_run_in_their_own_simulator(model):
    """A query never touches the model's simulator."""
    sim = model.sim
    base = sim.robot.get_base()
    d = bedside_directions(model.config, sim.shoulder)[0]
    response = model.sample_response(
        ReachQuery(d, (0.6,)), THETA, np.random.default_rng(0)
    )
    assert len(response) == 1 and sim.robot.get_base() == base
