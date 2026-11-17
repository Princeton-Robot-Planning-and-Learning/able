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
    """A plan found with a reach, executed one action at a time for a person whose
    reach is 30% longer, delivers both items, which the person grasps; a fresh
    simulator continues a state with a held item. A person whose reach ends at the
    last item grasps it about half the time, and the task succeeds exactly when
    they do."""
    monkeypatch.chdir(tmp_path)
    task = _task(model)
    planner_sim = ObjectCentricBedsideStocking3DEnv(2, 0, allow_state_access=True)
    planner_sim.reset(options={"init_state": task.initial_state.scene})
    plan = plan_bedside(
        planner_sim, task.initial_state.scene, theta_particles=THETA[None], seed=0
    )
    planner_sim.close()
    assert plan is not None
    rng, theta = np.random.default_rng(0), 1.3 * THETA
    state, deliveries = task.initial_state, []
    first_pick = next(i for i, (name, _) in enumerate(plan) if name == "pick")
    last_place = max(i for i, (name, _) in enumerate(plan) if name == "place")
    before_last = state
    for i, action in enumerate(plan):
        if i == last_place:
            before_last = state
        if i == first_pick + 1:
            # Continue in a fresh simulator while holding the item.
            fresh = ReachModel(num_requested=2, num_clutter=0)
            state = fresh.sample_next_state(state, action, theta, rng)
            fresh.close()
        else:
            state = model.sample_next_state(state, action, theta, rng)
        if state.delivery is not None:
            deliveries.append((state.delivery[0], state.delivery[2]))
    assert state.num_actions == len(plan)
    assert sorted(deliveries) == [("req0", True), ("req1", True)]
    assert state.grasped == {"req0", "req1"}
    assert model.check_goal(task, state, theta)
    assert model.get_terminal_cost(task, state, theta) == 0.0

    assert state.delivery is not None
    point = np.array(state.delivery[1]) - model.sim.shoulder
    edge = theta * np.linalg.norm(point / theta)
    grasps = []
    for seed in range(16):
        after = model.sample_next_state(
            before_last, plan[last_place], edge, np.random.default_rng(seed)
        )
        assert after.delivery is not None
        grasped = after.delivery[2]
        assert model.check_goal(task, after, edge) == grasped
        assert after.num_unreached == int(not grasped)
        grasps.append(grasped)
    assert 0 < sum(grasps) < len(grasps)


def test_queries_run_in_their_own_simulator(model):
    """A query never touches the model's simulator."""
    sim = model.sim
    base = sim.robot.get_base()
    d = bedside_directions(model.config, sim.shoulder)[0]
    response = model.sample_response(
        ReachQuery(d, (0.6,)), THETA, np.random.default_rng(0)
    )
    assert len(response) == 1 and sim.robot.get_base() == base
