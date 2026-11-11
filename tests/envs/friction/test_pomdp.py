"""Tests for envs/friction/pomdp.py."""

import math
from dataclasses import replace

import numpy as np

from able.envs.friction.pomdp import GOAL, HOLE, ONGOING, STUCK, PuckPushingModel
from able.envs.friction.world import THREE_ROUTES, PuckWorld, Push

FRICTIONS = (0.03, 0.07, 0.15)


def _model() -> PuckPushingModel:
    return PuckPushingModel(PuckWorld(THREE_ROUTES))


def _start(model: PuckPushingModel):
    return model.initial_state(
        np.array(THREE_ROUTES.puck_start), np.array(THREE_ROUTES.robot_start), FRICTIONS
    )


def test_transition_rewards_and_outcomes():
    """Pushes cost a push plus driving; the goal pays and a hole costs."""
    model = _model()
    world, rewards = model.world, model.rewards
    start = _start(model)
    push = Push(math.pi / 2, 0.5)
    nxt = model.transition(start, push, push.angle, push.speed)
    expected = world.slide(
        world.release_point(start.puck_array, push.angle), 0.5, push.angle, FRICTIONS
    )
    assert np.allclose(nxt.puck, expected.stop, atol=1e-3)
    assert nxt.outcome == ONGOING and nxt.pushes == 1
    assert nxt.last_reward < -rewards.push_cost  # it also drove
    # Into the green hole on a slippery green floor.
    near_hole = model.initial_state(
        np.array([1.6, 1.1]), np.array([1.0, 1.1]), (0.03, 0.07, 0.05)
    )
    fell = model.transition(near_hole, Push(0.0, 1.0), 0.0, 1.0)
    assert fell.outcome == HOLE and model.is_terminal(fell)
    assert fell.last_reward < -rewards.hole_penalty
    # Up into the goal from just below it.
    below_goal = model.initial_state(
        np.array([3.9, 1.8]), np.array([3.9, 1.2]), FRICTIONS
    )
    speed = math.sqrt(2 * FRICTIONS[1] * 9.81 * 0.1)
    won = model.transition(below_goal, Push(math.pi / 2, speed), math.pi / 2, speed)
    assert won.outcome == GOAL and won.last_reward > 0


def test_observation_likelihood_prefers_the_true_state():
    """The measurement density peaks at the state that produced it."""
    model = _model()
    rng = np.random.default_rng(0)
    start = _start(model)
    push = Push(math.pi / 2, 0.6)
    nxt = model.transition(start, push, push.angle, push.speed)
    other = model.transition(start, push, push.angle, push.speed * 1.3)
    obs = model.observe(nxt, push, rng)
    assert model.likelihood(obs, nxt, push) > model.likelihood(obs, other, push)
    assert model.likelihood((*obs[:3], True), nxt, push) == 0.0


def test_proposals_are_feasible():
    """The default proposer only returns pushes the robot can make."""
    model = _model()
    rng = np.random.default_rng(0)
    state = _start(model)
    limit = model.world.physics.max_stance
    for mode in ("grid", "continuous", "auto"):
        model.stance_mode = mode
        for _ in range(20):
            push = model.propose(state, [], rng)
            assert push is not None
            assert model.world.push_feasible(state.puck_array, push.angle, push.stance)
            if mode == "auto":
                assert push.stance is None
            else:
                assert push.stance is not None and abs(push.stance) <= limit + 1e-9
            if mode == "grid":
                assert np.isclose(model.stances, push.stance, atol=1e-4).any()


def test_stuck_stops_end_the_episode():
    """A puck in the slot or against a wall, where the robot cannot get behind it,
    is stuck: the episode ends at the stuck penalty. A push into the wall left of
    the slot is sometimes measured against the wall, and is then stuck too; a push
    across open floor never is."""
    model = _model()
    rng = np.random.default_rng(0)
    assert model.is_stuck(np.array([2.6, 2.2]))  # in the slot
    assert model.is_stuck(np.array([1.88, 2.6]))  # against the wall
    assert not model.is_stuck(np.array(THREE_ROUTES.puck_start))
    # Into the slot, whose blue floor is too rough to cross.
    sticky = model.initial_state(
        np.array([1.5, 2.2]), np.array([1.2, 1.0]), (0.15, 0.07, 0.15)
    )
    inside = model.transition(sticky, Push(0.0, 0.8), 0.0, 0.8)
    assert inside.outcome == STUCK and model.is_terminal(inside)
    assert inside.last_reward < -model.rewards.stuck_penalty
    assert model.value(inside) == 0.0
    near_wall = model.initial_state(
        np.array([1.2, 2.6]), np.array([1.2, 1.0]), FRICTIONS
    )
    into_wall = [model.step(near_wall, Push(0.0, 0.8), rng)[0] for _ in range(300)]
    num_stuck = sum(s.outcome == STUCK for s in into_wall)
    assert 0 < num_stuck < len(into_wall)
    start = _start(model)
    across = [model.step(start, Push(math.pi / 2, 0.3), rng)[0] for _ in range(100)]
    assert all(s.outcome == ONGOING for s in across)


def test_running_out_of_pushes_fails():
    """The last push costs the stuck penalty unless it ends the episode."""
    model = _model()
    last = replace(_start(model), pushes=model.max_pushes - 1)
    push = Push(math.pi / 2, 0.3)
    ran_out = model.transition(last, push, push.angle, push.speed)
    assert ran_out.outcome == ONGOING and model.is_terminal(ran_out)
    assert ran_out.last_reward < -model.rewards.stuck_penalty


def test_proposals_sweep_headings_before_giving_up():
    """Where only a few headings are feasible, the sweep finds one after the
    random tries fail; where none is, there is no push."""
    model = _model()
    rng = np.random.default_rng(0)
    model.max_proposal_tries = 0
    narrow = model.initial_state(np.array([1.899, 2.28]), np.zeros(2), FRICTIONS)
    push = model.propose(narrow, [], rng)
    assert push is not None
    assert model.world.push_feasible(narrow.puck_array, push.angle, push.stance)
    in_slot = model.initial_state(np.array([2.6, 2.2]), np.zeros(2), FRICTIONS)
    assert model.propose(in_slot, [], rng) is None


def test_greedy_rollout_reaches_the_goal_with_known_frictions():
    """With the true frictions, the rollout policy gets the puck to the goal."""
    model = _model()
    model.rollout_frictions = FRICTIONS
    state = _start(model)
    assert model.value(state) > 0
    path = model.path_to_goal(state.puck_array)
    assert np.allclose(path[-1], model.world.goal)
    # The rollout paths avoid the slot, which the robot cannot push from.
    assert not any(2.1 < p[0] < 3.1 and 2.0 < p[1] < 2.4 for p in path)
