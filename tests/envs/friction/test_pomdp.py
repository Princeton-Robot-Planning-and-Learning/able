"""Tests for envs/friction/pomdp.py."""

import math
from dataclasses import replace

import numpy as np
import pytest

from able.envs.friction.pomdp import GOAL, HOLE, ONGOING, PuckPushingModel
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


def test_transitions_shed_pucks_from_walls():
    """A push that stops the puck inside the slot leaves it where it can be pushed
    on."""
    model = _model()
    push = Push(0.0, 0.9)
    nxt = model.transition(_start(model), push, push.angle, push.speed)
    assert nxt.outcome == ONGOING and nxt.puck[0] < 2.0
    assert np.allclose(model.world.shed(nxt.puck_array), nxt.puck, atol=1e-4)


def test_leaf_values_follow_the_path_to_the_goal():
    """A leaf is worth the goal reward minus a push cost per push_length of the
    pushable path to the goal, or the stuck penalty without a path or with too few
    pushes left."""
    model = _model()
    rewards = model.rewards
    start = _start(model)
    distance = model.distance_to_goal(start.puck_array)
    assert distance > model.push_length
    expected = rewards.goal_reward - rewards.push_cost * distance / model.push_length
    assert model.value(start) == pytest.approx(expected)
    below_goal = model.initial_state(
        np.array([3.9, 1.8]), np.array([3.9, 1.2]), FRICTIONS
    )
    assert model.value(below_goal) == rewards.goal_reward - rewards.push_cost
    # The robot cannot push from inside the slot, so no path starts there.
    in_slot = model.initial_state(np.array([2.6, 2.2]), np.array([1.2, 2.2]), FRICTIONS)
    assert model.value(in_slot) == -rewards.stuck_penalty
    last_push = replace(start, pushes=model.max_pushes - 1)
    assert model.value(last_push) == -rewards.stuck_penalty
