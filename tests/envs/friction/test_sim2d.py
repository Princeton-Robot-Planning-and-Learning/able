"""Tests for envs/friction/sim2d.py."""

import math

import numpy as np

from able.envs.friction.layout import make_sim, make_world
from able.envs.friction.sim2d import (
    DynPuckPushing2DEnvConfig,
    ObjectCentricDynPuckPushing2DEnv,
    PuckType,
    parse_push_action,
)
from able.envs.friction.world import PuckPhysics

NOISELESS = PuckPhysics(
    speed_noise=0.0, angle_noise=0.0, speed_obs_noise=0.0, position_obs_noise=0.0
)
FRICTIONS = (0.03, 0.07, 0.15)


def _env_with_puck_at(puck_xy, frictions=FRICTIONS):
    env = ObjectCentricDynPuckPushing2DEnv(DynPuckPushing2DEnvConfig(physics=NOISELESS))
    state, _ = env.reset(seed=0, options={"frictions": frictions})
    puck = state.get_objects(PuckType)[0]
    state.set(puck, "x", puck_xy[0])
    state.set(puck, "y", puck_xy[1])
    env.reset(options={"init_state": state})
    return env


def test_actions_and_steps():
    """Actions are (angle, speed, stance), and two numbers let the robot pick the
    stance."""
    env = make_sim("2d", make_world())
    env.reset(seed=0)
    assert env.action_space.shape == (3,)
    _, reward, terminated, truncated, info = env.step(np.array([math.pi / 2, 0.5]))
    assert reward == -1.0 and not truncated
    assert info["push_feasible"] and not terminated
    _, _, _, _, info = env.step(np.array([math.pi / 2, 0.5, math.radians(30)]))
    assert info["push_feasible"]
    push = parse_push_action(np.array([0.1, 0.5]), math.radians(60))
    assert push.stance is None


def test_frictions_are_hidden_and_resampled():
    """Each reset draws new frictions unless given."""
    env = ObjectCentricDynPuckPushing2DEnv()
    env.reset(seed=0)
    first = env.frictions
    env.reset(seed=1)
    assert env.frictions != first
    env.reset(seed=2, options={"frictions": FRICTIONS})
    assert env.frictions == FRICTIONS


def test_matches_the_analytic_slide():
    """Without noise, the simulated stop matches the analytic model."""
    env = ObjectCentricDynPuckPushing2DEnv(DynPuckPushing2DEnvConfig(physics=NOISELESS))
    world = env.world
    for seed, (angle, speed) in enumerate(
        [(math.pi / 2, 0.6), (1.2, 1.0), (-2.0, 0.8)]
    ):
        env.reset(seed=seed, options={"frictions": FRICTIONS})
        puck = np.array(world.puck_map.puck_start)
        assert world.push_feasible(puck, angle)
        _, _, _, _, info = env.step(np.array([angle, speed]))
        predicted = world.slide(
            world.release_point(puck, angle), info["release_speed"], angle, FRICTIONS
        )
        assert np.linalg.norm(predicted.stop - info["stop_position"]) < 0.03


def test_hole_fails_the_episode():
    """Sliding over a red circle ends the episode."""
    # Along the green corridor, past the hole at (2.6, 1.0), on a slippery green floor.
    env = _env_with_puck_at((1.6, 1.1), frictions=(0.03, 0.07, 0.05))
    _, _, terminated, _, info = env.step(np.array([0.0, 1.0]))
    assert info["push_feasible"] and info["fell_in_hole"]
    assert terminated and not info["goal_reached"]


def test_goal_ends_the_episode():
    """A puck that stops in the goal circle succeeds."""
    env = _env_with_puck_at((3.9, 1.8))
    speed = math.sqrt(2 * FRICTIONS[1] * 9.81 * 0.1)  # stops 0.1 m past release
    _, _, terminated, _, info = env.step(np.array([math.pi / 2, speed]))
    assert terminated and info["goal_reached"] and not info["fell_in_hole"]


def test_infeasible_push_leaves_the_puck():
    """A push the robot cannot make does nothing."""
    env = _env_with_puck_at((0.55, 2.2))
    _, _, terminated, _, info = env.step(np.array([0.0, 1.0]))
    assert not info["push_feasible"] and not terminated
    assert np.allclose(info["stop_position"], (0.55, 2.2))
    assert env.render().ndim == 3


def test_reset_to_a_puck_and_robot_pose():
    """Reset can place the puck and the robot, as probes and the model need."""
    env = make_sim("2d", make_world(num_holes=0))
    puck, robot = (2.0, 3.2), (1.2, 3.2, 0.0)
    state, _ = env.reset(
        seed=0, options={"frictions": FRICTIONS, "puck": puck, "robot": robot}
    )
    puck_obj = state.get_objects(PuckType)[0]
    assert np.allclose([state.get(puck_obj, "x"), state.get(puck_obj, "y")], puck)
    assert env.frictions == FRICTIONS
