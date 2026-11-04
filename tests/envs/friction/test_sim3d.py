"""Tests for envs/friction/sim3d.py."""

import math

import numpy as np

from able.envs.friction.layout import make_sim, make_world
from able.envs.friction.model import FrictionModel
from able.envs.friction.query import FrictionQuery
from able.envs.friction.sim3d import (
    ObjectCentricPuckPushing3DEnv,
    PuckPushing3DEnvConfig,
)
from able.envs.friction.world import PuckPhysics, Push

NOISELESS = PuckPhysics(
    speed_noise=0.0, angle_noise=0.0, speed_obs_noise=0.0, position_obs_noise=0.0
)
FRICTIONS = (0.03, 0.07, 0.15)


def test_matches_the_analytic_slide():
    """Without noise, the simulated stop matches the analytic model, bounces
    included."""
    env = ObjectCentricPuckPushing3DEnv(PuckPushing3DEnvConfig(physics=NOISELESS))
    world = env.world
    # The second push bounces off the top wall.
    for seed, (angle, speed) in enumerate(
        [(math.pi / 2, 0.6), (1.2, 1.0), (-2.0, 0.8)]
    ):
        env.reset(seed=seed, options={"frictions": FRICTIONS})
        puck = np.array(world.puck_map.puck_start)
        _, reward, terminated, _, info = env.step(np.array([angle, speed]))
        assert reward == -1.0 and info["push_feasible"] and not terminated
        predicted = world.slide(
            world.release_point(puck, angle), info["release_speed"], angle, FRICTIONS
        )
        assert np.linalg.norm(predicted.stop - info["stop_position"]) < 0.03
    env.close()


def test_robot_follows_and_render():
    """The robot ends up behind where it let go, and the scene renders."""
    env = ObjectCentricPuckPushing3DEnv(PuckPushing3DEnvConfig(physics=NOISELESS))
    env.reset(seed=0, options={"frictions": FRICTIONS})
    puck = np.array(env.world.puck_map.puck_start)
    env.step(np.array([math.pi / 2, 0.6]))
    pose = env.robot_pose
    station = env.world.push_station(puck, math.pi / 2)
    assert math.isclose(pose[2], math.pi / 2, abs_tol=1e-6)
    assert math.isclose(
        pose[1] - station[1], env.config.physics.accel_distance, abs_tol=0.02
    )
    assert np.linalg.norm(env.puck_position - puck) > 0.3
    assert env.render().ndim == 3
    env.close()


def test_paddle_clears_the_arm_and_base():
    """Held for a strike at any stance, or carried for driving, the paddle touches
    no part of the robot except the gripper holding it."""
    env = make_sim("3d", make_world())
    env.reset(seed=0)
    base = env.robot.get_base()
    heading = np.array([math.cos(base.rot), math.sin(base.rot)])
    face = np.array([base.x, base.y]) + env.config.physics.paddle_offset * heading
    for stance in np.radians([-60, -30, 0, 30, 60]):
        env._place_paddle(face, base.rot - stance)  # pylint: disable=protected-access
        assert env.paddle_robot_gap() > 0.01
    env._stow_paddle()  # pylint: disable=protected-access
    assert env.paddle_robot_gap() > 0.05
    env.close()


def test_model_and_probes_in_3d():
    """The environment model pushes and probes in 3D as in 2D."""
    model = FrictionModel(sim="3d")
    rng = np.random.default_rng(0)
    frictions = np.array(FRICTIONS)
    state = model.get_initial_state()
    next_state = model.sample_next_state(
        state, Push(math.pi / 2, 0.5, 0.0), frictions, rng
    )
    assert next_state.last_push is not None and next_state.last_push.feasible
    assert next_state.pushes == 1 and next_state.robot_pose != state.robot_pose
    query = FrictionQuery(0, (0.95, 2.85), math.radians(20), 0.8)
    record = model.sample_response(query, frictions, rng)
    assert record.feasible and not record.fell_in_hole
    assert model.world.surface_at(np.array(record.stop)) == 0
    model.close()
