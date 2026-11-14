"""Tests for envs/limb/constraints.py."""

import numpy as np
import pytest
from pybullet_helpers.geometry import SE2Pose

from able.envs.limb.constraints import (
    HumanTorqueLimits,
    are_within_limits,
    base_in_collision,
    exceeds_robot_torque_limits,
    get_human_collision_ids,
    get_resting_penetration,
    get_robot_motion_violation,
    limb_in_collision,
)
from able.envs.limb.human import get_variant
from able.envs.limb.sim import HumanTorques, LimbSim


@pytest.fixture(name="sim", scope="module")
def _sim():
    # The bed scene has the torso and furniture, and its default base is clear.
    sim = LimbSim(get_variant("bed-left-arm"))
    yield sim
    sim.close()


def test_torque_limits(sim):
    """Tests the bounds on the person's and the robot's torques."""
    limits = HumanTorqueLimits.for_limb("human-left-arm")
    assert limits == HumanTorqueLimits(50.0, 25.0)
    assert HumanTorqueLimits.for_limb("human-right-leg").total == 100.0
    static, none = np.full(6, 10.0), np.zeros(6)
    assert not limits.are_exceeded(HumanTorques(static, none, static, none))
    assert limits.are_exceeded(HumanTorques(static + 30.0, none, static, none))
    assert limits.are_exceeded(HumanTorques(np.full(6, 60.0), none, static, none))
    # Holding the limb still against the person's walls loads their joints.
    walls = np.full(6, -30.0)
    assert limits.are_exceeded(HumanTorques(static, none, static, walls))

    _, upper = sim.robot_torque_limits
    assert not exceeds_robot_torque_limits(sim, upper)
    assert exceeds_robot_torque_limits(sim, upper + 1.0)


def test_robot_motion_violation(sim):
    """Tests that the robot's joint limits and speed limit are checked."""
    start = sim.capture()
    arm = sim.env.robot.arm
    assert get_robot_motion_violation(sim, max_velocity=2.0) == ""
    velocities = np.zeros(len(start.robot_velocities))
    velocities[0] = 3.0
    arm.set_joints(list(start.robot_positions), joint_velocities=list(velocities))
    assert get_robot_motion_violation(sim, max_velocity=2.0) == "robot velocity limit"
    positions = np.array(start.robot_positions)
    positions[1] = arm.joint_upper_limits[1] + 0.5
    arm.set_joints(list(positions))
    assert get_robot_motion_violation(sim, max_velocity=2.0) == "robot joint limits"
    sim.restore(start)


def test_collisions(sim):
    """Tests that resting contact is allowed and deeper contact is not."""
    start = sim.capture()
    assert get_human_collision_ids(sim)
    resting = get_resting_penetration(sim)
    assert all(depth <= 0.0 for depth in resting.values())
    assert not limb_in_collision(sim, resting)
    assert not base_in_collision(sim)

    # Swing the shoulder until the arm passes through the torso.
    positions = np.array(start.limb_positions)
    hits = []
    for angle in np.linspace(-np.pi, np.pi, 25):
        positions[1] = angle
        sim.env.limb.set_joints(list(positions))
        hits.append(limb_in_collision(sim, resting))
    assert any(hits)
    sim.restore(start)


def test_are_within_limits():
    """Tests the joint limit check."""
    limits = (np.zeros(2), np.ones(2))
    assert are_within_limits(np.array([0.0, 1.0]), limits)
    assert not are_within_limits(np.array([0.5, 1.1]), limits)


def test_base_collision(sim):
    """Tests that a base placed on the person collides."""
    start = sim.capture()
    torso_position = sim.env.limb.get_base_pose().position
    sim.env.robot.set_base(SE2Pose(torso_position[0], torso_position[1], 0.0))
    assert base_in_collision(sim)
    sim.restore(start)
