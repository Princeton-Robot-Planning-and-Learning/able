"""Safety constraints on the robot and the person's limb."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_utils import NUM_ROBOT_JOINTS
from numpy.typing import NDArray
from pybullet_helpers.inverse_kinematics import check_body_collisions

from able.envs.limb.sim import HumanTorques, LimbSim

# Placeholder limits on the torque at the person's joints (N m): the total, and the
# share the robot adds over the static load. Legs carry more than arms.
HUMAN_TORQUE_LIMITS = {"arm": 50.0, "leg": 100.0}
ROBOT_INDUCED_TORQUE_LIMITS = {"arm": 25.0, "leg": 50.0}
# How close bodies may come before a clearance query stops looking (m).
CLEARANCE_PROBE_DISTANCE = 0.1
# How much deeper than at rest the limb may press into the person (m).
COLLISION_MARGIN = 0.01
# How far the limb may press into furniture or the robot's base (m).
CONTACT_PENETRATION_TOLERANCE = 0.002


@dataclass(frozen=True)
class HumanTorqueLimits:
    """Bounds on the torque at each joint of the person's limb."""

    total: float
    robot_induced: float

    @classmethod
    def for_limb(cls, limb_name: str) -> "HumanTorqueLimits":
        """The placeholder limits for an arm or a leg."""
        kind = "leg" if "leg" in limb_name else "arm"
        return cls(HUMAN_TORQUE_LIMITS[kind], ROBOT_INDUCED_TORQUE_LIMITS[kind])

    def are_exceeded(self, torques: HumanTorques) -> bool:
        """Whether a step loads the person past either bound."""
        return bool(
            np.abs(torques.total).max() > self.total
            or np.abs(torques.robot).max() > self.robot_induced
        )


def exceeds_robot_torque_limits(sim: LimbSim, commanded: NDArray[np.float64]) -> bool:
    """Whether the robot is asked for a torque outside its limits."""
    lower, upper = sim.robot_torque_limits
    commanded = np.asarray(commanded, dtype=np.float64)
    return bool(np.any(commanded < lower - 1e-9) or np.any(commanded > upper + 1e-9))


def get_robot_motion_violation(sim: LimbSim, max_velocity: float) -> str:
    """The robot's joint-limit or speed violation, or an empty string."""
    arm = sim.env.robot.arm
    if not arm.check_joint_limits(list(arm.get_joint_positions())):
        return "robot joint limits"
    velocities = np.asarray(arm.get_joint_velocities()[:NUM_ROBOT_JOINTS])
    if np.abs(velocities).max() > max_velocity:
        return "robot velocity limit"
    return ""


def are_within_limits(
    positions: NDArray[np.float64],
    limits: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> bool:
    """Whether every joint position lies within its (lower, upper) limits."""
    lower, upper = limits
    return bool(np.all(lower <= positions) and np.all(positions <= upper))


def get_human_collision_ids(sim: LimbSim) -> list[int]:
    """The person's torso and every limb except the moved one."""
    scene, limb_id = sim.env.scene, sim.env.limb.robot_id
    ids = [
        limb.robot_id
        for limb in getattr(scene, "limbs", {}).values()
        if limb.robot_id != limb_id
    ]
    torso_id = getattr(scene, "torso_id", None)
    if torso_id is not None:
        ids.append(torso_id)
    return ids


def get_scene_collision_ids(sim: LimbSim) -> list[int]:
    """The furniture, if any."""
    return list(sim.env.scene.get_scene_collision_ids())


def get_clearance(sim: LimbSim, body_a: int, body_b: int) -> float:
    """The gap between two bodies (m), negative if they overlap, capped at the
    probe distance."""
    points = p.getClosestPoints(
        body_a, body_b, distance=CLEARANCE_PROBE_DISTANCE, physicsClientId=sim.client
    )
    if not points:
        return CLEARANCE_PROBE_DISTANCE
    return float(min(point[8] for point in points))


def get_resting_penetration(sim: LimbSim) -> dict[int, float]:
    """How far the limb overlaps each part of the person, in the current state.

    Call this in a task's start state: the limb may rest against the torso, and
    only deeper overlap counts as a collision.
    """
    limb_id = sim.env.limb.robot_id
    return {
        body_id: min(0.0, get_clearance(sim, limb_id, body_id))
        for body_id in get_human_collision_ids(sim)
    }


def limb_in_collision(sim: LimbSim, resting_penetration: Mapping[int, float]) -> bool:
    """Whether the limb presses into the person deeper than at rest, or into the
    furniture or the robot's base."""
    limb_id = sim.env.limb.robot_id
    for body_id in get_human_collision_ids(sim):
        resting = resting_penetration.get(body_id, 0.0)
        if get_clearance(sim, limb_id, body_id) < resting - COLLISION_MARGIN:
            return True
    for body_id in get_scene_collision_ids(sim) + [sim.env.robot.base.robot_id]:
        if get_clearance(sim, limb_id, body_id) < -CONTACT_PENETRATION_TOLERANCE:
            return True
    return False


def arm_in_collision(sim: LimbSim) -> bool:
    """Whether the robot's arm overlaps the person (except the moved limb) or the
    furniture."""
    arm_id = sim.env.robot.arm.robot_id
    obstacles = get_human_collision_ids(sim) + get_scene_collision_ids(sim)
    return any(check_body_collisions(arm_id, o, sim.client) for o in obstacles)


def base_in_collision(sim: LimbSim) -> bool:
    """Whether the robot's base overlaps the person, the moved limb included, or the
    furniture."""
    base_id = sim.env.robot.base.robot_id
    obstacles = (
        [sim.env.limb.robot_id]
        + get_human_collision_ids(sim)
        + get_scene_collision_ids(sim)
    )
    return any(check_body_collisions(base_id, o, sim.client) for o in obstacles)
