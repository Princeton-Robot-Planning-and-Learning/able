"""Queries in the human reach environment: the robot holds its gripper at points
along a direction from the person's shoulder, and the person answers whether they
can reach each one. Queries run in a simulator of their own, an empty copy of the
scene."""

from dataclasses import dataclass

import numpy as np
from kinder.envs.kinematic3d.utils import remove_fingers_from_extended_joints
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot

from able.envs.human_reach.execute import Driver
from able.envs.human_reach.scene import (
    BedsideStocking3DEnvConfig,
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
)
from able.envs.human_reach.streams import (
    CARRY_JOINTS,
    GRASP_ORIENTATIONS,
    HOME_JOINTS,
    NOMINAL_HALF_HEIGHT,
    _base_candidates,
)

# Margin scale of P(reached | theta) = sigmoid(margin / scale), for queries and
# deliveries alike.
REACH_SCALE = 0.03
# The answers to a query: each presented gripper point and whether it was reached.
ReachResponse = tuple[tuple[tuple[float, float, float], bool], ...]


@dataclass(frozen=True, eq=False)
class ReachQuery:
    """Present the gripper at shoulder + s * direction for each s in `distances`,
    going outward."""

    direction: NDArray[np.float64]  # unit vector from the shoulder
    distances: tuple[float, ...]  # m


def reach_distance(direction: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """The distance to the reach ellipsoid along unit `direction`: 1 / ||d / theta||."""
    ratio = np.asarray(direction, float) / np.asarray(theta, float)
    return 1.0 / np.linalg.norm(ratio, axis=-1)


def bedside_directions(
    config: BedsideStocking3DEnvConfig, shoulder: NDArray
) -> NDArray[np.float64]:
    """Unit vectors from the shoulder to the top of a nominal item on every bedside
    slot."""
    tops = np.concatenate([config.region_slots(r) for r in config.bedside_regions])
    d = tops + [0.0, 0.0, 2 * NOMINAL_HALF_HEIGHT] - np.asarray(shoulder)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


PRESENT_TOL = 0.01  # m, the gripper must end this close to the planned point


def present_conf(
    sim: ObjectCentricBedsideStocking3DEnv, point: NDArray
) -> tuple[SE2Pose, list] | None:
    """A collision-free base pose and arm conf with the gripper at `point`, else
    None. Restores the robot."""
    # pylint: disable=protected-access
    arm, static, cid = sim.robot.arm, sim.static_collision_ids, sim.physics_client_id
    assert isinstance(arm, FingeredSingleArmPyBulletRobot)
    base0, joints0 = sim.robot.get_base(), list(arm.get_joint_positions())
    finger = arm.get_finger_state()
    found = None
    for base in _base_candidates(point[0], point[1]):
        sim.robot.set_base(base)
        if any(check_body_collisions(sim.robot.base.robot_id, b, cid) for b in static):
            continue
        for quat in GRASP_ORIENTATIONS:
            arm.set_joints(HOME_JOINTS)
            try:
                q = inverse_kinematics(
                    arm, Pose(tuple(point), quat), validate=True, set_joints=True
                )
            except InverseKinematicsError:
                continue
            if not any(check_body_collisions(arm.robot_id, b, cid) for b in static):
                found = (base, list(q))
                break
        if found:
            break
    sim._set_robot_and_held_object(
        base0, remove_fingers_from_extended_joints(joints0), finger
    )
    return found


class QuerySimulator:
    """Presents queries in a simulator of its own, built lazily.

    With noise, the person answers yes with probability sigmoid(margin / scale);
    without, exactly when the margin is nonnegative. Presenting stops after the
    first no, if stop_at_miss.
    """

    def __init__(
        self,
        config: BedsideStocking3DEnvConfig = BedsideStocking3DEnvConfig(),
        scale: float = REACH_SCALE,
        noisy: bool = True,
        stop_at_miss: bool = True,
        seed: int = 0,
    ) -> None:
        self.config, self.scale, self.noisy = config, scale, noisy
        self.stop_at_miss, self.seed = stop_at_miss, seed
        self._sim: ObjectCentricBedsideStocking3DEnv | None = None

    @property
    def sim(self) -> ObjectCentricBedsideStocking3DEnv:
        """The query simulator."""
        if self._sim is None:
            self._sim = ObjectCentricBedsideStocking3DEnv(
                0, 0, config=self.config, allow_state_access=True
            )
            self._sim.reset(seed=self.seed)
        return self._sim

    def can_present(self, point: NDArray) -> bool:
        """Static collision-free IK with the gripper at `point`."""
        return present_conf(self.sim, point) is not None

    def close(self) -> None:
        """Close the simulator, if built."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def ask(
        self,
        theta: NDArray[np.float64],
        query: ReachQuery,
        rng: np.random.Generator,
    ) -> ReachResponse:
        """Each point along the query's direction that the robot could present,
        with whether the person reached it."""
        sim = self.sim
        sim.reset(seed=self.seed)
        sim.reach_theta = np.array(theta, float)
        drive = Driver(sim, self.seed)
        outcomes, base = [], None
        for s in query.distances:
            target = sim.shoulder + s * np.asarray(query.direction, float)
            conf = present_conf(sim, target)
            if conf is None:
                continue
            if conf[0] != base:
                drive.arm(CARRY_JOINTS)
                drive.base(conf[0])
                base = conf[0]
            drive.arm(conf[1])
            point = np.array(sim.robot.arm.get_end_effector_pose().position)
            if np.linalg.norm(point - target) > PRESENT_TOL:
                continue  # blocked on the way
            m = float(reach_margin(point, sim.reach_theta, sim.shoulder))
            if self.noisy:
                reached = bool(rng.random() < 1 / (1 + np.exp(-m / self.scale)))
            else:
                reached = m >= 0
            xyz = (float(point[0]), float(point[1]), float(point[2]))
            outcomes.append((xyz, reached))
            if self.stop_at_miss and not reached:
                break
        return tuple(outcomes)
