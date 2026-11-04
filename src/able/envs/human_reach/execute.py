"""Executing a human reach plan in the simulator: arm paths refined with BiRRT
and a straight-line descent, and dense base paths."""

from typing import Any

import numpy as np
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
    remove_fingers_from_extended_joints,
)
from numpy.typing import NDArray
from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.joint import get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    smoothly_follow_end_effector_path,
)

from able.envs.human_reach.scene import ObjectCentricBedsideStocking3DEnv
from able.envs.human_reach.streams import CARRY_JOINTS


def _clip(
    sim: ObjectCentricBedsideStocking3DEnv, action: list[float]
) -> NDArray[np.float32]:
    space = sim.action_space
    assert isinstance(space, Kinematic3DRobotActionSpace)
    return np.clip(np.asarray(action, dtype=np.float32), space.low, space.high)


def _arm_path(sim, target, exclude: set[int], seed: int) -> list:
    """BiRRT from the current arm conf to `target`; straight line if that fails."""
    # pylint: disable=protected-access
    arm = sim.robot.arm
    base, cur = sim.robot.get_base(), list(arm.get_joint_positions())
    if (
        np.abs(
            get_jointwise_difference(
                arm.get_arm_joint_infos()[:7], list(target[:7]), cur[:7]
            )
        ).max()
        < 1e-6
    ):
        return [cur]
    finger = arm.get_finger_state()
    held = sim._grasped_object_id
    bodies = (
        sim._get_collision_object_ids()
        - exclude
        - ({held} if held is not None else set())
    )
    path = run_motion_planning(
        arm,
        cur,
        list(target),
        bodies,
        seed=seed,
        physics_client_id=sim.physics_client_id,
        held_object=held,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    sim._set_robot_and_held_object(
        base, remove_fingers_from_extended_joints(cur), finger
    )
    if path is None:
        path = [cur, list(target)]
    return remap_joint_position_plan_to_constant_distance(
        path, arm, max_distance=sim.config.max_action_mag / 2
    )


def _descent(sim, hover, grasp, target: int, seed: int) -> list:
    """Straight-line end-effector path from the hover conf down to the grasp conf."""
    # pylint: disable=protected-access
    arm = sim.robot.arm
    base, cur = sim.robot.get_base(), list(arm.get_joint_positions())
    finger = arm.get_finger_state()
    held = sim._grasped_object_id
    bodies = (
        sim._get_collision_object_ids()
        - {target}
        - ({held} if held is not None else set())
    )
    waypoints = [
        arm.forward_kinematics(list(hover)),
        arm.forward_kinematics(list(grasp)),
    ]
    try:
        path = smoothly_follow_end_effector_path(
            arm,
            waypoints,
            initial_joints=list(hover),
            collision_ids=bodies,
            joint_distance_fn=create_joint_distance_fn(arm),
            seed=seed,
            include_start=False,
            held_object=held,
            base_link_to_held_obj=sim._grasped_object_transform,
        )
    except InverseKinematicsError:
        path = [list(grasp)]
    sim._set_robot_and_held_object(
        base, remove_fingers_from_extended_joints(cur), finger
    )
    return remap_joint_position_plan_to_constant_distance(
        [list(hover)] + list(path), arm, max_distance=0.02
    )[1:]


BASE_RESOLUTION = 0.05


def _base_path(sim, goal: SE2Pose) -> list[SE2Pose]:
    """Dense collision-free base waypoints to `goal`; tries both turn directions
    (a ~pi turn's is float noise)."""
    # pylint: disable=protected-access
    start = sim.robot.get_base()
    joints = remove_fingers_from_extended_joints(sim.robot.arm.get_joint_positions())
    finger = sim.robot.arm.get_finger_state()
    d = goal - start
    turns = (
        [d.rot] if abs(d.rot) < 1e-6 else [d.rot, d.rot - np.sign(d.rot) * 2 * np.pi]
    )
    shift = np.array([d.x, d.y, 0.0])
    candidates = []
    for t in turns:
        turn = np.array([0.0, 0.0, t])
        candidates += [[shift + turn], [turn, shift], [shift, turn]]
    paths = []
    for segments in candidates:
        path, at = [], np.zeros(3)
        for seg in segments:
            n = max(1, int(np.ceil(np.abs(seg).max() / BASE_RESOLUTION)))
            path += [at + seg * (i / n) for i in range(1, n + 1)]
            at = at + seg
        paths.append(
            [
                SE2Pose(
                    start.x + x,
                    start.y + y,
                    (start.rot + r + np.pi) % (2 * np.pi) - np.pi,
                )
                for x, y, r in path
            ]
        )
    found = paths[0]
    for path in paths:
        ok = True
        for q in path:
            sim._set_robot_and_held_object(q, joints, finger)
            if sim._robot_or_held_object_collision_exists():
                ok = False
                break
        if ok:
            found = path
            break
    sim._set_robot_and_held_object(start, joints, finger)
    return found


class Driver:
    """Steps the simulator toward base and joint targets."""

    def __init__(self, sim: ObjectCentricBedsideStocking3DEnv, seed: int = 0) -> None:
        self.sim, self.seed = sim, seed

    def step(self, action: list[float]) -> None:
        """One clipped simulator step."""
        self.sim.step(_clip(self.sim, action))

    def joints(
        self, target: list[float], tol: float = 1e-3, attempts: int = 40
    ) -> None:
        """Step the arm toward the joint target."""
        arm = self.sim.robot.arm
        for _ in range(attempts):
            cur = remove_fingers_from_extended_joints(arm.get_joint_positions())
            d = get_jointwise_difference(
                arm.get_arm_joint_infos()[:7], list(target[:7]), cur
            )
            self.step([0.0, 0.0, 0.0] + d + [0.0])
            if max(abs(x) for x in d) < tol:
                return

    def arm(self, target: list[float], exclude: frozenset[int] = frozenset()) -> None:
        """BiRRT to `target`, then follow it."""
        for q in _arm_path(self.sim, target, set(exclude), self.seed)[1:]:
            self.joints(q)

    def base(self, target: SE2Pose, attempts: int = 200) -> None:
        """Drive the base along a dense path to the target."""
        sim = self.sim
        for q in _base_path(sim, target):
            d = q - sim.robot.get_base()
            self.step([d.x, d.y, d.rot] + [0.0] * 8)
        for _ in range(attempts):
            d = target - sim.robot.get_base()
            if max(abs(d.x), abs(d.y), abs(d.rot)) < 1e-3:
                return
            self.step([d.x, d.y, d.rot] + [0.0] * 8)

    def gripper(self, close: bool) -> None:
        """Close or open the gripper until it holds an item or not."""
        for _ in range(5):
            self.step([0.0] * 10 + [-1.0 if close else 1.0])
            held = self.sim._grasped_object  # pylint: disable=protected-access
            if (held is not None) == close:
                return


def execute_action(
    sim: ObjectCentricBedsideStocking3DEnv,
    action: tuple[str, tuple[Any, ...]],
    seed: int = 0,
    start: bool = False,
) -> None:
    """Execute one move_base, pick, or place action, refining arm paths with
    BiRRT. At the start of a plan, the arm first moves to its carry conf."""
    drive = Driver(sim, seed)
    if start:
        drive.arm(CARRY_JOINTS)
    name, args = action
    if name == "move_base":
        drive.base(args[1].pose)
    elif name in ("pick", "place"):
        traj = args[4]
        # pylint: disable-next=protected-access
        target = sim._object_name_to_pybullet_id(args[0])
        _, hover, grasp = traj.joint_plan
        approach = _arm_path(sim, hover, {target}, seed)
        for q in approach[1:] + _descent(sim, hover, grasp, target, seed):
            drive.joints(q)
        drive.gripper(close=name == "pick")
        drive.joints(hover)
        drive.arm(traj.joint_plan[0])
    else:
        raise ValueError(f"Unknown action: {name}")


def execute_plan(
    sim: ObjectCentricBedsideStocking3DEnv, plan: list, seed: int = 0
) -> bool:
    """Execute a move_base/pick/place plan and return whether the goal was
    reached."""
    for i, action in enumerate(plan):
        execute_action(sim, action, seed, start=i == 0)
    return sim.goal_reached()
