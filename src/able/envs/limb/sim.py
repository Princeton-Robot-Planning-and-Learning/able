"""The PyBullet simulation of the robot and the person's limb.

LimbSim wraps kinder's limb repositioning environment. The robot grasps the limb with
a weld, and actions are torque corrections on top of the torque that holds the
coupled system still. The person holds up their own limb with their joint torques
(see person.py), so the robot holds only its arm. The limb's joint limits are the
widest range the prior allows; the person's range of motion enters the physics only
through the walls their joints put past it, when a step is given the limits.
"""

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_utils import NUM_ROBOT_JOINTS, joint_position_distance
from kinder.envs.dynamic3d.limbrepositioning3d import (
    ObjectCentricLimbRepositioning3DEnv,
    create_variant_config,
)
from kinder.envs.dynamic3d.limbs import BodyMass
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, matrix_from_quat

from able.envs.limb.human import LimbVariant
from able.envs.limb.person import JointLimits, PersonModel

# Kinova Gen3 joint torque limits (N m); the environment ships with +-1.
ROBOT_TORQUE_LIMITS = (200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 100.0)
# How far the base may move from its home pose (m), so motion planning has bounds.
BASE_REACH = 3.0
# Damping singular coupling Jacobians, as in limb-manipulation.
JACOBIAN_DAMPING = 0.1
SINGULARITY_THRESHOLD = 0.02
# Headroom below the robot's torque limits for holding the limb (N m).
TORQUE_FEASIBILITY_MARGIN = 1.0


@dataclass(frozen=True)
class LimbSimSettings:
    """Physics settings. The body mass is the 50th percentile adult female, and the
    person's joints act as the person model says."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    muscle_tone: str = "none"
    joint_limits_model: str = "box"
    limb_joint_damping: float = 0.5
    body_mass_kg: float = 62.5
    # The widest range-of-motion scale, used as the limb's physical limits.
    envelope_scale: float = 1.2
    person: PersonModel = PersonModel()


@dataclass(frozen=True)
class GraspFrame:
    """The frames and strength of the weld between the robot and the limb."""

    parent: Pose
    child: Pose
    max_force: float
    erp: float


@dataclass(frozen=True)
class CoupledState:
    """The robot and limb state, including the base pose and the grasp weld."""

    base_pose: SE2Pose
    robot_positions: tuple[float, ...]
    robot_velocities: tuple[float, ...]
    limb_positions: tuple[float, ...]
    limb_velocities: tuple[float, ...]
    grasp_frame: GraspFrame | None = None


@dataclass(frozen=True)
class HumanTorques:
    """The torque on the limb's joints over one step, the static loads in it, and
    the person's resistance: their joint torque beyond holding up the limb, such as
    the walls past their limits."""

    total: NDArray[np.float64]
    tone: NDArray[np.float64]
    gravity: NDArray[np.float64]
    resistance: NDArray[np.float64]

    @property
    def robot(self) -> NDArray[np.float64]:
        """The torque the robot adds over the static load. Where the robot pushes
        against the person's resistance, it loads the joint by that much even if
        the limb stays still."""
        return self.total - self.gravity - self.resistance


class LimbSim:
    """A PyBullet world with the robot, the person, and any furniture."""

    def __init__(
        self,
        variant: LimbVariant,
        settings: LimbSimSettings = LimbSimSettings(),
        _env: ObjectCentricLimbRepositioning3DEnv | None = None,
    ) -> None:
        self.variant = variant
        self.settings = settings
        self.env = self._build_env() if _env is None else _env
        self._apply_limb_joint_damping()

    def _build_env(self) -> ObjectCentricLimbRepositioning3DEnv:
        config = create_variant_config(self.variant.name)
        settings = self.settings
        envelope = np.full(len(self.variant.active_motions), settings.envelope_scale)
        config = dataclasses.replace(
            config,
            gravity=settings.gravity,
            torque_lower_limits=tuple(-t for t in ROBOT_TORQUE_LIMITS),
            torque_upper_limits=ROBOT_TORQUE_LIMITS,
            scene=dataclasses.replace(
                config.scene,
                limb_muscle_tone_model_name=settings.muscle_tone,
                limb_joint_limits_model_name=settings.joint_limits_model,
                limb_range_of_motion=self.variant.get_range_of_motion(envelope),
                limb_body_mass=BodyMass(total=settings.body_mass_kg),
            ),
        )
        env = ObjectCentricLimbRepositioning3DEnv(
            variant=self.variant.name, config=config, use_gui=False
        )
        env.reset(seed=0)
        home = env.robot.get_base()
        env.robot.base.pose_lower_bound = SE2Pose(
            home.x - BASE_REACH, home.y - BASE_REACH, -np.pi
        )
        env.robot.base.pose_upper_bound = SE2Pose(
            home.x + BASE_REACH, home.y + BASE_REACH, np.pi
        )
        return env

    @property
    def client(self) -> int:
        """The PyBullet client id."""
        return int(self.env.physics_client_id)

    @property
    def dt(self) -> float:
        """The duration of one step (s)."""
        return float(self.env.config.dt)

    @property
    def goal_tolerance(self) -> float:
        """The limb error within which the environment counts the goal reached."""
        return float(self.env.config.goal_atol)

    @property
    def robot_torque_limits(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The robot's (lower, upper) joint torque limits."""
        config = self.env.config
        return (
            np.asarray(config.torque_lower_limits, dtype=np.float64),
            np.asarray(config.torque_upper_limits, dtype=np.float64),
        )

    def close(self) -> None:
        """Disconnect the PyBullet client."""
        self.env.close()

    def clone(self) -> "LimbSim":
        """An independent world with the same dynamics, in the same state."""
        env = ObjectCentricLimbRepositioning3DEnv(
            variant=self.variant.name, config=self.env.config, use_gui=False
        )
        try:
            self._copy_dynamics(env)
            clone = LimbSim(self.variant, self.settings, _env=env)
            clone.restore(self.capture())
            return clone
        except BaseException:
            env.close()
            raise

    def _copy_dynamics(self, env: ObjectCentricLimbRepositioning3DEnv) -> None:
        params = p.getPhysicsEngineParameters(physicsClientId=self.client)
        keys = (
            "fixedTimeStep",
            "numSubSteps",
            "numSolverIterations",
            "numNonContactInnerIterations",
        )
        p.setPhysicsEngineParameter(
            **{key: params[key] for key in keys if key in params},
            physicsClientId=env.physics_client_id,
        )
        for source, target in (
            (self.env.limb, env.limb),
            (self.env.robot.arm, env.robot.arm),
        ):
            num_joints = p.getNumJoints(source.robot_id, physicsClientId=self.client)
            for link in range(-1, num_joints):
                info = p.getDynamicsInfo(
                    source.robot_id, link, physicsClientId=self.client
                )
                p.changeDynamics(
                    target.robot_id,
                    link,
                    mass=info[0],
                    localInertiaDiagonal=info[2],
                    lateralFriction=info[1],
                    restitution=info[5],
                    rollingFriction=info[6],
                    spinningFriction=info[7],
                    physicsClientId=env.physics_client_id,
                )

    def _apply_limb_joint_damping(self) -> None:
        """Damp the limb's joints, which the environment's torque control
        undamps."""
        for joint in self.env.limb.arm_joints:
            p.changeDynamics(
                self.env.limb.robot_id,
                joint,
                jointDamping=self.settings.limb_joint_damping,
                physicsClientId=self.client,
            )

    # Grasp weld.

    @property
    def is_grasping(self) -> bool:
        """Whether the robot is welded to the limb."""
        # pylint: disable-next=protected-access
        return self.env._grasp_constraint_id is not None

    def release_grasp(self) -> None:
        """Remove the weld, if there is one."""
        if not self.is_grasping:
            return
        p.removeConstraint(
            self.env._grasp_constraint_id,  # pylint: disable=protected-access
            physicsClientId=self.client,
        )
        # pylint: disable-next=protected-access
        self.env._grasp_constraint_id = None  # type: ignore[assignment]

    def engage_grasp(self, frame: GraspFrame | None = None) -> None:
        """Weld the robot to the limb with the frame, or where they stand."""
        env = self.env
        self.release_grasp()
        if frame is None:
            # pylint: disable-next=protected-access
            env._grasp_constraint_id = env._create_grasp_constraint()
            return
        constraint = p.createConstraint(
            env.robot.arm.robot_id,
            env.robot.arm.end_effector_id,
            env.limb.robot_id,
            env.limb.end_effector_id,
            jointType=p.JOINT_FIXED,
            jointAxis=(0.0, 0.0, 0.0),
            parentFramePosition=frame.parent.position,
            childFramePosition=frame.child.position,
            parentFrameOrientation=frame.parent.orientation,
            childFrameOrientation=frame.child.orientation,
            physicsClientId=self.client,
        )
        p.changeConstraint(
            constraint, maxForce=frame.max_force, physicsClientId=self.client
        )
        # An ERP of 0 means no override in Bullet.
        if frame.erp > 0.0:
            p.changeConstraint(constraint, erp=frame.erp, physicsClientId=self.client)
        env._grasp_constraint_id = constraint  # pylint: disable=protected-access

    # State.

    def capture(self) -> CoupledState:
        """The current state."""
        env, frame = self.env, None
        if self.is_grasping:
            info = p.getConstraintInfo(
                env._grasp_constraint_id,  # pylint: disable=protected-access
                physicsClientId=self.client,
            )
            frame = GraspFrame(
                Pose(info[6], info[8]), Pose(info[7], info[9]), info[10], info[14]
            )
        return CoupledState(
            base_pose=env.robot.get_base(),
            robot_positions=tuple(env.robot.arm.get_joint_positions()),
            robot_velocities=tuple(env.robot.arm.get_joint_velocities()),
            limb_positions=tuple(env.limb.get_joint_positions()),
            limb_velocities=tuple(env.limb.get_joint_velocities()),
            grasp_frame=frame,
        )

    def restore(self, state: CoupledState, regrasp: bool = True) -> None:
        """Put the world into the state; regrasp rebuilds the weld it stored."""
        env = self.env
        base = env.robot.get_base()
        target = state.base_pose
        if not np.allclose(
            (base.x, base.y, base.rot), (target.x, target.y, target.rot)
        ):
            self.release_grasp()
            env.robot.set_base(target)
        env.robot.arm.set_joints(
            list(state.robot_positions), joint_velocities=list(state.robot_velocities)
        )
        env.limb.set_joints(
            list(state.limb_positions), joint_velocities=list(state.limb_velocities)
        )
        if regrasp:
            self.engage_grasp(state.grasp_frame)

    @contextmanager
    def saved_state(self) -> Iterator[None]:
        """Restore the current state, including the weld, on the way out."""
        saved, was_grasping = self.capture(), self.is_grasping
        try:
            yield
        finally:
            if p.isConnected(physicsClientId=self.client):
                if not was_grasping:
                    self.release_grasp()
                self.restore(saved, regrasp=was_grasping)

    def get_limb_error(self, positions: Any, goal: Any) -> float:
        """The environment's distance between limb configurations."""
        infos = self.env.limb.get_arm_joint_infos()
        return float(joint_position_distance(infos, list(positions), list(goal)))

    # Torques and stepping.

    def advance(
        self, torque: NDArray[np.float64], person_limits: JointLimits | None = None
    ) -> HumanTorques:
        """Step once under the robot torque, clipped to the limits, with the
        person's joint torque, and return the torque on the limb. The person's
        joints put walls past person_limits, if given."""
        env = self.env
        space = env.torque_action_space
        positions, velocities = self._get_limb_state()
        tone = np.asarray(env.limb.get_muscle_tone_torque(), dtype=np.float64)
        gravity = self._get_gravity_torque(env.limb)
        person = self.settings.person
        limb_name = self.variant.limb_name
        person_torque = person.get_torque(
            np.asarray(positions),
            np.asarray(velocities),
            gravity,
            person_limits,
            limb_name,
        )
        # The environment adds muscle tone itself.
        clipped = list(np.clip(torque, space.low, space.high))
        # pylint: disable-next=protected-access
        env._apply_torques(clipped, extra_limb_torque=list(person_torque))
        _, new_velocities = self._get_limb_state()
        accelerations = (np.array(new_velocities) - np.array(velocities)) / self.dt
        total = self._inverse_dynamics(
            env.limb.robot_id, positions, velocities, list(accelerations)
        )
        resistance = person_torque - person.get_support_torque(gravity, limb_name)
        return HumanTorques(
            total=total, tone=tone, gravity=gravity, resistance=resistance
        )

    def get_hold_torque(self) -> NDArray[np.float64]:
        """The robot torque that holds the coupled system still: the arm's weight
        plus whatever of the limb's the person does not hold, clipped to the
        limits."""
        space = self.env.torque_action_space
        coupling = np.clip(self._get_limb_coupling_torque(), space.low, space.high)
        return self._get_arm_gravity_torque() + coupling

    def get_commanded_torque(
        self, correction: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The correction plus the hold torque: what the robot is asked for."""
        return np.asarray(correction, dtype=np.float64) + self.get_hold_torque()

    def advance_corrected(
        self,
        correction: NDArray[np.float64],
        person_limits: JointLimits | None = None,
    ) -> HumanTorques:
        """Step once under the correction on top of the hold torque."""
        return self.advance(self.get_commanded_torque(correction), person_limits)

    def get_control_saturation(self) -> float:
        """The torque needed to hold still as a fraction of the robot's limits,
        less a margin; infinite near a singular grasp. The robot carries the limb's
        weight here, so it can hold the limb if the person lets go."""
        limb_jacobian = self._get_tool_jacobian(self.env.limb)
        if np.linalg.svd(limb_jacobian, compute_uv=False)[-1] < SINGULARITY_THRESHOLD:
            return float("inf")
        hold = self._get_arm_gravity_torque() + self._get_limb_coupling_torque(
            supported=False
        )
        if not hold.any():
            return 0.0
        space = self.env.torque_action_space
        upper = np.asarray(space.high) - TORQUE_FEASIBILITY_MARGIN
        lower = np.asarray(space.low) + TORQUE_FEASIBILITY_MARGIN
        return float(np.max(np.maximum(hold / upper, hold / lower)))

    def get_limb_directed_torque(
        self, goal: NDArray[np.float64], gain: float
    ) -> NDArray[np.float64]:
        """The robot torque that pushes the limb toward the goal configuration."""
        error = np.asarray(goal) - np.asarray(self.env.limb.get_joint_positions())
        return self._get_robot_torque_for_limb_load(gain * error)

    def _get_limb_state(self) -> tuple[list[float], list[float]]:
        states = p.getJointStates(
            self.env.limb.robot_id,
            self.env.limb.arm_joints,
            physicsClientId=self.client,
        )
        return [s[0] for s in states], [s[1] for s in states]

    def _inverse_dynamics(
        self,
        body_id: int,
        positions: list[float],
        velocities: list[float],
        accelerations: list[float],
    ) -> NDArray[np.float64]:
        """Inverse dynamics with gravity in the body's base frame.

        PyBullet assumes an upright base, which is wrong for the bed and the
        wheelchair, where the limb's base is rolled.
        """
        gravity = np.asarray(self.env.config.gravity, dtype=np.float64)
        _, orientation = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self.client
        )
        local = matrix_from_quat(orientation).T @ gravity
        p.setGravity(*local, physicsClientId=self.client)
        try:
            return np.asarray(
                p.calculateInverseDynamics(
                    body_id,
                    positions,
                    velocities,
                    accelerations,
                    physicsClientId=self.client,
                ),
                dtype=np.float64,
            )
        finally:
            p.setGravity(*gravity, physicsClientId=self.client)

    def _get_gravity_torque(self, body: Any) -> NDArray[np.float64]:
        """The torque gravity puts on the body's joints."""
        joints = sorted(body.arm_joints)
        states = p.getJointStates(body.robot_id, joints, physicsClientId=self.client)
        rest = [0.0] * len(joints)
        return self._inverse_dynamics(body.robot_id, [s[0] for s in states], rest, rest)

    def _get_arm_gravity_torque(self) -> NDArray[np.float64]:
        if not any(self.env.config.gravity):
            return np.zeros(NUM_ROBOT_JOINTS)
        return self._get_gravity_torque(self.env.robot.arm)[:NUM_ROBOT_JOINTS]

    def _get_limb_coupling_torque(self, supported: bool = True) -> NDArray[np.float64]:
        """The robot torque that carries the limb's weight, net of muscle tone and,
        if supported, of the person holding up their own limb."""
        tone = np.asarray(self.env.limb.get_muscle_tone_torque(), dtype=np.float64)
        gravity = self._get_gravity_torque(self.env.limb)
        load = gravity - tone
        if supported:
            person = self.settings.person
            load = load - person.get_support_torque(gravity, self.variant.limb_name)
        return self._get_robot_torque_for_limb_load(load)

    def _get_tool_jacobian(self, body: Any) -> NDArray[np.float64]:
        """The 6 x N Jacobian at the body's tool link."""
        joints = sorted(body.arm_joints)
        states = p.getJointStates(body.robot_id, joints, physicsClientId=self.client)
        positions = [s[0] for s in states]
        rest = [0.0] * len(positions)
        translational, rotational = p.calculateJacobian(
            body.robot_id,
            body.tool_link_id,
            [0.0, 0.0, 0.0],
            positions,
            rest,
            rest,
            physicsClientId=self.client,
        )
        return np.vstack([np.asarray(translational), np.asarray(rotational)])

    def _get_robot_torque_for_limb_load(
        self, load: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The robot torque that applies the joint torque load to the limb through
        the grasp, with damped least squares near singularities."""
        if not load.any():
            return np.zeros(NUM_ROBOT_JOINTS)
        env = self.env
        arm_jacobian = self._get_tool_jacobian(env.robot.arm)
        limb_jacobian = self._get_tool_jacobian(env.limb)
        smallest = float(np.linalg.svd(limb_jacobian, compute_uv=False)[-1])
        damping = 0.0
        if smallest < SINGULARITY_THRESHOLD:
            damping = JACOBIAN_DAMPING**2 * (
                1.0 - (smallest / SINGULARITY_THRESHOLD) ** 2
            )
        gram = limb_jacobian @ limb_jacobian.T
        damped = np.linalg.solve(
            gram + damping * np.eye(len(gram)), limb_jacobian @ load
        )
        # Rotate the wrench from the limb's base frame to the robot's.
        robot_rotation = matrix_from_quat(env.robot.arm.get_base_pose().orientation)
        rotation = matrix_from_quat(env.limb.get_base_pose().orientation).T @ (
            robot_rotation
        )
        transform = np.eye(6)
        transform[:3, :3] = transform[3:, 3:] = rotation
        wrench = transform.T @ damped
        # Shift the moment from the limb's grasp frame to the robot's tool frame.
        offset = robot_rotation.T @ np.subtract(
            env.limb.get_end_effector_pose().position,
            env.robot.arm.get_end_effector_pose().position,
        )
        wrench[3:] = wrench[3:] + np.cross(offset, wrench[:3])
        return np.asarray((arm_jacobian.T @ wrench)[:NUM_ROBOT_JOINTS])
