"""The friction environment's 3D simulator: TidyBot strikes a puck with a paddle
across a floor maze of unknown friction.

Gravity is off and floor friction is applied by hand (avoids Bullet catching on tile
edges, matches the analytic slide model); the base teleports to each push station.
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from kinder.core import FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DPointType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions
from relational_structs import Array, Object
from relational_structs.utils import create_state_from_dict

from able.envs.friction.sim2d import (
    PushActionSpace,
    parse_push_action,
    sample_frictions,
)
from able.envs.friction.world import (
    DEFAULT_FRICTION_PRIOR,
    GOAL_RGB,
    GRAVITY,
    HOLE_RGB,
    MAPS,
    PUCK_RGB,
    SURFACE_RGB,
    WALL_RGB,
    FrictionPrior,
    PuckPhysics,
    PuckWorld,
)

# Start gap (m) so paddle and puck do not overlap.
_PADDLE_GAP = 0.005
# Carry pose in the base frame (m): clears base and arm by 11 cm, the route by 4 cm.
_CARRY_FORWARD = 0.3
_CARRY_HEIGHT = 0.55
# Gripper links hold the paddle: the only robot parts it does not collide with.
_GRIPPER_LINK_WORDS = (
    "robotiq",
    "finger",
    "knuckle",
    "end_effector",
    "tool_frame",
    "inertia",
)


@dataclass(frozen=True)
class PuckPushing3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for PuckPushing3DEnv()."""

    map_name: str = "threeroutes"
    physics: PuckPhysics = PuckPhysics()
    friction_prior: FrictionPrior = DEFAULT_FRICTION_PRIOR

    wall_height: float = 0.3
    paddle_thickness: float = 0.04
    paddle_height: float = 0.15
    paddle_mass: float = 20.0
    # Gripper points down just behind the paddle.
    initial_joints: JointPositions = field(
        default_factory=lambda: [2.439, -1.75, 0.866, -2.147, -2.156, -1.118, 1.648]
    )
    realistic_bg: bool = False

    physics_dt: float = 1.0 / 240.0
    # Bullet contacts come out bouncier than set; offset calibrated by shots at a wall.
    bullet_restitution_offset: float = -0.07
    stop_speed: float = 1e-3
    max_slide_seconds: float = 30.0

    def get_camera_kwargs(self) -> dict[str, Any]:
        puck_map = MAPS[self.map_name]
        return {
            "camera_target": (puck_map.width / 2, puck_map.height / 2, 0.0),
            "camera_yaw": 0,
            "camera_distance": 0.9 * max(puck_map.width, puck_map.height),
            "camera_pitch": -60,
        }


class PuckPushing3DObjectCentricState(Kinematic3DObjectCentricState):
    """A PuckPushing3DEnv state; the frictions are hidden, not part of it."""


class ObjectCentricPuckPushing3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        PuckPushing3DObjectCentricState, PuckPushing3DEnvConfig
    ]
):
    """Push a puck to a goal across floor patches of unknown friction."""

    def __init__(
        self,
        config: PuckPushing3DEnvConfig | None = None,
        map_name: str | None = None,
        **kwargs,
    ) -> None:
        config = config or PuckPushing3DEnvConfig()
        if map_name is not None:
            config = dataclasses.replace(config, map_name=map_name)
        puck_map = MAPS[config.map_name]
        config = dataclasses.replace(
            config,
            robot_base_home_pose=SE2Pose(*puck_map.robot_start, 0.0),
            x_lb=0.0,
            x_ub=puck_map.width,
            y_lb=0.0,
            y_ub=puck_map.height,
        )
        self.world = PuckWorld(puck_map, config.physics)
        # Puck (x, y) and robot (x, y, heading; None: home) at reset.
        self._start: tuple[Any, Any] = (puck_map.puck_start, None)
        super().__init__(config=config, **kwargs)
        client = self.physics_client_id
        phys = config.physics
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setTimeStep(config.physics_dt, physicsClientId=client)
        # Else Bullet drops the restitution of slow or early contacts.
        p.setPhysicsEngineParameter(
            restitutionVelocityThreshold=0.0, physicsClientId=client
        )

        # Floor patches, holes and goal are visual only.
        for surface, (x0, y0, x1, y1) in puck_map.surface_rects():
            _visual_box(
                client,
                ((x1 - x0) / 2, (y1 - y0) / 2, 0.005),
                ((x0 + x1) / 2, (y0 + y1) / 2, -0.005),
                (*SURFACE_RGB[surface], 1.0),
            )
        self._hole_ids = [
            _visual_disc(client, phys.hole_radius, (hx, hy, 0.001), (*HOLE_RGB, 1.0))
            for hx, hy in self.world.holes
        ]
        self._goal_id = _visual_disc(
            client, phys.goal_radius, (*self.world.goal, 0.0005), (*GOAL_RGB, 1.0)
        )

        self._walls: dict[str, int] = {}
        self._wall_half_extents: dict[str, tuple[float, float, float]] = {}
        for i, (x0, y0, x1, y1) in enumerate(puck_map.wall_rects()):
            half = ((x1 - x0) / 2, (y1 - y0) / 2, config.wall_height / 2)
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=client
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=(*WALL_RGB, 1.0),
                physicsClientId=client,
            )
            wall = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=((x0 + x1) / 2, (y0 + y1) / 2, half[2]),
                physicsClientId=client,
            )
            # Bullet multiplies the two restitutions, so the puck's is the wall's.
            p.changeDynamics(
                wall,
                -1,
                restitution=1.0,
                lateralFriction=0.0,
                contactProcessingThreshold=0.0,
                physicsClientId=client,
            )
            self._walls[f"wall{i}"] = wall
            self._wall_half_extents[f"wall{i}"] = half

        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=phys.puck_radius,
            height=phys.puck_height,
            physicsClientId=client,
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=phys.puck_radius,
            length=phys.puck_height,
            rgbaColor=(*PUCK_RGB, 1.0),
            physicsClientId=client,
        )
        self.puck_id = p.createMultiBody(
            baseMass=phys.puck_mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=(*puck_map.puck_start, self.puck_z),
            physicsClientId=client,
        )
        p.changeDynamics(
            self.puck_id,
            -1,
            restitution=phys.wall_restitution + config.bullet_restitution_offset,
            lateralFriction=0.0,
            spinningFriction=0.0,
            rollingFriction=0.0,
            linearDamping=0.0,
            angularDamping=0.0,
            contactProcessingThreshold=0.0,
            # Bullet puts cylinder contacts on the rim; a huge inertia keeps those
            # impulses from tipping the puck, which stays flat with its spin zeroed.
            localInertiaDiagonal=(1e3, 1e3, 1e3),
            physicsClientId=client,
        )

        paddle_half = (
            config.paddle_thickness / 2,
            phys.paddle_width / 2,
            config.paddle_height / 2,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=paddle_half, physicsClientId=client
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=paddle_half,
            rgbaColor=(0.2, 0.2, 0.8, 1.0),
            physicsClientId=client,
        )
        self.paddle_id = p.createMultiBody(
            baseMass=config.paddle_mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            physicsClientId=client,
        )
        p.changeDynamics(
            self.paddle_id,
            -1,
            restitution=0.0,
            lateralFriction=0.0,
            linearDamping=0.0,
            angularDamping=0.0,
            contactProcessingThreshold=0.0,
            physicsClientId=client,
        )

        # All collide except paddle vs gripper, and the invisible EE marker vs anything.
        base_id = self.robot.base.robot_id
        arm_id = self.robot.arm.robot_id
        self._paddle_robot_links = [(base_id, -1)] + [
            (base_id, link)
            for link in range(p.getNumJoints(base_id, physicsClientId=client))
        ]
        for link in range(-1, p.getNumJoints(arm_id, physicsClientId=client)):
            name = (
                p.getJointInfo(arm_id, link, physicsClientId=client)[12].decode()
                if link >= 0
                else "arm_base"
            )
            if any(word in name for word in _GRIPPER_LINK_WORDS):
                p.setCollisionFilterPair(
                    arm_id, self.paddle_id, link, -1, 0, physicsClientId=client
                )
            else:
                self._paddle_robot_links.append((arm_id, link))
        for other in (self.puck_id, self.paddle_id):
            p.setCollisionFilterPair(
                self.end_effector_viz_id, other, -1, -1, 0, physicsClientId=client
            )
        for link in range(p.getNumJoints(base_id, physicsClientId=client)):
            # Like the walls: the puck's restitution, no friction.
            p.changeDynamics(
                base_id,
                link,
                restitution=1.0,
                lateralFriction=0.0,
                contactProcessingThreshold=0.0,
                physicsClientId=client,
            )

        # Rigid grasp: paddle pose in the EE frame, from the initial arm pose.
        base = self.robot.get_base()
        face = np.array([base.x, base.y]) + phys.paddle_offset * np.array(
            [math.cos(base.rot), math.sin(base.rot)]
        )
        self._ee_to_paddle = multiply_poses(
            self.robot.arm.get_end_effector_pose().invert(),
            self._paddle_pose(face, base.rot),
        )

        self._frictions: tuple[float, ...] = tuple(
            float(f) for f in config.friction_prior.mean
        )
        self._outcome: str | None = None  # "goal", "hole" or None
        self._last_info: dict[str, Any] = {}

    @property
    def puck_z(self) -> float:
        """Height of the puck's center, resting on the floor."""
        return self.config.physics.puck_height / 2 + 1e-3

    @property
    def frictions(self) -> tuple[float, ...]:
        """True (hidden) friction per surface; for evaluation only, not planners."""
        return self._frictions

    @property
    def puck_position(self) -> NDArray[np.float64]:
        """The puck's true position."""
        return self._puck_position()

    @property
    def robot_pose(self) -> NDArray[np.float64]:
        """The base's true (x, y, heading)."""
        base = self.robot.get_base()
        return np.array([base.x, base.y, base.rot])

    # ------------------------------------------------------------------ spaces

    def _create_action_space(  # type: ignore[override]
        self, config: PuckPushing3DEnvConfig
    ) -> PushActionSpace:
        return PushActionSpace(config.physics.max_speed, config.physics.max_stance)

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return PuckPushing3DObjectCentricState

    # ------------------------------------------------------------------ objects

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict(
            [(name, Kinematic3DCuboidType) for name in self._walls]
            + [("goal", Kinematic3DPointType)]
            + [(f"hole{i}", Kinematic3DPointType) for i in range(len(self._hole_ids))]
        )

    def _get_obs(self) -> PuckPushing3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType), ("puck", Kinematic3DCuboidType)]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=PuckPushing3DObjectCentricState,
        )
        assert isinstance(state, PuckPushing3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        return self._outcome == "goal"

    def _reset_objects(self) -> None:
        start, robot = self._start
        if robot is not None:
            self.robot.set_base(SE2Pose(*robot))
        set_pose(self.puck_id, Pose((*start, self.puck_z)), self.physics_client_id)
        self._stop_puck()
        self._place_paddle()

    def _set_object_states(self, obs: PuckPushing3DObjectCentricState) -> None:
        pose = obs.get_object_pose("puck")
        position = (pose.position[0], pose.position[1], self.puck_z)
        set_pose(self.puck_id, Pose(position), self.physics_client_id)
        self._stop_puck()
        self._place_paddle()

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "puck":
            return self.puck_id
        if object_name == "goal":
            return self._goal_id
        if object_name.startswith("hole"):
            return self._hole_ids[int(object_name[len("hole") :])]
        if object_name in self._walls:
            return self._walls[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return set(self._walls.values())

    def _get_movable_object_names(self) -> set[str]:
        return set()

    def _get_surface_object_names(self) -> set[str]:
        return set()

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name == "puck":
            phys = self.config.physics
            return (phys.puck_radius, phys.puck_radius, phys.puck_height / 2)
        if object_name in self._wall_half_extents:
            return self._wall_half_extents[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    # ------------------------------------------------------------------ physics

    def _puck_position(self) -> NDArray[np.float64]:
        position = get_pose(self.puck_id, self.physics_client_id).position
        return np.array(position[:2])

    def _puck_velocity(self) -> NDArray[np.float64]:
        linear, _ = p.getBaseVelocity(
            self.puck_id, physicsClientId=self.physics_client_id
        )
        return np.array(linear[:2])

    def _set_puck_velocity(self, velocity: NDArray[np.float64]) -> None:
        # Pinning z and spin keeps the puck flat on the floor without gravity.
        p.resetBaseVelocity(
            self.puck_id,
            (float(velocity[0]), float(velocity[1]), 0.0),
            (0.0, 0.0, 0.0),
            physicsClientId=self.physics_client_id,
        )

    def _stop_puck(self) -> None:
        self._set_puck_velocity(np.zeros(2))

    def _apply_floor_friction(self) -> None:
        """Coulomb friction from the patch under the puck's center, for one step."""
        velocity = self._puck_velocity()
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            self._set_puck_velocity(velocity)
            return
        mu = self.world.friction_at(self._puck_position(), self._frictions)
        decel = mu * GRAVITY * self.config.physics_dt
        if speed <= decel:
            self._stop_puck()
        else:
            self._set_puck_velocity(velocity * (1.0 - decel / speed))

    def _paddle_pose(self, face: NDArray[np.float64], angle: float) -> Pose:
        """Paddle pose with the middle of its face at `face`, facing `angle`."""
        config = self.config
        back = config.paddle_thickness / 2
        x = face[0] - back * math.cos(angle)
        y = face[1] - back * math.sin(angle)
        z = config.paddle_height / 2 + 2e-3
        orientation = p.getQuaternionFromEuler((0.0, 0.0, angle))
        return Pose((x, y, z), orientation)

    def _place_paddle(
        self,
        face: NDArray[np.float64] | None = None,
        angle: float | None = None,
    ) -> None:
        """Put the paddle face at `face`, facing `angle` (default: in front)."""
        base = self.robot.get_base()
        if face is None:
            reach = self.config.physics.paddle_offset
            face = np.array([base.x, base.y]) + reach * np.array(
                [math.cos(base.rot), math.sin(base.rot)]
            )
        self._hold_paddle_at(
            self._paddle_pose(face, base.rot if angle is None else angle)
        )

    def _stow_paddle(self) -> None:
        """Lift the paddle to the carry pose, clear of the puck wherever the base is."""
        self._hold_paddle_at(self._carry_pose())

    def _carry_pose(self) -> Pose:
        """Paddle pose while driving."""
        base = self.robot.get_base()
        heading = np.array([math.cos(base.rot), math.sin(base.rot)])
        xy = np.array([base.x, base.y]) + _CARRY_FORWARD * heading
        orientation = p.getQuaternionFromEuler((0.0, 0.0, base.rot))
        return Pose((xy[0], xy[1], _CARRY_HEIGHT), orientation)

    def _hold_paddle_at(self, paddle_pose: Pose) -> bool:
        """IK the arm to put the paddle at `paddle_pose`; False (arm kept) if none."""
        arm = self.robot.arm
        joints = arm.get_joint_positions()
        target = multiply_poses(paddle_pose, self._ee_to_paddle.invert())
        try:
            inverse_kinematics(arm, target, validate=True, set_joints=True)
            reached = True
        except InverseKinematicsError:
            arm.set_joints(joints)
            reached = False
        self._attach_paddle()
        return reached

    def paddle_robot_gap(self) -> float:
        """Min paddle-to-robot distance (m), gripper excluded; negative on overlap."""
        gap = 1.0
        for body, link in self._paddle_robot_links:
            for point in p.getClosestPoints(
                self.paddle_id,
                body,
                1.0,
                -1,
                link,
                physicsClientId=self.physics_client_id,
            ):
                gap = min(gap, point[8])
        return gap

    def _attach_paddle(self) -> None:
        """Put the paddle where the gripper holds it, at rest."""
        pose = multiply_poses(
            self.robot.arm.get_end_effector_pose(), self._ee_to_paddle
        )
        set_pose(self.paddle_id, pose, self.physics_client_id)
        p.resetBaseVelocity(
            self.paddle_id, (0, 0, 0), (0, 0, 0), physicsClientId=self.physics_client_id
        )

    def _set_paddle_restitution(self, restitution: float) -> None:
        p.changeDynamics(
            self.paddle_id,
            -1,
            restitution=restitution,
            physicsClientId=self.physics_client_id,
        )

    def _set_paddle_puck_contact(self, enabled: bool) -> None:
        p.setCollisionFilterPair(
            self.paddle_id,
            self.puck_id,
            -1,
            -1,
            int(enabled),
            physicsClientId=self.physics_client_id,
        )

    # ------------------------------------------------------------------ reset/step

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[PuckPushing3DObjectCentricState, dict]:
        """Reset; `options` may fix "frictions", and the "puck" (x, y) and "robot"
        (x, y, heading) start."""
        opts = options or {}
        self._start = (
            opts.get("puck", self.world.puck_map.puck_start),
            opts.get("robot"),
        )
        obs, info = super().reset(seed=seed, options=options)
        if options is not None and "frictions" in options:
            frictions = tuple(float(f) for f in options["frictions"])
            assert len(frictions) == self.config.friction_prior.num_surfaces
            self._frictions = frictions
        elif options is None or "init_state" not in options:
            self._frictions = sample_frictions(
                self.np_random, self.config.friction_prior
            )
        self._outcome = None
        self._last_info = {}
        self._set_paddle_puck_contact(True)
        return obs, info

    def step(
        self, action: Array
    ) -> tuple[PuckPushing3DObjectCentricState, float, bool, bool, dict]:
        assert self._outcome is None, "Episode is over; call reset()"
        push = parse_push_action(action, self.config.physics.max_stance)
        angle, speed, stance = push.angle, push.speed, push.stance
        puck = self._puck_position()
        info: dict[str, Any] = {"push_feasible": False, "drive_distance": 0.0}
        if self.world.push_feasible(puck, angle, stance):
            info.update(self._strike(puck, angle, speed, stance))
        else:
            info.update(
                release_speed=0.0,
                stop_position=self._measure_position(puck),
                fell_in_hole=False,
                puck_trajectory=[puck.copy()],
            )
        if info["fell_in_hole"]:
            self._outcome = "hole"
        elif self.world.in_goal(self._puck_position()):
            self._outcome = "goal"
        info["goal_reached"] = self._outcome == "goal"
        self._last_info = info
        return self._get_obs(), -1.0, self._outcome is not None, False, info

    def _strike(
        self,
        puck: NDArray[np.float64],
        angle: float,
        speed: float,
        stance: float | None = None,
    ) -> dict:
        """Drive behind the puck, push it to speed with the paddle, let it slide."""
        config = self.config
        phys = config.physics
        client = self.physics_client_id
        rng = self.np_random
        # The robot aims and accelerates imperfectly.
        actual_angle = angle + float(rng.normal(0.0, phys.angle_noise))
        actual_speed = max(
            speed * (1.0 + float(rng.normal(0.0, phys.speed_noise))), 0.0
        )
        direction = np.array([math.cos(actual_angle), math.sin(actual_angle)])

        # If the noisy heading is infeasible, stand as for the commanded one.
        pose = self.world.push_pose(puck, actual_angle, stance) or self.world.push_pose(
            puck, angle, stance
        )
        assert pose is not None
        station, heading = pose
        start = self.robot.get_base()
        drive = self.world.drive_distance(
            self.world.nav_node_of(np.array([start.x, start.y])), station, puck
        )
        base_xy = station - _PADDLE_GAP * direction
        self.robot.set_base(SE2Pose(base_xy[0], base_xy[1], heading))
        face = puck - (phys.puck_radius + _PADDLE_GAP) * direction
        self._place_paddle(face, actual_angle)
        self._set_paddle_puck_contact(True)
        # Dead paddle during the strike, so it does not bat the puck ahead.
        self._set_paddle_restitution(0.0)
        paddle_start = np.array(get_pose(self.paddle_id, client).position[:2])
        base_from_paddle = base_xy - paddle_start

        dt = config.physics_dt
        trajectory = [puck.copy()]
        fell = False
        release_speed = 0.0
        if actual_speed > 0:
            accel = actual_speed**2 / (2.0 * phys.accel_distance)
            num_accel = max(int(round(actual_speed / accel / dt)), 1)
            for k in range(num_accel):
                v = accel * (k + 1) * dt * direction
                p.resetBaseVelocity(
                    self.paddle_id,
                    (float(v[0]), float(v[1]), 0.0),
                    (0.0, 0.0, 0.0),
                    physicsClientId=client,
                )
                p.stepSimulation(physicsClientId=client)
                self._apply_floor_friction()
                # The robot follows its paddle.
                paddle_xy = np.array(get_pose(self.paddle_id, client).position[:2])
                base_xy = paddle_xy + base_from_paddle
                self.robot.set_base(SE2Pose(base_xy[0], base_xy[1], heading))
                fell = fell or self.world.over_hole(self._puck_position())
                trajectory.append(self._puck_position())
            release_speed = float(np.linalg.norm(self._puck_velocity()))
        # Release: the paddle stops square to the push; the puck bounces off it.
        p.resetBaseVelocity(
            self.paddle_id, (0, 0, 0), (0, 0, 0), physicsClientId=client
        )
        self._set_paddle_restitution(1.0)
        for _ in range(int(config.max_slide_seconds / dt)):
            if fell or np.linalg.norm(self._puck_velocity()) < config.stop_speed:
                break
            p.stepSimulation(physicsClientId=client)
            self._apply_floor_friction()
            p.resetBaseVelocity(
                self.paddle_id, (0, 0, 0), (0, 0, 0), physicsClientId=client
            )
            fell = self.world.over_hole(self._puck_position())
            trajectory.append(self._puck_position())
        self._stop_puck()
        stop = self._puck_position()
        set_pose(self.puck_id, Pose((stop[0], stop[1], self.puck_z)), client)
        noisy_speed = release_speed + float(rng.normal(0.0, phys.speed_obs_noise))
        return {
            "push_feasible": True,
            "drive_distance": drive,
            "release_speed": max(noisy_speed, 0.0),
            "stop_position": self._measure_position(stop),
            "fell_in_hole": fell,
            # Analysis/videos only; planners see just the two measurements.
            "puck_trajectory": trajectory,
        }

    def _measure_position(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        noise = self.config.physics.position_obs_noise
        return point + self.np_random.normal(0.0, noise, size=2)


def _visual_box(
    client: int,
    half_extents: tuple[float, float, float],
    position: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> int:
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=position,
        physicsClientId=client,
    )


def _visual_disc(
    client: int,
    radius: float,
    position: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> int:
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=0.002,
        rgbaColor=rgba,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=position,
        physicsClientId=client,
    )
