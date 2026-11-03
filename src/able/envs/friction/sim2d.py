"""The friction environment's 2D simulator: a robot strikes a puck across a floor
maze of unknown friction.

Top-down PyMunk has no floor contact, so Coulomb floor friction is applied to the puck's
velocity directly (as in the analytic slide).
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymunk
from kinder.core import FinalConfigMeta
from kinder.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from kinder.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    KinRectangleType,
    KinRobotType,
    SmallCircleType,
)
from kinder.envs.kinematic2d.structs import ZOrder
from kinder.envs.utils import RobotActionSpace
from numpy.typing import NDArray
from relational_structs import Array, Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

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
    Push,
)

PuckType = Type("puck", parent=SmallCircleType)
Dynamic2DRobotEnvTypeFeatures[PuckType] = list(
    Dynamic2DRobotEnvTypeFeatures[SmallCircleType]
)
HoleMarkerType = Type("hole_marker", parent=SmallCircleType)
Dynamic2DRobotEnvTypeFeatures[HoleMarkerType] = list(
    Dynamic2DRobotEnvTypeFeatures[SmallCircleType]
)
GoalMarkerType = Type("goal_marker", parent=SmallCircleType)
Dynamic2DRobotEnvTypeFeatures[GoalMarkerType] = list(
    Dynamic2DRobotEnvTypeFeatures[SmallCircleType]
)

_PUCK_COLLISION_TYPE = 11
_WALL_COLLISION_TYPE = 12
_PADDLE_COLLISION_TYPE = 13
_ROBOT_COLLISION_TYPE = 14
# Paddle face reach (m) while stowed for driving: inside the base footprint.
_STOWED_REACH = 0.2
# Start gap (m) so paddle and puck do not overlap.
_PADDLE_GAP = 0.005


def parse_push_action(action: Array, max_stance: float) -> Push:
    """A push from an env action: (heading, speed) or (heading, speed, stance)."""
    action = np.asarray(action, dtype=np.float64)
    assert action.shape in ((2,), (3,)), f"Bad push action {action}"
    stance = float(action[2]) if len(action) == 3 else None
    assert stance is None or abs(stance) <= max_stance + 1e-9
    return Push(float(action[0]), float(action[1]), stance)


def sample_frictions(
    rng: np.random.Generator,
    prior: FrictionPrior = DEFAULT_FRICTION_PRIOR,
) -> tuple[float, ...]:
    """Draw one friction coefficient per surface from the prior."""
    return tuple(float(f) for f in prior.sample(rng))


class PushActionSpace(RobotActionSpace):
    """Push action (heading, speed, stance); envs also accept (heading, speed)."""

    def __init__(self, max_speed: float, max_stance: float) -> None:
        super().__init__(
            np.array([-np.pi, 0.0, -max_stance]),
            np.array([np.pi, max_speed, max_stance]),
            dtype=np.float64,
        )

    def create_markdown_description(self) -> str:
        """Markdown table of the action features."""
        features = [
            ("angle", "Release heading of the puck (rad, 0 is +x)"),
            ("speed", "Release speed of the puck (m/s)"),
            ("stance", "Robot base heading relative to the push direction (rad)"),
        ]
        md = "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        md += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            md += (
                f"\n| {idx} | {feature} | {description} | "
                f"{self.low[idx]:.3f} | {self.high[idx]:.3f} |"
            )
        return md + "\n"


@dataclass(frozen=True)
class DynPuckPushing2DEnvConfig(Dynamic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for DynPuckPushing2DEnv()."""

    map_name: str = "threeroutes"
    physics: PuckPhysics = PuckPhysics()
    friction_prior: FrictionPrior = DEFAULT_FRICTION_PRIOR

    # No in-plane gravity or damping; floor friction is applied directly.
    gravity_y: float = 0.0
    damping: float = 1.0
    sim_hz: int = 240
    stabilization_seconds: float = 0.0
    # A slide ends when the puck is slower than this, or after max_slide_seconds.
    stop_speed: float = 1e-3
    max_slide_seconds: float = 30.0

    # Matches the threeroutes map; other maps reset these from their own size.
    world_min_x: float = 0.0
    world_max_x: float = 5.2
    world_min_y: float = 0.0
    world_max_y: float = 3.6

    render_dpi: int = 100


class ObjectCentricDynPuckPushing2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynPuckPushing2DEnvConfig]
):
    """Push a puck to a goal across floor patches of unknown friction."""

    def __init__(
        self,
        config: DynPuckPushing2DEnvConfig | None = None,
        map_name: str | None = None,
        **kwargs,
    ) -> None:
        config = config or DynPuckPushing2DEnvConfig()
        if map_name is not None:
            config = dataclasses.replace(config, map_name=map_name)
        puck_map = MAPS[config.map_name]
        config = dataclasses.replace(
            config,
            world_min_x=0.0,
            world_max_x=puck_map.width,
            world_min_y=0.0,
            world_max_y=puck_map.height,
        )
        self.world = PuckWorld(puck_map, config.physics)
        # Puck (x, y) and robot (x, y, heading) at reset.
        self._start = (puck_map.puck_start, (*puck_map.robot_start, 0.0))
        super().__init__(config, **kwargs)
        self._frictions: tuple[float, ...] = tuple(
            float(f) for f in config.friction_prior.mean
        )
        self._robot_pose = np.array([*puck_map.robot_start, 0.0])
        self._puck_body: pymunk.Body | None = None
        self._paddle_body: pymunk.Body | None = None
        self._paddle_stowed = False
        self._robot_body: pymunk.Body | None = None
        self._outcome: str | None = None  # "goal", "hole" or None
        self._last_info: dict[str, Any] = {}
        # Built now so renders show walls and patches from the first reset.
        self._initial_constant_state = self._create_constant_initial_state()

    # ------------------------------------------------------------------ spaces

    def _create_action_space(  # type: ignore[override]
        self, config: DynPuckPushing2DEnvConfig
    ) -> PushActionSpace:
        return PushActionSpace(config.physics.max_speed, config.physics.max_stance)

    # ------------------------------------------------------------------ hidden

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
        return self._robot_pose.copy()

    # ------------------------------------------------------------------ state

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        state: dict[Object, dict[str, float]] = {}
        for i, (x0, y0, x1, y1) in enumerate(self.world.puck_map.wall_rects()):
            state[Object(f"wall{i}", KinRectangleType)] = _rect_features(
                x0, y0, x1, y1, WALL_RGB, ZOrder.SURFACE
            )
        for i, (surface, (x0, y0, x1, y1)) in enumerate(
            self.world.puck_map.surface_rects()
        ):
            state[Object(f"patch{i}", KinRectangleType)] = _rect_features(
                x0, y0, x1, y1, SURFACE_RGB[surface], ZOrder.FLOOR
            )
        return state

    def _sample_initial_state(self) -> ObjectCentricState:
        puck, robot = self._start
        return self._create_state(
            np.asarray(robot, np.float64), np.asarray(puck, np.float64)
        )

    def _create_state(
        self, robot_pose: NDArray[np.float64], puck: NDArray[np.float64]
    ) -> ObjectCentricState:
        phys = self.config.physics
        state: dict[Object, dict[str, float]] = {}
        state[Object("robot", KinRobotType)] = {
            "x": robot_pose[0],
            "y": robot_pose[1],
            "theta": robot_pose[2],
            "vx_base": 0.0,
            "vy_base": 0.0,
            "omega_base": 0.0,
            "vx_arm": 0.0,
            "vy_arm": 0.0,
            "omega_arm": 0.0,
            "vx_gripper_l": 0.0,
            "vy_gripper_l": 0.0,
            "omega_gripper_l": 0.0,
            "vx_gripper_r": 0.0,
            "vy_gripper_r": 0.0,
            "omega_gripper_r": 0.0,
            "static": False,
            "base_radius": phys.robot_radius,
            # The paddle is drawn as the gripper base.
            "arm_joint": phys.paddle_offset - phys.paddle_thickness / 2,
            "arm_length": phys.paddle_offset,
            "gripper_base_width": phys.paddle_thickness,
            "gripper_base_height": phys.paddle_width,
            "finger_gap": 0.0,
            "finger_height": 0.01,
            "finger_width": 0.01,
        }
        state[Object("puck", PuckType)] = _circle_features(
            puck[0], puck[1], phys.puck_radius, PUCK_RGB, ZOrder.ALL, phys.puck_mass
        )
        goal = Object("goal", GoalMarkerType)
        state[goal] = _circle_features(
            float(self.world.goal[0]),
            float(self.world.goal[1]),
            phys.goal_radius,
            GOAL_RGB,
            ZOrder.SURFACE,
            0.0,
        )
        state[goal]["static"] = True
        for i, (hx, hy) in enumerate(self.world.holes):
            hole = Object(f"hole{i}", HoleMarkerType)
            state[hole] = _circle_features(
                hx, hy, phys.hole_radius, HOLE_RGB, ZOrder.SURFACE, 0.0
            )
            state[hole]["static"] = True
        return create_state_from_dict(state, Dynamic2DRobotEnvTypeFeatures)

    # ------------------------------------------------------------------ physics

    def _setup_physics_space(self) -> None:
        self.pymunk_space = pymunk.Space()
        self.pymunk_space.gravity = 0, 0
        self.pymunk_space.damping = self.config.damping
        self.pymunk_space.collision_slop = self.config.collision_slop
        self.robot = None

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        assert self.pymunk_space is not None
        phys = self.config.physics
        walls = pymunk.Body(body_type=pymunk.Body.STATIC)
        shapes = []
        for x0, y0, x1, y1 in self.world.puck_map.wall_rects():
            shape = pymunk.Poly(walls, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            shape.elasticity = 1.0  # times the puck's, which is the restitution
            shape.friction = 0.0
            shape.collision_type = _WALL_COLLISION_TYPE
            shapes.append(shape)
        self.pymunk_space.add(walls, *shapes)

        puck_obj = state.get_objects(PuckType)[0]
        moment = pymunk.moment_for_circle(phys.puck_mass, 0, phys.puck_radius)
        puck = pymunk.Body(phys.puck_mass, moment)
        puck.position = (state.get(puck_obj, "x"), state.get(puck_obj, "y"))
        puck.velocity = (state.get(puck_obj, "vx"), state.get(puck_obj, "vy"))
        puck_shape = pymunk.Circle(puck, phys.puck_radius)
        puck_shape.elasticity = phys.wall_restitution
        puck_shape.friction = 0.0
        puck_shape.collision_type = _PUCK_COLLISION_TYPE
        puck.velocity_func = self._floor_friction
        self.pymunk_space.add(puck, puck_shape)
        self._puck_body = puck
        self._state_obj_to_pymunk_body[puck_obj] = puck

        robot_obj = state.get_objects(KinRobotType)[0]
        self._robot_pose = np.array(
            [state.get(robot_obj, "x"), state.get(robot_obj, "y")]
            + [state.get(robot_obj, "theta")]
        )
        # Paddle frame: middle of its face, facing +x.
        paddle = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        half_w = phys.paddle_width / 2
        paddle_shape = pymunk.Poly(
            paddle,
            [
                (-phys.paddle_thickness, -half_w),
                (0.0, -half_w),
                (0.0, half_w),
                (-phys.paddle_thickness, half_w),
            ],
        )
        paddle_shape.elasticity = 0.0
        paddle_shape.friction = 0.0
        paddle_shape.collision_type = _PADDLE_COLLISION_TYPE
        self.pymunk_space.add(paddle, paddle_shape)
        self._paddle_body = paddle
        self._place_paddle()

        robot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        robot_shape = pymunk.Circle(robot, phys.robot_radius)
        robot_shape.elasticity = 1.0
        robot_shape.friction = 0.0
        robot_shape.collision_type = _ROBOT_COLLISION_TYPE
        self.pymunk_space.add(robot, robot_shape)
        self._robot_body = robot
        self._set_robot_pose(self._robot_pose)

    def _set_robot_pose(self, pose: NDArray[np.float64]) -> None:
        """Move the robot's base (not its paddle) to `pose` (x, y, heading)."""
        self._robot_pose = np.array(pose, dtype=np.float64)
        if self._robot_body is not None:
            assert self.pymunk_space is not None
            self._robot_body.position = (
                float(self._robot_pose[0]),
                float(self._robot_pose[1]),
            )
            self.pymunk_space.reindex_shapes_for_body(self._robot_body)

    def _floor_friction(
        self, body: pymunk.Body, gravity, damping: float, dt: float
    ) -> None:
        """Coulomb floor friction, replacing PyMunk's velocity update."""
        del gravity, damping
        velocity = body.velocity
        speed = velocity.length
        body.angular_velocity = 0.0
        if speed <= 0.0:
            return
        point = np.array([body.position.x, body.position.y])
        decel = self.world.friction_at(point, self._frictions) * GRAVITY * dt
        if speed <= decel:
            body.velocity = (0.0, 0.0)
        else:
            body.velocity = velocity * (1.0 - decel / speed)

    def _place_paddle(
        self,
        face: NDArray[np.float64] | None = None,
        angle: float | None = None,
    ) -> None:
        """Put the paddle face at `face`, facing `angle` (default: in front)."""
        assert self._paddle_body is not None
        heading = float(self._robot_pose[2])
        if face is None:
            reach = self.config.physics.paddle_offset
            face = self._robot_pose[:2] + reach * np.array(
                [math.cos(heading), math.sin(heading)]
            )
        self._paddle_stowed = False
        self._paddle_body.position = (float(face[0]), float(face[1]))
        self._paddle_body.angle = heading if angle is None else angle
        self._paddle_body.velocity = (0.0, 0.0)
        assert self.pymunk_space is not None
        self.pymunk_space.reindex_shapes_for_body(self._paddle_body)

    def _stow_paddle(self) -> None:
        """Tuck the paddle over the base, clear of the puck wherever the base is."""
        assert self._paddle_body is not None and self.pymunk_space is not None
        heading = float(self._robot_pose[2])
        face = self._robot_pose[:2] + _STOWED_REACH * np.array(
            [math.cos(heading), math.sin(heading)]
        )
        self._paddle_stowed = True
        self._paddle_body.position = (float(face[0]), float(face[1]))
        self._paddle_body.angle = heading
        self._paddle_body.velocity = (0.0, 0.0)
        self.pymunk_space.reindex_shapes_for_body(self._paddle_body)

    def _read_state_from_space(self) -> None:
        assert self._current_state is not None
        state = self._current_state.copy()
        puck_obj = state.get_objects(PuckType)[0]
        if self._puck_body is not None:
            state.set(puck_obj, "x", self._puck_body.position.x)
            state.set(puck_obj, "y", self._puck_body.position.y)
            state.set(puck_obj, "vx", self._puck_body.velocity.x)
            state.set(puck_obj, "vy", self._puck_body.velocity.y)
        robot_obj = state.get_objects(KinRobotType)[0]
        state.set(robot_obj, "x", self._robot_pose[0])
        state.set(robot_obj, "y", self._robot_pose[1])
        state.set(robot_obj, "theta", self._robot_pose[2])
        phys = self.config.physics
        reach = _STOWED_REACH if self._paddle_stowed else phys.paddle_offset
        state.set(robot_obj, "arm_joint", reach - phys.paddle_thickness / 2)
        state.set(robot_obj, "arm_length", reach)
        self._current_state = state

    # ------------------------------------------------------------------ reset/step

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        """Reset; `options` may fix "frictions", and the "puck" (x, y) and "robot"
        (x, y, heading) start."""
        opts, puck_map = options or {}, self.world.puck_map
        self._start = (
            opts.get("puck", puck_map.puck_start),
            opts.get("robot", (*puck_map.robot_start, 0.0)),
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
        return obs, info

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        assert self.pymunk_space is not None and self._puck_body is not None
        assert self._outcome is None, "Episode is over; call reset()"
        push = parse_push_action(action, self.config.physics.max_stance)
        angle, speed, stance = push.angle, push.speed, push.stance
        puck = np.array([self._puck_body.position.x, self._puck_body.position.y])
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

        stop = np.array([self._puck_body.position.x, self._puck_body.position.y])
        if info["fell_in_hole"]:
            self._outcome = "hole"
        elif self.world.in_goal(stop):
            self._outcome = "goal"
        info["goal_reached"] = self._outcome == "goal"
        self._last_info = info
        reward, terminated = self._get_reward_and_done()
        return self._get_obs(), reward, terminated, False, info

    def _strike(
        self,
        puck: NDArray[np.float64],
        angle: float,
        speed: float,
        stance: float | None = None,
    ) -> dict:
        """Drive behind the puck, push it to speed with the paddle, let it slide."""
        assert self.pymunk_space is not None
        assert self._puck_body is not None and self._paddle_body is not None
        phys = self.config.physics
        rng = self.np_random
        # The robot aims and accelerates imperfectly.
        actual_angle = angle + float(rng.normal(0.0, phys.angle_noise))
        actual_speed = max(
            speed * (1.0 + float(rng.normal(0.0, phys.speed_noise))), 0.0
        )

        # If the noisy heading is infeasible, stand as for the commanded one.
        pose = self.world.push_pose(puck, actual_angle, stance) or self.world.push_pose(
            puck, angle, stance
        )
        assert pose is not None
        station, heading = pose
        direction = np.array([math.cos(actual_angle), math.sin(actual_angle)])
        start_node = self.world.nav_node_of(self._robot_pose[:2])
        drive = self.world.drive_distance(start_node, station, puck)
        self._set_robot_pose(np.array([*(station - _PADDLE_GAP * direction), heading]))
        face = puck - (phys.puck_radius + _PADDLE_GAP) * direction
        self._place_paddle(face, actual_angle)
        # Dead paddle during the strike, so it does not bat the puck ahead.
        self._set_paddle_elasticity(0.0)
        base_from_paddle = self._robot_pose[:2] - face

        dt = 1.0 / self.config.sim_hz
        trajectory = [puck.copy()]
        fell = False
        release_speed = 0.0
        if actual_speed > 0:
            accel = actual_speed**2 / (2.0 * phys.accel_distance)
            num_accel = max(int(round(actual_speed / accel / dt)), 1)
            for k in range(num_accel):
                v = accel * (k + 1) * dt
                self._paddle_body.velocity = tuple(v * direction)
                self.pymunk_space.step(dt)
                base_xy = np.array(self._paddle_body.position) + base_from_paddle
                self._set_robot_pose(np.array([*base_xy, heading]))
                fell = fell or self._over_hole()
                trajectory.append(self._puck_position())
            self._paddle_body.velocity = (0.0, 0.0)
            release_speed = float(self._puck_body.velocity.length)
        # Release: the paddle stops square to the push; the puck bounces off it.
        self._paddle_body.velocity = (0.0, 0.0)
        self._set_paddle_elasticity(1.0)
        num_slide = int(self.config.max_slide_seconds / dt)
        for _ in range(num_slide):
            if fell or self._puck_body.velocity.length < self.config.stop_speed:
                break
            self.pymunk_space.step(dt)
            fell = self._over_hole()
            trajectory.append(self._puck_position())
        self._puck_body.velocity = (0.0, 0.0)
        stop = self._puck_position()
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

    def _set_paddle_elasticity(self, elasticity: float) -> None:
        assert self._paddle_body is not None
        for shape in self._paddle_body.shapes:
            shape.elasticity = elasticity

    def _puck_position(self) -> NDArray[np.float64]:
        assert self._puck_body is not None
        return np.array([self._puck_body.position.x, self._puck_body.position.y])

    def _over_hole(self) -> bool:
        return self.world.over_hole(self._puck_position())

    def _measure_position(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        noise = self.config.physics.position_obs_noise
        return point + self.np_random.normal(0.0, noise, size=2)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        return -1.0, self._outcome is not None

    def _get_info(self) -> dict:
        return dict(self._last_info)

    def get_action_from_gui_input(self, gui_input: dict[str, Any]) -> NDArray[Any]:
        raise NotImplementedError("PuckPushing has no GUI controls")


def _rect_features(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rgb: tuple[float, float, float],
    z_order: ZOrder,
) -> dict[str, float]:
    return {
        "x": (x0 + x1) / 2,
        "y": (y0 + y1) / 2,
        "theta": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "static": True,
        "held": False,
        "color_r": rgb[0],
        "color_g": rgb[1],
        "color_b": rgb[2],
        "z_order": z_order.value,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def _circle_features(
    x: float,
    y: float,
    radius: float,
    rgb: tuple[float, float, float],
    z_order: ZOrder,
    mass: float,
) -> dict[str, float]:
    return {
        "x": x,
        "y": y,
        "theta": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "static": False,
        "held": False,
        "color_r": rgb[0],
        "color_g": rgb[1],
        "color_b": rgb[2],
        "z_order": z_order.value,
        "radius": radius,
        "mass": mass,
    }
