"""Tasks for the limb environment, and the domain for experiments.

Each variant has a fixed grasp and base pose, chosen offline. A task starts with the
robot at that base, holding the limb at its start configuration, and asks for a goal
configuration. Tasks are drawn so that the robot can grasp the limb and hold it
along the straight line to the goal.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS, NUM_ROBOT_JOINTS
from kinder.envs.dynamic3d.limbrepositioning3d import create_variant_config
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, matrix_from_quat, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.motion_planning import (
    check_collisions_with_held_object,
    run_motion_planning,
)

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.base import Domain, TaskDistribution
from able.envs.limb.constraints import (
    CONTACT_PENETRATION_TOLERANCE,
    arm_in_collision,
    base_in_collision,
    get_clearance,
    get_human_collision_ids,
    get_resting_penetration,
    get_scene_collision_ids,
    limb_in_collision,
)
from able.envs.limb.human import LimbVariant, RangeOfMotionPrior, get_variant
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.envs.limb.sim import CoupledState, LimbSim, LimbSimSettings
from able.envs.simulated import SimulatedEnvironment

FIXED_SETUPS_PATH = Path(__file__).with_name("fixed_setups.json")
# The number of points on the line from start to goal where the grasp must hold.
NUM_REACH_CHECKS = 3


@dataclass(frozen=True)
class FixedSetup:
    """A variant's grasp, offset m up the forearm or shank from the wrist or ankle
    and rolled about it, and its base pose (x, y, rotation)."""

    variant: str
    offset: float
    roll: float
    base: tuple[float, float, float]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def base_pose(self) -> SE2Pose:
        """The base pose."""
        return SE2Pose(*self.base)


def load_fixed_setup(variant: str, path: Path = FIXED_SETUPS_PATH) -> FixedSetup:
    """The fixed setup of the variant."""
    setups = json.loads(path.read_text(encoding="utf-8"))
    if variant not in setups:
        raise KeyError(f"No fixed setup for {variant}")
    entry = setups[variant]
    return FixedSetup(**{**entry, "base": tuple(entry["base"])})


def _get_link_position(sim: LimbSim, name: str) -> NDArray[np.float64]:
    limb = sim.env.limb
    for joint in range(p.getNumJoints(limb.robot_id, physicsClientId=sim.client)):
        info = p.getJointInfo(limb.robot_id, joint, physicsClientId=sim.client)
        if info[12].decode() == name:
            state = p.getLinkState(limb.robot_id, joint, physicsClientId=sim.client)
            return np.asarray(state[4])
    raise ValueError(f"No link named {name}")


def get_wrist_grasp(sim: LimbSim, offset: float, roll: float) -> Pose:
    """The environment's grasp, moved offset m up the forearm or shank from the
    wrist or ankle and rolled about it, relative to the limb's end effector."""
    limb = sim.env.limb
    is_arm = "arm" in limb.get_name()
    wrist = _get_link_position(sim, "hand_x" if is_arm else "foot_x")
    elbow = _get_link_position(sim, "lower_arm" if is_arm else "lower_leg")
    target = wrist + offset * (elbow - wrist) / np.linalg.norm(elbow - wrist)
    end_effector = limb.get_end_effector_pose()
    local = matrix_from_quat(end_effector.orientation).T @ (
        target - np.asarray(end_effector.position)
    )
    spin = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, roll))
    return multiply_poses(
        Pose(tuple(local.tolist())), sim.env.scene.grasp_transform, spin
    )


def _with_fingers(arm_positions: Any) -> list[float]:
    """Arm joint positions followed by the six closed finger joints."""
    return list(arm_positions)[:NUM_ROBOT_JOINTS] + [0.0] * 6


def solve_grasp_ik(
    sim: LimbSim,
    grasp: Pose,
    limb_positions: NDArray[np.float64],
    seed_positions: list[float],
    resting_penetration: dict[int, float],
    rng: np.random.Generator,
    num_attempts: int = 12,
) -> list[float] | None:
    """Collision-free arm joints that hold the grasp at the limb configuration.

    The first attempt starts from the seed and the rest from random
    configurations. Leaves the limb at the configuration and the arm wherever the
    last attempt left it.
    """
    env = sim.env
    env.limb.set_joints(list(limb_positions), joint_velocities=[0.0] * NUM_LIMB_JOINTS)
    if limb_in_collision(sim, resting_penetration):
        return None
    target = multiply_poses(env.limb.get_end_effector_pose(), grasp)
    arm = env.robot.arm
    lower = np.clip(
        np.asarray(arm.joint_lower_limits[:NUM_ROBOT_JOINTS]), -np.pi, np.pi
    )
    upper = np.clip(
        np.asarray(arm.joint_upper_limits[:NUM_ROBOT_JOINTS]), -np.pi, np.pi
    )
    for attempt in range(num_attempts):
        seed = (
            seed_positions if attempt == 0 else _with_fingers(rng.uniform(lower, upper))
        )
        arm.set_joints(list(seed))
        try:
            solution = inverse_kinematics(arm, target, validate=True, set_joints=False)
        except InverseKinematicsError:
            continue
        arm.set_joints(list(solution))
        if not arm_in_collision(sim):
            return list(solution)
    return None


@dataclass(frozen=True)
class TaskBox:
    """How tasks are drawn, as on the two-link arm: start and goal configurations of
    the active joints are uniform within the person's true range, and far enough
    apart. With probability edge_probability, each active joint of the goal is at
    the limit its motion sets, where reaching reveals the limit. Beyond
    max_separation, MPC does not converge."""

    edge_probability: float = 0.5
    min_separation: float = 0.5
    max_separation: float = 1.5
    robot_jitter: float = 0.2
    max_start_draws: int = 40
    max_goal_draws: int = 60
    max_configuration_draws: int = 200


class LimbTaskDistribution(TaskDistribution[LimbTask]):
    """Tasks for a person that the fixed grasp and base can carry out.

    A draw picks a start configuration where the robot can grasp the limb, and
    approach the grasp from a jittered arm configuration without collisions. It
    then picks a goal far enough away that the grasp holds, and the robot can
    support the limb, at points along the straight line to it. The checks run in
    a simulator of their own.
    """

    def __init__(
        self,
        model: LimbModel,
        setup: FixedSetup,
        true_theta: NDArray[np.float64],
        box: TaskBox = TaskBox(),
    ) -> None:
        self._model = model
        self._setup = setup
        self._box = box
        self._sim: LimbSim | None = None
        config = create_variant_config(model.variant.name)
        self._nominal_limb = np.array(config.scene.limb_init_joint_positions)
        self._nominal_robot = np.array(config.robot_initial_joints)
        self._low, self._high = model.variant.get_joint_limits(true_theta)
        self._edges = model.variant.get_bounds(true_theta)
        self._grasp: Pose | None = None
        self._resting: dict[int, float] = {}

    @property
    def sim(self) -> LimbSim:
        """The simulator for the checks, built on first use."""
        if self._sim is None:
            self._sim = LimbSim(self._model.variant, self._model.settings)
            env = self._sim.env
            self._sim.release_grasp()
            env.limb.set_joints(list(self._nominal_limb))
            self._grasp = get_wrist_grasp(
                self._sim, self._setup.offset, self._setup.roll
            )
            self._resting = get_resting_penetration(self._sim)
        return self._sim

    def close(self) -> None:
        """Disconnect the simulator."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def sample(self, rng: np.random.Generator) -> LimbTask:
        sim, box = self.sim, self._box
        assert self._grasp is not None
        for _ in range(box.max_start_draws):
            limb_init = self.draw_limb_configuration(rng)
            robot_init = self._draw_robot_configuration(rng)
            start = self._approach(limb_init, robot_init, rng)
            if start is None:
                continue
            for _ in range(box.max_goal_draws):
                limb_goal = self.draw_limb_configuration(rng, box.edge_probability)
                separation = sim.get_limb_error(limb_init, limb_goal)
                if not box.min_separation <= separation <= box.max_separation:
                    continue
                if self._can_carry(limb_init, limb_goal, start, rng):
                    initial_state = self._model.get_initial_state(start, limb_goal)
                    return LimbTask(initial_state=initial_state, goal=limb_goal)
        raise RuntimeError("No reachable task found")

    def _in_furniture(self, limb_positions: NDArray[np.float64]) -> bool:
        sim = self.sim
        with sim.saved_state():
            sim.env.limb.set_joints(list(limb_positions))
            limb_id = sim.env.limb.robot_id
            return any(
                get_clearance(sim, limb_id, body) < -CONTACT_PENETRATION_TOLERANCE
                for body in get_scene_collision_ids(sim)
            )

    def draw_limb_configuration(
        self, rng: np.random.Generator, edge_probability: float = 0.0
    ) -> NDArray[np.float64]:
        """Active joints uniform within the person's range, or at the limit of their
        motion with edge_probability, clear of the furniture. The other joints keep
        the variant's values."""
        variant = self._model.variant
        for _ in range(self._box.max_configuration_draws):
            positions = self._nominal_limb.copy()
            for motion, joint in enumerate(variant.active_joints):
                positions[joint] = rng.uniform(self._low[joint], self._high[joint])
                if rng.random() < edge_probability:
                    positions[joint] = self._edges[motion]
            if not self._in_furniture(positions):
                return positions
        raise RuntimeError("No limb configuration clear of the furniture")

    def _draw_robot_configuration(self, rng: np.random.Generator) -> list[float]:
        """The variant's arm configuration, jittered, without self-collisions."""
        sim, jitter = self.sim, self._box.robot_jitter
        for _ in range(self._box.max_configuration_draws):
            positions = _with_fingers(
                self._nominal_robot + rng.uniform(-jitter, jitter, NUM_ROBOT_JOINTS)
            )
            with sim.saved_state():
                colliding = check_collisions_with_held_object(
                    sim.env.robot.arm, [], sim.client, None, None, positions
                )
            if not colliding:
                return positions
        raise RuntimeError("No arm configuration without self-collisions")

    def _approach(
        self,
        limb_init: NDArray[np.float64],
        robot_init: list[float],
        rng: np.random.Generator,
    ) -> CoupledState | None:
        """The grasped start state, if the robot at the fixed base can move from its
        start configuration to the grasp."""
        sim = self.sim
        assert self._grasp is not None
        env = sim.env
        with sim.saved_state():
            sim.release_grasp()
            env.robot.set_base(self._setup.base_pose)
            if base_in_collision(sim):
                return None
            joints = solve_grasp_ik(
                sim, self._grasp, limb_init, robot_init, self._resting, rng
            )
            if joints is None:
                return None
            obstacles = get_human_collision_ids(sim) + get_scene_collision_ids(sim)
            path = run_motion_planning(
                env.robot.arm,
                initial_positions=robot_init,
                target_positions=joints,
                collision_bodies=obstacles,
                seed=int(rng.integers(2**31)),
                physics_client_id=sim.client,
            )
            if path is None:
                return None
            env.robot.arm.set_joints(list(joints), joint_velocities=[0.0] * len(joints))
            env.limb.set_joints(
                list(limb_init), joint_velocities=[0.0] * NUM_LIMB_JOINTS
            )
            sim.engage_grasp()
            return sim.capture()

    def _can_carry(
        self,
        limb_init: NDArray[np.float64],
        limb_goal: NDArray[np.float64],
        start: CoupledState,
        rng: np.random.Generator,
    ) -> bool:
        """Whether the grasp holds, and the robot can support the limb, along the
        line from start to goal."""
        if self._in_furniture(limb_goal):
            return False
        sim = self.sim
        assert self._grasp is not None
        with sim.saved_state():
            sim.release_grasp()
            sim.env.robot.set_base(self._setup.base_pose)
            seed = list(start.robot_positions)
            for waypoint in np.linspace(limb_init, limb_goal, NUM_REACH_CHECKS):
                solution = solve_grasp_ik(
                    sim, self._grasp, waypoint, seed, self._resting, rng
                )
                if solution is None or sim.get_control_saturation() > 1.0:
                    return False
                seed = solution
        return True


LimbDomain = Domain[
    LimbState,
    NDArray[np.float64],
    LimbTask,
    NDArray[np.float64],
    LimbQuery,
    QueryOutcome,
]


def make_domain(
    seed: int,
    variant: str = "human-left-arm",
    horizon: int = 300,
    prior: RangeOfMotionPrior = RangeOfMotionPrior(),
    true_theta: NDArray[np.float64] | None = None,
    settings: LimbSimSettings = LimbSimSettings(),
) -> LimbDomain:
    """A limb domain for the variant, with a person drawn from the prior unless
    true_theta is given. Tasks lie within the person's range, and the robot's
    initial belief is the prior."""
    limb_variant: LimbVariant = get_variant(variant)
    num_motions = len(limb_variant.active_motions)
    rng = np.random.default_rng(seed)
    if true_theta is None:
        true_theta = prior.sample(num_motions, rng)
    model = LimbModel(limb_variant, settings, horizon=horizon)
    return Domain(
        env_model=model,
        task_distribution=LimbTaskDistribution(
            model, load_fixed_setup(variant), np.asarray(true_theta)
        ),
        initial_belief=RangeOfMotionBelief.from_prior(
            prior.mean, prior.stddev, prior.low, prior.high, num_motions
        ),
        real_env=SimulatedEnvironment(model, np.asarray(true_theta), rng),
    )
