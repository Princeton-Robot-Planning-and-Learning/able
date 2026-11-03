"""Queries about the person's range of motion.

The robot asks the person to push one motion toward a target near its limit. The
person pushes with a constant effort, the end of their range resists, and where they
stop is the response. Each query runs in its own PyBullet world with only the limb.
"""

from dataclasses import dataclass

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS, PYBULLET_TIMESTEP
from kinder.envs.dynamic3d.limbs import create_human_limb
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose

from able.envs.limb.human import LimbVariant


@dataclass(frozen=True)
class LimbQuery:
    """Push an active motion toward a target magnitude (rad)."""

    motion: int
    target: float


@dataclass(frozen=True)
class QueryOutcome:
    """Where the person stopped, and why.

    The answer is the magnitude of the final joint angle. If reached_target, the
    person reached the target before their limit. If not settled, the joint never
    came to rest. A peak velocity at the joint's speed cap means the push was not
    passive.
    """

    motion: int
    target: float
    answer: float
    reached_target: bool
    effort: float
    settled: bool
    peak_velocity: float


@dataclass(frozen=True)
class QueryModel:
    """How the person pushes and how the end of their range resists.

    Within zone_width of the limit, a spring of the given stiffness resists, and past
    the limit a stiffer wall does. The effort is drawn from a normal distribution and
    stays below stiffness * zone_width, so the person stops short of the limit. The
    constants are placeholders.
    """

    zone_width: float = 0.3
    stiffness: float = 40.0
    wall_stiffness: float = 400.0
    effort_mean: float = 4.0
    effort_stddev: float = 2.0
    # Overdamps the spring; below about 8, the joint overshoots the limit.
    joint_damping: float = 15.0
    max_velocity: float = 2.0
    moving_velocity: float = 1e-2
    settle_velocity: float = 1e-3
    settle_steps: int = 100
    max_steps: int = 150000

    def __post_init__(self) -> None:
        if self.effort_mean >= self.stiffness * self.zone_width:
            raise ValueError("effort_mean must be below stiffness * zone_width")

    @property
    def short_fall(self) -> float:
        """The mean distance short of the limit where the person stops (rad)."""
        return self.zone_width - self.effort_mean / self.stiffness

    @property
    def noise_stddev(self) -> float:
        """The standard deviation of where the person stops (rad)."""
        return self.effort_stddev / self.stiffness

    def sample_effort(self, rng: np.random.Generator) -> float:
        """One push's effort, redrawn until in (0, stiffness * zone_width)."""
        ceiling = self.stiffness * self.zone_width
        while True:
            effort = float(rng.normal(self.effort_mean, self.effort_stddev))
            if 0.0 < effort < ceiling:
                return effort

    def get_restoring_torque(self, reach: float, limit: float) -> float:
        """The end-of-range torque, where the person pushes in the positive
        direction."""
        if reach <= limit - self.zone_width:
            return 0.0
        if reach < limit:
            return -self.stiffness * (reach - (limit - self.zone_width))
        return -self.stiffness * self.zone_width - self.wall_stiffness * (reach - limit)


class QueryLimb:
    """One person's limb, alone in a PyBullet world without gravity."""

    def __init__(
        self,
        variant: LimbVariant,
        theta: NDArray[np.float64],
        model: QueryModel = QueryModel(),
    ) -> None:
        self.variant = variant
        self.model = model
        self.limits = np.abs(variant.get_bounds(theta))
        self.client = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(
            fixedTimeStep=PYBULLET_TIMESTEP, physicsClientId=self.client
        )
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client)
        self.limb = create_human_limb(
            variant.limb_name,
            Pose.identity(),
            [0.0] * NUM_LIMB_JOINTS,
            "none",
            "none",
            self.client,
        )
        joints = self.limb.arm_joints
        # Free the joints of their default motors so torques act directly.
        p.setJointMotorControlArray(
            self.limb.robot_id,
            joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(joints)),
            physicsClientId=self.client,
        )
        for joint in joints:
            # One combined call drops the damping.
            p.changeDynamics(
                self.limb.robot_id,
                joint,
                maxJointVelocity=model.max_velocity,
                physicsClientId=self.client,
            )
            p.changeDynamics(
                self.limb.robot_id,
                joint,
                jointDamping=model.joint_damping,
                physicsClientId=self.client,
            )

    def ask(self, query: LimbQuery, rng: np.random.Generator) -> QueryOutcome:
        """Push from the neutral pose until the person stops or reaches the
        target."""
        model, robot_id = self.model, self.limb.robot_id
        joint = self.limb.arm_joints[self.variant.active_joints[query.motion]]
        sign = self.variant.active_signs[query.motion]
        limit = float(self.limits[query.motion])
        effort = model.sample_effort(rng)

        self.limb.set_joints(
            [0.0] * NUM_LIMB_JOINTS, joint_velocities=[0.0] * NUM_LIMB_JOINTS
        )
        held = [j for j in self.limb.arm_joints if j != joint]
        p.setJointMotorControlArray(
            robot_id,
            held,
            p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(held),
            forces=[1e4] * len(held),
            physicsClientId=self.client,
        )
        # Release a motor left on by an earlier query.
        p.setJointMotorControl2(
            robot_id, joint, p.VELOCITY_CONTROL, force=0.0, physicsClientId=self.client
        )

        num_settled, peak, moving, reached_target = 0, 0.0, False, False
        for _ in range(model.max_steps):
            position, velocity = p.getJointState(
                robot_id, joint, physicsClientId=self.client
            )[:2]
            reach, velocity = sign * position, sign * velocity
            if reach >= query.target:
                reached_target = True
                break
            peak = max(peak, abs(velocity))
            moving = moving or abs(velocity) > model.moving_velocity
            torque = effort + model.get_restoring_torque(reach, limit)
            p.setJointMotorControl2(
                robot_id,
                joint,
                p.TORQUE_CONTROL,
                force=sign * torque,
                physicsClientId=self.client,
            )
            p.stepSimulation(physicsClientId=self.client)
            # Pin the other joints; a motor cannot hold them at the gimbal
            # singularity.
            for other in held:
                p.resetJointState(
                    robot_id, other, 0.0, 0.0, physicsClientId=self.client
                )
            # Settling only counts once the joint has started moving.
            num_settled = (
                num_settled + 1
                if moving and abs(velocity) < model.settle_velocity
                else 0
            )
            if num_settled >= model.settle_steps:
                break

        final = p.getJointState(robot_id, joint, physicsClientId=self.client)[0]
        return QueryOutcome(
            motion=query.motion,
            target=query.target,
            answer=abs(final),
            reached_target=reached_target,
            effort=effort,
            settled=moving and num_settled >= model.settle_steps,
            peak_velocity=peak,
        )

    def close(self) -> None:
        """Disconnect the PyBullet client."""
        if p.isConnected(physicsClientId=self.client):
            p.disconnect(physicsClientId=self.client)


class QuerySimulator:
    """Answers queries for one person at a time, rebuilding the limb when the
    person changes."""

    def __init__(self, variant: LimbVariant, model: QueryModel = QueryModel()) -> None:
        self.variant = variant
        self.model = model
        self._key: bytes | None = None
        self._limb: QueryLimb | None = None

    def ask(
        self, theta: NDArray[np.float64], query: LimbQuery, rng: np.random.Generator
    ) -> QueryOutcome:
        """The response of the person with theta."""
        key = np.asarray(theta, dtype=np.float64).tobytes()
        if self._limb is None or key != self._key:
            self.close()
            self._limb = QueryLimb(self.variant, theta, self.model)
            self._key = key
        return self._limb.ask(query, rng)

    def close(self) -> None:
        """Disconnect the current limb's client, if any."""
        if self._limb is not None:
            self._limb.close()
        self._limb, self._key = None, None
