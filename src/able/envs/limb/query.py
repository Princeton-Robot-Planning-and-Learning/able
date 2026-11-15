"""Queries about the person's range of motion.

The robot asks the person to move one motion toward a target near its limit. The
person moves it with their own joint torques (see person.py) at a comfortable
speed, toward the target or, if the target lies beyond their reach, to a random
short fall before their limit. Where they stop is the response. Each query runs in
its own PyBullet world with only the limb, without gravity, since the person holds
up their own limb.
"""

from dataclasses import dataclass

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS, PYBULLET_TIMESTEP
from kinder.envs.dynamic3d.limbs import create_human_limb
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose

from able.envs.limb.human import LimbVariant
from able.envs.limb.person import PersonModel


@dataclass(frozen=True)
class LimbQuery:
    """Move an active motion toward a target magnitude (rad)."""

    motion: int
    target: float


@dataclass(frozen=True)
class QueryOutcome:
    """Where the person stopped, and why.

    The answer is the magnitude of the final joint angle. If reached_target, the
    person reached the target before their limit; short_fall is how far short of
    their limit they would stop. If not settled, the joint never came to rest. A
    peak velocity at the joint's speed cap means the motion was not the person's
    own.
    """

    motion: int
    target: float
    answer: float
    reached_target: bool
    short_fall: float
    settled: bool
    peak_velocity: float


@dataclass(frozen=True)
class QueryModel:
    """How the person moves in a query.

    The person stops a short fall before their limit, drawn from a normal
    distribution truncated to (0, max_short_fall). They aim at the target or that
    point, whichever comes first, and move their aim toward it at speed; the joint
    follows the aim with the given stiffness and damping. The constants are
    placeholders, chosen so that the short fall matches the earlier model, which
    pushed with a constant effort against a spring near the limit.
    """

    short_fall_mean: float = 0.2
    short_fall_stddev: float = 0.05
    max_short_fall: float = 0.3
    stiffness: float = 50.0
    damping: float = 10.0
    speed: float = 0.5
    max_velocity: float = 2.0
    moving_velocity: float = 1e-2
    settle_velocity: float = 1e-3
    settle_steps: int = 100
    max_steps: int = 20000

    def __post_init__(self) -> None:
        if not 0.0 < self.short_fall_mean < self.max_short_fall:
            raise ValueError("short_fall_mean must lie in (0, max_short_fall)")

    @property
    def short_fall(self) -> float:
        """The mean distance short of the limit where the person stops (rad)."""
        return self.short_fall_mean

    @property
    def noise_stddev(self) -> float:
        """The standard deviation of where the person stops (rad)."""
        return self.short_fall_stddev

    def sample_short_fall(self, rng: np.random.Generator) -> float:
        """One query's short fall, redrawn until in (0, max_short_fall)."""
        while True:
            short_fall = float(rng.normal(self.short_fall_mean, self.short_fall_stddev))
            if 0.0 < short_fall < self.max_short_fall:
                return short_fall


class QueryLimb:
    """One person's limb, alone in a PyBullet world without gravity."""

    def __init__(
        self,
        variant: LimbVariant,
        theta: NDArray[np.float64],
        model: QueryModel = QueryModel(),
        person: PersonModel = PersonModel(),
    ) -> None:
        self.variant = variant
        self.model = model
        self.person = person
        self.limits = np.abs(variant.get_bounds(theta))
        self.joint_limits = variant.get_joint_limits(theta)
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
            # One combined call drops the damping. The person's torque damps the
            # joint instead.
            p.changeDynamics(
                self.limb.robot_id,
                joint,
                maxJointVelocity=model.max_velocity,
                physicsClientId=self.client,
            )
            p.changeDynamics(
                self.limb.robot_id, joint, jointDamping=0.0, physicsClientId=self.client
            )

    def ask(self, query: LimbQuery, rng: np.random.Generator) -> QueryOutcome:
        """Move from the neutral pose until the person stops at their aim."""
        model, robot_id = self.model, self.limb.robot_id
        index = self.variant.active_joints[query.motion]
        joint = self.limb.arm_joints[index]
        sign = self.variant.active_signs[query.motion]
        limit = float(self.limits[query.motion])
        short_fall = model.sample_short_fall(rng)
        reached_target = query.target <= limit - short_fall
        aim = query.target if reached_target else limit - short_fall
        limits = (
            self.joint_limits[0][index : index + 1],
            self.joint_limits[1][index : index + 1],
        )

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

        num_settled, peak, moving, intent = 0, 0.0, False, 0.0
        step_size = model.speed * PYBULLET_TIMESTEP
        for _ in range(model.max_steps):
            position, velocity = p.getJointState(
                robot_id, joint, physicsClientId=self.client
            )[:2]
            speed = abs(velocity)
            peak = max(peak, speed)
            moving = moving or speed > model.moving_velocity
            # The person's aim moves toward where they want to stop.
            intent = intent + float(np.clip(aim - intent, -step_size, step_size))
            effort = model.stiffness * (sign * intent - position)
            effort -= model.damping * velocity
            torque = self.person.get_torque(
                np.array([position]),
                np.array([velocity]),
                np.zeros(1),
                limits,
                self.variant.limb_name,
                effort=np.array([effort]),
            )
            p.setJointMotorControl2(
                robot_id,
                joint,
                p.TORQUE_CONTROL,
                force=float(torque[0]),
                physicsClientId=self.client,
            )
            p.stepSimulation(physicsClientId=self.client)
            # Pin the other joints; a motor cannot hold them at the gimbal
            # singularity.
            for other in held:
                p.resetJointState(
                    robot_id, other, 0.0, 0.0, physicsClientId=self.client
                )
            # Settling only counts once the joint has started moving and the
            # person's aim has arrived.
            num_settled = (
                num_settled + 1
                if moving and intent == aim and speed < model.settle_velocity
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
            short_fall=short_fall,
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

    def __init__(
        self,
        variant: LimbVariant,
        model: QueryModel = QueryModel(),
        person: PersonModel = PersonModel(),
    ) -> None:
        self.variant = variant
        self.model = model
        self.person = person
        self._key: bytes | None = None
        self._limb: QueryLimb | None = None

    def ask(
        self, theta: NDArray[np.float64], query: LimbQuery, rng: np.random.Generator
    ) -> QueryOutcome:
        """The response of the person with theta."""
        key = np.asarray(theta, dtype=np.float64).tobytes()
        if self._limb is None or key != self._key:
            self.close()
            self._limb = QueryLimb(self.variant, theta, self.model, self.person)
            self._key = key
        return self._limb.ask(query, rng)

    def close(self) -> None:
        """Disconnect the current limb's client, if any."""
        if self._limb is not None:
            self._limb.close()
        self._limb, self._key = None, None
