"""The person's own joint torques.

Every joint of the person's limb is a motor that stands in for their muscles, as
Sprout's arm motors do on hardware. Each simulator step, joint j applies

    tau_j = clip(support * g_j(q) + effort_j - damping * qdot_j + W_j(q_j), +-max)

where g is the torque that holds the limb up against gravity, effort is where the
person is trying to move (none while the robot repositions the limb, since the
person relaxes), and W is a stiff wall past the person's limits. The limits are the
person's active range of motion, so the range the person reaches in a query is the
range the robot must keep to.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

JointLimits = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class PersonModel:
    """How the person's joints act.

    The torque limits are placeholders for Sprout's (N m); legs are stronger than arms.
    """

    support: float = 1.0
    damping: float = 0.0
    wall_stiffness: float = 400.0
    arm_max_torque: float = 50.0
    leg_max_torque: float = 100.0

    def get_max_torque(self, limb_name: str) -> float:
        """The torque limit of each joint of the limb."""
        return self.leg_max_torque if "leg" in limb_name else self.arm_max_torque

    def get_support_torque(
        self, gravity: NDArray[np.float64], limb_name: str
    ) -> NDArray[np.float64]:
        """The torque with which the person holds up their own limb."""
        max_torque = self.get_max_torque(limb_name)
        return np.clip(self.support * gravity, -max_torque, max_torque)

    def get_wall_torque(
        self, positions: NDArray[np.float64], limits: JointLimits
    ) -> NDArray[np.float64]:
        """The torque that pushes each joint back inside the limits; zero inside."""
        lower, upper = limits
        below = np.maximum(lower - positions, 0.0)
        above = np.maximum(positions - upper, 0.0)
        return self.wall_stiffness * (below - above)

    def get_torque(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        gravity: NDArray[np.float64],
        limits: JointLimits | None,
        limb_name: str,
        effort: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """The torque at each joint: support, effort, damping, and the walls of the
        limits, if given, within the torque limit."""
        positions = np.asarray(positions, dtype=np.float64)
        torque = self.support * np.asarray(gravity, dtype=np.float64)
        torque = torque - self.damping * np.asarray(velocities, dtype=np.float64)
        if effort is not None:
            torque = torque + effort
        if limits is not None:
            torque = torque + self.get_wall_torque(positions, limits)
        max_torque = self.get_max_torque(limb_name)
        return np.clip(torque, -max_torque, max_torque)
