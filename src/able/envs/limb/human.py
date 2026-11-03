"""The person whose limb the robot moves.

The hidden parameters theta scale three motions of the limb (for example, shoulder
flexion) relative to the normal adult range of motion. Each motion bounds one joint
of the limb from one side. A variant is a scene (isolated, human, bed, or
wheelchair) and a limb (left or right arm or leg).
"""

from dataclasses import dataclass

import numpy as np
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS
from kinder.envs.dynamic3d.limbs import (
    DEFAULT_RANGE_OF_MOTION,
    RangeOfMotion,
    get_human_joint_limits,
)
from numpy.typing import NDArray

SCENES = ("isolated", "human", "bed", "wheelchair")
LIMBS = ("left-arm", "right-arm", "left-leg", "right-leg")
VARIANTS = tuple(f"{scene}-{limb}" for scene in SCENES for limb in LIMBS)

# The active motions of each limb: (motion, joint it bounds, sign). The sign is +1
# if the motion bounds the joint from above and -1 if from below.
_LIMB_MOTIONS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "human-right-arm": (
        ("shoulder_flexion", 0, -1),
        ("shoulder_abduction", 1, +1),
        ("elbow_flexion", 3, -1),
    ),
    "human-left-arm": (
        ("shoulder_flexion", 0, +1),
        ("shoulder_abduction", 1, +1),
        ("elbow_flexion", 3, +1),
    ),
    "human-right-leg": (
        ("hip_flexion", 0, -1),
        ("hip_abduction", 1, +1),
        ("knee_flexion", 3, +1),
    ),
    "human-left-leg": (
        ("hip_flexion", 0, -1),
        ("hip_abduction", 1, -1),
        ("knee_flexion", 3, +1),
    ),
}


@dataclass(frozen=True)
class LimbVariant:
    """One variant, and how theta maps to the limb's joint limits."""

    name: str
    limb_name: str
    active_motions: tuple[str, ...]
    active_joints: tuple[int, ...]
    active_signs: tuple[int, ...]

    @property
    def inactive_joints(self) -> tuple[int, ...]:
        """The limb joints that no active motion bounds."""
        return tuple(j for j in range(NUM_LIMB_JOINTS) if j not in self.active_joints)

    @property
    def default_magnitudes(self) -> NDArray[np.float64]:
        """The normal range of each active motion, in radians."""
        defaults = DEFAULT_RANGE_OF_MOTION.as_dict()
        return np.deg2rad([defaults[motion] for motion in self.active_motions])

    def get_range_of_motion(self, theta: NDArray[np.float64]) -> RangeOfMotion:
        """The full range of motion; inactive motions keep their defaults."""
        factors = dict(zip(self.active_motions, map(float, theta), strict=True))
        return DEFAULT_RANGE_OF_MOTION.scaled(factors)

    def get_joint_limits(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The (lower, upper) limit of each limb joint, in radians."""
        lower, upper = get_human_joint_limits(
            self.limb_name, self.get_range_of_motion(theta)
        )
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def get_bounds(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """The signed joint angle at the limit of each active motion."""
        theta = np.asarray(theta, dtype=np.float64)
        return theta * self.default_magnitudes * np.array(self.active_signs)


def get_variant(name: str) -> LimbVariant:
    """The variant with the name, such as "bed-right-leg"."""
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; expected one of {VARIANTS}")
    limb_name = f"human-{name.split('-', 1)[1]}"
    motions = _LIMB_MOTIONS[limb_name]
    return LimbVariant(
        name=name,
        limb_name=limb_name,
        active_motions=tuple(m for m, _, _ in motions),
        active_joints=tuple(j for _, j, _ in motions),
        active_signs=tuple(s for _, _, s in motions),
    )


@dataclass(frozen=True)
class RangeOfMotionPrior:
    """The population that theta is drawn from: each scale is normal, truncated to
    [low, high]."""

    mean: float = 0.9
    stddev: float = 0.15
    low: float = 0.7
    high: float = 1.2

    def sample(self, num_motions: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """One scale per motion, redrawing any outside [low, high]."""
        draws = np.empty(num_motions)
        for i in range(num_motions):
            while True:
                value = rng.normal(self.mean, self.stddev)
                if self.low <= value <= self.high:
                    draws[i] = value
                    break
        return draws
