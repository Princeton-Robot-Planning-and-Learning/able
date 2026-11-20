"""A person with good days and bad days: the robot moves the limb of one person
whose range of motion changes from day to day.

On a bad day, some motions are stiff, and their range of motion shrinks. Each
mode, a set of stiff motions, is a site (see envs/multi_site.py) with its own
range-of-motion scales, and the robot keeps a belief about each mode's scales.
The day's mode follows a Markov chain, and the robot sleeps knowing only the
previous day's mode.

A therapist sets each day's routine of exercises. Each exercise moves one
motion toward its limit and keeps the other joints where every mode allows
them, so an exercise depends only on how far its own motion reaches. Modes in
which a motion is equally stiff therefore share that motion's exercises.
"""

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
from numpy.typing import NDArray

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.beliefs.sites import SiteBelief
from able.envs.base import Domain
from able.envs.limb.human import RangeOfMotionPrior, get_variant
from able.envs.limb.model import LimbModel, LimbTask
from able.envs.limb.tasks import (
    FixedSetup,
    LimbTaskDistribution,
    TaskBox,
    load_fixed_setup,
    make_domain,
)
from able.envs.multi_site import MarkovSiteTaskDistribution, MultiSiteModel
from able.envs.simulated import SimulatedEnvironment

LimbModeDomain = Domain[Any, Any, Any, Any, Any, Any]

# The modes by default: a good day, stiff shoulders, a stiff elbow, and a bad
# day, as indices into the left arm's motions (shoulder flexion, shoulder
# abduction, elbow flexion).
DEFAULT_STIFF_MOTIONS: tuple[tuple[int, ...], ...] = ((), (0, 1), (2,), (0, 1, 2))


class MotionExerciseDistribution(LimbTaskDistribution):
    """Exercises that each move one motion toward its limit, with the other
    active joints where every mode allows them.

    An exercise of a motion at a scale moves the motion's joint to a goal that
    leans toward goal_margin short of the limit at that scale, as goals lean
    toward limits in TaskBox. The margin keeps goals where a belief about the
    scale that is a little unsure still allows them. Each other active joint
    lies at a uniform fraction, up to hold_fraction, of the way from the far
    end of its range to the band short of its limit at shared_scales, where the
    person starts to resist. With shared_scales at most every mode's scales,
    those joints are where every mode allows them, so an exercise depends only
    on the scale of its motion and the random numbers it draws.

    With hold_others, the other joints stay still, and the motion's joint
    starts min_separation to max_separation short of its goal. Otherwise every
    active joint moves: the start is drawn as the goal is, but uniformly along
    the motion's range, and start and goal lie min_separation to
    max_separation apart, as in LimbTaskDistribution.
    """

    def __init__(
        self,
        model: LimbModel,
        setup: FixedSetup,
        shared_scales: NDArray[np.float64],
        box: TaskBox = TaskBox(),
        hold_fraction: float = 0.8,
        goal_margin: float = 0.0,
        hold_others: bool = True,
    ) -> None:
        super().__init__(model, setup, shared_scales, box)
        self._shared_scales = np.asarray(shared_scales, dtype=np.float64)
        self._hold_fraction = hold_fraction
        self._goal_margin = goal_margin
        self._hold_others = hold_others

    def sample(self, rng: np.random.Generator) -> LimbTask:
        """An exercise of a random motion at the shared scales."""
        motion = int(rng.integers(len(self._shared_scales)))
        return self.sample_exercise(motion, float(self._shared_scales[motion]), rng)

    def _get_ends(
        self, joint: int, sign: int, low: NDArray[np.float64], high: NDArray[np.float64]
    ) -> tuple[float, float]:
        """The far end of the joint's range and the limit of its motion."""
        return (low[joint], high[joint]) if sign > 0 else (high[joint], low[joint])

    def _draw_configuration(
        self, motion: int, scale: float, rng: np.random.Generator, is_goal: bool
    ) -> NDArray[np.float64]:
        """A configuration whose other active joints are where every mode allows
        them, and whose motion's joint leans toward goal_margin short of its
        limit at the scale if is_goal, and is uniform along its range if not."""
        box, variant = self._box, self._model.variant
        band = self._model.settings.person.band_width
        positions = self._nominal_limb.copy()
        for i, (joint, sign) in enumerate(
            zip(variant.active_joints, variant.active_signs, strict=True)
        ):
            if i == motion:
                continue
            far, limit = self._get_ends(joint, sign, self._low, self._high)
            limit -= sign * band
            fraction = self._hold_fraction * rng.uniform()
            positions[joint] = far + fraction * (limit - far)
        scales = self._shared_scales.copy()
        scales[motion] = scale
        low, high = variant.get_joint_limits(scales)
        joint, sign = variant.active_joints[motion], variant.active_signs[motion]
        far, limit = self._get_ends(joint, sign, low, high)
        if is_goal:
            limit -= sign * self._goal_margin
            fraction = box.max_goal_fraction * rng.beta(box.goal_lean, 1.0)
        else:
            fraction = rng.uniform()
        positions[joint] = far + fraction * (limit - far)
        return positions

    def sample_exercise(
        self, motion: int, scale: float, rng: np.random.Generator
    ) -> LimbTask:
        """An exercise of the motion, whose range of motion is the scale."""
        box, variant = self._box, self._model.variant
        scales = self._shared_scales.copy()
        scales[motion] = scale
        low, high = variant.get_joint_limits(scales)
        joint, sign = variant.active_joints[motion], variant.active_signs[motion]
        for _ in range(box.max_start_draws):
            if self._hold_others:
                goal = self._draw_configuration(motion, scale, rng, is_goal=True)
                start = goal.copy()
                separation = rng.uniform(box.min_separation, box.max_separation)
                start[joint] -= sign * separation
                goals = [goal]
            else:
                start = self._draw_configuration(motion, scale, rng, is_goal=False)
                goals = [
                    self._draw_configuration(motion, scale, rng, is_goal=True)
                    for _ in range(box.max_goal_draws)
                ]
            robot_init = self._draw_robot_configuration(rng)
            if not low[joint] <= start[joint] <= high[joint]:
                continue
            if self._in_furniture(start):
                continue
            coupled = self._approach(start, robot_init, rng)
            if coupled is None:
                continue
            for goal in goals:
                separation = self.sim.get_limb_error(start, goal)
                if (
                    not box.min_separation - 1e-9
                    <= separation
                    <= box.max_separation + 1e-9
                ):
                    continue
                if self._can_carry(start, goal, coupled, rng):
                    initial_state = self._model.get_initial_state(coupled, goal)
                    return LimbTask(initial_state=initial_state, goal=goal)
        raise RuntimeError(f"No reachable exercise of motion {motion}")


@dataclass(frozen=True)
class ExerciseBlocks:
    """Draws a block of exercises for a key (motion, whether it is stiff): num
    exercises of the motion at its good or stiff scale."""

    exercises: MotionExerciseDistribution
    good: NDArray[np.float64]
    stiff: NDArray[np.float64]
    num: int

    def __call__(self, key: Hashable, rng: np.random.Generator) -> list[LimbTask]:
        assert isinstance(key, tuple)
        motion, is_stiff = int(key[0]), bool(key[1])
        scale = float(self.stiff[motion] if is_stiff else self.good[motion])
        return [
            self.exercises.sample_exercise(motion, scale, rng) for _ in range(self.num)
        ]


def make_transition(
    num_modes: int, stay: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """A transition matrix that stays in each mode with probability stay and
    otherwise moves to another mode, with chances drawn uniformly from the
    simplex for each row."""
    transition = np.zeros((num_modes, num_modes))
    for mode in range(num_modes):
        others = [m for m in range(num_modes) if m != mode]
        transition[mode, mode] = stay
        transition[mode, others] = (1.0 - stay) * rng.dirichlet(np.ones(len(others)))
    return transition


def make_mode_domain(
    seed: int,
    stiff_motions: Sequence[Sequence[int]] = DEFAULT_STIFF_MOTIONS,
    good_low: float = 0.9,
    good_high: float = 1.1,
    stiffness: float = 0.2,
    stay: float = 0.5,
    exercises_per_motion: int = 1,
    revision_rate: float = 0.0,
    hold_fraction: float = 0.8,
    goal_margin: float = 0.3,
    hold_others: bool = False,
    survey_error: float = 0.02,
    survey_stddev: float = 0.04,
    variant: str = "human-left-arm",
    horizon: int = 300,
    min_separation: float = TaskBox.min_separation,
    max_separation: float = 0.8,
    goal_lean: float = TaskBox.goal_lean,
    prior: RangeOfMotionPrior = RangeOfMotionPrior(),
    band_width: float | None = None,
    torque_noise: float = 1.0,
) -> LimbModeDomain:
    """A domain with one mode per entry of stiff_motions, the motions that are
    stiff in that mode.

    The person's scale of each motion on a good day is uniform in [good_low,
    good_high], and stiffness less where the motion is stiff. The day's mode
    follows make_transition(stay). Exercises are drawn as in
    MotionExerciseDistribution, with hold_fraction, goal_margin (rad), and
    hold_others. Each
    motion has a block of
    exercises_per_motion exercises for each of its two scales, and a mode's
    routine is the block of each motion at the mode's scale. Each new day
    revises each block with probability revision_rate. The robot starts with a
    survey of each mode: its belief about each scale is the prior's truncated
    Gaussian with standard deviation survey_stddev, centered within
    survey_error of the truth.
    """
    rng = np.random.default_rng(seed)
    base = make_domain(
        seed,
        variant=variant,
        horizon=horizon,
        min_separation=min_separation,
        max_separation=max_separation,
        goal_lean=goal_lean,
        prior=prior,
        band_width=band_width,
        torque_noise=torque_noise,
    )
    model = base.env_model
    assert isinstance(model, LimbModel)
    num_motions = len(get_variant(variant).active_motions)
    _, _, low, high = prior.get_parameters(num_motions).T
    good = rng.uniform(good_low, good_high, num_motions)
    stiff = good - stiffness
    assert np.all(stiff >= low) and np.all(good <= high), "Scales leave the prior"
    is_stiff = np.array(
        [[m in motions for m in range(num_motions)] for motions in stiff_motions]
    )
    thetas = np.where(is_stiff, stiff, good)
    exercises = MotionExerciseDistribution(
        model,
        load_fixed_setup(variant),
        thetas.min(axis=0),
        TaskBox(
            min_separation=min_separation,
            max_separation=max_separation,
            goal_lean=goal_lean,
        ),
        hold_fraction,
        goal_margin,
        hold_others,
    )

    site_blocks = [
        [(m, bool(is_stiff[mode, m])) for m in range(num_motions)]
        for mode in range(len(stiff_motions))
    ]
    transition = make_transition(len(stiff_motions), stay, rng)
    draw_block = ExerciseBlocks(exercises, good, stiff, exercises_per_motion)
    routines = MarkovSiteTaskDistribution(
        site_blocks, draw_block, transition, rng, revision_rate
    )
    centers = thetas + rng.uniform(-survey_error, survey_error, thetas.shape)
    surveys = [
        RangeOfMotionBelief.from_prior(
            np.clip(center, low, high), survey_stddev, low, high, num_motions
        )
        for center in centers
    ]
    site_model = MultiSiteModel(model)
    return Domain(
        env_model=site_model,
        task_distribution=routines,
        initial_belief=SiteBelief(surveys),
        real_env=SimulatedEnvironment(site_model, thetas, rng),
    )


def get_mode_scales(domain: LimbModeDomain) -> NDArray[np.float64]:
    """Each mode's true range-of-motion scales, one row per mode."""
    return np.asarray(domain.real_env.hidden_params)  # type: ignore[attr-defined]


def close_mode_domain(domain: LimbModeDomain) -> None:
    """Disconnect the simulators of the domain's limb model and exercises."""
    model: Any = domain.env_model
    routines: Any = domain.task_distribution
    model.inner.close()
    routines.draw_block.exercises.close()
