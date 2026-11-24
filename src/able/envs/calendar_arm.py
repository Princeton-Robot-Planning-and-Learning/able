"""A two-link arm whose person is stiffer on some days than on others, following a
calendar that the robot knows.

The hidden parameters are one person's joint limits today, [lo1, hi1, lo2, hi2].
Each day, each joint has a stiffness, and its two limits move inward by it. The
calendar of stiffnesses is known, as when a person is stiffer at weekends, when no
therapist comes, or loosens through a week of therapy.

The person practices a routine of exercises. An exercise is a fraction of each
joint's range, so its goal moves with the day's limits, and a plan for one day's
limits works less well on a day whose limits differ more.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import Domain, FiniteTaskDistribution, TaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import ArmTask, JointQuery, TwoLinkArmModel
from able.structs import Task

CALENDAR_KINDS = ("week", "uneven", "steady")

# Which way each limit moves as its joint gets stiffer.
_INWARD = np.array([1.0, -1.0, 1.0, -1.0])

CalendarArmDomain = Domain[
    NDArray[np.float64],
    NDArray[np.float64],
    ArmTask,
    NDArray[np.float64],
    JointQuery,
    bool,
]


@dataclass(frozen=True, eq=False)
class ExerciseTask(Task[NDArray[np.float64], NDArray[np.float64]]):
    """A task to do one exercise of the person's routine, starting from q = 0."""

    exercise: int


@dataclass(frozen=True)
class Exercise:
    """A target for each joint, a fraction of the way from 0 to its upper limit
    if uppers[j] and to its lower limit otherwise."""

    uppers: tuple[bool, bool]
    fractions: tuple[float, float]

    def get_joints(self, limits: NDArray[np.float64]) -> NDArray[np.float64]:
        """The target joint angles under these limits."""
        return np.array(
            [
                fraction * limits[2 * j + 1 if upper else 2 * j]
                for j, (upper, fraction) in enumerate(zip(self.uppers, self.fractions))
            ]
        )


class StiffnessCalendar:
    """Each joint's stiffness on each day, with shape (num_days, 2). Days past the
    last repeat it."""

    def __init__(self, stiffness: NDArray[np.float64]) -> None:
        stiffness = np.array(stiffness, dtype=np.float64)
        assert stiffness.ndim == 2 and stiffness.shape[1] == 2, "Need (num_days, 2)"
        assert np.all(stiffness >= 0.0), "Stiffness is nonnegative"
        stiffness.setflags(write=False)
        self._stiffness = stiffness

    @property
    def stiffness(self) -> NDArray[np.float64]:
        """The read-only (num_days, 2) array of stiffnesses."""
        return self._stiffness

    def get_shift(self, day: int) -> NDArray[np.float64]:
        """How far each of the four limits has moved from its loosest position on a
        day."""
        assert day >= 0, "Days count from 0"
        joints = self._stiffness[min(day, len(self._stiffness) - 1)]
        return np.repeat(joints, 2) * _INWARD


def sample_calendar(
    kind: str,
    num_days: int,
    step: float,
    max_stiffness: float,
    rng: np.random.Generator,
) -> StiffnessCalendar:
    """A random calendar.

    In a "week", the stiffness moves by step each day in one direction for five
    weekdays, then sits at one place far from them for the two days of the
    weekend. Each week is in a new place. In "uneven", regimes of 1, 2, 3, and 6
    days come in a random order, each in a new place, and the stiffness moves by
    step each day within a regime. In "steady", the stiffness moves by step each
    day, turning around at the ends of its range.
    """
    assert kind in CALENDAR_KINDS, f"kind must be one of {CALENDAR_KINDS}"
    assert 0.0 < step and 0.0 < max_stiffness, "Need positive step and range"
    days: list[NDArray[np.float64]] = []
    if kind == "steady":
        place, direction = np.zeros(2), np.array([1.0, 0.35])
        while len(days) < num_days:
            days.append(place.copy())
            moved = place + step * direction
            direction = np.where(
                (moved < 0.0) | (moved > max_stiffness), -direction, direction
            )
            place = np.clip(place + step * direction, 0.0, max_stiffness)
    while len(days) < num_days:
        if kind == "week":
            weekdays = _sample_walk(5, step, max_stiffness, rng)
            weekend = weekdays[2]
            while np.abs(weekend - weekdays[2]).max() < 2.0 * step + 0.12:
                weekend = rng.uniform(0.0, max_stiffness, size=2)
            days += weekdays + [weekend, weekend]
        else:
            for length in rng.permutation([1, 2, 3, 6]):
                days += _sample_walk(int(length), step, max_stiffness, rng)
    return StiffnessCalendar(np.array(days[:num_days]))


def _sample_walk(
    length: int, step: float, max_stiffness: float, rng: np.random.Generator
) -> list[NDArray[np.float64]]:
    """Places length days apart in one random direction, all within range."""
    while True:
        start = rng.uniform(0.0, max_stiffness, size=2)
        directions = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.float64)
        direction = directions[rng.integers(4)] * rng.choice([-1.0, 1.0])
        places = [start + step * k * direction for k in range(length)]
        if all(np.all(p >= 0.0) and np.all(p <= max_stiffness) for p in places):
            return places


class CalendarTaskDistribution(FiniteTaskDistribution[ArmTask]):
    """The person's routine, whose tasks are one per exercise under today's
    limits, which get_true_limits() gives the harness. The robot knows the routine,
    the calendar, and drift, the most that each joint's stiffness can move each
    night beyond the calendar, but not the limits, so it uses given() and
    forecast()."""

    def __init__(
        self,
        model: TwoLinkArmModel,
        exercises: Sequence[Exercise],
        calendar: StiffnessCalendar,
        get_true_limits: Callable[[], NDArray[np.float64]],
        drift: float = 0.0,
    ) -> None:
        self._model = model
        self._exercises = tuple(exercises)
        self._calendar = calendar
        self._get_true_limits = get_true_limits
        self._drift = drift
        self._day = 0
        super().__init__(self._make_tasks(get_true_limits()))

    def _make_tasks(self, limits: NDArray[np.float64]) -> list[ArmTask]:
        return [
            ExerciseTask(
                initial_state=np.zeros(2),
                goal=self._model.forward_kinematics(e.get_joints(limits)),
                exercise=i,
            )
            for i, e in enumerate(self._exercises)
        ]

    @property
    def day(self) -> int:
        """The current day, counted from 0."""
        return self._day

    @property
    def num_exercises(self) -> int:
        """The exercises in the routine."""
        return len(self._exercises)

    @property
    def depends_on_hidden_params(self) -> bool:
        return True

    def given(self, hidden_params: NDArray[np.float64]) -> TaskDistribution[ArmTask]:
        """The routine's tasks under these limits."""
        return FiniteTaskDistribution(self._make_tasks(hidden_params))

    def get_task(self, exercise: int, limits: NDArray[np.float64]) -> ArmTask:
        """The task to do an exercise under these limits."""
        return self._make_tasks(limits)[exercise]

    def start_day(self, rng: np.random.Generator) -> None:
        self._day += 1
        super().__init__(self._make_tasks(self._get_true_limits()))

    def forecast(
        self, belief: Belief[NDArray[np.float64]], days_ahead: int
    ) -> IntervalBelief:
        """What a belief about today's limits becomes days_ahead days from now, if
        nothing more is learned: it moves with the calendar and widens by the drift
        of every night."""
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        shift = self._calendar.get_shift(
            self._day + days_ahead
        ) - self._calendar.get_shift(self._day)
        widening = np.array([-self._drift, self._drift]) * days_ahead
        return IntervalBelief(belief.intervals + shift[:, None] + widening)

    def predict_today(self, belief: Belief[NDArray[np.float64]]) -> IntervalBelief:
        """What a belief about yesterday's limits becomes today."""
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        assert self._day > 0, "The first day has no yesterday"
        shift = self._calendar.get_shift(self._day) - self._calendar.get_shift(
            self._day - 1
        )
        widening = np.array([-self._drift, self._drift])
        return IntervalBelief(belief.intervals + shift[:, None] + widening)


class _CalendarDrift:
    """Moves the true limits to the next day's, counting its own days. Each joint's
    stiffness also moves by a uniform draw from [-drift, drift] that the calendar
    does not know."""

    def __init__(self, calendar: StiffnessCalendar, drift: float) -> None:
        self._calendar = calendar
        self._drift = drift
        self._day = 0

    def __call__(
        self, limits: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        before = self._calendar.get_shift(self._day)
        self._day += 1
        moved = limits + self._calendar.get_shift(self._day) - before
        if self._drift > 0.0:
            surprise = rng.uniform(-self._drift, self._drift, size=2)
            moved = moved + np.repeat(surprise, 2) * _INWARD
        return moved


def make_calendar_arm_domain(
    seed: int,
    calendar: str = "week",
    step: float = 0.09,
    routine_size: int = 8,
    max_stiffness: float = 1.8,
    num_days: int = 60,
    goal_lean: float = 8.0,
    goal_tolerance: float = 0.15,
    intake_width: float = 0.04,
    drift: float = 0.0,
) -> CalendarArmDomain:
    """A person with a random routine and a random calendar of the given kind (see
    sample_calendar()).

    The person's loosest limits are each 2.4 to 3.1 from 0, so at max_stiffness up
    to 1.8 every limit is at least 0.6 from 0. An exercise's fraction of each
    limit is 0.98 times a draw from Beta(goal_lean, 1). The robot starts knowing
    each of today's limits to within intake_width, as after an intake assessment.
    Each night, each joint's stiffness also moves by up to drift in a way that the
    calendar does not know, and the robot widens its belief to match. The calendar
    covers num_days days, which must include every day the robot
    plans for.
    """
    assert routine_size > 0, "Need at least one exercise"
    assert max_stiffness <= 1.8, "The limits must stay away from 0"
    rng = np.random.default_rng(seed)
    model = TwoLinkArmModel(goal_tolerance=goal_tolerance)
    loosest = -_INWARD * rng.uniform(2.4, 3.1, size=4)
    exercises = [
        Exercise(
            uppers=(bool(rng.random() < 0.5), bool(rng.random() < 0.5)),
            fractions=(
                0.98 * float(rng.beta(goal_lean, 1.0)),
                0.98 * float(rng.beta(goal_lean, 1.0)),
            ),
        )
        for _ in range(routine_size)
    ]
    days = sample_calendar(calendar, num_days, step, max_stiffness, rng)
    truth = loosest + days.get_shift(0)
    starts = truth - rng.uniform(0.0, intake_width, size=4)
    real_env = SimulatedEnvironment(
        model, truth, rng, drift=_CalendarDrift(days, drift)
    )
    tasks = CalendarTaskDistribution(
        model, exercises, days, lambda: real_env.hidden_params, drift
    )
    return Domain(
        env_model=model,
        task_distribution=tasks,
        initial_belief=IntervalBelief(
            np.stack([starts, starts + intake_width], axis=1)
        ),
        real_env=real_env,
        predict_belief=tasks.predict_today,
    )
