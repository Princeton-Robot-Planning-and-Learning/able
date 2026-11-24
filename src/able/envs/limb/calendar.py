"""A person whose limb is stiffer on some days than on others, following a calendar
that the robot knows.

Each day, each active motion has a stiffness, and its range of motion shrinks by it.
The person's routine is a list of exercises. An exercise starts from the same posture
every day and moves the limb to a goal that lies a fixed distance short of each of
the day's limits, so its goal moves with the range of motion, and a plan for one day
reaches another day's goal only if the days are alike.

People come from limb_calendar_people.json. Their exercises were drawn and checked
offline: the robot plans and completes each of them with the limits at their loosest,
at their stiffest, and with each motion at its stiffest.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose

from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.base import Domain, FiniteTaskDistribution, TaskDistribution
from able.envs.calendar import (
    ExerciseTask,
    ExerciseTaskDistribution,
    StiffnessCalendar,
    sample_calendar,
)
from able.envs.limb.human import LimbVariant, get_variant
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.query import LimbQuery, QueryOutcome
from able.envs.limb.sim import CoupledState, GraspFrame
from able.envs.simulated import SimulatedEnvironment

PEOPLE_PATH = Path(__file__).with_name("limb_calendar_people.json")

LimbCalendarDomain = Domain[
    LimbState,
    NDArray[np.float64],
    LimbTask,
    NDArray[np.float64],
    LimbQuery,
    QueryOutcome,
]


@dataclass(frozen=True, eq=False)
class LimbExercise:
    """An exercise: the state the limb and robot start from, and how far short of
    each motion's limit the goal lies (rad)."""

    start: CoupledState
    short_of_limit: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class LimbPerson:
    """A person's loosest range of motion, as a scale per active motion, and their
    exercises."""

    variant: str
    loosest: NDArray[np.float64]
    exercises: tuple[LimbExercise, ...]


def coupled_to_json(state: CoupledState) -> dict[str, Any]:
    """The state as JSON data."""
    frame = state.grasp_frame
    return {
        "base": [state.base_pose.x, state.base_pose.y, state.base_pose.rot],
        "robot_positions": list(state.robot_positions),
        "robot_velocities": list(state.robot_velocities),
        "limb_positions": list(state.limb_positions),
        "limb_velocities": list(state.limb_velocities),
        "grasp": (
            None
            if frame is None
            else {
                "parent": [*frame.parent.position, *frame.parent.orientation],
                "child": [*frame.child.position, *frame.child.orientation],
                "max_force": frame.max_force,
                "erp": frame.erp,
            }
        ),
    }


def coupled_from_json(data: dict[str, Any]) -> CoupledState:
    """The state that coupled_to_json() made."""

    def pose(values: Sequence[float]) -> Pose:
        x, y, z, a, b, c, d = values
        return Pose((x, y, z), (a, b, c, d))

    grasp = data["grasp"]
    return CoupledState(
        base_pose=SE2Pose(*data["base"]),
        robot_positions=tuple(data["robot_positions"]),
        robot_velocities=tuple(data["robot_velocities"]),
        limb_positions=tuple(data["limb_positions"]),
        limb_velocities=tuple(data["limb_velocities"]),
        grasp_frame=(
            None
            if grasp is None
            else GraspFrame(
                pose(grasp["parent"]),
                pose(grasp["child"]),
                grasp["max_force"],
                grasp["erp"],
            )
        ),
    )


@lru_cache(maxsize=1)
def _load_people(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def load_person(index: int, path: Path = PEOPLE_PATH) -> LimbPerson:
    """The person with the index."""
    people = _load_people(path)
    if str(index) not in people:
        raise KeyError(f"No person {index} in {path.name}; have {sorted(people)}")
    data = people[str(index)]
    return LimbPerson(
        variant=data["variant"],
        loosest=np.array(data["loosest"]),
        exercises=tuple(
            LimbExercise(coupled_from_json(e["start"]), np.array(e["short_of_limit"]))
            for e in data["exercises"]
        ),
    )


class LimbCalendarTaskDistribution(ExerciseTaskDistribution):
    """The person's routine on a calendar of stiffness, whose tasks are one per
    exercise under today's range of motion, which get_true_scales() gives the
    harness. Beliefs are RangeOfMotionBeliefs over today's scales, and a midpoint
    is each limit's angle (rad)."""

    def __init__(
        self,
        model: LimbModel,
        person: LimbPerson,
        calendar: StiffnessCalendar,
        get_true_scales: Callable[[], NDArray[np.float64]],
    ) -> None:
        self._model = model
        self._person = person
        self._calendar = calendar
        self._get_true_scales = get_true_scales
        self._day = 0
        self._variant: LimbVariant = model.variant
        self._magnitudes = self._variant.default_magnitudes
        # The inactive joints rest where the first exercise starts.
        self._nominal = np.array(person.exercises[0].start.limb_positions)
        super().__init__(self._make_tasks(get_true_scales()))

    @property
    def num_exercises(self) -> int:
        return len(self._person.exercises)

    @property
    def day(self) -> int:
        """The current day, counted from 0."""
        return self._day

    def get_goal(
        self, exercise: int, scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The goal of an exercise under a range of motion: each active joint its
        distance short of the limit, and the others where the exercise starts."""
        spec = self._person.exercises[exercise]
        positions = np.array(spec.start.limb_positions, dtype=np.float64)
        low, high = self._variant.get_joint_limits(scales)
        for i, (joint, sign) in enumerate(
            zip(self._variant.active_joints, self._variant.active_signs, strict=True)
        ):
            limit = high[joint] if sign > 0 else low[joint]
            positions[joint] = limit - sign * spec.short_of_limit[i]
        return positions

    def _make_task(self, exercise: int, scales: NDArray[np.float64]) -> LimbTask:
        goal = self.get_goal(exercise, scales)
        state = self._model.get_initial_state(
            self._person.exercises[exercise].start, goal
        )
        return ExerciseTask(initial_state=state, goal=goal, exercise=exercise)

    def _make_tasks(self, scales: NDArray[np.float64]) -> list[LimbTask]:
        return [self._make_task(e, scales) for e in range(len(self._person.exercises))]

    def given(self, hidden_params: Any) -> TaskDistribution[Any]:
        """The routine's tasks under this range of motion."""
        return FiniteTaskDistribution(self._make_tasks(np.asarray(hidden_params)))

    def get_distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        # The limb's distance to a goal sums over its joints.
        return float(np.abs(a - b).sum())

    def get_midpoint(self, belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        return np.asarray(belief.mean * self._magnitudes)

    def get_task(self, exercise: int, midpoint: NDArray[np.float64]) -> LimbTask:
        return self._make_task(exercise, midpoint / self._magnitudes)

    def start_day(self, rng: np.random.Generator) -> None:
        self._day += 1
        super().__init__(self._make_tasks(self._get_true_scales()))

    def _shift_belief(
        self, belief: Belief[Any], change: NDArray[np.float64]
    ) -> RangeOfMotionBelief:
        """The belief after every limit moves inward by a change (rad)."""
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        delta = change / self._magnitudes
        return RangeOfMotionBelief(
            mean=belief.mean - delta,
            stddev=belief.stddev,
            low=belief.low - delta,
            high=belief.high - delta,
            floor=belief.floor - delta,
        )

    def forecast(self, belief: Belief[Any], days_ahead: int) -> RangeOfMotionBelief:
        change = self._calendar.get(self._day + days_ahead) - self._calendar.get(
            self._day
        )
        return self._shift_belief(belief, change)

    def predict_today(self, belief: Belief[Any]) -> RangeOfMotionBelief:
        """What a belief about yesterday's range of motion becomes today."""
        assert self._day > 0, "The first day has no yesterday"
        change = self._calendar.get(self._day) - self._calendar.get(self._day - 1)
        return self._shift_belief(belief, change)


class _LimbCalendarDrift:
    """Moves the true scales to the next day's, counting its own days."""

    def __init__(
        self, calendar: StiffnessCalendar, magnitudes: NDArray[np.float64]
    ) -> None:
        self._calendar = calendar
        self._magnitudes = magnitudes
        self._day = 0

    def __call__(
        self, scales: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        del rng  # unused because the calendar is fixed
        before = self._calendar.get(self._day)
        self._day += 1
        change = self._calendar.get(self._day) - before
        return np.asarray(scales - change / self._magnitudes)


def make_limb_calendar_domain(
    seed: int,
    calendar: str = "uneven",
    step: float = 0.06,
    max_stiffness: float = 0.3,
    num_days: int = 60,
    survey_stddev: float = 0.02,
    grid: float | None = 0.03,
    horizon: int = 300,
    people_path: Path = PEOPLE_PATH,
) -> LimbCalendarDomain:
    """Person number seed (see PEOPLE_PATH) with a random calendar of the kind.

    The stiffness of each of the three motions stays within max_stiffness (rad) and
    moves by step a day (see sample_calendar()), on a grid if it is given. The robot
    starts knowing today's
    range of motion to within survey_stddev, as a scale.
    """
    person = load_person(seed, people_path)
    rng = np.random.default_rng([seed, 7])
    variant = get_variant(person.variant)
    model = LimbModel(variant, horizon=horizon)
    num_motions = len(variant.active_motions)
    days = sample_calendar(
        calendar, num_days, step, max_stiffness, rng, num_motions, grid
    )
    scales = person.loosest - days.get(0) / variant.default_magnitudes
    real_env = SimulatedEnvironment(
        model,
        scales,
        rng,
        drift=_LimbCalendarDrift(days, variant.default_magnitudes),
    )
    tasks = LimbCalendarTaskDistribution(
        model, person, days, lambda: real_env.hidden_params
    )
    # Wide bounds, so that the truncation plays no part.
    belief = RangeOfMotionBelief.from_prior(
        scales,
        survey_stddev,
        np.full(num_motions, 0.3),
        np.full(num_motions, 1.5),
        num_motions,
    )
    return Domain(
        env_model=model,
        task_distribution=tasks,
        initial_belief=belief,
        real_env=real_env,
        predict_belief=tasks.predict_today,
    )
