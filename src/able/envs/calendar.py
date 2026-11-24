"""Calendars of a person's stiffness, which the robot knows, and routines of
exercises whose tasks move with the person's range of motion."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.envs.base import FiniteTaskDistribution
from able.structs import Task

CALENDAR_KINDS = ("week", "uneven", "steady")
# Draws of a place to start a walk before concluding that the walk cannot fit.
_MAX_WALK_DRAWS = 100000


class StiffnessCalendar:
    """Each motion's stiffness (rad) on each day, with shape (num_days,
    num_motions). Days past the last repeat it."""

    def __init__(self, stiffness: NDArray[np.float64]) -> None:
        stiffness = np.array(stiffness, dtype=np.float64)
        assert stiffness.ndim == 2, "Need (num_days, num_motions)"
        assert np.all(stiffness >= 0.0), "Stiffness is nonnegative"
        stiffness.setflags(write=False)
        self._stiffness = stiffness

    @property
    def stiffness(self) -> NDArray[np.float64]:
        """The read-only (num_days, num_motions) array of stiffnesses."""
        return self._stiffness

    def get(self, day: int) -> NDArray[np.float64]:
        """Each motion's stiffness on a day."""
        assert day >= 0, "Days count from 0"
        return self._stiffness[min(day, len(self._stiffness) - 1)]


def sample_calendar(
    kind: str,
    num_days: int,
    step: float,
    max_stiffness: float,
    rng: np.random.Generator,
    num_motions: int = 2,
    grid: float | None = None,
) -> StiffnessCalendar:
    """A random calendar.

    In a "week", the stiffness moves by step each day in one direction for five
    weekdays, then sits at one place far from them for the two days of the
    weekend. Each week is in a new place. In "uneven", regimes of 1, 2, 3, and 6
    days come in a random order, each in a new place, and the stiffness moves by
    step each day within a regime, as far as the range allows. In "steady", for
    two motions, the stiffness moves by step each day, turning around at the ends
    of its range.

    With grid, and more than two motions, every place is a multiple of it.
    """
    assert kind in CALENDAR_KINDS, f"kind must be one of {CALENDAR_KINDS}"
    assert 0.0 < step and 0.0 < max_stiffness, "Need positive step and range"
    days: list[NDArray[np.float64]] = []
    if kind == "steady":
        assert num_motions == 2, "The steady calendar has two motions"
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
            weekdays = _sample_walk(5, step, max_stiffness, rng, num_motions, grid)
            weekend = weekdays[2]
            far = min(2.0 * step + 0.12, 0.4 * max_stiffness)
            while np.abs(weekend - weekdays[2]).max() < far:
                weekend = rng.uniform(0.0, max_stiffness, size=num_motions)
                if grid is not None:
                    weekend = np.round(weekend / grid) * grid
            days += weekdays + [weekend, weekend]
        else:
            for length in rng.permutation([1, 2, 3, 6]):
                days += _sample_walk(
                    min(int(length), int(max_stiffness / step + 1e-9) + 1),
                    step,
                    max_stiffness,
                    rng,
                    num_motions,
                    grid,
                )
    return StiffnessCalendar(np.array(days[:num_days]))


def _sample_direction(
    num_motions: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """A random direction: for two motions, one of eight axis or diagonal ones, and
    for more, one motion gets looser or stiffer at a time."""
    if num_motions == 2:
        directions = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.float64)
        return np.asarray(directions[rng.integers(4)] * rng.choice([-1.0, 1.0]))
    direction = np.zeros(num_motions)
    direction[rng.integers(num_motions)] = rng.choice([-1.0, 1.0])
    return np.asarray(direction)


def _sample_walk(
    length: int,
    step: float,
    max_stiffness: float,
    rng: np.random.Generator,
    num_motions: int,
    grid: float | None = None,
) -> list[NDArray[np.float64]]:
    """Places length days apart in one random direction, all within range, and on
    the grid if there is one."""
    if num_motions > 2 and grid is not None:
        top, step_count = round(max_stiffness / grid), round(step / grid)
        for _ in range(_MAX_WALK_DRAWS):
            first = rng.integers(0, top + 1, size=num_motions)
            direction = _sample_direction(num_motions, rng)
            places = [first + step_count * k * direction for k in range(length)]
            if all(np.all(p >= 0) and np.all(p <= top) for p in places):
                return [grid * p for p in places]
        raise ValueError(f"A walk of {length} days does not fit in the stiffness grid")
    for _ in range(_MAX_WALK_DRAWS):
        start = rng.uniform(0.0, max_stiffness, size=num_motions)
        direction = _sample_direction(num_motions, rng)
        places = [start + step * k * direction for k in range(length)]
        if all(np.all(p >= 0.0) and np.all(p <= max_stiffness) for p in places):
            return places
    raise ValueError(f"A walk of {length} days does not fit in the stiffness range")


@dataclass(frozen=True, eq=False)
class ExerciseTask(Task[Any, Any]):
    """A task to do one exercise of the person's routine."""

    exercise: int = field(kw_only=True)


class ExerciseTaskDistribution(FiniteTaskDistribution[Any]):
    """A person's routine of exercises on a known calendar, whose tasks for one
    day depend on the person's limits that day. The robot knows the routine and the
    calendar and uses forecast() and get_task(); the day's true tasks are only for
    the experiment harness."""

    @property
    @abstractmethod
    def num_exercises(self) -> int:
        """The exercises in the routine."""

    @property
    def depends_on_hidden_params(self) -> bool:
        return True

    @abstractmethod
    def get_midpoint(self, belief: Belief[Any]) -> NDArray[np.float64]:
        """The most likely position of each limit under a belief (rad), where
        distances between days are measured."""

    def get_distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """How different two days' midpoints are, for the same exercise: by default,
        the largest difference between limits."""
        return float(np.abs(a - b).max())

    @abstractmethod
    def get_task(self, exercise: int, midpoint: NDArray[np.float64]) -> Any:
        """The task to do an exercise when the limits are at a midpoint (rad)."""

    @abstractmethod
    def forecast(self, belief: Belief[Any], days_ahead: int) -> Belief[Any]:
        """What a belief about today's limits becomes days_ahead days from now, if
        nothing more is learned."""
