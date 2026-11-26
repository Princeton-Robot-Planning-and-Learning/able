"""A friction site whose floors are polished on a calendar that the robot knows.

Each floor's friction is its base friction times exp(-slick), where slick is the
day's polish of the floor's zone (its color). The zones follow a calendar of slick
values (envs/calendar.py), one zone changing at a time. The routine is a fixed list
of pushing tasks, each with the same start every day. With a goal shift, each
zone's slick moves the task's goal along its own direction, the three directions
120 degrees apart, as the limb's goals move with its limits. Without one, the
goals are fixed and only the dynamics change with the calendar.

Routines come from friction_calendar_routines.json. Their base frictions and tasks
were drawn and checked offline: the night planner solves each task with every zone
at its least and its most slick, and with each zone alone at its most slick.
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.base import Domain, FiniteTaskDistribution, TaskDistribution
from able.envs.calendar import (
    ExerciseTask,
    ExerciseTaskDistribution,
    StiffnessCalendar,
    sample_calendar,
)
from able.envs.friction.model import FrictionModel, FrictionTask
from able.envs.friction.tasks import FrictionDomain
from able.envs.friction.world import PuckWorld
from able.envs.simulated import SimulatedEnvironment
from able.info_gatherers.base import InformationGatherer
from able.structs import Budget, QueryResponse

ROUTINES_PATH = Path(__file__).with_name("friction_calendar_routines.json")
# Bounds of the robot's particles, wide enough to play no part.
BELIEF_BOUNDS = (0.005, 1.0)


@dataclass(frozen=True, eq=False)
class FrictionRoutine:
    """A site's base friction per floor and its tasks, each a puck start, robot
    start and goal with no slick, and the angle (rad) of the first zone's direction
    of goal shift."""

    base: NDArray[np.float64]
    tasks: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
    ]
    angles: tuple[float, ...] | None = None

    def to_json(self) -> dict[str, Any]:
        """The routine as JSON data."""
        tasks: list[dict[str, Any]] = [
            {"puck": list(p), "robot": list(r), "goal": list(g)}
            for p, r, g in self.tasks
        ]
        if self.angles is not None:
            for task, angle in zip(tasks, self.angles, strict=True):
                task["angle"] = angle
        return {"base": [float(f) for f in self.base], "tasks": tasks}

    @staticmethod
    def from_json(data: dict[str, Any]) -> "FrictionRoutine":
        """The routine that to_json() made."""
        tasks = data["tasks"]
        return FrictionRoutine(
            base=np.array(data["base"], dtype=np.float64),
            tasks=tuple(
                (tuple(t["puck"]), tuple(t["robot"]), tuple(t["goal"])) for t in tasks
            ),
            angles=(
                tuple(float(t["angle"]) for t in tasks)
                if all("angle" in t for t in tasks)
                else None
            ),
        )


@lru_cache(maxsize=4)
def _load_routines(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def load_routine(index: int, path: Path = ROUTINES_PATH) -> FrictionRoutine:
    """The routine with the index."""
    routines = _load_routines(path)
    if str(index) not in routines:
        raise KeyError(f"No routine {index} in {path.name}; have {sorted(routines)}")
    return FrictionRoutine.from_json(routines[str(index)])


def get_zones(world: PuckWorld) -> NDArray[np.int64]:
    """Each floor's zone: the index of its color among the map's colors, in order."""
    colors = [name.split()[0] for name in world.puck_map.surface_names]
    order = list(dict.fromkeys(colors))
    return np.array([order.index(c) for c in colors], dtype=np.int64)


def get_frictions(
    base: NDArray[np.float64], zones: NDArray[np.int64], slick: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Each floor's friction when each zone has a slick (log units)."""
    return np.asarray(base * np.exp(-np.asarray(slick)[zones]))


def move_goal(
    goal: tuple[float, float],
    angle: float,
    slick: NDArray[np.float64],
    goal_shift: float,
) -> tuple[float, float]:
    """The goal moved goal_shift (m per unit of slick) by each zone's slick along
    its direction: the first at angle, each next one 120 degrees on."""
    x, y = goal
    for z, value in enumerate(np.asarray(slick)):
        direction = angle + 2.0 * math.pi * z / len(slick)
        x += goal_shift * float(value) * math.cos(direction)
        y += goal_shift * float(value) * math.sin(direction)
    return (round(x, 4), round(y, 4))


# A day's forecast: the chances of no rain, light rain and heavy rain, for clear,
# cloudy and stormy days.
FORECASTS = ((0.95, 0.05, 0.0), (0.7, 0.25, 0.05), (0.4, 0.4, 0.2))


@dataclass(frozen=True, eq=False)
class Weather:
    """A forecast of rain for every day, the rain that falls, and what it does:
    each level of rain makes one zone (the entrance) slick_per_level slicker, up to
    max_slick."""

    chances: NDArray[np.float64]
    levels: NDArray[np.int64]
    zone: int
    slick_per_level: float
    max_slick: float

    def slick(self, schedule: NDArray[np.float64], level: int) -> NDArray[np.float64]:
        """The day's slick per zone with the schedule and a level of rain."""
        slick = np.array(schedule, dtype=np.float64)
        slick[self.zone] = min(
            self.max_slick, slick[self.zone] + level * self.slick_per_level
        )
        return np.round(slick, 6)


def sample_weather(
    num_days: int,
    num_zones: int,
    slick_per_level: float,
    max_slick: float,
    rng: np.random.Generator,
) -> Weather:
    """A random forecast (clear, cloudy or stormy) for each day, the rain drawn from
    it, and a random entrance zone."""
    kinds = rng.integers(len(FORECASTS), size=num_days)
    chances = np.array([FORECASTS[k] for k in kinds])
    levels = np.array([rng.choice(3, p=c) for c in chances], dtype=np.int64)
    return Weather(
        chances, levels, int(rng.integers(num_zones)), slick_per_level, max_slick
    )


@dataclass(frozen=True)
class WeatherQuery:
    """Looking outside in the morning to see how much it rained."""


class FrictionCalendarTaskDistribution(ExerciseTaskDistribution):
    """The site's routine on a calendar of polish, whose tasks are one per routine
    task under today's slick. Beliefs are ParticleBeliefs over today's floor
    frictions, and a midpoint is each zone's believed slick: how far the mean log
    friction of its floors lies below their base, as the robot first believed it."""

    def __init__(
        self,
        model: FrictionModel,
        routine: FrictionRoutine,
        calendar: StiffnessCalendar,
        zones: NDArray[np.int64],
        initial_belief: ParticleBelief,
        goal_shift: float = 0.0,
        notice: str = "run",
        weather: Weather | None = None,
        dream_seed: int = 0,
    ) -> None:
        assert goal_shift == 0.0 or routine.angles is not None, "Need goal angles"
        assert notice in NOTICES, f"notice must be one of {NOTICES}"
        self._notice = notice
        self._model = model
        self._routine = routine
        self._calendar = calendar
        self._zones = zones
        self._num_zones = int(zones.max()) + 1
        self._goal_shift = goal_shift
        self._day = 0
        self._reference = self._zone_log_means(initial_belief) + calendar.get(0)
        self._weather = weather
        self._dream_rng = np.random.default_rng(dream_seed)
        self._believed = np.asarray(calendar.get(0))
        super().__init__(self._make_tasks(self.true_slick(0)))

    def get_goal(
        self, exercise: int, slick: NDArray[np.float64]
    ) -> tuple[float, float]:
        """The goal of a task when each zone has a slick."""
        goal = self._routine.tasks[exercise][2]
        if self._goal_shift == 0.0 or self._routine.angles is None:
            return goal
        return move_goal(goal, self._routine.angles[exercise], slick, self._goal_shift)

    def _make_task(self, exercise: int, slick: NDArray[np.float64]) -> FrictionTask:
        puck, robot, _ = self._routine.tasks[exercise]
        goal = self.get_goal(exercise, np.round(slick, 6))
        state = self._model.get_initial_state(puck, robot, goal)
        return ExerciseTask(initial_state=state, goal=goal, exercise=exercise)

    def _make_tasks(self, slick: NDArray[np.float64]) -> list[FrictionTask]:
        return [self._make_task(e, slick) for e in range(len(self._routine.tasks))]

    @property
    def num_exercises(self) -> int:
        return len(self.get_tasks())

    @property
    def day(self) -> int:
        """The current day, counted from 0."""
        return self._day

    @property
    def zones(self) -> NDArray[np.int64]:
        """Each floor's zone."""
        return self._zones

    def _zone_log_means(self, belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        logs = np.log(belief.mean)
        return np.array([logs[self._zones == z].mean() for z in range(self._num_zones)])

    def given(self, hidden_params: Any) -> TaskDistribution[Any]:
        """The routine's tasks under these frictions."""
        logs = np.log(self._routine.base) - np.log(np.asarray(hidden_params))
        slick = np.array(
            [logs[self._zones == z].mean() for z in range(self._num_zones)]
        )
        return FiniteTaskDistribution(self._make_tasks(slick))

    def get_distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.abs(a - b).sum())

    def get_midpoint(self, belief: Belief[Any]) -> NDArray[np.float64]:
        return np.asarray(self._reference - self._zone_log_means(belief))

    def get_task(self, exercise: int, midpoint: NDArray[np.float64]) -> FrictionTask:
        return self._make_task(exercise, midpoint)

    def start_day(self, rng: np.random.Generator) -> None:
        del rng  # unused because the calendar is fixed
        self._day += 1
        super().__init__(self._make_tasks(self.true_slick(self._day)))

    def _shift_belief(
        self, belief: Belief[Any], change: NDArray[np.float64]
    ) -> ParticleBelief:
        """The belief after each zone's slick grows by a change."""
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        return belief.scaled(np.exp(-change[self._zones]))

    def known_slick(self, day: int) -> NDArray[np.float64]:
        """Each zone's slick on a day, as the robot knows it today: the calendar's,
        or with notice "week", for a later week, this week's same weekday."""
        if self._notice == "week" and day // 7 > self._day // 7:
            day = 7 * (self._day // 7) + day % 7
        return self._calendar.get(day)

    def true_slick(self, day: int) -> NDArray[np.float64]:
        """Each zone's slick on a day: the schedule's, and the rain that falls."""
        schedule = self._calendar.get(day)
        if self._weather is None:
            return np.asarray(schedule)
        return self._weather.slick(schedule, int(self._weather.levels[day]))

    def _rain(self, day: int, likely: bool) -> NDArray[np.float64]:
        """A day's slick as the robot knows it, with a level of rain drawn from the
        forecast, or the most likely one."""
        schedule = self.known_slick(day)
        if self._weather is None:
            return np.asarray(schedule)
        chances = self._weather.chances[min(day, len(self._weather.chances) - 1)]
        level = (
            int(np.argmax(chances))
            if likely
            else int(self._dream_rng.choice(3, p=chances))
        )
        return self._weather.slick(schedule, level)

    def forecast(self, belief: Belief[Any], days_ahead: int) -> ParticleBelief:
        """The belief on a day days_ahead from now, under rain drawn from that day's
        forecast; today's rain is unknown until the robot looks outside."""
        return self._shift_belief(
            belief, self._rain(self._day + days_ahead, likely=False) - self._believed
        )

    def forecast_likely(self, belief: Belief[Any], days_ahead: int) -> ParticleBelief:
        """forecast() under each day's most likely rain."""
        return self._shift_belief(
            belief, self._rain(self._day + days_ahead, likely=True) - self._believed
        )

    def predict_today(self, belief: Belief[Any]) -> ParticleBelief:
        """What a belief about yesterday's frictions becomes this morning, before
        the robot looks outside: today's schedule, without rain."""
        assert self._day > 0, "The first day has no yesterday"
        today = np.asarray(self.known_slick(self._day))
        shifted = self._shift_belief(belief, today - self._believed)
        self._believed = today
        return shifted

    def reveal_weather(self, belief: Belief[Any]) -> ParticleBelief:
        """The belief after looking outside: today's rain is known."""
        today = self.true_slick(self._day)
        shifted = self._shift_belief(belief, today - self._believed)
        self._believed = today
        return shifted


CLEANING_WEEK = "cleaning_week"
# The cleaning week whose polish levels drift from week to week.
DRIFTING_WEEK = "drifting_week"
# When the robot learns each week's drift: at the start of the run, or on the
# week's Monday, before which it expects the current week to repeat.
NOTICES = ("run", "week")
# Draws of a cleaning week before concluding that its days cannot all differ.
_MAX_WEEK_DRAWS = 1000


def _week_steps(
    polish_days: NDArray[np.int64],
    polish_levels: NDArray[np.int64],
    sunday: tuple[int, int],
    wear: int,
) -> NDArray[np.int64]:
    """The 7 days' slick per zone, in grid steps, of a cleaning week."""
    num_zones = len(polish_days)
    week = np.zeros((7, num_zones), dtype=np.int64)
    for z in range(num_zones):
        for day in range(polish_days[z], 5):
            week[day, z] = max(0, polish_levels[z] - wear * (day - polish_days[z]))
    week[6, sunday[0]] = sunday[1]
    return week


def sample_cleaning_week(
    num_days: int,
    step: float,
    max_slick: float,
    rng: np.random.Generator,
    num_zones: int = 3,
    grid: float = 0.1,
    drift_zones: int = 0,
) -> StiffnessCalendar:
    """A floor-care week, drawn once: the 7 days' slick per zone, repeated.

    Each zone is polished on its own weekday (Monday to Friday), to its own level
    of slick, and the polish wears off by step a day. A zone is dry earlier in the
    week. On Saturday every zone is dry, because the polish is stripped on Friday
    night. On Sunday one zone is waxed for a weekend event, and the wax is stripped
    before Monday. The first week is drawn until its 7 days all differ. With
    drift_zones, each later week moves that many random zones' polish levels one
    grid step up or down, turning back at the ends of the levels. Every value is a
    multiple of grid within [0, max_slick].
    """
    top = round(max_slick / grid)
    wear = round(step / grid)
    levels = np.arange(max(1, top // 2), top + 1)
    for _ in range(_MAX_WEEK_DRAWS):
        polish_days = rng.permutation(5)[:num_zones]
        polish_levels = rng.choice(levels, size=num_zones)
        sunday = (int(rng.integers(num_zones)), int(rng.choice(levels)))
        week = _week_steps(polish_days, polish_levels, sunday, wear)
        if len({tuple(row) for row in week}) == 7:
            break
    else:
        raise ValueError("Found no cleaning week whose 7 days all differ")
    weeks = [week]
    while 7 * len(weeks) < num_days:
        polish_levels = polish_levels.copy()
        for z in rng.permutation(num_zones)[:drift_zones]:
            move = int(rng.choice([-1, 1]))
            if not levels[0] <= polish_levels[z] + move <= levels[-1]:
                move = -move
            polish_levels[z] += move
        weeks.append(_week_steps(polish_days, polish_levels, sunday, wear))
    days = np.concatenate(weeks)[:num_days].astype(np.float64)
    return StiffnessCalendar(days * grid)


class _FrictionCalendarDrift:
    """Moves the true frictions to the next day's, counting its own days."""

    def __init__(self, calendar: StiffnessCalendar, zones: NDArray[np.int64]) -> None:
        self._calendar = calendar
        self._zones = zones
        self._day = 0

    def __call__(
        self, frictions: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        del rng  # unused because the calendar is fixed
        before = self._calendar.get(self._day)
        self._day += 1
        change = self._calendar.get(self._day) - before
        return np.asarray(frictions * np.exp(-change[self._zones]))


def make_friction_calendar_domain(
    seed: int,
    calendar: str = "uneven",
    step: float = 0.2,
    max_slick: float = 1.0,
    num_days: int = 40,
    survey_stddev: float = 0.05,
    grid: float | None = 0.1,
    num_particles: int = 500,
    num_holes: int = 4,
    goal_radius: float = 0.2,
    goal_shift: float = 0.0,
    notice: str = "run",
    weather: bool = False,
    rain_slick: float = 0.3,
    horizon: int = 15,
    strike_noise: bool = False,
    measurement_noise: bool = False,
    routines: str = ROUTINES_PATH.name,
    calendar_seed: int | None = None,
) -> FrictionDomain:
    """Routine number seed from the routines file next to this module (see
    ROUTINES_PATH) on threeroutes with 8 floors, and a random calendar of the kind:
    one of sample_calendar()'s, or a cleaning week that repeats (CLEANING_WEEK) or
    drifts by one zone a week (DRIFTING_WEEK, see sample_cleaning_week()). With
    notice "week", the robot learns each week's calendar on its Monday and expects
    the current week to repeat until then; with "run", it knows the whole calendar.
    With weather, rain from a known forecast makes one zone rain_slick slicker per
    level of rain (see Weather), and the robot learns each day's rain only in the
    morning (WeatherQuery).

    Each zone's slick stays within max_slick (log friction) and moves by step a day
    (see sample_calendar()), on a grid if it is given. The robot starts knowing
    today's frictions to within survey_stddev in log friction. The calendar is drawn
    from calendar_seed, if given, instead of seed. Each task's goal moves goal_shift
    (m) per unit of each zone's slick (see move_goal()).
    """
    routine = load_routine(seed, ROUTINES_PATH.with_name(routines))
    rng = np.random.default_rng([seed if calendar_seed is None else calendar_seed, 11])
    model = FrictionModel(
        "2d",
        num_holes,
        goal_radius,
        horizon,
        num_floors=8,
        strike_noise=strike_noise,
        measurement_noise=measurement_noise,
    )
    zones = get_zones(model.world)
    num_zones = int(zones.max()) + 1
    if calendar in (CLEANING_WEEK, DRIFTING_WEEK):
        assert grid is not None, "A cleaning week needs a grid"
        drift_zones = 1 if calendar == DRIFTING_WEEK else 0
        days = sample_cleaning_week(
            num_days, step, max_slick, rng, num_zones, grid, drift_zones
        )
    else:
        days = sample_calendar(
            calendar, num_days, step, max_slick, rng, num_zones, grid
        )
    rain = None
    if weather:
        rain = sample_weather(
            num_days,
            num_zones,
            rain_slick,
            max_slick,
            np.random.default_rng([seed, 13]),
        )
        true_days = StiffnessCalendar(
            np.array(
                [rain.slick(days.get(d), int(rain.levels[d])) for d in range(num_days)]
            )
        )
    else:
        true_days = days
    # The robot's first belief is for the schedule; it learns the rain each morning.
    frictions = get_frictions(routine.base, zones, days.get(0))
    # From its own stream, so that a day's belief is the same on every calendar.
    noise_rng = np.random.default_rng([seed, 12])
    noise = noise_rng.normal(0.0, survey_stddev, size=(num_particles, len(frictions)))
    low, high = BELIEF_BOUNDS
    belief = ParticleBelief(
        frictions * np.exp(noise),
        np.full(len(frictions), low),
        np.full(len(frictions), high),
        seed=int(noise_rng.integers(2**31)),
    )
    real_env = SimulatedEnvironment(
        model,
        get_frictions(routine.base, zones, true_days.get(0)),
        rng,
        drift=_FrictionCalendarDrift(true_days, zones),
    )
    tasks = FrictionCalendarTaskDistribution(
        model,
        routine,
        days,
        zones,
        belief,
        goal_shift,
        notice,
        rain,
        dream_seed=seed,
    )
    return Domain(
        env_model=model,
        task_distribution=tasks,
        initial_belief=belief,
        real_env=real_env,
        predict_belief=tasks.predict_today,
    )


class WeatherGatherer(InformationGatherer[NDArray[np.float64], WeatherQuery, None]):
    """Looks outside each morning, which costs nothing and needs no budget."""

    def gather(
        self,
        belief: Belief[NDArray[np.float64]],
        env: Any,
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[WeatherQuery, None]]:
        del belief, env, budget, rng  # unused because looking outside always works
        return [QueryResponse(WeatherQuery(), None)]


class CalendarFrictionBeliefUpdater(FrictionBeliefUpdater):
    """The friction belief updater, which also learns the day's rain from a look
    outside (see FrictionCalendarTaskDistribution.reveal_weather())."""

    def __init__(
        self, world: PuckWorld, tasks: FrictionCalendarTaskDistribution
    ) -> None:
        super().__init__(world)
        self._tasks = tasks

    def update_from_queries(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[QueryResponse[Any, Any]],
    ) -> Belief[NDArray[np.float64]]:
        pushes = [d for d in data if not isinstance(d.query, WeatherQuery)]
        if len(pushes) < len(data):
            belief = self._tasks.reveal_weather(belief)
        return super().update_from_queries(belief, pushes)
