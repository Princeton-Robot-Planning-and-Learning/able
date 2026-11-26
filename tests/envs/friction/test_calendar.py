"""Tests for envs/friction/calendar.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

import numpy as np
import pytest

from able.beliefs.particles import ParticleBelief
from able.envs.friction.calendar import (
    FrictionCalendarTaskDistribution,
    FrictionRoutine,
    get_zones,
    load_routine,
    make_friction_calendar_domain,
    move_goal,
)
from able.envs.simulated import SimulatedEnvironment


@pytest.fixture(name="parts", scope="module")
def fixture_parts():
    """A domain, its tasks, and its environment, closed after the module."""
    domain = make_friction_calendar_domain(1)
    tasks, env = domain.task_distribution, domain.real_env
    assert isinstance(tasks, FrictionCalendarTaskDistribution)
    assert isinstance(env, SimulatedEnvironment)
    yield domain, tasks, env
    domain.env_model.close()


def test_routines_survive_json():
    """A routine has six tasks and loads what it stores."""
    routine = load_routine(1)
    assert len(routine.tasks) == 6
    again = FrictionRoutine.from_json(routine.to_json())
    assert np.array_equal(again.base, routine.base)
    assert again.tasks == routine.tasks


def test_zones_are_colors(parts):
    """The eight floors of threeroutes form three zones by color."""
    domain, _, _ = parts
    assert list(get_zones(domain.env_model.world)) == [0, 0, 0, 1, 1, 2, 2, 2]


def test_beliefs_follow_the_calendar(parts):
    """The predicted belief tracks the true frictions day by day, the forecast for a
    later day matches it, and the midpoint is each zone's slick on the grid."""
    domain, tasks, env = parts
    assert domain.predict_belief is not None
    rng = np.random.default_rng(0)
    belief = domain.initial_belief
    assert isinstance(belief, ParticleBelief)
    first = env.hidden_params.copy()
    forecasts = [tasks.forecast(belief, k) for k in range(5)]
    goals_before = [task.goal for task in tasks.get_tasks()]
    for day in range(5):
        if day > 0:
            env.start_day(rng)
            tasks.start_day(rng)
            belief = domain.predict_belief(belief)
        assert np.allclose(np.log(belief.mean), np.log(env.hidden_params), atol=0.02)
        assert np.allclose(belief.mean, forecasts[day].mean)
        slick = tasks.get_midpoint(belief)
        assert np.allclose(slick / 0.1, np.round(slick / 0.1), atol=1e-6)
        change = np.log(first) - np.log(env.hidden_params)
        assert np.allclose(
            change, slick[tasks.zones] - tasks.get_midpoint(forecasts[0])[tasks.zones]
        )
    assert [task.goal for task in tasks.get_tasks()] == goals_before


def test_goals_move_with_each_zone():
    """One step of slick in one zone moves a goal by the shift times the step, in
    that zone's direction, and equal slick in every zone cancels out."""
    goal = (2.0, 2.0)
    moved = move_goal(goal, 0.3, np.array([0.1, 0.0, 0.0]), 0.5)
    assert np.isclose(np.hypot(moved[0] - 2.0, moved[1] - 2.0), 0.05, atol=1e-4)
    assert np.isclose(np.arctan2(moved[1] - 2.0, moved[0] - 2.0), 0.3, atol=1e-2)
    assert np.allclose(move_goal(goal, 0.3, np.full(3, 0.7), 0.5), goal, atol=1e-4)


def test_cleaning_weeks_repeat_or_drift():
    """A cleaning week repeats exactly, and a drifting week changes one zone's
    polish by one step from one week to the next."""
    from able.envs.friction.calendar import (  # pylint: disable=import-outside-toplevel
        sample_cleaning_week,
    )

    rng = np.random.default_rng(3)
    week = sample_cleaning_week(21, 0.2, 1.0, rng).stiffness
    assert np.allclose(week[:7], week[7:14])
    assert len({tuple(np.round(row, 6)) for row in week[:7]}) == 7
    drift = sample_cleaning_week(21, 0.2, 1.0, rng, drift_zones=1).stiffness
    changed = np.abs(drift[7:14] - drift[:7]).max(axis=0) > 1e-9
    assert changed.sum() == 1
    assert np.isclose(np.abs(drift[7:14] - drift[:7]).max(), 0.1)


def test_weekly_notice_expects_this_week_to_repeat():
    """With a weekly notice, the forecast of next week is this week's same weekday
    until Monday comes, and with notice of the run it is the calendar's."""
    known = make_friction_calendar_domain(1, calendar="drifting_week", num_days=28)
    weekly = make_friction_calendar_domain(
        1, calendar="drifting_week", num_days=28, notice="week"
    )
    for domain in (known, weekly):
        domain.env_model.close()
    a, b = known.task_distribution, weekly.task_distribution
    rng = np.random.default_rng(0)
    for _ in range(5):
        a.start_day(rng)
        b.start_day(rng)
    calendar = a._calendar  # pylint: disable=protected-access
    assert np.allclose(a.known_slick(9), calendar.get(9))
    assert np.allclose(b.known_slick(9), calendar.get(2))
    assert np.allclose(b.known_slick(6), calendar.get(6))


def test_weather_is_learned_each_morning():
    """With weather, the belief at night is the schedule's, a look outside moves it
    to the day's true frictions, and forecasts draw rain from the forecast."""
    from able.envs.friction.calendar import (  # pylint: disable=import-outside-toplevel
        CalendarFrictionBeliefUpdater,
        WeatherGatherer,
    )

    domain = make_friction_calendar_domain(
        2, calendar="cleaning_week", num_days=28, weather=True
    )
    domain.env_model.close()
    tasks, env = domain.task_distribution, domain.real_env
    updater = CalendarFrictionBeliefUpdater(domain.env_model.world, tasks)
    rng = np.random.default_rng(0)
    belief = domain.initial_belief
    rainy = 0
    for day in range(14):
        if day > 0:
            env.start_day(rng)
            tasks.start_day(rng)
            belief = domain.predict_belief(belief)
        schedule = tasks._calendar.get(day)  # pylint: disable=protected-access
        assert np.allclose(tasks.get_midpoint(belief), schedule, atol=1e-6)
        data = WeatherGatherer().gather(belief, env, None, rng)
        belief = updater.update_from_queries(belief, data)
        assert np.allclose(tasks.get_midpoint(belief), tasks.true_slick(day), atol=1e-6)
        assert np.allclose(np.log(belief.mean), np.log(env.hidden_params), atol=0.02)
        rainy += not np.allclose(tasks.true_slick(day), schedule)
    assert rainy > 0
    draws = {
        tuple(np.round(tasks.get_midpoint(tasks.forecast(belief, 1)), 3))
        for _ in range(50)
    }
    assert len(draws) > 1
