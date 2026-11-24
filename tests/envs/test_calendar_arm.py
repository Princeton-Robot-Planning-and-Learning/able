"""Tests for calendar_arm.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.envs.calendar import (
    CALENDAR_KINDS,
    ExerciseTask,
    StiffnessCalendar,
    sample_calendar,
)
from able.envs.calendar_arm import (
    CalendarTaskDistribution,
    Exercise,
    make_calendar_arm_domain,
)
from able.envs.simulated import SimulatedEnvironment


@pytest.mark.parametrize("kind", CALENDAR_KINDS)
def test_calendars_stay_in_range(kind):
    """Every calendar has the days asked for, within the stiffness range."""
    calendar = sample_calendar(kind, 40, 0.1, 1.8, np.random.default_rng(0))
    assert calendar.stiffness.shape == (40, 2)
    assert np.all(calendar.stiffness >= 0.0) and np.all(calendar.stiffness <= 1.8)


def test_week_has_moving_weekdays_and_a_fixed_weekend():
    """Weekdays move by step, and the two weekend days are the same far place."""
    calendar = sample_calendar("week", 14, 0.1, 1.8, np.random.default_rng(0))
    weekdays, weekend = calendar.stiffness[:5], calendar.stiffness[5:7]
    moves = np.abs(np.diff(weekdays, axis=0)).max(axis=1)
    assert np.allclose(moves, 0.1)
    assert np.array_equal(weekend[0], weekend[1])
    assert np.abs(weekend[0] - weekdays[2]).max() >= 0.3


def test_days_past_the_calendar_repeat_the_last():
    """A calendar that ends keeps its last day."""
    calendar = StiffnessCalendar(np.array([[0.0, 0.0], [0.5, 0.1]]))
    assert np.array_equal(calendar.get(9), [0.5, 0.1])


def test_exercise_is_a_fraction_of_the_limits():
    """An exercise's targets scale with the limits on its sides."""
    exercise = Exercise(uppers=(True, False), fractions=(0.5, 0.8))
    joints = exercise.get_joints(np.array([-2.0, 1.0, -1.5, 3.0]))
    assert np.allclose(joints, [0.5, -1.2])


def test_tasks_follow_the_true_limits():
    """Each day's tasks match the real environment's limits that day."""
    domain = make_calendar_arm_domain(0)
    tasks, env = domain.task_distribution, domain.real_env
    assert isinstance(tasks, CalendarTaskDistribution)
    assert isinstance(env, SimulatedEnvironment)
    rng = np.random.default_rng(0)
    for _ in range(9):
        expected = tasks.given(env.hidden_params).get_tasks()  # type: ignore
        for task, other in zip(tasks.get_tasks(), expected):
            assert isinstance(task, ExerciseTask)
            assert np.allclose(task.goal, other.goal)
        env.start_day(rng)
        tasks.start_day(rng)


def test_beliefs_follow_the_calendar():
    """The predicted belief holds the true limits every day, and the forecast for
    a later day matches the belief predicted day by day."""
    domain = make_calendar_arm_domain(1)
    tasks, env = domain.task_distribution, domain.real_env
    assert isinstance(tasks, CalendarTaskDistribution)
    assert isinstance(env, SimulatedEnvironment)
    assert domain.predict_belief is not None
    rng = np.random.default_rng(0)
    belief = domain.initial_belief
    forecasts = [tasks.forecast(belief, k) for k in range(6)]
    assert isinstance(belief, IntervalBelief)
    for day in range(6):
        if day > 0:
            env.start_day(rng)
            tasks.start_day(rng)
            belief = domain.predict_belief(belief)
        assert belief.contains(env.hidden_params)
        assert np.allclose(belief.intervals, forecasts[day].intervals)


def test_drift_widens_beliefs_that_hold_the_truth():
    """With drift, the predicted belief and the forecasts for later days still hold
    the true limits, and forecasts widen with the days ahead."""
    domain = make_calendar_arm_domain(1, drift=0.03)
    tasks, env = domain.task_distribution, domain.real_env
    assert isinstance(tasks, CalendarTaskDistribution)
    assert isinstance(env, SimulatedEnvironment)
    assert domain.predict_belief is not None
    rng = np.random.default_rng(0)
    belief = domain.initial_belief
    assert isinstance(belief, IntervalBelief)
    forecasts = [tasks.forecast(belief, k) for k in range(8)]
    widths = [f.get_total_width() for f in forecasts]
    assert np.all(np.diff(widths) > 0.0)
    for day in range(8):
        if day > 0:
            env.start_day(rng)
            tasks.start_day(rng)
            belief = domain.predict_belief(belief)
        assert belief.contains(env.hidden_params)
        assert forecasts[day].contains(env.hidden_params)
