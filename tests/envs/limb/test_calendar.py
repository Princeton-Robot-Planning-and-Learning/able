"""Tests for envs/limb/calendar.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

import numpy as np
import pytest

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.calendar import ExerciseTask
from able.envs.limb.calendar import (
    LimbCalendarTaskDistribution,
    coupled_from_json,
    coupled_to_json,
    load_person,
    make_limb_calendar_domain,
)
from able.envs.simulated import SimulatedEnvironment


def _get_parts(seed=1, **kwargs):
    domain = make_limb_calendar_domain(seed, **kwargs)
    tasks, env = domain.task_distribution, domain.real_env
    assert isinstance(tasks, LimbCalendarTaskDistribution)
    assert isinstance(env, SimulatedEnvironment)
    return domain, tasks, env


@pytest.fixture(name="parts", scope="module")
def fixture_parts():
    """A domain, its tasks, and its environment, closed after the module."""
    parts = _get_parts()
    yield parts
    parts[0].env_model.close()


def test_people_survive_json():
    """A person's exercises load the same state that was stored."""
    person = load_person(1)
    assert len(person.exercises) == 6
    state = person.exercises[0].start
    assert coupled_from_json(coupled_to_json(state)) == state


def test_goals_follow_the_limits(parts):
    """A stiffer day's goals are closer by the change in each limit."""
    _, tasks, env = parts
    scales = env.hidden_params
    change = np.array([0.06, 0.0, 0.03])
    moved = scales - change / tasks._magnitudes  # pylint: disable=protected-access
    goal, stiff_goal = tasks.get_goal(0, scales), tasks.get_goal(0, moved)
    joints = [0, 1, 3]
    assert np.allclose(np.abs(goal - stiff_goal)[joints], change, atol=1e-6)


def test_tasks_and_beliefs_follow_the_calendar(request):
    """Each day's tasks and the predicted belief match the true range of motion, and
    the forecast for a later day matches the belief predicted day by day."""
    domain, tasks, env = _get_parts()
    request.addfinalizer(domain.env_model.close)
    assert domain.predict_belief is not None
    rng = np.random.default_rng(0)
    belief = domain.initial_belief
    assert isinstance(belief, RangeOfMotionBelief)
    forecasts = [tasks.forecast(belief, k) for k in range(4)]
    for day in range(4):
        if day > 0:
            env.start_day(rng)
            tasks.start_day(rng)
            belief = domain.predict_belief(belief)
        assert np.allclose(belief.mean, env.hidden_params)
        assert np.allclose(belief.mean, forecasts[day].mean)
        task = tasks.get_tasks()[0]
        assert isinstance(task, ExerciseTask)
        assert np.allclose(task.goal, tasks.get_goal(0, env.hidden_params))


def test_days_reached_differently_round_to_one_exercise(parts):
    """Beliefs about one day by different arithmetic give one task and belief, close
    to the day's own, while another day's differ."""
    domain, tasks, _ = parts
    belief = domain.initial_belief
    forecast = tasks.forecast(belief, 3)
    halves = RangeOfMotionBelief(
        mean=(forecast.mean * 3 + forecast.mean) / 4 + 1e-15,
        stddev=forecast.stddev,
        low=forecast.low - 1e-15,
        high=forecast.high + 1e-15,
        floor=forecast.floor - 1e-15,
    )
    task, rounded = tasks.round_exercise(2, forecast)
    other_task, other = tasks.round_exercise(2, halves)
    assert isinstance(task, ExerciseTask) and task.exercise == 2
    assert np.array_equal(task.goal, other_task.goal)
    for name in ("mean", "stddev", "low", "high", "floor"):
        assert np.array_equal(getattr(rounded, name), getattr(other, name))
    assert np.allclose(
        tasks.get_midpoint(rounded), tasks.get_midpoint(forecast), atol=5.1e-4
    )
    far, _ = tasks.round_exercise(2, tasks.forecast(belief, 0))
    assert not np.array_equal(task.goal, far.goal) or np.allclose(
        forecast.mean, belief.mean
    )


def test_distance_sums_over_joints(parts):
    """Two days differ by the sum of their limits' differences."""
    _, tasks, _ = parts
    a, b = np.zeros(3), np.array([0.03, 0.0, 0.06])
    assert np.isclose(tasks.get_distance(a, b), 0.09)
