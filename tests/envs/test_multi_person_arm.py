"""Tests for envs/multi_person_arm.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.envs.multi_person_arm import (
    MultiPersonArmModel,
    MultiPersonLimitUpdater,
    RoutineTaskDistribution,
    ScheduleTaskDistribution,
    drift_people_limits,
    get_block,
    get_person_plan_features,
    make_multi_person_domain,
    widen_people_belief,
)
from able.envs.two_link_arm import JointQuery
from able.structs import Task

LIMITS = np.array([-1.0, 1.0, -1.0, 1.0, -2.0, 0.5, -0.5, 2.0])


def test_model_uses_the_persons_limits():
    """Tests that motion is clipped by the limits of the person in the state."""
    model = MultiPersonArmModel(num_people=2)
    rng = np.random.default_rng(0)
    action = np.array([0.3, -0.3])
    first = model.sample_next_state(np.array([0.9, -0.9, 0.0]), action, LIMITS, rng)
    np.testing.assert_allclose(first, [1.0, -1.0, 0.0])
    second = model.sample_next_state(np.array([0.4, -0.4, 1.0]), action, LIMITS, rng)
    np.testing.assert_allclose(second, [0.5, -0.5, 1.0])
    actions = np.tile(action, (1, model.horizon, 1))
    states = model.rollout_batch(np.array([0.0, 0.0, 1.0]), actions, LIMITS[None], rng)
    np.testing.assert_allclose(states[0, 0, -1], [0.5, -0.5])
    task = Task(initial_state=np.array([0.0, 0.0, 1.0]), goal=np.array([2.0, 0.0]))
    assert model.check_goal(task, np.array([0.0, 0.0, 1.0]), LIMITS)
    np.testing.assert_allclose(model.get_cost_lower_bound(task, LIMITS[None]), [0.0])
    with pytest.raises(NotImplementedError):
        model.sample_response(JointQuery(joint=0, angle=0.0), LIMITS, rng)


def test_updater_narrows_only_the_moving_person():
    """Tests that a step narrows the belief about the person who moved."""
    belief = IntervalBelief(np.tile([[-2.0, 2.0]], (8, 1)))
    updater = MultiPersonLimitUpdater()
    updated = updater.update_from_transition(
        belief,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.3, 0.0]),
        np.array([0.3, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(updated.intervals[get_block(0)], [[-2.0, 2.0]] * 4)
    np.testing.assert_allclose(updated.intervals[get_block(1)][1], [0.3, 2.0])
    assert updater.update_from_queries(belief, []) is belief


def test_plan_features_use_the_persons_intervals():
    """Tests that plan features are the goal and the person's intervals."""
    belief = IntervalBelief(
        np.reshape(np.stack([LIMITS - 0.1, LIMITS + 0.1], 1), (8, 2))
    )
    task = Task(initial_state=np.array([0.0, 0.0, 1.0]), goal=np.array([1.0, 1.0]))
    features = get_person_plan_features(task, belief)
    np.testing.assert_allclose(features[:2], [1.0, 1.0])
    np.testing.assert_allclose(features[2:6], LIMITS[4:])
    np.testing.assert_allclose(features[6:], 0.2)


def test_schedule_task_distribution():
    """Tests that tasks follow the schedule and stay within the person's limits."""
    model = MultiPersonArmModel(num_people=2)
    tasks = ScheduleTaskDistribution(model, LIMITS, [0.0, 1.0])
    rng = np.random.default_rng(0)
    assert all(tasks.sample(rng).initial_state[2] == 1.0 for _ in range(10))
    assert tasks.for_person(0).sample(rng).initial_state[2] == 0.0
    np.testing.assert_allclose(tasks.given(LIMITS).schedule, [0.0, 1.0])


def test_routines_and_domain():
    """Tests that the domain's belief contains the truth, that ward-mates are
    similar, and that everyone in a ward practices the same routine."""
    # pylint does not narrow a domain's fields with isinstance.
    # pylint: disable=no-member
    domain = make_multi_person_domain(
        seed=0, num_wards=2, people_per_ward=2, ward_spread=0.1, routine_size=3
    )
    truth = domain.real_env.hidden_params  # type: ignore[attr-defined]
    assert domain.initial_belief.contains(truth)  # type: ignore[attr-defined]
    people = truth.reshape(4, 4)
    assert np.all(np.abs(people[0] - people[1]) <= 0.2 + 1e-9)
    routines = domain.task_distribution
    assert isinstance(routines, RoutineTaskDistribution)
    assert len(routines.get_tasks()) == 4 * 3
    assert routines.get_ward(1) == 0 and routines.get_ward(2) == 1
    goals = [[t.goal for t in routines.get_routine(p)] for p in range(4)]
    np.testing.assert_array_equal(goals[0], goals[1])
    assert routines.schedule.sum() == pytest.approx(1.0)
    task = routines.sample(np.random.default_rng(0))
    assert any(task is t for t in routines.get_routine(int(task.initial_state[2])))


def test_routine_revisions():
    """Tests that a new day revises routines at the revision rate, giving each
    revised routine new goals and a new index, and that the tasks follow."""
    # pylint does not narrow a domain's fields with isinstance.
    # pylint: disable=no-member
    fixed = make_multi_person_domain(
        seed=0, num_wards=2, people_per_ward=2, routine_size=3
    ).task_distribution
    assert isinstance(fixed, RoutineTaskDistribution)
    tasks = fixed.get_tasks()
    fixed.start_day(np.random.default_rng(0))
    assert fixed.get_tasks() == tasks
    domain = make_multi_person_domain(
        seed=0, num_wards=2, people_per_ward=2, routine_size=3, revision_rate=1.0
    )
    routines = domain.task_distribution
    assert isinstance(routines, RoutineTaskDistribution)
    old_goals = [t.goal for t in routines.get_tasks()]
    np.testing.assert_array_equal(old_goals, [t.goal for t in tasks])
    assert [routines.get_routine_id(p) for p in range(4)] == [0, 0, 1, 1]
    routines.start_day(np.random.default_rng(0))
    assert [routines.get_routine_id(p) for p in range(4)] == [2, 2, 3, 3]
    assert [routines.get_num_revisions(p) for p in range(4)] == [1] * 4
    new_goals = [t.goal for t in routines.get_tasks()]
    assert len(new_goals) == len(old_goals)
    assert not any(np.allclose(a, b) for a, b in zip(old_goals, new_goals))
    np.testing.assert_array_equal(
        [t.goal for t in routines.get_routine(1)], new_goals[3:6]
    )
    assert [int(t.initial_state[2]) for t in routines.get_tasks()] == sorted(
        [0, 1, 2, 3] * 3
    )


def test_drift():
    """Tests that drift moves each person's limits by at most the drift, within
    the prior, and that the predicted belief widens to keep the truth."""
    # pylint does not narrow a domain's fields with isinstance.
    # pylint: disable=no-member
    domain = make_multi_person_domain(seed=0, num_wards=2, people_per_ward=2, drift=0.1)
    assert domain.predict_belief is not None
    truth = domain.real_env.hidden_params.copy()  # type: ignore[attr-defined]
    domain.real_env.start_day(np.random.default_rng(0))
    moved = domain.real_env.hidden_params  # type: ignore[attr-defined]
    assert not np.array_equal(moved, truth)
    assert np.all(np.abs(moved - truth) <= 0.1)
    belief = domain.predict_belief(domain.initial_belief)
    assert isinstance(belief, IntervalBelief) and belief.contains(moved)
    edge = np.tile([-np.pi, 0.5, -np.pi, 0.5], 2)
    drifted = drift_people_limits(edge, np.random.default_rng(0), 1.0)
    prior = np.tile([[-np.pi, -0.5], [0.5, np.pi]] * 2, (2, 1))
    assert np.all((drifted >= prior[:, 0]) & (drifted <= prior[:, 1]))
    widened = widen_people_belief(IntervalBelief(prior), 0.5)
    np.testing.assert_allclose(widened.intervals, prior)
    assert make_multi_person_domain(seed=0).predict_belief is None
