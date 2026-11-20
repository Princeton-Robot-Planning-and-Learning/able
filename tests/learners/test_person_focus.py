"""Tests for learners/person_focus.py."""

from typing import Any

import numpy as np

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.multi_person_arm import RoutineTaskDistribution, make_multi_person_domain
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR
from able.learners.belief_sources import BeliefSource, CurrentBeliefSource
from able.learners.person_focus import (
    PersonBelief,
    PersonFocusSource,
    PersonModelLearner,
    get_same_person_transfer,
    make_person_features,
    make_person_key_features,
    make_person_moment_transfer,
    make_person_revision_features,
    make_person_routine_features,
    make_routine_transfer,
    make_same_person_transfer,
)
from able.planners.random_shooting import RandomShootingPlanner
from able.structs import Budget


def _two_people() -> IntervalBelief:
    return IntervalBelief(np.vstack([JOINT_LIMIT_PRIOR, JOINT_LIMIT_PRIOR * 0.5]))


def test_person_belief_equality():
    """Tests that person beliefs differ by person."""
    intervals = _two_people().intervals
    assert PersonBelief(intervals, 0) == PersonBelief(intervals, 0)
    assert PersonBelief(intervals, 0) != PersonBelief(intervals, 1)
    assert hash(PersonBelief(intervals, 1)) == hash(PersonBelief(intervals, 1))


def test_focus_source_follows_the_schedule():
    """Tests that the focus source names people by the schedule."""
    source: BeliefSource[Any, Any] = PersonFocusSource(CurrentBeliefSource(), [0, 1])
    rng = np.random.default_rng(0)
    belief = source.get_training_belief(1, _two_people(), None, rng)
    assert isinstance(belief, PersonBelief) and belief.person == 1


def test_features_and_transfer():
    """Tests that features start with the person and that transfer ignores who
    the person is, unless it is the same-person transfer."""
    intervals = _two_people().intervals
    features = make_person_features(IntervalBelief(JOINT_LIMIT_PRIOR))
    a, b = features(PersonBelief(intervals, 0)), features(PersonBelief(intervals, 1))
    assert a[0] == 0 and b[0] == 1
    routine_features = make_person_routine_features(
        IntervalBelief(JOINT_LIMIT_PRIOR), lambda person: 7
    )(PersonBelief(intervals, 1))
    np.testing.assert_array_equal(routine_features[[0, 1]], [1, 7])
    np.testing.assert_array_equal(routine_features[2:], b[1:])
    keys = make_person_key_features(0.1, lambda person: 7)(PersonBelief(intervals, 1))
    assert keys[1] == 7
    np.testing.assert_allclose(keys[2:6], np.round(intervals[4:].mean(axis=1) / 0.1))
    routine_rows = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    shared = make_routine_transfer(lambda x, y: np.ones((len(x), len(y))))
    np.testing.assert_array_equal(shared(routine_rows, routine_rows), np.eye(2))
    rows = np.array([[0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(get_same_person_transfer(rows, rows), np.eye(2))
    anyone = make_person_moment_transfer(lambda x, y: np.ones((len(x), len(y))))
    np.testing.assert_array_equal(anyone(rows, rows), np.ones((2, 2)))
    halves = make_same_person_transfer(lambda x, y: np.full((len(x), len(y)), 0.5))
    np.testing.assert_array_equal(halves(rows, rows), 0.5 * np.eye(2))


def test_revision_features():
    """Tests that beliefs about a person before and after a revision of their
    routine are farther apart than beliefs that differ only in their moments."""
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    revisions = {0: 0}
    get_features = make_person_revision_features(prior, lambda person: revisions[0])
    intervals = _two_people().intervals
    before = get_features(PersonBelief(intervals, 0))
    np.testing.assert_array_equal(
        before[:-1], make_person_features(prior)(PersonBelief(intervals, 0))
    )
    revisions[0] = 1
    after = get_features(PersonBelief(intervals, 0))
    other = get_features(PersonBelief(intervals * 0.1, 1))
    assert np.max(np.abs(after - before)) > np.max(np.abs(after - other))


def test_learner_trains_the_named_person_through_the_routine():
    """Tests that units for people in one ward plan the ward's routine goals in
    turn, and only for the person each unit names."""
    # pylint does not narrow a domain's fields with isinstance.
    # pylint: disable=no-member
    domain = make_multi_person_domain(
        seed=0, num_wards=2, people_per_ward=2, routine_size=4
    )
    routines = domain.task_distribution
    assert isinstance(routines, RoutineTaskDistribution)
    learner: PersonModelLearner[Any, Any] = PersonModelLearner(
        RandomShootingPlanner(domain.env_model),  # type: ignore[arg-type]
        routines,
        CurrentBeliefSource(),
        Budget(),
        num_training_tasks=3,
    )
    belief: Belief[Any] = domain.initial_belief
    assert isinstance(belief, IntervalBelief)
    rng = np.random.default_rng(0)
    first = learner.draw_training_tasks(PersonBelief(belief.intervals, 0), rng)
    second = learner.draw_training_tasks(PersonBelief(belief.intervals, 1), rng)
    routine = routines.get_routine(1)
    assert [t.initial_state[2] for t in first] == [0.0] * 3
    np.testing.assert_array_equal(
        [t.goal for t in second], [routine[3].goal, routine[0].goal, routine[1].goal]
    )
    assert len(learner.draw_training_tasks(belief, rng)) == len(routines.get_tasks())
