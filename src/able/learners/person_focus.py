"""Training beliefs that name the person to train for, for a robot that assists
several people (see envs/multi_person_arm.py).

A unit of sleep trains for one person: it plans that person's tasks under the
robot's belief about that person. A belief source chooses the person along with
the belief by returning a PersonBelief.
"""

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.multi_person_arm import (
    RoutineTaskDistribution,
    ScheduleTaskDistribution,
    get_block,
)
from able.learners.belief_features import get_moments
from able.learners.belief_sources import BeliefSource
from able.learners.model_learner import ModelLearner
from able.structs import HiddenParamsType, PlannerParamsType

# Larger than any difference between the other features of two beliefs.
_REVISION_SEPARATION = 1e6


class PersonBelief(IntervalBelief):
    """An interval belief over every person's limits, and the person a unit of
    sleep should train for."""

    def __init__(self, intervals: ArrayLike, person: int) -> None:
        super().__init__(intervals)
        self.person = person

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PersonBelief)
            and other.person == self.person
            and super().__eq__(other)
        )

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.person))


class PersonFocusSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Names a person for each belief from another source, drawn with the
    person's share of the schedule, so training goes to people in proportion to
    how often the robot will help them."""

    def __init__(
        self,
        source: BeliefSource[HiddenParamsType, PlannerParamsType],
        schedule: Sequence[float] | NDArray[np.float64],
    ) -> None:
        self._source = source
        self._schedule = np.asarray(schedule, dtype=np.float64) / np.sum(schedule)

    @property
    def draws_in_workers(self) -> bool:
        """Whether the other source's draws may be made in workers."""
        return self._source.draws_in_workers

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        drawn = self._source.get_training_belief(iteration, belief, params, rng)
        assert isinstance(drawn, IntervalBelief), "Need an IntervalBelief"
        person = int(rng.choice(len(self._schedule), p=self._schedule))
        return PersonBelief(drawn.intervals, person)  # type: ignore[return-value]


class PersonModelLearner(ModelLearner[HiddenParamsType, PlannerParamsType]):
    """A model learner whose training tasks under a PersonBelief are that
    person's tasks only.

    With routines (RoutineTaskDistribution), each unit plans the next
    num_training_tasks goals of the person's ward's routine, continuing where
    the ward's last unit stopped, so units for people in the same ward cover the
    routine's goals in turn."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._next_goal_by_ward: dict[int, int] = {}

    def draw_training_tasks(
        self, belief: Belief[HiddenParamsType], rng: np.random.Generator
    ) -> list[Any]:
        task_distribution = self._task_distribution
        if not isinstance(belief, PersonBelief):
            return super().draw_training_tasks(belief, rng)
        if isinstance(task_distribution, RoutineTaskDistribution):
            routine = task_distribution.get_routine(belief.person)
            ward = task_distribution.get_ward(belief.person)
            start = self._next_goal_by_ward.get(ward, 0)
            self._next_goal_by_ward[ward] = start + self._num_training_tasks
            return [
                routine[(start + i) % len(routine)]
                for i in range(self._num_training_tasks)
            ]
        assert isinstance(task_distribution, ScheduleTaskDistribution)
        tasks = task_distribution.for_person(belief.person)
        return [
            tasks.given(belief.sample(rng)).sample(rng)
            for _ in range(self._num_training_tasks)
        ]


def make_person_features(
    one_person_prior: IntervalBelief,
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """A function from a PersonBelief to its person's index, then the mean and
    standard deviation of each of that person's limits in units of the prior's
    standard deviation."""
    _, prior_std = get_moments(one_person_prior)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, PersonBelief), "Need a PersonBelief"
        block = IntervalBelief(belief.intervals[get_block(belief.person)])
        mean, std = get_moments(block)
        return np.concatenate([[belief.person], mean / prior_std, std / prior_std])

    return get_features


def make_person_routine_features(
    one_person_prior: IntervalBelief, get_routine: Callable[[int], int]
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """Features from make_person_features() with the index of the person's
    routine inserted after the person's index."""
    get_person_features = make_person_features(one_person_prior)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, PersonBelief), "Need a PersonBelief"
        features = get_person_features(belief)
        return np.insert(features, 1, get_routine(belief.person))

    return get_features


def make_person_revision_features(
    one_person_prior: IntervalBelief, get_num_revisions: Callable[[int], int]
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """Features from make_person_features(), then how many times the person's
    routine has been revised, scaled so that beliefs about a person before and
    after a revision differ more in some feature than any beliefs between
    revisions."""
    get_person_features = make_person_features(one_person_prior)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, PersonBelief), "Need a PersonBelief"
        revisions = get_num_revisions(belief.person)
        return np.append(get_person_features(belief), _REVISION_SEPARATION * revisions)

    return get_features


def get_same_person_transfer(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Transfer of 1 between beliefs about the same person and 0 otherwise, for
    features from make_person_features()."""
    return (a[:, None, 0] == b[None, :, 0]).astype(np.float64)


def make_same_person_transfer(
    transfer: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """The transfer, but only between beliefs about the same person, for features
    whose first entry is the person's index."""

    def same_person_transfer(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return transfer(a, b) * get_same_person_transfer(a, b)

    return same_person_transfer


def make_person_moment_transfer(
    transfer: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Transfer between beliefs about any people from every feature but the
    person's index, for features from make_person_features()."""

    def person_transfer(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return transfer(a[:, 1:], b[:, 1:])

    return person_transfer


def make_person_key_features(
    resolution: float, get_routine: Callable[[int], int]
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """A function from a PersonBelief to its person's index, the index of the
    person's routine, and then the belief entries of a plan cache key for that
    person at the resolution: each interval's midpoint, then each one's width,
    rounded to steps of resolution."""

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, PersonBelief), "Need a PersonBelief"
        intervals = belief.intervals[get_block(belief.person)]
        entries = np.concatenate(
            [intervals.mean(axis=1), intervals[:, 1] - intervals[:, 0]]
        )
        routine = get_routine(belief.person)
        return np.concatenate(
            [[belief.person, routine], np.round(entries / resolution)]
        )

    return get_features


def make_routine_transfer(
    transfer: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Transfer from the features after the routine's index, and only between
    people on the same routine, since plans serve only the goals they were
    planned for, for features from make_person_key_features() or
    make_person_routine_features()."""

    def routine_transfer(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        same_routine = a[:, None, 1] == b[None, :, 1]
        return transfer(a[:, 2:], b[:, 2:]) * same_routine

    return routine_transfer
