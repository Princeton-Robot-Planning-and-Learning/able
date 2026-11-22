"""Training beliefs that name the site to train for, for a robot that works at
several sites (see envs/multi_site.py).

A unit of sleep trains for one site: it plans that site's routine under the
robot's belief about that site. A belief source chooses the site along with the
belief by returning a SiteBelief focused on it.
"""

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import (
    MarkovSiteTaskDistribution,
    SiteRoutineTaskDistribution,
)
from able.learners.belief_features import get_moments
from able.learners.belief_sources import BeliefSource
from able.learners.model_learner import ModelLearner
from able.learners.rollout_model_learner import RolloutModelLearner
from able.structs import HiddenParamsType, PlannerParamsType

# Larger than any difference between the other features of two beliefs.
_REVISION_SEPARATION = 1e6

Schedule = Sequence[float] | NDArray[np.float64]


class SiteFocusSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Focuses each belief from another source on a site drawn with its share of
    the schedule, so training goes to sites in proportion to how often the robot
    will work there. The schedule may be a function that gives the current
    schedule, for one that changes between days."""

    def __init__(
        self,
        source: BeliefSource[HiddenParamsType, PlannerParamsType],
        schedule: Schedule | Callable[[], Schedule],
    ) -> None:
        self._source = source
        self._schedule = schedule

    @property
    def draws_in_workers(self) -> bool:
        """Whether workers may make the draws: only with a fixed schedule, since
        a worker's copy of a changing one is not kept current, and if the other
        source allows."""
        return not callable(self._schedule) and self._source.draws_in_workers

    def get_schedule(self) -> NDArray[np.float64]:
        """Each site's current share of the schedule."""
        schedule = self._schedule() if callable(self._schedule) else self._schedule
        schedule = np.asarray(schedule, dtype=np.float64)
        return schedule / np.sum(schedule)

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        drawn = self._source.get_training_belief(iteration, belief, params, rng)
        assert isinstance(drawn, SiteBelief), "Need a SiteBelief"
        schedule = self.get_schedule()
        site = int(rng.choice(len(schedule), p=schedule))
        return drawn.focused(site)  # type: ignore[return-value]


class SiteModelLearner(ModelLearner[HiddenParamsType, PlannerParamsType]):
    """A model learner whose training tasks under a focused SiteBelief are that
    site's routine.

    Each unit plans num_training_tasks tasks of the site's routine, those it has
    planned the fewest times first, in the routine's order among ties. Sites
    that share tasks, as sites in a group share their routine, count each
    other's plans, so units for them cover the shared tasks in turn."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Each task planned, by id(), and how many times; holding the task keeps
        # its id from going to another.
        self._num_plans: dict[int, tuple[Any, int]] = {}

    def draw_training_tasks(
        self, belief: Belief[HiddenParamsType], rng: np.random.Generator
    ) -> list[Any]:
        routines = self._task_distribution
        if not isinstance(belief, SiteBelief) or belief.focus is None:
            return super().draw_training_tasks(belief, rng)
        assert isinstance(
            routines, (SiteRoutineTaskDistribution, MarkovSiteTaskDistribution)
        ), "Need routines"
        routine = routines.get_routine(belief.focus)
        counts = [self._num_plans.get(id(t.inner), (t.inner, 0))[1] for t in routine]
        tasks = []
        for _ in range(self._num_training_tasks):
            i = int(np.argmin(counts))
            tasks.append(routine[i])
            counts[i] += 1
            self._num_plans[id(routine[i].inner)] = (routine[i].inner, counts[i])
        return tasks


class SiteRolloutModelLearner(
    RolloutModelLearner[HiddenParamsType, PlannerParamsType],
    SiteModelLearner[HiddenParamsType, PlannerParamsType],
):
    """A rollout model learner whose training tasks under a focused SiteBelief
    are that site's routine, as for SiteModelLearner, each paired with hidden
    parameters for its rollout."""


def make_site_key_features(
    resolution: Sequence[float] | NDArray[np.float64], get_routine: Callable[[int], int]
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """A function from a focused SiteBelief to its site's index, the index of the
    site's routine, and then the mean and standard deviation of each of the
    site's parameters, divided by the resolution and rounded, as in a plan cache
    key. resolution has one entry per mean and then one per standard deviation."""
    resolution = np.asarray(resolution, dtype=np.float64)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, SiteBelief) and belief.focus is not None
        mean, std = get_moments(belief.beliefs[belief.focus])
        entries = np.round(np.concatenate([mean, std]) / resolution)
        return np.concatenate([[belief.focus, get_routine(belief.focus)], entries])

    return get_features


def make_site_moment_features(
    prior: Belief[Any], get_routine: Callable[[int], int]
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """A function from a focused SiteBelief to its site's index, the index of the
    site's routine, and then the mean and standard deviation of each of the
    site's parameters in units of the prior's standard deviation."""
    _, prior_std = get_moments(prior)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, SiteBelief) and belief.focus is not None
        mean, std = get_moments(belief.beliefs[belief.focus])
        site = [belief.focus, get_routine(belief.focus)]
        return np.concatenate([site, mean / prior_std, std / prior_std])

    return get_features


def make_site_revision_features(
    prior: Belief[Any], get_num_revisions: Callable[[int], int]
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """A function from a focused SiteBelief to its site's index, the mean and
    standard deviation of each of the site's parameters in units of the prior's
    standard deviation, and how many times the site's routine has been revised,
    scaled so that beliefs about a site before and after a revision differ more
    in some feature than any beliefs between revisions."""
    _, prior_std = get_moments(prior)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, SiteBelief) and belief.focus is not None
        mean, std = get_moments(belief.beliefs[belief.focus])
        revisions = _REVISION_SEPARATION * get_num_revisions(belief.focus)
        return np.concatenate(
            [[belief.focus], mean / prior_std, std / prior_std, [revisions]]
        )

    return get_features


def make_site_block_features(
    prior: Belief[Any], get_block_ids: Callable[[int], Sequence[int]]
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """A function from a focused SiteBelief to its site's index, the index of
    each block of the site's routine, and then the mean and standard deviation
    of each of the site's parameters in units of the prior's standard
    deviation."""
    _, prior_std = get_moments(prior)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, SiteBelief) and belief.focus is not None
        mean, std = get_moments(belief.beliefs[belief.focus])
        blocks = [belief.focus, *get_block_ids(belief.focus)]
        return np.concatenate([blocks, mean / prior_std, std / prior_std])

    return get_features


def make_site_usage_features(
    prior: Belief[Any],
    get_routine: Callable[[int], int],
    get_usage: Callable[[int], NDArray[np.float64]],
) -> Callable[[Belief[Any]], NDArray[np.float64]]:
    """A function from a focused SiteBelief to its site's index, the index of the
    site's routine, how much the routine uses each parameter (get_usage), and
    then the mean and standard deviation of each of the site's parameters in
    units of the prior's standard deviation."""
    _, prior_std = get_moments(prior)

    def get_features(belief: Belief[Any]) -> NDArray[np.float64]:
        assert isinstance(belief, SiteBelief) and belief.focus is not None
        mean, std = get_moments(belief.beliefs[belief.focus])
        head = [belief.focus, get_routine(belief.focus)]
        usage = get_usage(belief.focus)
        return np.concatenate([head, usage, mean / prior_std, std / prior_std])

    return get_features


Transfer = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def make_block_overlap_transfer(num_blocks: int, transfer: Transfer) -> Transfer:
    """Transfer from the features after the blocks' indices, times the share of
    blocks two routines have in common, since plans serve only the tasks they
    were planned for, for features from make_site_block_features() of routines
    with num_blocks blocks of equal size."""

    def overlap_transfer(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        blocks = slice(1, 1 + num_blocks)
        shared = (a[:, None, blocks] == b[None, :, blocks]).mean(axis=2)
        return transfer(a[:, 1 + num_blocks :], b[:, 1 + num_blocks :]) * shared

    return overlap_transfer


def make_same_routine_transfer(num_blocks: int, transfer: Transfer) -> Transfer:
    """Transfer from the features after the blocks' indices, but only between
    beliefs about the same site with the same blocks, for features from
    make_site_block_features() of routines with num_blocks blocks. Each site's
    routine is then its own skill, as Deliberate Practice assumes."""

    def same_routine_transfer(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        head = slice(0, 1 + num_blocks)
        same = np.all(a[:, None, head] == b[None, :, head], axis=2)
        return transfer(a[:, 1 + num_blocks :], b[:, 1 + num_blocks :]) * same

    return same_routine_transfer


def make_usage_transfer(
    num_params: int, transfer: Transfer, shared: float, other: float
) -> Transfer:
    """Transfer from the moment features, times 1 between sites on the same
    routine and otherwise other + (shared - other) times the cosine similarity of
    the routines' usage of the parameters, for features from
    make_site_usage_features() with num_params parameters."""

    def usage_transfer(
        a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        usage = slice(2, 2 + num_params)
        norm_a = np.linalg.norm(a[:, usage], axis=1)[:, None]
        norm_b = np.linalg.norm(b[:, usage], axis=1)[None, :]
        cosine = (a[:, usage] @ b[:, usage].T) / np.maximum(norm_a * norm_b, 1e-12)
        same = a[:, None, 1] == b[None, :, 1]
        factor = np.where(same, 1.0, other + (shared - other) * cosine)
        return transfer(a[:, 2 + num_params :], b[:, 2 + num_params :]) * factor

    return usage_transfer
