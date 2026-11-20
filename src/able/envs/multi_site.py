"""A robot that works at several sites, each with its own hidden parameters.

Any single-site environment becomes a multi-site one here. A state is a
SiteState: the site's index and the single-site state. A task is a SiteTask,
which holds the single-site task for one site. The hidden parameters stack each
site's, so their first axis is the site, and the robot's belief is a SiteBelief
(beliefs/sites.py) with one belief per site. Every site shares the single-site
environment model; only the hidden parameters differ.

Sites come in groups. Sites in a group have similar hidden parameters and do
the same routine of tasks, so what the robot learns at one site can serve the
others in its group.

A site can also be one condition of a single person, whose condition changes
from day to day (see MarkovSiteTaskDistribution). The robot then works at one
site each day, and sites share parts of their routines.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Hashable, Sequence

import numpy as np
from numpy.typing import NDArray

from able.envs.base import EnvironmentModel, FiniteTaskDistribution
from able.structs import StateType, Task


@dataclass(frozen=True)
class SiteState(Generic[StateType]):
    """A single-site state at one site."""

    site: int
    inner: StateType


@dataclass(frozen=True, eq=False)
class SiteTask(Task[SiteState[Any], Any]):
    """A single-site task at one site. Its initial state is the single-site
    task's at the site, and its goal is the single-site task's."""

    inner: Task[Any, Any] = field(kw_only=True)

    @property
    def site(self) -> int:
        """The site the task is at."""
        return int(self.initial_state.site)


def make_site_task(site: int, inner: Task[Any, Any]) -> SiteTask:
    """The single-site task at the site."""
    return SiteTask(SiteState(site, inner.initial_state), inner.goal, inner=inner)


class MultiSiteModel(EnvironmentModel[SiteState[Any], Any, SiteTask, Any, Any, Any]):
    """The single-site model at the site in each state, with that site's hidden
    parameters. Sites take no queries."""

    def __init__(self, inner: EnvironmentModel[Any, Any, Any, Any, Any, Any]) -> None:
        self.inner = inner

    @property
    def horizon(self) -> int:
        return self.inner.horizon

    def sample_next_state(
        self,
        state: SiteState[Any],
        action: Any,
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> SiteState[Any]:
        next_state = self.inner.sample_next_state(
            state.inner, action, hidden_params[state.site], rng
        )
        return SiteState(state.site, next_state)

    def get_cost(
        self,
        task: SiteTask,
        state: SiteState[Any],
        action: Any,
        next_state: SiteState[Any],
        hidden_params: NDArray[np.float64],
    ) -> float:
        return self.inner.get_cost(
            task.inner, state.inner, action, next_state.inner, hidden_params[state.site]
        )

    def get_terminal_cost(
        self,
        task: SiteTask,
        state: SiteState[Any],
        hidden_params: NDArray[np.float64],
    ) -> float:
        return self.inner.get_terminal_cost(
            task.inner, state.inner, hidden_params[state.site]
        )

    def check_goal(
        self,
        task: SiteTask,
        state: SiteState[Any],
        hidden_params: NDArray[np.float64],
    ) -> bool:
        return self.inner.check_goal(task.inner, state.inner, hidden_params[state.site])

    def sample_response(
        self, query: Any, hidden_params: NDArray[np.float64], rng: np.random.Generator
    ) -> Any:
        raise NotImplementedError("Sites take no queries")


class SiteRoutineTaskDistribution(FiniteTaskDistribution[SiteTask]):
    """Tasks from routines: each group of sites has a list of single-site tasks,
    and every site in the group does them all. A task draws a site with its share
    of the schedule, then one of its group's tasks uniformly.

    The routines stay fixed unless revision_rate is above 0. Then at the start
    of each new day, each group's routine is replaced with that probability by
    draw_routine(group, rng).
    """

    def __init__(
        self,
        routines: Sequence[Sequence[Task[Any, Any]]],
        group_of: Sequence[int],
        schedule: Sequence[float] | NDArray[np.float64],
        revision_rate: float = 0.0,
        draw_routine: (
            Callable[[int, np.random.Generator], list[Task[Any, Any]]] | None
        ) = None,
    ) -> None:
        assert revision_rate == 0.0 or draw_routine is not None, "Need draw_routine"
        self._group_of = list(group_of)
        self._schedule = np.asarray(schedule, dtype=np.float64) / np.sum(schedule)
        self._revision_rate = revision_rate
        self._draw_routine = draw_routine
        self._routine_ids = list(range(len(routines)))
        self._num_revisions = [0] * len(routines)
        self._num_routines = len(routines)
        super().__init__(self._set_routines(routines))

    def _set_routines(
        self, routines: Sequence[Sequence[Task[Any, Any]]]
    ) -> list[SiteTask]:
        """Make each site's tasks from its group's routine, and return every
        site's tasks in turn."""
        self._group_routines = [list(tasks) for tasks in routines]
        self._routines = [
            [make_site_task(site, task) for task in routines[group]]
            for site, group in enumerate(self._group_of)
        ]
        return [task for tasks in self._routines for task in tasks]

    @property
    def schedule(self) -> NDArray[np.float64]:
        """Each site's share of the tasks."""
        return self._schedule

    def get_group(self, site: int) -> int:
        """The group a site belongs to."""
        return self._group_of[site]

    def get_routine_id(self, site: int) -> int:
        """The index of a site's current routine: its group's index until the
        group's routine is first revised, and a new index after each revision."""
        return self._routine_ids[self._group_of[site]]

    def get_num_revisions(self, site: int) -> int:
        """How many times the site's group's routine has been revised."""
        return self._num_revisions[self._group_of[site]]

    def get_routine(self, site: int) -> list[SiteTask]:
        """A site's tasks, one per task of its group's routine, in order."""
        return list(self._routines[site])

    def sample(self, rng: np.random.Generator) -> SiteTask:
        site = int(rng.choice(len(self._schedule), p=self._schedule))
        routine = self._routines[site]
        return routine[int(rng.integers(len(routine)))]

    def start_day(self, rng: np.random.Generator) -> None:
        """Revise each group's routine with probability revision_rate."""
        if self._revision_rate == 0.0:
            return
        assert self._draw_routine is not None
        routines = list(self._group_routines)
        for group in range(len(routines)):
            if rng.random() < self._revision_rate:
                routines[group] = self._draw_routine(group, rng)
                self._routine_ids[group] = self._num_routines
                self._num_revisions[group] += 1
                self._num_routines += 1
        super().__init__(self._set_routines(routines))


class MarkovSiteTaskDistribution(FiniteTaskDistribution[SiteTask]):
    """Tasks at one site each day, where the day's site follows a Markov chain, as
    when each site is one condition of a person whose condition changes from day
    to day.

    Each site's routine is made of blocks of single-site tasks, and sites can
    share blocks. Site s does the blocks that site_blocks[s] names by key, in
    order, and draw_block(key, rng) draws a block's tasks, so sites that name the
    same key do the same tasks. Each block has an index, which changes when the
    block is revised.

    The day's tasks are the routine of the day's site. The robot sleeps before
    it learns the day's site, knowing the chain and the previous day's site, so
    the schedule, each site's chance of being the day's, is the previous site's
    row of the transition matrix. At the start of each new day, each block is
    replaced with probability revision_rate, and the chain moves on.
    """

    def __init__(
        self,
        site_blocks: Sequence[Sequence[Hashable]],
        draw_block: Callable[[Hashable, np.random.Generator], list[Task[Any, Any]]],
        transition: NDArray[np.float64],
        rng: np.random.Generator,
        revision_rate: float = 0.0,
        initial_site: int | None = None,
    ) -> None:
        transition = np.asarray(transition, dtype=np.float64)
        num_sites = len(site_blocks)
        assert transition.shape == (num_sites, num_sites), "Need one row per site"
        assert np.allclose(transition.sum(axis=1), 1.0), "Rows must sum to 1"
        self._site_blocks = [list(keys) for keys in site_blocks]
        self._draw_block = draw_block
        self._transition = transition
        self._revision_rate = revision_rate
        self._keys = list(dict.fromkeys(k for keys in site_blocks for k in keys))
        self._blocks = {key: list(draw_block(key, rng)) for key in self._keys}
        self._block_ids = {key: i for i, key in enumerate(self._keys)}
        self._num_revisions = {key: 0 for key in self._keys}
        self._num_block_ids = len(self._keys)
        self._previous_site = (
            int(rng.integers(num_sites)) if initial_site is None else initial_site
        )
        self._site = self._draw_site(rng)
        super().__init__(self._set_routines())

    def _draw_site(self, rng: np.random.Generator) -> int:
        row = self._transition[self._previous_site]
        return int(rng.choice(len(row), p=row))

    def _set_routines(self) -> list[SiteTask]:
        """Make each site's tasks from its blocks, and return the day's site's."""
        self._routines = [
            [make_site_task(site, task) for key in keys for task in self._blocks[key]]
            for site, keys in enumerate(self._site_blocks)
        ]
        return list(self._routines[self._site])

    @property
    def num_sites(self) -> int:
        """The number of sites."""
        return len(self._site_blocks)

    @property
    def draw_block(
        self,
    ) -> Callable[[Hashable, np.random.Generator], list[Task[Any, Any]]]:
        """The function that draws a block's tasks."""
        return self._draw_block

    @property
    def transition(self) -> NDArray[np.float64]:
        """The chance of each site following each site, one row per site."""
        return self._transition

    @property
    def site(self) -> int:
        """The day's site."""
        return self._site

    @property
    def previous_site(self) -> int:
        """The previous day's site."""
        return self._previous_site

    @property
    def schedule(self) -> NDArray[np.float64]:
        """Each site's chance of being the day's, given the previous day's."""
        return self._transition[self._previous_site]

    def get_routine(self, site: int) -> list[SiteTask]:
        """A site's tasks, block by block, in order."""
        return list(self._routines[site])

    def get_block_ids(self, site: int) -> tuple[int, ...]:
        """The index of each of a site's blocks, in order."""
        return tuple(self._block_ids[key] for key in self._site_blocks[site])

    def get_num_revisions(self, site: int) -> int:
        """How many times the site's blocks have been revised, in total."""
        return sum(self._num_revisions[key] for key in self._site_blocks[site])

    def start_day(self, rng: np.random.Generator) -> None:
        """Revise each block with probability revision_rate, then draw the day's
        site from the chain."""
        for key in self._keys:
            if self._revision_rate > 0.0 and rng.random() < self._revision_rate:
                self._blocks[key] = list(self._draw_block(key, rng))
                self._block_ids[key] = self._num_block_ids
                self._num_block_ids += 1
                self._num_revisions[key] += 1
        self._previous_site = self._site
        self._site = self._draw_site(rng)
        super().__init__(self._set_routines())
