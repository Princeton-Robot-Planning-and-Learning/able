"""Tests for envs/multi_site.py."""

import numpy as np
import pytest

from able.envs.multi_site import (
    ImaginedDay,
    MarkovSiteTaskDistribution,
    MultiSiteModel,
    SiteQuery,
    SiteRoutineTaskDistribution,
    SiteState,
    make_site_task,
)
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import ArmTask, JointQuery, TwoLinkArmModel

WIDE = np.array([-2.0, 2.0, -2.0, 2.0])
NARROW = np.array([-0.2, 0.2, -0.2, 0.2])


def _task(goal: float) -> ArmTask:
    return ArmTask(initial_state=np.zeros(2), goal=np.array([goal, 0.0]))


def test_model_uses_each_sites_parameters():
    """Tests that steps, costs, and goals use the hidden parameters of the state's
    site, and that the environment runs on site states."""
    model = MultiSiteModel(TwoLinkArmModel())
    limits = np.stack([WIDE, NARROW])
    task = make_site_task(1, _task(2.0))
    assert task.site == 1 and task.goal is task.inner.goal
    assert model.horizon == TwoLinkArmModel().horizon
    action = np.array([0.5, 0.5])
    rng = np.random.default_rng(0)
    for site, expected in ((0, 0.5), (1, 0.2)):
        state = model.sample_next_state(
            SiteState(site, np.zeros(2)), action, limits, rng
        )
        assert state.site == site
        np.testing.assert_allclose(state.inner, [expected, expected])
    inner, state = TwoLinkArmModel(), SiteState(1, np.zeros(2))
    assert model.get_cost(task, state, action, state, limits) == inner.get_cost(
        task.inner, state.inner, action, state.inner, NARROW
    )
    assert model.get_terminal_cost(task, state, limits) == inner.get_terminal_cost(
        task.inner, state.inner, NARROW
    )
    assert model.check_goal(task, state, limits) == inner.check_goal(
        task.inner, state.inner, NARROW
    )
    env = SimulatedEnvironment(model, limits, rng)
    assert env.reset(task) == SiteState(1, task.inner.initial_state)
    assert env.step(action).state.site == 1
    query = JointQuery(joint=1, angle=1.0)
    assert model.sample_response(SiteQuery(0, query), limits, rng)
    assert not model.sample_response(SiteQuery(1, query), limits, rng)
    with pytest.raises(AssertionError):
        model.sample_response(query, limits, rng)


def test_routines():
    """Tests that sites in a group share their group's routine, that tasks follow
    the schedule, and that revisions replace a group's routine with a new index."""
    routines = [[_task(1.0), _task(1.5)], [_task(-1.0), _task(-1.5)]]
    drawn = [_task(0.5), _task(0.7)]
    tasks = SiteRoutineTaskDistribution(
        routines, [0, 1, 1], [0.0, 0.0, 1.0], 1.0, lambda group, rng: drawn
    )
    assert len(tasks.get_tasks()) == 6
    assert [tasks.get_group(s) for s in range(3)] == [0, 1, 1]
    assert [t.inner for t in tasks.get_routine(2)] == routines[1]
    assert [t.site for t in tasks.get_routine(2)] == [2, 2]
    np.testing.assert_allclose(tasks.schedule, [0.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    assert all(tasks.sample(rng).site == 2 for _ in range(10))
    assert [tasks.get_routine_id(s) for s in range(3)] == [0, 1, 1]
    tasks.start_day(rng)
    assert [tasks.get_routine_id(s) for s in range(3)] == [2, 3, 3]
    assert [tasks.get_num_revisions(s) for s in range(3)] == [1, 1, 1]
    assert [t.inner for t in tasks.get_routine(0)] == drawn
    fixed = SiteRoutineTaskDistribution(routines, [0, 1], [1.0, 1.0])
    before = fixed.get_tasks()
    fixed.start_day(rng)
    assert fixed.get_tasks() == before
    assert fixed.site is None


def test_imagined_day():
    """Tests that an imagined day draws its own site with the schedule and keeps
    its tasks there."""
    routines = [[_task(1.0), _task(1.5)], [_task(-1.0), _task(-1.5)]]
    rng = np.random.default_rng(0)
    tasks = SiteRoutineTaskDistribution(routines, [0, 1], [0.5, 0.5], rng=rng)
    day = ImaginedDay(tasks)
    sites = set()
    for _ in range(10):
        site = day.draw_site(rng)
        assert all(day.sample(rng).site == site for _ in range(3))
        sites.add(site)
    assert sites == {0, 1}


def test_routines_one_site_a_day():
    """Tests that with a random stream the robot works at one site each day, drawn
    again each new day."""
    routines = [[_task(1.0), _task(1.5)], [_task(-1.0), _task(-1.5)]]
    rng = np.random.default_rng(0)
    tasks = SiteRoutineTaskDistribution(routines, [0, 1], [0.5, 0.5], rng=rng)
    sites = []
    for _ in range(10):
        site = tasks.site
        assert site is not None
        assert all(tasks.sample(rng).site == site for _ in range(5))
        sites.append(site)
        tasks.start_day(rng)
    assert set(sites) == {0, 1}


def test_markov_sites():
    """Tests that sites that name a block share its tasks, that the day's tasks
    are the day's site's routine, that the schedule is the previous site's row
    of the chain, and that revisions replace blocks with new indices."""
    goals = iter(np.linspace(0.1, 2.0, 20))

    def draw_block(key, rng):
        del key, rng
        return [_task(next(goals))]

    transition = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    rng = np.random.default_rng(0)
    tasks = MarkovSiteTaskDistribution(
        [["a", "b"], ["a", "c"], ["d", "c"]],
        draw_block,
        transition,
        rng,
        initial_site=0,
    )
    assert tasks.num_sites == 3
    assert tasks.previous_site == 0 and tasks.site == 1
    np.testing.assert_allclose(tasks.schedule, [0.0, 1.0, 0.0])
    assert [t.site for t in tasks.get_tasks()] == [1, 1]
    assert tasks.get_tasks() == tuple(tasks.get_routine(1))
    first, second = tasks.get_routine(0), tasks.get_routine(1)
    assert first[0].inner is second[0].inner and first[0].site == 0
    assert first[1].inner is not second[1].inner
    assert [tasks.get_block_ids(s) for s in range(3)] == [(0, 1), (0, 2), (3, 2)]
    assert all(tasks.sample(rng).site == 1 for _ in range(5))
    tasks.start_day(rng)
    assert tasks.previous_site == 1 and tasks.site == 2
    assert tasks.get_routine(0)[0].inner is first[0].inner
    revised = MarkovSiteTaskDistribution(
        [["a"], ["a"]], draw_block, np.full((2, 2), 0.5), rng, revision_rate=1.0
    )
    before = revised.get_routine(0)[0].inner
    revised.start_day(rng)
    assert revised.get_routine(0)[0].inner is not before
    assert revised.get_routine(1)[0].inner is revised.get_routine(0)[0].inner
    assert revised.get_block_ids(1) == (1,)
    assert [revised.get_num_revisions(s) for s in range(2)] == [1, 1]
