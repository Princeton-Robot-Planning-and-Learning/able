"""Tests for learners/site_focus.py."""

from typing import Any

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.beliefs.particles import ParticleBelief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import (
    MarkovSiteTaskDistribution,
    MultiSiteModel,
    SiteRoutineTaskDistribution,
)
from able.envs.two_link_arm import ArmTask, TwoLinkArmModel
from able.learners.belief_sources import BeliefSource, CurrentBeliefSource
from able.learners.site_focus import (
    SiteFocusSource,
    SiteModelLearner,
    SiteRolloutModelLearner,
    make_block_overlap_transfer,
    make_same_routine_transfer,
    make_site_block_features,
    make_site_key_features,
    make_site_moment_features,
    make_site_revision_features,
)
from able.models.exact_match_cache import ExactMatchCache
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.planners.sites import MultiSitePlanner
from able.structs import Budget


def _particles(mean: float, spread: float) -> ParticleBelief:
    values = mean + spread * np.array([[-1.0, -1.0], [1.0, 1.0]])
    return ParticleBelief(values, np.zeros(2), np.ones(2))


def _task(goal: float) -> ArmTask:
    return ArmTask(initial_state=np.zeros(2), goal=np.array([goal, 0.0]))


def test_focus_source_follows_the_schedule():
    """Tests that the focus source focuses beliefs on sites by the schedule, a
    fixed one or the current one, and lets workers draw only with a fixed one."""
    source: BeliefSource[Any, Any] = SiteFocusSource(CurrentBeliefSource(), [0, 1])
    belief = SiteBelief([_particles(0.5, 0.1), _particles(0.5, 0.1)])
    drawn = source.get_training_belief(1, belief, None, np.random.default_rng(0))
    assert isinstance(drawn, SiteBelief) and drawn.focus == 1
    assert source.draws_in_workers
    schedule = [[1.0, 0.0]]
    changing: BeliefSource[Any, Any] = SiteFocusSource(
        CurrentBeliefSource(), lambda: schedule[0]
    )
    assert not changing.draws_in_workers
    rng = np.random.default_rng(0)
    assert changing.get_training_belief(1, belief, None, rng).focus == 0
    schedule[0] = [0.0, 1.0]
    assert changing.get_training_belief(1, belief, None, rng).focus == 1


def test_learner_trains_the_focused_sites_routine_in_turn():
    """Tests that units for sites in one group roll out or plan the group's
    routine in turn, rollouts each with hidden parameters, and that the
    learners take rows from them."""
    routines = SiteRoutineTaskDistribution(
        [[_task(1.0), _task(1.2), _task(1.4)]], [0, 0], [1.0, 1.0]
    )
    planner = MultiSitePlanner(
        RandomShootingPlanner(
            TwoLinkArmModel(), RandomShootingParams(model=ExactMatchCache())
        )
    )
    learner: SiteRolloutModelLearner[Any, Any] = SiteRolloutModelLearner(
        planner,
        routines,
        CurrentBeliefSource(),
        Budget(),
        MultiSiteModel(TwoLinkArmModel()),
        num_training_tasks=2,
    )
    limits = IntervalBelief(np.tile([[-2.0, 2.0]], (4, 1)))
    belief = SiteBelief([limits, limits])
    rng = np.random.default_rng(0)
    first = [t for t, _ in learner.draw_training_tasks(belief.focused(0), rng)]
    second = [t for t, _ in learner.draw_training_tasks(belief.focused(1), rng)]
    assert [t.site for t in first] == [0, 0] and [t.site for t in second] == [1, 1]
    goals = [t.goal[0] for t in first + second]
    assert goals == [1.0, 1.2, 1.4, 1.0]
    assert len(learner.draw_training_tasks(belief, rng)) == 6
    rows = learner.plan_training_tasks(
        belief.focused(0), planner.get_default_params(), rng
    )
    assert len(rows) <= 2
    plans: SiteModelLearner[Any, Any] = SiteModelLearner(
        planner, routines, CurrentBeliefSource(), Budget(), num_training_tasks=3
    )
    tasks = plans.draw_training_tasks(belief.focused(1), rng)
    assert [t.site for t in tasks] == [1, 1, 1]
    rows = plans.plan_training_tasks(
        belief.focused(1), planner.get_default_params(), rng
    )
    assert len(rows) == 3


def test_learner_counts_tasks_that_sites_share():
    """Tests that a unit for a site plans the tasks of its routine planned the
    fewest times, counting plans for other sites that share them."""
    goals = iter([1.0, 1.2, 1.4])
    routines = MarkovSiteTaskDistribution(
        [["shared", "first"], ["shared", "second"]],
        lambda key, rng: [_task(next(goals))],
        np.full((2, 2), 0.5),
        np.random.default_rng(0),
    )
    planner = MultiSitePlanner(
        RandomShootingPlanner(
            TwoLinkArmModel(), RandomShootingParams(model=ExactMatchCache())
        )
    )
    learner: SiteModelLearner[Any, Any] = SiteModelLearner(
        planner, routines, CurrentBeliefSource(), Budget(), num_training_tasks=1
    )
    limits = IntervalBelief(np.tile([[-2.0, 2.0]], (4, 1)))
    belief = SiteBelief([limits, limits])
    rng = np.random.default_rng(0)
    planned = [
        learner.draw_training_tasks(belief.focused(site), rng)[0]
        for site in (0, 1, 1, 0)
    ]
    assert [(t.site, t.goal[0]) for t in planned] == [
        (0, 1.0),
        (1, 1.4),
        (1, 1.0),
        (0, 1.2),
    ]


def test_block_features_and_transfer():
    """Tests that block features name the site and its blocks, that transfer
    counts the share of blocks in common, and that the same-routine transfer
    counts only the same site with the same blocks."""
    blocks = {0: (4, 5), 1: (4, 6)}
    prior = _particles(0.5, 0.2)
    get_features = make_site_block_features(prior, lambda site: blocks[site])
    belief = SiteBelief([_particles(0.5, 0.1), _particles(0.5, 0.1)])
    features = np.array([get_features(belief.focused(s)) for s in (0, 1)])
    np.testing.assert_allclose(features[1], [1, 4, 6, 2.5, 2.5, 0.5, 0.5])

    def same_moments(a, b):
        return np.ones((len(a), len(b)))

    overlap = make_block_overlap_transfer(2, same_moments)(features, features)
    np.testing.assert_allclose(overlap, [[1.0, 0.5], [0.5, 1.0]])
    same = make_same_routine_transfer(2, same_moments)(features, features)
    np.testing.assert_allclose(same, np.eye(2))


def test_features():
    """Tests the key, moment, and revision features of a focused belief."""
    belief = SiteBelief([_particles(0.2, 0.0), _particles(0.5, 0.1)], focus=1)
    keys = make_site_key_features([0.1, 0.1, 0.05, 0.05], lambda site: 7)(belief)
    np.testing.assert_allclose(keys, [1, 7, 5, 5, 2, 2])
    prior = _particles(0.5, 0.2)
    moments = make_site_moment_features(prior, lambda site: 3)(belief)
    np.testing.assert_allclose(moments, [1, 3, 2.5, 2.5, 0.5, 0.5])
    revisions = {1: 0}
    get_revision_features = make_site_revision_features(
        prior, lambda site: revisions[site]
    )
    before = get_revision_features(belief)
    np.testing.assert_allclose(before[:-1], [1, 2.5, 2.5, 0.5, 0.5])
    revisions[1] = 1
    after = get_revision_features(belief)
    assert np.max(np.abs(after - before)) > 1e3
