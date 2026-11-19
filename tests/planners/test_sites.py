"""Tests for planners/sites.py."""

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import MultiSiteModel, SiteState, make_site_task
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import ArmTask, TwoLinkArmModel
from able.episodes import run_episode
from able.models.exact_match_cache import ExactMatchCache
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.planners.sites import MultiSitePlanner, SitePolicy
from able.structs import Budget

LIMITS = np.array([-2.0, 2.0, -2.0, 2.0])


def test_plans_each_site_under_its_belief():
    """Tests that the planner plans a site's task under that site's belief, that
    its policy acts on site states, and that its rows are the single-site
    planner's."""
    inner = RandomShootingPlanner(
        TwoLinkArmModel(), RandomShootingParams(model=ExactMatchCache())
    )
    planner = MultiSitePlanner(inner)
    params = planner.get_default_params()
    known = IntervalBelief(np.stack([LIMITS, LIMITS], axis=1))
    belief = SiteBelief([IntervalBelief(np.tile([[-3.0, 3.0]], (4, 1))), known])
    inner_task = ArmTask(initial_state=np.zeros(2), goal=np.array([1.0, 1.0]))
    task = make_site_task(1, inner_task)
    policy = planner.plan(task, belief, params, Budget(), np.random.default_rng(0))
    assert isinstance(policy, SitePolicy)
    expected = inner.plan(inner_task, known, params, Budget(), np.random.default_rng(0))
    np.testing.assert_allclose(
        policy.get_action(SiteState(1, np.zeros(2))), expected.get_action(np.zeros(2))
    )
    assert policy.get_planning_stats() == expected.get_planning_stats()
    rows = planner.get_plan_examples(task, belief, policy)
    np.testing.assert_allclose(
        rows.x, inner.get_plan_examples(inner_task, known, policy.inner).x
    )
    env = SimulatedEnvironment(
        MultiSiteModel(TwoLinkArmModel()),
        np.stack([LIMITS, LIMITS]),
        np.random.default_rng(0),
    )
    policy = planner.plan(task, belief, params, Budget(), np.random.default_rng(1))
    trajectory = run_episode(env, task, policy)
    assert (
        len(planner.get_rollout_examples(task, belief, policy, trajectory, True)) == 1
    )
    assert (
        len(planner.get_rollout_examples(task, belief, policy, trajectory, False)) == 0
    )
    model = planner.get_model(planner.get_training_params(params))
    assert planner.set_model(params, model).model is model
