"""Tests for approaches.py."""

import numpy as np

from able.approaches import (
    make_current_belief_robot,
    make_dreaming_robot,
    make_prior_only_robot,
    make_scratch_robot,
)
from able.envs.two_link_arm import make_domain
from able.models.exact_match_cache import ExactMatchCache
from able.planners.random_shooting import RandomShootingParams
from able.structs import Budget


def _has_cached_plans(params: RandomShootingParams) -> bool:
    assert isinstance(params.model, ExactMatchCache)
    return bool(params.model.entries)


def test_make_scratch_robot():
    """Tests that the scratch robot plans and never changes its parameters."""
    domain = make_domain(seed=0)
    robot = make_scratch_robot(domain)
    rng = np.random.default_rng(0)
    params = robot.planner_params
    robot.sleep(Budget(max_iterations=2), rng)
    assert robot.planner_params is params
    robot.plan(domain.task_distribution.sample(rng), Budget(), rng)


def test_make_dreaming_robot():
    """Tests that the dreaming robot caches plans while it sleeps."""
    domain = make_domain(seed=0)
    robot = make_dreaming_robot(
        domain,
        gathering_budget=Budget(max_iterations=0),
        execution_budget=Budget(max_iterations=1),
        planning_budget=Budget(),
    )
    assert not _has_cached_plans(robot.planner_params)
    robot.sleep(Budget(max_iterations=1), np.random.default_rng(0))
    assert _has_cached_plans(robot.planner_params)


def test_make_current_belief_robot():
    """Tests that the current-belief robot caches plans while it sleeps."""
    domain = make_domain(seed=0)
    robot = make_current_belief_robot(domain, planning_budget=Budget())
    assert not _has_cached_plans(robot.planner_params)
    robot.sleep(Budget(max_iterations=1), np.random.default_rng(0))
    assert _has_cached_plans(robot.planner_params)


def test_make_prior_only_robot():
    """Tests that the prior-only robot caches plans in its first sleep only."""
    domain = make_domain(seed=0)
    robot = make_prior_only_robot(domain, planning_budget=Budget())
    rng = np.random.default_rng(0)
    robot.sleep(Budget(max_iterations=1), rng)
    params = robot.planner_params
    assert _has_cached_plans(params)
    robot.sleep(Budget(max_iterations=1), rng)
    assert robot.planner_params is params
