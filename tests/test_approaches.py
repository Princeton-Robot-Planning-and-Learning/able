"""Tests for approaches.py."""

import dataclasses

import numpy as np

from able.approaches import (
    make_current_belief_robot,
    make_dreaming_robot,
    make_friction_current_belief_robot,
    make_friction_dreaming_robot,
    make_friction_prior_only_robot,
    make_friction_scratch_robot,
    make_limb_current_belief_robot,
    make_limb_dreaming_robot,
    make_limb_prior_only_robot,
    make_limb_scratch_robot,
    make_prior_only_robot,
    make_scratch_robot,
)
from able.beliefs.interval import IntervalBelief
from able.cycle import CycleBudgets, run_cycles
from able.envs.base import FiniteTaskDistribution
from able.envs.friction.tasks import make_domain as make_friction_domain
from able.envs.limb.tasks import make_domain as make_limb_domain
from able.envs.two_link_arm import make_domain
from able.models.exact_match_cache import ExactMatchCache
from able.models.neural import MLPRegressor
from able.planners.random_shooting import RandomShootingParams, get_plan_features
from able.robot import Robot
from able.structs import Budget

TRUE_LIMITS = np.array([-1.2, 1.4, -2.0, 0.9])


def _budgets(num_dreams: int) -> CycleBudgets:
    return CycleBudgets(
        sleep=Budget(max_iterations=num_dreams),
        information_gathering=Budget(max_iterations=0),
        execution=Budget(max_iterations=1),
        planning=Budget(),
    )


def _make_dreaming_robot(domain) -> Robot:
    budgets = _budgets(num_dreams=0)
    return make_dreaming_robot(
        domain, budgets.information_gathering, budgets.execution, budgets.planning
    )


def _get_stats(robot, domain, num_dreams: int, num_cycles: int, seed: int) -> list:
    """Run cycles and return the planning statistics of every evaluation."""
    task_distribution = domain.task_distribution
    assert isinstance(task_distribution, FiniteTaskDistribution)
    results = run_cycles(
        robot,
        domain.real_env,
        task_distribution,
        list(task_distribution.get_tasks()),
        num_cycles,
        _budgets(num_dreams),
        np.random.default_rng(seed),
    )
    return [e.planning_stats for r in results for e in r.evaluations]


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
    robot = _make_dreaming_robot(make_domain(seed=0))
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


def test_known_limits_always_hit():
    """Tests that every evaluation uses a cached plan when the belief cannot change.

    With known limits, the current belief and every imagined next belief equal the
    real one, so the robots that learn always plan from a cached plan.
    """
    known = IntervalBelief(np.stack([TRUE_LIMITS, TRUE_LIMITS], axis=1))
    domain = dataclasses.replace(
        make_domain(seed=0, true_limits=TRUE_LIMITS), initial_belief=known
    )
    robots = {
        "scratch": make_scratch_robot(domain),
        "current_belief": make_current_belief_robot(domain, Budget()),
        "dreaming": _make_dreaming_robot(domain),
    }
    samples = {}
    for name, robot in robots.items():
        stats = _get_stats(robot, domain, num_dreams=1, num_cycles=2, seed=0)
        hits = [s["used_warm_start"] for s in stats]
        assert all(hits) if name != "scratch" else not any(hits)
        samples[name] = np.mean([s["num_samples"] for s in stats])
    assert samples["dreaming"] < samples["scratch"]


def test_dreaming_beats_current_belief():
    """Tests that dreaming hits the cache more often than training on the current
    belief, and that both plan with fewer samples than planning from scratch.

    Three goals keep the test fast, and 16 dreams cover most of the beliefs the
    robot can reach after executing one of them.
    """
    hits: dict[str, list[float]] = {"current_belief": [], "dreaming": []}
    samples: dict[str, list[float]] = {"scratch": [], **{k: [] for k in hits}}
    for seed in range(3):
        domain = make_domain(seed, num_goals=3)
        robots = {
            "scratch": make_scratch_robot(domain),
            "current_belief": make_current_belief_robot(domain, Budget()),
            "dreaming": _make_dreaming_robot(domain),
        }
        for name, robot in robots.items():
            stats = _get_stats(robot, domain, num_dreams=16, num_cycles=4, seed=seed)
            if name in hits:
                hits[name].extend(s["used_warm_start"] for s in stats)
            samples[name].extend(s["num_samples"] for s in stats)
    assert 0 < np.mean(hits["current_belief"]) < np.mean(hits["dreaming"])
    assert np.mean(hits["dreaming"]) > 0.5
    assert np.mean(samples["current_belief"]) < np.mean(samples["scratch"])
    assert np.mean(samples["dreaming"]) < 0.7 * np.mean(samples["scratch"])


def test_prior_only_plans_help_only_under_the_prior():
    """Tests that an evaluation uses a cached plan exactly when the belief still has
    the prior's key."""
    domain = make_domain(seed=0)
    prior = domain.initial_belief
    assert isinstance(prior, IntervalBelief)
    task_distribution = domain.task_distribution
    assert isinstance(task_distribution, FiniteTaskDistribution)
    robot = make_prior_only_robot(domain, Budget())
    rng = np.random.default_rng(0)
    eval_tasks = [task_distribution.sample(rng) for _ in range(4)]
    results = run_cycles(
        robot, domain.real_env, task_distribution, eval_tasks, 3, _budgets(2), rng
    )
    cache = robot.planner_params.model
    assert isinstance(cache, ExactMatchCache)
    num_hits = 0
    for result in results:
        belief = result.belief
        assert isinstance(belief, IntervalBelief)
        for evaluation in result.evaluations:
            key = cache.get_key(get_plan_features(evaluation.task, belief))
            prior_key = cache.get_key(get_plan_features(evaluation.task, prior))
            hit = evaluation.planning_stats["used_warm_start"] == 1.0
            assert hit == (key == prior_key)
            num_hits += hit
    # With every goal at a limit, the first execution changes the belief.
    assert num_hits == 0


def test_make_limb_scratch_robot():
    """Tests that the limb scratch robot uses the requested MPC settings and never
    changes its parameters."""
    domain = make_limb_domain(seed=0)
    robot = make_limb_scratch_robot(domain, num_rollouts=8, num_particles=4)
    params = robot.planner_params
    assert params.config.sampling.num_rollouts == 8
    assert params.num_particles == 4
    assert robot.belief is domain.initial_belief
    robot.sleep(Budget(max_iterations=1), np.random.default_rng(0))
    assert robot.planner_params is params


def test_make_limb_learning_robots():
    """Tests that the limb robots that learn seed MPC with the given model, sleep,
    and that prior-only learns in its first sleep only."""
    # A short horizon makes every plan fail fast.
    domain = make_limb_domain(seed=0, horizon=5)
    model = MLPRegressor(num_steps=5)
    options = {"model": model, "num_rollouts": 4, "num_particles": 4}
    budgets = _budgets(num_dreams=0)
    robots = [
        make_limb_current_belief_robot(domain, Budget(), **options),
        make_limb_prior_only_robot(domain, Budget(), **options),
        make_limb_dreaming_robot(
            domain,
            budgets.information_gathering,
            budgets.execution,
            Budget(),
            **options,
        ),
    ]
    rng = np.random.default_rng(0)
    for robot in robots:
        assert robot.planner_params.model is model
        assert robot.planner_params.config.sampling.num_rollouts == 4
        robot.sleep(Budget(max_iterations=1), rng)
    prior_only = robots[1]
    params = prior_only.planner_params
    prior_only.sleep(Budget(max_iterations=1), rng)
    assert prior_only.planner_params is params


def test_make_friction_scratch_robot():
    """Tests that the friction scratch robot uses the requested POMCPOW settings
    and never changes its parameters."""
    domain = make_friction_domain(seed=0, num_particles=10)
    robot = make_friction_scratch_robot(domain, num_sims=20, max_depth=3)
    params = robot.planner_params
    assert params.config.num_sims == 20 and params.config.max_depth == 3
    assert robot.belief is domain.initial_belief
    robot.sleep(Budget(max_iterations=1), np.random.default_rng(0))
    assert robot.planner_params is params


def test_make_friction_learning_robots():
    """Tests that the friction robots that learn guide POMCPOW with the given
    model, sleep, and that prior-only learns in its first sleep only."""
    # One push per episode keeps the rollouts short.
    domain = make_friction_domain(seed=0, horizon=1, num_particles=10)
    model = MLPRegressor(num_steps=5)
    options = {"model": model, "num_sims": 10, "max_depth": 2}
    budgets = _budgets(num_dreams=0)
    robots = [
        make_friction_current_belief_robot(domain, Budget(), **options),
        make_friction_prior_only_robot(domain, Budget(), **options),
        make_friction_dreaming_robot(
            domain,
            budgets.information_gathering,
            budgets.execution,
            Budget(),
            **options,
        ),
    ]
    rng = np.random.default_rng(0)
    for robot in robots:
        assert robot.planner_params.model is model
        assert robot.planner_params.config.num_sims == 10
        robot.sleep(Budget(max_iterations=1), rng)
    prior_only = robots[1]
    params = prior_only.planner_params
    prior_only.sleep(Budget(max_iterations=1), rng)
    assert prior_only.planner_params is params
