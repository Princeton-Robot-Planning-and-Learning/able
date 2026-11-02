"""Tests for approaches.py."""

import numpy as np

from able.approaches import make_dreaming_robot, make_scratch_robot
from able.envs.two_link_arm import make_domain
from able.structs import Budget


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
    assert not robot.planner_params.warm_starts
    robot.sleep(Budget(max_iterations=1), np.random.default_rng(0))
    assert robot.planner_params.warm_starts
