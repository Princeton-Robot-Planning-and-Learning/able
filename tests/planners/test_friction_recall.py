"""Tests for friction_recall.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

import dataclasses

import numpy as np
import pytest

from able.approaches import (
    _make_friction_calendar_planner,
    make_friction_calendar_allocation_robot,
    make_friction_calendar_belief_coverage_robot,
    make_friction_calendar_current_belief_robot,
    make_friction_calendar_dreaming_robot,
    make_friction_calendar_scratch_robot,
)
from able.envs.friction.calendar import make_friction_calendar_domain
from able.planners.friction_recall import StoredPush, decode_plan, encode_plan
from able.structs import Budget, PlanningFailure


@pytest.fixture(name="domain", scope="module")
def fixture_domain():
    """A domain whose simulators close after the module."""
    domain = make_friction_calendar_domain(1)
    yield domain
    domain.env_model.close()


def test_plans_survive_encoding():
    """A plan decodes to the pushes it encoded."""
    pushes = [
        StoredPush(0.5, 1.0, None, (1.0, 2.0)),
        StoredPush(-1.0, 0.3, 0.26, (2.0, 1.0)),
    ]
    assert decode_plan(encode_plan(pushes)) == pushes


def test_nothing_stored_fails(domain):
    """With no stored plan, planning while awake fails."""
    planner, tasks = _make_friction_calendar_planner(domain, 0.1, 200, 0, None)
    with pytest.raises(PlanningFailure):
        planner.plan(
            tasks.get_tasks()[0],
            domain.initial_belief,
            planner.get_default_params(),
            Budget(max_iterations=1),
            np.random.default_rng(0),
        )


def test_replay_reaches_the_goal_on_its_day_only(domain, tmp_path):
    """A training plan replays to the goal under its own day's frictions within a
    centimeter of every planned stop, is stored and reused, and its replay strays
    farther under frictions four steps of slick away in every zone."""
    planner, tasks = _make_friction_calendar_planner(domain, 0.1, 500, 0, str(tmp_path))
    task, belief = tasks.get_tasks()[0], domain.initial_belief
    default = planner.get_default_params()
    training = planner.get_training_params(default)
    rng = np.random.default_rng(0)
    policy = planner.plan(task, belief, training, Budget(), rng)
    assert list(tmp_path.glob("routine0/plan_*.json"))
    again = planner.plan(task, belief, training, Budget(), rng)
    assert again.pushes == policy.pushes
    params = dataclasses.replace(
        default,
        model=default.model.fit(planner.get_plan_examples(task, belief, policy)),
        tube=0.01,
    )
    replay = planner.plan(task, belief, params, Budget(max_iterations=1), rng)
    model, state = domain.env_model, task.initial_state
    truth = domain.real_env.hidden_params
    while not state.ended and state.pushes < model.horizon:
        state = model.sample_next_state(state, replay.get_action(state), truth, rng)
    assert state.outcome == "goal"
    far = belief.scaled(np.full(len(truth), np.exp(-0.4)))
    with pytest.raises(PlanningFailure):
        planner.plan(task, far, params, Budget(max_iterations=1), rng)


@pytest.mark.parametrize(
    "make_robot,kwargs",
    [
        (make_friction_calendar_scratch_robot, {}),
        (make_friction_calendar_current_belief_robot, {"planning_budget": Budget()}),
        (make_friction_calendar_dreaming_robot, {}),
        (make_friction_calendar_belief_coverage_robot, {}),
        (
            make_friction_calendar_allocation_robot,
            {"num_units": 2, "rule": "transfer_practice"},
        ),
        (
            make_friction_calendar_allocation_robot,
            {"num_units": 2, "rule": "deliberate_practice"},
        ),
        (
            make_friction_calendar_allocation_robot,
            {"num_units": 2, "rule": "one_step_greedy"},
        ),
    ],
)
def test_robots_sleep(domain, make_robot, kwargs, tmp_path):
    """Every robot builds and sleeps one unit; learners store a plan."""
    budgets = {}
    if make_robot not in (
        make_friction_calendar_scratch_robot,
        make_friction_calendar_current_belief_robot,
    ):
        budgets = {
            "gathering_budget": Budget(max_iterations=0),
            "execution_budget": Budget(max_iterations=0),
            "planning_budget": Budget(max_iterations=1),
        }
    robot = make_robot(
        domain, night_sims=200, plan_dir=str(tmp_path), **budgets, **kwargs
    )
    robot.sleep(Budget(max_iterations=1), np.random.default_rng(0))
    if make_robot is not make_friction_calendar_scratch_robot:
        assert len(robot.learner.dataset) <= 1
