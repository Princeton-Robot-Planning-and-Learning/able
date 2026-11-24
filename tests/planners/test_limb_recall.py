"""Tests for limb_recall.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

import dataclasses

import numpy as np
import pytest

from able.approaches import (
    make_limb_calendar_allocation_robot,
    make_limb_calendar_belief_coverage_robot,
    make_limb_calendar_current_belief_robot,
    make_limb_calendar_dreaming_robot,
    make_limb_calendar_scratch_robot,
)
from able.envs.limb.calendar import make_limb_calendar_domain
from able.models.base import Dataset
from able.planners.limb_recall import MAX_PLAN_LENGTH, LimbRecallPlanner, RecallParams
from able.planners.mpc import LimbMPCPlanner
from able.structs import Budget, PlanningFailure


@pytest.fixture(name="domain", scope="module")
def fixture_domain():
    """A domain whose simulators close after the module."""
    domain = make_limb_calendar_domain(1)
    yield domain
    domain.env_model.close()


def _make_planner(domain, request):
    tasks = domain.task_distribution

    def get_features(task, belief):
        return np.concatenate([[task.exercise], tasks.get_midpoint(belief)])

    planner = LimbRecallPlanner(
        domain.env_model,
        get_features,
        tasks.round_exercise,
        RecallParams(),
        seed=0,
    )
    request.addfinalizer(planner.close)
    return planner


def test_nothing_stored_fails(domain, request):
    """With no stored plan, planning while awake fails."""
    planner = _make_planner(domain, request)
    task = domain.task_distribution.get_tasks()[0]
    with pytest.raises(PlanningFailure):
        planner.plan(
            task,
            domain.initial_belief,
            planner.get_default_params(),
            Budget(max_iterations=1),
            np.random.default_rng(0),
        )


def test_a_plan_that_does_not_reach_the_goal_fails(domain, request):
    """A stored plan that leaves the limb where it is is not replayed as a plan."""
    planner = _make_planner(domain, request)
    tasks = domain.task_distribution
    task, belief = tasks.get_tasks()[0], domain.initial_belief
    features = np.concatenate([[task.exercise], tasks.get_midpoint(belief)])
    stay = np.full((MAX_PLAN_LENGTH, 7), np.nan)
    stay[:5] = 0.0
    model = planner.get_model(planner.get_default_params()).fit(
        Dataset(
            x=features[None],
            y=stay.reshape(1, -1),
            costs=np.zeros(1),
            groups=np.zeros(1, dtype=np.int64),
        )
    )
    params = planner.set_model(planner.get_default_params(), model)
    with pytest.raises(PlanningFailure):
        planner.plan(
            task, belief, params, Budget(max_iterations=1), np.random.default_rng(0)
        )


def test_training_plans_ignore_rounding_error(domain, request, monkeypatch):
    """A belief that differs in its last digits is searched as the same problem."""
    planner = _make_planner(domain, request)
    task, belief = domain.task_distribution.get_tasks()[0], domain.initial_belief
    nudged = dataclasses.replace(belief, mean=belief.mean + 2e-16)
    assert not np.array_equal(nudged.mean, belief.mean)
    searched = []

    def search(self, search_task, search_belief, *args):
        del self, args
        searched.append((search_task, search_belief))
        raise PlanningFailure("Only the problem matters here")

    monkeypatch.setattr(LimbMPCPlanner, "plan", search)
    params = planner.get_training_params(planner.get_default_params())
    for other in (belief, nudged):
        with pytest.raises(PlanningFailure):
            planner.plan(
                task, other, params, Budget(max_iterations=1), np.random.default_rng(0)
            )
    first, second = searched[0], searched[-1]
    assert np.array_equal(first[0].goal, second[0].goal)
    for name in ("mean", "stddev", "low", "high", "floor"):
        assert np.array_equal(getattr(first[1], name), getattr(second[1], name))


def test_training_plans_are_stored_whole(domain, request):
    """A training plan's row holds every control, padded to the longest plan."""
    planner = _make_planner(domain, request)
    tasks = domain.task_distribution
    task, belief = tasks.get_tasks()[2], domain.initial_belief
    controls = [np.full(7, 0.25)] * 4
    plan = planner._as_policy(task, controls)  # pylint: disable=protected-access
    examples = planner.get_plan_examples(task, belief, plan)
    assert examples.y.shape == (1, MAX_PLAN_LENGTH * 7)
    assert np.allclose(examples.y[0, : 4 * 7], 0.25)
    assert np.isnan(examples.y[0, 4 * 7 :]).all()


def test_limb_calendar_robots(domain):
    """Every limb-calendar approach builds."""
    unit = Budget(max_iterations=1)
    zero = Budget(max_iterations=0)
    make_limb_calendar_scratch_robot(domain)
    make_limb_calendar_current_belief_robot(domain, unit, days_ahead=2)
    make_limb_calendar_dreaming_robot(domain, zero, zero, unit, horizon=3)
    make_limb_calendar_belief_coverage_robot(domain, zero, zero, unit, horizon=3)
    for rule in ("transfer_practice", "deliberate_practice", "one_step_greedy"):
        make_limb_calendar_allocation_robot(
            domain, zero, zero, unit, num_units=2, rule=rule, horizon=3
        )
