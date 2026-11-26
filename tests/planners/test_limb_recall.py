"""Tests for limb_recall.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

import dataclasses
from types import SimpleNamespace

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
from able.models.kernel_regression import KernelRegressor
from able.planners.limb_recall import MAX_PLAN_LENGTH, LimbRecallPlanner, RecallParams
from able.planners.mpc import LimbMPCPlanner
from able.structs import Budget, PlanningFailure, PlanningTimeout


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
        RecallParams(
            model=KernelRegressor((0.1, 0.03, 0.03, 0.03), prior_weight=1e-12)
        ),
        seed=0,
    )
    request.addfinalizer(planner.close)
    return planner


def test_nothing_stored_plans_from_scratch(domain, request):
    """With no stored plan, MPC plans from scratch and runs out of a small budget."""
    planner = _make_planner(domain, request)
    task = domain.task_distribution.get_tasks()[0]
    with pytest.raises(PlanningTimeout):
        planner.plan(
            task,
            domain.initial_belief,
            planner.get_default_params(),
            Budget(max_iterations=1),
            np.random.default_rng(0),
        )


def _fit_plan(planner, task, belief, controls):
    get_features = planner._get_features  # pylint: disable=protected-access
    plan = np.zeros((MAX_PLAN_LENGTH, 7))
    plan[: len(controls)] = controls
    model = planner.get_model(planner.get_default_params()).fit(
        Dataset(
            x=get_features(task, belief)[None],
            y=plan.reshape(1, -1),
            costs=np.zeros(1),
            groups=np.zeros(1, dtype=np.int64),
        )
    )
    return planner.set_model(planner.get_default_params(), model)


def test_mpc_proposes_the_predicted_plan(domain, request):
    """Each MPC step proposes the predicted plan's controls from that step on, padded
    with zero controls, and nothing once the plan has ended."""
    planner = _make_planner(domain, request)
    task, belief = domain.task_distribution.get_tasks()[0], domain.initial_belief
    controls = np.arange(20 * 7, dtype=float).reshape(20, 7) + 1
    params = _fit_plan(planner, task, belief, controls)
    plan = planner._predict(  # pylint: disable=protected-access
        task, belief, params, np.random.default_rng(0)
    )
    assert np.allclose(plan, controls)
    attempt = SimpleNamespace(params=dataclasses.replace(params, plan=plan))
    previous = np.zeros(7)
    propose = planner._propose  # pylint: disable=protected-access
    (first,) = propose(attempt, None, 0, 1, previous)
    assert np.allclose(first, controls[:12])
    (second,) = propose(attempt, None, 12, 1, previous)
    assert np.allclose(second[:8], controls[12:]) and (second[8:] == 0).all()
    assert not propose(attempt, None, 20, 1, previous)


def test_a_plan_that_does_not_reach_the_goal_needs_more_search(domain, request):
    """A predicted plan that barely moves the limb leaves MPC searching past a small
    budget."""
    planner = _make_planner(domain, request)
    task, belief = domain.task_distribution.get_tasks()[0], domain.initial_belief
    params = _fit_plan(planner, task, belief, np.full((5, 7), 1e-3))
    with pytest.raises(PlanningTimeout):
        planner.plan(
            task, belief, params, Budget(max_iterations=2), np.random.default_rng(0)
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
    assert (examples.y[0, 4 * 7 :] == 0).all()


def test_predicted_plans_average_nearby_days_of_the_exercise(domain):
    """A day between two stored days gets the mean of their plans, whatever the plans
    of another exercise."""
    robot = make_limb_calendar_scratch_robot(domain)
    middle = domain.task_distribution.get_midpoint(domain.initial_belief)
    below, above = middle - [0.03, 0, 0], middle + [0.03, 0, 0]
    model = robot.planner_params.model.fit(
        Dataset(
            x=np.array([[0, *below], [0, *above], [1, *middle]]),
            y=np.array([[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]]),
            costs=np.zeros(3),
            groups=np.arange(3),
        )
    )
    predicted = model.sample(np.array([0, *middle]), 1, np.random.default_rng(0))
    assert np.allclose(predicted, [[2.0, 1.0]])


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
