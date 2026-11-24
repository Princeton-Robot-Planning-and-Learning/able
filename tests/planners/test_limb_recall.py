"""Tests for limb_recall.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

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
from able.structs import Budget, PlanningFailure


@pytest.fixture(name="domain", scope="module")
def fixture_domain():
    """A domain whose simulators close after the module."""
    domain = make_limb_calendar_domain(1)
    yield domain
    domain.env_model.close()


def _make_planner(domain, request, plan_dir=None):
    tasks = domain.task_distribution

    def get_features(task, belief):
        return np.concatenate([[task.exercise], tasks.get_midpoint(belief)])

    planner = LimbRecallPlanner(
        domain.env_model, get_features, RecallParams(), seed=0, plan_dir=plan_dir
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


def test_training_plans_are_kept_and_reused(domain, request, tmp_path):
    """A training plan on disk is returned without searching."""
    planner = _make_planner(domain, request, tmp_path)
    tasks = domain.task_distribution
    task, belief = tasks.get_tasks()[2], domain.initial_belief
    features = np.concatenate([[task.exercise], tasks.get_midpoint(belief)])
    key = "_".join(str(int(v)) for v in np.round(features * 1000))
    controls = np.full((4, 7), 0.25)
    np.save(tmp_path / f"{key}.npy", controls)
    params = planner.get_training_params(planner.get_default_params())
    plan = planner.plan(
        task, belief, params, Budget(max_iterations=1), np.random.default_rng(0)
    )
    assert np.allclose(np.array(plan.controls), controls)
    examples = planner.get_plan_examples(task, belief, plan)
    assert examples.y.shape == (1, MAX_PLAN_LENGTH * 7)
    assert np.isnan(examples.y[0, 4 * 7 :]).all()


def test_limb_calendar_robots(domain):
    """Every limb-calendar approach builds."""
    unit = Budget(max_iterations=1)
    zero = Budget(max_iterations=0)
    make_limb_calendar_scratch_robot(domain)
    make_limb_calendar_current_belief_robot(domain, unit, days_ahead=2)
    make_limb_calendar_dreaming_robot(domain, zero, zero, unit, horizon=3)
    make_limb_calendar_belief_coverage_robot(domain, zero, zero, unit, horizon=3)
    for rule in ("transfer_practice", "deliberate_practice"):
        make_limb_calendar_allocation_robot(
            domain, zero, zero, unit, num_units=2, rule=rule, horizon=3
        )
