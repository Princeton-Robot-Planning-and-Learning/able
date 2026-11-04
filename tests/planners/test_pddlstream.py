"""Tests for planners/pddlstream.py."""

import os
from collections import Counter

import numpy as np
import pytest

from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import NO_ACTION, ReachModel
from able.envs.human_reach.tasks import make_domain
from able.planners.pddlstream import (
    PDDLStreamParams,
    ReachPDDLStreamPlanner,
    counted,
    private_cwd,
)
from able.structs import Budget, PlanningFailure

THETA = np.array([0.7, 0.6, 0.65])


def test_private_cwd(tmp_path, monkeypatch):
    """Code runs in a fresh directory that is removed afterwards."""
    monkeypatch.chdir(tmp_path)
    with private_cwd():
        inside = os.getcwd()
        assert inside != str(tmp_path)
        with open("scratch", "w", encoding="utf-8") as f:
            f.write("x")
    assert os.getcwd() == str(tmp_path) and not os.path.exists(inside)


def test_counted_streams():
    """Each test call and each generated output counts once."""
    calls: Counter = Counter()
    test = counted(lambda x: x > 0, calls, "positive", generator=False)
    assert test(1) and not test(-1)

    def two_outputs(x):
        yield (x,)
        yield (x + 1,)

    gen = counted(two_outputs, calls, "next", generator=True)
    assert list(gen(1)) == [(1,), (2,)]
    assert calls == {"positive": 2, "next": 3}


def _make_planner(model: ReachModel) -> ReachPDDLStreamPlanner:
    return ReachPDDLStreamPlanner(model, PDDLStreamParams(max_time=60))


def _close(model: ReachModel, planner: ReachPDDLStreamPlanner) -> None:
    planner.close()
    model.close()


def _point_belief(theta, num_particles: int = 300) -> ParticleBelief:
    low, high = np.array([0.05] * 3), np.array([0.95, 0.80, 0.90])
    return ParticleBelief(np.tile(theta, (num_particles, 1)), low, high)


def test_reach_planner(tmp_path, monkeypatch):
    """With a belief at the true reach, the plan delivers both items within it;
    with a tiny reach, planning fails within the budget."""
    monkeypatch.chdir(tmp_path)
    domain = make_domain(seed=0, num_requested=2, num_clutter=0, true_theta=THETA)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    planner = _make_planner(model)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    rng = np.random.default_rng(0)
    policy = planner.plan(
        task, _point_belief(THETA), planner.get_default_params(), Budget(), rng
    )
    assert len(policy.actions) == model.horizon
    plan = [a for a in policy.actions if a != NO_ACTION]
    stats = policy.get_planning_stats()
    assert stats["plan_length"] == len(plan) and stats["stream_calls"] > 0
    places = [args[0] for name, args in plan if name == "place"]
    assert {"req0", "req1"} <= set(places)
    with pytest.raises(PlanningFailure, match="no plan"):
        planner.plan(
            task,
            _point_belief(np.array([0.1, 0.1, 0.1])),
            planner.get_default_params(),
            Budget(max_wall_time=5.0),
            rng,
        )
    assert not list(tmp_path.iterdir())
    _close(model, planner)
