"""Tests for planners/pddlstream.py."""

import os
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import NO_ACTION, ReachModel
from able.envs.human_reach.scene import BedsideStocking3DEnvConfig
from able.envs.human_reach.streams import Placement
from able.envs.human_reach.tasks import make_domain
from able.models.base import ConditionalModel, Dataset
from able.planners import pddlstream as pddlstream_planner
from able.planners.pddlstream import (
    MAX_REQUESTED,
    PDDLStreamParams,
    PlaceProposal,
    ReachPDDLStreamPlanner,
    build_stream_context,
    counted,
    get_bedside_slots,
    get_place_features,
    private_cwd,
)
from able.policies.base import OpenLoopPolicy
from able.structs import Budget, PlanningFailure, PlanningTimeout

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
    with a tiny reach, PDDLStream searches until the budget's time runs out, which
    is a timeout."""
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
    with pytest.raises(PlanningTimeout, match="no plan within its time limit"):
        planner.plan(
            task,
            _point_belief(np.array([0.1, 0.1, 0.1])),
            planner.get_default_params(),
            Budget(max_wall_time=5.0),
            rng,
        )
    # A budget already spent stops before searching, also a timeout.
    with pytest.raises(PlanningTimeout):
        planner.plan(
            task,
            _point_belief(THETA),
            planner.get_default_params(),
            Budget(max_wall_time=0.0),
            rng,
        )
    assert not list(tmp_path.iterdir())
    _close(model, planner)


class _FixedModel(ConditionalModel):
    """Always proposes the same placement."""

    def __init__(self, point) -> None:
        self.point = np.asarray(point, dtype=float)

    def fit(self, dataset: Dataset) -> ConditionalModel:
        return self

    def sample(self, x, num_samples, rng):
        return np.tile(self.point, (num_samples, 1))


class _FailingWithModel(ReachPDDLStreamPlanner):
    """Finds no plan whenever the model guides it."""

    def _solve(self, task, particles, params, budget, rng, proposal, stats, start_time):
        if proposal is not None:
            return None, None
        return super()._solve(
            task, particles, params, budget, rng, proposal, stats, start_time
        )


def _make_failing(model: ReachModel) -> ReachPDDLStreamPlanner:
    return _FailingWithModel(model, PDDLStreamParams(max_time=60))


def _placement(item: str, xyz) -> Placement:
    return Placement(item, Pose(tuple(xyz)), "nightstand")


def test_features_and_labels():
    """A row per item the plan places, requested items first: its one-hot (all
    zeros for clutter), the clutter's slots and the belief, labeled with its final
    placement."""
    domain = make_domain(seed=0, num_requested=2, num_clutter=2)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    belief = domain.initial_belief
    assert isinstance(belief, ParticleBelief)
    # make_domain's default scene config.
    scene, config = task.initial_state.scene, BedsideStocking3DEnvConfig()
    num_slots = len(get_bedside_slots(config))
    x = get_place_features(scene, config, belief, "req1")
    assert x.shape == (MAX_REQUESTED + num_slots + 6,)
    assert x[1] == 1 and x[:MAX_REQUESTED].sum() == 1
    assert x[MAX_REQUESTED : MAX_REQUESTED + num_slots].sum() == 2
    plan = (
        ("place", ("req0", _placement("req0", (0.7, 0.3, 0.6)))),
        ("place", ("clutter0", _placement("clutter0", (2.2, 0.0, 0.6)))),
        ("place", ("req1", _placement("req1", (0.7, 0.0, 0.7)))),
        ("place", ("req0", _placement("req0", (0.8, 0.3, 0.6)))),
    )
    planner = ReachPDDLStreamPlanner(model)
    rows = planner.get_plan_examples(task, belief, OpenLoopPolicy(list(plan)))
    assert rows.x.shape == (3, len(x)) and rows.y.shape == (3, 3)
    assert np.allclose(rows.y[0], (0.8, 0.3, 0.6))  # the final placement
    clutter = get_place_features(scene, config, belief, "clutter0")
    assert clutter[:MAX_REQUESTED].sum() == 0
    assert np.allclose(rows.x[2], clutter) and np.allclose(rows.y[2], (2.2, 0.0, 0.6))
    _close(model, planner)


def test_place_proposal_ranks_the_nearest_slot_first():
    """The slot nearest most draws comes first; items without draws are random."""
    domain = make_domain(seed=0, num_requested=2, num_clutter=0)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    planner = ReachPDDLStreamPlanner(model)
    scene = domain.task_distribution.sample(
        np.random.default_rng(0)
    ).initial_state.scene
    rng = np.random.default_rng(0)
    points = build_stream_context(planner.sim, scene).slot_points("req0")
    j = len(points) // 2
    samples = points[j] + rng.normal(0, 0.005, (16, 3))
    ctx = build_stream_context(
        planner.sim, scene, proposal=PlaceProposal({"req0": samples})
    )
    order = ctx.place_order("req0")
    assert order[0] == j and sorted(order) == list(range(len(points)))
    assert sorted(ctx.place_order("req1")) == list(range(len(points)))
    _close(model, planner)


def test_place_proposal_ranks_clutter_candidates():
    """Clutter tries the candidate nearest most of its draws first, and every
    candidate in a random order without draws or a proposal."""
    domain = make_domain(seed=0, num_requested=1, num_clutter=1)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    planner = ReachPDDLStreamPlanner(model)
    scene = domain.task_distribution.sample(
        np.random.default_rng(0)
    ).initial_state.scene
    ctx = build_stream_context(planner.sim, scene)
    candidates = [(s, xy) for s in ctx.all_surfaces for xy in ctx.slots[s][:, :2]]
    assert sorted(ctx.clutter_order("clutter0", candidates)) == list(
        range(len(candidates))
    )
    j = len(candidates) // 2
    surface, xy = candidates[j]
    point = np.array(planner.sim.resting_pose("clutter0", tuple(xy), surface).position)
    samples = point + np.random.default_rng(0).normal(0, 0.005, (16, 3))
    ctx = build_stream_context(
        planner.sim, scene, proposal=PlaceProposal({"clutter0": samples})
    )
    order = ctx.clutter_order("clutter0", candidates)
    assert order[0] == j and sorted(order) == list(range(len(candidates)))
    assert sorted(ctx.clutter_order("req0", candidates)) == list(range(len(candidates)))
    _close(model, planner)


def test_a_trained_model_guides_planning(tmp_path, monkeypatch):
    """A model's placements guide the plan, and planning that fails with the model
    is repeated without it."""
    monkeypatch.chdir(tmp_path)
    domain = make_domain(seed=0, num_requested=2, num_clutter=0, true_theta=THETA)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    planner = _make_planner(model)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    slot = build_stream_context(planner.sim, task.initial_state.scene).slot_points(
        "req0"
    )[0]
    params = planner.set_model(planner.get_default_params(), _FixedModel(slot))
    rng = np.random.default_rng(0)
    policy = planner.plan(task, _point_belief(THETA), params, Budget(), rng)
    stats = policy.get_planning_stats()
    assert stats["seeded"] == 1.0 and stats["fell_back"] == 0.0
    plain = planner.plan(
        task, _point_belief(THETA), planner.get_default_params(), Budget(), rng
    )
    assert plain.get_planning_stats()["seeded"] == 0.0
    failing = _make_failing(model)
    policy = failing.plan(task, _point_belief(THETA), params, Budget(), rng)
    assert policy.get_planning_stats()["fell_back"] == 1.0
    failing.close()
    _close(model, planner)


def test_memory_is_released_after_every_solve(tmp_path, monkeypatch):
    """Every solve returns its freed memory, whether it found a plan or not."""
    monkeypatch.chdir(tmp_path)
    pddlstream_planner.release_memory()  # runs where malloc_trim is missing too
    releases = []
    monkeypatch.setattr(
        pddlstream_planner, "release_memory", lambda: releases.append(1)
    )
    domain = make_domain(seed=0, num_requested=2, num_clutter=0, true_theta=THETA)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    planner = _make_planner(model)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    rng = np.random.default_rng(0)
    planner.plan(
        task, _point_belief(THETA), planner.get_default_params(), Budget(), rng
    )
    assert releases == [1]
    _close(model, planner)


def test_memory_limit(tmp_path, monkeypatch):
    """A solve stops once the process's memory has grown past its limit, checked
    every MEMORY_CHECK_EVERY stream evaluations, and planning then fails."""
    monkeypatch.chdir(tmp_path)
    # Each reading is 0.1 GB more than the last.
    readings = iter(range(0, 10**12, 10**8))
    monkeypatch.setattr(
        pddlstream_planner, "get_resident_bytes", lambda: next(readings)
    )
    calls: Counter = Counter()
    with pddlstream_planner.memory_limit(10**9):
        for _ in range(pddlstream_planner.MEMORY_CHECK_EVERY - 1):
            pddlstream_planner.count_evaluation(calls, "s")
        with pytest.raises(pddlstream_planner.OutOfMemory):
            for _ in range(20 * pddlstream_planner.MEMORY_CHECK_EVERY):
                pddlstream_planner.count_evaluation(calls, "s")
    # No limit: streams never check.
    pddlstream_planner.count_evaluation(calls, "s")
    domain = make_domain(seed=0, num_requested=2, num_clutter=0, true_theta=THETA)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    planner = _make_planner(model)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    params = replace(planner.get_default_params(), max_memory_gb=0.5)
    with pytest.raises(PlanningFailure, match="out of memory"):
        planner.plan(
            task,
            _point_belief(np.array([0.1, 0.1, 0.1])),
            params,
            Budget(max_wall_time=60.0),
            np.random.default_rng(0),
        )
    _close(model, planner)


def test_the_guided_try_leaves_the_fallback_time(tmp_path, monkeypatch):
    """A try the model guides gets guided_time_fraction of the time limit, and the
    fallback without the model gets what is left."""
    monkeypatch.chdir(tmp_path)
    limits: list[float] = []

    def solve_problem(problem, max_time, algorithm):
        del problem, algorithm
        limits.append(max_time)
        # The guided try finds nothing; the fallback finds an empty plan.
        return None if len(limits) % 2 == 1 else []

    monkeypatch.setattr(pddlstream_planner, "solve_problem", solve_problem)
    domain = make_domain(seed=0, num_requested=2, num_clutter=0, true_theta=THETA)
    model = domain.env_model
    assert isinstance(model, ReachModel)
    planner = _make_planner(model)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    slot = build_stream_context(planner.sim, task.initial_state.scene).slot_points(
        "req0"
    )[0]
    defaults = replace(planner.get_default_params(), max_time=600.0)
    params = planner.set_model(
        replace(defaults, guided_time_fraction=0.5), _FixedModel(slot)
    )
    belief, rng = _point_belief(THETA), np.random.default_rng(0)
    policy = planner.plan(task, belief, params, Budget(max_wall_time=600.0), rng)
    assert policy.get_planning_stats()["fell_back"] == 1.0
    assert limits[0] == pytest.approx(300.0)
    assert 590.0 < limits[1] <= 600.0  # what is left of the 600 s budget
    # Without a planning budget, the fallback gets all of max_time.
    limits.clear()
    planner.plan(task, belief, params, Budget(), rng)
    assert limits == [pytest.approx(300.0), 600.0]
    # A fraction of 1 gives the guided try the whole limit, as before.
    limits.clear()
    whole = replace(params, guided_time_fraction=1.0)
    planner.plan(task, belief, whole, Budget(max_wall_time=600.0), rng)
    assert limits[0] > 590.0
    _close(model, planner)
