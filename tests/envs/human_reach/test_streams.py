"""Tests for envs/human_reach/streams.py and problem.py."""

import numpy as np
import pytest

from able.envs.human_reach import streams
from able.envs.human_reach.scene import ObjectCentricBedsideStocking3DEnv, reach_margin
from able.planners.pddlstream import build_stream_context, plan_bedside

THETA = np.array([0.7, 0.6, 0.65])


class _NearestProposal:
    """Ranks slots by their distance to a point."""

    def __init__(self, target):
        self.target = target

    def order(self, o, points, rng):
        """Slots nearest the target first."""
        del o, rng
        return np.argsort(np.linalg.norm(points - self.target, axis=1))


@pytest.fixture(name="sim", scope="module")
def _sim():
    sim = ObjectCentricBedsideStocking3DEnv(num_requested=2, num_clutter=0)
    yield sim
    sim.close()


def test_candidate_slots_and_usable_poses(sim):
    """Candidate slots lie within the largest reach, and a pose is usable when
    enough particles reach it."""
    obs, _ = sim.reset(seed=0)
    candidates = streams.requested_candidate_slots(sim)
    ub = np.asarray(sim.config.reach_theta_ub)
    assert candidates and all(r in sim.config.bedside_regions for r, _ in candidates)
    for _, slot in candidates:
        assert reach_margin(slot, ub, sim.shoulder) >= 0
    ctx = build_stream_context(
        sim, obs, theta_particles=np.array([THETA] * 9 + [[0.1] * 3])
    )
    region, slot = candidates[0]
    pose = sim.resting_pose("req0", tuple(slot[:2]), region)
    probability = streams.usable_probability(ctx, "req0", pose)
    reached = reach_margin(sim.grasp_point("req0", pose), THETA, sim.shoulder) >= 0
    assert probability == (0.9 if reached else 0.0)


def test_place_order_follows_the_proposal(sim):
    """Without a proposal the slots come in random order; a proposal ranks them."""
    obs, _ = sim.reset(seed=0)
    plain = build_stream_context(sim, obs)
    points = plain.slot_points("req0")
    assert sorted(plain.place_order("req0")) == list(range(len(points)))
    j = len(points) // 2
    ctx = build_stream_context(sim, obs, proposal=_NearestProposal(points[j]))
    order = ctx.place_order("req0")
    assert order[0] == j and sorted(order) == list(range(len(points)))


def test_plan_with_the_true_reach(sim, tmp_path, monkeypatch):
    """With the true reach, the plan puts both items within it on the bedside, and
    PDDLStream runs in its own directory."""
    monkeypatch.chdir(tmp_path)
    obs, _ = sim.reset(seed=0)
    plan = plan_bedside(sim, obs, max_time=60, theta_particles=THETA[None], seed=0)
    assert plan is not None
    places = {args[0]: args[1] for name, args in plan if name == "place"}
    assert set(places) == {"req0", "req1"}
    for item, placement in places.items():
        assert placement.surface in sim.config.bedside_regions
        point = sim.grasp_point(item, placement.pose)
        assert reach_margin(point, THETA, sim.shoulder) >= 0
    assert not list(tmp_path.iterdir())


class _PointsProposal:
    """Proposes the same points for every item."""

    def __init__(self, points):
        self.points_ = np.asarray(points, dtype=float)

    def points(self, o):
        """The proposed points."""
        del o
        return self.points_


def _in_box(ctx, placement) -> bool:
    low, high = ctx.surface_box(placement.surface)
    xy = np.array(placement.pose.position[:2])
    return bool(np.all(xy >= low - 1e-9) and np.all(xy <= high + 1e-9))


def test_continuous_placements_cover_the_surfaces(sim):
    """Without a proposal, continuous placements spread over the bedside's area for
    requested items, rather than sitting at slot centers, and rest on it."""
    obs, _ = sim.reset(seed=0)
    ctx = build_stream_context(sim, obs, placement="continuous")
    samples = streams.sample_continuous_place_pose(ctx, "req0")
    placements = [next(samples)[0] for _ in range(300)]
    assert {p.surface for p in placements} == set(sim.config.bedside_regions)
    assert all(_in_box(ctx, p) for p in placements)
    for p in placements[:5]:
        assert p.pose == sim.resting_pose("req0", tuple(p.pose.position[:2]), p.surface)
    slots = np.concatenate(
        [sim.config.region_slots(s)[:, :2] for s in sim.config.bedside_regions]
    )
    xy = np.array([p.pose.position[:2] for p in placements])
    gap = np.abs(xy[:, None] - slots[None]).max(-1).min(1)
    assert (gap > streams.POSE_JITTER).mean() > 0.5


def test_continuous_placements_try_the_proposal_first(sim):
    """The proposal's points come first, each moved onto its nearest surface's box,
    and uniform samples follow."""
    obs, _ = sim.reset(seed=0)
    (x0, x1), (y0, y1), _ = sim.config.bedside_regions["nightstand"]
    inside = sim.resting_pose(
        "req0", ((x0 + x1) / 2 + 0.013, (y0 + y1) / 2), "nightstand"
    )
    outside = np.array([x1 + 0.3, (y0 + y1) / 2, inside.position[2]])
    proposal = _PointsProposal([inside.position, outside])
    ctx = build_stream_context(sim, obs, proposal=proposal, placement="continuous")
    samples = streams.sample_continuous_place_pose(ctx, "req0")
    first, second = next(samples)[0], next(samples)[0]
    assert first.surface == "nightstand" and first.pose == inside
    assert second.surface == "nightstand" and _in_box(ctx, second)
    assert second.pose.position[0] == pytest.approx(ctx.surface_box("nightstand")[1][0])
    assert second.pose.position[1] == pytest.approx(outside[1])
    later = [next(samples)[0] for _ in range(20)]
    assert len({p.surface for p in later}) > 1


def test_continuous_placements_for_clutter():
    """Clutter may go on every surface."""
    sim = ObjectCentricBedsideStocking3DEnv(num_requested=1, num_clutter=1)
    obs, _ = sim.reset(seed=0)
    ctx = build_stream_context(sim, obs, placement="continuous")
    assert ctx.placement_surfaces("clutter0") == list(sim.config.surface_regions)
    samples = streams.sample_continuous_place_pose(ctx, "clutter0")
    surfaces = {next(samples)[0].surface for _ in range(300)}
    assert surfaces == set(sim.config.surface_regions)
    sim.close()


def test_plan_with_continuous_placements(sim, tmp_path, monkeypatch):
    """With continuous placements and the true reach, the plan puts both items
    within it on the bedside."""
    monkeypatch.chdir(tmp_path)
    obs, _ = sim.reset(seed=0)
    plan = plan_bedside(
        sim,
        obs,
        max_time=60,
        theta_particles=THETA[None],
        seed=0,
        placement="continuous",
    )
    assert plan is not None
    places = {args[0]: args[1] for name, args in plan if name == "place"}
    assert set(places) == {"req0", "req1"}
    for item, placement in places.items():
        assert placement.surface in sim.config.bedside_regions
        point = sim.grasp_point(item, placement.pose)
        assert reach_margin(point, THETA, sim.shoulder) >= 0
