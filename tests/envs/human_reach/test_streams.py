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
