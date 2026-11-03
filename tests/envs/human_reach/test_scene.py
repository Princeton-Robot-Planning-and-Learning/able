"""Tests for envs/human_reach/scene.py."""

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from able.envs.human_reach.scene import (
    BedsideStocking3DEnvConfig,
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
)


@pytest.fixture(name="sim", scope="module")
def _sim():
    sim = ObjectCentricBedsideStocking3DEnv(num_requested=2, num_clutter=2)
    yield sim
    sim.close()


def test_reach_margin():
    """The margin is 1 at the shoulder and 0 on the ellipsoid."""
    shoulder = np.array([0.1, 0.2, 0.9])
    theta = np.array([0.8, 0.6, 0.7])
    assert reach_margin(shoulder, theta, shoulder) == pytest.approx(1.0)
    on_axes = shoulder + np.diag(theta)
    assert np.allclose(reach_margin(on_axes, theta, shoulder), 0.0)
    assert reach_margin(shoulder + [1.0, 0, 0], theta, shoulder) < 0


def test_reset_puts_requested_items_far_and_clutter_at_the_bedside(sim):
    """Requested items start on the far furniture, so the goal is not reached, and
    clutter starts on bedside slots."""
    obs, _ = sim.reset(seed=0)
    assert obs.requested_objects == ["req0", "req1"]
    assert obs.clutter_objects == ["clutter0", "clutter1"]
    for name in obs.requested_objects:
        assert obs.get_object_pose(name).position[0] > 2.0
    bedside = np.concatenate(
        [sim.config.region_slots(r) for r in sim.config.bedside_regions]
    )
    for name in obs.clutter_objects:
        xy = np.array(obs.get_object_pose(name).position[:2])
        assert np.abs(bedside[:, :2] - xy).max(1).min() < 1e-6
    assert not sim.goal_reached()


def test_reachable_items_and_obstruction(sim):
    """An item on the nightstand is reachable by a long reach, until clutter sits
    between it and the shoulder."""
    sim.reset(seed=0)
    sim.reach_theta = np.array(BedsideStocking3DEnvConfig().reach_theta_ub)
    (x0, _), (y0, y1), _ = sim.config.bedside_regions["nightstand"]
    item = (float(x0 + 0.06), float((y0 + y1) / 2))
    sim.place("req0", sim.resting_pose("req0", item, "nightstand"))
    assert sim.item_reachable("req0")
    middle = (sim.shoulder + sim.grasp_point("req0")) / 2
    sim.place("clutter0", Pose(tuple(middle)))
    assert not sim.is_unobstructed("req0") and not sim.item_reachable("req0")
    assert sim.render().ndim == 3
