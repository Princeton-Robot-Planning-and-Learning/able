"""Tests for envs/human_reach/execute.py."""

import numpy as np
import pytest
from pybullet_helpers.geometry import SE2Pose

from able.envs.human_reach.execute import _base_path, execute_plan
from able.envs.human_reach.scene import ObjectCentricBedsideStocking3DEnv
from able.planners.pddlstream import plan_bedside

THETA = np.array([0.7, 0.6, 0.65])


@pytest.fixture(name="sim", scope="module")
def _sim():
    sim = ObjectCentricBedsideStocking3DEnv(2, 0, allow_state_access=True)
    yield sim
    sim.close()


def test_base_path_turns_without_collision(sim):
    """A base path ends at the goal and leaves the robot where it was."""
    sim.reset(seed=0)
    sim.robot.set_base(SE2Pose(1.75, 0.9, 0.0))
    path = _base_path(sim, SE2Pose(1.28, 0.21, -np.pi))
    assert np.allclose([path[-1].x, path[-1].y, abs(path[-1].rot)], [1.28, 0.21, np.pi])
    assert sim.robot.get_base().x == pytest.approx(1.75)


def test_execute_a_plan(sim, tmp_path, monkeypatch):
    """A plan found with the true reach, executed with refined paths, reaches the
    goal."""
    monkeypatch.chdir(tmp_path)
    obs, _ = sim.reset(seed=0)
    sim.reach_theta = THETA
    plan = plan_bedside(sim, obs, max_time=60, theta_particles=THETA[None], seed=0)
    assert plan is not None
    sim.reset(options={"init_state": obs})
    sim.reach_theta = THETA
    assert not sim.goal_reached()
    assert execute_plan(sim, plan, seed=0)
    assert sim.goal_reached()
