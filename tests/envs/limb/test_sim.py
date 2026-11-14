"""Tests for envs/limb/sim.py."""

import numpy as np
import pytest

from able.envs.limb.human import get_variant
from able.envs.limb.sim import LimbSim


@pytest.fixture(name="sim", scope="module")
def _sim():
    # The environment's own grasp holds the isolated arm still.
    sim = LimbSim(get_variant("isolated-left-arm"))
    yield sim
    sim.close()


def _hold_still(sim: LimbSim, num_steps: int) -> float:
    """The largest limb joint drift after holding still from the current state."""
    start = np.array(sim.capture().limb_positions)
    for _ in range(num_steps):
        sim.advance_corrected(np.zeros(7))
    return float(np.max(np.abs(np.array(sim.capture().limb_positions) - start)))


def test_capture_and_restore(sim):
    """Tests that restoring a captured state puts the world back, weld included."""
    assert sim.is_grasping
    start = sim.capture()
    sim.advance_corrected(np.full(7, 5.0))
    moved = sim.capture()
    assert not np.allclose(moved.robot_positions, start.robot_positions)
    sim.restore(start)
    restored = sim.capture()
    assert np.allclose(restored.robot_positions, start.robot_positions)
    assert np.allclose(restored.limb_positions, start.limb_positions)
    assert np.allclose(restored.limb_velocities, start.limb_velocities)
    assert sim.is_grasping

    with sim.saved_state():
        sim.release_grasp()
        sim.advance_corrected(np.full(7, 5.0))
    assert sim.is_grasping
    assert np.allclose(sim.capture().limb_positions, start.limb_positions)
    assert sim.get_limb_error(start.limb_positions, start.limb_positions) == 0.0


def test_hold_still(sim):
    """Tests that a zero correction holds the grasped limb still under gravity."""
    sim.restore(sim.capture())
    assert _hold_still(sim, 200) < 0.02


@pytest.mark.parametrize("name", ["bed-left-arm", "wheelchair-left-arm"])
def test_hold_still_with_a_rolled_base(name):
    """Tests holding still where the limb's base is rolled, so inverse dynamics
    needs gravity in the base frame (without it, the arm drifts over 0.5 rad)."""
    sim = LimbSim(get_variant(name))
    try:
        assert _hold_still(sim, 200) < 0.02
    finally:
        sim.close()


def test_clone(sim):
    """Tests that a clone starts in the same state and steps the same way."""
    clone = sim.clone()
    try:
        assert clone.client != sim.client
        start = sim.capture()
        assert np.allclose(clone.capture().limb_positions, start.limb_positions)
        torque = np.array([2.0, -1.0, 0.5, 0.0, 0.3, 0.0, 0.0])
        for world in (sim, clone):
            world.restore(start)
            for _ in range(50):
                world.advance_corrected(torque)
        assert np.allclose(
            sim.capture().limb_positions, clone.capture().limb_positions, atol=1e-6
        )
    finally:
        clone.close()
        sim.restore(start)


def test_limb_directed_torque(sim):
    """Tests that the directed torque moves the limb toward a nearby goal."""
    start = sim.capture()
    goal = np.array(start.limb_positions) + np.array([0.1, 0.05, 0, 0.1, 0, 0])
    initial = sim.get_limb_error(start.limb_positions, goal)
    for _ in range(100):
        sim.advance_corrected(sim.get_limb_directed_torque(goal, gain=2.0))
    assert sim.get_limb_error(sim.capture().limb_positions, goal) < initial
    sim.restore(start)


def test_person_holds_up_their_limb(sim):
    """Tests that without the grasp, the person holds their limb still against
    gravity, where a limp limb would fall."""
    start = sim.capture()
    with sim.saved_state():
        sim.release_grasp()
        assert _hold_still(sim, 200) < 0.02
    sim.restore(start)


def test_walls_push_back_and_load_the_person(sim):
    """Tests that the person's walls act only past their limits, and that holding
    the limb past them counts as the robot loading the person's joints."""
    start = sim.capture()
    positions = np.array(start.limb_positions)
    wide = (positions - 1.0, positions + 1.0)
    torque = np.array([2.0, -1.0, 0.5, 0.0, 0.3, 0.0, 0.0])
    trajectories = []
    for limits in (None, wide):
        sim.restore(start)
        for _ in range(50):
            torques = sim.advance_corrected(torque, limits)
            assert not torques.resistance.any()
        trajectories.append(np.array(sim.capture().limb_positions))
    assert np.allclose(trajectories[0], trajectories[1])

    sim.restore(start)
    narrow = (positions - 1.0, positions + 1.0)
    narrow[1][3] = positions[3] - 0.05
    torques = sim.advance_corrected(np.zeros(7), narrow)
    assert torques.resistance[3] == pytest.approx(-20.0, abs=1.0)
    assert torques.robot[3] == pytest.approx(20.0, abs=5.0)
    for _ in range(100):
        sim.advance_corrected(np.zeros(7), narrow)
    assert sim.capture().limb_positions[3] < positions[3]
    sim.restore(start)
