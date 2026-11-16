"""Tests for envs/friction/query.py."""

import dataclasses

import numpy as np
import pytest

from able.envs.friction.query import (
    FrictionQuery,
    ProbeSimulator,
    PushRecord,
    probe_queries,
)
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, Push


@pytest.mark.parametrize("num_floors", [3, 8])
def test_probe_poses_fit_their_floor(num_floors):
    """Each floor's probe, at either extreme friction, stops on that floor and clear
    of holes."""
    simulator = ProbeSimulator("2d", num_floors=num_floors)
    world = simulator.world
    prior = dataclasses.replace(DEFAULT_FRICTION_PRIOR, num_surfaces=num_floors)
    low, high = prior.bounds
    queries = probe_queries(world, low)
    assert len(queries) == num_floors
    rng = np.random.default_rng(0)
    for floor, query in enumerate(queries):
        assert query.floor == floor == world.surface_at(np.array(query.puck))
        assert world.push_feasible(np.array(query.puck), query.angle)
        for mu in (low[floor], high[floor]):
            frictions = np.full(num_floors, 0.08)
            frictions[floor] = mu
            record = simulator.ask(frictions, query, rng)
            assert record.feasible and not record.fell_in_hole
            assert record.release_speed > 0 and record.puck == query.puck
            stop = np.array(record.stop)
            assert world.surface_at(stop) == floor
            gap = np.linalg.norm(stop - query.puck)
            assert gap > world.physics.accel_distance
    simulator.close()


def test_records_and_queries():
    """A record holds the push's measurements, and a query is a push with an
    automatic stance."""
    info = {
        "push_feasible": True,
        "release_speed": 0.5,
        "stop_position": np.array([1.0, 2.0]),
        "fell_in_hole": False,
        "goal_reached": True,
        "drive_distance": 0.3,
    }
    record = PushRecord.from_info(info, np.array([0.5, 2.0]), Push(0.0, 0.5, 0.2))
    assert record.stop == (1.0, 2.0) and record.puck == (0.5, 2.0)
    assert record.goal_reached and record.push.stance == 0.2
    query = FrictionQuery(1, (3.45, 1.65), 0.5, 0.7)
    assert query.push == Push(0.5, 0.7)
