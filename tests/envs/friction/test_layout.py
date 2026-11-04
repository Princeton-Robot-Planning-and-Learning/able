"""Tests for envs/friction/layout.py."""

import numpy as np
import pytest

from able.envs.friction.layout import HOLES, make_sim, make_world
from able.envs.friction.world import MAPS, THREE_ROUTES


def test_make_world_adds_holes_in_order():
    """Worlds take the first holes of the fixed order on the same maze, and the
    default world has the map's two holes."""
    worlds = [make_world(n) for n in range(len(HOLES) + 1)]
    assert [len(w.holes) for w in worlds] == list(range(len(HOLES) + 1))
    assert all(w.puck_map.rows == THREE_ROUTES.rows for w in worlds)
    assert np.allclose(make_world().holes, THREE_ROUTES.holes)
    assert make_world(goal_radius=0.2).physics.goal_radius == 0.2
    with pytest.raises(ValueError):
        make_world(len(HOLES) + 1)


def test_holes_are_on_the_floor():
    """Every hole lies on a floor, away from the start and the goal."""
    world = make_world(len(HOLES))
    for hole in world.holes:
        assert world.surface_at(hole) >= 0
        for point in (THREE_ROUTES.puck_start, THREE_ROUTES.goal):
            assert np.linalg.norm(hole - np.asarray(point)) > 0.5


def test_make_sim_registers_the_map():
    """The simulator finds the world's map by name."""
    world = make_world(4, 0.25)
    env = make_sim("2d", world)
    assert MAPS[world.puck_map.name] == world.puck_map
    assert env.world.physics.goal_radius == 0.25
    with pytest.raises(ValueError):
        make_sim("1d", world)
