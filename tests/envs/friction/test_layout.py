"""Tests for envs/friction/layout.py."""

import numpy as np
import pytest

from able.envs.friction.layout import (
    CORRIDOR_BOTTOMS,
    CORRIDOR_HOLES,
    CORRIDOR_LEFT,
    CORRIDOR_MAZES,
    FLOOR_SPLITS,
    HOLES,
    make_sim,
    make_world,
)
from able.envs.friction.world import CORRIDORS, MAPS, THREE_ROUTES


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


def test_make_world_cuts_the_floors():
    """Each number of floors cuts the same maze's colors where FLOOR_SPLITS says,
    and the simulator takes a friction per floor."""
    for num_floors, splits in FLOOR_SPLITS.items():
        world = make_world(num_floors=num_floors)
        assert world.puck_map.num_surfaces == num_floors
        assert world.puck_map.floor_splits == splits
        assert world.puck_map.rows == THREE_ROUTES.rows
    with pytest.raises(ValueError):
        make_world(num_floors=4)
    env = make_sim("2d", make_world(num_floors=8))
    env.reset(seed=0)
    assert len(env.frictions) == 8
    env.reset(seed=0, options={"frictions": (0.05,) * 8})
    assert env.frictions == (0.05,) * 8


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


def test_make_world_corridors():
    """Every corridor of the corridors maze has the same holes and its own color's
    floors, and the corridors join only at their left ends."""
    world = make_world(len(CORRIDOR_HOLES), 0.2, 9, maze="corridors")
    puck_map = world.puck_map
    assert puck_map.rows == CORRIDORS.rows and puck_map.num_surfaces == 9
    assert puck_map.surface_colors == (0, 0, 0, 1, 1, 1, 2, 2, 2)
    assert len(world.holes) == len(CORRIDOR_HOLES) * len(CORRIDOR_BOTTOMS)
    for color, bottom in enumerate(CORRIDOR_BOTTOMS):
        for dx, dy in CORRIDOR_HOLES:
            hole = np.array([CORRIDOR_LEFT + dx, bottom + dy])
            assert any(np.allclose(hole, h) for h in world.holes)
            assert puck_map.surface_colors[world.surface_at(hole)] == color
    assert world.surface_at(np.array([3.0, 3.4])) == -1
    assert world.surface_at(np.array([1.0, 3.4])) >= 0
    assert make_world(maze="corridors").puck_map.num_surfaces == 3
    with pytest.raises(ValueError):
        make_world(num_floors=8, maze="corridors")
    with pytest.raises(ValueError):
        make_world(maze="spiral")
    env = make_sim("2d", world)
    env.reset(seed=0)
    assert len(env.frictions) == 9


def test_make_world_four_corridors():
    """The four-corridor maze has the same holes in each corridor, a floor color
    per corridor, and three floors in each."""
    world = make_world(2, 0.2, 12, maze="corridors4")
    puck_map = world.puck_map
    assert puck_map.num_surfaces == 12
    assert puck_map.surface_colors == (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
    _, bottoms = CORRIDOR_MAZES["corridors4"]
    for color, bottom in enumerate(bottoms):
        for dx, dy in CORRIDOR_HOLES:
            hole = np.array([CORRIDOR_LEFT + dx, bottom + dy])
            assert puck_map.surface_colors[world.surface_at(hole)] == color
    assert make_world(maze="corridors4", num_floors=4).puck_map.num_surfaces == 4
    with pytest.raises(ValueError):
        make_world(maze="corridors4", num_floors=9)
    env = make_sim("2d", world)
    env.reset(seed=0)
    assert len(env.frictions) == 12
