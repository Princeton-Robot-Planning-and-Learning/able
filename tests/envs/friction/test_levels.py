"""Tests for envs/friction/levels.py."""

import numpy as np

from able.envs.friction.levels import LEVELS, level_world, register_level
from able.envs.friction.world import MAPS, THREE_ROUTES


def test_levels_add_holes_and_shrink_the_goal():
    """Higher levels have more holes and smaller goals on the same maze."""
    worlds = [level_world(level) for level in ("L1", "L2", "L3", "L4")]
    assert [len(w.holes) for w in worlds] == [0, 2, 4, 4]
    radii = [w.physics.goal_radius for w in worlds]
    assert radii == sorted(radii, reverse=True)
    assert all(w.puck_map.rows == THREE_ROUTES.rows for w in worlds)
    assert np.allclose(worlds[1].holes, THREE_ROUTES.holes)


def test_register_level():
    """A registered level's map can be found by name."""
    name, physics = register_level("L3")
    assert MAPS[name].holes == LEVELS["L3"][0]
    assert physics.goal_radius == LEVELS["L3"][1]
