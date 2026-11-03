"""Complexity levels of the friction environment on the threeroutes maze.

Each level sets the holes and the goal radius: L1 has no holes and a 0.40 m goal, L2
(the default) has the map's 2 holes and 0.30 m, L3 has 4 holes and 0.25 m, and L4 has 4
holes and 0.20 m.
"""

import dataclasses

from able.envs.friction.world import MAPS, THREE_ROUTES, PuckPhysics, PuckWorld

# Two more holes: mid blue corridor, and the green corridor's exit toward the goal.
EXTRA_HOLES = ((2.6, 3.3), (4.2, 1.2))

LEVELS: dict[str, tuple[tuple[tuple[float, float], ...], float]] = {
    "L1": ((), 0.40),
    "L2": (THREE_ROUTES.holes, 0.30),
    "L3": (THREE_ROUTES.holes + EXTRA_HOLES, 0.25),
    "L4": (THREE_ROUTES.holes + EXTRA_HOLES, 0.20),
}


def level_world(level: str) -> PuckWorld:
    """The threeroutes maze with the level's holes and goal radius."""
    holes, goal_radius = LEVELS[level]
    puck_map = dataclasses.replace(
        THREE_ROUTES, name=f"threeroutes_{level}", holes=holes
    )
    return PuckWorld(puck_map, PuckPhysics(goal_radius=goal_radius))


def register_level(level: str) -> tuple[str, PuckPhysics]:
    """Register the level's map in MAPS, so simulators can find it by name, and return
    the map's name and physics."""
    world = level_world(level)
    MAPS[world.puck_map.name] = world.puck_map
    return world.puck_map.name, world.physics
