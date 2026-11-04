"""The friction environment's maze: threeroutes with a number of holes and a goal
radius, which set how hard a task is.

Holes are added in a fixed order: the map's two, the middle of the blue corridor,
the green corridor's exit toward the goal, just past the goal, and then the far ends
of the blue and green corridors.
"""

import dataclasses
from typing import Any

from able.envs.friction.world import MAPS, THREE_ROUTES, PuckPhysics, PuckWorld

HOLES = THREE_ROUTES.holes + (
    (2.6, 3.3),
    (4.2, 1.2),
    (4.7, 2.2),
    (1.4, 3.6),
    (1.4, 0.8),
    (3.6, 3.6),
    (3.6, 0.8),
)


def make_world(num_holes: int = 2, goal_radius: float = 0.3) -> PuckWorld:
    """The threeroutes maze with the first num_holes holes and the goal radius."""
    if not 0 <= num_holes <= len(HOLES):
        raise ValueError(f"num_holes must be in [0, {len(HOLES)}], got {num_holes}")
    puck_map = dataclasses.replace(
        THREE_ROUTES, name=f"threeroutes_h{num_holes}", holes=HOLES[:num_holes]
    )
    return PuckWorld(puck_map, PuckPhysics(goal_radius=goal_radius))


def make_sim(sim: str, world: PuckWorld) -> Any:
    """The world's simulator: "2d" (PyMunk) or "3d" (PyBullet). The world's map is
    registered in MAPS, so the simulator can find it by name."""
    MAPS[world.puck_map.name] = world.puck_map
    map_name, physics = world.puck_map.name, world.physics
    if sim == "2d":
        # pylint: disable-next=import-outside-toplevel
        from able.envs.friction.sim2d import (
            DynPuckPushing2DEnvConfig,
            ObjectCentricDynPuckPushing2DEnv,
        )

        return ObjectCentricDynPuckPushing2DEnv(
            DynPuckPushing2DEnvConfig(map_name=map_name, physics=physics)
        )
    if sim == "3d":
        # pylint: disable-next=import-outside-toplevel
        from able.envs.friction.sim3d import (
            ObjectCentricPuckPushing3DEnv,
            PuckPushing3DEnvConfig,
        )

        return ObjectCentricPuckPushing3DEnv(
            PuckPushing3DEnvConfig(map_name=map_name, physics=physics)
        )
    raise ValueError(f"Unknown simulator {sim!r}; expected '2d' or '3d'")
