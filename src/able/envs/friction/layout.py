"""The friction environment's maze: threeroutes with a number of holes and a goal
radius, which set how hard a task is, and a number of floors.

Holes are added in a fixed order: the map's two, the middle of the blue corridor,
the green corridor's exit toward the goal, just past the goal, and then the far ends
of the blue and green corridors.

With more than 3 floors, each color is cut into stretches along x, each with its
own friction, so a task crosses only some of the floors.
"""

import dataclasses
from typing import Any

from able.envs.friction.world import (
    DEFAULT_FRICTION_PRIOR,
    MAPS,
    THREE_ROUTES,
    PuckPhysics,
    PuckWorld,
)

HOLES = THREE_ROUTES.holes + (
    (2.6, 3.3),
    (4.2, 1.2),
    (4.7, 2.2),
    (1.4, 3.6),
    (1.4, 0.8),
    (3.6, 3.6),
    (3.6, 0.8),
)

# Where the colors are cut (x, m), by the number of floors. 6: each color's left
# and right half. 8: the left, middle, and right of the blue and green corridors,
# and the tan floor on either side of the central block, whose slot is blue.
FLOOR_SPLITS: dict[int, tuple[float, ...]] = {3: (), 6: (2.8,), 8: (2.0, 3.2)}


def make_world(
    num_holes: int = 2, goal_radius: float = 0.3, num_floors: int = 3
) -> PuckWorld:
    """The threeroutes maze with the first num_holes holes, the goal radius, and
    num_floors floors (a key of FLOOR_SPLITS)."""
    if not 0 <= num_holes <= len(HOLES):
        raise ValueError(f"num_holes must be in [0, {len(HOLES)}], got {num_holes}")
    if num_floors not in FLOOR_SPLITS:
        raise ValueError(
            f"num_floors must be in {list(FLOOR_SPLITS)}, got {num_floors}"
        )
    puck_map = dataclasses.replace(
        THREE_ROUTES,
        name=f"threeroutes_h{num_holes}_f{num_floors}",
        holes=HOLES[:num_holes],
        floor_splits=FLOOR_SPLITS[num_floors],
    )
    assert puck_map.num_surfaces == num_floors
    return PuckWorld(puck_map, PuckPhysics(goal_radius=goal_radius))


def make_sim(sim: str, world: PuckWorld) -> Any:
    """The world's simulator: "2d" (PyMunk) or "3d" (PyBullet). The world's map is
    registered in MAPS, so the simulator can find it by name."""
    MAPS[world.puck_map.name] = world.puck_map
    map_name, physics = world.puck_map.name, world.physics
    # The simulator draws frictions from the prior only if a reset gives none.
    prior = dataclasses.replace(
        DEFAULT_FRICTION_PRIOR, num_surfaces=world.puck_map.num_surfaces
    )
    if sim == "2d":
        # pylint: disable-next=import-outside-toplevel
        from able.envs.friction.sim2d import (
            DynPuckPushing2DEnvConfig,
            ObjectCentricDynPuckPushing2DEnv,
        )

        return ObjectCentricDynPuckPushing2DEnv(
            DynPuckPushing2DEnvConfig(
                map_name=map_name, physics=physics, friction_prior=prior
            )
        )
    if sim == "3d":
        # pylint: disable-next=import-outside-toplevel
        from able.envs.friction.sim3d import (
            ObjectCentricPuckPushing3DEnv,
            PuckPushing3DEnvConfig,
        )

        return ObjectCentricPuckPushing3DEnv(
            PuckPushing3DEnvConfig(
                map_name=map_name, physics=physics, friction_prior=prior
            )
        )
    raise ValueError(f"Unknown simulator {sim!r}; expected '2d' or '3d'")
