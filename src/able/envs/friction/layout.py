"""The friction environment's maze: threeroutes with a number of holes and a goal
radius, which set how hard a task is, and a number of floors.

Holes are added in a fixed order: the map's two, the middle of the blue corridor,
the green corridor's exit toward the goal, just past the goal, and then the far ends
of the blue and green corridors.

With more than 3 floors, each color is cut into stretches along x, each with its
own friction, so a task crosses only some of the floors.

The corridors mazes, for several sites, have three (corridors: blue, tan, green)
or four (corridors4: and red) walled corridors joined at their left ends, each with
the same holes.
"""

import dataclasses
from typing import Any

from able.envs.friction.world import (
    CORRIDORS,
    CORRIDORS4,
    DEFAULT_FRICTION_PRIOR,
    MAPS,
    THREE_ROUTES,
    PuckMap,
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

# The corridors maze's holes, as offsets from each corridor's bottom-left corner,
# so every corridor has the same holes, and its floors: each color whole, or cut
# into thirds.
CORRIDOR_HOLES = ((2.2, 0.5), (4.4, 0.8))
CORRIDOR_BOTTOMS = (3.6, 2.0, 0.4)
CORRIDOR_LEFT = 0.4
CORRIDOR_FLOOR_SPLITS: dict[int, tuple[float, ...]] = {3: (), 9: (2.0, 3.6)}
# Each corridors maze and its corridors' bottoms (y, m), from the top.
CORRIDOR_MAZES: dict[str, tuple[PuckMap, tuple[float, ...]]] = {
    "corridors": (CORRIDORS, CORRIDOR_BOTTOMS),
    "corridors4": (CORRIDORS4, (5.2, 3.6, 2.0, 0.4)),
}


def make_world(
    num_holes: int = 2,
    goal_radius: float = 0.3,
    num_floors: int = 3,
    maze: str = "threeroutes",
    strike_noise: bool = True,
    measurement_noise: bool = True,
) -> PuckWorld:
    """The maze ("threeroutes", "corridors", or "corridors4") with num_holes
    holes, the goal radius, and num_floors floors: a key of FLOOR_SPLITS on
    threeroutes, and on a corridors maze one or three per corridor. On
    threeroutes the holes are the first num_holes of HOLES; on a corridors maze,
    every corridor gets the first num_holes of CORRIDOR_HOLES. Without
    strike_noise, every strike goes at exactly its heading and speed, and without
    measurement_noise, the robot measures where the puck stops to a tenth of a
    millimeter."""
    base: PuckMap
    if maze == "threeroutes":
        base, all_holes, splits = THREE_ROUTES, HOLES, FLOOR_SPLITS
        holes = all_holes[:num_holes]
    elif maze in CORRIDOR_MAZES:
        base, bottoms = CORRIDOR_MAZES[maze]
        all_holes = CORRIDOR_HOLES
        splits = {len(bottoms): (), 3 * len(bottoms): CORRIDOR_FLOOR_SPLITS[9]}
        holes = tuple(
            (CORRIDOR_LEFT + dx, bottom + dy)
            for bottom in bottoms
            for dx, dy in CORRIDOR_HOLES[:num_holes]
        )
    else:
        names = ["threeroutes", *CORRIDOR_MAZES]
        raise ValueError(f"Unknown maze {maze!r}; expected one of {names}")
    if not 0 <= num_holes <= len(all_holes):
        raise ValueError(f"num_holes must be in [0, {len(all_holes)}], got {num_holes}")
    if num_floors not in splits:
        raise ValueError(f"num_floors must be in {list(splits)}, got {num_floors}")
    puck_map = dataclasses.replace(
        base,
        name=f"{maze}_h{num_holes}_f{num_floors}",
        holes=holes,
        floor_splits=splits[num_floors],
    )
    assert puck_map.num_surfaces == num_floors
    physics = PuckPhysics(goal_radius=goal_radius)
    if not strike_noise:
        physics = dataclasses.replace(physics, speed_noise=0.0, angle_noise=0.0)
    if not measurement_noise:
        physics = dataclasses.replace(physics, position_obs_noise=1e-4)
    return PuckWorld(puck_map, physics)


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
