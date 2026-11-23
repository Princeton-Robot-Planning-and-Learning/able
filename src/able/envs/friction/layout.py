"""The friction environment's maze: threeroutes with a number of holes and a goal
radius, which set how hard a task is, and a number of floors.

Holes are added in a fixed order: the map's two, the middle of the blue corridor,
the green corridor's exit toward the goal, just past the goal, and then the far ends
of the blue and green corridors.

With more than 3 floors, each color is cut into stretches along x, each with its
own friction, so a task crosses only some of the floors.

The corridors mazes, for several sites, have three (corridors: blue, tan, green)
or four (corridors4: and red) walled corridors joined at their left ends, each with
the same holes. corridor1 is one walled corridor, whose holes and single-cell
obstacles a layout sets (draw_layout, get_layout_world).
"""

import dataclasses
from collections import OrderedDict
from typing import Any

import numpy as np

from able.envs.friction.world import (
    CORRIDOR1,
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
# corridor1's default holes, and its floors: whole, or cut into four stretches.
CORRIDOR1_HOLES = ((2.6, 1.0), (4.0, 1.8))
CORRIDOR1_FLOOR_SPLITS: dict[int, tuple[float, ...]] = {1: (), 4: (1.8, 3.2, 4.6)}


def make_world(
    num_holes: int = 2,
    goal_radius: float = 0.3,
    num_floors: int = 3,
    maze: str = "threeroutes",
) -> PuckWorld:
    """The maze ("threeroutes", "corridors", "corridors4", or "corridor1") with num_holes
    holes, the goal radius, and num_floors floors: a key of FLOOR_SPLITS on
    threeroutes, and on a corridors maze one or three per corridor. On
    threeroutes the holes are the first num_holes of HOLES; on a corridors maze,
    every corridor gets the first num_holes of CORRIDOR_HOLES."""
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
    elif maze == "corridor1":
        base, all_holes, splits = CORRIDOR1, CORRIDOR1_HOLES, CORRIDOR1_FLOOR_SPLITS
        holes = all_holes[:num_holes]
    else:
        names = ["threeroutes", *CORRIDOR_MAZES, "corridor1"]
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


# Layout worlds built so far, by base map, physics, and layout key; the oldest go.
_LAYOUT_WORLDS: OrderedDict[tuple[str, PuckPhysics, str], PuckWorld] = OrderedDict()
_MAX_LAYOUT_WORLDS = 64


def draw_layout(
    world: PuckWorld,
    num_holes: int,
    num_obstacles: int,
    rng: np.random.Generator,
    min_gap: float = 0.9,
) -> str:
    """A layout of the world's map, as a key for get_layout_world(): num_obstacles
    floor cells, not in the first or last two columns, made walls, and num_holes
    holes clear of the walls, the obstacles, and each other. Hole and obstacle
    centers are at least min_gap apart."""
    puck_map, phys = world.puck_map, world.physics
    rows, size = puck_map.rows, puck_map.cell_size
    cells = [
        (r, c)
        for r, row in enumerate(rows)
        for c, code in enumerate(row)
        if code != "#" and 2 <= c < puck_map.num_cols - 2
    ]

    def center(cell: tuple[int, int]) -> np.ndarray:
        r, c = cell
        return np.array([(c + 0.5) * size, (puck_map.num_rows - r - 0.5) * size])

    for _ in range(1000):
        picks = rng.choice(len(cells), size=num_obstacles, replace=False)
        obstacles = sorted(cells[int(i)] for i in picks)
        points = [center(cell) for cell in obstacles]
        if not _apart(points, min_gap):
            continue
        holes: list[np.ndarray] = []
        margin = phys.hole_radius + 0.1
        for _ in range(200):
            if len(holes) == num_holes:
                break
            point = rng.uniform([margin, margin], [puck_map.width, puck_map.height])
            point = np.round(point, 2)
            near_wall = any(
                _distance_to_rect(point, rect) < margin
                for rect in puck_map.wall_rects()
            )
            near_obstacle = any(
                np.max(np.abs(point - p)) < size / 2 + margin for p in points
            )
            if near_wall or near_obstacle or not _apart([*points, point], min_gap):
                continue
            holes.append(point)
            points.append(point)
        if len(holes) == num_holes:
            hole_text = ";".join(
                f"{x:.2f},{y:.2f}" for x, y in sorted(map(tuple, holes))
            )
            obstacle_text = ";".join(f"{r},{c}" for r, c in obstacles)
            return f"o:{obstacle_text}|h:{hole_text}"
    raise RuntimeError("Found no layout")


def get_layout_world(world: PuckWorld, key: str) -> PuckWorld:
    """The world's map with the layout's obstacles as walls and its holes in place
    of the map's, with the same physics (see draw_layout())."""
    cache_key = (world.puck_map.name, world.physics, key)
    if cache_key in _LAYOUT_WORLDS:
        _LAYOUT_WORLDS.move_to_end(cache_key)
        return _LAYOUT_WORLDS[cache_key]
    obstacle_text, hole_text = key.split("|")
    obstacles = [
        tuple(int(v) for v in item.split(","))
        for item in obstacle_text[2:].split(";")
        if item
    ]
    holes = tuple(
        (float(item.split(",")[0]), float(item.split(",")[1]))
        for item in hole_text[2:].split(";")
        if item
    )
    puck_map = world.puck_map
    rows = [list(row) for row in puck_map.rows]
    for r, c in obstacles:
        rows[r][c] = "#"
    size = puck_map.cell_size
    centers = tuple(
        (round((c + 0.5) * size, 4), round((puck_map.num_rows - r - 0.5) * size, 4))
        for r, c in obstacles
    )
    layout_map = dataclasses.replace(
        puck_map,
        name=f"{puck_map.name}~{key}",
        rows=tuple("".join(row) for row in rows),
        holes=holes,
        obstacles=centers,
    )
    layout_world = PuckWorld(layout_map, world.physics)
    _LAYOUT_WORLDS[cache_key] = layout_world
    if len(_LAYOUT_WORLDS) > _MAX_LAYOUT_WORLDS:
        _LAYOUT_WORLDS.popitem(last=False)
    return layout_world


def _apart(points: list[np.ndarray], gap: float) -> bool:
    """Whether every two points are at least gap apart."""
    return all(
        np.linalg.norm(a - b) >= gap
        for i, a in enumerate(points)
        for b in points[i + 1 :]
    )


def _distance_to_rect(
    point: np.ndarray, rect: tuple[float, float, float, float]
) -> float:
    """The distance from a point to an axis-aligned rectangle (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    dx = max(x0 - point[0], 0.0, point[0] - x1)
    dy = max(y0 - point[1], 0.0, point[1] - y1)
    return float(np.hypot(dx, dy))
