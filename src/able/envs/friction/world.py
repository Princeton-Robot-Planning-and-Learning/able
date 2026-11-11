"""The friction environment's world: maps, physics, the friction prior, the analytic
slide, and push geometry.

A robot pushes a puck over floors of unknown friction to a goal. The hidden parameters
are the friction coefficients of the floors, named by color. Everything else is known.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SURFACE_NAMES: tuple[str, ...] = ("blue", "tan", "green")
# Map chars, in SURFACE_NAMES order.
SURFACE_CHARS: tuple[str, ...] = ("b", "t", "g")
WALL_CHAR = "#"


@dataclass(frozen=True)
class FrictionPrior:
    """Friction prior: `levels` shuffled over surfaces, each scaled by U[1+-jitter]."""

    levels: tuple[float, ...] = (0.03, 0.08, 0.15)
    jitter: float = 0.2

    @property
    def num_surfaces(self) -> int:
        """The number of floors."""
        return len(self.levels)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Lowest and highest possible coefficient per surface."""
        n = self.num_surfaces
        lo = min(self.levels) * (1.0 - self.jitter)
        hi = max(self.levels) * (1.0 + self.jitter)
        return np.full(n, lo), np.full(n, hi)

    @property
    def mean(self) -> NDArray[np.float64]:
        """The mean coefficient per floor."""
        return np.full(self.num_surfaces, float(np.mean(self.levels)))

    def sample(
        self, rng: np.random.Generator, size: int | None = None
    ) -> NDArray[np.float64]:
        """One coefficient vector, or `size` of them stacked."""
        n = 1 if size is None else size
        levels = np.asarray(self.levels, dtype=np.float64)
        order = np.argsort(rng.random((n, self.num_surfaces)), axis=1)
        scale = rng.uniform(
            1.0 - self.jitter, 1.0 + self.jitter, (n, self.num_surfaces)
        )
        samples = levels[order] * scale
        return samples[0] if size is None else samples


DEFAULT_FRICTION_PRIOR = FrictionPrior()
SURFACE_RGB: tuple[tuple[float, float, float], ...] = (
    (0.80, 0.86, 0.92),  # blue
    (0.76, 0.60, 0.42),  # tan
    (0.30, 0.50, 0.32),  # green
)
WALL_RGB = (0.25, 0.25, 0.25)
HOLE_RGB = (0.9, 0.1, 0.1)
GOAL_RGB = (0.2, 0.8, 0.2)
PUCK_RGB = (0.95, 0.75, 0.1)

GRAVITY = 9.81

# One friction coefficient per floor, in SURFACE_NAMES order.
Frictions = Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True)
class PuckMap:
    """A floor maze (m); `rows` are top to bottom, row indices count from the bottom."""

    name: str
    rows: tuple[str, ...]
    cell_size: float
    puck_start: tuple[float, float]
    robot_start: tuple[float, float]
    goal: tuple[float, float]
    holes: tuple[tuple[float, float], ...] = ()

    @property
    def num_cols(self) -> int:
        """Cells along x."""
        return len(self.rows[0])

    @property
    def num_rows(self) -> int:
        """Cells along y."""
        return len(self.rows)

    @property
    def width(self) -> float:
        """Extent along x (m)."""
        return self.num_cols * self.cell_size

    @property
    def height(self) -> float:
        """Extent along y (m)."""
        return self.num_rows * self.cell_size

    def surface_grid(self) -> NDArray[np.int64]:
        """Surface index per cell, indexed [row_from_bottom, col]; -1 is a wall."""
        grid = np.full((self.num_rows, self.num_cols), -1, dtype=np.int64)
        for r, line in enumerate(self.rows):
            assert len(line) == self.num_cols, f"Ragged map row {r} in {self.name}"
            for c, char in enumerate(line):
                if char == WALL_CHAR:
                    continue
                grid[self.num_rows - 1 - r, c] = SURFACE_CHARS.index(char)
        return grid

    def wall_rects(self) -> list[tuple[float, float, float, float]]:
        """Walls as few axis-aligned rectangles (x0, y0, x1, y1)."""
        return self.cell_rects(-1)

    def surface_rects(self) -> list[tuple[int, tuple[float, float, float, float]]]:
        """Floor patches as (surface index, rectangle) pairs."""
        return [
            (surface, rect)
            for surface in range(len(SURFACE_CHARS))
            for rect in self.cell_rects(surface)
        ]

    def cell_rects(self, value: int) -> list[tuple[float, float, float, float]]:
        """Cells holding `value` merged into few rectangles (x0, y0, x1, y1)."""
        grid = self.surface_grid()
        h = self.cell_size
        runs: dict[tuple[int, int], list[int]] = {}
        for row in range(self.num_rows):
            col = 0
            while col < self.num_cols:
                if grid[row, col] != value:
                    col += 1
                    continue
                start = col
                while col < self.num_cols and grid[row, col] == value:
                    col += 1
                runs.setdefault((start, col), []).append(row)
        rects = []
        for (c0, c1), rows in sorted(runs.items()):
            r0 = prev = rows[0]
            for r in rows[1:]:
                if r == prev + 1:
                    prev = r
                    continue
                rects.append(_rect(c0, r0, c1, prev + 1, h))
                r0 = prev = r
            rects.append(_rect(c0, r0, c1, prev + 1, h))
        return rects


def _rect(
    c0: int, r0: int, c1: int, r1: int, h: float
) -> tuple[float, float, float, float]:
    """Cell-index bounds to meters, rounded so shared edges match exactly."""
    return (round(c0 * h, 9), round(r0 * h, 9), round(c1 * h, 9), round(r1 * h, 9))


# 1.2 m corridors fit the TidyBot base. The puck crosses the robot-blocked blue slot
# only if blue is slippery (else it sticks inside); top-corridor pushes reveal which.
THREE_ROUTES = PuckMap(
    name="threeroutes",
    rows=(
        "##############",
        "#bbbbbbbbbbbb#",
        "#bbbbbbbbbbbb#",
        "#bbbbbbbbbbbb#",
        "#tttt###ttttt#",
        "#ttttbbbbtttt#",
        "#tttt###ttttt#",
        "#gggggggggggg#",
        "#gggggggggggg#",
        "#gggggggggggg#",
        "##############",
    ),
    cell_size=0.4,
    puck_start=(1.2, 2.2),
    robot_start=(1.2, 1.0),
    goal=(3.9, 2.2),
    holes=((4.8, 3.6), (2.6, 1.0)),
)

MAPS: dict[str, PuckMap] = {m.name: m for m in (THREE_ROUTES,)}


@dataclass(frozen=True)
class PuckPhysics:
    """Known physical parameters of the puck, the robot and the strike."""

    puck_radius: float = 0.12
    puck_height: float = 0.12
    puck_mass: float = 1.0
    hole_radius: float = 0.2
    goal_radius: float = 0.3
    wall_restitution: float = 0.5
    # TidyBot base half extents (m) along/across its heading.
    base_half_extents: tuple[float, float] = (0.275, 0.255)
    # Base-to-wall gap where it stands and strikes.
    base_wall_margin: float = 0.01
    # Base disc for puck bounces and the 2D env.
    robot_radius: float = 0.3
    # Extra driving gap from the puck: the 3D base's box corners reach 0.375 m.
    drive_clearance: float = 0.08
    # Robot center to paddle face, along its heading.
    paddle_offset: float = 0.45
    # Stance: base heading relative to the push; if unset, first fitting `base_offsets`.
    max_stance: float = math.radians(60)
    base_offsets: tuple[float, ...] = (
        0.0,
        math.radians(30),
        -math.radians(30),
        math.radians(60),
        -math.radians(60),
    )
    # Narrower than the puck (0.24 m), so it can push a puck resting near a wall
    # along the wall. A paddle wider than the puck left a strip along every wall
    # where no push was feasible.
    paddle_width: float = 0.22
    paddle_thickness: float = 0.04
    # Paddle accelerates the puck over accel_distance, then brakes over brake_distance.
    accel_distance: float = 0.3
    brake_distance: float = 0.1
    max_speed: float = 1.2
    min_speed: float = 0.05
    # Strike noise: relative speed error, heading error (rad).
    speed_noise: float = 0.05
    angle_noise: float = math.radians(2.0)
    # Measurement noise: release speed (m/s), stop position (m).
    speed_obs_noise: float = 0.02
    position_obs_noise: float = 0.02
    # Drive-planning grid resolution (m).
    nav_resolution: float = 0.1


@dataclass
class SlideResult:
    """Outcome of one slide."""

    stop: NDArray[np.float64]
    # Corner points from release to stop.
    path: list[NDArray[np.float64]]
    # Hole the center passed over (slide ends there).
    hole: int | None = None
    distance: float = 0.0


@dataclass(frozen=True)
class StrikeEnd:
    """Robot rest pose after a strike (base disc, paddle face square to the push)."""

    base: NDArray[np.float64]
    face: NDArray[np.float64]
    angle: float


@dataclass(frozen=True)
class Push:
    """Release heading (rad), speed (m/s), stance (rad vs push; None: auto)."""

    angle: float
    speed: float
    stance: float | None = None

    @property
    def direction(self) -> NDArray[np.float64]:
        """Unit vector of the heading."""
        return np.array([math.cos(self.angle), math.sin(self.angle)])


class PuckWorld:
    """A map with the physics: slides, push feasibility and driving distances."""

    def __init__(self, puck_map: PuckMap, physics: PuckPhysics | None = None) -> None:
        self.puck_map = puck_map
        self.physics = physics if physics is not None else PuckPhysics()
        self.surfaces = self.puck_map.surface_grid()
        self.walls = np.asarray(self.puck_map.wall_rects(), dtype=np.float64)
        self.holes = np.asarray(self.puck_map.holes, dtype=np.float64).reshape(-1, 2)
        self.goal = np.asarray(self.puck_map.goal, dtype=np.float64)
        self._nav_cache: dict[int, tuple[NDArray[np.float64], NDArray[np.int32]]] = {}
        # LRU, keyed by (start node, puck x, puck y).
        self._nav_puck_cache: OrderedDict[
            tuple[int, float, float], tuple[NDArray[np.float64], NDArray[np.int32]]
        ] = OrderedDict()

    # ------------------------------------------------------------------ surfaces

    def cell_of(self, point: NDArray[np.float64]) -> tuple[int, int]:
        """(row_from_bottom, col) of the cell holding `point`, clamped to the map."""
        h = self.puck_map.cell_size
        col = min(max(int(point[0] // h), 0), self.puck_map.num_cols - 1)
        row = min(max(int(point[1] // h), 0), self.puck_map.num_rows - 1)
        return row, col

    def surface_at(self, point: NDArray[np.float64]) -> int:
        """Surface index under `point`, or -1 inside a wall."""
        return int(self.surfaces[self.cell_of(point)])

    def friction_at(self, point: NDArray[np.float64], frictions: Frictions) -> float:
        """Friction coefficient under `point` (the highest inside a wall)."""
        surface = self.surface_at(point)
        if surface < 0:
            return float(max(frictions))
        return float(frictions[surface])

    # ------------------------------------------------------------------ geometry

    def _ray_hit(
        self,
        rects: NDArray[np.float64],
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
        inflate: float,
    ) -> tuple[float, NDArray[np.float64]]:
        """(distance, entry-face normal axes) of the first entry into inflated rects.

        Exit faces are ignored, so a point resting on a face and moving away misses it.
        """
        best, best_axes = math.inf, np.zeros(2)
        if len(rects) == 0:
            return best, best_axes
        lo = rects[:, :2] - inflate
        hi = rects[:, 2:] + inflate
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / direction
            t1 = (lo - origin) * inv
            t2 = (hi - origin) * inv
        tmin = np.where(np.isnan(t1), -math.inf, np.minimum(t1, t2))
        tmax = np.where(np.isnan(t2), math.inf, np.maximum(t1, t2))
        # Axes with zero direction: inside the slab or never.
        for axis in range(2):
            if direction[axis] == 0.0:
                inside = (origin[axis] > lo[:, axis]) & (origin[axis] < hi[:, axis])
                tmin[:, axis] = np.where(inside, -math.inf, math.inf)
                tmax[:, axis] = np.where(inside, math.inf, -math.inf)
        enter = tmin.max(axis=1)
        leave = tmax.min(axis=1)
        eps = 1e-9
        candidates = np.nonzero((enter <= leave) & (leave > eps) & (enter > -1e-6))[0]
        for i in candidates:
            t = float(enter[i])
            if t >= best:
                continue
            axes = (np.abs(tmin[i] - t) < 1e-9).astype(float)
            # Skip unless moving into the entry face.
            if not np.any(axes * np.abs(direction) > 0):
                continue
            best, best_axes = max(t, 0.0), axes
        return best, best_axes

    def _segment_hits_rects(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        inflate: float,
    ) -> bool:
        """Whether a disc of radius `inflate` swept start to end hits a wall."""
        lo = self.walls[:, :2] - inflate
        hi = self.walls[:, 2:] + inflate
        delta = end - start
        t0 = np.zeros(len(self.walls))
        t1 = np.ones(len(self.walls))
        for axis in range(2):
            d = delta[axis]
            if abs(d) < 1e-12:
                outside = (start[axis] <= lo[:, axis]) | (start[axis] >= hi[:, axis])
                t0 = np.where(outside, 2.0, t0)
                continue
            a = (lo[:, axis] - start[axis]) / d
            b = (hi[:, axis] - start[axis]) / d
            t0 = np.maximum(t0, np.minimum(a, b))
            t1 = np.minimum(t1, np.maximum(a, b))
        return bool(np.any(t0 < t1))

    def _box_hits_walls(
        self,
        center: NDArray[np.float64],
        angle: float,
        half_length: float,
        half_width: float,
    ) -> bool:
        """Separating-axis test: does a box rotated by `angle` overlap a wall?"""
        u = np.array([math.cos(angle), math.sin(angle)])
        v = np.array([-u[1], u[0]])
        wall_center = 0.5 * (self.walls[:, :2] + self.walls[:, 2:])
        wall_half = 0.5 * (self.walls[:, 2:] - self.walls[:, :2])
        rel = center - wall_center
        eps = 1e-9
        # The walls' axes (x and y).
        box_x = half_length * abs(u[0]) + half_width * abs(v[0])
        box_y = half_length * abs(u[1]) + half_width * abs(v[1])
        overlap = (np.abs(rel[:, 0]) < wall_half[:, 0] + box_x - eps) & (
            np.abs(rel[:, 1]) < wall_half[:, 1] + box_y - eps
        )
        # The box's axes.
        for axis, half in ((u, half_length), (v, half_width)):
            wall_extent = wall_half[:, 0] * abs(axis[0]) + wall_half[:, 1] * abs(
                axis[1]
            )
            overlap &= np.abs(rel @ axis) < half + wall_extent - eps
        return bool(np.any(overlap))

    def _segment_hole(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        radius: float,
    ) -> tuple[int, float] | None:
        """First hole within `radius` of the segment, as (index, entry fraction)."""
        if len(self.holes) == 0:
            return None
        delta = end - start
        length2 = float(delta @ delta)
        best_index, best_t = -1, math.inf
        for i, center in enumerate(self.holes):
            rel = start - center
            if length2 < 1e-18:
                if float(rel @ rel) <= radius**2:
                    return i, 0.0
                continue
            # |start + t delta - center|^2 = radius^2
            b = float(rel @ delta)
            c = float(rel @ rel) - radius**2
            disc = b * b - length2 * c
            if disc < 0:
                continue
            t_in = (-b - math.sqrt(disc)) / length2
            t_out = (-b + math.sqrt(disc)) / length2
            if t_out < 0 or t_in > 1:
                continue
            t = max(t_in, 0.0)
            if t < best_t:
                best_index, best_t = i, t
        return None if best_index < 0 else (best_index, best_t)

    def puck_path_clear(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        hole_clearance: float | None = None,
    ) -> bool:
        """Whether a straight puck move clears walls and holes (by `hole_clearance`)."""
        if self._segment_hits_rects(start, end, self.physics.puck_radius):
            return False
        clearance = (
            self.physics.hole_radius if hole_clearance is None else hole_clearance
        )
        return self._segment_hole(start, end, clearance) is None

    def nearest_free_puck_position(
        self, point: NDArray[np.float64], max_shift: float = 0.1, step: float = 0.005
    ) -> NDArray[np.float64]:
        """Nearest wall-free puck position within `max_shift` (for noisy readings)."""
        point = np.asarray(point, dtype=np.float64)
        if self.puck_position_free(point):
            return point
        for radius in np.arange(step, max_shift + 1e-9, step):
            count = max(8, int(2 * math.pi * radius / step))
            for theta in np.linspace(0.0, 2 * math.pi, count, endpoint=False):
                candidate = point + radius * np.array(
                    [math.cos(theta), math.sin(theta)]
                )
                if self.puck_position_free(candidate):
                    return candidate
        return point

    def puck_position_free(self, point: NDArray[np.float64]) -> bool:
        """Whether the puck can rest at `point` without overlapping a wall."""
        return not self._segment_hits_rects(point, point, self.physics.puck_radius)

    def over_hole(self, point: NDArray[np.float64]) -> bool:
        """Whether the puck's center at `point` is over a hole."""
        if len(self.holes) == 0:
            return False
        gaps = np.linalg.norm(self.holes - point, axis=1)
        return bool(np.any(gaps <= self.physics.hole_radius))

    def in_goal(self, point: NDArray[np.float64]) -> bool:
        """Whether a puck stopped at `point` is in the goal."""
        return float(np.linalg.norm(point - self.goal)) <= self.physics.goal_radius

    def strike_end(
        self, station: NDArray[np.float64], puck: NDArray[np.float64], angle: float
    ) -> StrikeEnd:
        """Robot rest pose after a push: base and paddle advanced by accel_distance."""
        direction = np.array([math.cos(angle), math.sin(angle)])
        shift = self.physics.accel_distance * direction
        return StrikeEnd(
            base=np.asarray(station, dtype=np.float64) + shift,
            face=puck - self.physics.puck_radius * direction + shift,
            angle=angle,
        )

    def _robot_hit(
        self,
        robot: StrikeEnd,
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64] | None]:
        """First puck contact with the robot's base or paddle: (distance, normal)."""
        phys = self.physics
        best: tuple[float, NDArray[np.float64] | None] = (math.inf, None)
        # Base: a disc.
        reach = phys.robot_radius + phys.puck_radius
        rel = origin - robot.base
        b = float(rel @ direction)
        c = float(rel @ rel) - reach * reach
        disc = b * b - c
        if b < 0.0 <= disc:
            t = -b - math.sqrt(disc)
            if t > -1e-6:
                best = (max(t, 0.0), (rel + t * direction) / reach)
        # Paddle: a box square to the push, behind its face.
        cos_a, sin_a = math.cos(robot.angle), math.sin(robot.angle)

        def to_local(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([cos_a * v[0] + sin_a * v[1], -sin_a * v[0] + cos_a * v[1]])

        half_w = phys.paddle_width / 2
        rect = np.array([[-phys.paddle_thickness, -half_w, 0.0, half_w]])
        t, axes = self._ray_hit(
            rect, to_local(origin - robot.face), to_local(direction), phys.puck_radius
        )
        if t < best[0]:
            local_dir = to_local(direction)
            local_normal = -axes * np.sign(local_dir)
            local_normal /= max(float(np.linalg.norm(local_normal)), 1e-12)
            normal = np.array(
                [
                    cos_a * local_normal[0] - sin_a * local_normal[1],
                    sin_a * local_normal[0] + cos_a * local_normal[1],
                ]
            )
            best = (t, normal)
        return best

    # ------------------------------------------------------------------ slides

    def slide(
        self,
        start: NDArray[np.float64],
        speed: float,
        angle: float,
        frictions: Frictions,
        max_events: int = 1000,
        robot: StrikeEnd | None = None,
    ) -> SlideResult:
        """Slide the puck until it stops.

        Decel is friction * g under the center; frictionless bounces off walls, `robot`.
        """
        phys = self.physics
        h = self.puck_map.cell_size
        point = np.array(start, dtype=np.float64)
        direction = np.array([math.cos(angle), math.sin(angle)])
        path = [point.copy()]
        total = 0.0
        v = float(speed)
        for _ in range(max_events):
            if v <= 1e-6:
                break
            probe = point + 1e-7 * direction
            decel = self.friction_at(probe, frictions) * GRAVITY
            s_stop = v * v / (2.0 * decel)
            # Step to the next cell boundary, so friction is constant over the step.
            s_cell = math.inf
            for axis in range(2):
                d = direction[axis]
                if abs(d) < 1e-12:
                    continue
                idx = math.floor(probe[axis] / h)
                boundary = (idx + 1) * h if d > 0 else idx * h
                s_cell = min(s_cell, (boundary - point[axis]) / d)
            s_cell = max(s_cell, 0.0)
            s_wall, axes = self._ray_hit(self.walls, point, direction, phys.puck_radius)
            s_robot, normal = (
                self._robot_hit(robot, point, direction)
                if robot is not None
                else (math.inf, None)
            )
            step = min(s_stop, s_cell, s_wall, s_robot)
            end = point + step * direction
            hole = self._segment_hole(point, end, phys.hole_radius)
            if hole is not None:
                index, frac = hole
                point = point + frac * step * direction
                total += frac * step
                path.append(point.copy())
                return SlideResult(point, path, hole=index, distance=total)
            point = end
            total += step
            if step == s_stop:
                v = 0.0
                path.append(point.copy())
                break
            v = math.sqrt(max(v * v - 2.0 * decel * step, 0.0))
            if s_wall == step < s_cell + 1e-12:
                path.append(point.copy())
                velocity = v * direction
                velocity = np.where(
                    axes > 0, -phys.wall_restitution * velocity, velocity
                )
                v = float(np.linalg.norm(velocity))
                if v <= 1e-6:
                    break
                direction = velocity / v
            elif s_robot == step < s_cell + 1e-12:
                assert normal is not None
                path.append(point.copy())
                velocity = v * direction
                velocity = (
                    velocity
                    - (1.0 + phys.wall_restitution) * float(velocity @ normal) * normal
                )
                v = float(np.linalg.norm(velocity))
                if v <= 1e-6:
                    break
                direction = velocity / v
        if not np.allclose(path[-1], point):
            path.append(point.copy())
        return SlideResult(point, path, hole=None, distance=total)

    # ------------------------------------------------------------------ pushes

    def push_pose(
        self,
        puck: NDArray[np.float64],
        angle: float,
        stance: float | None = None,
    ) -> tuple[NDArray[np.float64], float] | None:
        """Base position and heading to push along `angle`, or None if infeasible.

        Uses `stance` (clipped to +-max_stance), else the first fitting `base_offsets`.
        """
        phys = self.physics
        direction = np.array([math.cos(angle), math.sin(angle)])
        release = self.release_point(puck, angle)
        if self._segment_hits_rects(puck, release, phys.puck_radius):
            return None
        contact = puck - phys.puck_radius * direction
        reach = phys.accel_distance + phys.brake_distance
        sweep = reach * direction
        # Paddle body from its start face to the end of the strike.
        paddle_center = contact + 0.5 * (reach - phys.paddle_thickness) * direction
        if self._box_hits_walls(
            paddle_center,
            angle,
            0.5 * (reach + phys.paddle_thickness),
            0.5 * phys.paddle_width,
        ):
            return None
        if stance is None:
            offsets: tuple[float, ...] = phys.base_offsets
        else:
            offsets = (min(max(stance, -phys.max_stance), phys.max_stance),)
        for offset in offsets:
            heading = math.atan2(math.sin(angle + offset), math.cos(angle + offset))
            base = contact - phys.paddle_offset * np.array(
                [math.cos(heading), math.sin(heading)]
            )
            if not self.robot_clear_of_puck(base, puck):
                continue
            # Base box along its lunge, sampled finer than any wall is thick.
            half_long, half_wide = (
                e + phys.base_wall_margin for e in phys.base_half_extents
            )
            if any(
                self._box_hits_walls(base + t * sweep, heading, half_long, half_wide)
                for t in np.linspace(0.0, 1.0, 5)
            ):
                continue
            return base, heading
        return None

    def push_station(
        self,
        puck: NDArray[np.float64],
        angle: float,
        stance: float | None = None,
    ) -> NDArray[np.float64]:
        """Base position for the push (straight behind the puck if infeasible)."""
        pose = self.push_pose(puck, angle, stance)
        if pose is not None:
            return pose[0]
        direction = np.array([math.cos(angle), math.sin(angle)])
        reach = self.physics.puck_radius + self.physics.paddle_offset
        return puck - reach * direction

    def release_point(
        self, puck: NDArray[np.float64], angle: float
    ) -> NDArray[np.float64]:
        """Where the puck leaves the paddle after the acceleration phase."""
        direction = np.array([math.cos(angle), math.sin(angle)])
        return puck + self.physics.accel_distance * direction

    def robot_position_free(self, point: NDArray[np.float64]) -> bool:
        """Whether the base, at any heading, clears the walls (it bridges holes)."""
        reach = float(np.hypot(*self.physics.base_half_extents))
        return not self._segment_hits_rects(point, point, reach)

    def robot_clear_of_puck(
        self, point: NDArray[np.float64], puck: NDArray[np.float64]
    ) -> bool:
        """Whether the robot's base at `point` keeps off the puck."""
        reach = self.physics.robot_radius + self.physics.puck_radius
        return float(np.linalg.norm(np.asarray(point) - puck)) >= reach

    def push_feasible(
        self,
        puck: NDArray[np.float64],
        angle: float,
        stance: float | None = None,
    ) -> bool:
        """Whether the push is possible (ignores reachability; see `drive_distance`)."""
        return self.push_pose(puck, angle, stance) is not None

    # ------------------------------------------------------------------ driving

    @cached_property
    def _nav_grid(self) -> tuple[NDArray[np.bool_], NDArray[np.int64], csr_matrix]:
        """Free robot positions on a grid, their node ids, and the 8-connected graph."""
        res = self.physics.nav_resolution
        nx = int(round(self.puck_map.width / res))
        ny = int(round(self.puck_map.height / res))
        free = np.zeros((ny, nx), dtype=bool)
        for j in range(ny):
            for i in range(nx):
                free[j, i] = self.robot_position_free(
                    np.array([(i + 0.5) * res, (j + 0.5) * res])
                )
        ids = -np.ones((ny, nx), dtype=np.int64)
        ids[free] = np.arange(int(free.sum()))
        rows, cols, weights = [], [], []
        for j, i in zip(*np.nonzero(free)):
            for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and free[jj, ii]:
                    # No corner cutting on diagonals.
                    if dj and di and not (free[j, ii] and free[jj, i]):
                        continue
                    w = res * math.hypot(dj, di)
                    rows += [ids[j, i], ids[jj, ii]]
                    cols += [ids[jj, ii], ids[j, i]]
                    weights += [w, w]
        n = int(free.sum())
        graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        return free, ids, graph

    @cached_property
    def _nav_centers(self) -> NDArray[np.float64]:
        """Position of each grid node, in node order."""
        free, _, _ = self._nav_grid
        res = self.physics.nav_resolution
        return (np.argwhere(free)[:, ::-1] + 0.5) * res

    def _puck_blocked(
        self, puck: NDArray[np.float64], from_node: int = -1
    ) -> NDArray[np.bool_]:
        """Nodes where the base would touch the puck (and closer than `from_node`)."""
        gaps = np.linalg.norm(self._nav_centers - puck, axis=1)
        phys = self.physics
        reach = phys.robot_radius + phys.puck_radius + phys.drive_clearance
        if from_node >= 0:
            reach = min(reach, float(gaps[from_node]) - 1e-9)
        return gaps < reach

    def nav_node_of(
        self, point: NDArray[np.float64], puck: NDArray[np.float64] | None = None
    ) -> int:
        """Nearest free grid node to `point` (-1 if none), avoiding `puck` if given."""
        free, ids, _ = self._nav_grid
        blocked = self._puck_blocked(puck) if puck is not None else None
        res = self.physics.nav_resolution
        ny, nx = free.shape
        i0 = int(point[0] // res)
        j0 = int(point[1] // res)
        best, best_d = -1, math.inf
        for j in range(max(j0 - 2, 0), min(j0 + 3, ny)):
            for i in range(max(i0 - 2, 0), min(i0 + 3, nx)):
                if not free[j, i]:
                    continue
                if blocked is not None and blocked[ids[j, i]]:
                    continue
                d = math.hypot((i + 0.5) * res - point[0], (j + 0.5) * res - point[1])
                if d < best_d:
                    best, best_d = int(ids[j, i]), d
        return best

    def _shortest_from(
        self, node: int, puck: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Dijkstra distances and predecessors from `node`, around `puck` if given."""
        _, _, graph = self._nav_grid
        if puck is None:
            if node not in self._nav_cache:
                self._nav_cache[node] = dijkstra(  # type: ignore[call-overload]
                    graph, directed=False, indices=node, return_predecessors=True
                )
            return self._nav_cache[node]
        key = (node, round(float(puck[0]), 3), round(float(puck[1]), 3))
        if key in self._nav_puck_cache:
            self._nav_puck_cache.move_to_end(key)
            return self._nav_puck_cache[key]
        keep = np.nonzero(~self._puck_blocked(puck, node))[0]
        sub = graph[keep][:, keep]
        start = int(np.searchsorted(keep, node))
        sub_dist, sub_pred = dijkstra(  # type: ignore[call-overload]
            sub, directed=False, indices=start, return_predecessors=True
        )
        n = graph.shape[0]
        dist = np.full(n, math.inf)
        dist[keep] = sub_dist
        pred = np.full(n, -9999, dtype=np.int32)
        pred[keep] = np.where(sub_pred >= 0, keep[np.maximum(sub_pred, 0)], -9999)
        self._nav_puck_cache[key] = (dist, pred)
        if len(self._nav_puck_cache) > 256:
            self._nav_puck_cache.popitem(last=False)
        return dist, pred

    @staticmethod
    def _path_nodes(pred: NDArray[np.int32], a: int, b: int) -> list[int]:
        """Shortest-path nodes from `b` back to `a` (empty if unreachable)."""
        if b != a and pred[b] < 0:
            return []
        nodes = [b]
        while nodes[-1] != a:
            nodes.append(int(pred[nodes[-1]]))
        return nodes

    def _shortest_to(
        self, a: int, b: int, puck: NDArray[np.float64] | None
    ) -> tuple[float, list[int]]:
        """Length and nodes (from `b` back to `a`) of a shortest drive."""
        dist, pred = self._shortest_from(a)
        nodes = self._path_nodes(pred, a, b)
        if puck is None or not nodes:
            return float(dist[b]), nodes
        # The puck-free path is fine if it already keeps clear of the puck.
        if not self._puck_blocked(puck, a)[nodes].any():
            return float(dist[b]), nodes
        dist, pred = self._shortest_from(a, puck)
        return float(dist[b]), self._path_nodes(pred, a, b)

    def drive_distance(
        self,
        from_node: int,
        to: NDArray[np.float64],
        puck: NDArray[np.float64] | None = None,
    ) -> float:
        """Shortest drive from a node to `to`, around `puck` if given (inf if none)."""
        to_node = self.nav_node_of(to, puck)
        if from_node < 0 or to_node < 0:
            return math.inf
        return self._shortest_to(from_node, to_node, puck)[0]

    def drive_path(
        self,
        start: NDArray[np.float64],
        goal: NDArray[np.float64],
        puck: NDArray[np.float64] | None = None,
    ) -> list[NDArray[np.float64]]:
        """Waypoints of a shortest drive, around `puck` if given (empty if none)."""
        a, b = self.nav_node_of(start), self.nav_node_of(goal, puck)
        if a < 0 or b < 0:
            return []
        _, nodes = self._shortest_to(a, b, puck)
        if not nodes:
            return []
        centers = self._nav_centers
        points = [centers[n].copy() for n in reversed(nodes)]
        return [np.asarray(start, dtype=np.float64)] + points + [np.asarray(goal)]
