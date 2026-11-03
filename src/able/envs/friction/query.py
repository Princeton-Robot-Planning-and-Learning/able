"""Queries in the friction environment: probe pushes on one floor.

A probe places the puck on a floor and pushes it along a clear straight run, and the
response is the push's measurements. Probe poses are searched per map: a puck start
and heading whose straight run (3-sigma heading cone, clear of walls, holes by 0.1 m
and other floors) fits the paddle's acceleration plus the slowest-friction slide at a
3-sigma fast strike. Where no run fits, the longest one is used at a lower speed.
On threeroutes L2 (lowest friction 0.024):

- blue: puck (0.95, 2.85), heading 20 deg, 0.80 m/s
- tan: puck (3.45, 1.65), heading 30 deg, 0.70 m/s (longest run 1.76 m)
- green: puck (2.85, 0.65), heading 10 deg, 0.80 m/s (right of the (2.6, 1.0) hole)
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.envs.friction.levels import level_world, make_level_env
from able.envs.friction.world import GRAVITY, PuckMap, PuckPhysics, PuckWorld, Push

_SIGMAS = 3.0  # strike noise a probe's run must cover
BACKOFF = 0.05  # robot start behind its push station (m)


@dataclass(frozen=True)
class FrictionQuery:
    """A probe push on a floor: the puck placed at `puck`, pushed along `angle` at
    `speed`, with the robot choosing its stance."""

    floor: int
    puck: tuple[float, float]
    angle: float
    speed: float

    @property
    def push(self) -> Push:
        """The probe as a push."""
        return Push(self.angle, self.speed)


@dataclass(frozen=True)
class PushRecord:
    """What the robot measures in one push, from where the puck started.

    The release speed and the stop are noisy. The robot sees whether the push was
    feasible, whether the puck fell in a hole or stopped in the goal, and how far it
    drove.
    """

    puck: tuple[float, float]
    push: Push
    feasible: bool
    release_speed: float
    stop: tuple[float, float]
    fell_in_hole: bool
    goal_reached: bool
    drive_distance: float

    @classmethod
    def from_info(
        cls, info: dict[str, Any], puck: ArrayLike, push: Push
    ) -> "PushRecord":
        """The record of a simulator step's info."""
        start = np.asarray(puck, dtype=np.float64)
        stop = np.asarray(info["stop_position"], dtype=np.float64)
        return cls(
            puck=(float(start[0]), float(start[1])),
            push=push,
            feasible=bool(info["push_feasible"]),
            release_speed=float(info["release_speed"]),
            stop=(float(stop[0]), float(stop[1])),
            fell_in_hole=bool(info["fell_in_hole"]),
            goal_reached=bool(info["goal_reached"]),
            drive_distance=float(info["drive_distance"]),
        )


def probe_queries(
    world: PuckWorld, low: ArrayLike, speed: float = 0.8, margin: float = 0.1
) -> tuple[FrictionQuery, ...]:
    """One probe per floor; `low` is the lowest possible friction per floor."""
    lows = tuple(float(v) for v in np.asarray(low))
    return _probe_queries(world.puck_map, world.physics, lows, speed, margin)


@lru_cache(maxsize=16)
def _probe_queries(
    puck_map: PuckMap,
    physics: PuckPhysics,
    low: tuple[float, ...],
    speed: float,
    margin: float,
) -> tuple[FrictionQuery, ...]:
    world = PuckWorld(puck_map, physics)
    queries = []
    fast = 1.0 + _SIGMAS * physics.speed_noise
    for floor, mu in enumerate(low):
        # Worst case: fastest strike on the slipperiest floor, after the paddle's
        # acceleration.
        need = physics.accel_distance + (speed * fast) ** 2 / (2.0 * mu * GRAVITY)
        puck, angle, run = _longest_run(world, floor, need + margin)
        room = run - physics.accel_distance - margin
        fit = math.sqrt(max(room, 0.0) * 2.0 * mu * GRAVITY) / fast
        probe_speed = round(max(min(speed, fit), physics.min_speed), 4)
        queries.append(FrictionQuery(floor, puck, angle, probe_speed))
    return tuple(queries)


def _longest_run(
    world: PuckWorld,
    floor: int,
    need: float,
    grid: float = 0.1,
    step: float = 0.04,
    max_run: float = 4.0,
) -> tuple[tuple[float, float], float, float]:
    """A feasible probe (puck, angle) on the floor with the longest clear straight
    run up to `need`, and that run.

    Ties go to the start farthest from holes.
    """
    phys, pm = world.physics, world.puck_map
    xs = np.arange(grid / 2, pm.width, grid)
    ys = np.arange(grid / 2, pm.height, grid)
    pucks = np.array([(x, y) for x in xs for y in ys])
    pucks = pucks[
        [world.surface_at(p) == floor and world.puck_position_free(p) for p in pucks]
    ]
    angles = np.radians(np.arange(-180.0, 180.0, 10.0))
    s = np.arange(step, max_run + 1e-9, step)
    run = np.full((len(pucks), len(angles)), max_run)
    spread = _SIGMAS * phys.angle_noise
    for offset in (-spread, 0.0, spread):
        dirs = np.stack([np.cos(angles + offset), np.sin(angles + offset)], -1)
        pts = pucks[:, None, None] + s[None, None, :, None] * dirs[None, :, None]
        row = np.clip((pts[..., 1] // pm.cell_size).astype(int), 0, pm.num_rows - 1)
        col = np.clip((pts[..., 0] // pm.cell_size).astype(int), 0, pm.num_cols - 1)
        ok = world.surfaces[row, col] == floor
        for x0, y0, x1, y1 in world.walls:
            dx = np.maximum(np.maximum(x0 - pts[..., 0], pts[..., 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - pts[..., 1], pts[..., 1] - y1), 0.0)
            ok &= dx**2 + dy**2 > phys.puck_radius**2
        for hole in world.holes:
            clearance = phys.hole_radius + phys.puck_radius + 0.1
            ok &= np.linalg.norm(pts - hole, axis=-1) > clearance
        first_bad = np.where(ok.all(-1), len(s), np.argmin(ok, -1))
        run = np.minimum(run, first_bad * step)  # clear up to the last good sample
    gaps = [np.linalg.norm(pucks - hole, axis=-1) for hole in world.holes]
    gap = np.min(gaps + [np.full(len(pucks), 1.0)], 0)
    score = np.minimum(run, need) + 1e-3 * gap[:, None]
    for k in np.argsort(-score, axis=None, kind="stable"):
        index = np.unravel_index(k, run.shape)
        i, j = int(index[0]), int(index[1])
        if world.push_feasible(pucks[i], angles[j]):
            puck = (round(float(pucks[i][0]), 4), round(float(pucks[i][1]), 4))
            return puck, round(float(angles[j]), 4), float(run[i, j])
    raise ValueError(f"No feasible probe on floor {floor} of {pm.name}")


def execute_push(env: Any, push: Push) -> tuple[bool, dict[str, Any]]:
    """Step the simulator with the push and return (terminated, info)."""
    stance = [] if push.stance is None else [push.stance]
    action = np.array([push.angle, push.speed] + stance)
    _, _, terminated, _, info = env.step(action)
    return bool(terminated), info


class ProbeSimulator:
    """Runs probes in a simulator of its own, built lazily, so a probe never
    disturbs an episode."""

    def __init__(self, sim: str = "2d", level: str = "L2") -> None:
        self._sim = sim
        self._level = level
        self.world = level_world(level)
        self._env: Any = None

    @property
    def env(self) -> Any:
        """The probe simulator."""
        if self._env is None:
            self._env = make_level_env(self._sim, self._level)
        return self._env

    def ask(
        self,
        frictions: NDArray[np.float64],
        query: FrictionQuery,
        rng: np.random.Generator,
    ) -> PushRecord:
        """The record of the probe at the given frictions."""
        start, push = np.array(query.puck, dtype=np.float64), query.push
        pose = self.world.push_pose(start, query.angle)
        assert pose is not None, query
        base, heading = pose
        robot = base - BACKOFF * push.direction
        options = {
            "frictions": tuple(float(f) for f in frictions),
            "puck": tuple(query.puck),
            "robot": (float(robot[0]), float(robot[1]), float(heading)),
        }
        self.env.reset(seed=int(rng.integers(2**31)), options=options)
        _, info = execute_push(self.env, push)
        return PushRecord.from_info(info, start, push)

    def close(self) -> None:
        """Close the simulator, if built."""
        if self._env is not None:
            self._env.close()
            self._env = None
