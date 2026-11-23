"""Tasks and domains for the friction environment."""

import dataclasses
from typing import Collection

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.beliefs.particles import ParticleBelief
from able.envs.base import Domain, TaskDistribution
from able.envs.friction.layout import get_layout_world
from able.envs.friction.model import (
    FrictionModel,
    FrictionState,
    FrictionTask,
    PuckRewards,
)
from able.envs.friction.pomdp import PuckPushingModel
from able.envs.friction.query import FrictionQuery, PushRecord
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, FrictionPrior, Push
from able.envs.simulated import SimulatedEnvironment

FrictionDomain = Domain[
    FrictionState,
    Push,
    FrictionTask,
    NDArray[np.float64],
    FrictionQuery,
    PushRecord,
]


class FrictionTaskDistribution(TaskDistribution[FrictionTask]):
    """Each task pushes the puck from a random start to a random goal.

    The goal is a pushable spot on the floor whose goal circle is clear of the
    holes. The start is a pushable spot clear of the holes, which the puck is not
    shed from, whose shortest pushable path to the goal is min_distance to
    max_distance long. So one task crosses some of the floors, and later tasks
    reach floors that earlier ones did not. Both lie on the grid of the paths
    behind the planner's leaf values (PuckPushingModel.value_resolution). The robot
    starts at a random spot robot_distance from the puck, facing +x. With floors,
    the start, the goal, and a shortest path between them lie on those floors.
    With a layout (layout.draw_layout()), tasks are in that layout's world.
    """

    def __init__(
        self,
        model: FrictionModel,
        min_distance: float = 1.5,
        max_distance: float = 3.5,
        robot_distance: tuple[float, float] = (0.6, 1.2),
        hole_clearance: float = 0.1,
        floors: Collection[int] | None = None,
        layout: str | None = None,
    ) -> None:
        assert 0.0 < min_distance <= max_distance, "Need 0 < min <= max distance"
        self._model = model
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._robot_distance = robot_distance
        self._layout = layout
        world = model.world if layout is None else get_layout_world(model.world, layout)
        self._world = world
        phys = world.physics
        self._paths = PuckPushingModel(world)
        pushable = self._paths.pushable_cells
        res = self._paths.value_resolution
        j, i = np.nonzero(pushable)
        points = np.stack([(i + 0.5) * res, (j + 0.5) * res], axis=1)
        gaps = np.full(len(points), np.inf)
        for hole in world.holes:
            gaps = np.minimum(gaps, np.linalg.norm(points - hole, axis=1))
        clear = phys.hole_radius + hole_clearance
        self._floors = None if floors is None else frozenset(floors)
        on_floors = np.ones(len(points), dtype=bool)
        if self._floors is not None:
            on_floors = np.isin(self._paths.cell_floors[j, i], list(self._floors))
        self._goals = points[(gaps >= clear + phys.goal_radius) & on_floors]
        starts = (
            (gaps >= clear + phys.puck_radius)
            & np.array([world.has_room(p) for p in points], dtype=bool)
            & on_floors
        )
        self._starts = points[starts]
        self._start_cells = (j * pushable.shape[1] + i)[starts]

    def sample(self, rng: np.random.Generator) -> FrictionTask:
        world = self._world
        for _ in range(1000):
            goal = self._goals[rng.integers(len(self._goals))]
            goal_point = (round(float(goal[0]), 4), round(float(goal[1]), 4))
            distances = self._paths.path_distances(goal_point).flat[self._start_cells]
            ok = (distances >= self._min_distance) & (distances <= self._max_distance)
            if self._floors is not None:
                paths = self._paths.path_distances(goal_point, self._floors)
                ok &= np.isclose(paths.flat[self._start_cells], distances)
            starts = [p for p in self._starts[ok] if not world.in_goal(p, goal_point)]
            if not starts:
                continue
            start = starts[rng.integers(len(starts))]
            robot = self._sample_robot(start, rng)
            if robot is None:
                continue
            puck = (round(float(start[0]), 4), round(float(start[1]), 4))
            state = self._model.get_initial_state(puck, robot, goal_point, self._layout)
            return FrictionTask(state, goal_point)
        raise RuntimeError("Found no start and goal for a friction task")

    def _sample_robot(
        self, puck: NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[float, float] | None:
        """A random spot on the robot's driving grid, robot_distance from the
        puck, where the robot clears the walls at any heading, or None if there is
        none."""
        world = self._world
        res = world.physics.nav_resolution
        low, high = self._robot_distance
        steps = np.arange(-high, high + 1e-9, res)
        candidates = puck + np.stack(np.meshgrid(steps, steps), -1).reshape(-1, 2)
        candidates = np.round((candidates - res / 2) / res) * res + res / 2
        gaps = np.linalg.norm(candidates - puck, axis=1)
        candidates = candidates[(gaps >= low) & (gaps <= high)]
        free = [c for c in candidates if world.robot_position_free(c)]
        if not free:
            return None
        robot = free[rng.integers(len(free))]
        return (round(float(robot[0]), 4), round(float(robot[1]), 4))


def make_prior_belief(
    prior: FrictionPrior, num_particles: int, rng: np.random.Generator
) -> ParticleBelief:
    """Particles drawn from the prior, with a resampling seed from the stream."""
    low, high = prior.bounds
    particles = prior.sample(rng, num_particles)
    return ParticleBelief(particles, low, high, seed=int(rng.integers(2**31)))


def make_domain(
    seed: int,
    sim: str = "2d",
    num_holes: int = 2,
    goal_radius: float = 0.3,
    horizon: int = 15,
    num_particles: int = 500,
    task_distances: tuple[float, float] = (1.5, 3.5),
    num_floors: int = 3,
    prior: FrictionPrior = DEFAULT_FRICTION_PRIOR,
    rewards: PuckRewards = PuckRewards(),
    true_frictions: ArrayLike | None = None,
    maze: str = "threeroutes",
) -> FrictionDomain:
    """A friction domain on the maze (see make_world) with num_holes holes, the
    goal radius, and num_floors floors, with floor frictions drawn from the prior
    unless given.
    Each task's start is task_distances (least, most) from its goal along the
    shortest pushable path. The robot's initial belief is particles from the
    prior."""
    rng = np.random.default_rng(seed)
    model = FrictionModel(
        sim, num_holes, goal_radius, horizon, rewards, num_floors, maze
    )
    prior = dataclasses.replace(prior, num_surfaces=num_floors)  # a draw per floor
    frictions = prior.sample(rng) if true_frictions is None else true_frictions
    frictions = np.asarray(frictions, dtype=np.float64)
    assert frictions.shape == (num_floors,), "Need one friction per floor"
    return Domain(
        env_model=model,
        task_distribution=FrictionTaskDistribution(model, *task_distances),
        initial_belief=make_prior_belief(prior, num_particles, rng),
        real_env=SimulatedEnvironment(model, frictions, rng),
    )
