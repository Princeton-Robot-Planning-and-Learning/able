"""A robot that pushes pucks at several sites, each with its own floor frictions
(see envs/multi_site.py).

Every site has the same maze, by default four corridors (see layout.py). Sites
come in groups: the sites in a group share each floor's friction level and
differ only in its jitter, as for buildings fitted out with the same materials.
Each group runs a routine of pushing tasks, which every site in the group does,
in its own region of the maze: on a corridors maze, group g's tasks stay in
corridor g mod the number of corridors.
"""

import dataclasses
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from able.beliefs.particles import ParticleBelief
from able.beliefs.sites import SiteBelief
from able.envs.base import Domain
from able.envs.friction.layout import draw_layout, make_world
from able.envs.friction.model import FrictionModel, FrictionTask
from able.envs.friction.pomdp import PuckPushingModel
from able.envs.friction.tasks import (
    FrictionTaskDistribution,
    make_domain,
    make_prior_belief,
)
from able.envs.friction.world import (
    DEFAULT_FRICTION_PRIOR,
    FrictionPrior,
    PuckWorld,
)
from able.envs.multi_site import MultiSiteModel, SiteRoutineTaskDistribution
from able.envs.simulated import SimulatedEnvironment

FrictionSiteDomain = Domain[Any, Any, Any, Any, Any, Any]


def make_site_domain(
    seed: int,
    group_sizes: Sequence[int] = (2, 3, 3, 4),
    routine_size: int = 8,
    revision_rate: float = 0.0,
    sim: str = "2d",
    num_holes: int = 2,
    goal_radius: float = 0.2,
    horizon: int = 15,
    num_particles: int = 500,
    task_distances: tuple[float, float] = (1.5, 3.5),
    num_floors: int = 12,
    prior: FrictionPrior = DEFAULT_FRICTION_PRIOR,
    maze: str = "corridors4",
    region_overlap: float = 0.0,
    revision_region: str = "same",
    layout_holes: int = 0,
    layout_obstacles: int = 0,
    survey: str = "prior",
    friction_groups: str = "same",
    group_friction: str = "base",
    base_levels: Sequence[float] = (0.025, 0.05, 0.1, 0.2),
    floor_noise: float = 0.1,
    site_jitter: float = 0.05,
) -> FrictionSiteDomain:
    """A domain with group_sizes[g] sites in group g.

    With group_friction "base", each friction group draws its own base friction
    from base_levels, without replacement, and each of its floors is the base
    times a draw from [1 - floor_noise, 1 + floor_noise]; each site scales its
    group's floors by its own draw from [1 - site_jitter, 1 + site_jitter]. Every
    belief's particles are drawn the same way from a base, so its floors are
    correlated and a probe of one floor informs them all. With "per_floor", each
    group draws each floor's level from the prior's levels, and each site scales
    them by its own draw from the prior's jitter.

    With friction_groups "same", a site's friction group is its group;
    with "crossed", the sites of each group take the friction groups in turn
    (0, 1, 2, 0, ...), so sites on one routine can differ in friction.
    The robot's belief about each site starts from the survey: "prior", particles
    from the prior, or "levels", which knows the site's group's base (or levels)
    but not the noise and jitter.
    Each group's routine has routine_size tasks drawn as for the
    single-site friction domain, each from anywhere in the maze with probability
    region_overlap and otherwise from the group's region: the floors of one
    color, group g's being color g mod the number of colors. Each new day
    revises each group's routine with probability revision_rate, in the same
    region with revision_region "same", or in another region drawn uniformly with
    "other", where what the robot learned for the old one barely helps. With
    layout_holes or layout_obstacles, each routine, and each revision of it,
    also draws its own layout of that many holes and single-cell obstacles
    (layout.draw_layout()), which all its tasks share, so a revision brings a
    new layout. The robot works
    at one site each day, drawn with its share of a random schedule.
    """
    rng = np.random.default_rng(seed)
    base = make_domain(
        seed,
        sim=sim,
        num_holes=num_holes,
        goal_radius=goal_radius,
        horizon=horizon,
        num_particles=num_particles,
        task_distances=(task_distances[0], task_distances[1]),
        num_floors=num_floors,
        prior=prior,
        maze=maze,
    )
    group_of = [g for g, size in enumerate(group_sizes) for _ in range(size)]
    assert friction_groups in ("same", "crossed"), f"Unknown {friction_groups!r}"
    friction_of = (
        group_of
        if friction_groups == "same"
        else [i % len(group_sizes) for size in group_sizes for i in range(size)]
    )
    assert survey in ("prior", "levels"), f"Unknown survey {survey!r}"
    if group_friction == "base":
        truths, surveys = _draw_base_frictions(
            rng,
            len(group_sizes),
            friction_of,
            num_floors,
            num_particles,
            np.asarray(base_levels, dtype=np.float64),
            (floor_noise, site_jitter),
            survey,
        )
    else:
        assert group_friction == "per_floor", f"Unknown {group_friction!r}"
        truths, surveys = _draw_floor_frictions(
            rng,
            len(group_sizes),
            friction_of,
            dataclasses.replace(prior, num_surfaces=num_floors),
            num_particles,
            survey,
        )

    assert 0.0 <= region_overlap <= 1.0, "Need a region overlap in [0, 1]"
    friction_model = base.env_model
    assert isinstance(friction_model, FrictionModel)
    maze_world = make_world(num_holes, goal_radius, num_floors, maze)
    colors = np.asarray(maze_world.puck_map.surface_colors)
    regions = [
        FrictionTaskDistribution(
            friction_model, *task_distances, floors=np.flatnonzero(colors == color)
        )
        for color in range(int(colors.max()) + 1)
    ]

    assert revision_region in ("same", "other"), f"Unknown {revision_region!r}"
    region_of = [g % len(regions) for g in range(len(group_sizes))]

    def draw_routine(
        group: int, routine_rng: np.random.Generator
    ) -> list[FrictionTask]:
        region = regions[region_of[group]]
        if layout_holes or layout_obstacles:
            layout = draw_layout(
                maze_world, layout_holes, layout_obstacles, routine_rng
            )
            floors = np.flatnonzero(colors == region_of[group] % len(regions))
            region = FrictionTaskDistribution(
                friction_model, *task_distances, floors=floors, layout=layout
            )
            return [region.sample(routine_rng) for _ in range(routine_size)]
        return [
            (
                base.task_distribution
                if routine_rng.random() < region_overlap
                else region
            ).sample(routine_rng)
            for _ in range(routine_size)
        ]

    def revise_routine(
        group: int, routine_rng: np.random.Generator
    ) -> list[FrictionTask]:
        if revision_region == "other":
            others = [r for r in range(len(regions)) if r != region_of[group]]
            region_of[group] = int(routine_rng.choice(others))
        return draw_routine(group, routine_rng)

    routines = [draw_routine(g, rng) for g in range(len(group_sizes))]
    schedule = rng.dirichlet(np.ones(len(group_of)))
    model = MultiSiteModel(base.env_model)
    return Domain(
        env_model=model,
        task_distribution=SiteRoutineTaskDistribution(
            routines, group_of, schedule, revision_rate, revise_routine, rng
        ),
        initial_belief=SiteBelief(surveys),
        real_env=SimulatedEnvironment(model, truths, rng),
    )


def count_route_floors(
    paths: PuckPushingModel, tasks: Sequence[FrictionTask], num_floors: int
) -> NDArray[np.float64]:
    """How many of the tasks' shortest pushable paths cross each floor."""
    counts = np.zeros(num_floors)
    for task in tasks:
        for floor in paths.path_floors(task.initial_state.puck, task.goal):
            counts[floor] += 1
    return counts


def make_route_usage(
    world: PuckWorld, routines: SiteRoutineTaskDistribution
) -> Callable[[int], NDArray[np.float64]]:
    """A function from a site to the share of its routine's shortest paths that
    cross each floor, computed once per routine."""
    paths = PuckPushingModel(world)
    num_floors = world.puck_map.num_surfaces
    shares: dict[int, NDArray[np.float64]] = {}

    def get_usage(site: int) -> NDArray[np.float64]:
        routine = routines.get_routine_id(site)
        if routine not in shares:
            tasks = [task.inner for task in routines.get_routine(site)]
            counts = count_route_floors(paths, tasks, num_floors)
            shares[routine] = counts / max(counts.sum(), 1.0)
        return shares[routine]

    return get_usage


def _draw_base_frictions(
    rng: np.random.Generator,
    num_groups: int,
    friction_of: Sequence[int],
    num_floors: int,
    num_particles: int,
    base_levels: NDArray[np.float64],
    noise: tuple[float, float],
    survey: str,
) -> tuple[NDArray[np.float64], list[ParticleBelief]]:
    """Each site's true frictions and initial belief, from one base per friction
    group (see make_site_domain)."""
    assert num_groups <= len(base_levels), "Need a base level per friction group"
    floor_noise, site_jitter = noise
    bases = rng.choice(base_levels, size=num_groups, replace=False)
    group_floors = bases[:, None] * rng.uniform(
        1.0 - floor_noise, 1.0 + floor_noise, (num_groups, num_floors)
    )
    truths = np.stack(
        [
            group_floors[g] * rng.uniform(1 - site_jitter, 1 + site_jitter, num_floors)
            for g in friction_of
        ]
    )
    spread = (1.0 - floor_noise) * (1.0 - site_jitter), (1 + floor_noise) * (
        1 + site_jitter
    )
    low = np.full(num_floors, base_levels.min() * spread[0])
    high = np.full(num_floors, base_levels.max() * spread[1])
    surveys = []
    for g in friction_of:
        particle_bases = (
            rng.choice(base_levels, size=num_particles)
            if survey == "prior"
            else np.full(num_particles, bases[g])
        )
        particles = (
            particle_bases[:, None]
            * rng.uniform(1 - floor_noise, 1 + floor_noise, (num_particles, num_floors))
            * rng.uniform(1 - site_jitter, 1 + site_jitter, (num_particles, num_floors))
        )
        surveys.append(
            ParticleBelief(
                particles,
                low,
                high,
                seed=int(rng.integers(2**31)),
                independent=False,
            )
        )
    return truths, surveys


def _draw_floor_frictions(
    rng: np.random.Generator,
    num_groups: int,
    friction_of: Sequence[int],
    prior: FrictionPrior,
    num_particles: int,
    survey: str,
) -> tuple[NDArray[np.float64], list[ParticleBelief]]:
    """Each site's true frictions and initial belief, from a level per floor and
    friction group (see make_site_domain)."""
    num_floors = prior.num_surfaces
    levels = np.asarray(prior.levels, dtype=np.float64)
    low, high = prior.bounds
    jitter = (1.0 - prior.jitter, 1.0 + prior.jitter)
    group_levels = levels[rng.integers(len(levels), size=(num_groups, num_floors))]
    truths = np.stack(
        [group_levels[g] * rng.uniform(*jitter, num_floors) for g in friction_of]
    )
    surveys = [
        (
            make_prior_belief(prior, num_particles, rng)
            if survey == "prior"
            else ParticleBelief(
                group_levels[g] * rng.uniform(*jitter, (num_particles, num_floors)),
                low,
                high,
                seed=int(rng.integers(2**31)),
            )
        )
        for g in friction_of
    ]
    return truths, surveys
