"""A robot that pushes pucks at several sites, each with its own floor frictions
(see envs/multi_site.py).

Every site has the same maze, by default three corridors (see layout.py). Sites
come in groups: the sites in a group share each floor's friction level and
differ only in its jitter, as for buildings fitted out with the same materials.
Each group runs a routine of pushing tasks, which every site in the group does,
in its own region of the maze: on the corridors maze, group g's tasks stay in
corridor g mod 3.
"""

import dataclasses
from typing import Any, Sequence

import numpy as np

from able.beliefs.particles import ParticleBelief
from able.beliefs.sites import SiteBelief
from able.envs.base import Domain
from able.envs.friction.layout import make_world
from able.envs.friction.model import FrictionModel, FrictionTask
from able.envs.friction.tasks import FrictionTaskDistribution, make_domain
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, FrictionPrior
from able.envs.multi_site import MultiSiteModel, SiteRoutineTaskDistribution
from able.envs.simulated import SimulatedEnvironment

FrictionSiteDomain = Domain[Any, Any, Any, Any, Any, Any]


def make_site_domain(
    seed: int,
    group_sizes: Sequence[int] = (2, 4, 6),
    routine_size: int = 8,
    revision_rate: float = 0.0,
    sim: str = "2d",
    num_holes: int = 2,
    goal_radius: float = 0.2,
    horizon: int = 15,
    num_particles: int = 500,
    task_distances: tuple[float, float] = (1.5, 3.5),
    num_floors: int = 9,
    prior: FrictionPrior = DEFAULT_FRICTION_PRIOR,
    maze: str = "corridors",
    region_overlap: float = 0.0,
) -> FrictionSiteDomain:
    """A domain with group_sizes[g] sites in group g.

    Each group draws each floor's friction level from the prior's levels, and
    each site scales its group's levels by its own draw from the prior's jitter.
    The robot starts with a survey of each site: it knows the site's levels but
    not the jitter, so its belief is particles with those levels and jitter drawn
    from the prior. Each group's routine has routine_size tasks drawn as for the
    single-site friction domain, each from anywhere in the maze with probability
    region_overlap and otherwise from the group's region: the floors of one
    color, group g's being color g mod 3. Each new day revises each group's
    routine with probability revision_rate. The schedule gives each site a
    random share of the robot's time.
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
    prior = dataclasses.replace(prior, num_surfaces=num_floors)
    levels = np.asarray(prior.levels, dtype=np.float64)
    low, high = prior.bounds
    jitter = (1.0 - prior.jitter, 1.0 + prior.jitter)
    group_levels = levels[
        rng.integers(len(levels), size=(len(group_sizes), num_floors))
    ]
    group_of = [g for g, size in enumerate(group_sizes) for _ in range(size)]
    truths = np.stack(
        [group_levels[g] * rng.uniform(*jitter, num_floors) for g in group_of]
    )
    surveys = [
        ParticleBelief(
            group_levels[g] * rng.uniform(*jitter, (num_particles, num_floors)),
            low,
            high,
            seed=int(rng.integers(2**31)),
        )
        for g in group_of
    ]

    assert 0.0 <= region_overlap <= 1.0, "Need a region overlap in [0, 1]"
    friction_model = base.env_model
    assert isinstance(friction_model, FrictionModel)
    colors = np.asarray(
        make_world(num_holes, goal_radius, num_floors, maze).puck_map.surface_colors
    )
    regions = [
        FrictionTaskDistribution(
            friction_model, *task_distances, floors=np.flatnonzero(colors == color)
        )
        for color in range(int(colors.max()) + 1)
    ]

    def draw_routine(
        group: int, routine_rng: np.random.Generator
    ) -> list[FrictionTask]:
        region = regions[group % len(regions)]
        return [
            (
                base.task_distribution
                if routine_rng.random() < region_overlap
                else region
            ).sample(routine_rng)
            for _ in range(routine_size)
        ]

    routines = [draw_routine(g, rng) for g in range(len(group_sizes))]
    schedule = rng.dirichlet(np.ones(len(group_of)))
    model = MultiSiteModel(base.env_model)
    return Domain(
        env_model=model,
        task_distribution=SiteRoutineTaskDistribution(
            routines, group_of, schedule, revision_rate, draw_routine
        ),
        initial_belief=SiteBelief(surveys),
        real_env=SimulatedEnvironment(model, truths, rng),
    )
