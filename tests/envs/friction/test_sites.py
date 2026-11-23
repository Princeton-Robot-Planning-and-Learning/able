"""Tests for envs/friction/sites.py."""

import numpy as np

from able.beliefs.particles import ParticleBelief
from able.beliefs.sites import SiteBelief
from able.envs.friction.pomdp import PuckPushingModel
from able.envs.friction.sites import (
    count_route_floors,
    make_route_usage,
    make_site_domain,
)
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR
from able.envs.multi_site import MultiSiteModel, SiteRoutineTaskDistribution


def test_make_site_domain():
    """Tests that sites in a group share friction levels and a routine, that each
    survey belief has its site's levels, and that routines can be revised."""
    # pylint does not narrow a domain's fields with isinstance.
    # pylint: disable=no-member
    domain = make_site_domain(
        seed=0,
        group_sizes=(1, 2),
        routine_size=3,
        num_particles=20,
        revision_rate=1.0,
        survey="levels",
        group_friction="per_floor",
    )
    assert isinstance(domain.env_model, MultiSiteModel)
    truths = domain.real_env.hidden_params  # type: ignore[attr-defined]
    assert truths.shape == (3, 12)
    levels = np.array(DEFAULT_FRICTION_PRIOR.levels)
    jitter = DEFAULT_FRICTION_PRIOR.jitter

    def get_levels(frictions: np.ndarray) -> np.ndarray:
        ratios = frictions[..., None] / levels
        return np.argmin(np.abs(np.log(ratios)), axis=-1)

    site_levels = get_levels(truths)
    np.testing.assert_array_equal(site_levels[1], site_levels[2])
    belief = domain.initial_belief
    assert isinstance(belief, SiteBelief) and belief.num_sites == 3
    for site, survey in enumerate(belief.beliefs):
        assert isinstance(survey, ParticleBelief)
        ratios = survey.particles / levels[site_levels[site]]
        assert np.all((ratios >= 1 - jitter - 1e-9) & (ratios <= 1 + jitter + 1e-9))
    routines = domain.task_distribution
    assert isinstance(routines, SiteRoutineTaskDistribution)
    assert len(routines.get_tasks()) == 9
    assert [t.inner for t in routines.get_routine(1)] == [
        t.inner for t in routines.get_routine(2)
    ]
    before = [t.inner for t in routines.get_routine(0)]
    routines.start_day(np.random.default_rng(0))
    assert [t.inner for t in routines.get_routine(0)] != before


def test_routines_stay_in_their_corridors():
    """Tests that with no region overlap each group's tasks stay on its corridor's
    floors, and that with full overlap some leave it."""
    # pylint: disable=no-member
    for overlap, inside in ((0.0, True), (1.0, False)):
        domain = make_site_domain(
            seed=0,
            group_sizes=(1, 1, 1),
            routine_size=8,
            num_particles=10,
            region_overlap=overlap,
        )
        world = domain.env_model.inner.world  # type: ignore[attr-defined]
        colors = world.puck_map.surface_colors
        routines = domain.task_distribution
        assert isinstance(routines, SiteRoutineTaskDistribution)
        in_corridor = [
            colors[world.surface_at(np.array(point))] == site
            for site in range(3)
            for task in routines.get_routine(site)
            for point in (task.inner.initial_state.puck, task.inner.goal)
        ]
        assert all(in_corridor) if inside else not all(in_corridor)


def test_revised_routines_move_to_other_corridors():
    """Tests that with revision_region "other" every revision moves a group's
    routine to a corridor other than its last one."""
    # pylint: disable=no-member
    domain = make_site_domain(
        seed=0,
        group_sizes=(1, 1),
        routine_size=4,
        num_particles=10,
        revision_rate=1.0,
        revision_region="other",
    )
    world = domain.env_model.inner.world  # type: ignore[attr-defined]
    colors = world.puck_map.surface_colors
    routines = domain.task_distribution
    assert isinstance(routines, SiteRoutineTaskDistribution)

    def get_corridors() -> list[int]:
        corridors = []
        for site in range(2):
            points = [
                point
                for task in routines.get_routine(site)
                for point in (task.inner.initial_state.puck, task.inner.goal)
            ]
            found = {colors[world.surface_at(np.array(p))] for p in points}
            assert len(found) == 1
            corridors.append(found.pop())
        return corridors

    last = get_corridors()
    assert last == [0, 1]
    rng = np.random.default_rng(0)
    for _ in range(4):
        routines.start_day(rng)
        corridors = get_corridors()
        assert all(c != b for c, b in zip(corridors, last))
        last = corridors


def test_prior_survey_and_route_floors():
    """Tests that the prior survey starts every site from the prior, that the robot
    works at one site a day, and that a corridor's routine crosses only its
    floors."""
    # pylint: disable=no-member
    domain = make_site_domain(seed=0, group_sizes=(1, 1, 1), num_particles=50)
    belief = domain.initial_belief
    assert isinstance(belief, SiteBelief)
    for survey in belief.beliefs:
        assert isinstance(survey, ParticleBelief)
        first_floor = survey.particles[:, 0]
        assert first_floor.min() < 0.05 and first_floor.max() > 0.1
    routines = domain.task_distribution
    assert isinstance(routines, SiteRoutineTaskDistribution)
    assert routines.site in (0, 1, 2)
    world = domain.env_model.inner.world  # type: ignore[attr-defined]
    paths = PuckPushingModel(world)
    colors = np.array(world.puck_map.surface_colors)
    for site in range(3):
        routine = [task.inner for task in routines.get_routine(site)]
        counts = count_route_floors(paths, routine, world.puck_map.num_surfaces)
        assert counts.sum() >= len(routine)
        assert np.all(counts[colors != site] == 0)
        usage = make_route_usage(world, routines)(site)
        np.testing.assert_allclose(usage, counts / counts.sum())


def test_crossed_friction_groups():
    """Tests that crossed friction groups give the sites of each group the
    friction groups in turn, independent of their routine."""
    # pylint: disable=no-member
    levels = np.array(DEFAULT_FRICTION_PRIOR.levels)
    domain = make_site_domain(
        seed=0,
        group_sizes=(2, 3),
        num_particles=10,
        friction_groups="crossed",
        group_friction="per_floor",
    )
    truths = domain.real_env.hidden_params  # type: ignore[attr-defined]
    ratios = truths[..., None] / levels
    site_levels = np.argmin(np.abs(np.log(ratios)), axis=-1)
    for a, b in ((0, 2), (1, 3), (0, 4)):
        np.testing.assert_array_equal(site_levels[a], site_levels[b])
    assert not np.array_equal(site_levels[0], site_levels[1])
    routines = domain.task_distribution
    assert isinstance(routines, SiteRoutineTaskDistribution)
    assert routines.get_group(0) == routines.get_group(1)


def test_base_frictions():
    """Tests that each friction group has its own base friction on every floor,
    that beliefs draw floors from a shared base, and that a probe of one floor
    informs the others."""
    # pylint: disable=no-member
    base_levels = (0.025, 0.05, 0.1, 0.2)
    domain = make_site_domain(seed=0, group_sizes=(2, 2, 2), num_particles=200)
    truths = domain.real_env.hidden_params  # type: ignore[attr-defined]
    bases = [
        min(base_levels, key=lambda b, t=t: abs(np.log(np.median(t) / b)))
        for t in truths
    ]
    assert bases[0] == bases[1] and bases[2] == bases[3] and bases[4] == bases[5]
    assert len({bases[0], bases[2], bases[4]}) == 3
    for t, b in zip(truths, bases):
        assert np.all(np.abs(t / b - 1.0) <= 1.1 * 1.05 - 1.0 + 1e-9)
    belief = domain.initial_belief
    assert isinstance(belief, SiteBelief)
    survey = belief.beliefs[0]
    assert isinstance(survey, ParticleBelief)
    ratios = survey.particles / survey.particles[:, :1]
    assert np.all((ratios > 0.75) & (ratios < 1.35))
    target = bases[0]
    narrowed = survey.resample(
        -(((survey.particles[:, 0] - target) / 0.005) ** 2), dims=[0]
    )
    assert np.all(np.abs(np.median(narrowed.particles, axis=0) / target - 1) < 0.2)
