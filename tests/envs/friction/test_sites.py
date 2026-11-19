"""Tests for envs/friction/sites.py."""

import numpy as np

from able.beliefs.particles import ParticleBelief
from able.beliefs.sites import SiteBelief
from able.envs.friction.sites import make_site_domain
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR
from able.envs.multi_site import MultiSiteModel, SiteRoutineTaskDistribution


def test_make_site_domain():
    """Tests that sites in a group share friction levels and a routine, that each
    survey belief has its site's levels, and that routines can be revised."""
    # pylint does not narrow a domain's fields with isinstance.
    # pylint: disable=no-member
    domain = make_site_domain(
        seed=0, group_sizes=(1, 2), routine_size=3, num_particles=20, revision_rate=1.0
    )
    assert isinstance(domain.env_model, MultiSiteModel)
    truths = domain.real_env.hidden_params  # type: ignore[attr-defined]
    assert truths.shape == (3, 8)
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
