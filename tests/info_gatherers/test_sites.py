"""Tests for info_gatherers/sites.py."""

import numpy as np

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import MultiSiteModel, SiteQuery
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import TwoLinkArmModel
from able.info_gatherers.bisect_limits import BisectLimitGatherer
from able.info_gatherers.sites import MultiSiteGatherer
from able.structs import Budget


def test_asks_at_the_days_site():
    """Tests that queries go to the day's site, under the belief about it, and
    that none are asked on a day with no site."""
    limits = np.array([[-2.0, 2.0, -2.0, 2.0], [-0.2, 0.2, -0.2, 0.2]])
    rng = np.random.default_rng(0)
    env = SimulatedEnvironment(MultiSiteModel(TwoLinkArmModel()), limits, rng)
    wide = IntervalBelief(np.tile([[-3.0, 3.0]], (4, 1)))
    belief = SiteBelief([wide, wide])
    sites: list[int | None] = [1]
    inner = BisectLimitGatherer(JointLimitBeliefUpdater())
    gatherer = MultiSiteGatherer(lambda site: inner, lambda _: sites[0])
    data = gatherer.gather(belief, env, Budget(max_iterations=2), rng)
    assert len(data) == 2
    for datum in data:
        assert isinstance(datum.query, SiteQuery) and datum.query.site == 1
        low, high = limits[1][2 * datum.query.inner.joint :][:2]
        assert datum.response == bool(low <= datum.query.inner.angle <= high)
    sites[0] = None
    assert not gatherer.gather(belief, env, Budget(max_iterations=2), rng)
