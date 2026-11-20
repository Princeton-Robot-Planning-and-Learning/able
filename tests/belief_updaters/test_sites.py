"""Tests for belief_updaters/sites.py."""

import numpy as np

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.belief_updaters.sites import MultiSiteUpdater
from able.beliefs.interval import IntervalBelief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import SiteState


def test_updates_only_the_sites_belief():
    """Tests that a step narrows the belief about the site where it happened."""
    wide = IntervalBelief(np.tile([[-2.0, 2.0]], (4, 1)))
    belief = SiteBelief([wide, wide], focus=0)
    updater = MultiSiteUpdater(JointLimitBeliefUpdater())
    updated = updater.update_from_transition(
        belief,
        SiteState(1, np.zeros(2)),
        np.array([0.3, 0.0]),
        SiteState(1, np.array([0.3, 0.0])),
    )
    assert updated.beliefs[0] is wide and updated.focus == 0
    assert isinstance(updated.beliefs[1], IntervalBelief)
    np.testing.assert_allclose(updated.beliefs[1].intervals[1], [0.3, 2.0])
    assert updater.update_from_queries(belief, []) is belief
