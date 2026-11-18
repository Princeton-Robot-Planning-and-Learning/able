"""Tests for the arm's bisecting gatherer."""

import numpy as np

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, JointQuery, make_domain
from able.info_gatherers.bisect_limits import BisectLimitGatherer
from able.structs import Budget


def test_bisect_limits():
    """Tests that bisection asks at the widest interval's midpoint, offers every
    fraction of unknown limits, and stops once all are known."""
    gatherer = BisectLimitGatherer(JointLimitBeliefUpdater())
    belief = IntervalBelief([[-3.0, -0.5], [0.5, 1.0], [-1.0, -1.0], [0.5, 2.5]])
    assert gatherer.choose(belief) == JointQuery(0, -1.75)
    candidates = gatherer.get_candidates(belief)
    assert len(candidates) == 9
    assert JointQuery(1, 1.0) in candidates
    known = IntervalBelief([[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]])
    assert gatherer.choose(known) is None

    domain = make_domain(0)
    rng = np.random.default_rng(0)
    data = gatherer.gather(
        domain.initial_belief, domain.real_env, Budget(max_iterations=3), rng
    )
    assert len(data) == 3
    belief = JointLimitBeliefUpdater().update_from_queries(domain.initial_belief, data)
    assert isinstance(belief, IntervalBelief)
    assert (
        belief.get_total_width() < IntervalBelief(JOINT_LIMIT_PRIOR).get_total_width()
    )
