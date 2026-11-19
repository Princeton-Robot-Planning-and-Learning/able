"""Tests for beliefs/sites.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.beliefs.sites import SiteBelief


def test_site_belief():
    """Tests that a sample stacks each site's sample, and that replacing a site's
    belief or the focus leaves the rest alone."""
    first = IntervalBelief(np.array([[0.0, 0.0], [1.0, 1.0]]))
    second = IntervalBelief(np.array([[2.0, 2.0], [3.0, 3.0]]))
    belief = SiteBelief([first, second])
    assert belief.num_sites == 2 and belief.focus is None
    np.testing.assert_array_equal(
        belief.sample(np.random.default_rng(0)), [[0.0, 1.0], [2.0, 3.0]]
    )
    focused = belief.focused(1)
    assert focused.focus == 1 and focused.beliefs == belief.beliefs
    replaced = focused.with_site(0, second)
    assert replaced.focus == 1 and replaced.beliefs == (second, second)
    with pytest.raises(AssertionError):
        belief.focused(2)
