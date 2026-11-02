"""Tests for policies/base.py."""

import pytest

from able.policies.base import OpenLoopPolicy
from able.structs import PlanningFailure


def test_open_loop_policy():
    """Tests for OpenLoopPolicy."""
    policy = OpenLoopPolicy([1, 2, 3], planning_stats={"num_samples": 10.0})
    assert [policy.get_action(state) for state in ("a", "b", "c")] == [1, 2, 3]
    with pytest.raises(PlanningFailure):
        policy.get_action("d")

    stats = policy.get_planning_stats()
    assert stats == {"num_samples": 10.0}
    stats["num_samples"] = 0.0
    assert policy.get_planning_stats() == {"num_samples": 10.0}

    assert not OpenLoopPolicy([1]).get_planning_stats()
