"""Tests for info_gatherers/joint_limits.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import JointQuery, TwoLinkArmModel
from able.info_gatherers.joint_limits import WidestGapGatherer
from able.structs import Budget

# Widths are 0.9, 2.5, 1.0, and 0.5.
BELIEF = IntervalBelief([[-2.0, -1.1], [0.5, 3.0], [-1.5, -0.5], [0.5, 1.0]])
TRUE_LIMITS = np.array([-1.5, 2.0, -1.2, 0.8])


def _env(true_limits: np.ndarray) -> SimulatedEnvironment:
    return SimulatedEnvironment(
        TwoLinkArmModel(), true_limits, np.random.default_rng(0)
    )


def test_queries_split_the_widest_gap():
    """Tests the order of the queries and that each response comes from the
    environment."""
    gatherer = WidestGapGatherer()
    env = _env(TRUE_LIMITS)
    data = gatherer.gather(
        BELIEF, env, Budget(max_iterations=4), np.random.default_rng(0)
    )
    assert len(data) == 4
    queries = [pair.query for pair in data]
    # The first query bisects the upper limit of joint 0, and the next two bisect
    # the halves, which are wider than any other interval.
    assert queries[0] == JointQuery(joint=0, angle=1.75)
    assert {(q.joint, q.angle) for q in queries[1:3]} == {(0, 1.125), (0, 2.375)}
    assert queries[3] == JointQuery(joint=1, angle=-1.0)
    for pair in data:
        assert pair.response == env.ask(pair.query)
    assert [pair.response for pair in data] == [True, True, False, True]


def test_queries_do_not_depend_on_responses():
    """Tests that the same belief leads to the same queries in any environment."""
    gatherer = WidestGapGatherer()
    query_sequences = []
    for seed, true_limits in enumerate([TRUE_LIMITS, np.array([-2.0, 0.6, -1.0, 1.0])]):
        data = gatherer.gather(
            BELIEF,
            _env(true_limits),
            Budget(max_iterations=6),
            np.random.default_rng(seed),
        )
        query_sequences.append([pair.query for pair in data])
    assert query_sequences[0] == query_sequences[1]


def test_known_limits_stop_gathering():
    """Tests that nothing is asked once every limit is known."""
    gatherer = WidestGapGatherer()
    known = IntervalBelief(np.stack([TRUE_LIMITS, TRUE_LIMITS], axis=1))
    data = gatherer.gather(
        known, _env(TRUE_LIMITS), Budget(max_iterations=5), np.random.default_rng(0)
    )
    assert not data
    with pytest.raises(AssertionError):
        gatherer.gather(BELIEF, _env(TRUE_LIMITS), Budget(), np.random.default_rng(0))
