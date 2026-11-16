"""Tests for learners/belief_tree.py."""

import numpy as np

from able.approaches import get_cache_key_features
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, TwoLinkArmModel, make_domain
from able.learners.belief_tree import BeliefTreeSource, _Node, _set_weights
from able.planners.random_shooting import RandomShootingPlanner
from tests.learners.test_belief_sources import _imagined_next_source


def _source(**kwargs) -> BeliefTreeSource:
    domain = make_domain(seed=0)
    options = {"num_training_beliefs": 4, "num_expansions": 12, **kwargs}
    return BeliefTreeSource(
        _imagined_next_source(domain),
        get_cache_key_features(0.1),
        lambda stats: 1.0 if stats is None else stats["num_samples"] / 1600,
        **options,
    )


def test_weights_split_among_children():
    """Each depth's weights sum to 1 when every node is expanded."""
    root = _Node(None, np.zeros(1), None, 0)
    a, b = _Node(None, np.zeros(1), root, 1), _Node(None, np.zeros(1), root, 1)
    root.children = [a, b]
    a.children = [_Node(None, np.zeros(1), a, 2) for _ in range(3)]
    _set_weights(root)
    assert a.weight == b.weight == 0.5
    assert np.isclose(sum(c.weight for c in a.children), 0.5)


def test_sleep_picks_distinct_beliefs_in_order():
    """The first iteration grows the tree; each iteration returns the next pick,
    and later sleeps learn a cost model from imagined and training plans."""
    source = _source()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    rng = np.random.default_rng(0)
    picks = [source.get_training_belief(i, belief, params, rng) for i in (1, 2, 3, 4)]
    assert len({id(p) for p in picks}) == 4
    source.observe_training_plans(2, picks[1], [{"num_samples": 64.0}, None])
    # pylint: disable=protected-access
    num_imagined = 12
    assert len(source._cost_data) == num_imagined + 2
    # The second training plan failed, so it costs 1.
    assert source._cost_data.y[-1, 0] == 1.0
    assert source._cost_model is None
    source.get_training_belief(1, belief, params, rng)
    assert source._cost_model is not None


def test_known_belief_stops_growing():
    """A belief that cannot move ends every branch at depth 2."""
    limits = np.array([-1.2, 1.4, -2.0, 0.9])
    known = IntervalBelief(np.stack([limits, limits], axis=1))
    source = _source(num_training_beliefs=1, num_expansions=40, max_children=2)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    nodes = source._grow_tree(  # pylint: disable=protected-access
        known, params, np.random.default_rng(0)
    )
    assert max(n.depth for n in nodes) == 2
    assert len(nodes) == 1 + 2 + 4


def test_untrained_scores_prefer_new_beliefs():
    """Before any cost data, a node far from trained beliefs outscores a node at
    a trained belief."""
    source = _source()
    root = _Node(None, np.zeros(2), None, 0)
    near = _Node(None, np.zeros(2), root, 1)
    far = _Node(None, np.full(2, 5.0), root, 1)
    root.children = [near, far]
    _set_weights(root)
    source._trained = [np.zeros(2)]  # pylint: disable=protected-access
    scores = source._scores([near, far], [])  # pylint: disable=protected-access
    assert scores[0] == 0.0 and scores[1] > 0.0


def test_small_tree_repeats_its_picks():
    """A tree with fewer nodes than the sleep budget repeats its picks."""
    limits = np.array([-1.2, 1.4, -2.0, 0.9])
    known = IntervalBelief(np.stack([limits, limits], axis=1))
    source = _source(num_training_beliefs=10, num_expansions=40, max_children=2)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    rng = np.random.default_rng(0)
    picks = [source.get_training_belief(i, known, params, rng) for i in range(1, 11)]
    assert picks[7] is picks[0]


def test_progressive_widening_widens_the_root():
    """With widening, the root's child limit grows with the expansions below it,
    and every node keeps within its limit."""
    source = _source(
        num_training_beliefs=1,
        num_expansions=40,
        max_children=None,
        widening_constant=1.0,
        wave_size=1,
    )
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    nodes = source._grow_tree(  # pylint: disable=protected-access
        IntervalBelief(JOINT_LIMIT_PRIOR), params, np.random.default_rng(0)
    )
    root = nodes[0]
    assert root.visits == len(nodes) == 41
    assert len(root.children) > 4
    for node in nodes:
        assert len(node.children) <= np.ceil(node.visits**0.5)
