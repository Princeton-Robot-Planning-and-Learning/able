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


def test_ablations():
    """Without leaf counts a leaf scores like an inner node, and without a learned
    cost model the source keeps the initial one."""
    source = _source(use_leaf_counts=False, learn_cost_model=False)
    root = _Node(None, np.zeros(2), None, 0)
    leaf = _Node(None, np.full(2, 5.0), root, 1)
    root.children = [leaf]
    _set_weights(root)
    # pylint: disable=protected-access
    full = _source()._scores([leaf], [])
    assert np.isclose(source._scores([leaf], [])[0] * 4, full[0])
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    rng = np.random.default_rng(0)
    for _ in range(2):
        source.get_training_belief(1, belief, params, rng)
    assert source._cost_model is None


def test_facility_gain_prefers_shared_beliefs():
    """The first pick is a belief that two likely nodes share, not a belief that
    one node holds alone, even though each node scores the same alone."""
    source = _source(match_threshold=0.5)
    root = _Node(None, np.full(2, 9.0), None, 0)
    shared = [_Node(None, np.zeros(2), root, 1) for _ in range(2)]
    alone = _Node(None, np.full(2, 5.0), root, 1)
    root.children = [*shared, alone]
    _set_weights(root)
    # pylint: disable=protected-access
    nodes = [root, *shared, alone]
    scores = source._scores(nodes, [])
    assert np.isclose(scores[1], scores[3])
    picks = source._pick_by_gain(nodes, 2, np.random.default_rng(0))
    # The lone leaf stands for 4 evaluations at weight 1/3, more than the root.
    assert picks[0] in shared and picks[1] is alone


def test_match_threshold_makes_distances_all_or_nothing():
    """Below the threshold a belief counts as trained; above it, as far away."""
    # pylint: disable=protected-access
    source = _source(match_threshold=0.5)
    assert list(source._match(np.array([0.0, 0.4, 1.0, 3.0]))) == [0, 0, 10, 10]
    assert list(_source()._match(np.array([0.4, 30.0]))) == [0.4, 10.0]


def test_cost_rows_interpolate_the_cost_model():
    """Cached cost rows give the cost model's values on the grid and between."""
    # pylint: disable=protected-access
    source = _source()
    node = _Node(None, np.zeros(2), None, 0)
    rows = source._get_cost_rows([node])
    grid = source._distance_grid
    assert np.allclose(rows[0], source._predict_cost(np.zeros((len(grid), 2)), grid))
    mid = (grid[1] + grid[2]) / 2
    expected = (rows[0, 1] + rows[0, 2]) / 2
    assert np.isclose(source._cost_at(rows, np.array([mid]))[0], expected)


def test_wave_size_caps_children_per_wave():
    """With waves of 8, the root gains at most one child per wave; one dream at a
    time lets it grow much wider."""
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    widths = {}
    for wave_size in (8, 1):
        source = _source(
            num_training_beliefs=1,
            num_expansions=64,
            max_children=None,
            widening_constant=1.0,
            widening_exponent=0.9,
            wave_size=wave_size,
        )
        nodes = source._grow_tree(  # pylint: disable=protected-access
            IntervalBelief(JOINT_LIMIT_PRIOR), params, np.random.default_rng(0)
        )
        widths[wave_size] = len(nodes[0].children)
    assert widths[1] > 2 * widths[8]


def _chain() -> list[_Node]:
    """A node at depth 1 (weight 1) with two leaves (weight 1/2 each), all with
    distinct cache keys; the root is not a candidate."""
    root = _Node(None, np.full(2, 9.0), None, 0)
    middle = _Node(None, np.zeros(2), root, 1)
    leaves = [_Node(None, np.full(2, x), middle, 2) for x in (3.0, 6.0)]
    root.children, middle.children = [middle], leaves
    _set_weights(root)
    return [middle, *leaves]


def test_later_sleeps_leave_deep_nodes_to_later_dreaming():
    """Counting ancestors as trained, a leaf (3 evaluations at weight 1/2) beats
    its parent; with later sleeps as dreaming, the parent's sleep covers both
    leaves, so the parent comes first and the leaves add nothing."""
    # pylint: disable=protected-access
    nodes = _chain()
    rng = np.random.default_rng(0)
    open_loop = _source(match_threshold=0.5)._pick_by_gain(nodes, 1, rng)
    assert open_loop[0] is nodes[1]
    source = _source(match_threshold=0.5, later_sleeps="dreaming")
    assert source._pick_by_gain(nodes, 1, rng)[0] is nodes[0]
    features = np.array([n.features for n in nodes])
    between = source._match(np.abs(features[:, None] - features[None]).max(axis=2))
    later = source._get_later_sleep_distances(nodes, between, rng)
    assert (later[1:] == 0).all()


def test_later_sleep_gain_matches_closed_form():
    """With one belief per later sleep and two equally likely leaves, each leaf is
    missed with chance 1/2, so its gain is w (1 - Q) [C(far) - C(0)] = 1/4."""
    # pylint: disable=protected-access
    nodes = _chain()
    source = _source(
        match_threshold=0.5,
        later_sleeps="dreaming",
        later_sleep_budget=1,
        num_later_sleep_draws=4000,
    )
    features = np.array([n.features for n in nodes])
    between = source._match(np.abs(features[:, None] - features[None]).max(axis=2))
    later = source._get_later_sleep_distances(nodes, between, np.random.default_rng(0))
    rows = source._get_cost_rows(nodes)
    values = np.array([n.weight for n in nodes])
    distances = np.minimum(source._get_distances_to(nodes, []), later.T).T
    gains, _ = source._get_gains(rows, values, distances, between)
    gap = rows[0, -1] - rows[0, 0]
    assert np.isclose(gains[0], gap)
    assert np.allclose(gains[1:], 0.25 * gap, atol=0.02)


def test_sleep_with_later_dreaming():
    """A whole sleep runs with later sleeps simulated as dreaming."""
    source = _source(later_sleeps="dreaming", match_threshold=0.5)
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    rng = np.random.default_rng(0)
    picks = [source.get_training_belief(i, belief, params, rng) for i in (1, 2, 3, 4)]
    assert len({id(p) for p in picks}) == 4
