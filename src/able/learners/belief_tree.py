"""Belief-tree dreaming: grow a tree of dreams from the current belief, then train
under the beliefs where training is predicted to remove the most planning cost."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.learners.belief_sources import BeliefSource, ImaginedNextBeliefSource
from able.models.base import Dataset
from able.models.kernel_regression import KernelRegressor
from able.structs import HiddenParamsType, PlannerParamsType

Features = NDArray[np.float64]


@dataclass(eq=False)
class _Node:
    belief: Belief[Any]
    features: Features
    parent: "_Node | None"
    depth: int
    children: list["_Node"] = field(default_factory=list)
    terminal: bool = False
    weight: float = 1.0
    # One plus the expansions made at this node or below it.
    visits: int = 1
    # C(b, delta) on the distance grid, cached because C is fixed in a sleep.
    cost_row: NDArray[np.float64] | None = None
    # Whether the belief matches one trained in an earlier sleep.
    trained: bool = False
    # The chance that a new child is a new belief, cached until a child is added.
    novelty: float | None = None

    def ancestors(self) -> list["_Node"]:
        """Ancestors below the root: beliefs future sleeps will train near."""
        nodes, node = [], self.parent
        while node is not None and node.depth >= 1:
            nodes.append(node)
            node = node.parent
        return nodes


def _distance(a: Features, b: Features) -> float:
    return float(np.max(np.abs(a - b)))


class BeliefTreeSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains under the nodes of a tree of dreams with the highest score

        score(n) = w_n m_n [C(b_n, delta_n) - C(b_n, 0)],

    the predicted planning cost per task that training at n removes. w_n is the
    probability that the future belief passes through n, m_n the number of
    evaluations n stands for (1, or max_depth - depth + 1 at a leaf), and delta_n
    the distance to the nearest belief trained before n is evaluated (trained
    beliefs, this sleep's picks, and n's ancestors). C is a kernel regression of
    the cost of every imagined and training plan on (features, distance), refit at
    the start of every sleep on the latest max_cost_rows plans; before any data it
    is 1 - exp(-delta^2 / init_radius^2). refit_per_iteration says whether a
    training plan's model already holds this sleep's earlier picks.

    The first iteration of each sleep grows the tree with num_expansions dreams,
    wave_size at a time from the highest-scoring open nodes, and picks
    num_training_beliefs nodes greedily (all of them if the tree is smaller); each
    iteration returns the next pick. A node is terminal at max_depth, or once its
    belief moved less than stop_threshold over two consecutive cycles. A node may
    have at most max_children children or, with progressive widening, at most
    ceil(widening_constant * visits^widening_exponent), so the tree grows wider
    where it is explored more. A wave expands each node at most once, so with
    wave_size > 1 a node gains at most one child per wave. Runs in one process
    only.

    Picks maximize the facility-location gain, the cost that a pick removes summed
    over every node, sum_n w_n m_n [C(b_n, delta_n) - C(b_n, delta_n with the
    pick)], so beliefs that many likely nodes share come first. With
    match_threshold, a belief closer than it counts as the same belief (distance 0)
    and any other as max_distance away, as an exact-match plan cache sees them.
    For ablations, use_facility_gain=False picks by score(n) alone,
    use_leaf_counts=False sets every m_n to 1, and learn_cost_model=False keeps the
    initial cost model.

    later_sleeps says what the picks assume about the sleeps after this one.
    "ancestors" (the default) counts each node's ancestors as trained, as if later
    sleeps trained at exactly those beliefs. "dreaming" simulates them instead:
    the sleep at each node below the root trains under later_sleep_budget of its
    children, drawn by weight as dreaming draws next beliefs, and the gain is
    averaged over num_later_sleep_draws such draws. A pick then counts only for
    the cost that later sleeps would not remove, and each leaf counts once (its
    later evaluations follow sleeps at the leaf itself, which train near it). The
    tree still grows by score(n) unless expansion="value".

    expansion="value" (with later_sleeps="dreaming") grows the tree where a new
    child is worth the most to this sleep's picks:

        value(n) = w_n / (k_n + 1) * novelty(n) * miss(n),

    the new child's weight, times the chance it is a belief not seen yet,
    novelty(n) = (f1 + 1) / (k_n + 2) with f1 the children whose belief no other
    child and no earlier sleep shares (a Good-Turing estimate), times the chance
    that the later sleep at n misses it: 1 at the root, whose children only this
    sleep can train, and alpha / (alpha + later_sleep_budget) below it, with alpha
    the Dirichlet-process concentration for which the next of k_n + 1 draws is new
    with chance novelty(n). Two beliefs are the same when their matched distance is
    0 (with match_threshold) or below stop_threshold. Only max_children, if set,
    limits the children; progressive widening is off.

    With train_root_first, the first iteration of each sleep trains under the
    current belief, and the tree grows in the second iteration, so its dreams
    plan with that training, as the robot will when it next executes a task
    under the current belief. The tree then picks the remaining beliefs.
    """

    def __init__(
        self,
        dreamer: ImaginedNextBeliefSource[HiddenParamsType, PlannerParamsType],
        get_features: Callable[[Belief[HiddenParamsType]], Features],
        get_cost: Callable[[dict[str, float] | None], float],
        num_training_beliefs: int,
        num_expansions: int = 256,
        max_depth: int = 4,
        max_children: int | None = 4,
        widening_constant: float | None = None,
        widening_exponent: float = 0.5,
        stop_threshold: float = 0.5,
        wave_size: int = 8,
        init_radius: float = 1.0,
        feature_bandwidth: float = 2.0,
        distance_bandwidth: float = 0.5,
        max_distance: float = 10.0,
        max_cost_rows: int = 2048,
        match_threshold: float | None = None,
        use_facility_gain: bool = True,
        refit_per_iteration: bool = True,
        use_leaf_counts: bool = True,
        learn_cost_model: bool = True,
        later_sleeps: str = "ancestors",
        later_sleep_budget: int | None = None,
        num_later_sleep_draws: int = 8,
        expansion: str = "score",
        train_root_first: bool = False,
    ) -> None:
        assert 0 < num_training_beliefs <= num_expansions + 1, "Too few nodes to pick"
        assert max_depth >= 1 and wave_size >= 1
        assert later_sleeps in ("ancestors", "dreaming")
        assert later_sleeps == "ancestors" or use_facility_gain
        assert num_later_sleep_draws >= 1
        assert expansion in ("score", "value")
        assert expansion == "score" or later_sleeps == "dreaming"
        assert max_children is not None or widening_constant is not None
        assert max_children is None or max_children >= 1
        self._dreamer = dreamer
        self._get_features = get_features
        self._get_cost = get_cost
        self._num_training_beliefs = num_training_beliefs
        self._num_expansions = num_expansions
        self._max_depth = max_depth
        self._max_children = max_children
        self._widening_constant = widening_constant
        self._widening_exponent = widening_exponent
        self._stop_threshold = stop_threshold
        self._wave_size = wave_size
        self._init_radius = init_radius
        self._feature_bandwidth = feature_bandwidth
        self._distance_bandwidth = distance_bandwidth
        self._max_distance = max_distance
        self._max_cost_rows = max_cost_rows
        self._match_threshold = match_threshold
        self._use_facility_gain = use_facility_gain
        self._distance_grid = np.linspace(0.0, max_distance, 21)
        self._refit_per_iteration = refit_per_iteration
        self._use_leaf_counts = use_leaf_counts
        self._learn_cost_model = learn_cost_model
        self._later_sleeps = later_sleeps
        self._later_sleep_budget = (
            num_training_beliefs if later_sleep_budget is None else later_sleep_budget
        )
        self._num_later_sleep_draws = num_later_sleep_draws
        self._expansion = expansion
        self._train_root_first = train_root_first
        self._trained: list[Features] = []
        self._num_trained_before_sleep = 0
        self._cost_data = Dataset.empty()
        self._cost_model: KernelRegressor | None = None
        self._picks: list[Belief[HiddenParamsType]] = []

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        if iteration == 1:
            self._refit_cost_model()
            self._num_trained_before_sleep = len(self._trained)
        root_first = self._train_root_first and self._num_training_beliefs > 1
        if root_first and iteration == 1:
            self._picks = [belief]
            self._trained.append(self._get_features(belief))
        elif root_first and iteration == 2:
            num_picks = self._num_training_beliefs - 1
            picks = self._plan_sleep(belief, params, rng, num_picks, root_picked=True)
            self._picks = [self._picks[0], *picks]
        elif iteration == 1:
            num_picks = self._num_training_beliefs
            self._picks = self._plan_sleep(belief, params, rng, num_picks)
        # A tree smaller than the budget repeats its picks.
        return self._picks[(iteration - 1) % len(self._picks)]

    def observe_training_plans(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        stats: list[dict[str, float] | None],
    ) -> None:
        """Adds the training plans' costs to the cost model's data."""
        num_seen = self._num_trained_before_sleep
        if self._refit_per_iteration:
            num_seen += min(iteration - 1, len(self._picks))
        self._add_cost_data([(belief, s) for s in stats], self._trained[:num_seen])

    def draw(
        self,
        index: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        raise NotImplementedError("The tree grows from earlier dreams, in order")

    def _plan_sleep(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
        num_picks: int,
        root_picked: bool = False,
    ) -> list[Belief[HiddenParamsType]]:
        """Grows the tree and picks num_picks training beliefs. root_picked says
        whether this sleep already trained under the root's belief (logged with
        the picks)."""
        nodes = self._grow_tree(belief, params, rng)
        num_picks = min(num_picks, len(nodes))
        if self._use_facility_gain:
            picks = self._pick_by_gain(nodes, num_picks, rng)
        else:
            picks = []
            for _ in range(num_picks):
                open_nodes = [n for n in nodes if n not in picks]
                scores = self._scores(open_nodes, [p.features for p in picks])
                picks.append(open_nodes[int(np.argmax(scores))])
        self._trained.extend(p.features for p in picks)
        picked_depths = Counter(p.depth for p in picks)
        picked_depths[0] += int(root_picked)
        logging.info(
            "Belief tree: %d nodes, depths %s, picked depths %s",
            len(nodes),
            dict(sorted(Counter(n.depth for n in nodes).items())),
            dict(sorted((d, c) for d, c in picked_depths.items() if c)),
        )
        return [p.belief for p in picks]

    def _grow_tree(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> list[_Node]:
        root = _Node(belief, self._get_features(belief), None, 0)
        nodes = [root]
        spent = 0
        while spent < self._num_expansions:
            budget = min(self._wave_size, self._num_expansions - spent)
            if spent == 0:
                wave = [root] * min(budget, self._get_max_children(root))
            else:
                expandable = [
                    n
                    for n in nodes
                    if not n.terminal and len(n.children) < self._get_max_children(n)
                ]
                if not expandable:
                    break
                if self._expansion == "value":
                    priorities = self._get_expansion_values(expandable)
                else:
                    priorities = self._scores(expandable, [])
                order = np.argsort(-priorities, kind="stable")
                wave = [expandable[i] for i in order[:budget]]
            for node in wave:
                cycle = self._dreamer.imagine_cycle(
                    node.belief, params, rng, f"Tree node {len(nodes)}"
                )
                child = _Node(
                    cycle.belief,
                    self._get_features(cycle.belief),
                    node,
                    node.depth + 1,
                )
                child.terminal = child.depth >= self._max_depth or (
                    child.depth >= 2
                    and _distance(child.features, node.features) < self._stop_threshold
                    and node.parent is not None
                    and _distance(node.features, node.parent.features)
                    < self._stop_threshold
                )
                child.trained = self._is_trained(child.features)
                node.children.append(child)
                node.novelty = None
                nodes.append(child)
                ancestor: _Node | None = node
                while ancestor is not None:
                    ancestor.visits += 1
                    ancestor = ancestor.parent
                self._add_cost_data(cycle.plans, self._trained)
            spent += len(wave)
            _set_weights(root)
        return nodes

    def _get_max_children(self, node: _Node) -> int:
        if self._expansion == "value":
            return (
                self._num_expansions
                if self._max_children is None
                else (self._max_children)
            )
        limits = []
        if self._max_children is not None:
            limits.append(self._max_children)
        if self._widening_constant is not None:
            widened = self._widening_constant * node.visits**self._widening_exponent
            limits.append(max(1, int(np.ceil(widened))))
        return min(limits)

    def _same(self, distances: NDArray) -> NDArray:
        """Whether beliefs this far apart count as the same belief."""
        if self._match_threshold is not None:
            return distances < self._match_threshold
        return distances < self._stop_threshold

    def _is_trained(self, features: Features) -> bool:
        if not self._trained:
            return False
        gaps = np.abs(np.array(self._trained) - features).max(axis=1)
        return bool(self._same(gaps).any())

    def _get_novelty(self, node: _Node) -> float:
        """The Good-Turing chance that node's next child is a belief that no child
        and no earlier sleep has: (children seen once and untrained + 1) /
        (children + 2)."""
        if node.novelty is None:
            singles = 0
            if node.children:
                features = np.array([c.features for c in node.children])
                gaps = np.abs(features[:, None] - features[None]).max(axis=2)
                counts = self._same(gaps).sum(axis=1)
                trained = np.array([c.trained for c in node.children])
                singles = int(((counts == 1) & ~trained).sum())
            node.novelty = (singles + 1) / (len(node.children) + 2)
        return node.novelty

    def _get_expansion_values(self, nodes: list[_Node]) -> NDArray:
        """value(n) = w_n / (k_n + 1) * novelty(n) * miss(n): what a new child of
        n is expected to add to this sleep's picks."""
        values = []
        for node in nodes:
            novelty = self._get_novelty(node)
            num_children = len(node.children)
            miss = 1.0
            if node.depth >= 1:
                alpha = novelty * (num_children + 1) / (1.0 - novelty)
                miss = alpha / (alpha + self._later_sleep_budget)
            values.append(node.weight / (num_children + 1) * novelty * miss)
        return np.array(values)

    def _pick_by_gain(
        self, nodes: list[_Node], num_picks: int, rng: np.random.Generator
    ) -> list[_Node]:
        """Greedy picks, each with the largest facility-location gain, averaged
        over the draws of the later sleeps."""
        rows = self._get_cost_rows(nodes)
        features = np.array([n.features for n in nodes])
        between = self._match(np.abs(features[:, None] - features[None]).max(axis=2))
        if self._later_sleeps == "dreaming":
            values = np.array([n.weight for n in nodes])
            later = self._get_later_sleep_distances(nodes, between, rng)
        else:
            values = self._get_values(nodes)
            later = self._get_ancestor_distances(nodes, between)
        # Row n, draw r: n's distance to the beliefs trained before it is evaluated.
        distances = np.minimum(self._get_distances_to(nodes, self._trained), later.T).T
        self._log_coverage(nodes, later)
        available = np.ones(len(nodes), dtype=bool)
        picks = []
        for _ in range(num_picks):
            gains, after = self._get_gains(rows, values, distances, between)
            gains[~available] = -np.inf
            best = int(np.argmax(gains))
            picks.append(nodes[best])
            available[best] = False
            distances = after[:, best]
        return picks

    def _get_gains(
        self, rows: NDArray, values: NDArray, distances: NDArray, between: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Each candidate's gain, sum_n values_n (C(b_n, d_n) - C(b_n, d_n with the
        candidate)), averaged over draws, and the distances after each candidate
        (node, candidate, draw)."""
        num_nodes, num_draws = distances.shape
        costs = self._cost_at(rows, distances).mean(axis=1)
        after = np.minimum(distances[:, None, :], between[:, :, None])
        costs_after = self._cost_at(rows, after.reshape(num_nodes, -1))
        costs_after = costs_after.reshape(num_nodes, num_nodes, num_draws).mean(axis=2)
        return values @ (costs[:, None] - costs_after), after

    def _get_ancestor_distances(self, nodes: list[_Node], between: NDArray) -> NDArray:
        """Each node's distance to its ancestors below the root, as one draw."""
        index = {id(n): i for i, n in enumerate(nodes)}
        distances = np.full((len(nodes), 1), self._max_distance)
        for i, node in enumerate(nodes):
            for ancestor in node.ancestors():
                distances[i, 0] = min(distances[i, 0], between[i, index[id(ancestor)]])
        return distances

    def _get_later_sleep_distances(
        self, nodes: list[_Node], between: NDArray, rng: np.random.Generator
    ) -> NDArray:
        """Each node's distance to the beliefs later sleeps on its path train
        under, per draw. The sleep at a node below the root trains under
        later_sleep_budget of its children, drawn by weight with replacement."""
        index = {id(n): i for i, n in enumerate(nodes)}
        shape = (self._num_later_sleep_draws, self._later_sleep_budget)
        trained_by = {}
        for node in nodes:
            if node.depth >= 1 and node.children:
                children = [index[id(c)] for c in node.children]
                weights = np.array([c.weight for c in node.children])
                trained_by[id(node)] = rng.choice(
                    children, size=shape, p=weights / weights.sum()
                )
        distances = np.full((len(nodes), shape[0]), self._max_distance)
        for i, node in enumerate(nodes):
            for ancestor in node.ancestors():
                drawn = trained_by[id(ancestor)]
                distances[i] = np.minimum(distances[i], between[i, drawn].min(axis=1))
        return distances

    def _log_coverage(self, nodes: list[_Node], later: NDArray) -> None:
        """Logs the share of each depth's nodes (and draws) that later sleeps
        train at distance 0, and each depth's total weight."""
        depths = np.array([n.depth for n in nodes])
        weights = np.array([n.weight for n in nodes])
        coverage = {
            int(d): round(float((later[depths == d] == 0).mean()), 3)
            for d in np.unique(depths)
        }
        mass = {
            int(d): round(float(weights[depths == d].sum()), 3)
            for d in np.unique(depths)
        }
        logging.info(
            "Belief tree: later-sleep coverage by depth %s, weight by depth %s",
            coverage,
            mass,
        )

    def _scores(self, nodes: list[_Node], picked: list[Features]) -> NDArray:
        """Each node's own score, with the picked features counted as trained."""
        rows = self._get_cost_rows(nodes)
        distances = self._get_distances(nodes, picked)
        reduction = self._cost_at(rows, distances) - rows[:, 0]
        return self._get_values(nodes) * reduction

    def _get_values(self, nodes: list[_Node]) -> NDArray:
        """w_n m_n: the expected evaluations each node stands for."""
        counts = [
            (
                1
                if n.children or not self._use_leaf_counts
                else self._max_depth - n.depth + 1
            )
            for n in nodes
        ]
        return np.array([n.weight for n in nodes]) * np.array(counts)

    def _get_distances_to(self, nodes: list[_Node], trained: list[Features]) -> NDArray:
        """Each node's matched distance to the nearest of the trained features."""
        if not trained:
            return self._match(np.full(len(nodes), np.inf))
        features = np.array([n.features for n in nodes])
        gaps = np.abs(features[:, None] - np.array(trained)[None]).max(axis=2)
        return self._match(gaps.min(axis=1))

    def _get_distances(self, nodes: list[_Node], picked: list[Features]) -> NDArray:
        """Each node's distance to the trained beliefs, the picked ones, and its
        ancestors."""
        features = np.array([n.features for n in nodes])
        distances = np.full(len(nodes), np.inf)
        trained = self._trained + picked
        if trained:
            gaps = np.abs(features[:, None] - np.array(trained)[None]).max(axis=2)
            distances = gaps.min(axis=1)
        for i, node in enumerate(nodes):
            for ancestor in node.ancestors():
                distances[i] = min(
                    distances[i], _distance(node.features, ancestor.features)
                )
        return self._match(distances)

    def _match(self, distances: NDArray) -> NDArray:
        if self._match_threshold is None:
            return np.minimum(distances, self._max_distance)
        return np.where(distances < self._match_threshold, 0.0, self._max_distance)

    def _get_cost_rows(self, nodes: list[_Node]) -> NDArray:
        """C(b_n, d) for each node and each d on the distance grid."""
        missing = [n for n in nodes if n.cost_row is None]
        if missing:
            grid = self._distance_grid
            features = np.repeat([n.features for n in missing], len(grid), axis=0)
            distances = np.tile(grid, len(missing))
            costs = self._predict_cost(features, distances).reshape(len(missing), -1)
            for node, row in zip(missing, costs):
                node.cost_row = row
        return np.array([n.cost_row for n in nodes])

    def _cost_at(self, rows: NDArray, distances: NDArray) -> NDArray:
        """Each row's cost at its distances (shape (N,) or (N, M)), interpolated
        on the grid."""
        position = distances / self._max_distance * (len(self._distance_grid) - 1)
        low = np.minimum(np.floor(position).astype(int), rows.shape[1] - 2)
        fraction = position - low
        if distances.ndim == 1:
            low, fraction = low[:, None], fraction[:, None]
        below = np.take_along_axis(rows, low, axis=1)
        above = np.take_along_axis(rows, low + 1, axis=1)
        result = below + fraction * (above - below)
        return result[:, 0] if distances.ndim == 1 else result

    def _predict_cost(self, features: NDArray, distances: NDArray) -> NDArray:
        if self._cost_model is None:
            return 1.0 - np.exp(-((distances / self._init_radius) ** 2))
        inputs = np.column_stack([features, distances])
        return self._cost_model.predict_batch(inputs)[:, 0]

    def _add_cost_data(
        self,
        plans: list[tuple[Belief[HiddenParamsType], dict[str, float] | None]],
        trained: list[Features],
    ) -> None:
        """Rows of (features, distance to the beliefs the planning model was
        trained under, cost) for plans."""
        rows = []
        for belief, stats in plans:
            features = self._get_features(belief)
            distance = np.inf
            if trained:
                distance = min(_distance(features, t) for t in trained)
            distance = float(self._match(np.array(distance)))
            rows.append((np.append(features, distance), self._get_cost(stats)))
        if rows:
            self._cost_data = Dataset.concatenate(
                [
                    self._cost_data,
                    Dataset(
                        x=np.array([x for x, _ in rows]),
                        y=np.array([[c] for _, c in rows]),
                        costs=np.zeros(len(rows)),
                        groups=np.zeros(len(rows), dtype=np.int64),
                    ),
                ]
            )

    def _refit_cost_model(self) -> None:
        if len(self._cost_data) == 0 or not self._learn_cost_model:
            return
        num_features = self._cost_data.x.shape[1] - 1
        bandwidths = (self._feature_bandwidth,) * num_features + (
            self._distance_bandwidth,
        )
        data = self._cost_data
        if len(data) > self._max_cost_rows:
            last = slice(len(data) - self._max_cost_rows, None)
            data = Dataset(
                data.x[last], data.y[last], data.costs[last], data.groups[last]
            )
            self._cost_data = data
        self._cost_model = KernelRegressor(bandwidths=bandwidths).fit(data)


def _set_weights(node: _Node) -> None:
    """Split each node's weight equally among its children."""
    for child in node.children:
        child.weight = node.weight / len(node.children)
        _set_weights(child)
