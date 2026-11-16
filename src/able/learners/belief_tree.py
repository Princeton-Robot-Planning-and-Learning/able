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
    the cost of every imagined plan on (features, distance), refit every sleep;
    before any data it is 1 - exp(-delta^2 / init_radius^2).

    The first iteration of each sleep grows the tree with num_expansions dreams,
    wave_size at a time from the highest-scoring open nodes, and picks
    num_training_beliefs nodes greedily (all of them if the tree is smaller); each
    iteration returns the next pick. A
    node is terminal at max_depth, or once its belief moved less than
    stop_threshold over two consecutive cycles. Runs in one process only.
    """

    def __init__(
        self,
        dreamer: ImaginedNextBeliefSource[HiddenParamsType, PlannerParamsType],
        get_features: Callable[[Belief[HiddenParamsType]], Features],
        get_cost: Callable[[dict[str, float] | None], float],
        num_training_beliefs: int,
        num_expansions: int = 256,
        max_depth: int = 4,
        max_children: int = 4,
        stop_threshold: float = 0.5,
        wave_size: int = 8,
        init_radius: float = 1.0,
        feature_bandwidth: float = 2.0,
        distance_bandwidth: float = 0.5,
        max_distance: float = 10.0,
    ) -> None:
        assert 0 < num_training_beliefs <= num_expansions + 1, "Too few nodes to pick"
        assert max_depth >= 1 and max_children >= 1 and wave_size >= 1
        self._dreamer = dreamer
        self._get_features = get_features
        self._get_cost = get_cost
        self._num_training_beliefs = num_training_beliefs
        self._num_expansions = num_expansions
        self._max_depth = max_depth
        self._max_children = max_children
        self._stop_threshold = stop_threshold
        self._wave_size = wave_size
        self._init_radius = init_radius
        self._feature_bandwidth = feature_bandwidth
        self._distance_bandwidth = distance_bandwidth
        self._max_distance = max_distance
        self._trained: list[Features] = []
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
            self._picks = self._plan_sleep(belief, params, rng)
        # A tree smaller than the budget repeats its picks.
        return self._picks[(iteration - 1) % len(self._picks)]

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
    ) -> list[Belief[HiddenParamsType]]:
        nodes = self._grow_tree(belief, params, rng)
        picks: list[_Node] = []
        for _ in range(min(self._num_training_beliefs, len(nodes))):
            open_nodes = [n for n in nodes if n not in picks]
            scores = self._scores(open_nodes, [p.features for p in picks])
            picks.append(open_nodes[int(np.argmax(scores))])
        self._trained.extend(p.features for p in picks)
        self._refit_cost_model()
        logging.info(
            "Belief tree: %d nodes, depths %s, picked depths %s",
            len(nodes),
            dict(sorted(Counter(n.depth for n in nodes).items())),
            dict(sorted(Counter(p.depth for p in picks).items())),
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
                wave = [root] * min(budget, self._max_children)
            else:
                expandable = [
                    n
                    for n in nodes
                    if not n.terminal and len(n.children) < self._max_children
                ]
                if not expandable:
                    break
                order = np.argsort(-self._scores(expandable, []), kind="stable")
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
                node.children.append(child)
                nodes.append(child)
                self._add_cost_data(cycle.plans)
            spent += len(wave)
            _set_weights(root)
        return nodes

    def _scores(self, nodes: list[_Node], picked: list[Features]) -> NDArray:
        """Each node's score, with the picked features counted as trained."""
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
        distances = np.minimum(distances, self._max_distance)
        reduction = self._predict_cost(features, distances) - self._predict_cost(
            features, np.zeros(len(nodes))
        )
        counts = np.array(
            [1 if n.children else self._max_depth - n.depth + 1 for n in nodes]
        )
        weights = np.array([n.weight for n in nodes])
        return weights * counts * reduction

    def _predict_cost(self, features: NDArray, distances: NDArray) -> NDArray:
        if self._cost_model is None:
            return 1.0 - np.exp(-((distances / self._init_radius) ** 2))
        inputs = np.column_stack([features, distances])
        return self._cost_model.predict_batch(inputs)[:, 0]

    def _add_cost_data(
        self, plans: list[tuple[Belief[HiddenParamsType], dict[str, float] | None]]
    ) -> None:
        """Rows of (features, distance to trained beliefs, cost) for imagined
        plans, which used the model trained on the beliefs trained so far."""
        rows = []
        for belief, stats in plans:
            features = self._get_features(belief)
            distance = self._max_distance
            if self._trained:
                distance = min(
                    distance, min(_distance(features, t) for t in self._trained)
                )
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
        if len(self._cost_data) == 0:
            return
        num_features = self._cost_data.x.shape[1] - 1
        bandwidths = (self._feature_bandwidth,) * num_features + (
            self._distance_bandwidth,
        )
        self._cost_model = KernelRegressor(bandwidths=bandwidths).fit(self._cost_data)


def _set_weights(node: _Node) -> None:
    """Split each node's weight equally among its children."""
    for child in node.children:
        child.weight = node.weight / len(node.children)
        _set_weights(child)
