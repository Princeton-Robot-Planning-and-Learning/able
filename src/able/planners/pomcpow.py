"""POMCPOW (Sunberg and Kochenderfer, ICAPS 2018), and a POMCPOW planner for the
friction environment.

The search is domain-free for continuous actions and observations: a domain supplies
a POMDPModel, an optional ActionProposer replaces its action sampler, and an optional
leaf value replaces its value estimate.
"""

import bisect
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Generic, Hashable, Protocol, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.friction.model import NO_PUSH, FrictionModel, FrictionState, FrictionTask
from able.envs.friction.pomdp import (
    ONGOING,
    PuckObservation,
    PuckPushingModel,
    PuckState,
)
from able.envs.friction.world import SURFACE_NAMES, PuckWorld, Push
from able.models.base import ConditionalModel, Dataset
from able.models.flow_matching import FlowMatchingModel
from able.models.kernel_regression import KernelRegressor
from able.planners.base import TrainablePlanner
from able.policies.base import Policy
from able.structs import Budget, PlanningFailure, PlanningTimeout, Trajectory

S = TypeVar("S")
A = TypeVar("A", bound=Hashable)
O = TypeVar("O", bound=Hashable)


class POMDPModel(Protocol[S, A, O]):
    """What POMCPOW needs from a domain."""

    def step(self, state: S, action: A, rng: np.random.Generator) -> tuple[S, float]:
        """Sample a next state and the reward of the transition."""

    def reward(self, state: S, action: A, next_state: S) -> float:
        """Reward of `state -> next_state` under `action`; used when resampling from
        a node's belief."""

    def observe(self, next_state: S, action: A, rng: np.random.Generator) -> O:
        """Sample an observation of `next_state`."""

    def likelihood(self, observation: O, next_state: S, action: A) -> float:
        """Density (up to a constant) of `observation` given `next_state`."""

    def is_terminal(self, state: S) -> bool:
        """Whether the episode is over in `state`."""

    def propose(
        self, state: S, tried: Sequence[A], rng: np.random.Generator
    ) -> A | None:
        """A new action to try from `state`, or None if there is none."""

    def value(self, state: S) -> float:
        """Estimated value of a leaf state."""


# (sampled state, all states at the node, actions already tried, rng) -> action or None.
ActionProposer = Callable[
    [S, Sequence[S], Sequence[A], np.random.Generator], "A | None"
]
# Nonterminal state -> estimated value, in place of the model's.
LeafValue = Callable[[S], float]


@dataclass(frozen=True)
class EarlyStopConfig:
    """When a search may stop before num_sims.

    From min_sims simulations on, every check_every simulations, the search stops
    if the most visited root action is settled in any of three ways:
    - its lead in visits over the runner-up exceeds the simulations left, so no
      action could overtake it and the full search would return it too;
    - its return's lower confidence bound (mean - z standard errors) is at least
      every other visited root action's upper bound minus epsilon, so it is
      likely within epsilon of the best of them; or
    - if aspiration is set and the search has a target value for the root, that
      lower bound is at least the target minus aspiration, so the action is about
      as good as the target expects of the best one.
    An action visited once takes the standard deviation of all root returns. The
    returns' distribution drifts as the tree below grows, so the bounds are
    approximate.
    """

    min_sims: int = 100
    check_every: int = 25
    epsilon: float = 1.0
    z: float = 2.0
    aspiration: float | None = None


@dataclass(frozen=True)
class POMCPOWConfig:
    """Search parameters. early_stop, if set, lets a search stop before num_sims."""

    num_sims: int = 1000
    max_time: float | None = None  # seconds; None runs all simulations
    max_depth: int = 10
    discount: float = 1.0
    exploration: float = 20.0
    # Progressive widening (k, alpha): up to k * N^alpha children at N visits.
    action_widening: tuple[float, float] = (2.0, 0.5)
    observation_widening: tuple[float, float] = (1.0, 0.25)
    rollout_depth: int = 0  # random steps before the model's leaf value
    early_stop: EarlyStopConfig | None = None


@dataclass
class _ObservationNode(Generic[S, A]):
    visits: int = 0
    actions: dict = field(default_factory=dict)
    # States that reached this node, with cumulative weights.
    states: list = field(default_factory=list)
    cumulative: list = field(default_factory=list)
    exhausted: bool = False


@dataclass
class _ActionNode:
    visits: int = 0
    value: float = 0.0
    # Sum of squared deviations of the returns from their mean (Welford).
    squared_deviations: float = 0.0
    children: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)


@dataclass
class SearchStats:
    """What one call to `plan` did."""

    num_sims: int = 0
    seconds: float = 0.0
    root_actions: int = 0
    max_depth_reached: int = 0
    root_values: list = field(
        default_factory=list
    )  # (action, visits, value), best first


class POMCPOW(Generic[S, A, O]):
    """`plan(particles)` returns the best root action and search statistics."""

    def __init__(
        self,
        model: POMDPModel[S, A, O],
        config: POMCPOWConfig = POMCPOWConfig(),
        proposer: ActionProposer | None = None,
        rng: np.random.Generator | None = None,
        leaf_value: LeafValue | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.proposer = proposer
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.leaf_value = model.value if leaf_value is None else leaf_value
        self.root: _ObservationNode | None = None
        self._depth_reached = 0
        self._target = math.inf

    def plan(
        self, particles: Sequence[S], target: float | None = None
    ) -> tuple[A | None, SearchStats]:
        """Search from equally weighted state particles, aspiring to the target
        value for the root if the early-stop rule does; the action is None if none
        could be proposed."""
        start = time.perf_counter()
        config = self.config
        particles = list(particles)
        self.root = _ObservationNode(
            states=particles, cumulative=list(range(1, len(particles) + 1))
        )
        self._depth_reached = 0
        self._target = math.inf if target is None else target
        sims = 0
        while sims < config.num_sims:
            if config.max_time is not None and sims > 0:
                if time.perf_counter() - start > config.max_time:
                    break
            state = particles[int(self.rng.integers(len(particles)))]
            self._simulate(state, self.root, config.max_depth)
            sims += 1
            if self.is_settled(sims):
                break
        stats = SearchStats(
            num_sims=sims,
            seconds=time.perf_counter() - start,
            root_actions=len(self.root.actions),
            max_depth_reached=self._depth_reached,
            root_values=sorted(
                ((a, n.visits, n.value) for a, n in self.root.actions.items()),
                key=lambda item: (item[1], item[2]),
                reverse=True,
            ),
        )
        if not self.root.actions:
            return None, stats
        # The most visited action: values of rarely tried continuous ones are noisy.
        best = max(
            self.root.actions.items(), key=lambda kv: (kv[1].visits, kv[1].value)
        )
        return best[0], stats

    def is_settled(self, sims: int) -> bool:
        """Whether the early-stop rule, if any, ends the search after sims
        simulations."""
        rule = self.config.early_stop
        if rule is None or self.root is None or sims >= self.config.num_sims:
            return False
        if sims < rule.min_sims or sims % rule.check_every != 0:
            return False
        visited = [n for n in self.root.actions.values() if n.visits > 0]
        aspires = rule.aspiration is not None and math.isfinite(self._target)
        if len(visited) < (1 if aspires else 2):
            return False
        visited.sort(key=lambda n: (n.visits, n.value), reverse=True)
        best, others = visited[0], visited[1:]
        if others and best.visits - others[0].visits > self.config.num_sims - sims:
            return True
        total = sum(n.visits for n in visited)
        mean = sum(n.visits * n.value for n in visited) / total
        # Within-action and between-action deviations of every root return.
        pooled_ss = sum(
            n.squared_deviations + n.visits * (n.value - mean) ** 2 for n in visited
        )
        pooled_sd = math.sqrt(pooled_ss / max(total - 1, 1))

        def standard_error(node: _ActionNode) -> float:
            if node.visits < 2:
                return pooled_sd
            return math.sqrt(node.squared_deviations / (node.visits - 1) / node.visits)

        lower = best.value - rule.z * standard_error(best)
        if aspires and rule.aspiration is not None:
            if lower >= self._target - rule.aspiration:
                return True
        if not others:
            return False
        upper = max(n.value + rule.z * standard_error(n) for n in others)
        return lower >= upper - rule.epsilon

    @staticmethod
    def _widened(visits: int, size: int, widening: tuple[float, float]) -> bool:
        k, alpha = widening
        return size <= k * max(visits, 1) ** alpha

    def _propose(self, state: S, node: _ObservationNode) -> A | None:
        tried = list(node.actions)
        if self.proposer is not None:
            return self.proposer(state, node.states, tried, self.rng)
        return self.model.propose(state, tried, self.rng)

    def _select_action(self, state: S, node: _ObservationNode):
        if not node.exhausted and self._widened(
            node.visits, len(node.actions), self.config.action_widening
        ):
            action = self._propose(state, node)
            if action is None:
                node.exhausted = True
            elif action not in node.actions:
                node.actions[action] = _ActionNode()
        if not node.actions:
            return None
        log_visits = math.log(max(node.visits, 1))

        def ucb(item) -> float:
            child = item[1]
            if child.visits == 0:
                return math.inf
            bonus = math.sqrt(log_visits / child.visits)
            return child.value + self.config.exploration * bonus

        return max(node.actions.items(), key=ucb)

    def _rollout(self, state: S, depth: int) -> float:
        model, config = self.model, self.config
        total, scale = 0.0, 1.0
        steps = min(depth, config.rollout_depth)
        for _ in range(steps):
            if model.is_terminal(state):
                return total
            action = model.propose(state, [], self.rng)
            if action is None:
                break
            state, reward = model.step(state, action, self.rng)
            total += scale * reward
            scale *= config.discount
        if model.is_terminal(state):
            return total
        return total + scale * self.leaf_value(state)

    def _simulate(self, state: S, node: _ObservationNode, depth: int) -> float:
        model, config = self.model, self.config
        if model.is_terminal(state):
            return 0.0
        if depth == 0:
            return self.leaf_value(state)
        self._depth_reached = max(self._depth_reached, config.max_depth - depth + 1)
        selected = self._select_action(state, node)
        if selected is None:
            return self.leaf_value(state)
        action, child = selected
        next_state, reward = model.step(state, action, self.rng)
        observation = model.observe(next_state, action, self.rng)
        if self._widened(
            child.visits, len(child.children), config.observation_widening
        ):
            child.counts[observation] = child.counts.get(observation, 0) + 1
        else:
            # Reuse an observation, picked by how often it was generated.
            seen = list(child.counts)
            counts = np.array([child.counts[o] for o in seen], dtype=float)
            observation = seen[self.rng.choice(len(seen), p=counts / counts.sum())]
        new = observation not in child.children
        if new:
            child.children[observation] = _ObservationNode()
        below = child.children[observation]
        weight = model.likelihood(observation, next_state, action)
        below.states.append(next_state)
        below.cumulative.append(
            (below.cumulative[-1] if below.cumulative else 0.0) + weight
        )
        if new:
            total = reward + config.discount * self._rollout(next_state, depth - 1)
        else:
            # Continue from a state drawn from the node's belief.
            total_weight = below.cumulative[-1]
            if total_weight > 0:
                pick = bisect.bisect_right(
                    below.cumulative, self.rng.random() * total_weight
                )
            else:
                pick = int(self.rng.integers(len(below.states)))
            next_state = below.states[min(pick, len(below.states) - 1)]
            reward = model.reward(state, action, next_state)
            total = reward + config.discount * self._simulate(
                next_state, below, depth - 1
            )
        node.visits += 1
        child.visits += 1
        deviation = total - child.value
        child.value += deviation / child.visits
        child.squared_deviations += deviation * (total - child.value)
        return total


# The push model's input and output sizes, and the value model's input size. A
# training row's input is the push model's input followed by the value model's, and
# its output is a push followed by a value; NaN marks a missing push or value.
X_DIM, Y_DIM, VALUE_X_DIM = 13, 4, 9


@dataclass(frozen=True)
class ProposalConfig:
    """How the model proposes pushes inside POMCPOW.

    The first model_first pushes tried at each node come from the model while its
    draws for the node's puck cell last; later ones come from it with probability
    mix. action_widening, if set, replaces the search's progressive widening
    whenever the model proposes, so that a trained model's pushes get the visits
    instead of new random ones.
    """

    mix: float = 0.5  # chance a proposal comes from the model
    num_draws: int = 16  # model draws cached per puck cell
    cell: float = 0.05  # cache resolution (m)
    model_first: int = 0
    action_widening: tuple[float, float] | None = None


@dataclass(frozen=True, eq=False)
class POMCPOWParams:
    """Parameters of the friction POMCPOW planner.

    The model proposes pushes from get_push_features(); an untrained model proposes
    nothing, and POMCPOW then searches as without it. The value model, if set,
    corrects the leaf values from get_value_features(): it predicts how much the
    return from a state exceeds the path-distance value, and an untrained one
    predicts nothing. If the early-stop rule aspires (EarlyStopConfig.aspiration),
    a search aspires to the leaf value of its root where the value model's data has
    at least min_target_weight of the weight there. Training tasks planned while
    sleeping search training_num_sims simulations per push, if set, instead of
    config.num_sims, and never stop early.
    """

    config: POMCPOWConfig = POMCPOWConfig(max_depth=8)
    model: ConditionalModel = field(default_factory=FlowMatchingModel)
    proposal: ProposalConfig = ProposalConfig()
    training_num_sims: int | None = None
    value_model: KernelRegressor | None = None
    min_target_weight: float = 0.5


@dataclass(frozen=True, eq=False)
class PushValueModel(ConditionalModel):
    """The push model and the value model trained together on rows laid out as
    X_DIM, Y_DIM, and VALUE_X_DIM describe. Each model fits the rows that have its
    output; sample() takes the push model's input and proposes pushes."""

    push: ConditionalModel
    value: KernelRegressor | None = None

    def fit(self, dataset: Dataset) -> "PushValueModel":
        push, value = self.push, self.value
        rows = np.all(np.isfinite(dataset.y[:, :Y_DIM]), axis=1)
        if rows.any():
            push = push.fit(_select(dataset, rows, slice(0, X_DIM), slice(0, Y_DIM)))
        rows = np.isfinite(dataset.y[:, Y_DIM])
        if value is not None and rows.any():
            value = value.fit(
                _select(dataset, rows, slice(X_DIM, None), slice(Y_DIM, None))
            )
        return replace(self, push=push, value=value)

    def sample(
        self, x: NDArray[np.float64], num_samples: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return self.push.sample(x, num_samples, rng)


def _select(dataset: Dataset, rows: NDArray[np.bool_], x: slice, y: slice) -> Dataset:
    """The dataset's chosen rows, with the chosen input and output columns."""
    return Dataset(
        dataset.x[rows, x],
        dataset.y[rows, y],
        dataset.costs[rows],
        dataset.groups[rows],
    )


def get_push_features(
    puck: NDArray[np.float64],
    goal: tuple[float, float],
    world: PuckWorld,
    belief: ParticleBelief,
) -> NDArray[np.float64]:
    """The model's input for a push: puck, goal, the floor under the puck
    (one-hot), and the belief's mean and standard deviation."""
    floor = np.zeros(len(SURFACE_NAMES))
    surface = world.surface_at(puck)
    if surface >= 0:
        floor[surface] = 1.0
    return np.concatenate([puck, goal, floor, belief.mean, belief.std])


def get_value_features(
    puck: NDArray[np.float64] | tuple[float, float],
    pushes_left: int,
    belief: ParticleBelief,
) -> NDArray[np.float64]:
    """The value model's input for a state: puck, pushes left, and the belief's
    mean and standard deviation."""
    return np.concatenate([puck, [pushes_left], belief.mean, belief.std])


def encode_push(push: Push) -> list[float]:
    """The model's output for a push: (cos, sin) of the heading, speed, stance."""
    stance = 0.0 if push.stance is None else push.stance
    return [math.cos(push.angle), math.sin(push.angle), push.speed, stance]


def decode_push(pomdp: PuckPushingModel, row: NDArray[np.float64]) -> Push:
    """The push for a model output, rounded and limited as the POMDP's sampler
    does."""
    phys = pomdp.world.physics
    angle = round(math.atan2(row[1], row[0]), 4)
    speed = round(float(np.clip(row[2], phys.min_speed, phys.max_speed)), 4)
    if pomdp.stance_mode == "auto":
        return Push(angle, speed)
    if pomdp.stance_mode == "grid":
        stance = float(pomdp.stances[np.argmin(np.abs(pomdp.stances - row[3]))])
    else:
        stance = float(row[3])
    limit = phys.max_stance
    return Push(angle, speed, min(max(round(stance, 4), -limit), limit))


class _ModelProposer:
    """POMCPOW's action proposer: with probability mix, an untried feasible push
    from the model's draws for the node's puck cell, else the POMDP's sampler."""

    def __init__(
        self,
        pomdp: PuckPushingModel,
        model: ConditionalModel,
        config: ProposalConfig,
        goal: tuple[float, float],
        belief: ParticleBelief,
    ) -> None:
        self._pomdp = pomdp
        self._model = model
        self._config = config
        self._goal = goal
        self._belief = belief
        self._cache: dict[tuple[int, ...], list[Push]] = {}

    def draw(self, puck: NDArray[np.float64], rng: np.random.Generator) -> list[Push]:
        """The cached feasible pushes for the puck's cell, drawn on first use."""
        key = self._get_key(puck)
        if key not in self._cache:
            x = get_push_features(puck, self._goal, self._pomdp.world, self._belief)
            self.fill(puck, self._model.sample(x, self._config.num_draws, rng))
        return self._cache[key]

    def fill(self, puck: NDArray[np.float64], rows: NDArray[np.float64]) -> None:
        """Cache the feasible pushes among the model's outputs for the puck's cell."""
        pushes = (decode_push(self._pomdp, row) for row in rows)
        feasible = self._pomdp.world.push_feasible
        self._cache[self._get_key(puck)] = list(
            dict.fromkeys(p for p in pushes if feasible(puck, p.angle, p.stance))
        )

    def _get_key(self, puck: NDArray[np.float64]) -> tuple[int, ...]:
        return tuple(int(v) for v in np.round(puck / self._config.cell))

    def __call__(
        self,
        state: PuckState,
        states: Sequence[PuckState],
        tried: Sequence[Push],
        rng: np.random.Generator,
    ) -> Push | None:
        del states  # unused because proposals depend on the sampled state
        if len(tried) < self._config.model_first or rng.random() < self._config.mix:
            for push in self.draw(state.puck_array, rng):
                if push not in tried:
                    return push
        return self._pomdp.propose(state, tried, rng)


class LearnedLeafValue:
    """Leaf values under a belief: the path-distance value plus the value model's
    correction. It counts the leaves it scores and sums the weight the model's
    data had at each."""

    def __init__(
        self,
        pomdp: PuckPushingModel,
        value_model: KernelRegressor,
        belief: ParticleBelief,
    ) -> None:
        self._pomdp = pomdp
        self._value_model = value_model
        self._belief = belief
        self.num_leaves = 0
        self.data_weight = 0.0

    def __call__(self, state: PuckState) -> float:
        value, weight = self.estimate(state)
        if state.outcome == ONGOING:
            self.num_leaves += 1
            self.data_weight += weight
        return value

    def estimate(self, state: PuckState) -> tuple[float, float]:
        """The state's value, and the weight the model's data had there (0 for a
        terminal state), without counting the state as a leaf."""
        pomdp = self._pomdp
        value = pomdp.value(state)
        if state.outcome != ONGOING:
            return value, 0.0
        pushes_left = pomdp.max_pushes - state.pushes
        x = get_value_features(state.puck, pushes_left, self._belief)
        correction, weight = self._value_model.predict(x)
        return value + float(correction[0]), weight


@dataclass(frozen=True)
class PushSearch:
    """One push's search: the state and belief it planned from, the push it chose,
    and the root actions with their visits and values, most visited first."""

    state: FrictionState
    belief: ParticleBelief
    push: Push
    root_values: tuple[tuple[Push, int, float], ...]


class FrictionPOMCPOWPolicy(Policy[FrictionState, Push]):
    """Plans each push with POMCPOW, from the belief updated with the pushes so far.

    The policy keeps its own copy of the belief and updates it after every push
    with the robot's belief updater, so each search knows what earlier pushes
    showed. If the model proposes pushes, they guide the search; if that search
    finds no push, the policy searches again without the model. After the task ends
    it returns NO_PUSH without searching. All searches share the planning budget,
    where one iteration is one simulation, so a fallback costs the task's later
    pushes simulations. Running out of the budget's simulations or time is a
    timeout (PlanningTimeout), and every failure carries the statistics so far.
    """

    def __init__(
        self,
        planner: "FrictionPOMCPOWPlanner",
        task: FrictionTask,
        belief: ParticleBelief,
        params: POMCPOWParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> None:
        self._planner = planner
        self._goal = task.goal
        self._belief = belief
        self._params = params
        self._budget = budget
        self._rng = rng
        self._stats = {
            "num_sims": 0,
            "num_seeded_searches": 0,
            "num_fallbacks": 0,
            "num_targeted_searches": 0,
        }
        self._search_time = 0.0
        self._num_value_leaves = 0
        self._value_data_weight = 0.0
        self._previous: tuple[FrictionState, Push] | None = None
        self._planned: tuple[FrictionState, Push] | None = None
        self.searches: list[PushSearch] = []

    @property
    def belief(self) -> ParticleBelief:
        """The policy's belief, updated with the pushes so far."""
        return self._belief

    @property
    def params(self) -> POMCPOWParams:
        """The parameters the policy plans with."""
        return self._params

    def plan_first_push(self, state: FrictionState) -> None:
        """Search the push for the task's initial state ahead of the episode."""
        self._planned = (state, self._search(state))

    def get_action(self, state: FrictionState) -> Push:
        if state.ended:
            return NO_PUSH
        if self._previous is not None:
            previous_state, previous_push = self._previous
            belief = self._planner.belief_updater.update_from_transition(
                self._belief, previous_state, previous_push, state
            )
            assert isinstance(belief, ParticleBelief)
            self._belief = belief
        if self._planned is not None and self._planned[0] == state:
            push = self._planned[1]
        else:
            push = self._search(state)
        self._planned = None
        self._previous = (state, push)
        return push

    def get_planning_stats(self) -> dict[str, float]:
        """The statistics in the planner's description: the mean weight the value
        model's data had at the leaves it scored (0 without a trained value model),
        and the seconds spent searching, which leave out the belief updates."""
        weight = self._value_data_weight / max(self._num_value_leaves, 1)
        return {
            **self._stats,
            "num_searches": len(self.searches),
            "value_data_weight": weight,
            "search_time": self._search_time,
        }

    def _search(self, state: FrictionState) -> Push:
        proposer = self._make_proposer(state)
        if proposer is not None:
            self._stats["num_seeded_searches"] += 1
        push, root_values = self._run(state, proposer)
        if push is None and proposer is not None:
            self._stats["num_fallbacks"] += 1
            push, root_values = self._run(state, None)
        if push is None:
            raise PlanningFailure("No feasible push", self.get_planning_stats())
        self.searches.append(PushSearch(state, self._belief, push, root_values))
        return push

    def _run(
        self, state: FrictionState, proposer: ActionProposer | None
    ) -> tuple[Push | None, tuple[tuple[Push, int, float], ...]]:
        config = self._get_config()
        widening = self._params.proposal.action_widening
        if proposer is not None and widening is not None:
            # A config file gives a list.
            k, alpha = widening
            config = replace(config, action_widening=(float(k), float(alpha)))
        leaf_value = self._planner.make_leaf_value(self._belief, self._params)
        target = self._get_target(state, config, leaf_value)
        if target is not None:
            self._stats["num_targeted_searches"] += 1
        push, stats = self._planner.search(
            state, self._belief, config, self._rng, proposer, leaf_value, target
        )
        if leaf_value is not None:
            self._num_value_leaves += leaf_value.num_leaves
            self._value_data_weight += leaf_value.data_weight
        self._stats["num_sims"] += stats.num_sims
        self._search_time += stats.seconds
        return push, tuple(stats.root_values)

    def _get_target(
        self,
        state: FrictionState,
        config: POMCPOWConfig,
        leaf_value: LearnedLeafValue | None,
    ) -> float | None:
        """The value the search aspires to: the root's leaf value, if the early-stop
        rule aspires and the value model's data has enough weight at the root."""
        rule = config.early_stop
        if leaf_value is None or rule is None or rule.aspiration is None:
            return None
        pomdp = self._planner.pomdp
        root = pomdp.initial_state(
            state.puck_array, state.robot_array, tuple(self._belief.mean), state.pushes
        )
        value, weight = leaf_value.estimate(root)
        return value if weight >= self._params.min_target_weight else None

    def _make_proposer(self, state: FrictionState) -> _ModelProposer | None:
        """A proposer from the model, or None if it proposes nothing at the root."""
        params, pomdp = self._params, self._planner.pomdp
        x = get_push_features(state.puck_array, self._goal, pomdp.world, self._belief)
        rows = params.model.sample(x, params.proposal.num_draws, self._rng)
        if len(rows) == 0:
            return None
        proposer = _ModelProposer(
            pomdp, params.model, params.proposal, self._goal, self._belief
        )
        proposer.fill(state.puck_array, rows)
        return proposer

    def _get_config(self) -> POMCPOWConfig:
        """The search configuration within what is left of the budget."""
        config, budget = self._params.config, self._budget
        num_sims, max_time = config.num_sims, config.max_time
        if budget.max_iterations is not None:
            num_sims = min(num_sims, budget.max_iterations - self._stats["num_sims"])
        if budget.max_wall_time is not None:
            left = budget.max_wall_time - self._search_time
            max_time = left if max_time is None else min(max_time, left)
        if max_time is not None and max_time <= 0:
            raise PlanningTimeout(
                "The planning budget ran out of time", self.get_planning_stats()
            )
        if num_sims <= 0:
            raise PlanningTimeout(
                "The planning budget ran out of simulations",
                self.get_planning_stats(),
            )
        return replace(config, num_sims=num_sims, max_time=max_time)


class FrictionPOMCPOWPlanner(
    TrainablePlanner[
        FrictionState, Push, FrictionTask, NDArray[np.float64], POMCPOWParams
    ]
):
    """Plans friction tasks with POMCPOW, one push at a time.

    The root states put each belief particle's frictions at the robot's view of the
    puck and itself. Leaf values come from the shortest pushable path to the goal
    (PuckPushingModel.value), corrected by the value model if it is trained.
    plan() searches the first push, and the policy searches each later one after
    updating its belief. The model in the parameters proposes pushes. Planning
    fails if no push is feasible, and times out if the budget runs out. The
    statistics are the number of simulations, searches, searches the model guided,
    fallbacks without it, and searches with a target value, the mean weight of the
    value model's data at the leaves, and the time spent searching. The default
    parameters, if given, replace POMCPOWParams().
    """

    def __init__(
        self,
        model: FrictionModel,
        belief_updater: FrictionBeliefUpdater,
        stance_mode: str = "grid",
        default_params: POMCPOWParams | None = None,
    ) -> None:
        self.belief_updater = belief_updater
        self.pomdp = PuckPushingModel(
            model.world,
            rewards=model.rewards,
            max_pushes=model.horizon,
            stance_mode=stance_mode,
        )
        self._default_params = (
            POMCPOWParams() if default_params is None else default_params
        )

    def get_default_params(self) -> POMCPOWParams:
        return self._default_params

    def get_training_params(self, params: POMCPOWParams) -> POMCPOWParams:
        if params.training_num_sims is None:
            return params
        config = replace(
            params.config, num_sims=params.training_num_sims, early_stop=None
        )
        return replace(params, config=config)

    def get_model(self, params: POMCPOWParams) -> PushValueModel:
        return PushValueModel(params.model, params.value_model)

    def set_model(
        self, params: POMCPOWParams, model: ConditionalModel
    ) -> POMCPOWParams:
        assert isinstance(model, PushValueModel), "Need a push and value model"
        return replace(params, model=model.push, value_model=model.value)

    def make_leaf_value(
        self, belief: ParticleBelief, params: POMCPOWParams
    ) -> "LearnedLeafValue | None":
        """The leaf values under the belief from the trained value model, or None
        without one."""
        value_model = params.value_model
        if value_model is None or value_model.x is None:
            return None
        return LearnedLeafValue(self.pomdp, value_model, belief)

    def get_plan_examples(
        self,
        task: FrictionTask,
        belief: Belief[NDArray[np.float64]],
        policy: Policy[FrictionState, Push],
    ) -> Dataset:
        """One row per push the policy searched so far: the features under the
        belief that search planned from, the push it chose, and no value. A row's
        cost is the negative of the push's value, and its group is the search."""
        del belief  # unused because each search keeps the belief it planned from
        assert isinstance(policy, FrictionPOMCPOWPolicy), "Need a plan from plan()"
        return self._get_rows(task, policy.searches, None, True)

    def get_rollout_examples(
        self,
        task: FrictionTask,
        belief: Belief[NDArray[np.float64]],
        policy: Policy[FrictionState, Push],
        trajectory: Trajectory[FrictionState, Push],
        solved: bool,
    ) -> Dataset:
        """The plan's rows for every push of the trajectory, with the pushes only
        if it solved the task. Each row's value is how much the return from its
        state exceeded the path-distance value there."""
        del belief  # unused because each search keeps the belief it planned from
        assert isinstance(policy, FrictionPOMCPOWPolicy), "Need a plan from plan()"
        searches = policy.searches
        assert all(
            search.state == state for search, state in zip(searches, trajectory.states)
        ), "Need the searches of the trajectory's pushes"
        returns = -np.cumsum(np.asarray(trajectory.costs)[::-1])[::-1]
        return self._get_rows(task, searches, returns, solved)

    def _get_rows(
        self,
        task: FrictionTask,
        searches: Sequence[PushSearch],
        returns: NDArray[np.float64] | None,
        with_pushes: bool,
    ) -> Dataset:
        """A row per search, with its push if with_pushes and the excess of its
        return over the path-distance value if returns are given."""
        pomdp = self.pomdp
        x, y, costs = [], [], []
        for i, search in enumerate(searches):
            state = search.state
            pushes_left = pomdp.max_pushes - state.pushes
            x.append(
                np.concatenate(
                    [
                        get_push_features(
                            state.puck_array, task.goal, pomdp.world, search.belief
                        ),
                        get_value_features(state.puck, pushes_left, search.belief),
                    ]
                )
            )
            push = encode_push(search.push) if with_pushes else [math.nan] * Y_DIM
            value = math.nan
            if returns is not None:
                root = pomdp.initial_state(
                    state.puck_array, state.robot_array, (), state.pushes
                )
                value = float(returns[i]) - pomdp.value(root)
            y.append([*push, value])
            values = [v for p, _, v in search.root_values if p == search.push]
            costs.append(-values[0] if values else 0.0)
        if not x:
            return Dataset.empty(X_DIM + VALUE_X_DIM, Y_DIM + 1)
        return Dataset(
            x=np.array(x),
            y=np.array(y),
            costs=np.array(costs),
            groups=np.arange(len(x), dtype=np.int64),
        )

    def search(
        self,
        state: FrictionState,
        belief: ParticleBelief,
        config: POMCPOWConfig,
        rng: np.random.Generator,
        proposer: ActionProposer | None = None,
        leaf_value: LeafValue | None = None,
        target: float | None = None,
    ) -> tuple[Push | None, SearchStats]:
        """One POMCPOW search for the state's push under the belief, with the leaf
        values, if given, in place of the POMDP's, aspiring to the target value, if
        given."""
        pomdp = self.pomdp
        particles = [
            pomdp.initial_state(state.puck_array, state.robot_array, f, state.pushes)
            for f in belief.particles
        ]
        solver: POMCPOW[PuckState, Push, PuckObservation] = POMCPOW(
            pomdp, config, proposer=proposer, rng=rng, leaf_value=leaf_value
        )
        return solver.plan(particles, target)

    def plan(
        self,
        task: FrictionTask,
        belief: Belief[NDArray[np.float64]],
        params: POMCPOWParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> FrictionPOMCPOWPolicy:
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        policy = FrictionPOMCPOWPolicy(self, task, belief, params, budget, rng)
        policy.plan_first_push(task.initial_state)
        return policy
