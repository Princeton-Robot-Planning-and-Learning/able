"""POMCPOW (Sunberg and Kochenderfer, ICAPS 2018), and a POMCPOW planner for the
friction environment.

The search is domain-free for continuous actions and observations: a domain supplies
a POMDPModel, and an optional ActionProposer replaces its action sampler.
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
from able.envs.friction.pomdp import PuckObservation, PuckPushingModel, PuckState
from able.envs.friction.world import SURFACE_NAMES, PuckWorld, Push
from able.models.base import ConditionalModel, Dataset
from able.models.flow_matching import FlowMatchingModel
from able.planners.base import TrainablePlanner
from able.policies.base import Policy
from able.structs import Budget, PlanningFailure, PlanningTimeout

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


@dataclass(frozen=True)
class EarlyStopConfig:
    """When a search may stop before num_sims.

    From min_sims simulations on, every check_every simulations, the search stops
    if the most visited root action is settled in either of two ways:
    - its lead in visits over the runner-up exceeds the simulations left, so no
      action could overtake it and the full search would return it too; or
    - its return's lower confidence bound (mean - z standard errors) is at least
      every other visited root action's upper bound minus epsilon, so it is
      likely within epsilon of the best of them.
    An action visited once takes the standard deviation of all root returns. The
    returns' distribution drifts as the tree below grows, so the bounds are
    approximate.
    """

    min_sims: int = 100
    check_every: int = 25
    epsilon: float = 1.0
    z: float = 2.0


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
    ) -> None:
        self.model = model
        self.config = config
        self.proposer = proposer
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.root: _ObservationNode | None = None
        self._depth_reached = 0

    def plan(self, particles: Sequence[S]) -> tuple[A | None, SearchStats]:
        """Search from equally weighted state particles; the action is None if none
        could be proposed."""
        start = time.perf_counter()
        config = self.config
        particles = list(particles)
        self.root = _ObservationNode(
            states=particles, cumulative=list(range(1, len(particles) + 1))
        )
        self._depth_reached = 0
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
        if len(visited) < 2:
            return False
        visited.sort(key=lambda n: (n.visits, n.value), reverse=True)
        best, others = visited[0], visited[1:]
        if best.visits - others[0].visits > self.config.num_sims - sims:
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
        return total + scale * model.value(state)

    def _simulate(self, state: S, node: _ObservationNode, depth: int) -> float:
        model, config = self.model, self.config
        if model.is_terminal(state):
            return 0.0
        if depth == 0:
            return model.value(state)
        self._depth_reached = max(self._depth_reached, config.max_depth - depth + 1)
        selected = self._select_action(state, node)
        if selected is None:
            return model.value(state)
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


# The model's input and output sizes for a push.
X_DIM, Y_DIM = 13, 4


@dataclass(frozen=True)
class ProposalConfig:
    """How the model proposes pushes inside POMCPOW, and which root pushes become
    its training labels.

    The first model_first pushes tried at each node come from the model while its
    draws for the node's puck cell last; later ones come from it with probability
    mix. action_widening, if set, replaces the search's progressive widening
    whenever the model proposes, so that a trained model's pushes get the visits
    instead of new random ones.
    """

    mix: float = 0.5  # chance a proposal comes from the model
    num_draws: int = 16  # model draws cached per puck cell
    cell: float = 0.05  # cache resolution (m)
    min_visit_fraction: float = 0.2  # labels, relative to the most visited push
    model_first: int = 0
    action_widening: tuple[float, float] | None = None


@dataclass(frozen=True, eq=False)
class POMCPOWParams:
    """Parameters of the friction POMCPOW planner.

    The model proposes pushes from get_push_features(); an untrained model proposes
    nothing, and POMCPOW then searches as without it.
    """

    config: POMCPOWConfig = POMCPOWConfig(max_depth=8)
    model: ConditionalModel = field(default_factory=FlowMatchingModel)
    proposal: ProposalConfig = ProposalConfig()


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


@dataclass(frozen=True)
class PushSearch:
    """One push's search: the state it planned from, the push it chose, and the root
    actions with their visits and values, most visited first."""

    state: FrictionState
    push: Push
    root_values: tuple[tuple[Push, int, float], ...]


class FrictionPOMCPOWPolicy(Policy[FrictionState, Push]):
    """Plans each push with POMCPOW, from the belief updated with the pushes so far.

    The policy keeps its own copy of the belief and updates it after every push
    with the robot's belief updater, so each search knows what earlier pushes
    showed. If the model proposes pushes, they guide the search; if that search
    finds no push, the policy searches again without the model. After the task ends
    it returns NO_PUSH without searching. All searches share the planning budget,
    where one iteration is one simulation.
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
        self._stats = {"num_sims": 0, "num_seeded_searches": 0, "num_fallbacks": 0}
        self._search_time = 0.0
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
        return {**self._stats, "num_searches": len(self.searches)}

    def _search(self, state: FrictionState) -> Push:
        proposer = self._make_proposer(state)
        if proposer is not None:
            self._stats["num_seeded_searches"] += 1
        push, root_values = self._run(state, proposer)
        if push is None and proposer is not None:
            self._stats["num_fallbacks"] += 1
            push, root_values = self._run(state, None)
        if push is None:
            raise PlanningFailure("No feasible push")
        self.searches.append(PushSearch(state, push, root_values))
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
        push, stats = self._planner.search(
            state, self._belief, config, self._rng, proposer
        )
        self._stats["num_sims"] += stats.num_sims
        self._search_time += stats.seconds
        return push, tuple(stats.root_values)

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
            raise PlanningTimeout("The planning budget ran out of time")
        if num_sims <= 0:
            raise PlanningFailure("The planning budget ran out of simulations")
        return replace(config, num_sims=num_sims, max_time=max_time)


class FrictionPOMCPOWPlanner(
    TrainablePlanner[
        FrictionState, Push, FrictionTask, NDArray[np.float64], POMCPOWParams
    ]
):
    """Plans friction tasks with POMCPOW, one push at a time.

    The root states put each belief particle's frictions at the robot's view of the
    puck and itself. Leaf values roll out greedy pushes whose speeds assume the
    belief's mean frictions. plan() searches the first push, and the policy
    searches each later one after updating its belief. The model in the parameters
    proposes pushes. Planning fails if no push is feasible or the budget runs out.
    The statistics are the number of simulations, searches, searches the model
    guided, and fallbacks without it. The default parameters, if given, replace
    POMCPOWParams().
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

    def get_model(self, params: POMCPOWParams) -> ConditionalModel:
        return params.model

    def set_model(
        self, params: POMCPOWParams, model: ConditionalModel
    ) -> POMCPOWParams:
        return replace(params, model=model)

    def get_plan_examples(
        self,
        task: FrictionTask,
        belief: Belief[NDArray[np.float64]],
        policy: Policy[FrictionState, Push],
    ) -> Dataset:
        """Rows from each push the policy searched so far: the features under the
        belief, and each root push visited at least min_visit_fraction as often as
        the most visited one, plus the chosen push. A row's cost is the negative of
        the push's value, and its group is the search."""
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        assert isinstance(policy, FrictionPOMCPOWPolicy), "Need a plan from plan()"
        fraction = policy.params.proposal.min_visit_fraction
        x, y, costs, groups = [], [], [], []
        for index, search in enumerate(policy.searches):
            features = get_push_features(
                search.state.puck_array, task.goal, self.pomdp.world, belief
            )
            top = max(visits for _, visits, _ in search.root_values)
            for push, visits, value in search.root_values:
                if visits >= fraction * top or push == search.push:
                    x.append(features)
                    y.append(encode_push(push))
                    costs.append(-value)
                    groups.append(index)
        if not x:
            return Dataset.empty(X_DIM, Y_DIM)
        return Dataset(
            x=np.array(x),
            y=np.array(y),
            costs=np.array(costs),
            groups=np.array(groups, dtype=np.int64),
        )

    def search(
        self,
        state: FrictionState,
        belief: ParticleBelief,
        config: POMCPOWConfig,
        rng: np.random.Generator,
        proposer: ActionProposer | None = None,
    ) -> tuple[Push | None, SearchStats]:
        """One POMCPOW search for the state's push under the belief."""
        pomdp = self.pomdp
        pomdp.rollout_frictions = tuple(float(f) for f in belief.mean)
        particles = [
            pomdp.initial_state(state.puck_array, state.robot_array, f, state.pushes)
            for f in belief.particles
        ]
        solver: POMCPOW[PuckState, Push, PuckObservation] = POMCPOW(
            pomdp, config, proposer=proposer, rng=rng
        )
        return solver.plan(particles)

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
