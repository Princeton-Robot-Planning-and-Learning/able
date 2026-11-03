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
from able.envs.friction.world import Push
from able.planners.base import Planner
from able.policies.base import Policy
from able.structs import Budget, PlanningFailure

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
class POMCPOWConfig:
    """Search parameters."""

    num_sims: int = 1000
    max_time: float | None = None  # seconds; None runs all simulations
    max_depth: int = 10
    discount: float = 1.0
    exploration: float = 20.0
    # Progressive widening (k, alpha): up to k * N^alpha children at N visits.
    action_widening: tuple[float, float] = (2.0, 0.5)
    observation_widening: tuple[float, float] = (1.0, 0.25)
    rollout_depth: int = 0  # random steps before the model's leaf value


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
        child.value += (total - child.value) / child.visits
        return total


@dataclass(frozen=True)
class POMCPOWParams:
    """Parameters of the friction POMCPOW planner."""

    config: POMCPOWConfig = POMCPOWConfig(max_depth=8)


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
    showed. After the task ends it returns NO_PUSH without searching. All searches
    share the planning budget, where one iteration is one simulation.
    """

    def __init__(
        self,
        planner: "FrictionPOMCPOWPlanner",
        belief: ParticleBelief,
        params: POMCPOWParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> None:
        self._planner = planner
        self._belief = belief
        self._params = params
        self._budget = budget
        self._rng = rng
        self._num_sims = 0
        self._search_time = 0.0
        self._previous: tuple[FrictionState, Push] | None = None
        self._planned: tuple[FrictionState, Push] | None = None
        self.searches: list[PushSearch] = []

    @property
    def belief(self) -> ParticleBelief:
        """The policy's belief, updated with the pushes so far."""
        return self._belief

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
        return {"num_sims": self._num_sims, "num_searches": len(self.searches)}

    def _search(self, state: FrictionState) -> Push:
        config = self._get_config()
        push, stats = self._planner.search(state, self._belief, config, self._rng)
        self._num_sims += stats.num_sims
        self._search_time += stats.seconds
        if push is None:
            raise PlanningFailure("No feasible push")
        self.searches.append(PushSearch(state, push, tuple(stats.root_values)))
        return push

    def _get_config(self) -> POMCPOWConfig:
        """The search configuration within what is left of the budget."""
        config, budget = self._params.config, self._budget
        num_sims, max_time = config.num_sims, config.max_time
        if budget.max_iterations is not None:
            num_sims = min(num_sims, budget.max_iterations - self._num_sims)
        if budget.max_wall_time is not None:
            left = budget.max_wall_time - self._search_time
            max_time = left if max_time is None else min(max_time, left)
        if num_sims <= 0 or (max_time is not None and max_time <= 0):
            raise PlanningFailure("The planning budget ran out")
        return replace(config, num_sims=num_sims, max_time=max_time)


class FrictionPOMCPOWPlanner(
    Planner[FrictionState, Push, FrictionTask, NDArray[np.float64], POMCPOWParams]
):
    """Plans friction tasks with POMCPOW, one push at a time.

    The root states put each belief particle's frictions at the robot's view of the
    puck and itself. Leaf values roll out greedy pushes whose speeds assume the
    belief's mean frictions. plan() searches the first push, and the policy
    searches each later one after updating its belief. Planning fails if no push is
    feasible or the budget runs out. The statistics are the number of simulations
    and searches. The default parameters, if given, replace POMCPOWParams().
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
        policy = FrictionPOMCPOWPolicy(self, belief, params, budget, rng)
        policy.plan_first_push(task.initial_state)
        return policy
