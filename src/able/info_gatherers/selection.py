"""Gatherers that pick the first query of each gathering by a rule of their own
and ask any later ones as a base gatherer does."""

import logging
import time
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.envs.base import EnvironmentModel, TaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.info_gatherers.base import SequentialGatherer
from able.learners.belief_sources import (
    BeliefSource,
    Dream,
    ImaginedNextBeliefSource,
    PlannedSource,
)
from able.planners.base import Planner
from able.structs import Budget, PlanningFailure, QueryResponse

BeliefFeatures = Callable[[Belief[Any]], NDArray[np.float64]]


class SelectingGatherer(SequentialGatherer[Any, Any, Any]):
    """Asks a committed first query if there is one, else the base gatherer's,
    then follows the base gatherer.

    Copies in workers adopt the commitment through set_commitment().
    """

    def __init__(self, base: SequentialGatherer[Any, Any, Any]) -> None:
        super().__init__(base.belief_updater)
        self._base = base
        self._commitment: Any = None

    def choose(self, belief: Belief[Any], rng: np.random.Generator) -> Any:
        return self._base.choose(belief, rng)

    def get_candidates(
        self, belief: Belief[Any], rng: np.random.Generator
    ) -> list[Any]:
        """The base gatherer's choice first, then its other candidates."""
        first = self._base.choose(belief, rng)
        candidates = [] if first is None else [first]
        seen = {repr(q) for q in candidates}
        for query in self._base.get_candidates(belief, rng):
            if repr(query) not in seen:
                seen.add(repr(query))
                candidates.append(query)
        return candidates

    def choose_first(self, belief: Belief[Any], rng: np.random.Generator) -> Any:
        if self._commitment is not None:
            return self._commitment
        return self._base.choose(belief, rng)

    def get_commitment(self) -> Any:
        return self._commitment

    def set_commitment(self, commitment: Any) -> None:
        self._commitment = commitment


class FixedFirstQueryGatherer(SelectingGatherer):
    """Always asks the given query first."""

    def __init__(self, base: SequentialGatherer[Any, Any, Any], query: Any) -> None:
        super().__init__(base)
        self._commitment = query


class RandomQueryGatherer(SelectingGatherer):
    """Asks a uniformly random candidate first."""

    def choose_first(self, belief: Belief[Any], rng: np.random.Generator) -> Any:
        candidates = self._base.get_candidates(belief, rng)
        if not candidates:
            return None
        return candidates[int(rng.integers(len(candidates)))]


class PlanningAwareGatherer(SelectingGatherer):
    """Before each sleep, commits to the first query whose outcomes the robot's
    planner will be best prepared for.

    For each candidate, num_eval_dreams dreams that ask it first sample the
    beliefs the robot may be evaluated under. A dreamed belief is covered if its
    features are within tolerance of a belief the learner has trained under or,
    if sleep_aware, of one of the beliefs the learner's source would train under
    tonight with dreams that ask the candidate. The candidate with the fewest
    uncovered dreams wins; ties go to the base gatherer's choice, then to the
    earliest candidate.
    """

    def __init__(
        self,
        base: SequentialGatherer[Any, Any, Any],
        dreamer: BeliefSource[Any, Any],
        get_features: BeliefFeatures,
        tolerance: float = 0.0,
        num_eval_dreams: int = 32,
        sleep_aware: bool = True,
        num_training_beliefs: int = 64,
    ) -> None:
        super().__init__(base)
        self._dreamer = dreamer
        self._get_features = get_features
        self._tolerance = tolerance
        self._num_eval_dreams = num_eval_dreams
        self._sleep_aware = sleep_aware
        self._num_training_beliefs = num_training_beliefs
        self.scores: list[float] = []

    def prepare(
        self,
        belief: Belief[Any],
        params: Any,
        learner: Any,
        budget: Budget,
        rng: np.random.Generator,
    ) -> None:
        candidates = self.get_candidates(belief, rng)
        self._commitment = candidates[0] if candidates else None
        self.scores = []
        source = getattr(learner, "belief_source", None)
        if len(candidates) <= 1 or source is None:
            return
        num_training = budget.max_iterations or self._num_training_beliefs
        trained = [self._get_features(b) for b in learner.training_beliefs]
        for query in candidates:
            fixed = FixedFirstQueryGatherer(self._base, query)
            covered = list(trained)
            if self._sleep_aware:
                tonight = source.with_info_gatherer(fixed)
                covered += [
                    self._get_features(
                        tonight.get_training_belief(i, belief, params, rng)
                    )
                    for i in range(1, num_training + 1)
                ]
            dreamer = self._dreamer.with_info_gatherer(fixed)
            dreams = [
                self._get_features(dreamer.get_training_belief(i, belief, params, rng))
                for i in range(1, self._num_eval_dreams + 1)
            ]
            self.scores.append(self._get_uncovered(dreams, covered))
        self._commitment = candidates[int(np.argmin(self.scores))]

    def _get_uncovered(
        self,
        dreams: Sequence[NDArray[np.float64]],
        covered: Sequence[NDArray[np.float64]],
    ) -> float:
        if not covered:
            return 1.0
        distances = np.max(
            np.abs(np.array(dreams)[:, None] - np.array(covered)[None]), axis=2
        )
        return float(np.mean(distances.min(axis=1) > self._tolerance))


class OracleQueryGatherer(SelectingGatherer):
    """Before each sleep, commits to the first query whose cycle goes best under
    the true hidden parameters.

    For each candidate, a copy of the learner sleeps with dreams that ask it, the
    candidate is asked of an environment with the true parameters, the cycle's
    tasks are executed, and the evaluation tasks are planned and run. Every
    branch uses the same random numbers. The best branch has the highest success
    rate, then the lowest mean planning time. Later queries in a gathering are
    not simulated.
    """

    def __init__(
        self,
        base: SequentialGatherer[Any, Any, Any],
        planner: Planner[Any, Any, Any, Any, Any],
        env_model: EnvironmentModel[Any, Any, Any, Any, Any, Any],
        task_distribution: TaskDistribution[Any],
        hidden_params: Any,
        eval_tasks: Sequence[Any],
        execution_budget: Budget,
        planning_budget: Budget,
    ) -> None:
        super().__init__(base)
        self._planner = planner
        self._env_model = env_model
        self._task_distribution = task_distribution
        self._hidden_params = hidden_params
        self._eval_tasks = list(eval_tasks)
        self._num_executions = execution_budget.max_iterations or 1
        self._planning_budget = planning_budget
        self.scores: list[tuple[float, float]] = []

    def prepare(
        self,
        belief: Belief[Any],
        params: Any,
        learner: Any,
        budget: Budget,
        rng: np.random.Generator,
    ) -> None:
        candidates = self.get_candidates(belief, rng)
        self._commitment = candidates[0] if candidates else None
        self.scores = []
        if len(candidates) <= 1:
            return
        seed = int(rng.integers(2**63))
        for query in candidates:
            branch_rng = np.random.default_rng(seed)
            branch = learner
            if hasattr(learner, "with_belief_source"):
                fixed = FixedFirstQueryGatherer(self._base, query)
                branch = learner.with_belief_source(
                    learner.belief_source.with_info_gatherer(fixed)
                )
            branch_params = branch.learn(belief, params, budget, branch_rng, None)
            self.scores.append(
                self._run_branch(belief, branch_params, query, branch_rng)
            )
        best = max(
            range(len(candidates)),
            key=lambda i: (self.scores[i][0], -self.scores[i][1]),
        )
        self._commitment = candidates[best]

    def _run_branch(
        self, belief: Belief[Any], params: Any, query: Any, rng: np.random.Generator
    ) -> tuple[float, float]:
        env = SimulatedEnvironment(self._env_model, self._hidden_params, rng)
        belief = self.belief_updater.update_from_queries(
            belief, [QueryResponse(query, env.ask(query))]
        )

        def update(state: Any, action: Any, next_state: Any) -> None:
            nonlocal belief
            belief = self.belief_updater.update_from_transition(
                belief, state, action, next_state
            )

        for _ in range(self._num_executions):
            task = self._task_distribution.sample(rng)
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
                run_episode(env, task, policy, on_step=update)
            except PlanningFailure:
                continue
        successes, times = [], []
        for task in self._eval_tasks:
            start_time = time.perf_counter()
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
                times.append(time.perf_counter() - start_time)
                trajectory = run_episode(env, task, policy)
                successes.append(env.check_goal(task, trajectory.states[-1]))
            except PlanningFailure:
                times.append(time.perf_counter() - start_time)
                successes.append(False)
        return float(np.mean(successes)), float(np.mean(times))


class JointSleepGatherer(SelectingGatherer):
    """Before each sleep, commits to the first query and to the beliefs the
    learner trains under tonight, together.

    For each candidate, dreams_per_query dreams ask it first. Their beliefs are
    clustered by features within tolerance, starting from the beliefs the learner
    has trained under. A dream's saving is its planning cost above cost_floor.
    The first half of a candidate's dreams plans tonight's training: untrained
    clusters ranked by probability times mean saving, one belief each, cycling if
    there are fewer than the sleep's iterations. Its score is uncertainty_weight
    times the mean drop in belief error to the dreams' hidden parameters, plus
    planning_weight times the second half's saving in planned clusters, as a share
    of the mean planning cost of all dreams. With choose_query, the best candidate
    is committed; otherwise the first query is random and all dreams plan.
    """

    def __init__(
        self,
        base: SequentialGatherer[Any, Any, Any],
        dreamer: ImaginedNextBeliefSource[Any, Any],
        source: PlannedSource[Any, Any],
        get_features: BeliefFeatures,
        get_error: Callable[[Belief[Any], Any], float],
        tolerance: float = 0.0,
        dreams_per_query: int = 64,
        uncertainty_weight: float = 1.0,
        planning_weight: float = 1.0,
        cost_floor: float = 0.0,
        choose_query: bool = True,
        num_training_beliefs: int = 64,
    ) -> None:
        super().__init__(base)
        self._dreamer = dreamer
        self._source = source
        self._get_features = get_features
        self._get_error = get_error
        self._tolerance = tolerance
        self._dreams_per_query = dreams_per_query
        self._uncertainty_weight = uncertainty_weight
        self._planning_weight = planning_weight
        self._cost_floor = cost_floor
        self._choose_query = choose_query
        self._num_training_beliefs = num_training_beliefs
        self.scores: list[float] = []
        self.uncertainty_terms: list[float] = []
        self.planning_terms: list[float] = []

    def choose_first(self, belief: Belief[Any], rng: np.random.Generator) -> Any:
        if self._choose_query:
            return super().choose_first(belief, rng)
        candidates = self._base.get_candidates(belief, rng)
        if not candidates:
            return None
        return candidates[int(rng.integers(len(candidates)))]

    def prepare(
        self,
        belief: Belief[Any],
        params: Any,
        learner: Any,
        budget: Budget,
        rng: np.random.Generator,
    ) -> None:
        candidates: list[Any] = self.get_candidates(belief, rng) or [None]
        num_training = budget.max_iterations or self._num_training_beliefs
        dreams = [self._dream(q, belief, params, rng) for q in candidates]
        centers: list[NDArray[np.float64]] = []
        for b in learner.training_beliefs:
            self._get_cluster(centers, self._get_features(b))
        num_trained = len(centers)
        clusters = [
            [self._get_cluster(centers, self._get_features(d.belief)) for d in ds]
            for ds in dreams
        ]
        mean_cost = max(
            float(np.mean([d.planning_cost for ds in dreams for d in ds])), 1e-9
        )
        if not self._choose_query:
            pooled = [d for ds in dreams for d in ds]
            plan, _ = self._plan(pooled, sum(clusters, []), num_trained, num_training)
            self._commitment = None
            self._source.set_plan(plan)
            return
        plans = []
        self.scores, self.uncertainty_terms, self.planning_terms = [], [], []
        half = len(dreams[0]) // 2
        for ds, cs in zip(dreams, clusters):
            plan, planned = self._plan(ds[:half], cs[:half], num_trained, num_training)
            saved = sum(
                self._saving(d) for d, c in zip(ds[half:], cs[half:]) if c in planned
            )
            planning = saved / max(len(ds) - half, 1) / mean_cost
            uncertainty = float(
                np.mean(
                    [
                        self._get_error(belief, d.hypothesis)
                        - self._get_error(d.belief, d.hypothesis)
                        for d in ds
                    ]
                )
            )
            plans.append(plan)
            self.uncertainty_terms.append(uncertainty)
            self.planning_terms.append(planning)
            self.scores.append(
                self._uncertainty_weight * uncertainty
                + self._planning_weight * planning
            )
        best = int(np.argmax(self.scores))
        self._commitment = candidates[best]
        self._source.set_plan(plans[best])
        logging.info(
            "Joint sleep: query %d of %d, uncertainty %.3f, planning %.3f",
            best,
            len(candidates),
            self.uncertainty_terms[best],
            self.planning_terms[best],
        )

    def _saving(self, dream: Dream[Any]) -> float:
        return max(dream.planning_cost - self._cost_floor, 0.0)

    def _dream(
        self, query: Any, belief: Belief[Any], params: Any, rng: np.random.Generator
    ) -> list[Dream[Any]]:
        gatherer = (
            self._base if query is None else FixedFirstQueryGatherer(self._base, query)
        )
        dreamer = self._dreamer.with_info_gatherer(gatherer)
        return [
            dreamer.dream(f"Query dream {i}", belief, params, rng)
            for i in range(1, self._dreams_per_query + 1)
        ]

    def _get_cluster(
        self, centers: list[NDArray[np.float64]], features: NDArray[np.float64]
    ) -> int:
        for i, center in enumerate(centers):
            if np.max(np.abs(center - features)) <= self._tolerance:
                return i
        centers.append(features)
        return len(centers) - 1

    def _plan(
        self,
        dreams: Sequence[Dream[Any]],
        clusters: Sequence[int],
        num_trained: int,
        num_training: int,
    ) -> tuple[list[Belief[Any]], set[int]]:
        """The training beliefs from the dreams and the untrained clusters they
        cover."""
        value: dict[int, float] = {}
        first: dict[int, Belief[Any]] = {}
        for d, c in zip(dreams, clusters):
            value[c] = value.get(c, 0.0) + self._saving(d) / len(dreams)
            first.setdefault(c, d.belief)
        ranked = sorted(value, key=lambda c: -value[c])
        new = [c for c in ranked if c >= num_trained][:num_training]
        plan = [first[c] for c in new]
        while plan and len(plan) < num_training:
            plan += [first[c] for c in ranked][: num_training - len(plan)]
        return plan, set(new)
