"""Belief sources, which choose the beliefs a model learner trains under."""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Hashable

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.envs.base import EnvironmentModel, TaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.info_gatherers.base import InformationGatherer
from able.planners.base import Planner
from able.recording.base import CycleRecorder, NullRecorder
from able.structs import Budget, HiddenParamsType, PlannerParamsType, PlanningFailure


class BeliefSource(ABC, Generic[HiddenParamsType, PlannerParamsType]):
    """Chooses the belief to train under in each iteration of a sleep.

    The choice is the research variable: whether the robot anticipates the belief
    updates it will make before it is evaluated.
    """

    @abstractmethod
    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        """The belief to train under in an iteration, numbered from 1, given the
        robot's current belief and planner parameters."""

    def observe_training_time(
        self, belief: Belief[HiddenParamsType], training_time: float
    ) -> None:
        """Observe the wall time of planning for an iteration's training tasks
        under the belief this source returned, with the planner parameters from
        before the iteration's training.

        Sources that do not use training times ignore it.
        """
        del belief, training_time  # unused because most sources ignore it


class CurrentBeliefSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains under the robot's current belief, never anticipating belief
    updates."""

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        del iteration, params, rng  # unused because the belief is always the same
        return belief


class PriorSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains under the prior, whatever the robot's current belief is."""

    def __init__(self, prior: Belief[HiddenParamsType]) -> None:
        self._prior = prior

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        del iteration, belief, params, rng  # unused because training uses the prior
        return self._prior


class ImaginedNextBeliefSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains under beliefs the robot may hold when it is next evaluated.

    Each iteration is one dream:
    1. Sample hidden parameters from the current belief and treat them as the truth.
    2. In an environment simulated with those parameters, imagine the rest of the
       cycle before evaluation. Ask queries with the robot's information gatherer
       and update the belief with the responses. Then execute tasks sampled from
       the task distribution, updating the belief after every step.
    3. Return the resulting belief, which is a sample of the belief the robot will
       be evaluated under.

    Step 2 uses the robot's gatherer, belief updater, planner, and planner
    parameters, so imagined beliefs are distributed like real ones. The budgets
    should match the cycle budgets for information gathering, task execution, and
    planning, and one iteration of the execution budget is one task.
    """

    def __init__(
        self,
        env_model: EnvironmentModel[Any, Any, Any, HiddenParamsType, Any, Any],
        task_distribution: TaskDistribution[Any],
        planner: Planner[Any, Any, Any, HiddenParamsType, PlannerParamsType],
        belief_updater: BeliefUpdater[Any, Any, HiddenParamsType, Any, Any],
        info_gatherer: InformationGatherer[HiddenParamsType, Any, Any],
        gathering_budget: Budget,
        execution_budget: Budget,
        planning_budget: Budget,
        recorder: CycleRecorder[Any, Any, HiddenParamsType] | None = None,
    ) -> None:
        assert (
            execution_budget.max_iterations is not None
            or execution_budget.max_wall_time is not None
        ), "Imagined task execution needs a finite budget"
        self._env_model = env_model
        self._task_distribution = task_distribution
        self._planner = planner
        self._belief_updater = belief_updater
        self._info_gatherer = info_gatherer
        self._gathering_budget = gathering_budget
        self._execution_budget = execution_budget
        self._planning_budget = planning_budget
        self._recorder: CycleRecorder[Any, Any, HiddenParamsType] = (
            NullRecorder() if recorder is None else recorder
        )

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        label = f"Dream {iteration}"
        hypothesis = belief.sample(rng)
        imagined_env = SimulatedEnvironment(self._env_model, hypothesis, rng)
        imagined_belief = self._imagine_until_evaluation(
            label, belief, params, imagined_env, rng
        )
        self._recorder.record_belief(
            f"{label} · imagined belief", imagined_belief, hypothesis
        )
        return imagined_belief

    def _imagine_until_evaluation(
        self,
        label: str,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        env: SimulatedEnvironment[Any, Any, Any, HiddenParamsType, Any, Any],
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        """Imagine information gathering and task execution in the environment and
        return the resulting belief."""
        hypothesis = env.hidden_params
        data = self._info_gatherer.gather(belief, env, self._gathering_budget, rng)
        belief = self._belief_updater.update_from_queries(belief, data)
        self._recorder.record_queries(data, belief, hypothesis)

        def update(state: Any, action: Any, next_state: Any) -> None:
            nonlocal belief
            belief = self._belief_updater.update_from_transition(
                belief, state, action, next_state
            )
            self._recorder.record_step(next_state, belief)

        start_time = time.perf_counter()
        num_tasks = 0
        while not self._execution_budget.is_exhausted(
            time.perf_counter() - start_time, num_tasks
        ):
            num_tasks += 1
            task = self._task_distribution.sample(rng)
            self._recorder.start_episode(
                f"{label} · imagined task {num_tasks}", task, belief, hypothesis
            )
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
                trajectory = run_episode(env, task, policy, on_step=update)
            except PlanningFailure:
                self._recorder.end_episode("planning failed")
                continue
            success = env.check_goal(task, trajectory.states[-1])
            self._recorder.end_episode(
                "imagined success" if success else "imagined failure"
            )
        return belief


class BeliefCoverageSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains under the candidate belief farthest from the beliefs it has
    already chosen.

    This adapts a belief-expansion heuristic from point-based POMDP planning,
    stochastic simulation with exploratory action, which adds the reachable
    belief farthest from the current set of belief points (Pineau, Gordon, and
    Thrun, JAIR 2006). Each iteration draws num_candidates beliefs from the
    candidate source and returns the one whose nearest previously chosen belief
    is farthest. Chosen beliefs count in every later sleep, as the training data
    does.

    The distance between two beliefs is the largest absolute difference between
    their features. Beliefs that call for the same training should have the same
    features, so that a candidate like a chosen belief is at distance zero. Ties
    go to the earliest candidate, so with one candidate the source returns the
    candidate source's belief.

    The candidate source is usually an ImaginedNextBeliefSource, so the
    candidates are dreams. It sees num_candidates iterations per iteration of
    this source, numbered consecutively from 1 in each sleep.
    """

    def __init__(
        self,
        candidate_source: BeliefSource[HiddenParamsType, PlannerParamsType],
        get_features: Callable[[Belief[HiddenParamsType]], NDArray[np.float64]],
        num_candidates: int,
    ) -> None:
        assert num_candidates > 0, "Need at least one candidate"
        self._candidate_source = candidate_source
        self._get_features = get_features
        self._num_candidates = num_candidates
        self._chosen_features: list[NDArray[np.float64]] = []

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        first = (iteration - 1) * self._num_candidates + 1
        candidates = [
            self._candidate_source.get_training_belief(i, belief, params, rng)
            for i in range(first, first + self._num_candidates)
        ]
        features = np.array([self._get_features(c) for c in candidates])
        best = 0
        if self._chosen_features:
            chosen = np.array(self._chosen_features)
            # The distance from each candidate (row) to each chosen belief (column).
            distances = np.max(np.abs(features[:, None] - chosen[None]), axis=2)
            best = int(np.argmax(np.min(distances, axis=1)))
        self._chosen_features.append(features[best])
        return candidates[best]


class NextCallSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains under a belief from the region where one more update is expected
    to save the most planning time at the next planning call.

    This adapts continual computation (Horvitz, AIJ 2001), which spends idle time
    on the problem with the largest product of the probability that it arrives
    next and the reduction in response time from computing more on it. At the
    first iteration of each sleep, the source draws num_candidates beliefs from
    the candidate source and groups them into regions with get_region. A region's
    probability is the fraction of the candidates in it. Each iteration returns
    the first candidate of the region with the largest product of probability and
    estimated saving, with ties going to the more probable region. Update counts
    persist across sleeps, as the training data does.

    The saving is estimated from a profile of training times shared by all
    regions. T(n) is the mean training time observed in regions with n earlier
    updates, and the saving of an update to a region with n updates is T(n) -
    T(n + 1), or zero if that is negative. An update whose saving has not been
    measured is assumed to save as much as the last measured update. Before any
    saving has been measured, an update is assumed to remove the whole training
    time, and before any training time has been observed, every region has the
    same saving. The profile assumes the learner refits after every iteration, so
    each training plans with the updates before it.

    The candidate source is usually an ImaginedNextBeliefSource, so the regions'
    probabilities are those of the beliefs at evaluation.
    """

    def __init__(
        self,
        candidate_source: BeliefSource[HiddenParamsType, PlannerParamsType],
        get_region: Callable[[Belief[HiddenParamsType]], Hashable],
        num_candidates: int,
    ) -> None:
        assert num_candidates > 0, "Need at least one candidate"
        self._candidate_source = candidate_source
        self._get_region = get_region
        self._num_candidates = num_candidates
        self._num_updates: dict[Hashable, int] = {}
        # The training times observed in regions with each number of updates.
        self._training_times: list[list[float]] = []
        # This sleep's regions, most probable first: (region, probability, belief).
        self._regions: list[tuple[Hashable, float, Belief[HiddenParamsType]]] = []

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        if iteration == 1:
            self._regions = self._group_candidates(belief, params, rng)
        scores = [
            probability * self.estimate_saving(self._num_updates.get(region, 0))
            for region, probability, _ in self._regions
        ]
        region, _, training_belief = self._regions[int(np.argmax(scores))]
        self._num_updates[region] = self._num_updates.get(region, 0) + 1
        return training_belief

    def observe_training_time(
        self, belief: Belief[HiddenParamsType], training_time: float
    ) -> None:
        num_earlier_updates = self._num_updates.get(self._get_region(belief), 0) - 1
        assert num_earlier_updates >= 0, "Observed a belief this source did not return"
        while len(self._training_times) <= num_earlier_updates:
            self._training_times.append([])
        self._training_times[num_earlier_updates].append(training_time)

    def estimate_saving(self, num_updates: int) -> float:
        """The estimated saving in training time of one more update to a region
        with num_updates updates."""
        profile = [float(np.mean(times)) for times in self._training_times]
        if not profile:
            return 1.0
        savings = [max(0.0, t - t_next) for t, t_next in zip(profile, profile[1:])]
        if num_updates < len(savings):
            return savings[num_updates]
        return savings[-1] if savings else profile[0]

    def _group_candidates(
        self,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> list[tuple[Hashable, float, Belief[HiddenParamsType]]]:
        """Draw the candidates and return each region, its probability, and its
        first candidate, most probable first."""
        counts: dict[Hashable, int] = {}
        first_candidates: dict[Hashable, Belief[HiddenParamsType]] = {}
        for i in range(1, self._num_candidates + 1):
            candidate = self._candidate_source.get_training_belief(
                i, belief, params, rng
            )
            region = self._get_region(candidate)
            counts[region] = counts.get(region, 0) + 1
            first_candidates.setdefault(region, candidate)
        regions = [
            (region, count / self._num_candidates, first_candidates[region])
            for region, count in counts.items()
        ]
        return sorted(regions, key=lambda entry: -entry[1])
