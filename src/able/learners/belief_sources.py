"""Belief sources, which choose the beliefs a model learner trains under."""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Sequence

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

    # A learner that runs a sleep's iterations in parallel splits each one in two:
    # num_draws independent draws, which workers make at once, then choose(), which
    # this process calls once per iteration, in order. get_training_belief() must
    # return what choose() returns for its iteration's draws.

    @property
    def num_draws(self) -> int:
        """The draws each training belief is chosen from."""
        return 1

    def draw(
        self,
        index: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        """One draw, numbered from 1 in each sleep. It depends only on its
        arguments, so a sleep's draws can be made in any order or at once. By
        default the draw is the training belief."""
        return self.get_training_belief(index, belief, params, rng)

    def choose(
        self, iteration: int, draws: Sequence[Belief[HiddenParamsType]]
    ) -> Belief[HiddenParamsType]:
        """The training belief for an iteration, from its num_draws draws in
        order. By default the only draw."""
        del iteration  # unused because there is one draw
        assert len(draws) == 1, "Need exactly one draw"
        return draws[0]


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
            self.draw(i, belief, params, rng)
            for i in range(first, first + self._num_candidates)
        ]
        return self.choose(iteration, candidates)

    @property
    def num_draws(self) -> int:
        """The candidates each training belief is chosen from."""
        return self._num_candidates

    def draw(
        self,
        index: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        """One candidate from the candidate source."""
        return self._candidate_source.get_training_belief(index, belief, params, rng)

    def choose(
        self, iteration: int, draws: Sequence[Belief[HiddenParamsType]]
    ) -> Belief[HiddenParamsType]:
        """The candidate farthest from the beliefs chosen so far, which then counts
        as chosen."""
        del iteration  # unused because the chosen beliefs carry the history
        candidates = list(draws)
        assert len(candidates) == self._num_candidates, "Need every candidate"
        features = np.array([self._get_features(c) for c in candidates])
        best = 0
        if self._chosen_features:
            chosen = np.array(self._chosen_features)
            # The distance from each candidate (row) to each chosen belief (column).
            distances = np.max(np.abs(features[:, None] - chosen[None]), axis=2)
            best = int(np.argmax(np.min(distances, axis=1)))
        self._chosen_features.append(features[best])
        return candidates[best]
