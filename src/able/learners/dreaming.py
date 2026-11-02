"""Learning warm starts for random shooting by imagining future beliefs."""

import dataclasses
import time
from typing import Hashable

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import ArmTask, JointQuery, TwoLinkArmModel
from able.episodes import run_episode
from able.info_gatherers.base import InformationGatherer
from able.learners.base import Learner
from able.planners.random_shooting import (
    RandomShootingParams,
    RandomShootingPlanner,
    WarmStart,
    add_warm_start,
    get_warm_start_key,
)
from able.structs import Budget, PlanningFailure

ArmArray = NDArray[np.float64]


class DreamingLearner(Learner[ArmArray, RandomShootingParams]):
    """Caches plans for beliefs the robot may hold when it is next evaluated.

    One dream has three steps:
    1. Sample joint limits from the current belief and treat them as the truth.
    2. In an environment simulated with those limits, imagine the rest of the
       cycle before evaluation. Ask queries with the robot's information gatherer
       and update the belief with the responses. Then execute tasks sampled from
       the task distribution, updating the belief after every step. The result is
       a sample of the belief the robot will be evaluated under.
    3. Plan for every task under the imagined belief, and add the plans to the
       warm starts.

    Step 2 uses the robot's gatherer, belief updater, planner, and planner
    parameters, so imagined beliefs are distributed like real ones. Each dream
    starts from the warm starts cached by the previous dreams. One iteration of
    the budget is one dream, and one iteration of the execution budget is one
    task.
    """

    def __init__(
        self,
        env_model: TwoLinkArmModel,
        task_distribution: FiniteTaskDistribution[ArmTask],
        planner: RandomShootingPlanner,
        belief_updater: BeliefUpdater[ArmArray, ArmArray, ArmArray, JointQuery, bool],
        info_gatherer: InformationGatherer[ArmArray, JointQuery, bool],
        gathering_budget: Budget,
        execution_budget: Budget,
        planning_budget: Budget,
        max_warm_starts_per_key: int = 4,
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
        self._max_warm_starts_per_key = max_warm_starts_per_key

    def learn(
        self,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> RandomShootingParams:
        """Dream until the budget runs out and return parameters with the new
        warm starts."""
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Dreaming needs a finite budget"
        warm_starts: dict[Hashable, tuple[WarmStart, ...]] = dict(params.warm_starts)
        start_time = time.perf_counter()
        num_dreams = 0
        while not budget.is_exhausted(time.perf_counter() - start_time, num_dreams):
            num_dreams += 1
            dream_params = dataclasses.replace(params, warm_starts=dict(warm_starts))
            self._dream(belief, dream_params, warm_starts, rng)
        return dataclasses.replace(params, warm_starts=warm_starts)

    def _dream(
        self,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        warm_starts: dict[Hashable, tuple[WarmStart, ...]],
        rng: np.random.Generator,
    ) -> None:
        hypothesis = belief.sample(rng)
        imagined_env = SimulatedEnvironment(self._env_model, hypothesis, rng)
        imagined_belief = self._imagine_until_evaluation(
            belief, params, imagined_env, rng
        )
        assert isinstance(imagined_belief, IntervalBelief)
        for task in self._task_distribution.get_tasks():
            try:
                policy = self._planner.plan(
                    task, imagined_belief, params, self._planning_budget, rng
                )
            except PlanningFailure:
                continue
            key = get_warm_start_key(task, imagined_belief, params.key_resolution)
            add_warm_start(
                warm_starts,
                key,
                policy.get_planning_stats()["expected_cost"],
                np.array(policy.actions),
                self._max_warm_starts_per_key,
            )

    def _imagine_until_evaluation(
        self,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        env: SimulatedEnvironment,
        rng: np.random.Generator,
    ) -> Belief[ArmArray]:
        """Imagine information gathering and task execution in the environment and
        return the resulting belief."""
        data = self._info_gatherer.gather(belief, env, self._gathering_budget, rng)
        belief = self._belief_updater.update_from_queries(belief, data)

        def update(state: ArmArray, action: ArmArray, next_state: ArmArray) -> None:
            nonlocal belief
            belief = self._belief_updater.update_from_transition(
                belief, state, action, next_state
            )

        start_time = time.perf_counter()
        num_tasks = 0
        while not self._execution_budget.is_exhausted(
            time.perf_counter() - start_time, num_tasks
        ):
            num_tasks += 1
            task = self._task_distribution.sample(rng)
            try:
                policy = self._planner.plan(
                    task, belief, params, self._planning_budget, rng
                )
                run_episode(env, task, policy, on_step=update)
            except PlanningFailure:
                continue
        return belief
