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
    """Caches plans for beliefs the robot may hold after its next information
    gathering.

    One dream has three steps:
    1. Sample joint limits from the current belief and treat them as the truth.
    2. Run the robot's information gatherer in an environment simulated with those
       limits, and update the belief with the imagined responses. The result is a
       sample of the robot's next belief.
    3. Plan for every task under the imagined belief, and add the plans to the
       warm starts.

    Step 2 uses the same gatherer and belief updater the robot runs for real, so
    imagined next beliefs are distributed like real ones. Each dream starts from
    the warm starts cached by the previous dreams. One iteration of the budget is
    one dream.
    """

    def __init__(
        self,
        env_model: TwoLinkArmModel,
        task_distribution: FiniteTaskDistribution[ArmTask],
        planner: RandomShootingPlanner,
        belief_updater: BeliefUpdater[ArmArray, JointQuery, bool],
        info_gatherer: InformationGatherer[ArmArray, JointQuery, bool],
        gathering_budget: Budget,
        planning_budget: Budget,
        max_warm_starts_per_key: int = 4,
    ) -> None:
        self._env_model = env_model
        self._task_distribution = task_distribution
        self._planner = planner
        self._belief_updater = belief_updater
        self._info_gatherer = info_gatherer
        self._gathering_budget = gathering_budget
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
        data = self._info_gatherer.gather(
            belief, imagined_env, self._gathering_budget, rng
        )
        imagined_belief = self._belief_updater.update(belief, data)
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
