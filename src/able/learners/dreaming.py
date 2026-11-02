"""Learning warm starts for random shooting by imagining future beliefs."""

import time

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
from able.learners.plan_caching import PlanCachingLearner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.recording.base import CycleRecorder, NullRecorder
from able.structs import Budget, PlanningFailure

ArmArray = NDArray[np.float64]


class DreamingLearner(PlanCachingLearner):
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
        recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] | None = None,
    ) -> None:
        assert (
            execution_budget.max_iterations is not None
            or execution_budget.max_wall_time is not None
        ), "Imagined task execution needs a finite budget"
        super().__init__(task_distribution, planner, planning_budget)
        self._env_model = env_model
        self._belief_updater = belief_updater
        self._info_gatherer = info_gatherer
        self._gathering_budget = gathering_budget
        self._execution_budget = execution_budget
        self._recorder: CycleRecorder[ArmArray, ArmTask, ArmArray] = (
            NullRecorder() if recorder is None else recorder
        )

    def _get_training_belief(
        self,
        iteration: int,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        rng: np.random.Generator,
    ) -> IntervalBelief:
        label = f"Dream {iteration}"
        hypothesis = belief.sample(rng)
        imagined_env = SimulatedEnvironment(self._env_model, hypothesis, rng)
        imagined_belief = self._imagine_until_evaluation(
            label, belief, params, imagined_env, rng
        )
        assert isinstance(imagined_belief, IntervalBelief)
        self._recorder.record_belief(
            f"{label} · imagined belief", imagined_belief, hypothesis
        )
        return imagined_belief

    def _imagine_until_evaluation(
        self,
        label: str,
        belief: Belief[ArmArray],
        params: RandomShootingParams,
        env: SimulatedEnvironment,
        rng: np.random.Generator,
    ) -> Belief[ArmArray]:
        """Imagine information gathering and task execution in the environment and
        return the resulting belief."""
        hypothesis = env.hidden_params
        data = self._info_gatherer.gather(belief, env, self._gathering_budget, rng)
        belief = self._belief_updater.update_from_queries(belief, data)
        self._recorder.record_queries(data, belief, hypothesis)

        def update(state: ArmArray, action: ArmArray, next_state: ArmArray) -> None:
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
