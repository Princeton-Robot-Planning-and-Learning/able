"""Information gathering as a side effect of solving tasks."""

import time

import numpy as np

from able.beliefs.base import Belief
from able.envs.base import Environment, TaskDistribution
from able.episodes import run_episode
from able.info_gatherers.base import InformationGatherer
from able.planners.base import Planner
from able.structs import (
    ActType,
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    PlanningFailure,
    StateType,
    TaskType,
    Trajectory,
)


class TaskExecutionGatherer(
    InformationGatherer[
        StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType
    ]
):
    """Gathers information incidentally by solving tasks.

    Each iteration samples a task, plans for it under the current belief with the
    robot's planner and parameters, and executes the resulting policy. The robot
    never plans in order to learn; it learns from whatever the executions reveal.
    Tasks for which planning fails produce no data.
    """

    def __init__(
        self,
        planner: Planner[
            StateType, ActType, TaskType, HiddenParamsType, PlannerParamsType
        ],
        task_distribution: TaskDistribution[TaskType],
        planning_budget: Budget,
    ) -> None:
        self._planner = planner
        self._task_distribution = task_distribution
        self._planning_budget = planning_budget

    def gather(
        self,
        belief: Belief[HiddenParamsType],
        planner_params: PlannerParamsType,
        env: Environment[StateType, ActType, TaskType],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[Trajectory[StateType, ActType]]:
        """Solve tasks until the budget runs out.

        One iteration of the budget is one task. Tasks are sampled from their own
        random stream, so the sequence of tasks does not depend on how much
        randomness planning consumes. Two robots with different planner parameters
        therefore solve the same tasks.
        """
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Information gathering needs a finite budget"
        task_rng, planning_rng = rng.spawn(2)
        start_time = time.perf_counter()
        data: list[Trajectory[StateType, ActType]] = []
        num_tasks = 0
        while not budget.is_exhausted(time.perf_counter() - start_time, num_tasks):
            num_tasks += 1
            task = self._task_distribution.sample(task_rng)
            try:
                policy = self._planner.plan(
                    task, belief, planner_params, self._planning_budget, planning_rng
                )
                data.append(run_episode(env, task, policy))
            except PlanningFailure:
                continue
        return data
