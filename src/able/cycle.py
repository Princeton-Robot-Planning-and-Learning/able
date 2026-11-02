"""The cycle of sleep, information gathering, belief update, and evaluation."""

import time
from dataclasses import dataclass
from typing import Generic, Sequence

import numpy as np

from able.beliefs.base import Belief
from able.envs.base import Environment
from able.episodes import run_episode
from able.policies.base import Policy
from able.robot import Robot
from able.structs import (
    ActType,
    Budget,
    HiddenParamsType,
    PlannerParamsType,
    PlanningFailure,
    QueryResponse,
    QueryType,
    ResponseType,
    StateType,
    TaskType,
    Trajectory,
)


@dataclass(frozen=True)
class EvaluationResult(Generic[StateType, ActType, TaskType]):
    """The outcome of planning for and executing one evaluation task.

    The planning wall time covers plan() and every call to get_action(). If
    planning fails, the trajectory is None, the total cost is infinite, and the
    failure holds the error message.
    """

    task: TaskType
    trajectory: Trajectory[StateType, ActType] | None
    total_cost: float
    planning_wall_time: float
    planning_stats: dict[str, float]
    failure: str | None = None


@dataclass(frozen=True)
class CycleResult(
    Generic[StateType, ActType, TaskType, HiddenParamsType, QueryType, ResponseType]
):
    """Everything recorded in one cycle.

    The belief is the one after the update, which is also the belief the robot
    was evaluated under.
    """

    cycle: int
    sleep_wall_time: float
    gathered_data: list[QueryResponse[QueryType, ResponseType]]
    belief: Belief[HiddenParamsType]
    evaluations: list[EvaluationResult[StateType, ActType, TaskType]]


@dataclass(frozen=True)
class CycleBudgets:
    """Budgets for each phase of a cycle.

    The planning budget applies to each evaluation task.
    """

    sleep: Budget
    information_gathering: Budget
    planning: Budget


class _TimedPolicy(Policy[StateType, ActType]):
    """Wraps a policy and accumulates the wall time spent choosing actions."""

    def __init__(self, policy: Policy[StateType, ActType]) -> None:
        self._policy = policy
        self.total_time = 0.0

    def get_action(self, state: StateType) -> ActType:
        """Choose an action with the wrapped policy and time it."""
        start_time = time.perf_counter()
        try:
            return self._policy.get_action(state)
        finally:
            self.total_time += time.perf_counter() - start_time

    def get_planning_stats(self) -> dict[str, float]:
        return self._policy.get_planning_stats()


def evaluate(
    robot: Robot[
        StateType,
        ActType,
        TaskType,
        HiddenParamsType,
        PlannerParamsType,
        QueryType,
        ResponseType,
    ],
    task: TaskType,
    env: Environment[StateType, ActType, TaskType, QueryType, ResponseType],
    planning_budget: Budget,
    rng: np.random.Generator,
) -> EvaluationResult[StateType, ActType, TaskType]:
    """Plan for the task and execute the policy for one episode.

    Nothing from the evaluation is passed back to the robot.
    """
    start_time = time.perf_counter()
    try:
        policy = _TimedPolicy(robot.plan(task, planning_budget, rng))
    except PlanningFailure as failure:
        return EvaluationResult(
            task=task,
            trajectory=None,
            total_cost=float("inf"),
            planning_wall_time=time.perf_counter() - start_time,
            planning_stats={},
            failure=str(failure),
        )
    plan_time = time.perf_counter() - start_time
    try:
        trajectory = run_episode(env, task, policy)
    except PlanningFailure as failure:
        return EvaluationResult(
            task=task,
            trajectory=None,
            total_cost=float("inf"),
            planning_wall_time=plan_time + policy.total_time,
            planning_stats=policy.get_planning_stats(),
            failure=str(failure),
        )
    return EvaluationResult(
        task=task,
        trajectory=trajectory,
        total_cost=trajectory.total_cost,
        planning_wall_time=plan_time + policy.total_time,
        planning_stats=policy.get_planning_stats(),
    )


def run_cycles(
    robot: Robot[
        StateType,
        ActType,
        TaskType,
        HiddenParamsType,
        PlannerParamsType,
        QueryType,
        ResponseType,
    ],
    env: Environment[StateType, ActType, TaskType, QueryType, ResponseType],
    eval_tasks: Sequence[TaskType],
    num_cycles: int,
    budgets: CycleBudgets,
    rng: np.random.Generator,
) -> list[
    CycleResult[StateType, ActType, TaskType, HiddenParamsType, QueryType, ResponseType]
]:
    """Run the robot for a number of cycles.

    Each cycle has the robot sleep, gather information in the environment, and
    update its belief. It is then evaluated on every evaluation task under the
    updated belief, which it has not slept on yet. Sleep, information gathering,
    and evaluation each draw from their own random stream.
    """
    sleep_rng, gather_rng, eval_rng = rng.spawn(3)
    results = []
    for cycle in range(num_cycles):
        start_time = time.perf_counter()
        robot.sleep(budgets.sleep, sleep_rng)
        sleep_wall_time = time.perf_counter() - start_time
        data = robot.gather_information(env, budgets.information_gathering, gather_rng)
        robot.update_belief(data)
        evaluations = [
            evaluate(robot, task, env, budgets.planning, eval_rng)
            for task in eval_tasks
        ]
        results.append(
            CycleResult(
                cycle=cycle,
                sleep_wall_time=sleep_wall_time,
                gathered_data=data,
                belief=robot.belief,
                evaluations=evaluations,
            )
        )
    return results
