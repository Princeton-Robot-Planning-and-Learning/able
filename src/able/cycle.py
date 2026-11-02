"""The cycle of sleep, information gathering, task execution, and evaluation."""

import time
from dataclasses import dataclass
from typing import Callable, Generic, Sequence

import numpy as np

from able.beliefs.base import Belief
from able.envs.base import Environment, TaskDistribution
from able.episodes import run_episode
from able.policies.base import Policy
from able.recording.base import CycleRecorder, NullRecorder, Stage
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
class EpisodeResult(Generic[StateType, ActType, TaskType]):
    """The outcome of planning for and executing one task.

    Success means the environment's goal check passed on the final state. The
    planning wall time covers plan() and every call to get_action(). If planning
    fails, the trajectory is None, the total cost is infinite, the episode is not
    a success, and the failure holds the error message.
    """

    task: TaskType
    trajectory: Trajectory[StateType, ActType] | None
    total_cost: float
    success: bool
    planning_wall_time: float
    planning_stats: dict[str, float]
    failure: str | None = None


@dataclass(frozen=True)
class CycleResult(
    Generic[StateType, ActType, TaskType, HiddenParamsType, QueryType, ResponseType]
):
    """Everything recorded in one cycle.

    The belief is the one after information gathering and task execution, which
    is also the belief the robot was evaluated under.
    """

    cycle: int
    sleep_wall_time: float
    gathered_data: list[QueryResponse[QueryType, ResponseType]]
    executions: list[EpisodeResult[StateType, ActType, TaskType]]
    belief: Belief[HiddenParamsType]
    evaluations: list[EpisodeResult[StateType, ActType, TaskType]]


@dataclass(frozen=True)
class CycleBudgets:
    """Budgets for each phase of a cycle.

    One iteration of the execution budget is one task. The planning budget
    applies to each executed task and each evaluation task.
    """

    sleep: Budget
    information_gathering: Budget
    execution: Budget
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


def _plan_and_run(
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
    on_step: Callable[[StateType, ActType, StateType], None] | None,
) -> EpisodeResult[StateType, ActType, TaskType]:
    """Plan for the task under the robot's current belief and run one episode."""
    start_time = time.perf_counter()
    try:
        policy = _TimedPolicy(robot.plan(task, planning_budget, rng))
    except PlanningFailure as failure:
        return EpisodeResult(
            task=task,
            trajectory=None,
            total_cost=float("inf"),
            success=False,
            planning_wall_time=time.perf_counter() - start_time,
            planning_stats={},
            failure=str(failure),
        )
    plan_time = time.perf_counter() - start_time
    try:
        trajectory = run_episode(env, task, policy, on_step=on_step)
    except PlanningFailure as failure:
        return EpisodeResult(
            task=task,
            trajectory=None,
            total_cost=float("inf"),
            success=False,
            planning_wall_time=plan_time + policy.total_time,
            planning_stats=policy.get_planning_stats(),
            failure=str(failure),
        )
    return EpisodeResult(
        task=task,
        trajectory=trajectory,
        total_cost=trajectory.total_cost,
        success=env.check_goal(task, trajectory.states[-1]),
        planning_wall_time=plan_time + policy.total_time,
        planning_stats=policy.get_planning_stats(),
    )


def execute(
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
    recorder: CycleRecorder[StateType, TaskType, HiddenParamsType] | None = None,
    label: str = "Executed task",
) -> EpisodeResult[StateType, ActType, TaskType]:
    """Plan for the task and execute the policy, updating the robot's belief after
    every step.

    The policy was planned under the belief at the start of the episode and does
    not see the updates. If the policy fails partway, the belief keeps the updates
    from the steps already taken. After a complete episode, the robot's learner
    observes it. The recorder, if given, sees the episode under the label.
    """
    recorder = NullRecorder() if recorder is None else recorder
    recorder.start_episode(label, task, robot.belief)

    def on_step(state: StateType, action: ActType, next_state: StateType) -> None:
        robot.update_belief_from_transition(state, action, next_state)
        recorder.record_step(next_state, robot.belief)

    result = _plan_and_run(robot, task, env, planning_budget, rng, on_step)
    if result.trajectory is not None:
        robot.observe_execution(task, result.trajectory)
    recorder.end_episode(_describe_outcome(result))
    return result


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
    recorder: CycleRecorder[StateType, TaskType, HiddenParamsType] | None = None,
    label: str = "Evaluation task",
) -> EpisodeResult[StateType, ActType, TaskType]:
    """Plan for the task and execute the policy for one episode.

    Nothing from the evaluation is passed back to the robot. The recorder, if
    given, sees the episode under the label.
    """
    recorder = NullRecorder() if recorder is None else recorder
    recorder.start_episode(label, task, robot.belief)

    def on_step(state: StateType, action: ActType, next_state: StateType) -> None:
        del state, action  # unused because the belief does not change
        recorder.record_step(next_state, robot.belief)

    result = _plan_and_run(robot, task, env, planning_budget, rng, on_step)
    recorder.end_episode(_describe_outcome(result))
    return result


def _describe_outcome(result: EpisodeResult) -> str:
    if result.failure is not None:
        return "planning failed"
    outcome = "success" if result.success else "failure"
    return f"{outcome}, planning time {1000 * result.planning_wall_time:.1f} ms"


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
    task_distribution: TaskDistribution[TaskType],
    eval_tasks: Sequence[TaskType],
    num_cycles: int,
    budgets: CycleBudgets,
    rng: np.random.Generator,
    recorder: CycleRecorder[StateType, TaskType, HiddenParamsType] | None = None,
) -> list[
    CycleResult[StateType, ActType, TaskType, HiddenParamsType, QueryType, ResponseType]
]:
    """Run the robot for a number of cycles.

    In each cycle, the robot sleeps, asks queries and updates its belief with the
    responses, and then executes tasks sampled from the task distribution,
    updating its belief during execution. It is then evaluated on every
    evaluation task under the resulting belief, which it has not slept on yet.

    Sleep, information gathering, task sampling, execution, and evaluation each
    draw from their own random stream. Robots that plan differently therefore
    execute the same sequence of tasks. The recorder, if given, sees every stage.
    """
    assert (
        budgets.execution.max_iterations is not None
        or budgets.execution.max_wall_time is not None
    ), "Task execution needs a finite budget"
    recorder = NullRecorder() if recorder is None else recorder
    sleep_rng, gather_rng, task_rng, execution_rng, eval_rng = rng.spawn(5)
    results = []
    for cycle in range(num_cycles):
        recorder.start_stage(cycle, Stage.SLEEP)
        start_time = time.perf_counter()
        robot.sleep(budgets.sleep, sleep_rng)
        sleep_wall_time = time.perf_counter() - start_time
        recorder.end_stage()

        recorder.start_stage(cycle, Stage.QUERIES)
        data = robot.gather_information(env, budgets.information_gathering, gather_rng)
        robot.update_belief_from_queries(data)
        recorder.record_queries(data, robot.belief)
        recorder.end_stage()

        recorder.start_stage(cycle, Stage.EXECUTION)
        executions: list[EpisodeResult[StateType, ActType, TaskType]] = []
        start_time = time.perf_counter()
        while not budgets.execution.is_exhausted(
            time.perf_counter() - start_time, len(executions)
        ):
            task = task_distribution.sample(task_rng)
            label = f"Executed task {len(executions) + 1}"
            executions.append(
                execute(
                    robot, task, env, budgets.planning, execution_rng, recorder, label
                )
            )
        recorder.end_stage()

        recorder.start_stage(cycle, Stage.EVALUATION)
        evaluations = []
        for i, task in enumerate(eval_tasks):
            label = f"Evaluation task {i + 1} of {len(eval_tasks)}"
            evaluations.append(
                evaluate(robot, task, env, budgets.planning, eval_rng, recorder, label)
            )
        recorder.end_stage()
        results.append(
            CycleResult(
                cycle=cycle,
                sleep_wall_time=sleep_wall_time,
                gathered_data=data,
                executions=executions,
                belief=robot.belief,
                evaluations=evaluations,
            )
        )
    return results
