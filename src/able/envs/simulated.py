"""An environment simulated from an environment model and fixed hidden parameters."""

from typing import Generic

import numpy as np

from able.envs.base import Environment, EnvironmentModel
from able.structs import (
    ActType,
    HiddenParamsType,
    QueryType,
    ResponseType,
    StateType,
    StepResult,
    TaskType,
)


class SimulatedEnvironment(
    Environment[StateType, ActType, TaskType, QueryType, ResponseType],
    Generic[StateType, ActType, TaskType, HiddenParamsType, QueryType, ResponseType],
):
    """Simulates an environment model with fixed hidden parameters.

    The experiment harness uses this with the true hidden parameters as the real
    environment. Learners use it with sampled hidden parameters to imagine
    interactions.
    """

    def __init__(
        self,
        model: EnvironmentModel[
            StateType, ActType, TaskType, HiddenParamsType, QueryType, ResponseType
        ],
        hidden_params: HiddenParamsType,
        rng: np.random.Generator,
    ) -> None:
        self._model = model
        self._hidden_params = hidden_params
        self._rng = rng
        self._task: TaskType | None = None
        self._state: StateType | None = None
        self._num_steps = 0

    def reset(self, task: TaskType) -> StateType:
        self._task = task
        self._state = task.initial_state
        self._num_steps = 0
        return task.initial_state

    def step(self, action: ActType) -> StepResult[StateType]:
        assert (
            self._task is not None and self._state is not None
        ), "Call reset() before step()"
        assert self._num_steps < self._model.horizon, "Episode is over; call reset()"
        next_state = self._model.sample_next_state(
            self._state, action, self._hidden_params, self._rng
        )
        cost = self._model.get_cost(
            self._task, self._state, action, next_state, self._hidden_params
        )
        self._num_steps += 1
        terminated = self._num_steps >= self._model.horizon
        if terminated:
            cost += self._model.get_terminal_cost(
                self._task, next_state, self._hidden_params
            )
        self._state = next_state
        return StepResult(state=next_state, cost=cost, terminated=terminated)

    def ask(self, query: QueryType) -> ResponseType:
        return self._model.sample_response(query, self._hidden_params, self._rng)

    def check_goal(self, task: TaskType, state: StateType) -> bool:
        return self._model.check_goal(task, state, self._hidden_params)
