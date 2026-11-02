"""Base classes for environments, environment models, and task distributions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Sequence

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.structs import ActType, HiddenParamsType, StateType, StepResult, TaskType


class TaskDistribution(ABC, Generic[TaskType]):
    """A distribution over tasks. The robot knows this distribution."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> TaskType:
        """Draw one task."""


class FiniteTaskDistribution(TaskDistribution[TaskType]):
    """A uniform distribution over a fixed list of tasks."""

    def __init__(self, tasks: Sequence[TaskType]) -> None:
        assert len(tasks) > 0, "Need at least one task"
        self._tasks = tuple(tasks)

    def get_tasks(self) -> tuple[TaskType, ...]:
        """All tasks in the support of the distribution."""
        return self._tasks

    def sample(self, rng: np.random.Generator) -> TaskType:
        return self._tasks[rng.integers(len(self._tasks))]


class EnvironmentModel(ABC, Generic[StateType, ActType, TaskType, HiddenParamsType]):
    """The robot's model of the environment, known except for the hidden parameters.

    Every episode lasts exactly `horizon` steps. The total cost of an episode is the
    sum of its step costs plus the terminal cost of its final state.
    """

    @property
    @abstractmethod
    def horizon(self) -> int:
        """The number of steps in every episode."""

    @abstractmethod
    def sample_next_state(
        self,
        state: StateType,
        action: ActType,
        hidden_params: HiddenParamsType,
        rng: np.random.Generator,
    ) -> StateType:
        """Sample the state that results from taking the action in the state."""

    @abstractmethod
    def get_cost(
        self,
        task: TaskType,
        state: StateType,
        action: ActType,
        next_state: StateType,
        hidden_params: HiddenParamsType,
    ) -> float:
        """The cost of one step."""

    @abstractmethod
    def get_terminal_cost(
        self, task: TaskType, state: StateType, hidden_params: HiddenParamsType
    ) -> float:
        """The cost of ending an episode in the state."""


class BatchRolloutModel(
    EnvironmentModel[
        NDArray[np.float64], NDArray[np.float64], TaskType, NDArray[np.float64]
    ]
):
    """A model that simulates many action sequences under many hidden parameter
    values at once.

    States, actions, and hidden parameters are arrays. Planners that evaluate
    thousands of rollouts per call need this, because simulating one transition at
    a time in Python is too slow. In the shapes below, K is the number of action
    sequences, P is the number of hidden parameter values, and T is the length of
    each action sequence (normally the horizon).
    """

    @abstractmethod
    def rollout_batch(
        self,
        initial_state: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Simulate every action sequence under every hidden parameter value.

        The action sequences have shape (K, T, *action_shape) and the hidden
        parameters have shape (P, *hidden_params_shape). Returns states with shape
        (K, P, T + 1, *state_shape), each sequence starting at the initial state.
        """

    @abstractmethod
    def get_cost_batch(
        self,
        task: TaskType,
        state_sequences: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """The total cost of each rollout, with shape (K, P).

        The total includes the terminal cost of the final state. The inputs have
        the shapes described in rollout_batch().
        """

    def get_cost_lower_bound(
        self, task: TaskType, hidden_params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """A lower bound on the total cost achievable under each hidden parameter
        value, with shape (P,).

        Planners can use this to stop once a plan is close to the best achievable
        cost. The default bound is zero.
        """
        del task  # unused in the default bound
        return np.zeros(len(hidden_params))


class Environment(ABC, Generic[StateType, ActType, TaskType]):
    """An environment whose hidden parameters are fixed and not accessible."""

    @abstractmethod
    def reset(self, task: TaskType) -> StateType:
        """Start an episode for the task and return the initial state."""

    @abstractmethod
    def step(self, action: ActType) -> StepResult[StateType]:
        """Take one action in the current episode."""


@dataclass(frozen=True)
class Domain(Generic[StateType, ActType, TaskType, HiddenParamsType]):
    """One problem instance for an experiment.

    The robot receives the environment model, the task distribution, and the
    initial belief. Only the experiment harness uses the real environment, which
    holds the true hidden parameters.
    """

    env_model: EnvironmentModel[StateType, ActType, TaskType, HiddenParamsType]
    task_distribution: TaskDistribution[TaskType]
    initial_belief: Belief[HiddenParamsType]
    real_env: Environment[StateType, ActType, TaskType]
