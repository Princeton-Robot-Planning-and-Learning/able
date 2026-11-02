"""A one-dimensional environment model used in tests.

A point moves along a line toward a goal, and a wall at an unknown position stops it
from moving past the wall. The hidden parameter is the wall's position.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from able.envs.base import BatchRolloutModel
from able.structs import Task

ACTION_COST = 0.1


@dataclass(frozen=True, eq=False)
class WallTask(Task):
    """Reach the goal position."""

    goal: float


class WallModel(BatchRolloutModel[WallTask]):
    """States, actions, and hidden parameters all have shape (1,)."""

    def __init__(self, horizon: int = 3) -> None:
        self._horizon = horizon

    @property
    def horizon(self) -> int:
        return self._horizon

    def sample_next_state(
        self,
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return np.minimum(state + action, hidden_params)

    def get_cost(
        self,
        task: WallTask,
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        next_state: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> float:
        return ACTION_COST * float(np.abs(action).sum())

    def get_terminal_cost(
        self,
        task: WallTask,
        state: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> float:
        return float(np.abs(state - task.goal).sum())

    def rollout_batch(
        self,
        initial_state: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        num_seqs, length = action_sequences.shape[:2]
        num_params = hidden_params.shape[0]
        states = np.empty((num_seqs, num_params, length + 1, 1))
        states[:, :, 0] = initial_state
        for t in range(length):
            states[:, :, t + 1] = np.minimum(
                states[:, :, t] + action_sequences[:, None, t], hidden_params[None]
            )
        return states

    def get_cost_batch(
        self,
        task: WallTask,
        state_sequences: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        action_costs = ACTION_COST * np.abs(action_sequences).sum(axis=(1, 2))
        terminal_costs = np.abs(state_sequences[:, :, -1] - task.goal).sum(axis=-1)
        return action_costs[:, None] + terminal_costs
