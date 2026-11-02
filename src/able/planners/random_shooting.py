"""Random shooting for the two-link arm, with warm starts from cached plans."""

import time
from dataclasses import dataclass, field
from typing import Hashable, Mapping

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import ArmTask, TwoLinkArmModel
from able.planners.base import Planner
from able.policies.base import OpenLoopPolicy
from able.structs import Budget, PlanningFailure

# A cached plan: its expected cost and its action sequence with shape (T, 2).
WarmStart = tuple[float, NDArray[np.float64]]


@dataclass(frozen=True, eq=False)
class RandomShootingParams:
    """Parameters of the random shooting planner.

    The planner evaluates batches of sampled action sequences. It stops when the
    best expected cost is within `threshold` of the cost lower bound, when
    `patience` consecutive batches fail to improve on it, or after `max_batches`.

    The warm starts map a discretized (goal, belief) key to cached plans. When the
    key for a planning problem is present, the first batch consists of the cached
    plans plus perturbations of them.
    """

    batch_size: int = 64
    max_batches: int = 25
    threshold: float = 0.08
    patience: int = 8
    num_belief_samples: int = 32
    action_noise_fraction: float = 0.15
    warm_start_noise: float = 0.05
    key_resolution: float = 0.1
    warm_starts: Mapping[Hashable, tuple[WarmStart, ...]] = field(default_factory=dict)


def get_warm_start_key(
    task: ArmTask, belief: IntervalBelief, resolution: float
) -> Hashable:
    """The cache key for planning for the task under the belief."""
    goal_key = tuple(int(v) for v in np.round(task.goal / resolution))
    return (goal_key, belief.get_discretized_key(resolution))


def add_warm_start(
    warm_starts: dict[Hashable, tuple[WarmStart, ...]],
    key: Hashable,
    cost: float,
    actions: NDArray[np.float64],
    max_per_key: int = 4,
) -> None:
    """Add a plan to the warm starts in place, keeping the cheapest plans per key."""
    entries = list(warm_starts.get(key, ())) + [(cost, actions.copy())]
    entries.sort(key=lambda entry: entry[0])
    warm_starts[key] = tuple(entries[:max_per_key])


class RandomShootingPlanner(
    Planner[
        NDArray[np.float64],
        NDArray[np.float64],
        ArmTask,
        NDArray[np.float64],
        RandomShootingParams,
    ]
):
    """Plans by sampling action sequences and picking the one with the lowest
    expected cost over samples from the belief.

    Each sampled sequence is a random base displacement plus per-step noise. The
    planner returns an open-loop policy. Its planning statistics are the number of
    sequences evaluated, the expected cost of the plan, and whether a warm start
    was used.
    """

    def __init__(self, model: TwoLinkArmModel) -> None:
        self._model = model

    def get_default_params(self) -> RandomShootingParams:
        return RandomShootingParams()

    def plan(
        self,
        task: ArmTask,
        belief: Belief[NDArray[np.float64]],
        params: RandomShootingParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> OpenLoopPolicy[NDArray[np.float64], NDArray[np.float64]]:
        """Find an open-loop plan.

        One iteration of the budget is one batch of action sequences.
        """
        start_time = time.perf_counter()
        assert isinstance(belief, IntervalBelief), "Warm starts need IntervalBelief"
        hidden_params = np.stack(
            [belief.sample(rng) for _ in range(params.num_belief_samples)]
        )
        lower_bound = float(
            self._model.get_cost_lower_bound(task, hidden_params).mean()
        )
        target_cost = lower_bound + params.threshold
        key = get_warm_start_key(task, belief, params.key_resolution)
        warm_start_batch = self._get_warm_start_batch(key, params, rng)

        best_cost = np.inf
        best_actions: NDArray[np.float64] | None = None
        num_samples = 0
        num_stale_batches = 0
        for batch_idx in range(params.max_batches):
            if budget.is_exhausted(time.perf_counter() - start_time, batch_idx):
                break
            if batch_idx == 0 and warm_start_batch is not None:
                actions = warm_start_batch
            else:
                actions = self._sample_action_sequences(params, rng)
            states = self._model.rollout_batch(
                task.initial_state, actions, hidden_params, rng
            )
            costs = self._model.get_cost_batch(
                task, states, actions, hidden_params
            ).mean(axis=1)
            num_samples += len(actions)
            best_idx = int(np.argmin(costs))
            # Only a meaningful improvement resets the patience counter.
            if costs[best_idx] < 0.99 * best_cost:
                best_cost = float(costs[best_idx])
                best_actions = actions[best_idx]
                num_stale_batches = 0
            else:
                num_stale_batches += 1
            if best_cost < target_cost or num_stale_batches >= params.patience:
                break

        if best_actions is None:
            raise PlanningFailure("No action sequences were evaluated within budget")
        stats = {
            "num_samples": float(num_samples),
            "expected_cost": best_cost,
            "used_warm_start": float(warm_start_batch is not None),
        }
        return OpenLoopPolicy(list(best_actions), stats)

    def _sample_action_sequences(
        self, params: RandomShootingParams, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        bound = self._model.max_joint_displacement
        num, horizon = params.batch_size, self._model.horizon
        base = rng.uniform(-bound, bound, size=(num, 1, 2))
        noise = rng.normal(0.0, params.action_noise_fraction * bound, (num, horizon, 2))
        return np.clip(base + noise, -bound, bound)

    def _get_warm_start_batch(
        self, key: Hashable, params: RandomShootingParams, rng: np.random.Generator
    ) -> NDArray[np.float64] | None:
        entries = params.warm_starts.get(key)
        if not entries:
            return None
        seeds = np.stack([actions for _, actions in entries])
        reps = max(1, params.batch_size // len(seeds))
        perturbed = np.repeat(seeds, reps, axis=0) + rng.normal(
            0.0, params.warm_start_noise, size=(len(seeds) * reps,) + seeds.shape[1:]
        )
        bound = self._model.max_joint_displacement
        return np.clip(np.concatenate([seeds, perturbed]), -bound, bound)
