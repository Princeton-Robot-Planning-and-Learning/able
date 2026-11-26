"""A limb planner that follows a plan its model predicts while awake."""

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.calendar import ExerciseTask
from able.envs.limb.human import LimbVariant
from able.envs.limb.model import STOP, LimbModel, LimbTask
from able.envs.limb.sim import CoupledState
from able.models.base import Dataset
from able.planners.mpc import (
    LimbMPCConfig,
    LimbMPCPlanner,
    LimbMPCPolicy,
    MPCParams,
    PredictiveSamplingConfig,
    SeedConfig,
    _Attempt,
)
from able.policies.base import Policy
from able.structs import Budget, PlanningFailure

# Stored plans are padded with STOP to this many actions.
MAX_PLAN_LENGTH = 300
_NUM_JOINTS = 7


def _make_search(num_rollouts: int, commit_steps: int) -> LimbMPCConfig:
    return LimbMPCConfig(
        sampling=PredictiveSamplingConfig(
            num_rollouts=num_rollouts, commit_steps=commit_steps
        )
    )


# The searches that training plans try in turn, sharing the training budget.
TRAINING_SEARCHES = (_make_search(12, 4), _make_search(16, 3), _make_search(24, 2))

# While awake, each MPC step first checks the predicted plan's next horizon of
# controls and takes all of it if it breaks no constraint. Otherwise, or once the
# plan has ended, the step scores num_rollouts candidates in all (the plan's rejected
# controls among them), with no candidates pushed toward the goal, so the plan never
# loses to one. MPC stops at the goal tolerance, as a replay of the plan would.
AWAKE_SEARCH = LimbMPCConfig(
    sampling=PredictiveSamplingConfig(num_rollouts=4, commit_steps=12),
    directed_gains=(),
    replay_slack=0.0,
)
AWAKE_SEED = SeedConfig(num_rollouts=1, commit_steps=12)


@dataclass(frozen=True, eq=False)
class RecallParams(MPCParams):
    """MPC parameters, whether to search for a training plan instead of following a
    predicted one, and the predicted plan to follow."""

    config: LimbMPCConfig = AWAKE_SEARCH
    seed: SeedConfig = AWAKE_SEED
    training: bool = False
    searches: tuple[LimbMPCConfig, ...] = field(default=TRAINING_SEARCHES)
    plan: NDArray[np.float64] | None = None


class LimbRecallPlanner(LimbMPCPlanner):
    """Plans with MPC that follows the plan the model predicts from the training plans.

    While awake it predicts a plan for the task's exercise and the belief's features,
    and MPC proposes the plan's controls for the steps they cover (see AWAKE_SEARCH),
    within the budget's rollouts. With no training plans, MPC plans from scratch.

    Training plans (see get_training_params()) try MPC searches in turn within the
    budget. A plan is stored whole, up to where MPC ends, rather than only until it
    reaches the goal tolerance, so it can also serve nearby goals. The search is for
    the exercise under the belief that round_exercise() gives, and its random numbers
    come from the exercise and the belief's features, so a training plan does not
    depend on when it is made or on who makes it.

    get_features gives the model's input for a task under a belief, and starts with
    the exercise. round_exercise gives an exercise's task under a belief rounded to a
    resolution finer than the features' differences between days, and that belief.
    """

    def __init__(
        self,
        model: LimbModel,
        get_features: Callable[[LimbTask, RangeOfMotionBelief], NDArray[np.float64]],
        round_exercise: Callable[
            [int, RangeOfMotionBelief], tuple[LimbTask, RangeOfMotionBelief]
        ],
        default_params: RecallParams,
        seed: int = 0,
    ) -> None:
        super().__init__(model, default_params)
        self._get_features = get_features
        self._round_exercise = round_exercise
        self._seed = seed

    @property
    def variant(self) -> LimbVariant:
        """The variant of the limb that the planner moves."""
        return self._model.variant

    def get_training_params(self, params: MPCParams) -> MPCParams:
        assert isinstance(params, RecallParams), "Need RecallParams"
        return dataclasses.replace(params, training=True)

    def get_plan_examples(
        self,
        task: LimbTask,
        belief: Belief[NDArray[np.float64]],
        policy: Policy,
    ) -> Dataset:
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        assert isinstance(policy, LimbMPCPolicy), "Need a plan from plan()"
        controls = np.zeros((MAX_PLAN_LENGTH, _NUM_JOINTS))
        length = min(len(policy.controls), MAX_PLAN_LENGTH)
        controls[:length] = np.array(policy.controls)[:length]
        return Dataset(
            x=self._get_features(task, belief)[None],
            y=controls.reshape(1, -1),
            costs=np.zeros(1),
            groups=np.zeros(1, dtype=np.int64),
        )

    def plan(
        self,
        task: LimbTask,
        belief: Belief[NDArray[np.float64]],
        params: MPCParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> LimbMPCPolicy:
        assert isinstance(params, RecallParams), "Need RecallParams"
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        assert isinstance(task, ExerciseTask), "Need an exercise"
        if params.training:
            return self._plan_for_training(task, belief, params, budget)
        plan = self._predict(task, belief, params, rng)
        return super().plan(
            task, belief, dataclasses.replace(params, plan=plan), budget, rng
        )

    def _predict(
        self,
        task: LimbTask,
        belief: RangeOfMotionBelief,
        params: RecallParams,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """The plan that the model predicts, without its trailing zero controls, or
        no controls before the model has plans to predict from."""
        proposals = params.model.sample(self._get_features(task, belief), 1, rng)
        if len(proposals) == 0:
            return np.zeros((0, _NUM_JOINTS))
        controls = proposals[0].reshape(MAX_PLAN_LENGTH, _NUM_JOINTS)
        moving = np.flatnonzero(np.abs(controls).max(axis=1) > 0)
        return controls[: moving[-1] + 1] if len(moving) else controls[:0]

    def _propose(
        self,
        attempt: _Attempt,
        state: CoupledState,
        step: int,
        num: int,
        previous: NDArray[np.float64],
    ) -> list[NDArray[np.float64]]:
        """The predicted plan's controls from the step on, padded with zero controls
        to the horizon, or none once the plan has ended."""
        params = attempt.params
        if not isinstance(params, RecallParams) or params.plan is None:
            return super()._propose(attempt, state, step, num, previous)
        controls = params.plan[step : step + params.config.sampling.horizon]
        if len(controls) == 0:
            return []
        padded = np.zeros((params.config.sampling.horizon, _NUM_JOINTS))
        padded[: len(controls)] = controls
        return [padded]

    def _plan_for_training(
        self,
        task: ExerciseTask,
        belief: RangeOfMotionBelief,
        params: RecallParams,
        budget: Budget,
    ) -> LimbMPCPolicy:
        search_task, search_belief = self._round_exercise(task.exercise, belief)
        features = self._get_features(search_task, search_belief)
        key = tuple(int(v) for v in np.round(features * 1000))
        rng = np.random.default_rng([self._seed, *(v + 10**6 for v in key)])
        start, used = time.perf_counter(), 0.0
        controls_found: list[NDArray[np.float64]] = []
        failure = PlanningFailure("No search was tried")
        for search in params.searches:
            remaining = None
            if budget.max_iterations is not None:
                remaining = int(budget.max_iterations - used)
                if remaining <= 0:
                    break
            try:
                plan = super().plan(
                    search_task,
                    search_belief,
                    MPCParams(config=search, confidence=params.confidence),
                    Budget(
                        max_wall_time=(
                            None
                            if budget.max_wall_time is None
                            else budget.max_wall_time - (time.perf_counter() - start)
                        ),
                        max_iterations=remaining,
                    ),
                    rng,
                )
                controls_found = list(plan.controls)
                break
            except PlanningFailure as error:
                failure = error
                used += (error.stats or {}).get("num_rollouts", 0.0)
        if not controls_found:
            raise failure
        return self._as_policy(task, controls_found)

    def _as_policy(
        self, task: LimbTask, controls: list[NDArray[np.float64]]
    ) -> LimbMPCPolicy:
        """A plan whose replay is the whole control sequence."""
        return LimbMPCPolicy(
            controls[: self._model.horizon]
            + [STOP] * (self._model.horizon - len(controls)),
            {"num_rollouts": 1.0},
            np.asarray(task.goal),
            1,
            [],
            controls,
        )
