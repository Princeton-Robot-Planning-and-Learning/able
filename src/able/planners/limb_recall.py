"""A limb planner that replays stored plans while awake."""

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.calendar import ExerciseTask
from able.envs.limb.human import LimbVariant
from able.envs.limb.model import STOP, LimbModel, LimbTask
from able.models.base import Dataset
from able.planners.mpc import (
    LimbMPCConfig,
    LimbMPCPlanner,
    LimbMPCPolicy,
    MPCParams,
    ParticleLimits,
    PredictiveSamplingConfig,
    _Checker,
    _with_start_reach,
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


@dataclass(frozen=True, eq=False)
class RecallParams(MPCParams):
    """MPC parameters, and whether to search for a training plan instead of
    replaying a stored one."""

    training: bool = False
    searches: tuple[LimbMPCConfig, ...] = field(default=TRAINING_SEARCHES)


class LimbRecallPlanner(LimbMPCPlanner):
    """Plans by replaying the stored plan for the same exercise whose day is nearest.

    While awake it looks up the plan for the task's exercise and the belief's
    features, replays it in its own simulator, and returns it if it reaches the goal
    without breaking a constraint under the believed limits. That counts as one
    iteration of the budget. Otherwise planning fails, since nothing else is cheap
    enough.

    Training plans (see get_training_params()) try MPC searches in turn within the
    budget. A plan is stored whole, up to where MPC ends, rather than only until it
    reaches the goal tolerance, so it can also serve nearby goals. The random numbers
    come from the exercise and the belief's features, so a training plan does not
    depend on when it is made or on who makes it. If plan_dir is given, training
    plans are kept there, by exercise and features, and reused.

    get_features gives the model's input for a task under a belief, and starts with
    the exercise.
    """

    def __init__(
        self,
        model: LimbModel,
        get_features: Callable[[LimbTask, RangeOfMotionBelief], NDArray[np.float64]],
        default_params: RecallParams,
        seed: int = 0,
        plan_dir: Path | None = None,
    ) -> None:
        super().__init__(model, default_params)
        self._get_features = get_features
        self._seed = seed
        self._plan_dir = plan_dir

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
        controls = np.full((MAX_PLAN_LENGTH, _NUM_JOINTS), np.nan)
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
        return self._recall(task, belief, params, rng)

    def _recall(
        self,
        task: LimbTask,
        belief: RangeOfMotionBelief,
        params: RecallParams,
        rng: np.random.Generator,
    ) -> LimbMPCPolicy:
        """The stored plan for the nearest day, if it replays to the goal."""
        proposals = params.model.sample(self._get_features(task, belief), 1, rng)
        if len(proposals) == 0:
            raise PlanningFailure("No stored plan for the exercise")
        controls = proposals[0].reshape(MAX_PLAN_LENGTH, _NUM_JOINTS)
        controls = controls[~np.isnan(controls).all(axis=1)]
        reached = _with_start_reach(belief, task.initial_state)
        particles = np.stack([reached.sample(rng) for _ in range(params.num_particles)])
        limits = ParticleLimits.from_particles(
            self._model.variant, particles, params.confidence
        )
        start = task.initial_state
        world, _ = self._get_worlds(start.coupled)
        checker = _Checker(self._model, limits, dict(start.resting_penetration))
        world.restore(start.coupled)
        used: list[NDArray[np.float64]] = []
        for control in controls:
            for _ in range(self._model.action_repeat):
                broken = checker.step(world, control)
                if broken:
                    raise PlanningFailure(f"The replay broke a constraint: {broken}")
            used.append(control)
            error = world.get_limb_error(
                world.env.limb.get_joint_positions(), task.goal
            )
            if error < world.goal_tolerance:
                return LimbMPCPolicy(
                    used + [STOP] * (self._model.horizon - len(used)),
                    {"num_rollouts": 1.0, "replay_error": error},
                    np.asarray(task.goal),
                    1,
                    [],
                    used,
                )
        raise PlanningFailure("The stored plan does not reach the goal")

    def _plan_for_training(
        self,
        task: ExerciseTask,
        belief: RangeOfMotionBelief,
        params: RecallParams,
        budget: Budget,
    ) -> LimbMPCPolicy:
        features = self._get_features(task, belief)
        key = tuple(int(v) for v in np.round(features * 1000))
        path = None
        if self._plan_dir is not None:
            self._plan_dir.mkdir(parents=True, exist_ok=True)
            path = self._plan_dir / ("_".join(map(str, key)) + ".npy")
            if path.exists():
                controls = np.load(path)
                if len(controls) == 0:
                    raise PlanningFailure("No training plan was found before")
                return self._as_policy(task, list(controls))
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
                    task,
                    belief,
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
        if path is not None:
            temporary = path.with_name(path.name + f".{id(self)}.tmp.npy")
            np.save(temporary, np.array(controls_found).reshape(-1, _NUM_JOINTS))
            temporary.replace(path)
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
