"""A friction planner that replays stored push plans while awake."""

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.calendar import ExerciseTask
from able.envs.friction.model import NO_PUSH, FrictionModel, FrictionState, FrictionTask
from able.envs.friction.pomdp import GOAL, HOLE, PuckPushingModel
from able.envs.friction.world import Push
from able.models.base import ConditionalModel, Dataset
from able.models.exact_match_cache import ExactMatchCache
from able.planners.base import TrainablePlanner
from able.planners.pomcpow import FrictionPOMCPOWPlanner, POMCPOWConfig
from able.policies.base import Policy
from able.structs import Budget, PlanningFailure

# Stored plans are padded with NaN to this many pushes.
MAX_PLAN_PUSHES = 15
# Each push: heading, speed, stance, and where the puck stopped (x, y).
PUSH_DIM = 5


@dataclass(frozen=True, eq=False)
class RecallParams:
    """The plan cache, whether to search for a training plan instead of replaying a
    stored one, and how: POMCPOW simulations per push and attempts with new random
    numbers. A replayed push may turn by up to max_turn (rad) to aim at where its
    puck stopped when it was planned. With a tube (m), the plan is followed only
    while each push leaves the puck within the tube of where it stopped when
    planned. Robust training plans search each push under the whole training
    belief, rather than its mean frictions alone. With finish, a training plan
    takes a finishing shot (see finishing_shot()) whenever one stops the puck in
    the goal, and searches otherwise."""

    model: ConditionalModel = field(default_factory=ExactMatchCache)
    training: bool = False
    night_sims: int = 500
    night_attempts: int = 3
    max_turn: float = math.radians(20.0)
    tube: float | None = None
    robust: bool = False
    finish: bool = False


@dataclass(frozen=True)
class StoredPush:
    """A push of a stored plan, and where its puck stopped when it was planned."""

    angle: float
    speed: float
    stance: float | None
    stop: tuple[float, float]


def encode_plan(pushes: list[StoredPush]) -> NDArray[np.float64]:
    """The plan as one row, padded with NaN."""
    row = np.full((MAX_PLAN_PUSHES, PUSH_DIM), np.nan)
    for i, p in enumerate(pushes[:MAX_PLAN_PUSHES]):
        stance = np.nan if p.stance is None else p.stance
        row[i] = [p.angle, p.speed, stance, p.stop[0], p.stop[1]]
    return row.reshape(-1)


def decode_plan(row: NDArray[np.float64]) -> list[StoredPush]:
    """The plan that encode_plan() made."""
    pushes = []
    for angle, speed, stance, x, y in row.reshape(MAX_PLAN_PUSHES, PUSH_DIM):
        if np.isnan(angle):
            break
        pushes.append(
            StoredPush(
                float(angle),
                float(speed),
                None if np.isnan(stance) else float(stance),
                (float(x), float(y)),
            )
        )
    return pushes


class FrictionReplayPolicy(Policy[FrictionState, Push]):
    """Replays a stored plan: push k goes at its stored speed, turned to stop as
    near as the believed frictions predict to where it stopped when planned. After
    the plan's pushes, or once the puck stops farther than the tube from where the
    plan's last push left it, it does nothing."""

    def __init__(
        self,
        planner: "FrictionRecallPlanner",
        pushes: list[StoredPush],
        frictions: NDArray[np.float64],
        goal: tuple[float, float],
        max_turn: float,
        tube: float | None = None,
    ) -> None:
        self.pushes = pushes
        self._planner = planner
        self._frictions = frictions
        self._goal = goal
        self._max_turn = max_turn
        self._tube = tube

    def get_action(self, state: FrictionState) -> Push:
        if state.ended or state.pushes >= len(self.pushes):
            return NO_PUSH
        if self._tube is not None and state.pushes > 0:
            planned = np.array(self.pushes[state.pushes - 1].stop)
            if np.linalg.norm(state.puck_array - planned) > self._tube:
                return NO_PUSH
        push = self._planner.aim(
            state.puck_array,
            self.pushes[state.pushes],
            self._frictions,
            self._goal,
            self._max_turn,
        )
        return NO_PUSH if push is None else push

    def get_planning_stats(self) -> dict[str, float]:
        return {"num_sims": 1.0, "plan_pushes": float(len(self.pushes))}


class FrictionRecallPlanner(
    TrainablePlanner[
        FrictionState, Push, FrictionTask, NDArray[np.float64], RecallParams
    ]
):
    """Plans by replaying the stored plan for the same task whose day is nearest.

    While awake it looks up the plan for the task's exercise and the belief's
    features and replays it under the believed frictions (their mean), each push
    turned toward where it stopped when planned. It returns the replay if the puck
    stops in the goal, which counts as one iteration of the budget. Otherwise
    planning fails, since nothing else is cheap enough.

    Training plans (see get_training_params()) run POMCPOW one push at a time under
    the training belief's mean frictions alone, and keep the pushes and where the
    puck stopped if they reach the goal and their replay does too, trying again
    with new random numbers up to night_attempts times. The random numbers come
    from the exercise and the belief's features, so a training plan does not depend
    on when it is made or on who makes it, and with plan_dir, plans are stored
    there and reused.

    get_features gives the model's input for a task under a belief, and starts with
    the exercise.
    """

    def __init__(
        self,
        model: FrictionModel,
        get_features: Callable[
            [FrictionTask, Belief[NDArray[np.float64]]], NDArray[np.float64]
        ],
        default_params: RecallParams,
        seed: int = 0,
        plan_dir: str | Path | None = None,
    ) -> None:
        self._model = model
        self._get_features = get_features
        self._default_params = default_params
        self._seed = seed
        self._plan_dir = None if plan_dir is None else Path(plan_dir)
        self.pomdp = PuckPushingModel(
            model.world, rewards=model.rewards, max_pushes=model.horizon
        )
        self._searcher = FrictionPOMCPOWPlanner(
            model, FrictionBeliefUpdater(model.world)
        )

    def get_default_params(self) -> RecallParams:
        return self._default_params

    def get_model(self, params: RecallParams) -> ConditionalModel:
        return params.model

    def set_model(self, params: RecallParams, model: ConditionalModel) -> RecallParams:
        return replace(params, model=model)

    def get_training_params(self, params: RecallParams) -> RecallParams:
        return replace(params, training=True)

    def get_plan_examples(
        self,
        task: FrictionTask,
        belief: Belief[NDArray[np.float64]],
        policy: Policy[FrictionState, Push],
    ) -> Dataset:
        assert isinstance(policy, FrictionReplayPolicy), "Need a plan from plan()"
        return Dataset(
            x=self._get_features(task, belief)[None],
            y=encode_plan(policy.pushes)[None],
            costs=np.zeros(1),
            groups=np.zeros(1, dtype=np.int64),
        )

    def plan(
        self,
        task: FrictionTask,
        belief: Belief[NDArray[np.float64]],
        params: RecallParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> FrictionReplayPolicy:
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        assert isinstance(task, ExerciseTask), "Need an exercise"
        if params.training:
            return self._plan_for_training(task, belief, params)
        return self._recall(task, belief, params, rng)

    def predict_stop(
        self,
        puck: NDArray[np.float64],
        push: Push,
        frictions: NDArray[np.float64],
        goal: tuple[float, float],
    ) -> tuple[NDArray[np.float64], int] | None:
        """Where a push leaves the puck under the frictions, and whether it ends in
        the goal (GOAL), a hole (HOLE) or neither (0), or None if it is infeasible."""
        world = self._model.world
        if not world.push_feasible(puck, push.angle, push.stance):
            return None
        station = world.push_station(puck, push.angle, push.stance)
        result = world.slide(
            world.release_point(puck, push.angle),
            push.speed,
            push.angle,
            frictions,
            robot=world.strike_end(station, puck, push.angle),
        )
        if result.hole is not None:
            return result.stop, HOLE
        if world.in_goal(result.stop, goal):
            return result.stop, GOAL
        return world.shed(result.stop, goal), 0

    def aim(
        self,
        puck: NDArray[np.float64],
        stored: StoredPush,
        frictions: NDArray[np.float64],
        goal: tuple[float, float],
        max_turn: float,
    ) -> Push | None:
        """The stored push at its speed, turned by up to max_turn so that the
        frictions stop the puck nearest where it stopped when planned, keeping clear
        of holes; or None if no turn is feasible."""
        target = np.array(stored.stop)
        stances = [stored.stance] + sorted(
            (float(s) for s in self.pomdp.stances),
            key=lambda s: abs(s - (stored.stance or 0.0)),
        )

        def score(turn: float) -> tuple[float, Push | None]:
            angle = stored.angle + turn
            for stance in stances:
                push = Push(angle, stored.speed, stance)
                outcome = self.predict_stop(puck, push, frictions, goal)
                if outcome is None:
                    continue
                stop, kind = outcome
                if kind == HOLE:
                    return math.inf, push
                return float(np.linalg.norm(stop - target)), push
            return math.inf, None

        def rank(cost: float, turn: float) -> tuple[float, float]:
            # Stops within a centimeter are equally good, and smaller turns better.
            return (round(cost, 2) if math.isfinite(cost) else math.inf, abs(turn))

        best: tuple[tuple[float, float], Push | None, float] = (
            (math.inf, math.inf),
            None,
            0.0,
        )
        coarse = np.radians(2.0)
        turns = np.arange(0.0, max_turn + 1e-9, coarse)
        for turn in np.concatenate([turns, -turns[1:]]):
            cost, push = score(float(turn))
            if push is not None and rank(cost, turn) < best[0]:
                best = (rank(cost, turn), push, float(turn))
        if best[1] is None:
            return None
        center = best[2]
        for turn in center + np.radians(np.arange(-2.0, 2.0 + 1e-9, 0.1)):
            if abs(turn) > max_turn + 1e-9:
                continue
            cost, push = score(float(turn))
            if push is not None and rank(cost, turn) < best[0]:
                best = (rank(cost, turn), push, float(turn))
        return best[1]

    def _recall(
        self,
        task: FrictionTask,
        belief: ParticleBelief,
        params: RecallParams,
        rng: np.random.Generator,
    ) -> FrictionReplayPolicy:
        """The stored plan for the nearest day, if it replays to the goal."""
        proposals = params.model.sample(self._get_features(task, belief), 1, rng)
        if len(proposals) == 0:
            raise PlanningFailure("No stored plan for the exercise", {"num_sims": 1.0})
        pushes = decode_plan(proposals[0])
        frictions = np.asarray(belief.mean)
        if not self.replays(task, pushes, frictions, params):
            raise PlanningFailure(
                "The stored plan does not reach the goal", {"num_sims": 1.0}
            )
        return FrictionReplayPolicy(
            self, pushes, frictions, task.goal, params.max_turn, params.tube
        )

    def replays(
        self,
        task: FrictionTask,
        pushes: list[StoredPush],
        frictions: NDArray[np.float64],
        params: RecallParams,
    ) -> bool:
        """Whether the plan, each push aimed (see aim()), stops the puck in the goal
        under the frictions, staying within the tube if there is one."""
        puck = task.initial_state.puck_array
        for stored in pushes:
            push = self.aim(puck, stored, frictions, task.goal, params.max_turn)
            outcome = (
                None
                if push is None
                else self.predict_stop(puck, push, frictions, task.goal)
            )
            if outcome is None or outcome[1] == HOLE:
                return False
            puck = outcome[0]
            if outcome[1] == GOAL:
                return True
            off = np.linalg.norm(puck - np.array(stored.stop))
            if params.tube is not None and off > params.tube:
                return False
        return False

    def _plan_path(self, key: tuple[int, ...]) -> Path | None:
        if self._plan_dir is None:
            return None
        return self._plan_dir / f"plan_{'_'.join(str(k) for k in key)}.json"

    def _plan_for_training(
        self, task: ExerciseTask, belief: ParticleBelief, params: RecallParams
    ) -> FrictionReplayPolicy:
        frictions = np.asarray(np.round(belief.mean, 6), dtype=np.float64)
        features = self._get_features(task, belief)
        key = tuple(int(v) for v in np.round(features * 1000))
        path = self._plan_path(key)
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["pushes"] is None:
                raise PlanningFailure("No training plan reached the goal")
            pushes = [
                StoredPush(p[0], p[1], p[2], (p[3], p[4])) for p in data["pushes"]
            ]
            return FrictionReplayPolicy(
                self, pushes, frictions, task.goal, params.max_turn
            )
        found: list[StoredPush] | None = None
        for attempt in range(params.night_attempts):
            rng = np.random.default_rng(
                [self._seed, attempt, *(v + 10**6 for v in key)]
            )
            found = self._search(
                task,
                frictions,
                params.night_sims,
                rng,
                belief if params.robust else None,
                params.finish,
            )
            if found is not None and self.replays(task, found, frictions, params):
                break
            found = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = (
                None
                if found is None
                else [[p.angle, p.speed, p.stance, p.stop[0], p.stop[1]] for p in found]
            )
            tmp = path.with_suffix(f".{np.random.default_rng().integers(2**31)}.tmp")
            tmp.write_text(json.dumps({"pushes": rows}), encoding="utf-8")
            tmp.replace(path)
        if found is None:
            raise PlanningFailure("No training plan reached the goal")
        return FrictionReplayPolicy(self, found, frictions, task.goal, params.max_turn)

    def finishing_shot(
        self,
        puck: NDArray[np.float64],
        frictions: NDArray[np.float64],
        goal: tuple[float, float],
    ) -> Push | None:
        """The push that the frictions stop nearest the goal's center, among those
        within 30 degrees of the goal's direction, if any stops in the goal."""
        phys = self._model.world.physics
        toward = math.atan2(goal[1] - puck[1], goal[0] - puck[0])
        best: tuple[float, Push | None] = (math.inf, None)
        for turn in np.radians(np.arange(-30.0, 30.0 + 1e-9, 1.0)):
            for speed in np.linspace(phys.min_speed, phys.max_speed, 60):
                push = Push(round(toward + float(turn), 4), round(float(speed), 4))
                outcome = self.predict_stop(puck, push, frictions, goal)
                if outcome is None or outcome[1] != GOAL:
                    continue
                miss = float(np.linalg.norm(outcome[0] - np.asarray(goal)))
                if miss < best[0]:
                    best = (miss, push)
        return best[1]

    def _search(
        self,
        task: FrictionTask,
        frictions: NDArray[np.float64],
        num_sims: int,
        rng: np.random.Generator,
        belief: ParticleBelief | None = None,
        finish: bool = False,
    ) -> list[StoredPush] | None:
        """POMCPOW's feasible pushes under the frictions alone, or under the belief
        if it is given, and where the analytic slide (predict_stop()) says each
        leaves the puck under the frictions, if they reach the goal within the
        horizon. With finish, a finishing shot replaces the search when there is
        one. A push planned from a measured position may be infeasible from the true
        one and is left out."""
        horizon = self._model.horizon
        config = POMCPOWConfig(num_sims=num_sims, max_depth=horizon)
        if belief is None:
            belief = ParticleBelief(frictions[None], frictions * 0.5, frictions * 2.0)
        state = task.initial_state
        pushes: list[StoredPush] = []
        try:
            while not state.ended and state.pushes < horizon:
                push = None
                if finish:
                    push = self.finishing_shot(
                        np.array(state.true_puck), frictions, task.goal
                    )
                if push is None:
                    push, _ = self._searcher.search(state, belief, config, rng)
                if push is None:
                    return None
                predicted = self.predict_stop(
                    np.array(state.true_puck), push, frictions, task.goal
                )
                state = self._model.sample_next_state(state, push, frictions, rng)
                record = state.last_push
                if predicted is not None and record is not None and record.feasible:
                    stop = (float(predicted[0][0]), float(predicted[0][1]))
                    pushes.append(StoredPush(push.angle, push.speed, push.stance, stop))
        except PlanningFailure:
            return None
        return pushes if state.outcome == "goal" else None
