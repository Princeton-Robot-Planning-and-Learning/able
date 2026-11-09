"""Solving PDDLStream problems, with each stream evaluation counted, and the
PDDLStream problem and planner for the human reach environment."""

import contextlib
import ctypes
import dataclasses
import gc
import io
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import AND, PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read
from pybullet_helpers.geometry import get_pose

from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import (
    NO_ACTION,
    ReachAction,
    ReachModel,
    ReachState,
    ReachTask,
)
from able.envs.human_reach.scene import (
    BedsideStocking3DEnvConfig,
    BedsideStocking3DObjectCentricState,
    ObjectCentricBedsideStocking3DEnv,
)
from able.envs.human_reach.streams import (
    PLACEMENTS,
    BaseConf,
    BedsideStreamContext,
    Placement,
    inverse_kinematics_stream,
    requested_candidate_slots,
    sample_continuous_place_pose,
    sample_grasp,
    sample_place_pose,
    test_cfree_pose_pose,
    test_cfree_traj_pose,
    test_clear,
    test_usable,
)
from able.models.base import ConditionalModel, Dataset
from able.models.flow_matching import FlowMatchingModel
from able.planners.base import TrainablePlanner
from able.policies.base import OpenLoopPolicy, Policy
from able.structs import Budget, PlanningFailure, PlanningTimeout


@contextmanager
def private_cwd() -> Iterator[None]:
    """Run in a fresh temp dir: Fast Downward writes ./temp in the cwd, so parallel
    runs would collide."""
    old, tmp = os.getcwd(), tempfile.mkdtemp(prefix="pddlstream_")
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)


class OutOfMemory(Exception):
    """Raised by a stream when the solve in progress has grown past its memory
    limit."""


class OutOfStreamCalls(Exception):
    """Raised by a stream when the solve in progress has used all its stream
    evaluations."""


# Streams check the process's memory every this many evaluations.
MEMORY_CHECK_EVERY = 64
# The solve in progress's limit: (resident bytes at its start, growth allowed).
_MEMORY_LIMIT: tuple[int, int] | None = None
# The most stream evaluations the solve in progress may make, or None.
_CALL_LIMIT: int | None = None


def get_resident_bytes() -> int | None:
    """This process's resident memory, or None where /proc is unavailable."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


@contextmanager
def memory_limit(max_growth: int | None) -> Iterator[None]:
    """Streams evaluated inside raise OutOfMemory once this process's resident
    memory has grown more than max_growth bytes (None for no limit).

    PDDLStream's own max_memory measures virtual memory, which libraries such as
    torch make large from the start, so it cannot serve here.
    """
    global _MEMORY_LIMIT  # pylint: disable=global-statement
    previous = _MEMORY_LIMIT
    start = None if max_growth is None else get_resident_bytes()
    _MEMORY_LIMIT = None if max_growth is None or start is None else (start, max_growth)
    try:
        yield
    finally:
        _MEMORY_LIMIT = previous


@contextmanager
def call_limit(max_calls: int | None) -> Iterator[None]:
    """Streams evaluated inside raise OutOfStreamCalls instead of making more than
    max_calls evaluations in all (None for no limit).

    Unlike a time limit, this caps the planner's work the same way on any machine
    and under any load.
    """
    global _CALL_LIMIT  # pylint: disable=global-statement
    previous = _CALL_LIMIT
    _CALL_LIMIT = max_calls
    try:
        yield
    finally:
        _CALL_LIMIT = previous


def release_memory() -> None:
    """Return freed memory to the operating system.

    glibc keeps the heap pages that a large solve freed, so without this the
    process stays as large as its largest solve, and each memory limit starts
    from there: across solves, a worker's memory climbs until it is killed. A
    no-op where glibc's malloc_trim is unavailable.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def count_evaluation(calls: Counter, name: str) -> None:
    """Count an evaluation of stream `name`, unless the stream-call limit is used
    up, and check the memory limit every MEMORY_CHECK_EVERY evaluations."""
    if _CALL_LIMIT is not None and calls.total() >= _CALL_LIMIT:
        raise OutOfStreamCalls(f"Used all {_CALL_LIMIT} stream calls")
    calls[name] += 1
    if _MEMORY_LIMIT is None or calls.total() % MEMORY_CHECK_EVERY:
        return
    resident = get_resident_bytes()
    start, max_growth = _MEMORY_LIMIT
    if resident is not None and resident - start > max_growth:
        raise OutOfMemory(f"Memory grew by {(resident - start) / 1e9:.1f} GB")


def counted(fn: Callable, calls: Counter, name: str, generator: bool) -> Callable:
    """`fn` with each evaluation of stream `name` counted in `calls`, which raises
    OutOfStreamCalls if the solve has used all its stream calls, or OutOfMemory if
    it has grown past its memory limit."""

    def test(*args):
        count_evaluation(calls, name)
        return fn(*args)

    def gen(*args):
        outputs = fn(*args)
        while True:
            count_evaluation(calls, name)
            try:
                out = next(outputs)
            except StopIteration:
                return
            yield out

    return gen if generator else test


def stream_map(
    gens: dict[str, Callable], tests: dict[str, Callable], calls: Counter
) -> dict:
    """PDDLStream stream map with every evaluation counted."""
    out = {n: from_gen_fn(counted(f, calls, n, True)) for n, f in gens.items()}
    return out | {n: from_test(counted(f, calls, n, False)) for n, f in tests.items()}


def solve_problem(
    problem: PDDLProblem,
    max_time: float,
    algorithm: str = "adaptive",
    search_sample_ratio: float = 1.0,
    max_complexity: float = float("inf"),
    verbose: bool = False,
):
    """The plan (list of actions) or None."""
    # pddlstream walks its binding tree recursively; long plans exceed the default limit.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    output = contextlib.nullcontext() if verbose else redirect_stdout(io.StringIO())
    with private_cwd(), output:
        solution = solve(
            problem,
            algorithm=algorithm,
            unit_costs=True,
            max_time=max_time,
            search_sample_ratio=search_sample_ratio,
            max_complexity=max_complexity,
            verbose=verbose,
        )
    if verbose:
        print_solution(solution)
    return solution[0]


_PDDL = Path(__file__).parents[1] / "envs" / "human_reach" / "pddl"
DOMAIN_PDDL = read(str(_PDDL / "domain.pddl"))
STREAM_PDDL = read(str(_PDDL / "stream.pddl"))


def _surface_of(sim: ObjectCentricBedsideStocking3DEnv, name: str) -> str:
    body = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
    # pylint: disable-next=protected-access
    supporting = sim._get_surfaces_supporting_object(body)
    for s in sim.config.surface_regions:
        if supporting & {sim.surface_id(s)}:
            return s
    return "unknown"


def build_stream_context(
    sim: ObjectCentricBedsideStocking3DEnv,
    state: BedsideStocking3DObjectCentricState,
    theta_particles: np.ndarray | None = None,
    delta: float = 0.1,
    seed: int = 0,
    proposal=None,
    placement: str = "slots",
) -> BedsideStreamContext:
    """The streams' shared context for planning in the state, with the reach
    particles (the true reach if none), an optional placement proposal, and how
    placements are sampled (one of PLACEMENTS)."""
    assert placement in PLACEMENTS, f"Unknown placement: {placement}"
    if theta_particles is None:
        theta_particles = sim.reach_theta[None]
    cfg = sim.config
    surfaces = list(cfg.surface_regions)
    return BedsideStreamContext(
        sim=sim,
        theta_particles=np.asarray(theta_particles, dtype=float),
        delta=delta,
        rng=np.random.default_rng(seed),
        requested=state.requested_objects,
        bedside_surfaces=list(cfg.bedside_regions),
        all_surfaces=surfaces,
        slots={s: cfg.region_slots(s) for s in surfaces},
        requested_slots=(
            requested_candidate_slots(sim) if placement == "slots" else []
        ),
        proposal=proposal,
        placement=placement,
    )


def create_problem(
    sim: ObjectCentricBedsideStocking3DEnv,
    state: BedsideStocking3DObjectCentricState,
    ctx: BedsideStreamContext | None = None,
    **ctx_kwargs,
) -> PDDLProblem:
    """The problem: deliver every requested item to a usable pose that no other
    item blocks, with the hand empty."""
    if ctx is None:
        ctx = build_stream_context(sim, state, **ctx_kwargs)
    q0 = BaseConf(sim.robot.get_base())
    init: list[tuple] = [("BConf", q0), ("AtBConf", q0), ("HandEmpty",), ("CanMove",)]
    goal: list[tuple] = [("HandEmpty",)]
    for name in state.requested_objects + state.clutter_objects:
        body = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
        p0 = Placement(
            name, get_pose(body, sim.physics_client_id), _surface_of(sim, name)
        )
        init += [("Item", name), ("Pose", name, p0), ("AtPose", name, p0)]
        if name in ctx.requested:
            init.append(("Requested", name))
            goal.append(("Delivered", name))
    gens: dict[str, Callable[..., Any]] = {
        "sample-place-pose": (
            sample_place_pose
            if ctx.placement == "slots"
            else sample_continuous_place_pose
        ),
        "sample-grasp": sample_grasp,
        "inverse-kinematics": inverse_kinematics_stream,
    }
    tests: dict[str, Callable[..., Any]] = {
        "test-usable": test_usable,
        "test-clear": test_clear,
        "test-cfree-pose-pose": test_cfree_pose_pose,
        "test-cfree-traj-pose": test_cfree_traj_pose,
    }
    streams = stream_map(
        {n: partial(f, ctx) for n, f in gens.items()},
        {n: partial(f, ctx) for n, f in tests.items()},
        ctx.calls,
    )
    return PDDLProblem(DOMAIN_PDDL, {}, STREAM_PDDL, streams, init, (AND, *goal))


def plan_bedside(
    sim: ObjectCentricBedsideStocking3DEnv,
    state: BedsideStocking3DObjectCentricState,
    max_time: float = 600.0,
    algorithm: str = "adaptive",
    search_sample_ratio: float = 1.0,
    max_complexity: float = float("inf"),
    verbose: bool = False,
    ctx: BedsideStreamContext | None = None,
    **ctx_kwargs,
):
    """Plan under `ctx` (built from `ctx_kwargs` if None); None if no plan is found."""
    problem = create_problem(sim, state, ctx=ctx, **ctx_kwargs)
    return solve_problem(
        problem, max_time, algorithm, search_sample_ratio, max_complexity, verbose
    )


# The largest number of requested items, which sets the item one-hot's size.
MAX_REQUESTED = 8


@dataclass(frozen=True, eq=False)
class PDDLStreamParams:
    """Parameters of the human reach PDDLStream planner.

    A requested item's pose is usable if at least 1 - delta of num_particles belief
    particles reach it. Planning stops after max_time seconds, or once a solve has
    grown the process's resident memory by max_memory_gb, if set. With a model,
    the guided try gets at most guided_budget_fraction of the time limit (the
    planning budget's wall time if set, else max_time) and of the planning
    budget's stream calls, so that a poor ranking leaves the fallback without the
    model the budget to plan as scratch would. The
    model proposes num_draws placements per item, requested or clutter, from
    get_place_features(). With "slots" placement, they rank the item's candidate
    slots; with "continuous", they are the placements tried first, anywhere on a
    surface, before uniform samples. An untrained model proposes nothing.
    """

    max_time: float = 300.0
    num_particles: int = 256
    delta: float = 0.1
    # The most a solve may grow this process's resident memory (GB), or None.
    max_memory_gb: float | None = None
    # The share of the time limit and of the stream calls the model-guided try may
    # use; the fallback without the model gets the rest.
    guided_budget_fraction: float = 1.0
    algorithm: str = "adaptive"
    model: ConditionalModel = field(default_factory=FlowMatchingModel)
    num_draws: int = 16
    # How placements are sampled: "slots" or "continuous" (see PLACEMENTS).
    placement: str = "slots"


def get_bedside_slots(config: BedsideStocking3DEnvConfig) -> NDArray[np.float64]:
    """Every bedside slot's (x, y, top z)."""
    return np.concatenate([config.region_slots(r) for r in config.bedside_regions])


def get_place_features(
    scene: BedsideStocking3DObjectCentricState,
    config: BedsideStocking3DEnvConfig,
    belief: ParticleBelief,
    item: str,
) -> NDArray[np.float64]:
    """The model's input for an item: which requested item it is (one-hot, all
    zeros for clutter), which bedside slots the clutter occupies, and the belief's
    mean and standard deviation."""
    onehot = np.zeros(MAX_REQUESTED)
    if item in scene.requested_objects:
        onehot[int(item.removeprefix("req"))] = 1.0
    slots = get_bedside_slots(config)
    occupied = np.zeros(len(slots))
    clutter = [scene.get_object_pose(c).position[:2] for c in scene.clutter_objects]
    if clutter:
        gap = np.abs(slots[:, None, :2] - np.array(clutter)[None]).max(-1)
        occupied = (gap < config.slot_spacing / 2).any(1).astype(float)
    return np.concatenate([onehot, occupied, belief.mean, belief.std])


def get_final_placements(
    plan: Sequence[ReachAction],
) -> dict[str, tuple[float, float, float]]:
    """Where the plan places each item last."""
    final = {}
    for name, args in plan:
        if name == "place":
            x, y, z = args[1].pose.position
            final[args[0]] = (float(x), float(y), float(z))
    return final


@dataclass(frozen=True)
class PlaceProposal:
    """Model draws of each item's placement, requested or clutter, which rank its
    candidate slots, or are its first placements when placement is continuous."""

    samples: dict[str, NDArray[np.float64]]  # item -> (k, 3)

    def points(self, o: str) -> NDArray[np.float64]:
        """The draws for `o`, (k, 3), or none."""
        return self.samples.get(o, np.empty((0, 3)))

    def order(
        self, o: str, points: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.int64]:
        """Slots nearest a draw, most drawn first, then the rest at random."""
        if o not in self.samples:
            return rng.permutation(len(points))
        near = np.linalg.norm(self.samples[o][:, None] - points[None], axis=-1).argmin(
            1
        )
        ids, first, counts = np.unique(near, return_index=True, return_counts=True)
        ranked = ids[np.lexsort((first, -counts))]
        rest = rng.permutation(np.setdiff1d(np.arange(len(points)), ranked))
        return np.concatenate([ranked, rest]).astype(np.int64)


class ReachPDDLStreamPlanner(
    TrainablePlanner[
        ReachState, ReachAction, ReachTask, NDArray[np.float64], PDDLStreamParams
    ]
):
    """Plans the whole human reach task with PDDLStream, in a simulator of its own.

    The policy executes the plan's actions, then NO_ACTION until the horizon; a
    plan longer than the horizon fails. The model in the parameters proposes where
    each item, requested or clutter, is placed first; if planning with it fails,
    the planner plans again without it. The planning budget caps all planning, both
    tries together: its wall time, if set, and its iterations, if set, which count
    stream evaluations. The statistics are the number of stream evaluations, the
    plan's length, whether the model guided the plan, whether the guided try ran out
    of its budget (time or stream calls), and whether it fell back; a failed plan
    raises with them. The default parameters, if given, replace
    PDDLStreamParams().
    """

    def __init__(
        self, model: ReachModel, default_params: PDDLStreamParams | None = None
    ) -> None:
        self._model = model
        self._default_params = (
            PDDLStreamParams() if default_params is None else default_params
        )
        self._sim: ObjectCentricBedsideStocking3DEnv | None = None

    @property
    def sim(self) -> ObjectCentricBedsideStocking3DEnv:
        """The planner's simulator, built on first use."""
        if self._sim is None:
            model = self._model
            self._sim = ObjectCentricBedsideStocking3DEnv(
                model.num_requested,
                model.num_clutter,
                config=model.config,
                allow_state_access=True,
            )
        return self._sim

    def close(self) -> None:
        """Close the planner's simulator."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def get_default_params(self) -> PDDLStreamParams:
        return self._default_params

    def get_model(self, params: PDDLStreamParams) -> ConditionalModel:
        return params.model

    def set_model(
        self, params: PDDLStreamParams, model: ConditionalModel
    ) -> PDDLStreamParams:
        return dataclasses.replace(params, model=model)

    def get_plan_examples(
        self,
        task: ReachTask,
        belief: Belief[NDArray[np.float64]],
        policy: Policy[ReachState, ReachAction],
    ) -> Dataset:
        """One row per item the plan places, requested or clutter: its features
        under the belief, and where the plan places it last."""
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        assert isinstance(policy, OpenLoopPolicy), "Need a plan from plan()"
        final = get_final_placements(policy.actions)
        scene, config = task.initial_state.scene, self._model.config
        objects = [*scene.requested_objects, *scene.clutter_objects]
        items = [o for o in objects if o in final]
        if not items:
            x_dim = MAX_REQUESTED + len(get_bedside_slots(config)) + 6
            return Dataset.empty(x_dim, 3)
        return Dataset(
            x=np.array([get_place_features(scene, config, belief, o) for o in items]),
            y=np.array([final[o] for o in items]),
            costs=np.zeros(len(items)),
            groups=np.zeros(len(items), dtype=np.int64),
        )

    def plan(
        self,
        task: ReachTask,
        belief: Belief[NDArray[np.float64]],
        params: PDDLStreamParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> OpenLoopPolicy[ReachState, ReachAction]:
        """Plan under num_particles of the belief's particles.

        Raises PlanningFailure if PDDLStream finds no plan, with and without the
        model, or the last try ran out of memory, and PlanningTimeout if it ran out
        of time or stream calls.
        """
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        start_time = time.perf_counter()
        particles = belief.particles
        if len(particles) > params.num_particles:
            keep = rng.choice(len(particles), params.num_particles, replace=False)
            particles = particles[keep]
        proposal = self._propose(task, belief, params, rng)
        stats = {"stream_calls": 0.0, "seeded": float(proposal is not None)}
        stats["fell_back"] = 0.0
        stats["seeded_timed_out"] = 0.0
        plan, stopped_by = self._solve(
            task, particles, params, budget, rng, proposal, stats, start_time
        )
        if plan is None and proposal is not None:
            stats["seeded_timed_out"] = float(stopped_by in ("time", "calls"))
            stats["fell_back"] = 1.0
            plan, stopped_by = self._solve(
                task, particles, params, budget, rng, None, stats, start_time
            )
        if plan is None and stopped_by == "time":
            raise PlanningTimeout(
                "PDDLStream found no plan within its time limit", stats
            )
        if plan is None and stopped_by == "calls":
            raise PlanningTimeout(
                "PDDLStream found no plan within its stream calls", stats
            )
        if plan is None and stopped_by == "memory":
            raise PlanningFailure("PDDLStream ran out of memory", stats)
        if plan is None:
            raise PlanningFailure("PDDLStream found no plan", stats)
        horizon = self._model.horizon
        if len(plan) > horizon:
            raise PlanningFailure("The plan is longer than the horizon", stats)
        stats["plan_length"] = len(plan)
        actions = [tuple(action) for action in plan]
        return OpenLoopPolicy(actions + [NO_ACTION] * (horizon - len(plan)), stats)

    def _propose(
        self,
        task: ReachTask,
        belief: ParticleBelief,
        params: PDDLStreamParams,
        rng: np.random.Generator,
    ) -> PlaceProposal | None:
        """The model's placements for each item, requested or clutter, or None if
        it proposes nothing."""
        scene, config = task.initial_state.scene, self._model.config
        samples = {}
        for item in [*scene.requested_objects, *scene.clutter_objects]:
            x = get_place_features(scene, config, belief, item)
            draws = params.model.sample(x, params.num_draws, rng)
            if len(draws):
                samples[item] = draws
        return PlaceProposal(samples) if samples else None

    def _solve(
        self,
        task: ReachTask,
        particles: NDArray[np.float64],
        params: PDDLStreamParams,
        budget: Budget,
        rng: np.random.Generator,
        proposal: PlaceProposal | None,
        stats: dict[str, float],
        start_time: float,
    ) -> tuple[list | None, str | None]:
        """The plan or None, and what stopped a search without one: "time" for its
        time limit (PDDLStream only gives up once its time is used up), "calls" for
        the planning budget's stream calls, "memory" for its memory limit, or
        None."""
        max_time = params.max_time
        if budget.max_wall_time is not None:
            left = budget.max_wall_time - (time.perf_counter() - start_time)
            if left <= 0:
                return None, "time"
            max_time = min(max_time, left)
        max_calls = budget.max_iterations
        if max_calls is not None:
            max_calls -= int(stats["stream_calls"])
            if max_calls <= 0:
                return None, "calls"
        if proposal is not None:
            fraction = params.guided_budget_fraction
            limit = params.max_time
            if budget.max_wall_time is not None:
                limit = min(limit, budget.max_wall_time)
            max_time = min(max_time, fraction * limit)
            if max_calls is not None and budget.max_iterations is not None:
                max_calls = min(max_calls, int(fraction * budget.max_iterations))
        sim, scene = self.sim, task.initial_state.scene
        sim.reset(options={"init_state": scene})
        ctx = build_stream_context(
            sim,
            scene,
            theta_particles=particles,
            delta=params.delta,
            seed=int(rng.integers(2**31 - 1)),
            proposal=proposal,
            placement=params.placement,
        )
        solve_start = time.perf_counter()
        max_growth = (
            None if params.max_memory_gb is None else int(params.max_memory_gb * 1e9)
        )
        try:
            with memory_limit(max_growth), call_limit(max_calls):
                plan = solve_problem(
                    create_problem(sim, scene, ctx=ctx), max_time, params.algorithm
                )
        except OutOfMemory:
            stats["stream_calls"] += float(sum(ctx.calls.values()))
            return None, "memory"
        except OutOfStreamCalls:
            stats["stream_calls"] += float(sum(ctx.calls.values()))
            return None, "calls"
        finally:
            release_memory()
        stats["stream_calls"] += float(sum(ctx.calls.values()))
        if plan is None and time.perf_counter() - solve_start >= max_time:
            return None, "time"
        return plan, None
