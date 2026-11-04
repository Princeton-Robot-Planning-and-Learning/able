"""Solving PDDLStream problems, with each stream evaluation counted, and the
PDDLStream problem and planner for the human reach environment."""

import contextlib
import io
import os
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

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
    BedsideStocking3DObjectCentricState,
    ObjectCentricBedsideStocking3DEnv,
)
from able.envs.human_reach.streams import (
    BaseConf,
    BedsideStreamContext,
    Placement,
    inverse_kinematics_stream,
    requested_candidate_slots,
    sample_grasp,
    sample_place_pose,
    test_cfree_pose_pose,
    test_cfree_traj_pose,
    test_clear,
    test_usable,
)
from able.planners.base import Planner
from able.policies.base import OpenLoopPolicy
from able.structs import Budget, PlanningFailure


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


def counted(fn: Callable, calls: Counter, name: str, generator: bool) -> Callable:
    """`fn` with each evaluation of stream `name` counted in `calls`."""

    def test(*args):
        calls[name] += 1
        return fn(*args)

    def gen(*args):
        outputs = fn(*args)
        while True:
            calls[name] += 1
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
) -> BedsideStreamContext:
    """The streams' shared context for planning in the state, with the reach
    particles (the true reach if none) and an optional placement proposal."""
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
        requested_slots=requested_candidate_slots(sim),
        proposal=proposal,
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
        "sample-place-pose": sample_place_pose,
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


@dataclass(frozen=True)
class PDDLStreamParams:
    """Parameters of the human reach PDDLStream planner.

    A requested item's pose is usable if at least 1 - delta of num_particles belief
    particles reach it. Planning stops after max_time seconds.
    """

    max_time: float = 300.0
    num_particles: int = 256
    delta: float = 0.1
    algorithm: str = "adaptive"


class ReachPDDLStreamPlanner(
    Planner[ReachState, ReachAction, ReachTask, NDArray[np.float64], PDDLStreamParams]
):
    """Plans the whole human reach task with PDDLStream, in a simulator of its own.

    The policy executes the plan's actions, then NO_ACTION until the horizon; a
    plan longer than the horizon fails. The planning budget's wall time, if set,
    caps max_time. The statistics are the number of stream evaluations and the
    plan's length. The default parameters, if given, replace PDDLStreamParams().
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

    def plan(
        self,
        task: ReachTask,
        belief: Belief[NDArray[np.float64]],
        params: PDDLStreamParams,
        budget: Budget,
        rng: np.random.Generator,
    ) -> OpenLoopPolicy[ReachState, ReachAction]:
        """Plan under num_particles of the belief's particles.

        Raises PlanningFailure if PDDLStream finds no plan in time.
        """
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        particles = belief.particles
        if len(particles) > params.num_particles:
            keep = rng.choice(len(particles), params.num_particles, replace=False)
            particles = particles[keep]
        max_time = params.max_time
        if budget.max_wall_time is not None:
            max_time = min(max_time, budget.max_wall_time)
        sim, scene = self.sim, task.initial_state.scene
        sim.reset(options={"init_state": scene})
        ctx = build_stream_context(
            sim,
            scene,
            theta_particles=particles,
            delta=params.delta,
            seed=int(rng.integers(2**31 - 1)),
        )
        plan = solve_problem(
            create_problem(sim, scene, ctx=ctx), max_time, params.algorithm
        )
        if plan is None:
            raise PlanningFailure("PDDLStream found no plan")
        horizon = self._model.horizon
        if len(plan) > horizon:
            raise PlanningFailure("The plan is longer than the horizon")
        stats = {
            "stream_calls": float(sum(ctx.calls.values())),
            "plan_length": len(plan),
        }
        actions = [tuple(action) for action in plan]
        return OpenLoopPolicy(actions + [NO_ACTION] * (horizon - len(plan)), stats)
