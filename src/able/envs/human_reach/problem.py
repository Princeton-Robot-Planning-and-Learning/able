"""The PDDLStream problem for the human reach environment."""

from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pddlstream.language.constants import AND, PDDLProblem
from pddlstream.utils import read
from pybullet_helpers.geometry import get_pose

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
from able.planners.pddlstream import solve_problem, stream_map

_PDDL = Path(__file__).parent / "pddl"
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
