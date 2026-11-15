"""PDDLStream streams for the human reach environment.

Requested items are usable at a pose if, under the belief's particles, the person
reaches it with probability at least 1 - delta. Planning is IK-only: carry, hover,
and grasp waypoints and unchecked base moves, which execution refines.

Placements come from one of two streams. With "slots", items try the centers of a
grid of slots on each surface, jittered, so a model can only rank the slots. With
"continuous", items are placed anywhere on a surface: the model's draws are the
placements, and uniform samples follow them.
"""

from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot

from able.envs.human_reach.scene import (
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
    segment_blocked,
)

# kinder_models' HOME_JOINT_POSITIONS for the TidyBot arm.
HOME_JOINTS = np.deg2rad([0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]).tolist()
CARRY_JOINTS = [0.0, -0.605, np.pi, -1.635, 0.0, -2.112, np.pi / 2] + HOME_JOINTS[7:]
GRASP_ORIENTATIONS = [(0.707, 0.707, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
GRASP_BELOW_TOP = 0.005
HOVER_HEIGHT = 0.08
BASE_DISTANCES = (0.45, 0.55, 0.65, 0.75)
BASE_ROTATIONS = (0.0, 0.4, -0.4)
CORRIDOR_X = 1.5
POSE_JITTER = 0.015
MAX_PLACE_POSE_ATTEMPTS = 100_000
# Draws per uniform sample of a requested item's placement, which keeps it within
# the largest reach the prior allows.
MAX_WITHIN_REACH_DRAWS = 1000
# How placements are sampled; see the module docstring.
PLACEMENTS = ("slots", "continuous")


@dataclass(eq=False)
class Placement:
    """An item pose; identity-hashed so streams never emit equal values."""

    obj: str
    pose: Pose
    surface: str


@dataclass(eq=False)
class Grasp:
    """An item-specific grasp transform; identity-hashed."""

    obj: str
    transform: Pose


@dataclass(eq=False)
class BaseConf:
    """A base pose; identity-hashed."""

    pose: SE2Pose


@dataclass(eq=False)
class ArmTrajectory:
    """Carry -> hover -> grasp joint waypoints for one target item."""

    joint_plan: list
    target_object: str
    base: SE2Pose


@dataclass
class BedsideStreamContext:
    """Shared context for all streams of one problem."""

    sim: ObjectCentricBedsideStocking3DEnv
    theta_particles: NDArray
    delta: float
    rng: np.random.Generator
    requested: list[str]
    bedside_surfaces: list[str]
    all_surfaces: list[str]
    slots: dict[str, NDArray] = field(default_factory=dict)
    requested_slots: list[tuple[str, NDArray]] = field(default_factory=list)
    # psi hook: with slots, `.order(o, points, rng)` ranks placements; with
    # continuous placements, `.points(o)` are the placements to try first.
    proposal: Any = None
    calls: Counter = field(default_factory=Counter)  # stream evaluations by name
    placement: str = "slots"  # one of PLACEMENTS

    def slot_points(self, o: str) -> NDArray[np.float64]:
        """Resting position of `o` at each requested slot."""
        return np.array(
            [
                self.sim.resting_pose(o, tuple(slot[:2]), s).position
                for s, slot in self.requested_slots
            ]
        )

    def place_order(self, o: str) -> NDArray[np.int64]:
        """Order in which `o` tries the requested slots: psi's ranking, else random."""
        if self.proposal is None:
            return self.rng.permutation(len(self.requested_slots))
        return np.asarray(self.proposal.order(o, self.slot_points(o), self.rng))

    def clutter_order(
        self, o: str, candidates: list[tuple[str, NDArray]]
    ) -> NDArray[np.int64]:
        """Order in which clutter `o` tries the candidate slots, given as (surface,
        xy): psi's ranking of their resting positions, else random."""
        if self.proposal is None:
            return self.rng.permutation(len(candidates))
        points = np.array(
            [self.sim.resting_pose(o, tuple(xy), s).position for s, xy in candidates]
        )
        return np.asarray(self.proposal.order(o, points, self.rng))

    def proposed_points(self, o: str) -> NDArray[np.float64]:
        """psi's draws of where `o` goes, (k, 3), or none without a proposal."""
        if self.proposal is None:
            return np.empty((0, 3))
        return np.asarray(self.proposal.points(o), dtype=float).reshape(-1, 3)

    def surface_box(self, s: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The lowest and highest (x, y) of an item's center on surface `s`: half a
        slot in from each edge, the area the slot grid's centers span."""
        (x0, x1), (y0, y1), _ = self.sim.config.surface_regions[s]
        margin = self.sim.config.slot_spacing / 2
        return np.array([x0 + margin, y0 + margin]), np.array(
            [x1 - margin, y1 - margin]
        )

    def placement_surfaces(self, o: str) -> list[str]:
        """The surfaces `o` may go on: the bedside for requested items, where they
        are usable, and every surface for clutter."""
        return self.bedside_surfaces if o in self.requested else self.all_surfaces


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@contextmanager
def _scratch_pose(sim: ObjectCentricBedsideStocking3DEnv, name: str, pose: Pose):
    body = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
    original = get_pose(body, sim.physics_client_id)
    set_pose(body, pose, sim.physics_client_id)
    try:
        yield
    finally:
        set_pose(body, original, sim.physics_client_id)


# Far from the scene: where _alone() puts every body it moves out of the way.
_AWAY = (50.0, 50.0, -50.0)


@contextmanager
def _alone(sim: ObjectCentricBedsideStocking3DEnv, keep: str) -> Iterator[None]:
    """Every item but `keep`, and the robot, out of the scene.

    The ray tests of segment_blocked report only the first body each ray hits. In
    the planner's simulator, the other items sit where the scene starts and the
    robot where the last IK test left it, not where the plan puts them, so either
    could hide `keep`.
    """
    away = Pose(_AWAY)
    with ExitStack() as stack:
        for name in sim.requested_objects + sim.clutter_objects:
            if name != keep:
                stack.enter_context(_scratch_pose(sim, name, away))
        base = sim.robot.get_base()
        sim.robot.set_base(SE2Pose(_AWAY[0], _AWAY[1], 0.0))
        try:
            yield
        finally:
            sim.robot.set_base(base)


def usable_probability(ctx: BedsideStreamContext, o: str, pose: Pose) -> float:
    """P_b(m(grasp point, theta) >= 0) under the particle belief."""
    point = ctx.sim.grasp_point(o, pose)
    m = reach_margin(point[None], ctx.theta_particles, ctx.sim.shoulder)
    return float(np.mean(m >= 0))


NOMINAL_HALF_HEIGHT = 0.04
_CANDIDATE_CACHE: dict[tuple, list[tuple[str, NDArray]]] = {}


def _slot_ik_reachable(sim: ObjectCentricBedsideStocking3DEnv, slot: NDArray) -> bool:
    arm = sim.robot.arm
    static = sim.static_collision_ids
    item = Pose((slot[0], slot[1], slot[2] + NOMINAL_HALF_HEIGHT))
    for quat in GRASP_ORIENTATIONS:
        target = multiply_poses(
            item, Pose((0.0, 0.0, NOMINAL_HALF_HEIGHT - GRASP_BELOW_TOP), quat)
        )
        hover = Pose(
            (target.position[0], target.position[1], target.position[2] + HOVER_HEIGHT),
            quat,
        )
        for base in _base_candidates(slot[0], slot[1]):
            sim.robot.set_base(base)
            if any(
                check_body_collisions(sim.robot.base.robot_id, b, sim.physics_client_id)
                for b in static
            ):
                continue
            arm.set_joints(HOME_JOINTS)
            ok = True
            for ee in (hover, target):
                try:
                    inverse_kinematics(arm, ee, validate=True, set_joints=True)
                except InverseKinematicsError:
                    ok = False
                    break
                if any(
                    check_body_collisions(arm.robot_id, b, sim.physics_client_id)
                    for b in static
                ):
                    ok = False
                    break
            if ok:
                return True
    return False


def requested_candidate_slots(
    sim: ObjectCentricBedsideStocking3DEnv,
) -> list[tuple[str, NDArray]]:
    """Bedside slots within the largest prior reach and reachable by IK (belief
    independent, cached)."""
    cfg = sim.config
    key = (
        tuple(sorted(cfg.bedside_regions.items())),
        cfg.reach_theta_ub,
        tuple(sim.shoulder),
    )
    if key not in _CANDIDATE_CACHE:
        base, joints = sim.robot.get_base(), sim.robot.arm.get_joint_positions()
        out = []
        for region in cfg.bedside_regions:
            for slot in cfg.region_slots(region):
                top = slot + [0.0, 0.0, 2 * NOMINAL_HALF_HEIGHT]
                ub = np.asarray(cfg.reach_theta_ub)
                if reach_margin(top, ub, sim.shoulder) < 0:
                    continue
                if _slot_ik_reachable(sim, slot):
                    out.append((region, slot))
        sim.robot.set_base(base)
        sim.robot.arm.set_joints(joints)
        _CANDIDATE_CACHE[key] = out
    return _CANDIDATE_CACHE[key]


def sample_place_pose(ctx: BedsideStreamContext, o: str) -> Iterator[tuple[Placement]]:
    """Requested items cycle through the candidate slots; clutter samples every
    surface."""
    if o in ctx.requested:
        cands = [(s, slot[:2]) for s, slot in ctx.requested_slots]
        while True:
            for k in ctx.place_order(o):
                s, xy = cands[k]
                xy = xy + ctx.rng.uniform(-POSE_JITTER, POSE_JITTER, size=2)
                yield (Placement(o, ctx.sim.resting_pose(o, tuple(xy), s), s),)
    surfaces = ctx.all_surfaces
    cands = [(s, xy) for s in surfaces for xy in ctx.slots[s][:, :2]]
    for k in ctx.clutter_order(o, cands):
        s, xy = cands[k]
        xy = xy + ctx.rng.uniform(-POSE_JITTER, POSE_JITTER, size=2)
        yield (Placement(o, ctx.sim.resting_pose(o, tuple(xy), s), s),)
    regions = ctx.sim.config.surface_regions
    for _ in range(MAX_PLACE_POSE_ATTEMPTS):
        s = surfaces[ctx.rng.integers(len(surfaces))]
        (x0, x1), (y0, y1), _ = regions[s]
        xy = (
            ctx.rng.uniform(x0 + 0.05, x1 - 0.05),
            ctx.rng.uniform(y0 + 0.05, y1 - 0.05),
        )
        yield (Placement(o, ctx.sim.resting_pose(o, xy, s), s),)


def snap_placement(ctx: BedsideStreamContext, o: str, point: NDArray) -> Placement:
    """`o` resting on one of its surfaces as near `point` as it can: the point's
    (x, y) moved into each surface's box, and the nearest of those poses."""
    options = []
    for s in ctx.placement_surfaces(o):
        low, high = ctx.surface_box(s)
        xy = np.clip(point[:2], low, high)
        pose = ctx.sim.resting_pose(o, (float(xy[0]), float(xy[1])), s)
        options.append(Placement(o, pose, s))
    distances = [np.linalg.norm(np.array(p.pose.position) - point) for p in options]
    return options[int(np.argmin(distances))]


def sample_uniform_placement(ctx: BedsideStreamContext, o: str) -> Placement:
    """`o` resting anywhere on its surfaces, uniformly over their area. A requested
    item stays within the largest reach the prior allows, as the candidate slots do:
    no belief makes a pose beyond it usable."""
    surfaces = ctx.placement_surfaces(o)
    boxes = [ctx.surface_box(s) for s in surfaces]
    areas = np.array([np.prod(high - low) for low, high in boxes])
    largest = np.asarray(ctx.sim.config.reach_theta_ub)
    for _ in range(MAX_WITHIN_REACH_DRAWS):
        k = int(ctx.rng.choice(len(surfaces), p=areas / areas.sum()))
        xy = ctx.rng.uniform(*boxes[k])
        pose = ctx.sim.resting_pose(o, (float(xy[0]), float(xy[1])), surfaces[k])
        if o not in ctx.requested:
            break
        if reach_margin(ctx.sim.grasp_point(o, pose), largest, ctx.sim.shoulder) >= 0:
            break
    return Placement(o, pose, surfaces[k])


def sample_continuous_place_pose(
    ctx: BedsideStreamContext, o: str
) -> Iterator[tuple[Placement]]:
    """psi's draws for `o`, each snapped onto its nearest surface, then uniform
    samples over its surfaces (the bedside for requested items, every surface for
    clutter)."""
    for point in ctx.proposed_points(o):
        yield (snap_placement(ctx, o, point),)
    for _ in range(MAX_PLACE_POSE_ATTEMPTS):
        yield (sample_uniform_placement(ctx, o),)


def sample_grasp(ctx: BedsideStreamContext, o: str) -> Iterator[tuple[Grasp]]:
    """Top-down grasps just below the item's top."""
    # pylint: disable-next=protected-access
    z = ctx.sim._get_half_extents(o)[2] - GRASP_BELOW_TOP
    for quat in GRASP_ORIENTATIONS:
        yield (Grasp(o, Pose((0.0, 0.0, z), quat)),)


def _base_candidates(x: float, y: float) -> list[SE2Pose]:
    facing = np.pi if x < CORRIDOR_X else 0.0
    out = []
    for dist in BASE_DISTANCES:
        for rot in BASE_ROTATIONS:
            ang = _wrap(facing + rot)
            out.append(SE2Pose(x - dist * np.cos(ang), y - dist * np.sin(ang), ang))
    return out


def inverse_kinematics_stream(
    ctx: BedsideStreamContext, o: str, p: Placement, g: Grasp
) -> Iterator[tuple[BaseConf, ArmTrajectory]]:
    """A base pose and arm trajectory that put `o` at `p` with `g`, if any."""
    sim = ctx.sim
    static = sim.static_collision_ids
    target_ee = multiply_poses(p.pose, g.transform)
    hover_ee = Pose(
        (
            target_ee.position[0],
            target_ee.position[1],
            target_ee.position[2] + HOVER_HEIGHT,
        ),
        target_ee.orientation,
    )
    target_id = sim._object_name_to_pybullet_id(o)  # pylint: disable=protected-access
    static = static | {target_id}
    with _scratch_pose(sim, o, p.pose):
        found = _ik_at(ctx, o, p, g, hover_ee, target_ee, static)
    if found is not None:
        yield found


def _carry_collides(
    sim: ObjectCentricBedsideStocking3DEnv, o: str, g: Grasp, static: set[int]
) -> bool:
    """Whether the arm in the carry conf, holding `o` with `g`, hits a static body."""
    arm = sim.robot.arm
    arm.set_joints(CARRY_JOINTS)
    body = sim._object_name_to_pybullet_id(o)  # pylint: disable=protected-access
    if any(
        check_body_collisions(arm.robot_id, b, sim.physics_client_id)
        for b in static - {body}
    ):
        return True
    with _scratch_pose(
        sim, o, multiply_poses(arm.get_end_effector_pose(), g.transform.invert())
    ):
        return any(
            check_body_collisions(body, b, sim.physics_client_id)
            for b in static - {body}
        )


def _ik_at(
    ctx: BedsideStreamContext,
    o: str,
    p: Placement,
    g: Grasp,
    hover_ee: Pose,
    target_ee: Pose,
    static: set[int],
) -> tuple[BaseConf, ArmTrajectory] | None:
    """A base pose and carry, hover, and grasp confs that place `o` at `p`."""
    sim = ctx.sim
    arm = sim.robot.arm
    x, y = p.pose.position[:2]
    for base in _base_candidates(x, y):
        sim.robot.set_base(base)
        if any(
            check_body_collisions(sim.robot.base.robot_id, b, sim.physics_client_id)
            for b in static
        ):
            continue
        if _carry_collides(sim, o, g, static):
            continue
        confs = []
        arm.set_joints(HOME_JOINTS)
        assert isinstance(arm, FingeredSingleArmPyBulletRobot)
        arm.open_fingers()
        for ee in (hover_ee, target_ee):
            try:
                q = inverse_kinematics(arm, ee, validate=True, set_joints=True)
            except InverseKinematicsError:
                break
            if any(
                check_body_collisions(arm.robot_id, b, sim.physics_client_id)
                for b in static
            ):
                break
            confs.append(list(q))
        if len(confs) < 2:
            continue
        return (BaseConf(base), ArmTrajectory([CARRY_JOINTS] + confs, o, base))
    return None


def test_usable(ctx: BedsideStreamContext, o: str, p: Placement) -> bool:
    """Whether `p` is on the bedside and reachable with probability 1 - delta."""
    return (
        p.surface in ctx.bedside_surfaces
        and usable_probability(ctx, o, p.pose) >= 1 - ctx.delta
    )


def test_clear(
    ctx: BedsideStreamContext, o: str, p: Placement, o2: str, p2: Placement
) -> bool:
    """Whether `o2` at `p2` leaves the path from the shoulder to `o` at `p` clear,
    whatever the other items and the robot are doing."""
    if o == o2:
        return True
    sim = ctx.sim
    other = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    with _alone(sim, o2), _scratch_pose(sim, o2, p2.pose):
        return not segment_blocked(
            sim.shoulder,
            sim.grasp_point(o, p.pose),
            {other},
            sim.config.arm_radius,
            sim.physics_client_id,
        )


def test_cfree_pose_pose(
    ctx: BedsideStreamContext, o1: str, p1: Placement, o2: str, p2: Placement
) -> bool:
    """Whether two items at their poses keep apart."""
    if o1 == o2:
        return True
    if p1.surface != p2.surface:
        return True
    sim = ctx.sim
    id1 = sim._object_name_to_pybullet_id(o1)  # pylint: disable=protected-access
    id2 = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    with ExitStack() as stack:
        stack.enter_context(_scratch_pose(sim, o1, p1.pose))
        stack.enter_context(_scratch_pose(sim, o2, p2.pose))
        return not check_body_collisions(
            id1, id2, sim.physics_client_id, distance_threshold=0.01
        )


def test_cfree_traj_pose(
    ctx: BedsideStreamContext, t: ArmTrajectory, o2: str, p2: Placement
) -> bool:
    """Whether the arm trajectory misses `o2` at `p2`."""
    if o2 == t.target_object:
        return True
    sim = ctx.sim
    other = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    sim.robot.set_base(t.base)
    with _scratch_pose(sim, o2, p2.pose):
        for q in t.joint_plan[1:]:
            sim.robot.arm.set_joints(q)
            if check_body_collisions(
                sim.robot.arm.robot_id, other, sim.physics_client_id
            ):
                return False
    return True
