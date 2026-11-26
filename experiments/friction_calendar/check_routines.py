"""Draw friction calendar routines and keep those the night planner solves at the
corners of the calendar: every combination of each zone least and most slick. With
a goal shift, each task also gets the angle of its goal's directions of shift, for
which every day's goal is a valid goal. Writes a routines JSON file.

    python experiments/friction_calendar/check_routines.py --num-routines 10
    python experiments/friction_calendar/check_routines.py --goal-shift 0.5 \
        --goal-radius 0.1 \
        --out src/able/envs/friction/friction_calendar_moving_routines.json
"""

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from able.envs.calendar import ExerciseTask
from able.envs.friction.calendar import (
    ROUTINES_PATH,
    FrictionRoutine,
    get_frictions,
    get_zones,
    move_goal,
)
from able.envs.friction.model import FrictionModel
from able.envs.friction.tasks import FrictionTaskDistribution
from able.models.exact_match_cache import ExactMatchCache
from able.planners.friction_recall import FrictionRecallPlanner, RecallParams

# Goal-shift angles tried per task before drawing another task.
_NUM_ANGLES = 24
# Slick values per zone at which every goal must be valid.
_LOCUS_STEPS = 5
_HOLE_CLEARANCE = 0.1


def _make_model(goal_radius: float) -> FrictionModel:
    return FrictionModel(
        "2d",
        4,
        goal_radius,
        15,
        num_floors=8,
        strike_noise=False,
        measurement_noise=False,
    )


def _valid_goal(model, paths, goal, start) -> bool:
    """Whether a goal lies on a pushable cell, its circle clear of the holes, and
    away from the start."""
    world, res = model.world, paths.value_resolution
    point = np.asarray(goal)
    i, j = int(point[0] / res), int(point[1] / res)
    cells = paths.pushable_cells
    if not (0 <= j < cells.shape[0] and 0 <= i < cells.shape[1] and cells[j, i]):
        return False
    phys = world.physics
    clear = phys.hole_radius + _HOLE_CLEARANCE + phys.goal_radius
    if any(np.linalg.norm(point - np.asarray(h)) < clear for h in world.holes):
        return False
    return not world.in_goal(np.asarray(start), goal)


def _find_angle(  # type: ignore[no-untyped-def]
    model, paths, start, goal, goal_shift, max_slick, rng
) -> float | None:
    """An angle for which the goal under every slick on a grid is valid, or None."""
    values = np.linspace(0.0, max_slick, _LOCUS_STEPS)
    angles = np.linspace(0.0, 2 * np.pi, _NUM_ANGLES, endpoint=False)
    for angle in rng.permutation(angles):
        if all(
            _valid_goal(
                model,
                paths,
                move_goal(goal, float(angle), np.array(s), goal_shift),
                start,
            )
            for s in itertools.product(values, repeat=3)
        ):
            return float(angle)
    return None


def draw_routine(
    candidate: int,
    num_tasks: int,
    base_range: tuple[float, float],
    min_zones: int,
    goal_radius: float,
    goal_shift: float,
    max_slick: float,
) -> FrictionRoutine:
    """A random routine whose tasks' shortest paths each cross min_zones zones."""
    model = _make_model(goal_radius)
    zones = get_zones(model.world)
    rng = np.random.default_rng([candidate, 3])
    base = rng.uniform(*base_range, size=len(zones))
    tasks_dist = FrictionTaskDistribution(model, 1.5, 3.5)
    paths = tasks_dist._paths  # pylint: disable=protected-access
    tasks: list[tuple[Any, Any, Any]] = []
    angles: list[float] = []
    while len(tasks) < num_tasks:
        task = tasks_dist.sample(rng)
        state = task.initial_state
        floors = paths.path_floors(state.puck, task.goal)
        if len({int(zones[f]) for f in floors}) < min_zones:
            continue
        angle: float | None = 0.0
        if goal_shift > 0.0:
            angle = _find_angle(
                model, paths, state.puck, task.goal, goal_shift, max_slick, rng
            )
        if angle is None:
            continue
        tasks.append((state.puck, state.robot, task.goal))
        angles.append(angle)
    return FrictionRoutine(
        base, tuple(tasks), tuple(angles) if goal_shift > 0.0 else None
    )


def check_routine(args) -> dict:
    """The routine for a candidate and, for each corner of the calendar, which tasks
    the night planner solved with a plan that also replays."""
    (
        candidate,
        num_tasks,
        base_range,
        min_zones,
        max_slick,
        night_sims,
        goal_radius,
        goal_shift,
    ) = args
    routine = draw_routine(
        candidate, num_tasks, base_range, min_zones, goal_radius, goal_shift, max_slick
    )
    model = _make_model(goal_radius)
    zones = get_zones(model.world)
    num_zones = int(zones.max()) + 1
    corners = [
        np.array(c) for c in itertools.product([0.0, max_slick], repeat=num_zones)
    ]
    planner = FrictionRecallPlanner(
        model, lambda t, b: np.zeros(1), RecallParams(ExactMatchCache()), seed=candidate
    )
    params = planner.get_default_params()
    solved = []
    for e, slick in enumerate(corners):
        frictions = np.round(get_frictions(routine.base, zones, slick), 6)
        row = []
        for i, (puck, robot, goal) in enumerate(routine.tasks):
            if routine.angles is not None:
                goal = move_goal(goal, routine.angles[i], slick, goal_shift)
            state = model.get_initial_state(puck, robot, goal)
            task = ExerciseTask(initial_state=state, goal=goal, exercise=i)
            ok = False
            for attempt in range(3):
                rng = np.random.default_rng([candidate, e, i, attempt])
                found = planner._search(  # pylint: disable=protected-access
                    task, frictions, night_sims, rng, finish=goal_shift > 0.0
                )
                if found is not None and planner.replays(
                    task, found, frictions, params
                ):
                    ok = True
                    break
            row.append(ok)
        solved.append(row)
    return {"candidate": candidate, "routine": routine.to_json(), "solved": solved}


def main() -> None:
    """Check candidates in parallel until enough routines pass, and write them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-routines", type=int, default=10)
    parser.add_argument("--num-tasks", type=int, default=6)
    parser.add_argument("--base", type=float, nargs=2, default=(0.08, 0.2))
    parser.add_argument("--min-zones", type=int, default=2)
    parser.add_argument("--max-slick", type=float, default=1.0)
    parser.add_argument("--night-sims", type=int, default=500)
    parser.add_argument("--goal-radius", type=float, default=0.2)
    parser.add_argument("--goal-shift", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROUTINES_PATH)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    jobs = [
        (
            c,
            args.num_tasks,
            tuple(args.base),
            args.min_zones,
            args.max_slick,
            args.night_sims,
            args.goal_radius,
            args.goal_shift,
        )
        for c in range(args.max_candidates)
    ]
    accepted: dict[str, dict] = {}
    log = []
    with Pool(args.workers) as pool:
        for result in pool.imap(check_routine, jobs):
            ok = all(all(row) for row in result["solved"])
            log.append(result)
            print(result["candidate"], "ok" if ok else "rejected", flush=True)
            if ok and len(accepted) < args.num_routines:
                accepted[str(len(accepted))] = {
                    **result["routine"],
                    "candidate": result["candidate"],
                }
            if len(accepted) >= args.num_routines:
                pool.terminate()
                break
    args.out.write_text(json.dumps(accepted, indent=1) + "\n", encoding="utf-8")
    if args.log is not None:
        args.log.write_text(json.dumps(log) + "\n", encoding="utf-8")
    print(f"Accepted {len(accepted)} routines into {args.out}")


if __name__ == "__main__":
    main()
