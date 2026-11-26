"""How well a stored friction plan replays on another day: for random plan days, plan
every task of a routine, then replay each plan on days 0 to max_steps grid steps
away (summed over zones) and run it in the simulator with that day's true
frictions. Writes one row per replay to a CSV.

    python experiments/friction_calendar/probe_replay.py --routines 8 9 --out g1.csv
"""

# pylint: disable=no-member,use-dict-literal  # Domain fields hold base classes.

import argparse
import csv
import dataclasses
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from able.approaches import _make_friction_calendar_planner
from able.envs.friction.calendar import (
    ROUTINES_PATH,
    FrictionCalendarTaskDistribution,
    get_frictions,
    load_routine,
    make_friction_calendar_domain,
)
from able.structs import Budget, PlanningFailure


def _random_day(rng, num_zones, top):
    return rng.integers(0, top + 1, size=num_zones)


def _day_at(rng, day, steps, top, max_tries=1000):
    """A random grid day exactly steps steps from day, summed over zones."""
    for _ in range(max_tries):
        other = day.copy()
        for _ in range(steps):
            z = rng.integers(len(day))
            other[z] += rng.choice([-1, 1])
        if (
            np.abs(other - day).sum() == steps
            and np.all(other >= 0)
            and np.all(other <= top)
        ):
            return other
    return None


def probe(job):
    """The replays of one routine's plans for one random plan day."""
    routine, plan_day_index, args = job
    domain = make_friction_calendar_domain(
        routine,
        step=args.grid,
        max_slick=args.max_slick,
        grid=args.grid,
        goal_radius=args.goal_radius,
        horizon=args.horizon,
        survey_stddev=args.survey_stddev,
        goal_shift=args.goal_shift,
        routines=args.routines_file,
    )
    tasks = domain.task_distribution
    assert isinstance(tasks, FrictionCalendarTaskDistribution)
    model = domain.env_model
    zones = tasks.zones
    num_zones = int(zones.max()) + 1
    top = round(args.max_slick / args.grid)
    base = load_routine(routine, ROUTINES_PATH.with_name(args.routines_file)).base
    belief0 = domain.initial_belief
    slick0 = tasks.get_midpoint(belief0)

    def belief_at(day):
        return belief0.scaled(np.exp(-(day * args.grid - slick0)[zones]))

    rng = np.random.default_rng([routine, plan_day_index, 5])
    plan_day = _random_day(rng, num_zones, top)
    planner, _ = _make_friction_calendar_planner(
        domain, args.grid, args.night_sims, routine, args.plan_dir
    )
    default = dataclasses.replace(
        planner.get_default_params(),
        robust=args.robust,
        night_sims=args.night_sims,
        finish=args.finish,
    )
    rows = []
    for exercise in range(tasks.num_exercises):
        task = tasks.get_task(exercise, plan_day * args.grid)
        plan_belief = belief_at(plan_day)
        try:
            policy = planner.plan(
                task, plan_belief, planner.get_training_params(default), Budget(), rng
            )
        except PlanningFailure:
            rows.append(
                dict(
                    routine=routine,
                    plan_day=plan_day_index,
                    exercise=task.exercise,
                    steps=-1,
                    check=0,
                    success=0,
                    miss=float("nan"),
                    check_dev=float("nan"),
                    exec_dev=float("nan"),
                    diff="",
                    pushes=0,
                )
            )
            continue
        cache = default.model.fit(planner.get_plan_examples(task, plan_belief, policy))
        params = dataclasses.replace(default, model=cache)
        for steps in range(args.max_steps + 1):
            for _ in range(args.replays_per_step):
                day = _day_at(rng, plan_day, steps, top)
                if day is None:
                    continue
                belief = belief_at(day)
                truth = get_frictions(base, zones, day * args.grid)
                task = tasks.get_task(exercise, day * args.grid)
                check, check_dev, replay = 0, float("nan"), None
                frictions = np.asarray(belief.mean)
                puck, devs = task.initial_state.puck_array, [0.0]
                for stored in policy.pushes:
                    push = planner.aim(
                        puck, stored, frictions, task.goal, default.max_turn
                    )
                    out = (
                        None
                        if push is None
                        else planner.predict_stop(puck, push, frictions, task.goal)
                    )
                    if out is None or out[1] == 2:
                        break
                    puck = out[0]
                    if out[1] == 1:
                        check, check_dev = 1, max(devs)
                        break
                    devs.append(float(np.linalg.norm(puck - np.array(stored.stop))))
                if check:
                    replay = planner.plan(
                        task, belief, params, Budget(max_iterations=1), rng
                    )
                success, miss = 0, float("nan")
                if replay is not None:
                    state = task.initial_state
                    exec_devs = [0.0]
                    for _ in range(model.horizon):
                        if state.ended:
                            break
                        state = model.sample_next_state(
                            state, replay.get_action(state), truth, rng
                        )
                        k = state.pushes - 1
                        if not state.ended and k < len(policy.pushes):
                            exec_devs.append(
                                float(
                                    np.linalg.norm(
                                        state.puck_array
                                        - np.array(policy.pushes[k].stop)
                                    )
                                )
                            )
                    success = int(state.outcome == "goal")
                    miss = (
                        -1.0
                        if state.outcome == "hole"
                        else float(
                            np.linalg.norm(
                                np.array(state.true_puck) - np.array(task.goal)
                            )
                        )
                    )
                exec_dev = max(exec_devs) if replay is not None else float("nan")
                rows.append(
                    dict(
                        routine=routine,
                        plan_day=plan_day_index,
                        exercise=task.exercise,
                        steps=steps,
                        check=check,
                        success=success,
                        miss=miss,
                        check_dev=check_dev,
                        exec_dev=exec_dev,
                        diff=" ".join(str(int(v)) for v in day - plan_day),
                        pushes=len(policy.pushes),
                    )
                )
    return rows


def main() -> None:
    """Probe every routine and plan day in parallel and write the replays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--routines", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--plan-days", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--replays-per-step", type=int, default=3)
    parser.add_argument("--grid", type=float, default=0.1)
    parser.add_argument("--max-slick", type=float, default=1.0)
    parser.add_argument("--goal-radius", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--night-sims", type=int, default=500)
    parser.add_argument("--plan-dir", type=str, default=None)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--survey-stddev", type=float, default=0.05)
    parser.add_argument("--goal-shift", type=float, default=0.0)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--routines-file", default=ROUTINES_PATH.name)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jobs = [(r, i, args) for r in args.routines for i in range(args.plan_days)]
    with Pool(args.workers) as pool, args.out.open("w", newline="") as f:
        writer = None
        for rows in pool.imap_unordered(probe, jobs):
            for row in rows:
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
