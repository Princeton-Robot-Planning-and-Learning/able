"""A fast simulation of the friction calendar without physics: the real robots,
belief sources and learner choose what to plan each night, every night plan is
made instantly, and each day a task succeeds with the chance, measured by
probe_replay.py, that a plan that many grid steps away (summed over zones) replays.
Prints each approach's share of days with a failed task. The chances are paired:
every approach draws the same random numbers for a day and task.

    python experiments/friction_calendar/simulate_allocation.py --success 1 .9 .6 .3 .1 0
"""

# pylint: disable=no-member,protected-access  # Domain fields hold base classes.

import argparse
import itertools
from multiprocessing import Pool

import numpy as np

from able import approaches as ap
from able.envs.friction.calendar import make_friction_calendar_domain
from able.planners.friction_recall import FrictionRecallPlanner, StoredPush
from able.structs import Budget

APPROACHES = {
    "current_belief": (ap.make_friction_calendar_current_belief_robot, {}),
    "dreaming": (ap.make_friction_calendar_dreaming_robot, {}),
    "coverage": (ap.make_friction_calendar_belief_coverage_robot, {}),
    "one_step_greedy": (
        ap.make_friction_calendar_allocation_robot,
        {"rule": "one_step_greedy"},
    ),
    "deliberate_practice": (
        ap.make_friction_calendar_allocation_robot,
        {"rule": "deliberate_practice"},
    ),
    "transfer_practice": (
        ap.make_friction_calendar_allocation_robot,
        {"rule": "transfer_practice"},
    ),
}


def _stub_search(*_args, **_kwargs):
    return [StoredPush(0.0, 0.0, None, (0.0, 0.0))]


def run(job):
    """One approach on one calendar: its slow-day share and task success."""
    calendar_seed, name, args = job
    FrictionRecallPlanner._search = _stub_search  # pylint: disable=protected-access
    FrictionRecallPlanner.replays = lambda *_: True
    domain = make_friction_calendar_domain(
        calendar_seed % 8,
        calendar=args.calendar,
        step=args.step,
        max_slick=args.max_slick,
        calendar_seed=calendar_seed,
        notice=args.notice,
        weather=args.weather,
        rain_slick=args.rain_slick,
        num_days=max(40, args.days + args.lookahead + 1),
    )
    factory, extra = APPROACHES[name]
    budgets = Budget(max_iterations=0)
    kwargs = dict(extra)
    if name == "current_belief":
        kwargs.update(planning_budget=budgets, days_ahead=0)
    else:
        kwargs.update(
            gathering_budget=budgets,
            execution_budget=budgets,
            planning_budget=budgets,
            horizon=args.lookahead,
        )
    if "rule" in kwargs:
        kwargs.update(
            num_units=args.units,
            tier_values=tuple(args.tiers),
            first_radius=args.first_radius,
            day_type_steps=args.day_type_steps,
        )
    robot = factory(domain, **kwargs)
    tasks = domain.task_distribution
    planner = robot.planner
    sleep_rng = np.random.default_rng([calendar_seed, 1])
    draws = np.random.default_rng([calendar_seed, 2]).random(
        (args.days, tasks.num_exercises)
    )
    slow, solved = [], []
    for day in range(args.days):
        if day > 0:
            tasks.start_day(sleep_rng)
            robot.predict_belief(domain.predict_belief)
        robot.sleep(Budget(max_iterations=args.units), sleep_rng)
        robot.update_belief_from_queries(
            robot.gather_information(None, Budget(max_iterations=0), sleep_rng)
        )
        cache = robot.planner_params.model
        failed, num_solved = False, 0
        for task in tasks.get_tasks():
            x = planner._get_features(
                task, robot.belief
            )  # pylint: disable=protected-access
            key = cache.get_key(x)
            near = key if key in cache.entries else cache.get_nearest_key(key)
            if near is None:
                failed = True
                continue
            steps = int(np.abs(np.array(near[1:]) - np.array(key[1:])).sum())
            chance = args.success[steps] if steps < len(args.success) else 0.0
            ok = draws[day, task.exercise] < chance
            failed |= not ok
            num_solved += int(ok)
        slow.append(failed)
        solved.append(num_solved / tasks.num_exercises)
    late = slice(args.eval_from, None)
    return (
        name,
        calendar_seed,
        float(np.mean(slow[late])),
        float(np.mean(solved[late])),
    )


def main() -> None:
    """Simulate every approach on every calendar and print the means."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=float, nargs="+", required=True)
    parser.add_argument("--tiers", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--first-radius", type=int, default=0)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--max-slick", type=float, default=1.0)
    parser.add_argument("--calendar", default="uneven")
    parser.add_argument("--notice", default="run")
    parser.add_argument("--weather", action="store_true")
    parser.add_argument("--rain-slick", type=float, default=0.3)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--day-type-steps", type=int, default=1)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--eval-from", type=int, default=5)
    parser.add_argument("--calendars", type=int, default=24)
    parser.add_argument("--approaches", nargs="+", default=list(APPROACHES))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    jobs = list(itertools.product(range(args.calendars), args.approaches, [args]))
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs)
    table = {name: np.zeros(args.calendars) for name in args.approaches}
    success = {name: np.zeros(args.calendars) for name in args.approaches}
    for name, seed, rate, solved in results:
        table[name][seed] = rate
        success[name][seed] = solved
    print(
        f"units={args.units} step={args.step} max_slick={args.max_slick} "
        f"lookahead={args.lookahead} tiers={args.tiers}"
    )
    for name in args.approaches:
        first8 = table[name][:8].mean()
        print(
            f"  {name:20s} slow-day {table[name].mean():.2f}  "
            f"task-success {success[name].mean():.2f}  (calendars 0-7: {first8:.2f})"
        )


if __name__ == "__main__":
    main()
