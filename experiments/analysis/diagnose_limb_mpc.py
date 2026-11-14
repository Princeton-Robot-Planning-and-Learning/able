"""Plan and execute one limb task per seed with MPC, in every variant.

Each job draws a person and a task as make_domain does for its seed, plans with the
scratch robot's MPC under a belief, and executes the plan open loop on the person.
The belief is the prior, or "true": the person's own scales with a standard
deviation of 0.01, which shows how well MPC reaches goals it knows are inside the
range. One row per job goes to a JSON lines file, so runs on two commits can be
compared. Rerunning with the same file skips the jobs it already holds, except those
that raised.

Example, on 12 processes:
    python experiments/analysis/diagnose_limb_mpc.py --seeds 0 1 2 3 \
        --beliefs prior true --jobs 12 --out limb_mpc.jsonl
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Planning stops after this many MPC rollouts, as in experiments/conf/env/limb.yaml.
MAX_ROLLOUTS = 2000
# Only guards against a slow machine; the rollouts bind first.
MAX_WALL_TIME = 1800.0
# The task band of the experiments.
SEPARATION = (0.5, 0.8)
TRUE_BELIEF_STDDEV = 0.01


def run_job(variant: str, seed: int, belief_name: str) -> dict[str, Any]:
    """Plan and execute the seed's task in the variant under the belief."""
    # Pylint does not narrow the domain's parts by the asserts below.
    # pylint: disable=import-outside-toplevel,no-member
    from able.beliefs.range_of_motion import RangeOfMotionBelief
    from able.envs.limb.human import RangeOfMotionPrior
    from able.envs.limb.model import LimbModel
    from able.envs.limb.tasks import LimbTaskDistribution, make_domain
    from able.episodes import run_episode
    from able.planners.mpc import LimbMPCPlanner, MPCParams
    from able.structs import Budget, PlanningFailure, PlanningTimeout

    domain = make_domain(
        seed, variant, min_separation=SEPARATION[0], max_separation=SEPARATION[1]
    )
    model, tasks = domain.env_model, domain.task_distribution
    assert isinstance(model, LimbModel)
    assert isinstance(tasks, LimbTaskDistribution)
    real_env = domain.real_env
    theta = np.asarray(real_env.hidden_params)  # type: ignore[attr-defined]
    row: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "belief": belief_name,
        "theta": theta.tolist(),
    }
    planner = LimbMPCPlanner(model, MPCParams())
    try:
        task = tasks.sample(np.random.default_rng(seed))
        row["separation"] = model.sim.get_limb_error(
            task.initial_state.limb_positions, task.goal
        )
        if belief_name == "prior":
            belief = domain.initial_belief
        else:
            prior = RangeOfMotionPrior()
            belief = RangeOfMotionBelief(
                mean=theta.copy(),
                stddev=np.full(len(theta), TRUE_BELIEF_STDDEV),
                low=prior.low,
                high=prior.high,
                floor=np.full(len(theta), prior.low),
            )
        budget = Budget(max_wall_time=MAX_WALL_TIME, max_iterations=MAX_ROLLOUTS)
        start = time.perf_counter()
        try:
            policy = planner.plan(
                task, belief, MPCParams(), budget, np.random.default_rng(seed + 1000)
            )
        except PlanningFailure as failure:
            row.update(
                planned=False,
                timed_out=isinstance(failure, PlanningTimeout),
                plan_failure=str(failure),
                plan_time=time.perf_counter() - start,
                stats=getattr(failure, "stats", None) or {},
                success=False,
            )
            return row
        row.update(
            planned=True,
            timed_out=False,
            plan_failure="",
            plan_time=time.perf_counter() - start,
            stats=policy.get_planning_stats(),
        )
        trajectory = run_episode(real_env, task, policy, max_steps=model.horizon)
        final = trajectory.states[-1]
        row.update(
            success=bool(model.check_goal(task, final, theta)),
            failure=final.failure,
            violated=final.violated,
            reached=final.reached,
            final_error=model.get_goal_error(task, final),
            num_steps=len(trajectory.actions),
        )
        return row
    finally:
        planner.close()
        model.close()
        tasks.close()


def main() -> None:
    """Run every job and append its row to the output file."""
    # pylint: disable-next=import-outside-toplevel
    from able.envs.limb.human import VARIANTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--beliefs", nargs="+", default=["prior", "true"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add((row["variant"], row["seed"], row["belief"]))
    jobs = [
        (variant, seed, belief)
        for belief in args.beliefs
        for seed in args.seeds
        for variant in args.variants
        if (variant, seed, belief) not in done
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.jobs, mp_context=context) as pool:
        futures = {pool.submit(run_job, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:  # pylint: disable=broad-exception-caught
                variant, seed, belief = futures[future]
                row = {
                    "variant": variant,
                    "seed": seed,
                    "belief": belief,
                    "error": repr(error),
                }
            with args.out.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
