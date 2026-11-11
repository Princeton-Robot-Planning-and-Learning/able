"""Find what limits friction's success with ample compute (#130, step 1).

Runs the scratch planner on single friction tasks, with no per-task budget, under
a grid of settings, and records how each episode ends:

- goal: the puck stopped in the goal;
- hole: the puck passed over a hole;
- stuck: the planner found no feasible push ("No feasible push");
- out_of_pushes: the horizon ran out first.

A setting is a belief (the prior, or particles concentrated near the true
frictions), POMCPOW simulations per push, strike noise on or off (in the real
pushes and the planner alike), and the horizon. Each seed draws true frictions
from the prior as the experiments do (make_domain), and each episode reruns the
task from the same belief with new noise. One JSON line per episode is appended
to the output file, so an interrupted run resumes where it stopped.

Example, the step-1 grid on 24 processes:
    python experiments/analysis/diagnose_friction.py --out logs/diagnose/step1.jsonl \
        --grid step1 --workers 24
    python experiments/analysis/diagnose_friction.py --out logs/diagnose/step1.jsonl \
        --summarize
"""

import argparse
import dataclasses
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.beliefs.particles import ParticleBelief
from able.envs.friction.model import FrictionModel
from able.envs.friction.tasks import make_domain
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, PuckWorld
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.models.base import ConditionalModel
from able.planners.pomcpow import (
    FrictionPOMCPOWPlanner,
    FrictionPOMCPOWPolicy,
    POMCPOWConfig,
    POMCPOWParams,
)
from able.structs import Budget, PlanningFailure

OUTCOMES = ("goal", "hole", "stuck", "out_of_pushes", "other_failure")
SETTING_KEYS = (
    "belief",
    "num_sims",
    "noise",
    "horizon",
    "num_holes",
    "goal_radius",
    "rollout_pushes",
)


@dataclasses.dataclass(frozen=True)
class Setting:
    """One cell of the grid."""

    belief: str  # "prior" or "near"
    num_sims: int
    noise: bool = True
    horizon: int = 15
    num_holes: int = 4
    goal_radius: float = 0.3
    # Greedy pushes per leaf rollout (0: the planner's default).
    rollout_pushes: int = 0


@dataclasses.dataclass(frozen=True)
class Job:
    """One episode of a setting."""

    setting: Setting
    seed: int
    episode: int
    num_particles: int
    near_spread: float
    max_depth: int
    # A trained push model to guide the search, and whether a solved episode
    # returns its training rows (under "dataset", not written to the output).
    push_model: ConditionalModel | None = None
    collect_rows: bool = False

    @property
    def key(self) -> tuple[Any, ...]:
        """What identifies the job's line in the output."""
        return (*dataclasses.astuple(self.setting), self.seed, self.episode)


def make_grid(name: str) -> list[Setting]:
    """The settings of a named grid."""
    if name == "step1":
        sims = (20, 100, 300, 1000, 5000)
        base = [Setting(b, n) for b in ("prior", "near") for n in sims]
        no_noise = [
            Setting(b, n, noise=False) for b in ("prior", "near") for n in (100, 1000)
        ]
        long = [
            Setting(b, n, horizon=30) for b in ("prior", "near") for n in (100, 1000)
        ]
        return base + no_noise + long
    if name == "budgets":
        budgets = (20, 50, 100, 300, 1000, 5000)
        return [Setting(b, n) for b in ("prior", "near") for n in budgets]
    if name == "harder":
        harder = (
            {"goal_radius": 0.2},
            {"horizon": 10},
            {"num_holes": 6},
            {"goal_radius": 0.2, "horizon": 10},
        )
        return [Setting("near", n, **h) for h in harder for n in (20, 1000)]
    if name == "short_rollouts":
        few_budgets = (20, 100, 1000, 5000)
        return [
            Setting(b, n, rollout_pushes=6)
            for b in ("prior", "near")
            for n in few_budgets
        ]
    if name == "smoke":
        return [Setting("near", 20), Setting("prior", 20, noise=False, horizon=30)]
    raise ValueError(f"Unknown grid {name!r}")


def _near_belief(
    frictions: np.ndarray, spread: float, num_particles: int, rng: np.random.Generator
) -> ParticleBelief:
    """Particles at the true frictions, each scaled by 1 + N(0, spread), within
    the prior's bounds."""
    low, high = DEFAULT_FRICTION_PRIOR.bounds
    scale = 1.0 + rng.normal(0.0, spread, (num_particles, len(frictions)))
    particles = np.clip(frictions * scale, low, high)
    return ParticleBelief(particles, low, high, seed=int(rng.integers(2**31)))


def run_job(job: Job) -> dict[str, Any]:
    """Run one episode and describe how it ended."""
    setting = job.setting
    domain = make_domain(
        seed=job.seed,
        num_holes=setting.num_holes,
        goal_radius=setting.goal_radius,
        horizon=setting.horizon,
        num_particles=job.num_particles,
    )
    model = domain.env_model
    assert isinstance(model, FrictionModel)
    # Before the simulator or the planner is built from the world.
    physics = model.world.physics
    if not setting.noise:
        physics = dataclasses.replace(physics, speed_noise=0.0, angle_noise=0.0)
    model.world = PuckWorld(model.world.puck_map, physics)
    # make_domain's first draw from its seed's stream.
    frictions = DEFAULT_FRICTION_PRIOR.sample(np.random.default_rng(job.seed))
    rng = np.random.default_rng([job.seed, job.episode])
    if setting.belief == "prior":
        belief = domain.initial_belief
    else:
        belief = _near_belief(frictions, job.near_spread, job.num_particles, rng)
    assert isinstance(belief, ParticleBelief)
    planner = FrictionPOMCPOWPlanner(model, FrictionBeliefUpdater(model.world))
    if setting.rollout_pushes:
        planner.pomdp.rollout_pushes = setting.rollout_pushes
    params = POMCPOWParams(
        POMCPOWConfig(num_sims=setting.num_sims, max_depth=job.max_depth)
    )
    if job.push_model is not None:
        params = planner.set_model(params, job.push_model)
    env = SimulatedEnvironment(model, frictions, rng)
    task = domain.task_distribution.sample(rng)
    start = time.perf_counter()
    policy: FrictionPOMCPOWPolicy | None = None
    states: list[Any] = []
    message = ""

    def record(state: Any, action: Any, next_state: Any) -> None:
        del state, action
        states.append(next_state)

    try:
        policy = planner.plan(task, belief, params, Budget(), rng)
        run_episode(env, task, policy, max_steps=setting.horizon, on_step=record)
    except PlanningFailure as failure:
        message = str(failure)
    last = states[-1] if states else task.initial_state
    if last.outcome is not None:
        outcome = last.outcome
    elif message == "No feasible push":
        outcome = "stuck"
    elif message:
        outcome = "other_failure"
    else:
        outcome = "out_of_pushes"
    stats = policy.get_planning_stats() if policy is not None else {}
    model.close()  # pylint: disable=no-member
    dataset = None
    if job.collect_rows and outcome == "goal" and policy is not None:
        dataset = planner.get_plan_examples(task, belief, policy)
    return {
        "dataset": dataset,
        **dataclasses.asdict(setting),
        "seed": job.seed,
        "episode": job.episode,
        "near_spread": job.near_spread,
        "frictions": [round(float(f), 4) for f in frictions],
        "outcome": outcome,
        "message": message,
        "pushes": last.pushes,
        "final_puck": [round(v, 3) for v in last.puck],
        "path": [[round(v, 3) for v in s.puck] for s in states],
        "seconds": round(time.perf_counter() - start, 1),
        **{f"plan_{k}": int(v) for k, v in stats.items()},
    }


def _done_keys(out: Path) -> set[tuple[Any, ...]]:
    if not out.exists():
        return set()
    keys = set()
    for line in out.read_text().splitlines():
        row = json.loads(line)
        row = {**dataclasses.asdict(Setting("", 0)), **row}  # older lines
        keys.add((*(row[k] for k in SETTING_KEYS), row["seed"], row["episode"]))
    return keys


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per setting: episodes, the fraction of each outcome, and mean pushes and
    simulations per episode."""
    groups = rows.groupby(list(SETTING_KEYS))
    table = groups["outcome"].value_counts(normalize=True).unstack(fill_value=0)
    table = table.reindex(columns=list(OUTCOMES), fill_value=0.0).round(2)
    table.insert(0, "episodes", groups.size())
    table["pushes"] = groups["pushes"].mean().round(1)
    table["minutes"] = (groups["seconds"].mean() / 60).round(1)
    return table.reset_index()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", default="step1")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num_particles", type=int, default=500)
    parser.add_argument("--near_spread", type=float, default=0.03)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        rows = pd.read_json(args.out, lines=True)
        for key, default in dataclasses.asdict(Setting("", 0)).items():
            rows[key] = rows[key].fillna(default) if key in rows else default
        with pd.option_context("display.width", 200, "display.max_rows", None):
            print(summarize(rows).to_string(index=False))
        return
    done = _done_keys(args.out)
    jobs = [
        Job(s, seed, ep, args.num_particles, args.near_spread, args.max_depth)
        for s, seed, ep in itertools.product(
            make_grid(args.grid), range(args.seeds), range(args.episodes)
        )
    ]
    jobs = [j for j in jobs if j.key not in done]
    # The longest jobs first, so they do not finish last.
    jobs.sort(key=lambda j: -j.setting.num_sims * j.setting.horizon)
    print(f"{len(jobs)} episodes to run", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, maxtasksperchild=1) as pool, args.out.open("a") as f:
        for row in pool.imap_unordered(run_job, jobs):
            del row["dataset"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(
                f"{row['belief']} sims={row['num_sims']} noise={row['noise']} "
                f"H={row['horizon']} seed={row['seed']} ep={row['episode']}: "
                f"{row['outcome']} in {row['pushes']} pushes, {row['seconds']} s",
                flush=True,
            )


if __name__ == "__main__":
    _main()
