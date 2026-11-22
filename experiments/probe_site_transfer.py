"""Probe how training for one friction site changes planning at every site.

A robot built from env=friction_sites and an approach trains only for one site
(--train_site, default 0), one unit per sleep, and after each depth in --depths
plans every site's routine under its belief. Sites 0 and 1 share group 0; sites
2, 3, and 4 are groups 1, 2, and 3, each with its own corridor on the default
corridors4 maze. Each evaluation task gets the same planning and
environment seeds at every depth, so depths differ only by the model. One CSV
row per evaluation task.

    python experiments/probe_site_transfer.py --seed 0 --overlap 0 --out probe.csv
"""

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from able.cycle import EpisodeResult, evaluate
from able.envs.multi_site import SiteRoutineTaskDistribution, focus_routines
from able.envs.simulated import SimulatedEnvironment
from able.structs import Budget

_CONF = str(Path(__file__).parent / "conf")


def main() -> None:
    """Train for one site and evaluate every site at each depth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--approach", default="current_belief")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--train_site", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    with initialize_config_dir(config_dir=_CONF, version_base=None):
        cfg = compose(
            "config",
            overrides=[
                "env=friction_sites",
                f"approach={args.approach}",
                f"seed={args.seed}",
                f"env.domain.region_overlap={args.overlap}",
                "env.domain.group_sizes=[2,1,1,1]",
                *args.overrides,
            ],
        )
    domain = hydra.utils.instantiate(cfg.env.domain, seed=args.seed)
    routines = domain.task_distribution
    assert isinstance(routines, SiteRoutineTaskDistribution)
    num_sites = len(routines.schedule)
    group_of = [routines.get_group(site) for site in range(num_sites)]
    world: Any = getattr(domain.env_model, "inner").world
    num_corridors = max(world.puck_map.surface_colors) + 1
    focused = focus_routines(routines, args.train_site)
    make_robot = hydra.utils.instantiate(cfg.approach, _partial_=True)
    robot = make_robot(
        domain=dataclasses.replace(domain, task_distribution=focused), recorder=None
    )
    planning = hydra.utils.instantiate(cfg.budgets.planning)
    truths = domain.real_env.hidden_params  # type: ignore[attr-defined]
    rows = []

    def evaluate_sites(depth: int) -> None:
        for site in range(num_sites):
            for i, task in enumerate(focused.get_routine(site)):
                for repeat in range(args.repeats):
                    stream = np.random.default_rng([args.seed, site, i, repeat])
                    env_rng, plan_rng = stream.spawn(2)
                    env = SimulatedEnvironment(domain.env_model, truths, env_rng)
                    result = evaluate(robot, task, env, planning, plan_rng)
                    row = _row(args, depth, site, group_of[site], i, repeat, result)
                    rows.append({**row, "corridor": group_of[site] % num_corridors})
        pd.DataFrame(rows).to_csv(args.out, index=False)

    start = time.perf_counter()
    evaluate_sites(0)
    rng = np.random.default_rng(args.seed)
    for depth in range(1, max(args.depths) + 1):
        robot.sleep(Budget(max_iterations=1), rng)
        if depth in args.depths:
            evaluate_sites(depth)
        print(f"depth {depth} done after {time.perf_counter() - start:.0f}s")


def _row(
    args: argparse.Namespace,
    depth: int,
    site: int,
    group: int,
    task: int,
    repeat: int,
    result: EpisodeResult,
) -> dict[str, float]:
    return {
        "seed": args.seed,
        "overlap": args.overlap,
        "train_site": args.train_site,
        "depth": depth,
        "site": site,
        "group": group,
        "task": task,
        "repeat": repeat,
        "success": result.success,
        "num_sims": result.planning_stats.get("num_sims", np.nan),
        "planning_time": result.planning_wall_time,
    }


if __name__ == "__main__":
    main()
