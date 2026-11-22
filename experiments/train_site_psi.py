"""Train the friction-site push model for one site, or for every site evenly, and
save it after chosen numbers of units.

A robot built from env=friction_sites with one site per group trains only for
--train_site, one unit per sleep, or for every site in turn if --train_site is -1.
The pickle holds the planner parameters at each depth in --depths, the training
data, the beliefs, the true frictions, each site's corridor and routine, and the
POMDP, for experiments/analysis/plot_theta_psi.py.

    python experiments/train_site_psi.py --seed 0 --train_site 1 --out s0_t1.pkl
"""

import argparse
import dataclasses
import pickle
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra import compose, initialize_config_dir

from able.envs.multi_site import SiteRoutineTaskDistribution, focus_routines
from able.structs import Budget

_CONF = str(Path(__file__).parent / "conf")


def main() -> None:
    """Train and save the snapshots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_site", type=int, default=0)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    with initialize_config_dir(config_dir=_CONF, version_base=None):
        cfg = compose(
            "config",
            overrides=[
                "env=friction_sites",
                "approach=current_belief",
                f"seed={args.seed}",
                "env.domain.group_sizes=[1,1,1,1]",
                "env.domain.survey=levels",
                *args.overrides,
            ],
        )
    domain = hydra.utils.instantiate(cfg.env.domain, seed=args.seed)
    routines = domain.task_distribution
    assert isinstance(routines, SiteRoutineTaskDistribution)
    num_sites = len(routines.schedule)
    focused = focus_routines(routines, args.train_site)
    make_robot = hydra.utils.instantiate(cfg.approach, _partial_=True)
    robot = make_robot(
        domain=dataclasses.replace(domain, task_distribution=focused), recorder=None
    )
    pomdp: Any = getattr(robot.planner, "inner").pomdp
    num_corridors = max(pomdp.world.puck_map.surface_colors) + 1
    snapshots = {0: robot.planner_params} if 0 in args.depths else {}
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    for depth in range(1, max(args.depths) + 1):
        robot.sleep(Budget(max_iterations=1), rng)
        if depth in args.depths:
            snapshots[depth] = robot.planner_params
        print(f"depth {depth} done after {time.perf_counter() - start:.0f}s")
    learner: Any = robot.learner
    out = {
        "seed": args.seed,
        "train_site": args.train_site,
        "truths": np.asarray(domain.real_env.hidden_params),  # type: ignore[attr-defined]
        "beliefs": robot.belief,
        "corridors": [routines.get_group(s) % num_corridors for s in range(num_sites)],
        "routines": [list(focused.get_routine(s)) for s in range(num_sites)],
        "snapshots": snapshots,
        "dataset": learner.dataset,
        "pomdp": pomdp,
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
