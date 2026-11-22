"""Fit the friction-site allocation model to transfer probes from
experiments/probe_site_transfer.py: the ramp of the trained site's gain, in
units and in passes over its routine, and the transfer to the same routine and
to other corridors, in effective units per unit.

    python experiments/analysis/fit_site_allocation.py --probe_dir probes/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEPTHS = (1, 2, 4, 8)


def get_ramp(units: np.ndarray, start: float, end: float) -> np.ndarray:
    """A gain of 0 up to start units that rises linearly to 1 at end."""
    return np.clip((units - start) / (end - start), 0.0, 1.0)


def get_saved(df: pd.DataFrame) -> pd.DataFrame:
    """Simulations saved per task at each depth, against depth 0, averaged over
    seeds, tasks, and repeats, per trained and evaluated site."""
    by_depth = (
        df.groupby(["train_site", "seed", "site", "corridor", "depth"])["num_sims"]
        .mean()
        .unstack("depth")
    )
    saved = by_depth.rsub(by_depth[0], axis=0).reset_index()
    return saved.groupby(["train_site", "site", "corridor"])[list(DEPTHS)].mean()


def fit_transfer(curve: np.ndarray, start: float, end: float) -> float:
    """The effective units per unit whose ramp best matches the curve."""
    units = np.array(DEPTHS, dtype=np.float64)
    errors = {
        t: float(np.sum((get_ramp(units * t, start, end) - curve) ** 2))
        for t in np.arange(0.0, 1.51, 0.01)
    }
    return min(errors, key=errors.__getitem__)


def main() -> None:
    """Print the fitted ramp and transfers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dir", type=Path, required=True)
    parser.add_argument("--routine_size", type=int, default=8)
    parser.add_argument("--num_training_tasks", type=int, default=2)
    args = parser.parse_args()
    df = pd.concat(pd.read_csv(p) for p in sorted(args.probe_dir.glob("*.csv")))
    saved = get_saved(df).reset_index()
    own = saved[saved["train_site"] == saved["site"]].set_index("train_site")
    scale = own[8].mean()
    curve = (own[list(DEPTHS)].mean() / scale).to_numpy()
    units = np.array(DEPTHS, dtype=np.float64)
    fits = [
        (float(np.sum((get_ramp(units, s, e) - curve) ** 2)), s, e)
        for s in np.arange(0.0, 2.01, 0.05)
        for e in np.arange(2.0, 8.01, 0.1)
        if e > s + 0.2
    ]
    _, start, end = min(fits)
    per_pass = args.routine_size / args.num_training_tasks
    print(f"own gain by depth {dict(zip(DEPTHS, curve.round(2)))}")
    print(
        f"ramp: {start:.2f} to {end:.2f} units, "
        f"{start / per_pass:.2f} to {end / per_pass:.2f} passes"
    )
    own_corridor = own["corridor"]
    trained = saved["train_site"].map(own_corridor)
    groups = {
        "same routine": (saved["site"] != saved["train_site"])
        & (saved["corridor"] == trained)
        & (saved["site"] == 1)
        & (saved["train_site"] == 0),
        "other corridor": saved["corridor"] != trained,
    }
    for name, mask in groups.items():
        rows = saved[mask]
        if rows.empty:
            continue
        relative = (rows[list(DEPTHS)].mean() / scale).to_numpy()
        transfer = fit_transfer(relative, start, end)
        print(f"{name}: {dict(zip(DEPTHS, relative.round(2)))}, transfer {transfer}")


if __name__ == "__main__":
    main()
