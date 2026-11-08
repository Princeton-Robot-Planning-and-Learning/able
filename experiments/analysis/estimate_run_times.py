"""Estimate how long runs will take from short runs, and the Slurm time limits to
ask for.

Each run's cycle_results.csv records the wall time of every cycle. A run of
num_cycles cycles is estimated to take num_cycles times its mean cycle time, plus
the time before its first cycle. Della puts jobs of at most an hour in its test
QOS, which runs only 2 jobs per user at once, so a limit is never below 61
minutes.

Example, after a 2-cycle calibration sweep:
    python experiments/analysis/estimate_run_times.py \
        --results_dir logs/comparison/limb/<sweep> --num_cycles 20
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from analysis_utils import combine_results_csvs

# Della's test QOS takes jobs of up to 60 minutes and runs 2 at once per user.
MIN_LIMIT_MINUTES = 61
# Della's short QOS takes jobs of up to 24 hours.
SHORT_QOS_MINUTES = 24 * 60


def estimate_run_times(
    cycle_df: pd.DataFrame, num_cycles: int, margin: float, setup_seconds: float
) -> pd.DataFrame:
    """One row per environment and approach: the mean and slowest cycle times over
    its runs (s), the estimated time of the slowest run (h), and the time limit to
    ask for (min), margin times the estimate."""
    per_run = cycle_df.groupby(["env", "approach", "seed"])["cycle_wall_time"].mean()
    rows = []
    for (env, approach), cycle_times in per_run.groupby(["env", "approach"]):
        slowest = float(cycle_times.max())
        hours = (setup_seconds + num_cycles * slowest) / 3600
        minutes = max(MIN_LIMIT_MINUTES, math.ceil(margin * hours * 60))
        rows.append(
            {
                "env": env,
                "approach": approach,
                "mean_cycle_s": round(float(cycle_times.mean())),
                "slowest_cycle_s": round(slowest),
                "estimated_hours": round(hours, 1),
                "limit_minutes": minutes,
                "short_qos": minutes <= SHORT_QOS_MINUTES,
            }
        )
    return pd.DataFrame(rows)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--num_cycles", type=int, default=20)
    parser.add_argument("--margin", type=float, default=1.3)
    parser.add_argument(
        "--setup_seconds",
        type=float,
        default=300.0,
        help="Time before the first cycle: building the domain and the workers.",
    )
    args = parser.parse_args()
    cycle_df = combine_results_csvs(args.results_dir, "cycle_results.csv")
    estimates = estimate_run_times(
        cycle_df, args.num_cycles, args.margin, args.setup_seconds
    )
    print(estimates.to_string(index=False))


if __name__ == "__main__":
    _main()
