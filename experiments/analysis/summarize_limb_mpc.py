"""Compare two runs of diagnose_limb_mpc.py, paired by variant, seed, and belief.

Prints markdown tables: per belief, the success rate, how often planning failed or
timed out, how often execution passed the limit or broke a constraint, and the mean
rollouts and planning time; then the success of each variant under each belief.

Example:
    python experiments/analysis/summarize_limb_mpc.py \
        --runs main=main.jsonl branch=branch.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load(path: Path, name: str) -> pd.DataFrame:
    """One row per job, with the run's name."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    # A rerun appends the jobs that raised; keep each job's last row.
    df = pd.DataFrame(rows).drop_duplicates(["variant", "seed", "belief"], keep="last")
    df["run"] = name
    df["rollouts"] = [
        (stats or {}).get("num_rollouts", float("nan")) for stats in df["stats"]
    ]
    df["retargeted"] = [
        (stats or {}).get("retargeted", float("nan")) for stats in df["stats"]
    ]
    for column in ("violated", "failure", "timed_out", "error"):
        if column not in df:
            df[column] = None
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run and belief."""
    groups = df.groupby(["belief", "run"])
    return pd.DataFrame(
        {
            "tasks": groups.size(),
            "success": groups["success"].mean(),
            "plan failed": groups["planned"].apply(lambda s: float((~s).mean())),
            "timed out": groups["timed_out"].apply(lambda s: float(s.fillna(0).mean())),
            "passed limit": groups["violated"].apply(
                lambda s: float(s.map(lambda v: v is True).mean())
            ),
            "broke constraint": groups["failure"].apply(
                lambda s: float(
                    s.fillna("")
                    .map(lambda f: bool(f) and f != "stopped before the goal")
                    .mean()
                )
            ),
            "retargeted": groups["retargeted"].mean(),
            "rollouts": groups["rollouts"].mean(),
            "plan time (s)": groups["plan_time"].mean(),
        }
    )


def main() -> None:
    """Print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="name=path.jsonl")
    args = parser.parse_args()
    frames = []
    for spec in args.runs:
        name, path = spec.split("=", 1)
        frames.append(load(Path(path), name))
    df = pd.concat(frames, ignore_index=True)
    errors = df[df["error"].notna()]
    if len(errors):
        print(f"{len(errors)} jobs raised:\n{errors[['run', 'variant', 'seed']]}\n")
    df = df[df["error"].isna()]
    print(summarize(df).round(2).to_markdown())
    print()
    by_variant = df.pivot_table(
        index="variant", columns=["belief", "run"], values="success", aggfunc="mean"
    )
    print(by_variant.round(2).to_markdown())
    failures = df[~df["success"].astype(bool)]
    reasons = (
        failures["plan_failure"]
        .fillna("")
        .where(
            ~failures["planned"].astype(bool), "exec: " + failures["failure"].fillna("")
        )
    )
    # Drop the numbers, such as the error a plan diverged at.
    reasons = reasons.str.replace(r"[-\d.]+\d", "#", regex=True)
    print()
    print(
        reasons.groupby([failures["belief"], failures["run"], reasons])
        .size()
        .to_markdown()
    )


if __name__ == "__main__":
    main()
