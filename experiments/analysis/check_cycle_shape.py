"""Check a comparison's cycle curves against the shape we want (PRs #143, #154).

Graph, per environment and number of queries per cycle: each learner's success
rises, ends above scratch and prior only, and its planning time falls to at most
half its start and of scratch's; scratch's cycle-0 success is not near 1. Belief:
without queries its error and standard deviation (in prior standard deviations) end
below the prior's; with queries its error ends near zero.

Example:
    python experiments/analysis/check_cycle_shape.py --results_dir logs/belief
"""

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from plot_belief_cycles import load_belief_rows

LEARNERS = ("current_belief", "dreaming", "belief_coverage")
BASELINES = ("scratch", "prior_only")


def load_eval_rows(results_dir: Path) -> pd.DataFrame:
    """Every run's eval_results.csv, with its environment, approach, seed and
    queries per cycle."""
    frames = []
    for config_path in sorted(results_dir.rglob("config.yaml")):
        results_path = config_path.parent / "eval_results.csv"
        if not results_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig)
        df = pd.read_csv(results_path)
        df["env"] = cfg.env_name
        df["approach"] = cfg.approach_name
        df["seed"] = cfg.seed
        df["queries"] = int(
            OmegaConf.select(cfg, "budgets.information_gathering.max_iterations") or 0
        )
        frames.append(df)
    assert frames, f"No eval_results.csv found under {results_dir}"
    return pd.concat(frames, ignore_index=True)


def _window_means(df: pd.DataFrame, column: str, cycles: range) -> pd.Series:
    rows = df[df["cycle"].isin(cycles)]
    per_seed = rows.groupby(["approach", "seed"])[column].mean()
    return per_seed.groupby("approach").mean()


def check_graph(df: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    """One row per learner and check."""
    last = int(df["cycle"].max())
    early, late = range(0, args.window), range(last - args.window + 1, last + 1)
    success_early = _window_means(df, "success", early)
    success_late = _window_means(df, "success", late)
    time_early = _window_means(df, "planning_wall_time", early)
    time_late = _window_means(df, "planning_wall_time", late)
    baselines = [a for a in BASELINES if a in success_late]
    rows = []
    if "scratch" in success_late:
        start = _window_means(df, "success", range(0, 1))["scratch"]
        rows.append(_row("scratch", "cycle-0 success", start, start <= args.max_start))
    for learner in [a for a in LEARNERS if a in success_late]:
        rise = success_late[learner] - success_early[learner]
        rows.append(_row(learner, "success rise", rise, rise >= args.min_gap))
        if baselines:
            gap = success_late[learner] - success_late[baselines].max()
            rows.append(
                _row(learner, "late gap to baselines", gap, gap >= args.min_gap)
            )
        drop = time_late[learner] / time_early[learner]
        rows.append(_row(learner, "planning time, late / early", drop, drop <= 0.5))
        if "scratch" in time_late:
            ratio = time_late[learner] / time_late["scratch"]
            rows.append(
                _row(learner, "planning time, late / scratch", ratio, ratio <= 0.5)
            )
    return rows


def check_belief(
    df: pd.DataFrame, queries: int, args: argparse.Namespace
) -> list[dict]:
    """One row per approach and check, on the error and std averaged over
    parameters."""
    per_seed = (
        df.groupby(["approach", "seed", "cycle"])[["error_ratio", "std_ratio"]]
        .mean()
        .reset_index()
    )
    last = int(per_seed["cycle"].max())
    late = per_seed[per_seed["cycle"] > last - args.window]
    prior = per_seed[per_seed["cycle"] == -1]
    rows = []
    for approach in sorted(per_seed["approach"].unique()):
        end = late[late["approach"] == approach][["error_ratio", "std_ratio"]].mean()
        start = prior[prior["approach"] == approach][
            ["error_ratio", "std_ratio"]
        ].mean()
        if queries == 0:
            for measure in ("error_ratio", "std_ratio"):
                change = end[measure] / start[measure]
                rows.append(
                    _row(
                        approach,
                        f"belief {measure}, late / prior",
                        change,
                        change <= 1 - args.min_belief_drop,
                    )
                )
        else:
            error = end["error_ratio"]
            rows.append(
                _row(
                    approach, "belief error, late", error, error <= args.max_query_error
                )
            )
    return rows


def _row(approach: str, check: str, value: float, passed: bool) -> dict:
    return {
        "approach": approach,
        "check": check,
        "value": round(float(value), 3),
        "pass": "PASS" if passed else "FAIL",
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min_gap", type=float, default=0.15)
    parser.add_argument("--max_start", type=float, default=0.8)
    parser.add_argument("--min_belief_drop", type=float, default=0.05)
    parser.add_argument("--max_query_error", type=float, default=0.2)
    parser.add_argument("--no_belief", action="store_true")
    args = parser.parse_args()

    eval_df = load_eval_rows(args.results_dir)
    belief_df = None if args.no_belief else load_belief_rows(args.results_dir)
    for (env, queries), group in eval_df.groupby(["env", "queries"]):
        rows = check_graph(group, args)
        if belief_df is not None:
            beliefs = belief_df[
                (belief_df["env"] == env) & (belief_df["queries"] == queries)
            ]
            if len(beliefs):
                rows += check_belief(beliefs, int(str(queries)), args)
        table = pd.DataFrame(rows)
        passed = (table["pass"] == "PASS").sum() if len(table) else 0
        print(f"\n## {env}, {queries} queries per cycle: {passed}/{len(table)} pass")
        print(table.to_string(index=False))


if __name__ == "__main__":
    _main()
