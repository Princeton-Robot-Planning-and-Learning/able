"""Summarize how much one approach gains over a baseline, optionally for each value of a
swept setting.

Both approaches must be run with the same seeds. For each seed and window of cycles,
the time saved is the baseline's planning time per evaluation task minus the
approach's, and the hit-rate gain is the approach's rate of planning with cached plans
minus the baseline's. Hit rates are also compared on revision cycles only, the cycles
whose belief changed between sleep and evaluation. A learner that ignores belief
updates is most likely to miss on those cycles. The summary reports the mean and
standard error over seeds, along with each approach's success rate and the fraction of
cycles that revised the belief.

The table goes to gap_summary.md in the results directory. With --lever, which names
a setting in the resolved config such as env.domain.edge_probability, there is one
row group per value and a plot of the gaps against the value (gap_summary.png).

Example:
    python experiments/analysis/summarize_gap.py --results_dir logs/gap_sweep/<run> \
        --lever env.domain.edge_probability
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from analysis_utils import combine_results_csvs
from matplotlib import pyplot as plt
from plot_cycle_results import APPROACH_TO_COLOR, APPROACH_TO_DISPLAY_NAME

# The group column when no setting is swept.
NO_LEVER = "setting"

# Early windows are drawn lighter than late ones, in one hue.
WINDOW_COLORS = ["#86b6ef", "#1c5cab"]
WINDOW_MARKERS = ["o", "s"]


def get_windows(num_cycles: int) -> list[tuple[str, int, int]]:
    """Name and inclusive cycle range of each window: an early window, the second half
    of the run, and all cycles."""
    half = num_cycles // 2
    return [
        (f"Cycles 1–{min(5, num_cycles - 1)}", 1, min(5, num_cycles - 1)),
        (f"Cycles {half}–{num_cycles - 1}", half, num_cycles - 1),
        ("All cycles", 0, num_cycles - 1),
    ]


def get_seed_metrics(
    eval_df: pd.DataFrame, cycle_df: pd.DataFrame, lever: str
) -> pd.DataFrame:
    """One row per lever value, approach, seed, and window.

    A planning failure counts as a miss. The revision hit rate is NaN for a seed with no
    revision cycles in the window.
    """
    keys = [lever, "approach", "seed"]
    df = eval_df.merge(
        cycle_df[keys + ["cycle", "belief_changed"]], on=keys + ["cycle"]
    )
    df["hit"] = df["used_warm_start"].fillna(0.0)
    num_cycles = int(df["cycle"].max()) + 1
    frames = []
    for window, low, high in get_windows(num_cycles):
        in_window = df[(df["cycle"] >= low) & (df["cycle"] <= high)]
        grouped = in_window.groupby(keys)
        metrics = grouped.agg(
            planning_wall_time=("planning_wall_time", "mean"),
            hit_rate=("hit", "mean"),
            success=("success", "mean"),
        )
        revisions = in_window[in_window["belief_changed"]]
        metrics["revision_hit_rate"] = revisions.groupby(keys)["hit"].mean()
        cycles = cycle_df[(cycle_df["cycle"] >= low) & (cycle_df["cycle"] <= high)]
        metrics["revision_rate"] = cycles.groupby(keys)["belief_changed"].mean()
        metrics["window"] = window
        frames.append(metrics.reset_index())
    return pd.concat(frames, ignore_index=True)


def get_gaps(
    seed_metrics: pd.DataFrame, approach: str, baseline: str, lever: str
) -> pd.DataFrame:
    """Mean, standard error, and number of seeds of each gap, per lever value and
    window.

    Seeds missing from either approach are left out.
    """
    index = [lever, "window", "seed"]
    ours = seed_metrics[seed_metrics["approach"] == approach].set_index(index)
    theirs = seed_metrics[seed_metrics["approach"] == baseline].set_index(index)
    assert ours.index.is_unique and theirs.index.is_unique, (
        "Each approach has several runs with the same seed and lever value, so the "
        "runs differ in a setting that the lever does not name"
    )
    ours, theirs = ours.align(theirs, join="inner")
    assert len(ours), f"No seeds have results for both {approach} and {baseline}"
    gaps = pd.DataFrame(
        {
            "time_saved": theirs["planning_wall_time"] - ours["planning_wall_time"],
            "hit_rate_gain": ours["hit_rate"] - theirs["hit_rate"],
            "revision_hit_rate_gain": ours["revision_hit_rate"]
            - theirs["revision_hit_rate"],
            "approach_success": ours["success"],
            "baseline_success": theirs["success"],
            "baseline_revision_rate": theirs["revision_rate"],
        }
    )
    summary = gaps.groupby([lever, "window"]).agg(["mean", "sem", "count"])
    summary.columns = ["_".join(column) for column in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    window_order = list(dict.fromkeys(seed_metrics["window"]))
    summary["window_index"] = summary["window"].map(window_order.index)
    summary = summary.sort_values([lever, "window_index"], ignore_index=True)
    return summary.drop(columns="window_index")


def _format(mean: float, sem: float, scale: float = 1.0, digits: int = 2) -> str:
    if np.isnan(mean):
        return "–"
    sem = 0.0 if np.isnan(sem) else sem
    return f"{scale * mean:.{digits}f} ± {scale * sem:.{digits}f}"


def format_table(gaps: pd.DataFrame, approach: str, baseline: str, lever: str) -> str:
    """A markdown table of the gaps."""
    ours = APPROACH_TO_DISPLAY_NAME.get(approach, approach)
    theirs = APPROACH_TO_DISPLAY_NAME.get(baseline, baseline)
    header = ["Window"] if lever == NO_LEVER else [lever, "Window"]
    header += [
        "Time saved (ms)",
        "Hit-rate gain",
        "Hit-rate gain on revision cycles",
        f"Revision rate ({theirs})",
        f"Success ({ours})",
        f"Success ({theirs})",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for _, row in gaps.iterrows():
        cells = [] if lever == NO_LEVER else [str(row[lever])]
        cells += [
            row["window"],
            _format(row["time_saved_mean"], row["time_saved_sem"], 1000.0),
            _format(row["hit_rate_gain_mean"], row["hit_rate_gain_sem"]),
            _format(
                row["revision_hit_rate_gain_mean"], row["revision_hit_rate_gain_sem"]
            ),
            _format(
                row["baseline_revision_rate_mean"], row["baseline_revision_rate_sem"]
            ),
            _format(row["approach_success_mean"], row["approach_success_sem"]),
            _format(row["baseline_success_mean"], row["baseline_success_sem"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    num_seeds = int(gaps["time_saved_count"].max())
    lines.append("")
    lines.append(
        f"{ours} compared with {theirs}: mean ± standard error over {num_seeds} seeds."
    )
    return "\n".join(lines)


def plot_gaps(
    gaps: pd.DataFrame,
    approach: str,
    baseline: str,
    lever: str,
    outfile: Path,
    log_x: bool = False,
) -> None:
    """Plot the gaps in the early and late windows, and both approaches' success rates
    over all cycles, against the lever value."""
    windows = list(dict.fromkeys(gaps["window"]))
    panels = [
        ("time_saved", "Planning time saved per task (ms)", 1000.0),
        ("revision_hit_rate_gain", "Hit-rate gain on revision cycles", 1.0),
    ]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(15.0, 4.0))
    for ax, (metric, label, scale) in zip(axes, panels):
        for i, window in enumerate(windows[:2]):
            rows = gaps[gaps["window"] == window].sort_values(lever)
            mean = rows[f"{metric}_mean"].to_numpy() * scale
            sem = rows[f"{metric}_sem"].fillna(0.0).to_numpy() * scale
            ax.errorbar(
                rows[lever],
                mean,
                yerr=sem,
                color=WINDOW_COLORS[i],
                marker=WINDOW_MARKERS[i],
                markersize=7,
                linewidth=2,
                capsize=3,
                label=window,
            )
        ax.axhline(0.0, color="#c3c2b7", linewidth=1)
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=9)
    ax = axes[-1]
    rows = gaps[gaps["window"] == windows[-1]].sort_values(lever)
    for name, column in (
        (approach, "approach_success"),
        (baseline, "baseline_success"),
    ):
        ax.errorbar(
            rows[lever],
            rows[f"{column}_mean"],
            yerr=rows[f"{column}_sem"].fillna(0.0),
            color=APPROACH_TO_COLOR.get(name, "#52514e"),
            marker="o",
            markersize=7,
            linewidth=2,
            capsize=3,
            label=APPROACH_TO_DISPLAY_NAME.get(name, name),
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate (all cycles)")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlabel(lever)
        if log_x:
            ax.set_xscale("log", base=2)
        ax.grid(color="#e4e3df", linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    num_seeds = int(gaps["time_saved_count"].max())
    ours = APPROACH_TO_DISPLAY_NAME.get(approach, approach)
    theirs = APPROACH_TO_DISPLAY_NAME.get(baseline, baseline)
    fig.suptitle(f"{ours} compared with {theirs} ({num_seeds} seeds)", fontsize=11)
    fig.text(
        0.5,
        -0.02,
        "Points show the mean over seeds; error bars show one standard error.",
        ha="center",
        fontsize=9,
        color="#52514e",
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _main(
    results_dir: Path, approach: str, baseline: str, lever: str | None, log_x: bool
) -> None:
    keys = [] if lever is None else [lever]
    eval_df = combine_results_csvs(results_dir, "eval_results.csv", keys)
    cycle_df = combine_results_csvs(results_dir, "cycle_results.csv", keys)
    if lever is None:
        lever = NO_LEVER
        eval_df[lever] = "default"
        cycle_df[lever] = "default"
    seed_metrics = get_seed_metrics(eval_df, cycle_df, lever)
    gaps = get_gaps(seed_metrics, approach, baseline, lever)
    table = format_table(gaps, approach, baseline, lever)
    print(table)
    (results_dir / "gap_summary.md").write_text(table + "\n")
    print(f"Wrote table to {results_dir / 'gap_summary.md'}")
    if lever != NO_LEVER:
        outfile = results_dir / "gap_summary.png"
        plot_gaps(gaps, approach, baseline, lever, outfile, log_x)
        print(f"Wrote plot to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--approach", default="dreaming")
    parser.add_argument("--baseline", default="current_belief")
    parser.add_argument(
        "--lever",
        default=None,
        help="A setting in the resolved config to group by, such as "
        "env.domain.edge_probability.",
    )
    parser.add_argument(
        "--log_x", action="store_true", help="Plot the lever on a log scale."
    )
    args = parser.parse_args()
    _main(args.results_dir, args.approach, args.baseline, args.lever, args.log_x)
