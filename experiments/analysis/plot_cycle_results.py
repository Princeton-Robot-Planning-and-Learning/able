"""Plot offline evaluation results against cycles, with one line per approach.

Example:
    python experiments/analysis/plot_cycle_results.py --results_dir logs/comparison
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from analysis_utils import combine_results_csvs, summarize_over_seeds
from matplotlib import pyplot as plt

APPROACH_TO_DISPLAY_NAME = {
    "belief_coverage": "Belief coverage",
    "current_belief": "Current belief only",
    "dreaming": "Dreaming",
    "prior_only": "Prior only (first sleep)",
    "scratch": "Plan from scratch",
}

# Each approach keeps its color in every plot. Approaches without an entry take
# the remaining colors in this order.
APPROACH_TO_COLOR = {
    "dreaming": "#2a78d6",
    "scratch": "#eb6834",
    "current_belief": "#1baf7a",
    "prior_only": "#eda100",
}
EXTRA_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]

# The prior-only line often overlaps the scratch line, and their colors are close,
# so it is also dashed.
APPROACH_TO_LINESTYLE = {
    "prior_only": "--",
}

ENV_TO_DISPLAY_NAME = {
    "two_link_arm": "Two-link arm",
    "limb": "Limb repositioning",
    "friction": "Friction",
    "human_reach": "Human reach",
}

# Each panel plots one column of eval_results.csv: (column, axis label). Runs that
# record timeouts get a third panel; their planning time leaves out timed-out tasks.
METRICS = [
    ("planning_wall_time", "Planning time per task"),
    ("success", "Success rate"),
]
TIMEOUT_METRIC = ("timed_out", "Timeout rate")
RATE_METRICS = ("success", "timed_out")

CAPTION = "Lines show the mean over seeds; shaded bands show one standard error."
TIMEOUT_CAPTION = (
    "Timed-out tasks are excluded from planning time, counted as failures, and "
    "shown on the right."
)

# The table's windows of cycles: (name, first cycle, last cycle or None for the end).
TABLE_WINDOWS = [
    ("cycle 0", 0, 0),
    ("cycles 1–4", 1, 4),
    ("last 5 cycles", -5, None),
    ("all cycles", 0, None),
]


def _get_colors(approaches: list[str]) -> dict[str, str]:
    extra = iter(EXTRA_COLORS)
    return {a: APPROACH_TO_COLOR.get(a) or next(extra) for a in sorted(approaches)}


def _has_timeouts(df: pd.DataFrame) -> bool:
    """Whether the runs record timeouts; runs from before they were recorded do
    not."""
    return "timed_out" in df.columns


def _finished(df: pd.DataFrame) -> pd.DataFrame:
    """The rows whose planning did not time out, which planning time averages."""
    if not _has_timeouts(df):
        return df
    return df[~df["timed_out"]]


def _main(results_dir: Path, outfile: Path) -> None:
    df = combine_results_csvs(results_dir, "eval_results.csv")
    metrics = list(METRICS)
    if _has_timeouts(df):
        # Runs without the column, mixed with runs that have it, had no timeouts
        # recorded.
        df["timed_out"] = df["timed_out"].fillna(False).astype(bool)
        metrics.append(TIMEOUT_METRIC)
    envs = sorted(df["env"].unique())
    approaches = sorted(df["approach"].unique())
    colors = _get_colors(approaches)

    fig, axes = plt.subplots(
        len(envs),
        len(metrics),
        figsize=(5.5 * len(metrics), 4.0 * len(envs)),
        squeeze=False,
    )
    for row, env in enumerate(envs):
        env_df = df[df["env"] == env]
        num_seeds = env_df["seed"].nunique()
        # Every panel spans every cycle, so a cycle whose tasks all timed out
        # leaves a gap in the planning-time line instead of being skipped.
        cycles = np.sort(env_df["cycle"].unique())
        for col, (metric, label) in enumerate(metrics):
            scale, unit = 1.0, ""
            metric_df = env_df
            # Markers keep a planning time between two timed-out cycles visible.
            marker = None
            if metric == "planning_wall_time":
                scale, unit = _get_time_scale(env_df[metric])
                label = f"{label} ({unit})"
                metric_df = _finished(env_df)
                marker = "o" if _has_timeouts(df) else None
            ax = axes[row, col]
            summary = summarize_over_seeds(metric_df, metric)
            for approach in approaches:
                approach_summary = (
                    summary[summary["approach"] == approach]
                    .set_index("cycle")
                    .reindex(cycles)
                )
                mean = approach_summary["mean"].to_numpy(dtype=np.float64) * scale
                sem = approach_summary["sem"].to_numpy(dtype=np.float64) * scale
                ax.fill_between(
                    cycles, mean - sem, mean + sem, color=colors[approach], alpha=0.2
                )
                ax.plot(
                    cycles,
                    mean,
                    color=colors[approach],
                    linestyle=APPROACH_TO_LINESTYLE.get(approach, "-"),
                    linewidth=2,
                    marker=marker,
                    markersize=3,
                    solid_capstyle="round",
                    label=APPROACH_TO_DISPLAY_NAME.get(approach, approach),
                )
            pad = 0.05 * (cycles[-1] - cycles[0]) if len(cycles) > 1 else 0.5
            ax.set_xlim(cycles[0] - pad, cycles[-1] + pad)
            ax.set_xlabel("Cycle")
            ax.set_ylabel(label)
            ax.set_title(
                f"{ENV_TO_DISPLAY_NAME.get(env, env)} ({num_seeds} seeds)",
                fontsize=11,
            )
            if metric in RATE_METRICS:
                ax.set_ylim(0.0, 1.05)
            else:
                ax.set_ylim(bottom=0.0)
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.grid(color="#e4e3df", linewidth=1)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(approaches),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    caption = CAPTION
    if _has_timeouts(df):
        caption += "\n" + TIMEOUT_CAPTION
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        va="top",
        fontsize=9,
        color="#52514e",
    )
    # Leave room above the panel titles for the legend.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.08 / len(envs)))
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _print_final_cycle(df)
    print(f"Wrote plot to {outfile}")
    table = _make_table(df, combine_results_csvs(results_dir, "cycle_results.csv"))
    table_file = outfile.with_name("summary_table.md")
    table_file.write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote table to {table_file}")


def _get_time_scale(times: pd.Series) -> tuple[float, str]:
    """Milliseconds for fast planners and seconds for slow ones."""
    return (1.0, "s") if times.mean() > 1.0 else (1000.0, "ms")


def _over_seeds(df: pd.DataFrame, metric: str, first: int, last: int | None) -> str:
    """The metric averaged over each seed's rows in the cycles, then the mean and
    standard error over seeds."""
    max_cycle = int(df["cycle"].max())
    first = first if first >= 0 else max(0, max_cycle + 1 + first)
    last = max_cycle if last is None else last
    window = df[(df["cycle"] >= first) & (df["cycle"] <= last)]
    # A seed whose every task in the window timed out has no planning time.
    per_seed = window.groupby("seed")[metric].mean().dropna()
    if per_seed.empty:
        return "–"
    values = per_seed.to_numpy(dtype=np.float64)
    sem = (
        0.0 if len(values) < 2 else float(np.std(values, ddof=1) / np.sqrt(len(values)))
    )
    return f"{values.mean():.2f} ± {sem:.2f}"


def _make_table(eval_df: pd.DataFrame, cycle_df: pd.DataFrame) -> str:
    """A Markdown table per environment of planning time, success rate, timeout
    rate (when the runs record timeouts), and sleep time. Planning time leaves out
    timed-out tasks. ± is one standard error over seeds."""
    sections = []
    for env in sorted(eval_df["env"].unique()):
        env_df = eval_df[eval_df["env"] == env]
        approaches = sorted(env_df["approach"].unique())
        scale, unit = _get_time_scale(env_df["planning_wall_time"])
        planning_time = env_df["planning_wall_time"] * scale
        if _has_timeouts(env_df):
            # NaN rows are skipped in each seed's mean.
            planning_time = planning_time.where(~env_df["timed_out"])
        env_df = env_df.assign(planning_time=planning_time)
        names = [APPROACH_TO_DISPLAY_NAME.get(a, a) for a in approaches]
        lines = [
            f"### {ENV_TO_DISPLAY_NAME.get(env, env)}",
            "",
            "| | " + " | ".join(names) + " |",
            "| --- |" + " --- |" * len(names),
        ]
        rows = [
            (f"Planning time ({unit}), {name}", "planning_time", first, last)
            for name, first, last in TABLE_WINDOWS
        ]
        rows.append(("Success rate, all cycles", "success", 0, None))
        if _has_timeouts(env_df):
            rows.append(("Timeout rate, all cycles", "timed_out", 0, None))
        for label, metric, first, last in rows:
            cells = [
                _over_seeds(env_df[env_df["approach"] == a], metric, first, last)
                for a in approaches
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        env_cycles = cycle_df[cycle_df["env"] == env]
        cells = [
            _over_seeds(
                env_cycles[env_cycles["approach"] == a], "sleep_wall_time", 0, None
            )
            for a in approaches
        ]
        lines.append("| Sleep time (s), all cycles | " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    footer = "± is one standard error over seeds."
    return "\n\n".join(sections) + "\n\n" + footer + "\n"


def _print_final_cycle(df: pd.DataFrame) -> None:
    """Print the final cycle's results, so they are visible without the plot.

    Planning time leaves out timed-out tasks, which are counted after it.
    """
    final = df[df["cycle"] == df["cycle"].max()]
    for (env, approach), group in final.groupby(["env", "approach"]):
        scale, unit = _get_time_scale(df[df["env"] == env]["planning_wall_time"])
        finished = _finished(group)
        time = (
            f"{scale * np.mean(finished['planning_wall_time']):.2f} {unit}"
            if len(finished)
            else "none finished"
        )
        success = np.mean(group["success"])
        line = (
            f"{env}, {approach}, cycle {group['cycle'].iloc[0]}: "
            f"planning time {time}, success rate {success:.2f}"
        )
        if len(finished) < len(group):
            line += f", {len(group) - len(finished)} timed out"
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Defaults to cycle_results.png in the results directory.",
    )
    args = parser.parse_args()
    _main(args.results_dir, args.outfile or args.results_dir / "cycle_results.png")
