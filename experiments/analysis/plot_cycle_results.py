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
    "current_belief": "Current belief only",
    "dreaming": "Dreaming",
    "scratch": "Plan from scratch",
}

# Each approach keeps its color in every plot. Approaches without an entry take
# the remaining colors in this order.
APPROACH_TO_COLOR = {
    "dreaming": "#2a78d6",
    "scratch": "#eb6834",
    "current_belief": "#1baf7a",
}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

ENV_TO_DISPLAY_NAME = {
    "two_link_arm": "Two-link arm",
}

# Each panel plots one column of eval_results.csv: (column, axis label, scale).
METRICS = [
    ("planning_wall_time", "Planning time per task (ms)", 1000.0),
    ("success", "Success rate", 1.0),
]


def _get_colors(approaches: list[str]) -> dict[str, str]:
    extra = iter(EXTRA_COLORS)
    return {a: APPROACH_TO_COLOR.get(a) or next(extra) for a in sorted(approaches)}


def _main(results_dir: Path, outfile: Path) -> None:
    df = combine_results_csvs(results_dir, "eval_results.csv")
    envs = sorted(df["env"].unique())
    approaches = sorted(df["approach"].unique())
    colors = _get_colors(approaches)

    fig, axes = plt.subplots(
        len(envs),
        len(METRICS),
        figsize=(5.5 * len(METRICS), 4.0 * len(envs)),
        squeeze=False,
    )
    for row, env in enumerate(envs):
        env_df = df[df["env"] == env]
        num_seeds = env_df["seed"].nunique()
        for col, (metric, label, scale) in enumerate(METRICS):
            ax = axes[row, col]
            summary = summarize_over_seeds(env_df, metric)
            for approach in approaches:
                approach_summary = summary[summary["approach"] == approach]
                cycles = approach_summary["cycle"].to_numpy()
                mean = approach_summary["mean"].to_numpy() * scale
                sem = approach_summary["sem"].to_numpy() * scale
                ax.fill_between(
                    cycles, mean - sem, mean + sem, color=colors[approach], alpha=0.2
                )
                ax.plot(
                    cycles,
                    mean,
                    color=colors[approach],
                    linewidth=2,
                    solid_capstyle="round",
                    label=APPROACH_TO_DISPLAY_NAME.get(approach, approach),
                )
            ax.set_xlabel("Cycle")
            ax.set_ylabel(label)
            ax.set_title(
                f"{ENV_TO_DISPLAY_NAME.get(env, env)} ({num_seeds} seeds)",
                fontsize=11,
            )
            if metric == "success":
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
    fig.text(
        0.5,
        -0.02,
        "Lines show the mean over seeds; shaded bands show one standard error.",
        ha="center",
        fontsize=9,
        color="#52514e",
    )
    # Leave room above the panel titles for the legend.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.08 / len(envs)))
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _print_final_cycle(df)
    print(f"Wrote plot to {outfile}")


def _print_final_cycle(df: pd.DataFrame) -> None:
    """Print the final cycle's results, so they are visible without the plot."""
    final = df[df["cycle"] == df["cycle"].max()]
    for (env, approach), group in final.groupby(["env", "approach"]):
        time_ms = 1000.0 * np.mean(group["planning_wall_time"])
        success = np.mean(group["success"])
        print(
            f"{env}, {approach}, cycle {group['cycle'].iloc[0]}: "
            f"planning time {time_ms:.2f} ms, success rate {success:.2f}"
        )


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
