"""Plot what training for one friction site does at every site, from the CSVs of
experiments/probe_site_transfer.py.

    python experiments/analysis/plot_site_transfer.py --results_dir probes/ \
        --out transfer.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

# How each evaluated site relates to the trained site 0.
RELATIONS = {
    0: "trained site",
    1: "same routine",
    4: "same corridor, other routine",
    2: "other corridor",
    3: "other corridor",
}
COLORS = {
    "trained site": "#1f4fd1",
    "same routine": "#00a0b0",
    "same corridor, other routine": "#e07b00",
    "other corridor": "#888888",
}


def load(results_dir: Path) -> pd.DataFrame:
    """Every probe row, with each site's relation to the trained site."""
    df = pd.concat(pd.read_csv(path) for path in sorted(results_dir.glob("*.csv")))
    df["relation"] = df["site"].map(RELATIONS)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per overlap, relation, and depth: the mean and standard error over seeds of
    each seed's mean simulations and success, and the change in simulations from
    depth 0 on the same tasks."""
    per_seed = df.groupby(["overlap", "relation", "seed", "depth"])[
        ["num_sims", "success"]
    ].mean()
    sims = per_seed["num_sims"].unstack("depth")
    saved = sims.rsub(sims[0], axis=0).stack()
    per_seed = per_seed.join(pd.DataFrame({"saved": saved}))
    grouped = per_seed.groupby(["overlap", "relation", "depth"])
    return grouped.mean().join(grouped.sem(), rsuffix="_se")


def plot(summary: pd.DataFrame, out: Path) -> None:
    """Simulations saved and success by depth, one column per overlap."""
    overlaps = sorted(summary.index.get_level_values("overlap").unique())
    fig, axes = plt.subplots(2, len(overlaps), figsize=(5.5 * len(overlaps), 7))
    axes = np.atleast_2d(axes).reshape(2, len(overlaps))
    for col, overlap in enumerate(overlaps):
        for relation, color in COLORS.items():
            if (overlap, relation) not in summary.droplevel("depth").index:
                continue
            rows = summary.loc[(overlap, relation)]
            for row, (metric, label) in enumerate(
                (("saved", "Simulations saved per task"), ("success", "Success"))
            ):
                ax = axes[row, col]
                ax.errorbar(
                    rows.index,
                    rows[metric],
                    yerr=rows[f"{metric}_se"],
                    label=relation,
                    color=color,
                    marker="o",
                    capsize=3,
                )
                ax.set_ylabel(label)
                ax.set_xlabel("Units trained for site 0")
        axes[0, col].axhline(0.0, color="black", lw=0.5)
        axes[0, col].set_title(f"region_overlap = {overlap:g}")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    """Load, summarize, print, and plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(load(args.results_dir))
    print(summary[["num_sims", "saved", "saved_se", "success", "success_se"]].round(2))
    plot(summary, args.out)


if __name__ == "__main__":
    main()
