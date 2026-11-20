"""Plot the planning-cost landscape written by experiments/run_landscape.py.

The cost of planning for a task is the planner's work as a fraction of the
largest work in the results, plus 1 if the task failed. Contexts are placed on
one axis by one of these orders:
- transfer: a spectral ordering of the contexts by which runs lower their cost,
  so that contexts that benefit from the same training sit together;
- features: the first principal component of the beliefs' moment features;
- cycle: visited contexts by cycle, then trajectory.

The figures are:
- learning_grid.png: each training context's cost at each depth of its own run,
  as in a grid of θ columns and steps of compute;
- transfer_<depth>.png: how much each run at a cross depth lowers the cost of
  every context, relative to the untrained model;
- kernel.png: that reduction against the distance between the beliefs' moment
  features;
- embedding.png: the contexts in two dimensions, colored by the reduction from a
  few runs;
- curves.png: each training context's learning curve.

Usage:
    python experiments/analysis/plot_landscape.py --results_dir <out_dir>
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# pylint: disable-next=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402

# pylint: disable-next=wrong-import-position
import numpy as np  # noqa: E402

# pylint: disable-next=wrong-import-position
import pandas as pd  # noqa: E402

# pylint: disable-next=wrong-import-position
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# pylint: disable-next=wrong-import-position
from omegaconf import OmegaConf  # noqa: E402

# The planner statistic that counts each environment's planning work.
WORK_STATS = {
    "two_link_arm": "num_samples",
    "two_link_arm_sampled": "num_samples",
    "friction": "num_sims",
    "limb": "num_rollouts",
    "human_reach": "stream_calls",
}
BASE_JOB = -1

# Darker is cheaper, as in a grid of planning costs.
COST_CMAP = LinearSegmentedColormap.from_list(
    "cost", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f7f9fc"]
)
# Blue lowers the cost and red raises it, around a gray zero.
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "gain", ["#8f1f1f", "#e34948", "#f0efec", "#3987e5", "#0d366b"]
)
INK = "#3a3a38"


@dataclass(frozen=True)
class Landscape:
    """Mean costs over tasks and replicates.

    base[i] is the untrained model's cost on context i, diag[j][depth] is run j's
    cost on its own context, and cross[depth] is a (training context, context)
    table of costs.
    """

    contexts: pd.DataFrame
    base: pd.Series
    diag: pd.DataFrame
    cross: dict[int, pd.DataFrame]
    depths: list[int]


def load_landscape(results_dir: Path) -> Landscape:
    """Read the contexts and every finished job's rows and average their costs."""
    cfg = OmegaConf.load(results_dir / "landscape_config.yaml")
    contexts = pd.read_csv(results_dir / "contexts.csv").set_index("index")
    units = sorted((results_dir / "evals").glob("*.csv"))
    assert units, f"No finished evaluations in {results_dir / 'evals'}"
    rows = pd.concat([pd.read_csv(path) for path in units], ignore_index=True)
    work = rows[f"stat_{WORK_STATS[str(cfg.env_name)]}"].fillna(0.0)
    rows["cost"] = work / work.max() + (1.0 - rows["success"].astype(float))
    per_run = (
        rows.groupby(["train_context", "replicate", "depth", "eval_context"])["cost"]
        .mean()
        .groupby(["train_context", "depth", "eval_context"])
        .mean()
        .reset_index()
    )
    is_base = per_run["train_context"] == BASE_JOB
    base = per_run[is_base].set_index("eval_context")["cost"]
    trained = per_run[~is_base]
    own = trained[trained["train_context"] == trained["eval_context"]]
    diag = own.pivot(index="train_context", columns="depth", values="cost")
    diag.insert(0, 0, base.reindex(diag.index))
    cross = {
        depth: trained[trained["depth"] == depth].pivot(
            index="train_context", columns="eval_context", values="cost"
        )
        for depth in sorted(cfg.landscape.cross_depths)
        if np.any(trained["depth"] == depth)
    }
    return Landscape(contexts, base, diag, cross, [0] + sorted(cfg.landscape.depths))


def get_gains(landscape: Landscape, depth: int) -> pd.DataFrame:
    """How much each run at the depth lowers each context's cost below the
    untrained model's."""
    costs = landscape.cross[depth]
    return landscape.base.reindex(costs.columns) - costs


def spectral_order(profiles: np.ndarray) -> np.ndarray:
    """A coordinate per row that places rows with similar profiles together: the
    Fiedler vector of a Gaussian affinity between the rows."""
    distances = np.linalg.norm(profiles[:, None] - profiles[None], axis=-1)
    scale = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    affinity = np.exp(-((distances / scale) ** 2))
    degree = affinity.sum(axis=1)
    laplacian = np.eye(len(profiles)) - affinity / np.sqrt(np.outer(degree, degree))
    _, vectors = np.linalg.eigh(laplacian)
    return vectors[:, 1] / np.sqrt(degree)


def get_order(landscape: Landscape, how: str) -> pd.Series:
    """A one-dimensional coordinate for every context."""
    contexts = landscape.contexts
    features = contexts.filter(like="feature_").to_numpy()
    if how == "cycle":
        return contexts["cycle"] * len(contexts) + contexts["trajectory"]
    if how == "transfer" and landscape.cross:
        gains = get_gains(landscape, max(landscape.cross))
        profiles = gains.reindex(columns=contexts.index).fillna(0.0).to_numpy().T
        coordinate = spectral_order(profiles)
    else:
        centered = features - features.mean(axis=0)
        scale = centered.std(axis=0)
        centered = centered / np.where(scale > 0, scale, 1.0)
        coordinate = np.linalg.svd(centered, full_matrices=False)[2][0] @ centered.T
    if np.corrcoef(coordinate, features[:, 0])[0, 1] < 0:
        coordinate = -coordinate
    return pd.Series(coordinate, index=contexts.index)


def plot_learning_grid(landscape: Landscape, order: pd.Series, path: Path) -> None:
    """Training contexts as columns and depths as rows, colored by cost."""
    diag = landscape.diag.loc[order.reindex(landscape.diag.index).sort_values().index]
    diag = diag.reindex(columns=landscape.depths)
    num_cols = len(diag)
    fig, ax = plt.subplots(figsize=(min(16, 2 + 0.28 * num_cols), 3.6))
    image = ax.imshow(
        diag.to_numpy().T,
        origin="lower",
        aspect="auto",
        cmap=COST_CMAP,
        vmin=0.0,
        vmax=float(np.nanmax(diag.to_numpy())),
        interpolation="nearest",
    )
    if num_cols <= 40:
        for (row, col), value in np.ndenumerate(diag.to_numpy().T):
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if value < 0.5 else INK,
                )
    ax.set_yticks(range(len(landscape.depths)), [str(d) for d in landscape.depths])
    ax.set_ylabel("Sleep iterations (depth)")
    ax.set_xlabel("Training context, in one-dimensional order")
    step = max(1, num_cols // 30)
    ax.set_xticks(
        range(0, num_cols, step), [str(i) for i in diag.index[::step]], fontsize=6
    )
    fig.colorbar(image, ax=ax, label="Cost (darker is cheaper)", pad=0.01)
    ax.set_title("Each context's cost under its own run", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_transfer(landscape: Landscape, order: pd.Series, depth: int, path: Path):
    """Runs as rows and contexts as columns, colored by the cost reduction."""
    gains = get_gains(landscape, depth)
    rows = order.reindex(gains.index).sort_values().index
    cols = order.reindex(gains.columns).sort_values().index
    gains = gains.loc[rows, cols]
    values = gains.to_numpy()
    limit = float(np.nanpercentile(np.abs(values), 98)) or 1.0
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap=GAIN_CMAP,
        norm=TwoSlopeNorm(0.0, -limit, limit),
        interpolation="nearest",
    )
    positions = {c: k for k, c in enumerate(cols)}
    own = [(r, positions[c]) for r, c in enumerate(rows) if c in positions]
    if own:
        ax.scatter(
            [c for _, c in own],
            [r for r, _ in own],
            s=4,
            marker="s",
            facecolors="none",
            edgecolors=INK,
            linewidths=0.4,
        )
    ax.set_xlabel("Evaluation context, in one-dimensional order")
    ax.set_ylabel("Training context of the run")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(
        image, ax=ax, label="Cost reduction from the untrained model", pad=0.01
    )
    ax.set_title(
        f"Transfer after {depth} sleep iterations (outlines: own context)", color=INK
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_kernel(landscape: Landscape, path: Path) -> None:
    """Each run's cost reduction on each context against the distance between
    their beliefs' moment features, with binned medians."""
    features = landscape.contexts.filter(like="feature_")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    for color, depth in zip(colors, sorted(landscape.cross)):
        gains = (
            get_gains(landscape, depth)
            .reset_index()
            .melt(id_vars="train_context", var_name="eval_context", value_name="gain")
            .dropna()
        )
        source = features.loc[gains["train_context"]].to_numpy()
        target = features.loc[gains["eval_context"]].to_numpy()
        distance = np.linalg.norm(source - target, axis=1)
        ax.scatter(distance, gains["gain"], s=3, alpha=0.15, color=color, linewidths=0)
        bins = np.quantile(distance, np.linspace(0, 1, 13))
        which = np.clip(np.digitize(distance, bins[1:-1]), 0, len(bins) - 2)
        centers = [np.median(distance[which == b]) for b in range(len(bins) - 1)]
        medians = [np.median(gains["gain"][which == b]) for b in range(len(bins) - 1)]
        ax.plot(centers, medians, color=color, linewidth=2, label=f"{depth} iterations")
    ax.axhline(0.0, color="#9a9a96", linewidth=0.8)
    ax.set_xlabel("Distance between beliefs (prior standard deviations)")
    ax.set_ylabel("Cost reduction from the untrained model")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Transfer against belief distance (lines: binned medians)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_embedding(landscape: Landscape, order: pd.Series, path: Path) -> None:
    """The contexts on the first two principal components of their features,
    colored by the reduction from four runs spread along the order."""
    depth = max(landscape.cross)
    gains = get_gains(landscape, depth)
    features = landscape.contexts.filter(like="feature_").to_numpy()
    centered = features - features.mean(axis=0)
    basis = np.linalg.svd(centered, full_matrices=False)[2][:2]
    points = pd.DataFrame(centered @ basis.T, index=landscape.contexts.index)
    runs = order.reindex(gains.index).sort_values().index
    picks = [runs[int(q * (len(runs) - 1))] for q in (0.0, 0.33, 0.67, 1.0)]
    limit = float(np.nanpercentile(np.abs(gains.to_numpy()), 98)) or 1.0
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    for ax, run in zip(axes, picks):
        row = gains.loc[run].reindex(points.index)
        image = ax.scatter(
            points[0],
            points[1],
            c=row,
            s=14,
            cmap=GAIN_CMAP,
            norm=TwoSlopeNorm(0.0, -limit, limit),
            linewidths=0,
        )
        ax.scatter(
            points.loc[run, 0],
            points.loc[run, 1],
            marker="*",
            s=140,
            facecolors="none",
            edgecolors=INK,
            linewidths=1.0,
        )
        ax.set_title(f"Run under context {run}", color=INK, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=list(axes), label="Cost reduction", pad=0.01)
    fig.suptitle(
        f"Transfer after {depth} iterations on the first two principal "
        "components of belief features (star: training context)",
        color=INK,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_curves(landscape: Landscape, path: Path) -> None:
    """Every training context's cost against depth, with the median."""
    diag = landscape.diag.reindex(columns=landscape.depths)
    x = np.arange(len(landscape.depths))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for _, curve in diag.iterrows():
        ax.plot(x, curve.to_numpy(), color="#9a9a96", linewidth=0.6, alpha=0.4)
    ax.plot(x, diag.median().to_numpy(), color="#2a78d6", linewidth=2, label="Median")
    best = np.minimum.accumulate(diag.to_numpy(), axis=1)
    ax.plot(
        x,
        np.nanmedian(best, axis=0),
        color="#eb6834",
        linewidth=2,
        linestyle="--",
        label="Median, keeping the best so far",
    )
    ax.set_xticks(x, [str(d) for d in landscape.depths])
    ax.set_xlabel("Sleep iterations (depth)")
    ax.set_ylabel("Cost on the training context")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Learning curves (gray: one context each)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--order", choices=["transfer", "features", "cycle"], default="transfer"
    )
    args = parser.parse_args()
    landscape = load_landscape(args.results_dir)
    order = get_order(landscape, args.order)
    out = args.results_dir
    plot_learning_grid(landscape, order, out / "learning_grid.png")
    plot_curves(landscape, out / "curves.png")
    for depth in landscape.cross:
        plot_transfer(landscape, order, depth, out / f"transfer_{depth}.png")
    if landscape.cross:
        plot_kernel(landscape, out / "kernel.png")
        plot_embedding(landscape, order, out / "embedding.png")
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    _main()
