"""Plot the friction-site cost landscape and transfer from the CSVs of
experiments/probe_site_transfer.py, one or more runs per --train_site.

The figures are:
- site_landscape.png: per trained site, each evaluated site's mean POMCPOW
  simulations per task (darker is slower) after each number of units, with
  sites ordered by corridor and group. With --run_dir, a run's sites as a
  learning-curve grid under their corridor's own curve, with ticks where the
  units a rule gave each site over the first --nights nights took it, and bars
  of each site's schedule share. Needs a training_sites column in the run's
  cycle_results.csv; without one the overlay is skipped.
- site_transfer.png: simulations saved at each of --depths (default the
  deepest) as a share of the trained site's own saving, per trained and
  evaluated site. The corridor table printed is for the deepest of them.

    python experiments/analysis/plot_site_landscape.py --probe_dir probes/ \
        --out_dir figures/ [--run_dir <run> --nights 2]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import hydra
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
from matplotlib.figure import Figure, SubFigure  # noqa: E402

# pylint: disable-next=wrong-import-position
from matplotlib.patches import Rectangle  # noqa: E402

# pylint: disable-next=wrong-import-position
from numpy.typing import NDArray  # noqa: E402

# pylint: disable-next=wrong-import-position
from omegaconf import OmegaConf  # noqa: E402

# pylint: disable-next=wrong-import-position
from able.envs.friction.world import COLOR_RGB  # noqa: E402

# Darker is slower.
SIMS_CMAP = LinearSegmentedColormap.from_list(
    "sims", ["#f7f9fc", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
)
# Blue saves simulations and red costs them, around a gray zero.
SHARE_CMAP = LinearSegmentedColormap.from_list(
    "share", ["#8f1f1f", "#e34948", "#f0efec", "#3987e5", "#0d366b"]
)
INK = "#3a3a38"
MUTED = "#73726c"


# Each group's corridor, from the probes once loaded.
_CORRIDOR_OF: dict[int, int] = {}


def get_corridor(group: int) -> int:
    """The corridor of a group's routine, as the probes recorded it."""
    return _CORRIDOR_OF[group]


def load_probes(probe_dir: Path) -> pd.DataFrame:
    """Every probe row."""
    paths = sorted(probe_dir.glob("*.csv"))
    assert paths, f"No probe CSVs in {probe_dir}"
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    missing = {"train_site", "corridor"} - set(df.columns)
    assert not missing, f"Probes lack {sorted(missing)}; rerun probe_site_transfer.py"
    _CORRIDOR_OF.update(zip(df["group"].astype(int), df["corridor"].astype(int)))
    return df


def get_site_order(df: pd.DataFrame) -> list[int]:
    """Evaluated sites by corridor, then group, then index."""
    sites = df[["site", "group", "corridor"]].drop_duplicates()
    return sites.sort_values(["corridor", "group", "site"])["site"].tolist()


def get_trained_order(df: pd.DataFrame) -> list[int]:
    """Trained sites in the same order as get_site_order."""
    order = get_site_order(df)
    return sorted(df["train_site"].unique(), key=order.index)


def get_mean_sims(df: pd.DataFrame) -> pd.DataFrame:
    """Mean simulations per task over seeds of each seed's mean, indexed by
    (train_site, depth) with a column per evaluated site."""
    per_seed = df.groupby(["train_site", "depth", "site", "seed"])["num_sims"].mean()
    return per_seed.groupby(["train_site", "depth", "site"]).mean().unstack("site")


def get_saved(df: pd.DataFrame, depth: int) -> pd.Series:
    """Per (train_site, seed, site): mean simulations at depth 0 minus at depth,
    on the same tasks."""
    keys = ["train_site", "seed", "site", "depth"]
    sims = df.groupby(keys)["num_sims"].mean().unstack("depth")
    return sims[0] - sims[depth]


def get_shares(df: pd.DataFrame, depth: int) -> pd.DataFrame:
    """Mean simulations saved at each site over the trained site's own, per
    trained site (rows) and evaluated site (columns)."""
    saved = get_saved(df, depth).groupby(["train_site", "site"]).mean().unstack()
    own = pd.Series({t: saved.loc[t, t] for t in saved.index})
    return saved.div(own, axis=0)


def get_corridor_table(df: pd.DataFrame, depth: int) -> pd.DataFrame:
    """Simulations saved per task at depth, mean and standard error over seeds,
    and share of the trained site's own saving, per trained corridor and
    evaluated corridor. The trained corridor's own cell is the trained site;
    other cells average every site in the evaluated corridor."""
    saved = get_saved(df, depth).rename("saved").reset_index()
    sites = df[["site", "corridor"]].drop_duplicates().set_index("site")["corridor"]
    saved["eval_corridor"] = saved["site"].map(sites)
    saved["train_corridor"] = saved["train_site"].map(sites)
    is_own = saved["site"] == saved["train_site"]
    in_own_corridor = saved["eval_corridor"] == saved["train_corridor"]
    saved = saved[is_own | ~in_own_corridor]
    per_seed = saved.groupby(["train_corridor", "eval_corridor", "seed"])[
        "saved"
    ].mean()
    grouped = per_seed.groupby(["train_corridor", "eval_corridor"])
    table = pd.DataFrame({"saved": grouped.mean(), "se": grouped.sem()})
    trains = table.index.get_level_values(0)
    own = {t: table.loc[(t, t), "saved"] for t in set(trains)}
    table["share"] = table["saved"].to_numpy() / [own[t] for t in trains]
    return table


def format_corridor_table(table: pd.DataFrame) -> str:
    """A markdown table with trained corridors as rows."""
    corridors = sorted(set(table.index.get_level_values(1)))
    names = [f"corridor {c}" for c in corridors]
    lines = [
        "| Trained \\ evaluated | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for train in sorted(set(table.index.get_level_values(0))):
        cells = []
        for corridor in corridors:
            if (train, corridor) not in table.index:
                cells.append("")
                continue
            row = table.loc[(train, corridor)]
            cells.append(f"{row['saved']:.0f} ± {row['se']:.0f} ({row['share']:.2f})")
        lines.append(f"| corridor {train} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class Overlay:
    """A run's units per night (rows) and site (columns), each site's group,
    and the schedule."""

    units: NDArray[np.int64]
    groups: list[int]
    schedule: NDArray[np.float64]


def count_units(
    training_sites: pd.Series, num_sites: int, nights: int | None = None
) -> NDArray[np.int64]:
    """Units per night and site from ';'-joined site indices, one entry per
    night, over the first nights nights."""
    entries = training_sites.iloc[:nights] if nights else training_sites
    units = np.zeros((len(entries), num_sites), dtype=np.int64)
    for night, entry in enumerate(entries):
        if pd.isna(entry) or str(entry).strip() == "":
            continue
        for site in str(entry).split(";"):
            units[night, int(float(site))] += 1
    return units


def load_overlay(run_dir: Path, nights: int | None) -> Overlay | None:
    """The overlay of a friction_sites run, or None if its cycle_results.csv
    has no training_sites column."""
    cycles = pd.read_csv(run_dir / "cycle_results.csv")
    if "training_sites" not in cycles:
        print(f"No training_sites column in {run_dir}; skipping the overlay")
        return None
    cfg = OmegaConf.load(run_dir / "config.yaml")
    routines = hydra.utils.instantiate(cfg.env.domain, seed=cfg.seed).task_distribution
    schedule = np.asarray(routines.schedule, dtype=np.float64)
    groups = [routines.get_group(site) for site in range(len(schedule))]
    units = count_units(cycles["training_sites"], len(schedule), nights)
    return Overlay(units, groups, schedule)


def _row_position(units: NDArray[np.float64], depths: list[int]) -> NDArray:
    """Where a number of units falls among the rows of depths."""
    return np.interp(units, depths, np.arange(len(depths)))


def _corridor_strip(ax: plt.Axes, corridors: list[int], y: float) -> None:
    """A row of boxes from y to y + 0.3 under the columns, each with its
    column's corridor number, labeled "corridor" on the left."""
    for col, corridor in enumerate(corridors):
        ax.add_patch(
            Rectangle(
                (col - 0.5, y),
                1.0,
                0.3,
                facecolor="white",
                edgecolor="0.6",
                linewidth=0.8,
            )
        )
        ax.text(col, y + 0.15, str(corridor), ha="center", va="center", fontsize=8)
    ax.text(-0.6, y + 0.15, "corridor", ha="right", va="center", fontsize=8)


def _draw_grid(
    ax: plt.Axes, grid: NDArray[np.float64], corridors: list[int], vmax: float
) -> matplotlib.image.AxesImage:
    """A cost grid with a strip of corridor colors under its columns."""
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=SIMS_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    _corridor_strip(ax, corridors, -0.85)
    for col in range(1, len(corridors)):
        if corridors[col] != corridors[col - 1]:
            ax.axvline(col - 0.5, color="white", linewidth=2.0)
    ax.set_xlim(-0.5, len(corridors) - 0.5)
    ax.set_ylim(-0.85, len(grid) - 0.5)
    return image


def _draw_overlay(
    fig: SubFigure,
    overlay: Overlay,
    own_curves: dict[int, pd.Series],
    depths: list[int],
    vmax: float,
) -> None:
    ax, bar_ax = fig.subplots(2, 1, sharex=True, height_ratios=[1.0, 0.5])
    order = sorted(
        range(len(overlay.groups)),
        key=lambda s: (get_corridor(overlay.groups[s]), overlay.groups[s], s),
    )
    corridors = [get_corridor(overlay.groups[s]) for s in order]
    grid = np.full((len(depths), len(order)), np.nan)
    for col, corridor in enumerate(corridors):
        if corridor in own_curves:
            grid[:, col] = own_curves[corridor].reindex(depths).to_numpy()
    _draw_grid(ax, grid, corridors, vmax)
    cumulative = np.cumsum(overlay.units[:, order], axis=0)
    for col in range(len(order)):
        reached = np.unique(cumulative[:, col])
        reached = reached[reached > 0]
        rows = _row_position(np.minimum(reached, depths[-1]), depths)
        ax.hlines(rows, col - 0.35, col + 0.35, color="#eb6834", linewidth=2.0)
        ax.text(
            col,
            len(depths) - 0.45,
            str(int(cumulative[-1, col])),
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.set_yticks(range(len(depths)), [str(d) for d in depths])
    ax.set_ylabel("Units trained")
    ax.set_title(
        "A run's sites on their corridor's own curve; ticks: units after each of "
        f"the first {len(overlay.units)} nights (total above)",
        color=INK,
        fontsize=9,
        pad=16,
    )
    share = overlay.units[:, order].sum(axis=0) / max(overlay.units.sum(), 1)
    x = np.arange(len(order))
    bar_ax.bar(
        x,
        overlay.schedule[order],
        width=0.8,
        color=[COLOR_RGB[c] for c in corridors],
        edgecolor=MUTED,
        linewidth=0.6,
        label="Schedule share",
    )
    bar_ax.scatter(x, share, color="#eb6834", s=18, zorder=3, label="Share of units")
    bar_ax.set_xticks(x, [f"{s}\ng{overlay.groups[s]}" for s in order], fontsize=7)
    bar_ax.tick_params(axis="x", length=0)
    bar_ax.set_xlabel("Run site and group")
    bar_ax.set_ylabel("Share")
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.legend(frameon=False, fontsize=7, loc="upper left", ncols=2)


def plot_landscape(
    df: pd.DataFrame, path: Path, overlay: Overlay | None = None
) -> None:
    """One panel per trained site: evaluated sites as columns, units trained as
    rows, colored by mean simulations per task; the trained site is outlined."""
    sims = get_mean_sims(df)
    trained = get_trained_order(df)
    sites = get_site_order(df)
    depths = sorted(df["depth"].unique())
    groups = df.drop_duplicates("site").set_index("site")["group"]
    corridors = [get_corridor(groups[s]) for s in sites]
    vmax = float(np.nanmax(sims.to_numpy()))
    width = max(3.3 * len(trained), 7.0)
    fig = plt.figure(
        figsize=(width, 3.4 if overlay is None else 7.4), layout="constrained"
    )
    top: Figure | SubFigure = fig
    bottom: SubFigure | None = None
    if overlay is not None:
        top, bottom = fig.subfigures(  # type: ignore[misc]
            2, 1, height_ratios=[1.0, 1.15], hspace=0.04
        )
    axes = np.atleast_1d(top.subplots(1, len(trained), sharey=True))
    for ax, train_site in zip(axes, trained):
        grid = sims.loc[train_site].reindex(depths)[sites].to_numpy()
        image = _draw_grid(ax, grid, corridors, vmax)
        for (row, col), value in np.ndenumerate(grid):
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.55 * vmax else INK,
                )
        col = sites.index(train_site)
        ax.add_patch(
            Rectangle(
                (col - 0.5, -0.5),
                1.0,
                len(depths),
                fill=False,
                edgecolor="#eb6834",
                linewidth=2.0,
                zorder=3,
            )
        )
        ax.set_xticks(
            range(len(sites)), [f"{s}\ng{groups[s]}" for s in sites], fontsize=7
        )
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Evaluated site and group")
        ax.set_title(
            f"Trained for site {train_site} "
            f"(corridor {get_corridor(groups[train_site])})",
            color=INK,
            fontsize=10,
        )
    axes[0].set_yticks(range(len(depths)), [str(d) for d in depths])
    axes[0].set_ylabel("Units trained")
    top.colorbar(
        image, ax=list(axes), label="Mean simulations per task\n(darker is slower)"
    )
    if overlay is not None and bottom is not None:
        own_curves: dict[int, list[pd.Series]] = {}
        for train_site in trained:
            corridor = get_corridor(groups[train_site])
            own_curves.setdefault(corridor, []).append(sims.loc[train_site][train_site])
        curves = {c: pd.concat(v, axis=1).mean(axis=1) for c, v in own_curves.items()}
        _draw_overlay(bottom, overlay, curves, depths, vmax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_transfer(df: pd.DataFrame, depths: list[int], path: Path) -> None:
    """One panel per depth: trained sites as rows and evaluated sites as
    columns, colored by the share of the trained site's own saving."""
    trained = get_trained_order(df)
    sites = get_site_order(df)
    groups = df.drop_duplicates("site").set_index("site")["group"]
    corridors = [get_corridor(groups[s]) for s in sites]
    shares = {
        depth: get_shares(df, depth).reindex(index=trained, columns=sites)
        for depth in depths
    }
    limit = max(1.0, *(float(np.nanmax(np.abs(v.to_numpy()))) for v in shares.values()))
    fig, axes = plt.subplots(
        1,
        len(depths),
        figsize=(2.2 + 0.9 * len(sites) * len(depths), 1.0 + 0.6 * len(trained)),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for ax, depth in zip(axes[0], depths):
        values = shares[depth].to_numpy()
        image = ax.imshow(
            values,
            aspect="auto",
            cmap=SHARE_CMAP,
            norm=TwoSlopeNorm(0.0, -limit, limit),
            interpolation="nearest",
        )
        for (row, col), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.6 * limit else INK,
                )
        for row, train_site in enumerate(trained):
            ax.add_patch(
                Rectangle(
                    (sites.index(train_site) - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=INK,
                    linewidth=1.5,
                )
            )
        _corridor_strip(ax, corridors, len(trained) - 0.5)
        ax.set_ylim(len(trained) - 0.2, -0.5)
        ax.set_xticks(
            range(len(sites)), [f"{s}\ng{groups[s]}" for s in sites], fontsize=8
        )
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Evaluated site and group")
        own = get_saved(df, depth).groupby(["train_site", "site"]).mean()
        ax.set_title(
            f"After {depth} units (own saving: "
            + ", ".join(f"{own[(t, t)]:.0f}" for t in trained)
            + ")",
            color=INK,
            fontsize=9,
        )
    axes[0, 0].set_yticks(
        range(len(trained)),
        [f"site {t} (corridor {get_corridor(groups[t])})" for t in trained],
        fontsize=8,
    )
    axes[0, 0].set_ylabel("Trained site")
    fig.colorbar(image, ax=list(axes[0]), label="Share of the trained\nsite's saving")
    fig.suptitle(
        "Simulations saved per task, as a share of the trained site's own saving",
        color=INK,
        fontsize=10,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--probe_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--nights", type=int, default=None)
    args = parser.parse_args()
    df = load_probes(args.probe_dir)
    depths = args.depths or [int(df["depth"].max())]
    depth = max(depths)
    overlay = load_overlay(args.run_dir, args.nights) if args.run_dir else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_landscape(df, args.out_dir / "site_landscape.png", overlay)
    plot_transfer(df, depths, args.out_dir / "site_transfer.png")
    print(get_shares(df, depth).round(2))
    print(format_corridor_table(get_corridor_table(df, depth)))
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    _main()
