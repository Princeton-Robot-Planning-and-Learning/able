"""Figures for runs of the multi-site friction environment (env=friction_sites).

From every friction_sites run under --results_dir, writes to --output_dir:

- site_allocation.png: the units of each night's sleep per routine group,
  stacked, one panel per approach (columns) and seed (rows), from the
  training_sites column of cycle_results.csv. Dashed lines mark each group's
  share of the schedule times the night's units, as boundaries of the stack.
- site_staircase_error.png and site_staircase_std.png: per routine group, its
  belief error (or standard deviation) against its planning cost, one point per
  cycle in order, one line per approach (mean over seeds). Error is the mean of
  |belief mean - truth| over the floors the group's routine crosses, and both
  are in units of the prior's standard deviation; cost is the mean POMCPOW
  simulations on the group's routine tasks. Sites are weighted by their share
  of the schedule. Each step down is a night of sleep, and each step sideways
  a day awake (probe and task).
- site_map.png: the corridors maze, one panel per routine group (columns) and
  seed (rows), with the group's corridor shaded by its friction levels.

Each run's domain is rebuilt from its config.yaml, and its days replayed, for
the truths, the schedule, and each day's routines, so run this at the runs'
code version.

Example:
    python experiments/analysis/plot_site_runs.py --results_dir logs/sites
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import hydra
import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from able.envs.base import Environment, TaskDistribution
from able.envs.friction.model import FrictionModel
from able.envs.friction.sites import make_route_usage
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, PuckWorld
from able.envs.multi_site import MultiSiteModel, SiteRoutineTaskDistribution
from able.learners.belief_features import get_moments
from able.learners.site_focus import parse_training_sites

# run_cycles() spawns this many streams from the seed's generator; the days
# draw from the last.
_CYCLE_STREAMS = 6
_DAY_STREAM = 5

APPROACH_TO_DISPLAY_NAME = {
    "belief_coverage": "Belief coverage",
    "current_belief": "Current belief only",
    "deliberate_practice": "Deliberate practice",
    "dreaming": "Dreaming",
    "one_step_greedy": "One-step greedy",
    "scratch": "Plan from scratch",
    "transfer_practice": "Transfer practice",
}
APPROACH_TO_STYLE = {
    "transfer_practice": ("#6250d6", "o"),
    "deliberate_practice": ("#eda100", "s"),
    "current_belief": ("#1baf7a", "^"),
    "dreaming": ("#2a78d6", "D"),
    "scratch": ("#eb6834", "v"),
    "belief_coverage": ("#e87ba4", "P"),
    "one_step_greedy": ("#e34948", "X"),
}
# Each routine group's corridor color, for groups 0, 1, 2, ...
GROUP_COLORS = ("#2a78d6", "#c08a3e", "#1f7a3a", "#c94040")
# Each corridor's floors, light to dark by friction level.
CORRIDOR_SHADES = (
    ("#c9dcf5", "#5f97e0", "#1c4f9c"),
    ("#f1dfc2", "#c99a5b", "#7d5523"),
    ("#cde6cf", "#5aa463", "#1f5c2b"),
    ("#f5cfcd", "#dc6b67", "#8f2420"),
)
OTHER_FLOOR = "#e4e3df"
WALL = "#3b3b3b"
HOLE = "#111111"
INK = "#2b2b2b"
MUTED = "#6b6b6b"


@dataclass(frozen=True)
class SiteDomain:
    """What the figures need of a run's domain: each site's routine group,
    share of the schedule, and true friction, prior mean and prior standard
    deviation per floor; each cycle's share of each site's routine paths that
    cross each floor (cycles x sites x floors); the routine size, the prior's
    friction levels, and the maze."""

    group_of: NDArray[np.int64]
    schedule: NDArray[np.float64]
    truths: NDArray[np.float64]
    prior_mean: NDArray[np.float64]
    prior_std: NDArray[np.float64]
    usage: NDArray[np.float64]
    routine_size: int
    levels: NDArray[np.float64]
    world: PuckWorld

    @property
    def num_sites(self) -> int:
        """The number of sites."""
        return len(self.group_of)

    @property
    def num_groups(self) -> int:
        """The number of routine groups."""
        return int(self.group_of.max()) + 1


@dataclass(frozen=True)
class SiteRun:
    """One run's results and its domain."""

    approach: str
    seed: int
    cycles: pd.DataFrame
    evals: pd.DataFrame
    domain: SiteDomain


def replay_days(
    seed: int,
    num_cycles: int,
    env: Environment[Any, Any, Any, Any, Any],
    task_distribution: TaskDistribution[Any],
) -> Iterator[int]:
    """Start each cycle's day in the environment and task distribution as
    run_cycles() does from the seed, yielding each cycle once its day starts."""
    day_rng = np.random.default_rng(seed).spawn(_CYCLE_STREAMS)[_DAY_STREAM]
    for cycle in range(num_cycles):
        if cycle > 0:
            env.start_day(day_rng)
            task_distribution.start_day(day_rng)
        yield cycle


def describe_domain(cfg: DictConfig, num_cycles: int) -> SiteDomain:
    """Rebuild a run's domain and replay its days for each day's routines."""
    domain = hydra.utils.instantiate(cfg.env.domain, seed=cfg.seed)
    routines = domain.task_distribution
    model = domain.env_model
    assert isinstance(routines, SiteRoutineTaskDistribution)
    assert isinstance(model, MultiSiteModel) and isinstance(model.inner, FrictionModel)
    world = model.inner.world
    num_sites = len(routines.schedule)
    get_usage = make_route_usage(world, routines)
    usage = [
        [get_usage(site) for site in range(num_sites)]
        for _ in replay_days(cfg.seed, num_cycles, domain.real_env, routines)
    ]
    mean, std = get_moments(domain.initial_belief)
    truths = np.asarray(domain.real_env.hidden_params, dtype=np.float64)
    return SiteDomain(
        group_of=np.array([routines.get_group(s) for s in range(num_sites)]),
        schedule=np.asarray(routines.schedule, dtype=np.float64),
        truths=truths,
        prior_mean=np.asarray(mean).reshape(truths.shape),
        prior_std=np.asarray(std).reshape(truths.shape),
        usage=np.asarray(usage, dtype=np.float64),
        routine_size=len(routines.get_routine(0)),
        levels=np.asarray(DEFAULT_FRICTION_PRIOR.levels, dtype=np.float64),
        world=world,
    )


def load_runs(results_dir: Path) -> list[SiteRun]:
    """Every friction_sites run under the directory, with its domain."""
    domains: dict[str, SiteDomain] = {}
    runs = []
    for config_path in sorted(results_dir.rglob("config.yaml")):
        run_dir = config_path.parent
        if not (run_dir / "cycle_results.csv").exists():
            continue
        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig)
        if cfg.env_name != "friction_sites":
            continue
        cycles = pd.read_csv(
            run_dir / "cycle_results.csv", dtype={"training_sites": str}
        )
        evals = pd.read_csv(run_dir / "eval_results.csv")
        # A running run may have written one file's next cycle but not the other's.
        cycles = cycles[cycles.cycle.isin(evals.cycle)].reset_index(drop=True)
        evals = evals[evals.cycle.isin(cycles.cycle)]
        key = json.dumps(OmegaConf.to_container(cfg.env.domain), sort_keys=True)
        key += f"|{cfg.seed}|{cfg.num_cycles}"
        if key not in domains:
            domains[key] = describe_domain(cfg, cfg.num_cycles)
        runs.append(SiteRun(cfg.approach_name, cfg.seed, cycles, evals, domains[key]))
    assert runs, f"No friction_sites runs found under {results_dir}"
    return runs


def count_group_units(
    training_sites: Sequence[Any],
    group_of: Sequence[int] | NDArray[np.int64],
    num_groups: int,
) -> NDArray[np.float64]:
    """Units per night (rows) and routine group (columns), from each night's
    training_sites entry."""
    counts = np.zeros((len(training_sites), num_groups))
    for night, value in enumerate(training_sites):
        for site in parse_training_sites(value):
            counts[night, group_of[site]] += 1
    return counts


def get_group_shares(
    schedule: NDArray[np.float64], group_of: NDArray[np.int64], num_groups: int
) -> NDArray[np.float64]:
    """Each routine group's share of the schedule."""
    shares = np.bincount(group_of, weights=schedule, minlength=num_groups)
    return np.asarray(shares, dtype=np.float64)


def get_belief_moments(
    cycles: pd.DataFrame, num_sites: int, num_floors: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The belief mean and standard deviation of each floor of each site per
    cycle (cycles x sites x floors), from the belief columns, which list the
    sites in turn."""
    size = num_sites * num_floors
    means = cycles[[f"belief_mean_{i}" for i in range(size)]].to_numpy()
    stds = cycles[[f"belief_std_{i}" for i in range(size)]].to_numpy()
    shape = (len(cycles), num_sites, num_floors)
    return means.reshape(shape), stds.reshape(shape)


def mean_over_routine_floors(
    values: NDArray[np.float64], usage: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The mean of values (cycles x sites x floors) over the floors each site's
    routine crosses that cycle, per cycle and site."""
    crossed = usage > 0
    return np.sum(values * crossed, axis=-1) / np.maximum(crossed.sum(axis=-1), 1)


def get_site_sims(
    evals: pd.DataFrame, num_sites: int, routine_size: int
) -> NDArray[np.float64]:
    """Mean POMCPOW simulations per routine task, per cycle and site; tasks are
    listed site by site."""
    site = evals["task_id"] // routine_size
    table = evals.assign(site=site).pivot_table(
        index="cycle", columns="site", values="num_sims", aggfunc="mean"
    )
    return table.reindex(columns=range(num_sites)).to_numpy(dtype=np.float64)


def weight_by_schedule(
    values: NDArray[np.float64],
    schedule: NDArray[np.float64],
    group_of: NDArray[np.int64],
    num_groups: int,
) -> NDArray[np.float64]:
    """Each group's mean of values (cycles x sites) over its sites, weighted
    by their shares of the schedule."""
    weights = np.zeros((len(schedule), num_groups))
    weights[np.arange(len(schedule)), group_of] = schedule
    weights /= weights.sum(axis=0, keepdims=True)
    return values @ weights


def get_staircase(run: SiteRun, measure: str) -> pd.DataFrame:
    """One row per cycle and routine group: the group's belief measure ("error"
    or "std", in prior standard deviations, over its routine's floors) and its
    mean simulations per routine task."""
    domain = run.domain
    num_sites, num_floors = domain.truths.shape
    means, stds = get_belief_moments(run.cycles, num_sites, num_floors)
    if measure == "error":
        values = np.abs(means - domain.truths) / domain.prior_std
    else:
        values = stds / domain.prior_std
    belief = mean_over_routine_floors(values, domain.usage[: len(run.cycles)])
    sims = get_site_sims(run.evals, num_sites, domain.routine_size)
    args = (domain.schedule, domain.group_of, domain.num_groups)
    x = weight_by_schedule(belief, *args)
    y = weight_by_schedule(sims, *args)
    cycles = np.arange(len(run.cycles))
    return pd.DataFrame(
        {
            "cycle": np.repeat(cycles, domain.num_groups),
            "group": np.tile(np.arange(domain.num_groups), len(cycles)),
            "belief": x.ravel(),
            "sims": y.ravel(),
        }
    )


def get_prior_measure(domain: SiteDomain, measure: str) -> NDArray[np.float64]:
    """Each group's prior error or standard deviation, as get_staircase()
    measures them, over its routine's first-day floors."""
    if measure == "error":
        values = np.abs(domain.prior_mean - domain.truths) / domain.prior_std
    else:
        values = np.ones_like(domain.prior_std)
    belief = mean_over_routine_floors(values[None], domain.usage[:1])
    args = (domain.schedule, domain.group_of, domain.num_groups)
    return weight_by_schedule(belief, *args)[0]


def step_path(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The path through the points that moves down (or up) first, then
    sideways, between each point and the next."""
    path_x = np.empty(2 * len(x) - 1)
    path_y = np.empty(2 * len(y) - 1)
    path_x[0::2], path_y[0::2] = x, y
    path_x[1::2], path_y[1::2] = x[:-1], y[1:]
    return path_x, path_y


def get_friction_levels(
    truths: NDArray[np.float64], levels: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Each true friction's level, the nearest by ratio, as an index into
    levels."""
    ratios = np.abs(np.log(truths[..., None] / levels))
    return np.argmin(ratios, axis=-1)


def _style(approach: str) -> tuple[str, str]:
    return APPROACH_TO_STYLE.get(approach, ("#6b6b6b", "o"))


def _label(approach: str) -> str:
    return APPROACH_TO_DISPLAY_NAME.get(approach, approach)


def _ordered(approaches: set[str]) -> list[str]:
    known = [a for a in APPROACH_TO_STYLE if a in approaches]
    return known + sorted(approaches - set(known))


def _group_sites(domain: SiteDomain, group: int) -> str:
    sites = np.flatnonzero(domain.group_of == group)
    return f"site {sites[0]}" if len(sites) == 1 else f"sites {sites[0]}–{sites[-1]}"


def _shares_text(domain: SiteDomain) -> str:
    shares = get_group_shares(domain.schedule, domain.group_of, domain.num_groups)
    return " / ".join(f"{100 * s:.0f}%" for s in shares)


def plot_allocation(runs: Sequence[SiteRun], path: Path) -> None:
    """Units per routine group per night, one panel per approach and seed."""
    runs = [r for r in runs if "training_sites" in r.cycles]
    runs = [
        r for r in runs if any(parse_training_sites(v) for v in r.cycles.training_sites)
    ]
    assert runs, "No runs logged training_sites"
    approaches = _ordered({r.approach for r in runs})
    seeds = sorted({r.seed for r in runs})
    fig, axes = plt.subplots(
        len(seeds),
        len(approaches),
        figsize=(3.6 * len(approaches), 2.6 * len(seeds) + 0.9),
        squeeze=False,
        sharey=True,
    )
    for row, seed in enumerate(seeds):
        for col, approach in enumerate(approaches):
            ax = axes[row][col]
            match = [r for r in runs if r.seed == seed and r.approach == approach]
            if not match:
                ax.set_visible(False)
                continue
            _plot_allocation_panel(ax, match[0])
            if row == 0:
                ax.set_title(_label(approach), fontsize=10, color=INK)
            if col == 0:
                shares = _shares_text(match[0].domain)
                ax.set_ylabel(f"Seed {seed}: units\n(schedule {shares})", fontsize=9)
            if row == len(seeds) - 1:
                ax.set_xlabel("Night (cycle)")
    num_groups = max(r.domain.num_groups for r in runs)
    handles: list[Artist] = [
        Patch(color=GROUP_COLORS[g % len(GROUP_COLORS)], label=f"Group {g}")
        for g in range(num_groups)
    ]
    handles.append(
        Line2D([], [], color=MUTED, linestyle="--", label="Schedule share boundary")
    )
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_allocation_panel(ax: Any, run: SiteRun) -> None:
    domain = run.domain
    counts = count_group_units(
        list(run.cycles.training_sites), domain.group_of, domain.num_groups
    )
    nights = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for group in range(domain.num_groups):
        ax.bar(
            nights,
            counts[:, group],
            bottom=bottom,
            width=0.7,
            color=GROUP_COLORS[group % len(GROUP_COLORS)],
            edgecolor="white",
            linewidth=1.0,
        )
        bottom += counts[:, group]
    shares = get_group_shares(domain.schedule, domain.group_of, domain.num_groups)
    units = float(np.max(bottom))
    for group, boundary in enumerate(np.cumsum(shares)[:-1]):
        ax.axhline(
            boundary * units,
            color=GROUP_COLORS[group % len(GROUP_COLORS)],
            linestyle="--",
            linewidth=1.2,
        )
    ax.set_xticks(nights)
    ax.set_ylim(0, units * 1.05)
    ax.spines[["top", "right"]].set_visible(False)


def plot_staircase(runs: Sequence[SiteRun], measure: str, path: Path) -> None:
    """Belief measure against planning cost per cycle, one panel per routine
    group, one line per approach (mean over seeds)."""
    frames = [
        get_staircase(run, measure).assign(approach=run.approach, seed=run.seed)
        for run in runs
    ]
    df = pd.concat(frames, ignore_index=True)
    mean = df.groupby(["approach", "group", "cycle"], as_index=False)[
        ["belief", "sims"]
    ].mean()
    approaches = _ordered(set(df["approach"]))
    num_groups = int(df["group"].max()) + 1
    domains = {run.seed: run.domain for run in runs}
    prior = np.mean([get_prior_measure(d, measure) for d in domains.values()], axis=0)
    fig, axes = plt.subplots(1, num_groups, figsize=(4.4 * num_groups, 4.2))
    axes = np.atleast_1d(axes)
    for group, ax in enumerate(axes):
        ax.axvline(prior[group], color=MUTED, linestyle=":", linewidth=1.2)
        for approach in approaches:
            points = mean[(mean["approach"] == approach) & (mean["group"] == group)]
            if points.empty:
                continue
            x = points["belief"].to_numpy(dtype=np.float64)
            y = points["sims"].to_numpy(dtype=np.float64)
            color, marker = _style(approach)
            ax.plot(*step_path(x, y), color=color, linewidth=1.6, alpha=0.9)
            ax.plot(x, y, marker, color=color, markersize=6, markeredgecolor="white")
            ax.annotate(
                "0",
                (x[0], y[0]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
            ax.annotate(
                str(len(x) - 1),
                (x[-1], y[-1]),
                xytext=(4, -10),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        ax.set_title(f"Group {group} ({_group_sites(runs[0].domain, group)})")
        ax.set_xlabel(
            "Belief error on routine floors (prior std)"
            if measure == "error"
            else "Belief std on routine floors (prior std)"
        )
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("POMCPOW simulations per routine task")
    handles = [
        Line2D([], [], color=_style(a)[0], marker=_style(a)[1], label=_label(a))
        for a in approaches
    ]
    handles.append(Line2D([], [], color=MUTED, linestyle=":", label="Prior"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    num_seeds = df["seed"].nunique()
    fig.text(
        0.5,
        0.01,
        f"One point per cycle (0 to last), mean over {num_seeds} seeds; sites "
        "weighted by schedule. A step down is a night of sleep, a step sideways "
        "a day awake (probe and task).",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_group_map(domains: dict[int, SiteDomain], path: Path) -> None:
    """The maze per seed (rows) and routine group (columns), with the group's
    corridor shaded by its friction levels and the other corridors grey."""
    seeds = sorted(domains)
    num_groups = max(d.num_groups for d in domains.values())
    any_domain = domains[seeds[0]]
    puck_map = any_domain.world.puck_map
    scale = 3.6 / puck_map.width
    fig, axes = plt.subplots(
        len(seeds),
        num_groups,
        figsize=(
            num_groups * puck_map.width * scale + 0.4,
            len(seeds) * (puck_map.height * scale + 0.4) + 0.6,
        ),
        squeeze=False,
    )
    differ = [False] * num_groups
    for row, seed in enumerate(seeds):
        for group in range(num_groups):
            differ[group] |= _draw_group_map(axes[row][group], domains[seed], group)
            axes[row][group].set_title(
                f"Seed {seed}, group {group} ({_group_sites(domains[seed], group)})",
                fontsize=9,
                color=INK,
            )
    for group in range(num_groups):
        shades = CORRIDOR_SHADES[group % len(CORRIDOR_SHADES)]
        handles: list[Artist] = [
            Patch(color=shade, label=f"μ ≈ {level:g}")
            for shade, level in zip(shades, any_domain.levels)
        ]
        handles += [
            Patch(color=OTHER_FLOOR, label="Other corridors"),
            Line2D([], [], color=HOLE, marker="o", linestyle="", label="Hole"),
        ]
        if differ[group]:
            handles.append(
                Patch(
                    facecolor="white",
                    edgecolor=MUTED,
                    hatch="///",
                    label="Sites differ",
                )
            )
        axes[-1][group].legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            fontsize=7,
            frameon=False,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def get_group_floor_levels(
    domain: SiteDomain, group: int
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Each floor's friction level for a routine group, the one most of its
    sites' schedule share has, and whether its sites differ there."""
    sites = np.flatnonzero(domain.group_of == group)
    levels = get_friction_levels(domain.truths[sites], domain.levels)
    votes = np.zeros((levels.shape[1], len(domain.levels)))
    for site_levels, weight in zip(levels, domain.schedule[sites]):
        votes[np.arange(len(site_levels)), site_levels] += weight
    return np.argmax(votes, axis=1), np.any(levels != levels[0], axis=0)


def _draw_group_map(ax: Any, domain: SiteDomain, group: int) -> bool:
    """Draw the maze for one group; return whether the group's sites differ in
    some floor's level, which is hatched."""
    world = domain.world
    puck_map = world.puck_map
    colors = puck_map.surface_colors
    corridor = group % (max(colors) + 1)
    shades = CORRIDOR_SHADES[corridor % len(CORRIDOR_SHADES)]
    levels, differ = get_group_floor_levels(domain, group)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(0, puck_map.width)
    ax.set_ylim(0, puck_map.height)
    largest: dict[int, tuple[float, float, float, float]] = {}
    for floor, (x0, y0, x1, y1) in puck_map.surface_rects():
        mine = colors[floor] == corridor
        color = shades[levels[floor]] if mine else OTHER_FLOOR
        hatched = mine and differ[floor]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=color,
                edgecolor="white" if hatched else color,
                hatch="///" if hatched else None,
                linewidth=0,
            )
        )
        area = (x1 - x0) * (y1 - y0)
        best = largest.get(floor)
        if mine and (best is None or area > (best[2] - best[0]) * (best[3] - best[1])):
            largest[floor] = (x0, y0, x1, y1)
    for x0, y0, x1, y1 in puck_map.wall_rects():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=WALL))
    for hole in world.holes:
        ax.add_patch(Circle(tuple(hole), world.physics.hole_radius, color=HOLE))
    for floor, (x0, _, _, y1) in largest.items():
        ax.text(
            x0 + 0.08,
            y1 - 0.08,
            f"floor {floor}",
            ha="left",
            va="top",
            fontsize=7,
            color="white" if levels[floor] == len(shades) - 1 else INK,
        )
    return bool(np.any(differ[np.asarray(colors) == corridor]))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results_dir)
    plot_allocation(runs, output_dir / "site_allocation.png")
    for measure in ("error", "std"):
        plot_staircase(runs, measure, output_dir / f"site_staircase_{measure}.png")
    domains = {run.seed: run.domain for run in runs}
    plot_group_map(domains, output_dir / "site_map.png")


if __name__ == "__main__":
    _main()
