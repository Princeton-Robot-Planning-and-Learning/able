"""Plot what the friction-site push model psi proposes across believed friction
theta, from the snapshots of experiments/train_site_psi.py.

`measure` evaluates psi at the near-goal states of its own training pushes, under
the site's belief rescaled to each theta on a grid and at the belief itself. For
each theta it records the median speed of psi's draws, their mean direction
relative to the goal, the share that land in the goal when executed at theta, and
psi*(theta): the speeds that land the puck when aimed straight at the goal. The
untrained psi (depth 0) proposes nothing, so its row is the POMDP's random pushes.

`plot` draws the robot that trained every site, for one seed: speed on the y axis,
with arrows for direction (up is at the goal).

    python experiments/analysis/plot_theta_psi.py measure s0_t1.pkl --out s0_t1.csv
    python experiments/analysis/plot_theta_psi.py plot s0_*.csv --out theta_psi.png
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
from matplotlib.lines import Line2D  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch  # pylint: disable=wrong-import-position
from matplotlib.transforms import offset_copy  # pylint: disable=wrong-import-position
from numpy.typing import NDArray  # pylint: disable=wrong-import-position

from able.beliefs.particles import (  # pylint: disable=wrong-import-position
    ParticleBelief,
)
from able.envs.friction.pomdp import (  # pylint: disable=wrong-import-position
    GOAL,
    PuckPushingModel,
)
from able.envs.friction.world import Push  # pylint: disable=wrong-import-position
from able.models.base import Dataset  # pylint: disable=wrong-import-position
from able.planners.pomcpow import (  # pylint: disable=wrong-import-position
    decode_push,
    get_push_features,
)

THETAS = np.geomspace(0.015, 0.3, 25)
LEVELS = np.array([0.025, 0.05, 0.1, 0.2])
SPEEDS = np.linspace(0.05, 1.2, 116)
BLUE, ORANGE, RED, GRAY = "#a9c8ec", "#f3bfa8", "#a8322d", "#8c8c8c"
GLYPH = 20.0  # arrow length in points for draws that all share one direction


class NearGoal(NamedTuple):
    """A puck state, the heading and stance that push it straight at the goal."""

    puck: NDArray[np.float64]
    heading: float
    stance: float
    goal: tuple[float, float]


def scale_belief(belief: ParticleBelief, theta: float) -> ParticleBelief:
    """The belief with its particles rescaled so their mean is theta."""
    particles = np.asarray(belief.particles)
    scale = theta / float(particles.mean())
    return ParticleBelief(particles * scale, belief.low, belief.high, independent=False)


def landing_speeds(
    pomdp: PuckPushingModel, state: NearGoal, frictions: Sequence[float]
) -> tuple[float, float]:
    """The slowest and fastest speeds that land the puck when aimed at the goal,
    or NaNs if none do."""
    root = pomdp.initial_state(state.puck, state.puck, frictions, state.goal)
    push = Push(round(state.heading, 4), 0.0, state.stance)
    good = [
        v
        for v in SPEEDS
        if pomdp.transition(root, push, state.heading, v).outcome == GOAL
    ]
    return (min(good), max(good)) if good else (math.nan, math.nan)


def near_goal_states(
    dataset: Dataset,
    pomdp: PuckPushingModel,
    corridor: int,
    distances: tuple[float, float] = (0.5, 0.85),
    most: int = 40,
) -> list[NearGoal]:
    """The puck states of the training pushes in the corridor whose goal is within
    distances, from which some speed aimed at the goal lands the puck."""
    world = pomdp.world
    colors = np.asarray(world.puck_map.surface_colors)
    rows = np.all(np.isfinite(dataset.y[:, :4]), axis=1)
    found = []
    for x in dataset.x[rows]:
        puck, goal = np.asarray(x[0:2]), (float(x[2]), float(x[3]))
        distance = math.hypot(goal[0] - puck[0], goal[1] - puck[1])
        surface = world.surface_at(puck)
        if not distances[0] <= distance <= distances[1] or surface < 0:
            continue
        if colors[surface] != corridor:
            continue
        heading = math.atan2(goal[1] - puck[1], goal[0] - puck[0])
        stances = [s for s in pomdp.stances if world.push_feasible(puck, heading, s)]
        if not stances:
            continue
        state = NearGoal(puck, heading, float(min(stances, key=abs)), goal)
        if np.isfinite(landing_speeds(pomdp, state, [0.05] * len(colors))[0]):
            found.append(state)
    if len(found) > most:
        keep = np.random.default_rng(0).choice(len(found), most, replace=False)
        found = [found[i] for i in sorted(keep)]
    return found


def measure_site(
    snapshot: dict[str, Any],
    site: int,
    thetas: Sequence[float],
    num_draws: int,
) -> list[dict[str, float]]:
    """Rows for one queried site, at each theta and the site's own belief."""
    pomdp: PuckPushingModel = snapshot["pomdp"]
    truth = np.asarray(snapshot["truths"][site])
    belief: ParticleBelief = snapshot["beliefs"].beliefs[site]
    own = float(np.asarray(belief.particles).mean())
    states = near_goal_states(snapshot["dataset"], pomdp, snapshot["corridors"][site])
    train_site = snapshot["train_site"]
    rng = np.random.default_rng([snapshot["seed"], site, train_site + 1])
    rows = []
    for theta, exact in [*((t, False) for t in thetas), (own, True)]:
        frictions = tuple(truth * (theta / float(truth.mean())))
        stars = np.array([landing_speeds(pomdp, s, frictions) for s in states])
        for depth, params in sorted(snapshot["snapshots"].items()):
            speeds, offsets, hits = [], [], 0
            for state in states:
                root = pomdp.initial_state(
                    state.puck, state.puck, frictions, state.goal
                )
                if depth == 0:
                    pushes = [pomdp.propose(root, [], rng) for _ in range(num_draws)]
                else:
                    x = get_push_features(
                        state.puck, state.goal, pomdp.world, scale_belief(belief, theta)
                    )
                    draws = params.model.sample(x, num_draws, rng)
                    pushes = [decode_push(pomdp, y) for y in draws]
                for push in pushes:
                    if push is None:
                        continue
                    speeds.append(push.speed)
                    offsets.append(math.remainder(push.angle - state.heading, math.tau))
                    if pomdp.world.push_feasible(state.puck, push.angle, push.stance):
                        hits += pomdp.step(root, push, rng)[0].outcome == GOAL
            rows.append(
                {
                    "seed": snapshot["seed"],
                    "train_site": train_site,
                    "site": site,
                    "train_theta": (
                        float(np.mean(snapshot["truths"][train_site]))
                        if train_site >= 0
                        else math.nan
                    ),
                    "depth": depth,
                    "theta": theta,
                    "exact": exact,
                    "num_states": len(states),
                    "speed": float(np.median(speeds)) if speeds else math.nan,
                    "cos_mean": (
                        float(np.mean(np.cos(offsets))) if offsets else math.nan
                    ),
                    "sin_mean": (
                        float(np.mean(np.sin(offsets))) if offsets else math.nan
                    ),
                    "landed": hits / max(len(speeds), 1),
                    "star_lo": float(np.nanmedian(stars[:, 0])) if states else math.nan,
                    "star_hi": float(np.nanmedian(stars[:, 1])) if states else math.nan,
                }
            )
    return rows


def measure(
    snapshot: dict[str, Any],
    thetas: Sequence[float],
    num_draws: int,
    sites: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Rows for the sites, by default the trained site, or every site if every
    site was trained."""
    train_site = snapshot["train_site"]
    if sites is None:
        sites = range(len(snapshot["truths"])) if train_site < 0 else [train_site]
    rows = [r for s in sites for r in measure_site(snapshot, s, thetas, num_draws)]
    return pd.DataFrame(rows)


def nearest_level(theta: float) -> float:
    """The base friction level nearest theta."""
    return float(LEVELS[np.argmin(np.abs(np.log(LEVELS) - math.log(theta)))])


def _draw_glyph(ax: Any, x: float, y: float, angle: float, length: float) -> None:
    """An arrow centered at (x, y), pointing up for angle 0."""
    dx, dy = -math.sin(angle) * length, math.cos(angle) * length
    head = offset_copy(ax.transData, fig=ax.figure, x=dx / 2, y=dy / 2, units="points")
    ax.annotate(
        "",
        xy=(x, y),
        xycoords=head,
        xytext=(-dx, -dy),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "k",
            "lw": 1.0,
            "mutation_scale": 6,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )


def _draw_panel(
    ax: Any, rows: pd.DataFrame, depth: int, trained: list[float], title: str
) -> None:
    """One panel: psi* band, speeds before and after, direction arrows, and where
    more or fewer of psi's draws land."""
    rows = rows.copy()
    rows.loc[rows.exact, "theta"] = rows.loc[rows.exact, "theta"].map(nearest_level)
    agg = (
        rows.groupby(["depth", "theta"])
        .agg(
            speed=("speed", "median"),
            cos=("cos_mean", "mean"),
            sin=("sin_mean", "mean"),
            landed=("landed", "mean"),
            lo=("star_lo", "median"),
            hi=("star_hi", "median"),
            exact=("exact", "max"),
        )
        .reset_index()
    )
    before = agg[agg.depth == 0].set_index("theta")
    after = agg[agg.depth == depth].set_index("theta").reindex(before.index)
    th = before.index.values
    ax.fill_between(th, before.lo, before.hi, color=GRAY, alpha=0.22, lw=0)
    ax.plot(th, (before.lo + before.hi) / 2, color=GRAY, lw=1.1)
    change = after.landed.to_numpy(dtype=float) - before.landed.to_numpy(dtype=float)
    for sign, color in ((1, BLUE), (-1, ORANGE)):
        ax.fill_between(
            th,
            before.speed.values,
            after.speed.values,
            where=sign * change > 0.01,
            color=color,
            lw=0,
            interpolate=True,
        )
    exact = after.exact.to_numpy(dtype=bool)
    for i in np.flatnonzero(exact & (np.abs(change) > 0.01)):
        ax.fill_between(
            [th[i] / 1.08, th[i] * 1.08],
            before.speed.values[i],
            after.speed.values[i],
            color=BLUE if change[i] > 0 else ORANGE,
            lw=0,
            zorder=2.5,
        )
    ax.plot(th, before.speed.values, "k--", lw=0.9)
    ax.plot(th, after.speed.values, "k-", lw=1.6)
    for i in sorted(set(np.flatnonzero(~exact)[::3]) | set(np.flatnonzero(exact))):
        cos, sin = after.cos.values[i], after.sin.values[i]
        length = math.hypot(cos, sin) * GLYPH
        if length < 2.5:
            ax.plot(th[i], after.speed.values[i], "o", ms=2.2, color="k", zorder=4)
        else:
            _draw_glyph(ax, th[i], after.speed.values[i], math.atan2(sin, cos), length)
    top = ax.get_ylim()[1]
    units = depth // len(trained)
    for theta in trained:
        ax.annotate(
            "",
            xy=(theta, top),
            xytext=(0, 14),
            textcoords="offset points",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.8, "shrinkB": 0},
        )
        ax.annotate(
            str(units),
            xy=(theta, top),
            xytext=(4, 9),
            textcoords="offset points",
            color=RED,
            fontsize=7,
            va="center",
            annotation_clip=False,
        )
    ax.set_title(title, pad=16)


def plot(rows: pd.DataFrame, seed: int, out: Path) -> None:
    """The robot that trained every site, for one seed."""
    rows = rows[(rows.seed == seed) & (rows.train_site < 0)]
    assert len(rows), f"No rows for seed {seed} that trained every site"
    depth = int(rows.depth.max())
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.set_ylim(0, 1.3)
    ax.set_xscale("log")
    ax.set_xlim(0.015, 0.3)
    title = f"{depth // len(LEVELS)} units at each θ"
    _draw_panel(ax, rows, depth, list(LEVELS), title)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"{v:g}" for v in LEVELS])
    ax.minorticks_off()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("θ (friction)")
    ax.set_ylabel("ψ: push speed (m/s)")
    handles = [
        Patch(
            color=GRAY, alpha=0.35, label="ψ*(θ): speeds that land the puck (aimed ↑)"
        ),
        Line2D(
            [], [], color="k", ls="--", lw=0.9, label="before sleep (random pushes)"
        ),
        Line2D(
            [], [], color="k", lw=1.6, label="after sleep: median speed of ψ's pushes"
        ),
        Line2D(
            [],
            [],
            color="k",
            lw=0,
            marker=r"$\uparrow$",
            markersize=8,
            label="ψ's direction: ↑ at the goal; short = scattered",
        ),
        Line2D([], [], color=RED, lw=1.6, label="units trained at that θ"),
        Patch(color=BLUE, label="more of ψ's pushes land"),
        Patch(color=ORANGE, label="fewer of ψ's pushes land"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=7.5,
    )
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Measure snapshots or plot measurements."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("snapshot", type=Path)
    measure_parser.add_argument("--num_draws", type=int, default=256)
    measure_parser.add_argument("--sites", type=int, nargs="+")
    measure_parser.add_argument("--out", type=Path, required=True)
    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("csvs", type=Path, nargs="+")
    plot_parser.add_argument("--seed", type=int, default=0)
    plot_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "measure":
        with open(args.snapshot, "rb") as f:
            snapshot = pickle.load(f)
        rows = measure(snapshot, THETAS.tolist(), args.num_draws, args.sites)
        rows.to_csv(args.out, index=False)
    else:
        rows = pd.concat(pd.read_csv(path) for path in args.csvs)
        plot(rows, args.seed, args.out)


if __name__ == "__main__":
    main()
