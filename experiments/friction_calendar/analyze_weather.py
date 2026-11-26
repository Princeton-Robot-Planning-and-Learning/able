"""Tables and figures for the friction weather sweep, laid out as
<runs>/<calendar>_wx_u<units>_L<lookahead>/<approach>_s<seed>/eval_results.csv.

    python experiments/friction_calendar/analyze_weather.py --runs runs --out figs
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

APPROACHES = {
    "transfer_practice": ("Ours", "#2a78d6"),
    "deliberate_practice": ("Deliberate Practice", "#eb6834"),
    "current_belief": ("Current belief", "#6250d6"),
    "dreaming": ("Dreaming", "#e87ba4"),
    "belief_coverage": ("Coverage", "#eda100"),
    "one_step_greedy": ("One-step greedy", "#1baf7a"),
}
SIMPLE = ("current_belief", "dreaming", "belief_coverage")
CALENDARS = {"cw": "Cleaning week + weather", "dr": "Drifting week A + weather"}
FIXED = ("current_belief", "scratch")


def load(roots: list[Path]) -> pd.DataFrame:
    """Every run's evaluations, with current belief and scratch, which ignore the
    lookahead, copied to every lookahead."""
    frames = []
    for root in roots:
        for path in root.glob("*_wx_u*_L*/*_s*/eval_results.csv"):
            m = re.fullmatch(r"(\w+)_wx_u(\d+)_L(\d+)", path.parent.parent.name)
            assert m, path
            approach, seed = path.parent.name.rsplit("_s", 1)
            d = pd.read_csv(path)
            d["calendar"], d["units"], d["lookahead"] = (
                m.group(1),
                int(m.group(2)),
                int(m.group(3)),
            )
            d["approach"], d["seed"] = approach, int(seed)
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    fixed = d[d.approach.isin(FIXED) & (d.lookahead == 4)]
    extra = [
        fixed[(fixed.calendar == c) & (fixed.units == u)].assign(lookahead=L)
        for c, u, L in d[["calendar", "units", "lookahead"]]
        .drop_duplicates()
        .itertuples(index=False)
        if L != 4
    ]
    return pd.concat([d] + extra, ignore_index=True)


def bootstrap(diffs: np.ndarray, rng: np.random.Generator, n: int = 10000):
    """The mean of diffs and the 95% bootstrap interval of the mean."""
    means = rng.choice(diffs, size=(n, len(diffs))).mean(axis=1)
    return diffs.mean(), *np.quantile(means, [0.025, 0.975])


def main() -> None:
    """Write the summary table and the figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-day", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = load(args.runs)
    keys = ["calendar", "units", "lookahead", "approach", "seed", "cycle"]
    day = (
        d.groupby(keys)
        .agg(
            slow=("success", lambda s: float(not s.all())),
            success=("success", "mean"),
            time=("planning_wall_time", "mean"),
        )
        .reset_index()
    )
    late = day[(day.cycle >= args.first_day) & (day.approach != "scratch")]
    per_seed = late.groupby(keys[:-1])[["slow", "success", "time"]].mean()
    means = per_seed.groupby(keys[:-2]).mean()
    rng = np.random.default_rng(0)
    settings = sorted(per_seed.index.droplevel(["approach", "seed"]).unique())
    lines = [
        f"Days {args.first_day}-27. Cells: slow-day share / task success / "
        "planning time (s).",
        "",
    ]
    header = (
        "| Approach | "
        + " | ".join(
            f"{CALENDARS[c].split(' +')[0]}, {u} units, L={L}" for c, u, L in settings
        )
        + " |"
    )
    lines += [header, "|---" * (len(settings) + 1) + "|"]
    for a, (label, _) in APPROACHES.items():
        cells = []
        for s in settings:
            row = means.loc[(*s, a)]
            cells.append(f"{row.slow:.2f} / {row.success:.2f} / {row.time:.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Gap (mean [95% CI]) | " + header.split("| Approach | ")[1],
        "|---" * (len(settings) + 1) + "|",
    ]
    gaps: list[tuple[str, str, str | None, str | None]] = [
        ("Task success: ours − best simple", "success", "transfer_practice", None),
        ("Task success: DP − best simple", "success", "deliberate_practice", None),
        (
            "Task success: ours − DP",
            "success",
            "transfer_practice",
            "deliberate_practice",
        ),
        ("Slow days: best simple − ours", "slow", None, "transfer_practice"),
        ("Slow days: best simple − DP", "slow", None, "deliberate_practice"),
    ]
    for label, metric, first, second in gaps:
        cells = []
        for s in settings:
            table = per_seed.loc[s][metric].unstack("approach")
            simple_means = {x: float(table[x].mean()) for x in SIMPLE}
            if metric == "success":
                best = max(simple_means, key=simple_means.__getitem__)
            else:
                best = min(simple_means, key=simple_means.__getitem__)
            a2 = best if first is None else first
            b2 = best if second is None else second
            m, lo, hi = bootstrap((table[a2] - table[b2]).to_numpy(), rng)
            cells.append(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    shown = [s for s in settings if s[2] == 4]
    fig, axes = plt.subplots(3, len(shown), figsize=(3.8 * len(shown), 10), sharex=True)
    for col, (c, u, L) in enumerate(shown):
        sub = day[
            (day.calendar == c)
            & (day.units == u)
            & (day.lookahead == L)
            & (day.approach != "scratch")
        ]
        for row, (metric, ylabel) in enumerate(
            [
                ("slow", "Chance of a slow day"),
                ("success", "Task success rate"),
                ("time", "Planning time per task (s)"),
            ]
        ):
            ax = axes[row, col]
            for a, (label, color) in APPROACHES.items():
                m = sub[sub.approach == a].groupby("cycle")[metric].mean()
                r = m.rolling(3, center=True, min_periods=2).mean()
                thick = a in ("transfer_practice", "deliberate_practice")
                ax.plot(
                    r.index,
                    r.values,
                    color=color,
                    lw=2.4 if thick else 1.4,
                    alpha=1 if thick else 0.85,
                    label=label,
                )
            for w in (7, 14, 21):
                ax.axvline(w - 0.5, color="#8a8984", lw=0.8, ls=":")
            ax.set_ylim(0, 1.02 if row < 2 else None)
            if row < 2 and col > 0:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"{CALENDARS[c]}, {u} units", fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Day (cycle); dotted = new week")
            ax.set_xlim(0, 27)
            ax.set_xticks([0, 7, 14, 21, 27])
            ax.grid(alpha=0.25, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out / "weather_per_day.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(settings), figsize=(3.3 * len(settings), 3.6), sharey=True
    )
    for ax, s in zip(np.atleast_1d(axes), settings):
        table = per_seed.loc[s]["success"].unstack("approach")
        order = list(APPROACHES)
        for i, a in enumerate(order):
            m, lo, hi = bootstrap(table[a].to_numpy(), rng)
            ax.errorbar(
                m,
                i,
                xerr=[[m - lo], [hi - m]],
                fmt="o",
                color=APPROACHES[a][1],
                ms=7,
                capsize=3,
            )
        ax.set_yticks(range(len(order)), [APPROACHES[a][0] for a in order])
        ax.invert_yaxis()
        c, u, L = s
        ax.set_title(
            f"{CALENDARS[c].split(' +')[0]}\n{u} units, lookahead {L}", fontsize=9.5
        )
        ax.set_xlabel(f"Task success, days {args.first_day}+")
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "weather_success.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
