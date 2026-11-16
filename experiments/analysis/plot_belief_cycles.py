"""Plot how the robot's belief moves over cycles, with one line per approach.

For each hidden parameter, a cycle's belief (belief_mean_<i> and belief_std_<i> in
cycle_results.csv) is compared with the prior and with the true value: its
standard deviation and its mean's distance from the truth, both divided by the
prior's standard deviation. Cycle -1 is the prior. The prior's moments and the true
parameters come from rebuilding each run's domain from its config.yaml, so run this
at the runs' code version.

Writes belief_error.png and belief_std.png (averaged over parameters; one row per
environment, one column per number of queries per cycle,
budgets.information_gathering) and belief_per_parameter_<env>.png, and prints a
table of both measures at a few cycles. The dotted line is the prior's.

Example:
    python experiments/analysis/plot_belief_cycles.py --results_dir logs/belief
"""

import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from omegaconf import DictConfig, OmegaConf
from plot_cycle_results import (
    APPROACH_TO_COLOR,
    APPROACH_TO_DISPLAY_NAME,
    APPROACH_TO_LINESTYLE,
    ENV_TO_DISPLAY_NAME,
    EXTRA_COLORS,
)

from able.learners.belief_features import (  # pylint: disable=wrong-import-order
    get_moments,
)

MEASURES = [
    ("error_ratio", "Belief error (prior std)", "belief_error.png"),
    ("std_ratio", "Belief std (prior std)", "belief_std.png"),
]
TABLE_CYCLES = [-1, 0, 3, 6, 9, 11, 14, 19]
CAPTION = (
    "Error is the distance from the belief's mean to the true parameter. Lines show "
    "the mean over seeds of the mean over parameters; shaded bands show one standard "
    "error; the dotted line is the prior's."
)


def _prior_and_truth(cfg: DictConfig) -> dict[str, list[float]]:
    domain = hydra.utils.instantiate(cfg.env.domain, seed=cfg.seed)
    mean, std = get_moments(domain.initial_belief)
    truth = np.asarray(domain.real_env.hidden_params, dtype=np.float64).ravel()
    return {
        "prior_mean": np.asarray(mean, dtype=np.float64).tolist(),
        "prior_std": np.asarray(std, dtype=np.float64).tolist(),
        "truth": truth.tolist(),
    }


def load_belief_rows(results_dir: Path) -> pd.DataFrame:
    """One row per run, cycle and hidden parameter, with the cycle -1 prior."""
    cache: dict[str, dict[str, list[float]]] = {}
    frames = []
    for config_path in sorted(results_dir.rglob("config.yaml")):
        cycles_path = config_path.parent / "cycle_results.csv"
        if not cycles_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig)
        cycles = pd.read_csv(cycles_path)
        if "belief_mean_0" not in cycles:
            continue
        key = json.dumps(OmegaConf.to_container(cfg.env.domain), sort_keys=True)
        key += f"|{cfg.seed}"
        if key not in cache:
            cache[key] = _prior_and_truth(cfg)
        moments = cache[key]
        queries = int(
            OmegaConf.select(cfg, "budgets.information_gathering.max_iterations") or 0
        )
        for i, (prior_mean, prior_std, truth) in enumerate(
            zip(moments["prior_mean"], moments["prior_std"], moments["truth"])
        ):
            means = np.concatenate([[prior_mean], cycles[f"belief_mean_{i}"]])
            stds = np.concatenate([[prior_std], cycles[f"belief_std_{i}"]])
            frames.append(
                pd.DataFrame(
                    {
                        "env": cfg.env_name,
                        "approach": cfg.approach_name,
                        "queries": queries,
                        "seed": cfg.seed,
                        "cycle": np.concatenate([[-1], cycles["cycle"]]),
                        "parameter": i,
                        "std_ratio": stds / prior_std,
                        "error_ratio": np.abs(means - truth) / prior_std,
                    }
                )
            )
    assert frames, f"No cycle_results.csv with beliefs found under {results_dir}"
    return pd.concat(frames, ignore_index=True)


def _per_seed(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """The measure averaged over parameters, per approach, setting, seed and cycle."""
    keys = ["approach", "queries", "seed", "cycle"]
    return df.groupby(keys)[measure].mean().reset_index()


def _query_title(queries: int) -> str:
    if queries == 0:
        return "no queries"
    return f"{queries} quer{'y' if queries == 1 else 'ies'} per cycle"


def _approach_styles(df: pd.DataFrame) -> dict[str, dict]:
    extra = iter(EXTRA_COLORS)
    styles = {}
    for approach in sorted(df["approach"].unique()):
        styles[approach] = {
            "label": APPROACH_TO_DISPLAY_NAME.get(approach, approach),
            "color": APPROACH_TO_COLOR.get(approach) or next(extra),
            "linestyle": APPROACH_TO_LINESTYLE.get(approach, "-"),
        }
    return styles


def _plot_panel(ax, df: pd.DataFrame, measure: str, styles: dict) -> None:
    """One line per approach from cycle 0, and the prior (cycle -1) dotted."""
    per_seed = _per_seed(df, measure)
    prior = per_seed[per_seed["cycle"] == -1][measure].mean()
    ax.axhline(prior, color="0.3", linestyle=":", linewidth=1.2)
    cycles = per_seed[per_seed["cycle"] >= 0]
    for approach, group in cycles.groupby("approach"):
        summary = group.groupby("cycle")[measure].agg(["mean", "sem"]).fillna(0.0)
        style = styles[approach]
        ax.plot(summary.index, summary["mean"], **style)
        ax.fill_between(
            summary.index,
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            color=style["color"],
            alpha=0.2,
            linewidth=0,
        )
    ax.set_xlabel("Cycle")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0.0, max(1.1, ax.get_ylim()[1]))
    ax.grid(alpha=0.3)


def _add_legend(fig, styles: dict, top_inches: float = 0.0) -> float:
    """Puts the legend above the panels; returns the panels' top as a fraction."""
    handles = [
        Line2D([], [], color=s["color"], linestyle=s["linestyle"])
        for s in styles.values()
    ]
    labels = [s["label"] for s in styles.values()]
    ncol = min(5, len(labels))
    rows = -(-len(labels) // ncol)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False)
    return 1.0 - (0.28 * rows + 0.15 + top_inches) / fig.get_figheight()


def plot_measure(df: pd.DataFrame, measure: str, label: str, path: Path) -> None:
    """Rows are environments, columns are query settings."""
    styles = _approach_styles(df)
    envs = sorted(df["env"].unique())
    settings = sorted(df["queries"].unique())
    fig, axes = plt.subplots(
        len(envs),
        len(settings),
        figsize=(4.6 * len(settings), 3.4 * len(envs)),
        squeeze=False,
    )
    for row, env in enumerate(envs):
        for col, queries in enumerate(settings):
            ax = axes[row][col]
            panel = df[(df["env"] == env) & (df["queries"] == queries)]
            if panel.empty:
                ax.set_visible(False)
                continue
            _plot_panel(ax, panel, measure, styles)
            num_seeds = panel["seed"].nunique()
            ax.set_title(
                f"{ENV_TO_DISPLAY_NAME.get(env, env)}, {_query_title(queries)} "
                f"({num_seeds} seeds)",
                fontsize=10,
            )
            if col == 0:
                ax.set_ylabel(label)
    top = _add_legend(fig, styles)
    fig.text(0.5, 0.005, CAPTION, ha="center", fontsize=8, color="0.4", wrap=True)
    fig.tight_layout(rect=(0, 0.35 / fig.get_figheight(), 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_per_parameter(df: pd.DataFrame, env: str, path: Path) -> None:
    """Both measures for each parameter of one environment and query setting."""
    env_df = df[df["env"] == env]
    styles = _approach_styles(env_df)
    params = sorted(env_df["parameter"].unique())
    rows = [
        (measure, label, queries)
        for measure, label, _ in MEASURES
        for queries in sorted(env_df["queries"].unique())
    ]
    fig, axes = plt.subplots(
        len(rows),
        len(params),
        figsize=(2.8 * len(params), 2.7 * len(rows)),
        squeeze=False,
    )
    for col, param in enumerate(params):
        param_df = env_df[env_df["parameter"] == param]
        for row, (measure, label, queries) in enumerate(rows):
            ax = axes[row][col]
            _plot_panel(ax, param_df[param_df["queries"] == queries], measure, styles)
            ax.set_title(f"Parameter {param}, {_query_title(queries)}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label)
    top = _add_legend(fig, styles, top_inches=0.3)
    num_seeds = env_df["seed"].nunique()
    fig.text(
        0.5,
        top + 0.12 / fig.get_figheight(),
        f"{ENV_TO_DISPLAY_NAME.get(env, env)} ({num_seeds} seeds)",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def print_table(df: pd.DataFrame) -> None:
    """Error / std at a few cycles, averaged over parameters and seeds."""
    measures = [m for m, _, _ in MEASURES]
    per_run = df.groupby(
        ["env", "approach", "queries", "seed", "cycle"], as_index=False
    )[measures].mean()
    means = per_run.groupby(["env", "approach", "queries", "cycle"])[measures].mean()
    cycles = [c for c in TABLE_CYCLES if c in set(per_run["cycle"])]
    header = " / ".join("prior" if c == -1 else str(c) for c in cycles)
    print(
        f"\n| Env | Approach | Queries | Error (prior std), cycle {header} "
        "| Std (prior std), same cycles |"
    )
    print("| --- | --- | --- | --- | --- |")
    for key, group in means.groupby(level=["env", "approach", "queries"]):
        group = group.droplevel(["env", "approach", "queries"])
        cells = [
            " / ".join(f"{value:.2f}" for c, value in group[m].items() if c in cycles)
            for m in measures
        ]
        env, approach, queries = (str(k) for k in key)
        print(f"| {env} | {approach} | {queries} | {cells[0]} | {cells[1]} |")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_belief_rows(args.results_dir)
    df.to_csv(output_dir / "belief_cycles.csv", index=False)
    for measure, label, filename in MEASURES:
        plot_measure(df, measure, label, output_dir / filename)
    for env in sorted(df["env"].unique()):
        plot_per_parameter(df, env, output_dir / f"belief_per_parameter_{env}.png")
    print_table(df)


if __name__ == "__main__":
    _main()
