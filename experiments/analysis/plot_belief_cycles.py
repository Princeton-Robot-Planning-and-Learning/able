"""Plot how the robot's belief moves over cycles, with one line per approach.

For each hidden parameter, a cycle's belief (belief_mean_<i> and belief_std_<i> in
cycle_results.csv) is compared with the prior and with the true value: its
standard deviation and its mean's distance from the truth, both divided by the
prior's standard deviation. Cycle -1 is the prior. The prior's moments and the true
parameters come from rebuilding each run's domain from its config.yaml, so run this
at the runs' code version.

Writes belief_cycles.png (averaged over parameters, one row per environment) and
belief_per_parameter_<env>.png, and prints a table of both measures at a few cycles.
Runs that differ in queries per cycle (budgets.information_gathering) get one line
per setting.

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
from matplotlib.ticker import MaxNLocator
from omegaconf import DictConfig, OmegaConf
from plot_cycle_results import (
    APPROACH_TO_COLOR,
    APPROACH_TO_DISPLAY_NAME,
    ENV_TO_DISPLAY_NAME,
    EXTRA_COLORS,
)

from able.learners.belief_features import (  # pylint: disable=wrong-import-order
    get_moments,
)

MEASURES = [
    ("std_ratio", "Belief std ÷ prior std"),
    ("error_ratio", "|Mean − truth| ÷ prior std"),
]
TABLE_CYCLES = [-1, 0, 3, 6, 9, 11, 14, 19]
CAPTION = (
    "Lines show the mean over seeds (of the mean over parameters); shaded bands show "
    "one standard error. Cycle −1 is the prior."
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


def _summarize(df: pd.DataFrame, measure: str, keys: list[str]) -> pd.DataFrame:
    per_seed = df.groupby(keys + ["seed", "cycle"], as_index=False)[measure].mean()
    summary = per_seed.groupby(keys + ["cycle"], as_index=False)[measure].agg(
        mean="mean", sem="sem"
    )
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary


def _line_styles(df: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """A label, color and line style for each approach and query setting."""
    many_query_settings = df["queries"].nunique() > 1
    extra = iter(EXTRA_COLORS)
    colors: dict[str, str] = {}
    styles = {}
    for approach, queries in sorted(
        df[["approach", "queries"]].drop_duplicates().itertuples(index=False)
    ):
        if approach not in colors:
            colors[approach] = APPROACH_TO_COLOR.get(approach) or next(extra)
        label = APPROACH_TO_DISPLAY_NAME.get(approach, approach)
        if many_query_settings:
            label += f", {queries} quer{'y' if queries == 1 else 'ies'} per cycle"
        styles[(approach, queries)] = {
            "label": label,
            "color": colors[approach],
            "linestyle": "--" if many_query_settings and queries > 0 else "-",
        }
    return styles


def _plot_lines(ax, summary: pd.DataFrame, styles: dict) -> None:
    for (approach, queries), group in summary.groupby(["approach", "queries"]):
        style = styles[(approach, queries)]
        group = group.sort_values("cycle")
        ax.plot(group["cycle"], group["mean"], marker=".", **style)
        ax.fill_between(
            group["cycle"],
            group["mean"] - group["sem"],
            group["mean"] + group["sem"],
            color=style["color"],
            alpha=0.2,
            linewidth=0,
        )
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("Cycle")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)


def _add_legend(fig, ax, top_inches: float = 0.0) -> float:
    """Puts the legend above the panels; returns the panels' top as a fraction."""
    handles, labels = ax.get_legend_handles_labels()
    ncol = min(3, len(labels))
    rows = -(-len(labels) // ncol)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False)
    return 1.0 - (0.28 * rows + 0.15 + top_inches) / fig.get_figheight()


def plot_average(df: pd.DataFrame, path: Path) -> None:
    """Both measures averaged over parameters, one row per environment."""
    styles = _line_styles(df)
    envs = sorted(df["env"].unique())
    # Average over parameters within each seed and cycle first.
    per_run = df.groupby(
        ["env", "approach", "queries", "seed", "cycle"], as_index=False
    )[[m for m, _ in MEASURES]].mean()
    fig, axes = plt.subplots(
        len(envs), len(MEASURES), figsize=(11, 3.4 * len(envs)), squeeze=False
    )
    for row, env in enumerate(envs):
        for col, (measure, label) in enumerate(MEASURES):
            ax = axes[row][col]
            env_df = per_run[per_run["env"] == env]
            summary = _summarize(env_df, measure, ["approach", "queries"])
            _plot_lines(ax, summary, styles)
            num_seeds = env_df["seed"].nunique()
            ax.set_title(f"{ENV_TO_DISPLAY_NAME.get(env, env)} ({num_seeds} seeds)")
            ax.set_ylabel(label)
    top = _add_legend(fig, axes[0][0])
    fig.text(0.5, 0.005, CAPTION, ha="center", fontsize=9, color="0.4")
    fig.tight_layout(rect=(0, 0.3 / fig.get_figheight(), 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_per_parameter(df: pd.DataFrame, env: str, path: Path) -> None:
    """Both measures for each parameter of one environment."""
    env_df = df[df["env"] == env]
    styles = _line_styles(env_df)
    params = sorted(env_df["parameter"].unique())
    fig, axes = plt.subplots(
        len(MEASURES),
        len(params),
        figsize=(3.2 * len(params), 6.2),
        squeeze=False,
    )
    for col, param in enumerate(params):
        for row, (measure, label) in enumerate(MEASURES):
            ax = axes[row][col]
            summary = _summarize(
                env_df[env_df["parameter"] == param], measure, ["approach", "queries"]
            )
            _plot_lines(ax, summary, styles)
            ax.set_title(f"Parameter {param}")
            if col == 0:
                ax.set_ylabel(label)
    top = _add_legend(fig, axes[0][0], top_inches=0.3)
    num_seeds = env_df["seed"].nunique()
    fig.text(
        0.5,
        top + 0.15 / fig.get_figheight(),
        f"{ENV_TO_DISPLAY_NAME.get(env, env)} ({num_seeds} seeds)",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def print_table(df: pd.DataFrame) -> None:
    """Std / error at a few cycles, averaged over parameters and seeds."""
    per_run = df.groupby(
        ["env", "approach", "queries", "seed", "cycle"], as_index=False
    )[[m for m, _ in MEASURES]].mean()
    means = per_run.groupby(["env", "approach", "queries", "cycle"])[
        [m for m, _ in MEASURES]
    ].mean()
    cycles = [c for c in TABLE_CYCLES if c in set(per_run["cycle"])]
    header = " / ".join("prior" if c == -1 else str(c) for c in cycles)
    print(
        f"\n| Env | Approach | Queries | Std ÷ prior std, cycle {header} "
        f"| Error ÷ prior std, same cycles |"
    )
    print("| --- | --- | --- | --- | --- |")
    for key, group in means.groupby(level=["env", "approach", "queries"]):
        group = group.droplevel(["env", "approach", "queries"])
        cells = [
            " / ".join(f"{value:.2f}" for c, value in group[m].items() if c in cycles)
            for m, _ in MEASURES
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
    plot_average(df, output_dir / "belief_cycles.png")
    for env in sorted(df["env"].unique()):
        plot_per_parameter(df, env, output_dir / f"belief_per_parameter_{env}.png")
    print_table(df)


if __name__ == "__main__":
    _main()
