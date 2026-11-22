"""Plot offline evaluation results against cycles, with one line per approach.

An environment whose runs use several simulation budgets (friction's
approach.num_sims) is plotted against cycles at its largest budget, and its success
rate is also plotted against the budget, in budget_results.png.

An environment whose evaluation covers several sites (friction_sites) plots each
metric weighted by the sites' shares of the schedule, and the chance of a slow day.

Example:
    python experiments/analysis/plot_cycle_results.py --results_dir logs/comparison
"""

import argparse
import functools
from pathlib import Path
from typing import Any, Sequence

import hydra
import numpy as np
import pandas as pd
from analysis_utils import combine_results_csvs, summarize_over_seeds
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

APPROACH_TO_DISPLAY_NAME = {
    "belief_coverage": "Belief coverage",
    "current_belief": "Current belief only",
    "deliberate_practice": "Deliberate practice",
    "dreaming": "Dreaming",
    "one_step_greedy": "One-step greedy",
    "prior_only": "Prior only (first sleep)",
    "scratch": "Plan from scratch",
    "transfer_practice": "Transfer practice",
}

# Each approach keeps its color in every plot. Approaches without an entry take
# the remaining colors in this order.
APPROACH_TO_COLOR = {
    "dreaming": "#2a78d6",
    "scratch": "#eb6834",
    "current_belief": "#1baf7a",
    "prior_only": "#eda100",
    "belief_coverage": "#e87ba4",
    "transfer_practice": "#6250d6",
    "deliberate_practice": "#008300",
    "one_step_greedy": "#eda100",
}
EXTRA_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]

# The prior-only line often overlaps the scratch line, and their colors are close,
# so it is also dashed. One-step greedy shares prior only's color, so it is dotted.
APPROACH_TO_LINESTYLE = {
    "prior_only": "--",
    "one_step_greedy": ":",
}

ENV_TO_DISPLAY_NAME = {
    "two_link_arm": "Two-link arm",
    "limb": "Limb repositioning",
    "friction": "Friction",
    "friction_sites": "Friction sites",
    "human_reach": "Human reach",
}

# Each panel plots one column of eval_results.csv: (column, axis label). Runs that
# record timeouts get a third panel. Planning time averages every task, a timed-out
# one at the time it took, so a robot that stops timing out does not seem slower
# for the hard tasks it now finishes.
METRICS = [
    ("planning_wall_time", "Planning time per task"),
    ("success", "Success rate"),
]
TIMEOUT_METRIC = ("timed_out", "Timeout rate")
# Environments whose rows also plot the planner's work per task, which does not
# depend on the machine: limb's MPC rollouts.
WORK_PANEL_ENVS = ("limb",)
RATE_METRICS = ("success", "timed_out", "slow_day")

CAPTION = "Lines show the mean over seeds; shaded bands show one standard error."
TIMEOUT_CAPTION = (
    "Timed-out tasks count in planning time at the time they took, count as "
    "failures, and are shown on the right."
)

# The planner's work per evaluation task, from its statistics, which do not depend
# on the machine: (label, numerator column, denominator column or None).
ENV_TO_PLANNER_WORK = {
    "two_link_arm": ("Samples per task", "num_samples", None),
    "limb": ("MPC rollouts per task", "num_rollouts", None),
    "friction": ("Simulations per push", "num_sims", "num_searches"),
    "friction_sites": ("Simulations per task", "num_sims", None),
    "human_reach": ("Stream calls per task", "stream_calls", None),
}

# Rows with a site and its share of the schedule are averaged per site and then
# weighted by the shares. A slow day is one whose site has a failed task or a task
# whose planner work is above a threshold. Runs of these environments from before
# the rows recorded sites get them from the run's domain.
SITE_DOMAIN_ENVS = ("friction_sites",)
SLOW_DAY_METRIC = ("slow_day", "Chance of a slow day")

# Success against the simulation budget is taken over the last cycles, after the
# robots have learned.
BUDGET_WINDOW = 5
BUDGET_CAPTION = (
    "Each point is the mean over seeds of the success rate over the last {window}"
    " cycles; bars show one standard error.\nThe dotted line is planning from scratch"
    " at {budget} simulations per push."
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


def _get_env_panels(
    env: str, env_df: pd.DataFrame, metrics: list[tuple[str, str]]
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """One environment's rows and panels: the shared metrics, plus the planner's
    work per task in WORK_PANEL_ENVS when the runs record it. Schedule-weighted
    rows also plot their planner work and the chance of a slow day."""
    if _is_weighted(env_df):
        panels = [metric for metric in metrics if metric != TIMEOUT_METRIC]
        if _has_values(env_df, "planner_work"):
            panels.append(("planner_work", ENV_TO_PLANNER_WORK[env][0]))
        panels.append(SLOW_DAY_METRIC)
        return env_df, panels + [m for m in metrics if m == TIMEOUT_METRIC]
    work = _get_planner_work(env, env_df) if env in WORK_PANEL_ENVS else None
    if work is None:
        return env_df, list(metrics)
    label, values = work
    return env_df.assign(planner_work=values), [*metrics, ("planner_work", label)]


def add_missing_sites(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """For a run in SITE_DOMAIN_ENVS whose rows lack them, each task's site and
    its share of the schedule, from the run's domain, whose evaluation tasks go
    site by site, a routine each."""
    if cfg.env_name not in SITE_DOMAIN_ENVS or "site" in df.columns:
        return df
    routines = _get_task_distribution(OmegaConf.to_yaml(cfg.env.domain), int(cfg.seed))
    sites = df["task_id"].to_numpy() // len(routines.get_routine(0))
    return df.assign(site=sites, schedule_share=routines.schedule[sites])


@functools.cache
def _get_task_distribution(domain_yaml: str, seed: int) -> Any:
    domain = hydra.utils.instantiate(OmegaConf.create(domain_yaml), seed=seed)
    return domain.task_distribution


def has_sites(df: pd.DataFrame) -> bool:
    """Whether every row has a site and the site's share of the schedule."""
    columns = ["site", "schedule_share"]
    return set(columns) <= set(df.columns) and bool(df[columns].notna().all().all())


def get_slow_work(
    df: pd.DataFrame, slow_work: float | None
) -> tuple[float | None, str]:
    """The planner work per task above which a task makes its day slow, and where
    it comes from: slow_work if given, else the median of scratch's tasks, else
    None, so only failures do."""
    if slow_work is not None:
        return slow_work, "given"
    scratch = pd.Series(dtype=np.float64)
    if "planner_work" in df.columns:
        scratch = df.loc[df["approach"] == "scratch", "planner_work"].dropna()
    if scratch.empty:
        return None, "no scratch runs"
    return float(scratch.median()), "scratch's median"


def weigh_sites(
    df: pd.DataFrame, metrics: Sequence[str], slow_work: float | None
) -> pd.DataFrame:
    """One row per run and cycle: each metric averaged over each site's tasks and
    then over sites by their shares of the schedule, and slow_day, the share at
    sites with a failed task or a task whose planner_work is above slow_work.
    Shares are normalized over the sites evaluated, and a site without a value
    of a metric is left out of its average."""
    keys = [
        key
        for key in ("env", "approach", "seed", "cycle", "simulation_budget")
        if key in df.columns
    ]
    slow = ~df["success"].astype(bool)
    if slow_work is not None:
        slow |= df["planner_work"] > slow_work
    per_site = (
        df.assign(slow_day=slow)
        .astype({metric: np.float64 for metric in [*metrics, "slow_day"]})
        .groupby([*keys, "site"], dropna=False)
        .agg(
            {
                **{metric: "mean" for metric in metrics},
                "slow_day": "max",
                "schedule_share": "first",
            }
        )
    )
    values = per_site[[*metrics, "slow_day"]]
    shares = per_site["schedule_share"]
    totals = values.mul(shares, axis=0).groupby(level=keys, dropna=False).sum()
    weights = values.notna().mul(shares, axis=0).groupby(level=keys, dropna=False)
    return (totals / weights.sum()).reset_index()


def _weigh_site_envs(
    df: pd.DataFrame, metrics: Sequence[str], slow_work: float | None
) -> tuple[pd.DataFrame, list[str]]:
    """The rows, with each environment's that have sites replaced by their
    schedule-weighted rows per run and cycle, and a caption line for each such
    environment."""
    frames, captions = [], []
    for env, env_df in df.groupby("env"):
        if not has_sites(env_df):
            frames.append(env_df)
            continue
        columns = list(metrics)
        threshold = None
        caption = (
            f"{ENV_TO_DISPLAY_NAME.get(str(env), env)}: each site's mean is weighted"
            " by its share of the schedule. A slow day's site has a failed task"
        )
        work = _get_planner_work(str(env), env_df)
        if work is not None:
            env_df = env_df.assign(planner_work=work[1])
            columns.append("planner_work")
            threshold, source = get_slow_work(env_df, slow_work)
            caption += (
                f" ({source} to set a threshold on {work[0].lower()})"
                if threshold is None
                else f" or a task above {threshold:g} {work[0].lower()} ({source})"
            )
        frames.append(weigh_sites(env_df, columns, threshold))
        captions.append(caption + ".")
    if not captions:
        return df, captions
    # Weighted rates are fractions, so every environment's become floats.
    combined = pd.concat(frames, ignore_index=True)
    return combined.astype({metric: np.float64 for metric in metrics}), captions


def _is_weighted(df: pd.DataFrame) -> bool:
    return _has_values(df, "slow_day")


def _has_values(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and bool(df[column].notna().any())


def _main(results_dir: Path, outfile: Path, slow_work: float | None = None) -> None:
    all_df = combine_results_csvs(results_dir, "eval_results.csv", add_missing_sites)
    budget_envs = _get_budget_envs(all_df)
    # Against cycles, an environment with several budgets shows its largest.
    df = _at_full_budget(all_df)
    metrics = list(METRICS)
    if _has_timeouts(df):
        # Runs without the column, mixed with runs that have it, had no timeouts
        # recorded.
        df["timed_out"] = df["timed_out"].fillna(False).astype(bool)
        metrics.append(TIMEOUT_METRIC)
    df, site_captions = _weigh_site_envs(df, [m for m, _ in metrics], slow_work)
    envs = sorted(df["env"].unique())
    approaches = sorted(df["approach"].unique())
    colors = _get_colors(approaches)
    panels = {env: _get_env_panels(env, df[df["env"] == env], metrics) for env in envs}
    num_cols = max(len(env_metrics) for _, env_metrics in panels.values())

    fig, axes = plt.subplots(
        len(envs),
        num_cols,
        figsize=(5.5 * num_cols, 4.0 * len(envs)),
        squeeze=False,
    )
    for row, env in enumerate(envs):
        env_df, env_metrics = panels[env]
        num_seeds = env_df["seed"].nunique()
        # Every panel spans every cycle, so a cycle without results leaves a gap.
        cycles = np.sort(env_df["cycle"].unique())
        for col in range(len(env_metrics), num_cols):
            axes[row, col].axis("off")
        for col, (metric, label) in enumerate(env_metrics):
            scale, unit = 1.0, ""
            if metric == "planning_wall_time":
                scale, unit = _get_time_scale(env_df[metric])
                label = f"{label} ({unit})"
            ax = axes[row, col]
            summary = summarize_over_seeds(env_df, metric)
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
                    solid_capstyle="round",
                    label=APPROACH_TO_DISPLAY_NAME.get(approach, approach),
                )
            pad = 0.05 * (cycles[-1] - cycles[0]) if len(cycles) > 1 else 0.5
            ax.set_xlim(cycles[0] - pad, cycles[-1] + pad)
            ax.set_xlabel("Cycle")
            ax.set_ylabel(label)
            title = f"{ENV_TO_DISPLAY_NAME.get(env, env)} ({num_seeds} seeds"
            if env in budget_envs:
                budget = int(env_df["simulation_budget"].max())
                title += f", {budget} simulations per push"
            if _is_weighted(env_df):
                title += ", schedule-weighted"
            ax.set_title(title + ")", fontsize=11)
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
    caption = "\n".join([CAPTION, *site_captions])
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
    for line in site_captions:
        print(line)
    print(f"Wrote plot to {outfile}")
    cycle_df = _at_full_budget(combine_results_csvs(results_dir, "cycle_results.csv"))
    table = _make_table(df, cycle_df)
    if budget_envs:
        budget_df = all_df[all_df["env"].isin(budget_envs)]
        budget_file = outfile.with_name("budget_results.png")
        _plot_budgets(budget_df, colors, budget_file)
        print(f"Wrote plot to {budget_file}")
        table += "\n" + _make_budget_table(budget_df)
    table_file = outfile.with_name("summary_table.md")
    table_file.write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote table to {table_file}")


def _get_budget_envs(df: pd.DataFrame) -> list[str]:
    """The environments whose runs use more than one simulation budget."""
    if "simulation_budget" not in df.columns:
        return []
    budgets = df.groupby("env")["simulation_budget"].nunique()
    return sorted(budgets[budgets > 1].index)


def _at_full_budget(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of each environment at its largest simulation budget, and every row
    of environments without one."""
    if "simulation_budget" not in df.columns:
        return df
    largest = df.groupby("env")["simulation_budget"].transform("max")
    keep = df["simulation_budget"].isna() | (df["simulation_budget"] == largest)
    return df[keep]


def summarize_budgets(df: pd.DataFrame, window: int = BUDGET_WINDOW) -> pd.DataFrame:
    """For each environment, approach, and simulation budget: the success rate over
    the last `window` cycles, averaged over each seed's tasks and then over seeds
    (mean and standard error)."""
    last = df.groupby(["env", "approach", "simulation_budget", "seed"])[
        "cycle"
    ].transform("max")
    recent = df[df["cycle"] > last - window]
    per_seed = recent.groupby(["env", "approach", "simulation_budget", "seed"])[
        "success"
    ].mean()
    summary = per_seed.groupby(["env", "approach", "simulation_budget"]).agg(
        mean="mean", sem="sem", num_seeds="count"
    )
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary.reset_index()


def budget_to_match(
    budgets: np.ndarray, success: np.ndarray, target: float
) -> float | None:
    """The smallest budget at which the success rate, interpolated linearly in the
    budget's logarithm between measured budgets, reaches the target: the first
    budget if it already does, and None if no budget does."""
    order = np.argsort(budgets)
    budgets, success = budgets[order], success[order]
    for i, (budget, rate) in enumerate(zip(budgets, success)):
        if rate < target:
            continue
        if i == 0:
            return float(budget)
        low, high = np.log(budgets[i - 1]), np.log(budget)
        fraction = (target - success[i - 1]) / (rate - success[i - 1])
        return float(np.exp(low + fraction * (high - low)))
    return None


def _plot_budgets(df: pd.DataFrame, colors: dict[str, str], outfile: Path) -> None:
    """A panel per environment of the success rate against the simulation budget,
    with a line per approach and scratch's success at the largest budget as a
    reference."""
    summary = summarize_budgets(df)
    envs = sorted(summary["env"].unique())
    fig, axes = plt.subplots(
        1, len(envs), figsize=(6.0 * len(envs), 4.2), squeeze=False
    )
    full_budget = int(summary["simulation_budget"].max())
    for col, env in enumerate(envs):
        ax = axes[0, col]
        env_summary = summary[summary["env"] == env]
        for approach in sorted(env_summary["approach"].unique()):
            rows = env_summary[env_summary["approach"] == approach].sort_values(
                "simulation_budget"
            )
            ax.errorbar(
                rows["simulation_budget"],
                rows["mean"],
                yerr=rows["sem"],
                color=colors[approach],
                linestyle=APPROACH_TO_LINESTYLE.get(approach, "-"),
                linewidth=2,
                marker="o",
                markersize=4,
                capsize=3,
                label=APPROACH_TO_DISPLAY_NAME.get(approach, approach),
            )
        reference = env_summary[
            (env_summary["approach"] == "scratch")
            & (env_summary["simulation_budget"] == full_budget)
        ]
        if not reference.empty:
            ax.axhline(
                float(reference["mean"].iloc[0]),
                color="#8a8984",
                linestyle=":",
                linewidth=1.5,
            )
        budgets = sorted(env_summary["simulation_budget"].unique())
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(int(b)) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel("Simulations per push")
        ax.set_ylabel(f"Success rate, last {BUDGET_WINDOW} cycles")
        ax.set_ylim(0.0, 1.05)
        num_seeds = int(env_summary["num_seeds"].max())
        ax.set_title(
            f"{ENV_TO_DISPLAY_NAME.get(env, env)} ({num_seeds} seeds)", fontsize=11
        )
        ax.grid(color="#e4e3df", linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.text(
        0.5,
        -0.02,
        BUDGET_CAPTION.format(window=BUDGET_WINDOW, budget=full_budget),
        ha="center",
        va="top",
        fontsize=9,
        color="#52514e",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_budget_table(df: pd.DataFrame) -> str:
    """A Markdown table per environment of each approach's success rate at each
    simulation budget, and the budget it needs to match scratch at the largest one.
    """
    summary = summarize_budgets(df)
    sections = []
    for env in sorted(summary["env"].unique()):
        env_summary = summary[summary["env"] == env]
        budgets = sorted(int(b) for b in env_summary["simulation_budget"].unique())
        full = budgets[-1]
        reference = env_summary[
            (env_summary["approach"] == "scratch")
            & (env_summary["simulation_budget"] == full)
        ]
        target = None if reference.empty else float(reference["mean"].iloc[0])
        header = [f"{b} sims" for b in budgets]
        match_label = f"Sims per push to match scratch at {full}"
        lines = [
            f"### {ENV_TO_DISPLAY_NAME.get(env, env)}: success rate over the last"
            f" {BUDGET_WINDOW} cycles by simulation budget",
            "",
            "| | " + " | ".join(header + [match_label]) + " |",
            "| --- |" + " --- |" * (len(header) + 1),
        ]
        for approach in sorted(env_summary["approach"].unique()):
            rows = env_summary[env_summary["approach"] == approach].set_index(
                "simulation_budget"
            )
            cells = [
                (
                    f"{rows.loc[b, 'mean']:.2f} ± {rows.loc[b, 'sem']:.2f}"
                    if b in rows.index
                    else "–"
                )
                for b in budgets
            ]
            match = "–"
            if approach == "scratch":
                match = "reference"
            elif target is not None:
                needed = budget_to_match(
                    rows.index.to_numpy(dtype=np.float64),
                    rows["mean"].to_numpy(dtype=np.float64),
                    target,
                )
                match = f"> {full}" if needed is None else f"{needed:.0f}"
            name = APPROACH_TO_DISPLAY_NAME.get(approach, approach)
            lines.append(f"| {name} | " + " | ".join(cells + [match]) + " |")
        sections.append("\n".join(lines))
    footer = "± is one standard error over seeds. Budgets between measured ones are"
    footer += " interpolated linearly in their logarithm."
    return "\n\n".join(sections) + "\n\n" + footer + "\n"


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
    # A seed with no values in the window, such as planner work for tasks that
    # failed before any work, is left out.
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
    rate (when the runs record timeouts), the planner's work (when the runs record
    it), and sleep time. Planning time counts timed-out tasks at the time they
    took. ± is one standard error over seeds."""
    sections = []
    for env in sorted(eval_df["env"].unique()):
        env_df = eval_df[eval_df["env"] == env]
        approaches = sorted(env_df["approach"].unique())
        scale, unit = _get_time_scale(env_df["planning_wall_time"])
        env_df = env_df.assign(planning_time=env_df["planning_wall_time"] * scale)
        names = [APPROACH_TO_DISPLAY_NAME.get(a, a) for a in approaches]
        weighted = _is_weighted(env_df)
        lines = [
            f"### {ENV_TO_DISPLAY_NAME.get(env, env)}"
            + (" (schedule-weighted)" if weighted else ""),
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
        work = (
            (ENV_TO_PLANNER_WORK[env][0], env_df["planner_work"])
            if weighted and _has_values(env_df, "planner_work")
            else _get_planner_work(env, env_df)
        )
        if work is not None:
            work_label, work_values = work
            env_df = env_df.assign(planner_work=work_values)
            rows += [
                (f"{work_label}, {name}", "planner_work", first, last)
                for name, first, last in TABLE_WINDOWS
                if name in ("cycle 0", "last 5 cycles")
            ]
        if weighted:
            rows += [
                (f"{SLOW_DAY_METRIC[1]}, {name}", "slow_day", first, last)
                for name, first, last in TABLE_WINDOWS
                if name != "cycles 1–4"
            ]
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


def _get_planner_work(env: str, df: pd.DataFrame) -> tuple[str, pd.Series] | None:
    """The environment's planner work per evaluation task and its label, or None
    if the runs do not record it. Tasks whose planning failed before any work
    count as missing."""
    if env not in ENV_TO_PLANNER_WORK:
        return None
    label, numerator, denominator = ENV_TO_PLANNER_WORK[env]
    if numerator not in df.columns or (
        denominator is not None and denominator not in df.columns
    ):
        return None
    values = df[numerator].to_numpy(dtype=np.float64)
    if denominator is not None:
        counts = df[denominator].to_numpy(dtype=np.float64)
        values = np.divide(
            values, counts, out=np.full_like(values, np.nan), where=counts > 0
        )
    return label, pd.Series(values, index=df.index)


def _print_final_cycle(df: pd.DataFrame) -> None:
    """Print the final cycle's results, so they are visible without the plot.

    Planning time counts timed-out tasks at the time they took.
    """
    final = df[df["cycle"] == df["cycle"].max()]
    for (env, approach), group in final.groupby(["env", "approach"]):
        scale, unit = _get_time_scale(df[df["env"] == env]["planning_wall_time"])
        time = scale * np.mean(group["planning_wall_time"])
        success = np.mean(group["success"])
        line = (
            f"{env}, {approach}, cycle {group['cycle'].iloc[0]}: "
            f"planning time {time:.2f} {unit}, success rate {success:.2f}"
        )
        if _is_weighted(group):
            line += f", slow-day chance {np.mean(group['slow_day']):.2f}"
            line += " (schedule-weighted)"
            print(line)
            continue
        num_timed_out = int(group["timed_out"].sum()) if _has_timeouts(group) else 0
        if num_timed_out:
            line += f", {num_timed_out} timed out"
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
    parser.add_argument(
        "--slow_work",
        type=float,
        default=None,
        help="Planner work per task (friction_sites: simulations) above which a "
        "task makes its site's day slow. Defaults to the median of scratch's "
        "tasks, or to none, so only failures do, without scratch runs.",
    )
    args = parser.parse_args()
    _main(
        args.results_dir,
        args.outfile or args.results_dir / "cycle_results.png",
        args.slow_work,
    )
