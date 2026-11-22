"""Tests for experiments/analysis/plot_cycle_results.py and analysis_utils.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from able.envs.friction.sites import make_site_domain
from able.envs.multi_site import SiteRoutineTaskDistribution

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"


def _load_analysis_utils() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analysis_utils", ANALYSIS_DIR / "analysis_utils.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plot_cycle_results(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The script imports analysis_utils from its own directory.
    monkeypatch.syspath_prepend(str(ANALYSIS_DIR))
    spec = importlib.util.spec_from_file_location(
        "plot_cycle_results", ANALYSIS_DIR / "plot_cycle_results.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(
    run_dir: Path,
    approach: str,
    seed: int,
    times: list[float],
    env: str = "two_link_arm",
    timed_out: list[bool] | None = None,
    extra: dict[str, list[float]] | None = None,
) -> None:
    """Write a run with two cycles of two evaluation tasks each, with a timed_out
    column if given (runs from before timeouts were recorded have none)."""
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        {"env_name": env, "approach_name": approach, "seed": seed},
        run_dir / "config.yaml",
    )
    pd.DataFrame({"cycle": [0, 1], "sleep_wall_time": [1.0, 3.0]}).to_csv(
        run_dir / "cycle_results.csv", index=False
    )
    eval_df = pd.DataFrame(
        {
            "cycle": [0, 0, 1, 1],
            "task_id": [0, 1, 0, 1],
            "success": [True, False, True, True],
            "planning_wall_time": times,
        }
    )
    if timed_out is not None:
        eval_df["timed_out"] = timed_out
    for column, values in (extra or {}).items():
        eval_df[column] = values
    eval_df.to_csv(run_dir / "eval_results.csv", index=False)


def test_plot_timeouts(tmp_path: Path) -> None:
    """Tests that timed-out tasks count in planning time at the time they took and
    are counted separately, including when some runs predate the timed_out
    column."""
    # The second task of cycle 0 failed by running out of its 600 s.
    _write_run(
        tmp_path / "0",
        "scratch",
        0,
        [100.0, 600.0, 200.0, 300.0],
        "human_reach",
        timed_out=[False, True, False, False],
    )
    # The first task of cycle 1 also ran out of its 600 s.
    _write_run(
        tmp_path / "2",
        "belief_coverage",
        0,
        [100.0, 200.0, 600.0, 200.0],
        "human_reach",
        timed_out=[False, False, True, False],
    )
    _write_run(tmp_path / "1", "dreaming", 0, [50.0, 60.0, 70.0, 80.0], "human_reach")
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_DIR / "plot_cycle_results.py"),
            "--results_dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cycle_results.png").stat().st_size > 0
    table = (tmp_path / "summary_table.md").read_text(encoding="utf-8")
    # Columns: Belief coverage, Dreaming, Plan from scratch.
    row = (
        "| Planning time (s), cycle 0 | 150.00 ± 0.00 | 55.00 ± 0.00 | 350.00 ± 0.00 |"
    )
    assert row in table
    assert (
        "| Success rate, all cycles | 0.75 ± 0.00 | 0.75 ± 0.00 | 0.75 ± 0.00 |"
        in table
    )
    assert (
        "| Timeout rate, all cycles | 0.25 ± 0.00 | 0.00 ± 0.00 | 0.25 ± 0.00 |"
        in table
    )
    assert "human_reach, scratch, cycle 1: planning time 250.00 s" in result.stdout
    final = next(
        line for line in result.stdout.splitlines() if "belief_coverage" in line
    )
    assert "planning time 400.00 s" in final and final.endswith("1 timed out")


def test_plot_cycle_results(tmp_path: Path) -> None:
    """Tests combining and summarizing results, and plotting them."""
    _write_run(tmp_path / "0", "scratch", 0, [0.1, 0.3, 0.2, 0.2])
    _write_run(tmp_path / "1", "scratch", 1, [0.3, 0.5, 0.1, 0.1])
    _write_run(tmp_path / "2", "dreaming", 0, [0.1, 0.1, 0.0, 0.0])
    _write_run(tmp_path / "3", "scratch", 0, [100.0, 200.0, 50.0, 50.0], "limb")
    # Hydra's own copy of the config has no results next to it and is skipped.
    (tmp_path / "0" / ".hydra").mkdir()
    OmegaConf.save({"seed": 0}, tmp_path / "0" / ".hydra" / "config.yaml")

    utils = _load_analysis_utils()
    df = utils.combine_results_csvs(tmp_path, "eval_results.csv")
    assert len(df) == 16
    assert sorted(df["approach"].unique()) == ["dreaming", "scratch"]

    arm = df[df["env"] == "two_link_arm"]
    summary = utils.summarize_over_seeds(arm, "planning_wall_time")
    scratch = summary[summary["approach"] == "scratch"].set_index("cycle")
    # Seed means in cycle 0 are 0.2 and 0.4.
    assert np.isclose(scratch.loc[0, "mean"], 0.3)
    assert np.isclose(scratch.loc[0, "sem"], np.std([0.2, 0.4], ddof=1) / np.sqrt(2))
    assert scratch.loc[0, "num_seeds"] == 2
    success = utils.summarize_over_seeds(arm, "success")
    dreaming = success[success["approach"] == "dreaming"].set_index("cycle")
    assert np.isclose(dreaming.loc[0, "mean"], 0.5)
    assert dreaming.loc[0, "sem"] == 0.0

    outfile = tmp_path / "plot.png"
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_DIR / "plot_cycle_results.py"),
            "--results_dir",
            str(tmp_path),
            "--outfile",
            str(outfile),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert outfile.stat().st_size > 0
    assert "two_link_arm, scratch, cycle 1: planning time 150.00 ms" in result.stdout
    # Slow planners are reported in seconds.
    assert "limb, scratch, cycle 1: planning time 50.00 s" in result.stdout
    table = (tmp_path / "summary_table.md").read_text(encoding="utf-8")
    assert "### Limb repositioning" in table
    assert "| Planning time (s), cycle 0 | 150.00 ± 0.00 |" in table
    assert "| Sleep time (s), all cycles | 2.00 ± 0.00 | 2.00 ± 0.00 |" in table


def test_planner_work_in_the_table(tmp_path: Path) -> None:
    """Tests that the table reports the planner's work per task, here friction's
    simulations per push, when the runs record it."""
    # Cycle 0's tasks ran 1000 simulations over 4 pushes and 600 over 2.
    _write_run(
        tmp_path / "0",
        "scratch",
        0,
        [40.0, 20.0, 30.0, 30.0],
        "friction",
        extra={"num_sims": [1000, 600, 300, 300], "num_searches": [4, 2, 3, 3]},
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_DIR / "plot_cycle_results.py"),
            "--results_dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "summary_table.md").read_text(encoding="utf-8")
    assert "| Simulations per push, cycle 0 | 275.00 ± 0.00 |" in table
    assert "| Simulations per push, last 5 cycles | 187.50 ± 0.00 |" in table


def test_limb_plots_rollouts(tmp_path: Path) -> None:
    """Tests that a limb row gets a panel of MPC rollouts per task when the runs
    record them, and that other environments' rows do not."""
    widths = {}
    for name, extra in (("without", None), ("with", {"num_rollouts": [1.0] * 4})):
        results_dir = tmp_path / name
        _write_run(
            results_dir / "0", "scratch", 0, [1.0, 2.0, 3.0, 4.0], "limb", extra=extra
        )
        _write_run(results_dir / "1", "scratch", 0, [1.0, 2.0, 3.0, 4.0], "friction")
        result = subprocess.run(
            [
                sys.executable,
                str(ANALYSIS_DIR / "plot_cycle_results.py"),
                "--results_dir",
                str(results_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        widths[name] = plt.imread(results_dir / "cycle_results.png").shape[1]
    # Two panels per row without the rollouts, three with them.
    assert widths["with"] > 1.3 * widths["without"]


def _write_budget_run(
    run_dir: Path, approach: str, seed: int, num_sims: int, success: float
) -> None:
    """A friction run with the simulation budget whose 4 tasks over 2 cycles
    succeed at the given rate."""
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        {
            "env_name": "friction",
            "approach_name": approach,
            "seed": seed,
            "approach": {"num_sims": num_sims},
        },
        run_dir / "config.yaml",
    )
    pd.DataFrame({"cycle": [0, 1], "sleep_wall_time": [1.0, 1.0]}).to_csv(
        run_dir / "cycle_results.csv", index=False
    )
    num_successes = round(4 * success)
    pd.DataFrame(
        {
            "cycle": [0, 0, 1, 1],
            "task_id": [0, 1, 0, 1],
            "success": [i < num_successes for i in range(4)],
            "planning_wall_time": [float(num_sims)] * 4,
        }
    ).to_csv(run_dir / "eval_results.csv", index=False)


def test_success_against_the_simulation_budget(tmp_path: Path) -> None:
    """Tests that a sweep over simulation budgets plots against cycles at the largest
    budget, plots success against the budget, and reports the budget each approach
    needs to match scratch at the largest one."""
    runs = [
        ("scratch", 100, 0.25),
        ("scratch", 1000, 0.5),
        ("dreaming", 100, 0.25),
        ("dreaming", 1000, 0.75),
        ("prior_only", 100, 0.25),
        ("prior_only", 1000, 0.25),
    ]
    for i, (approach, num_sims, success) in enumerate(runs):
        _write_budget_run(tmp_path / str(i), approach, 0, num_sims, success)
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_DIR / "plot_cycle_results.py"),
            "--results_dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "budget_results.png").stat().st_size > 0
    # Against cycles, only the 1000-simulation runs: planning time 1000 s.
    assert "friction, scratch, cycle 1: planning time 1000.00 s" in result.stdout
    table = (tmp_path / "summary_table.md").read_text(encoding="utf-8")
    assert "| Planning time (s), cycle 0 | 1000.00 ± 0.00 |" in table
    # Dreaming reaches scratch's 0.5 halfway between 100 and 1000 in log budget,
    # at 316; prior only never does.
    assert "| Dreaming | 0.25 ± 0.00 | 0.75 ± 0.00 | 316 |" in table
    assert "| Prior only (first sleep) | 0.25 ± 0.00 | 0.25 ± 0.00 | > 1000 |" in table
    assert "| Plan from scratch | 0.25 ± 0.00 | 0.50 ± 0.00 | reference |" in table


def test_weigh_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that each metric is averaged per site and then weighted by the sites'
    shares, and that a slow day's site has a failure or a task above the planner
    work threshold, with shares normalized over the sites evaluated."""
    plot = _load_plot_cycle_results(monkeypatch)
    # Cycle 0 evaluates sites 0 and 1 with two tasks each; cycle 1 only site 1,
    # whose second task failed before any planner work.
    df = pd.DataFrame(
        {
            "env": "friction_sites",
            "approach": "scratch",
            "seed": 0,
            "cycle": [0, 0, 0, 0, 1, 1],
            "site": [0, 0, 1, 1, 1, 1],
            "schedule_share": [0.2, 0.2, 0.8, 0.8, 0.8, 0.8],
            "success": [True, False, True, True, True, False],
            "planner_work": [100.0, 300.0, 50.0, 150.0, 80.0, np.nan],
        }
    )
    assert plot.has_sites(df) and not plot.has_sites(df.drop(columns="site"))
    for threshold, slow_day in ((None, 0.2), (120.0, 1.0), (200.0, 0.2)):
        weighted = plot.weigh_sites(df, ["success", "planner_work"], threshold)
        assert list(weighted.cycle) == [0, 1]
        assert np.allclose(weighted.success, [0.2 * 0.5 + 0.8 * 1.0, 0.5])
        assert np.allclose(weighted.planner_work, [0.2 * 200 + 0.8 * 100, 80.0])
        assert np.allclose(weighted.slow_day, [slow_day, 1.0])


def test_get_slow_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that a given threshold wins, then scratch's median, then none."""
    plot = _load_plot_cycle_results(monkeypatch)
    df = pd.DataFrame(
        {
            "approach": ["scratch", "scratch", "scratch", "dreaming"],
            "planner_work": [100.0, 400.0, 200.0, 1000.0],
        }
    )
    assert plot.get_slow_work(df, 50.0) == (50.0, "given")
    assert plot.get_slow_work(df, None) == (200.0, "scratch's median")
    assert plot.get_slow_work(df[df.approach == "dreaming"], None)[0] is None


def test_add_missing_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that friction site runs from before the eval results recorded sites
    get them from the run's domain, and that other runs are left alone."""
    plot = _load_plot_cycle_results(monkeypatch)
    domain_cfg = {
        "_target_": "able.envs.friction.sites.make_site_domain",
        "group_sizes": [1, 2],
        "routine_size": 2,
        "num_particles": 10,
    }
    cfg = OmegaConf.create(
        {"env_name": "friction_sites", "seed": 3, "env": {"domain": domain_cfg}}
    )
    df = pd.DataFrame({"cycle": 0, "task_id": range(6)})
    added = plot.add_missing_sites(df, cfg)
    assert list(added.site) == [0, 0, 1, 1, 2, 2]
    domain = make_site_domain(
        seed=3, group_sizes=(1, 2), routine_size=2, num_particles=10
    )
    routines = domain.task_distribution
    assert isinstance(routines, SiteRoutineTaskDistribution)
    schedule = routines.schedule  # pylint: disable=no-member
    assert np.allclose(added.schedule_share, schedule[[0, 0, 1, 1, 2, 2]])
    assert plot.add_missing_sites(added.assign(site=5), cfg).site.eq(5).all()
    cfg.env_name = "friction"
    assert "site" not in plot.add_missing_sites(df, cfg).columns


def test_friction_sites_metrics(tmp_path: Path) -> None:
    """Tests that friction site runs are plotted and tabulated weighted by the
    schedule, with the chance of a slow day at scratch's median simulations."""
    sites: dict[str, list[float]] = {
        "site": [0, 1, 0, 1],
        "schedule_share": [0.2, 0.8, 0.2, 0.8],
    }
    runs: list[tuple[str, list[float]]] = [
        ("scratch", [100, 200, 100, 300]),
        ("transfer_practice", [100, 100, 50, 120]),
    ]
    for i, (approach, num_sims) in enumerate(runs):
        _write_run(
            tmp_path / str(i),
            approach,
            0,
            [1.0, 2.0, 1.0, 2.0],
            "friction_sites",
            timed_out=[False] * 4,
            extra={**sites, "num_sims": num_sims},
        )
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_DIR / "plot_cycle_results.py"),
            "--results_dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "above 150 simulations per task (scratch's median)" in result.stdout
    assert "slow-day chance 0.00 (schedule-weighted)" in result.stdout
    table = (tmp_path / "summary_table.md").read_text(encoding="utf-8")
    # Columns: Plan from scratch, Transfer practice.
    for row in (
        "### Friction sites (schedule-weighted)",
        "| Planning time (s), cycle 0 | 1.80 ± 0.00 | 1.80 ± 0.00 |",
        "| Success rate, all cycles | 0.60 ± 0.00 | 0.60 ± 0.00 |",
        "| Simulations per task, cycle 0 | 180.00 ± 0.00 | 100.00 ± 0.00 |",
        "| Simulations per task, last 5 cycles | 220.00 ± 0.00 | 103.00 ± 0.00 |",
        "| Chance of a slow day, cycle 0 | 0.80 ± 0.00 | 0.80 ± 0.00 |",
        "| Chance of a slow day, all cycles | 0.80 ± 0.00 | 0.40 ± 0.00 |",
    ):
        assert row in table
