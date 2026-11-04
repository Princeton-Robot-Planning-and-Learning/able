"""Tests for experiments/analysis/plot_cycle_results.py and analysis_utils.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"


def _load_analysis_utils() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analysis_utils", ANALYSIS_DIR / "analysis_utils.py"
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
) -> None:
    """Write a run with two cycles of two evaluation tasks each."""
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        {"env_name": env, "approach_name": approach, "seed": seed},
        run_dir / "config.yaml",
    )
    pd.DataFrame({"cycle": [0, 1], "sleep_wall_time": [1.0, 3.0]}).to_csv(
        run_dir / "cycle_results.csv", index=False
    )
    pd.DataFrame(
        {
            "cycle": [0, 0, 1, 1],
            "task_id": [0, 1, 0, 1],
            "success": [True, False, True, True],
            "planning_wall_time": times,
        }
    ).to_csv(run_dir / "eval_results.csv", index=False)


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
