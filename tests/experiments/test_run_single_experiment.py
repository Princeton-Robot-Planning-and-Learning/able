"""Tests for experiments/run_single_experiment.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

SCRIPT = (
    Path(__file__).resolve().parents[2] / "experiments" / "run_single_experiment.py"
)


def test_run_single_experiment(tmp_path: Path) -> None:
    """Tests that each approach runs for two short cycles and writes its results."""
    for approach in ("scratch", "dreaming"):
        run_dir = tmp_path / approach
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                f"approach={approach}",
                "num_cycles=2",
                "budgets.sleep.max_iterations=1",
                f"hydra.run.dir={run_dir}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        cfg = OmegaConf.load(run_dir / "config.yaml")
        assert cfg.approach_name == approach
        assert cfg.env_name == "two_link_arm"

        # Two cycles of six evaluation tasks each.
        eval_df = pd.read_csv(run_dir / "eval_results.csv")
        assert len(eval_df) == 12
        assert list(eval_df.cycle.unique()) == [0, 1]
        for column in ("success", "planning_wall_time", "total_cost", "num_samples"):
            assert column in eval_df.columns

        cycle_df = pd.read_csv(run_dir / "cycle_results.csv")
        assert list(cycle_df.cycle) == [0, 1]
        assert list(cycle_df.num_executed_tasks) == [1, 1]
        assert (cycle_df.sleep_wall_time >= 0).all()
