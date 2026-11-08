"""Tests for experiments/analysis/estimate_run_times.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"


def _write_run(run_dir: Path, approach: str, seed: int, cycle_times: list[float]):
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        {"env_name": "limb", "approach_name": approach, "seed": seed},
        run_dir / "config.yaml",
    )
    pd.DataFrame(
        {"cycle": range(len(cycle_times)), "cycle_wall_time": cycle_times}
    ).to_csv(run_dir / "cycle_results.csv", index=False)


def test_estimate_run_times(tmp_path: Path) -> None:
    """Tests that the slowest run's mean cycle time sets the estimate, and that a
    short run's limit stays above Della's one-hour test QOS."""
    _write_run(tmp_path / "0", "dreaming", 0, [1700.0, 1900.0])
    _write_run(tmp_path / "1", "dreaming", 1, [2000.0, 2200.0])
    _write_run(tmp_path / "2", "scratch", 0, [60.0, 60.0])
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_DIR / "estimate_run_times.py"),
            "--results_dir",
            str(tmp_path),
            "--num_cycles",
            "20",
            "--setup_seconds",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = {line.split()[1]: line.split() for line in result.stdout.splitlines()[1:]}
    # Dreaming: the slower seed's 2100 s per cycle, times 20 cycles, is 11.7 h, and
    # 1.3 times that is 910 minutes, within the short QOS.
    assert lines["dreaming"][2:] == ["1950", "2100", "11.7", "910", "True"]
    # Scratch: 20 minutes, raised to 61 minutes.
    assert lines["scratch"][2:] == ["60", "60", "0.3", "61", "True"]
