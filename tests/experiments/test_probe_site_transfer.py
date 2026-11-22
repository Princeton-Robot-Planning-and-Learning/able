"""Tests for experiments/probe_site_transfer.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "probe_site_transfer.py"


def test_probe_with_train_site(tmp_path: Path) -> None:
    """Tests a tiny probe that trains for site 3."""
    out = tmp_path / "probe.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--depths",
            "1",
            "--train_site",
            "3",
            "--out",
            str(out),
            "approach.num_sims=5",
            "approach.training_num_sims=5",
            "approach.model.num_steps=5",
            "env.domain.routine_size=2",
            "env.domain.num_particles=10",
            "budgets.planning.max_iterations=5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out)
    assert (df["train_site"] == 3).all()
    assert sorted(df["depth"].unique()) == [0, 1]
    assert len(df) == 2 * 5 * 2
