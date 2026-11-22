"""Tests for experiments/train_site_psi.py."""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "train_site_psi.py"


def test_trains_every_site_and_saves_snapshots(tmp_path: Path) -> None:
    """Tests that training every site saves the requested depths, the training
    data, and each site's corridor and belief."""
    out = tmp_path / "psi.pkl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train_site=-1",
            "--depths",
            "0",
            "1",
            "--out",
            str(out),
            "approach.model.num_steps=20",
            "approach.training_num_sims=20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with open(out, "rb") as f:
        snapshot = pickle.load(f)
    assert sorted(snapshot["snapshots"]) == [0, 1]
    assert snapshot["train_site"] == -1
    assert snapshot["corridors"] == [0, 1, 2, 3]
    assert len(snapshot["routines"]) == 4 and snapshot["beliefs"].num_sites == 4
    assert np.asarray(snapshot["truths"]).shape == (4, 12)
    assert len(snapshot["dataset"]) > 0
