"""Tests for experiments/analysis/plot_landscape.py."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"


def _write_results(out_dir: Path, num_contexts: int = 6) -> None:
    """Results in which each run lowers the cost of nearby contexts the most."""
    (out_dir / "evals").mkdir(parents=True)
    OmegaConf.save(
        {
            "env_name": "two_link_arm_sampled",
            "landscape": {"depths": [1, 4], "cross_depths": [4]},
        },
        out_dir / "landscape_config.yaml",
    )
    positions = np.linspace(-1.0, 1.0, num_contexts)
    pd.DataFrame(
        {
            "index": range(num_contexts),
            "trajectory": range(num_contexts),
            "cycle": -2,
            "train": True,
            "feature_0": positions,
            "feature_1": positions**2,
        }
    ).to_csv(out_dir / "contexts.csv", index=False)

    def rows(train: int, depth: int, contexts: list[int]) -> pd.DataFrame:
        records = []
        for context in contexts:
            near = 0.0 if train < 0 else np.exp(-abs(train - context)) * depth / 4
            for task in range(2):
                records.append(
                    {
                        "eval_context": context,
                        "task": task,
                        "success": True,
                        "stat_num_samples": 640.0 * (1.0 - 0.8 * near),
                        "train_context": train,
                        "replicate": 0,
                        "depth": depth,
                    }
                )
        return pd.DataFrame(records)

    rows(-1, 0, list(range(num_contexts))).to_csv(out_dir / "evals" / "base_c0.csv")
    for train in range(num_contexts):
        run = pd.concat(
            [rows(train, 1, [train]), rows(train, 4, list(range(num_contexts)))]
        )
        run.to_csv(out_dir / "evals" / f"context{train}.csv", index=False)


def test_plot_landscape(tmp_path: Path) -> None:
    """Tests that every figure is written for each order."""
    _write_results(tmp_path)
    for order in ("transfer", "features", "cycle"):
        result = subprocess.run(
            [
                sys.executable,
                str(ANALYSIS_DIR / "plot_landscape.py"),
                "--results_dir",
                str(tmp_path),
                "--order",
                order,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    names = ["learning_grid", "curves", "transfer_4", "kernel", "embedding"]
    for name in names:
        assert (tmp_path / f"{name}.png").exists()


def test_transfer_order_groups_neighbors(tmp_path: Path) -> None:
    """Tests that the transfer order nearly follows the feature that sets which
    runs help each context."""
    _write_results(tmp_path, num_contexts=8)
    sys.path.insert(0, str(ANALYSIS_DIR))
    try:
        import plot_landscape  # pylint: disable=import-outside-toplevel
    finally:
        sys.path.pop(0)
    landscape = plot_landscape.load_landscape(tmp_path)
    order = plot_landscape.get_order(landscape, "transfer")
    ranks = order.rank().to_numpy()
    assert np.corrcoef(ranks, np.arange(8))[0, 1] > 0.9
