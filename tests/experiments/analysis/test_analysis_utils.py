"""Tests for experiments/analysis/analysis_utils.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"


def test_combine_results_csvs_config_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that each requested config key becomes a column."""
    monkeypatch.syspath_prepend(str(ANALYSIS_DIR))
    monkeypatch.delitem(sys.modules, "analysis_utils", raising=False)
    # pylint: disable-next=import-outside-toplevel
    from analysis_utils import combine_results_csvs

    for i, edge_probability in enumerate([0.0, 1.0]):
        run_dir = tmp_path / str(i)
        run_dir.mkdir()
        OmegaConf.save(
            {
                "env_name": "two_link_arm",
                "approach_name": "dreaming",
                "seed": 0,
                "env": {"domain": {"edge_probability": edge_probability}},
            },
            run_dir / "config.yaml",
        )
        pd.DataFrame({"cycle": [0, 1]}).to_csv(run_dir / "results.csv", index=False)
    df = combine_results_csvs(
        tmp_path, "results.csv", ["env.domain.edge_probability", "missing.key"]
    )
    assert sorted(df["env.domain.edge_probability"]) == [0.0, 0.0, 1.0, 1.0]
    assert df["missing.key"].isna().all()
