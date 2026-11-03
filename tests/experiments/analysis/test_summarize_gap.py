"""Tests for experiments/analysis/summarize_gap.py."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"
LEVER = "env.domain.edge_probability"
NUM_CYCLES = 6
# The belief changes in even cycles.
CHANGED = [True, False, True, False, True, False]


def _load_summarize_gap(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ANALYSIS_DIR))
    for name in ("analysis_utils", "plot_cycle_results", "summarize_gap"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    # pylint: disable-next=import-outside-toplevel
    import summarize_gap

    return summarize_gap


def _write_run(
    run_dir: Path, approach: str, seed: int, edge_probability: float
) -> None:
    """Write a run with two evaluation tasks per cycle.

    The current-belief robot takes 4 ms per task and hits the cache only when the belief
    is unchanged. The dreaming robot takes 2 ms per task and hits the cache after cycle
    0, and its planning fails on one task in cycle 0.
    """
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        {
            "env_name": "two_link_arm",
            "approach_name": approach,
            "seed": seed,
            "env": {"domain": {"edge_probability": edge_probability}},
        },
        run_dir / "config.yaml",
    )
    rows = []
    for cycle in range(NUM_CYCLES):
        for task_id in range(2):
            if approach == "dreaming":
                hit = np.nan if (cycle, task_id) == (0, 1) else float(cycle > 0)
                time = 0.002
            else:
                hit = float(not CHANGED[cycle])
                time = 0.004
            rows.append(
                {
                    "cycle": cycle,
                    "task_id": task_id,
                    "success": True,
                    "planning_wall_time": time,
                    "used_warm_start": hit,
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "eval_results.csv", index=False)
    pd.DataFrame({"cycle": range(NUM_CYCLES), "belief_changed": CHANGED}).to_csv(
        run_dir / "cycle_results.csv", index=False
    )


def _write_sweep(results_dir: Path, edge_probabilities=(0.0, 1.0)) -> None:
    i = 0
    for edge_probability in edge_probabilities:
        for approach in ("current_belief", "dreaming"):
            for seed in (0, 1):
                _write_run(results_dir / str(i), approach, seed, edge_probability)
                i += 1


def test_get_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests the gaps computed from hand-checkable results."""
    module = _load_summarize_gap(monkeypatch)
    _write_sweep(tmp_path)
    eval_df = module.combine_results_csvs(tmp_path, "eval_results.csv", [LEVER])
    cycle_df = module.combine_results_csvs(tmp_path, "cycle_results.csv", [LEVER])
    seed_metrics = module.get_seed_metrics(eval_df, cycle_df, LEVER)
    gaps = module.get_gaps(seed_metrics, "dreaming", "current_belief", LEVER)

    assert list(gaps["window"][:3]) == ["Cycles 1–5", "Cycles 3–5", "All cycles"]
    assert list(gaps[LEVER]) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    row = gaps[(gaps[LEVER] == 0.0) & (gaps["window"] == "All cycles")].iloc[0]
    assert np.isclose(row["time_saved_mean"], 0.002)
    assert row["time_saved_sem"] == 0.0
    assert row["time_saved_count"] == 2
    # The failed plan counts as a miss, so dreaming hits on 10 of 12 tasks and the
    # baseline on 6 of 12.
    assert np.isclose(row["hit_rate_gain_mean"], 10 / 12 - 6 / 12)
    # On the revision cycles 0, 2, and 4, dreaming hits on 4 of 6 tasks and the
    # baseline on none.
    assert np.isclose(row["revision_hit_rate_gain_mean"], 4 / 6)
    assert np.isclose(row["baseline_revision_rate_mean"], 0.5)
    assert row["approach_success_mean"] == 1.0
    early = gaps[(gaps[LEVER] == 0.0) & (gaps["window"] == "Cycles 1–5")].iloc[0]
    assert np.isclose(early["hit_rate_gain_mean"], 1.0 - 3 / 5)
    assert np.isclose(early["revision_hit_rate_gain_mean"], 1.0)


def test_summarize_gap_script(tmp_path: Path) -> None:
    """Tests that the script writes the table, and the plot when given a lever."""
    _write_sweep(tmp_path)
    script = str(ANALYSIS_DIR / "summarize_gap.py")
    result = subprocess.run(
        [sys.executable, script, "--results_dir", str(tmp_path), "--lever", LEVER],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "gap_summary.md").read_text()
    assert "| 0.0 | All cycles | 2.00 ± 0.00 | 0.33 ± 0.00 | 0.67 ± 0.00 |" in table
    assert (tmp_path / "gap_summary.png").stat().st_size > 0

    single_dir = tmp_path / "single"
    _write_sweep(single_dir, edge_probabilities=(0.0,))
    result = subprocess.run(
        [sys.executable, script, "--results_dir", str(single_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    table = (single_dir / "gap_summary.md").read_text()
    assert "| All cycles | 2.00 ± 0.00 | 0.33 ± 0.00 | 0.67 ± 0.00 |" in table
    assert not (single_dir / "gap_summary.png").exists()
