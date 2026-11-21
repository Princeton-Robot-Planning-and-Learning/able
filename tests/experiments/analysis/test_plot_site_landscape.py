"""Tests for experiments/analysis/plot_site_landscape.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from able.envs.multi_site import SiteRoutineTaskDistribution
from able.structs import Task

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "analysis"
    / "plot_site_landscape.py"
)
# Probe sites 0 and 1 run group 0's routine, 4 group 3's, and 2 and 3 groups 1
# and 2, as with group_sizes [2, 1, 1, 1].
GROUPS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}


@pytest.fixture(name="script", scope="module")
def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_site_landscape", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saved(train_site: int, site: int) -> float:
    """Full saving at the trained site, half in its corridor, a quarter
    elsewhere."""
    if site == train_site:
        return 1.0
    same_corridor = GROUPS[site] % 3 == GROUPS[train_site] % 3
    return 0.5 if same_corridor else 0.25


def _write_probes(probe_dir: Path, legacy: bool = False) -> None:
    probe_dir.mkdir(parents=True)
    for train_site in (0,) if legacy else (0, 2, 3):
        for seed in range(2):
            rows = []
            for depth in (0, 1, 8):
                for site, group in GROUPS.items():
                    for task in range(2):
                        rows.append(
                            {
                                "seed": seed,
                                "overlap": 0.0,
                                "train_site": train_site,
                                "depth": depth,
                                "site": site,
                                "group": group,
                                "task": task,
                                "repeat": 0,
                                "success": True,
                                "num_sims": 200.0
                                - 100.0 * _saved(train_site, site) * depth / 8
                                + 10.0 * seed,
                                "planning_time": 1.0,
                            }
                        )
            df = pd.DataFrame(rows)
            if legacy:
                df = df.drop(columns="train_site")
            df.to_csv(probe_dir / f"train{train_site}_seed{seed}.csv", index=False)


def test_site_order_and_shares(script: ModuleType, tmp_path: Path) -> None:
    """Tests that sites are ordered by corridor and shares are relative to the
    trained site's own saving."""
    _write_probes(tmp_path / "probes")
    df = script.load_probes(tmp_path / "probes")
    assert script.get_site_order(df) == [0, 1, 4, 2, 3]
    assert script.get_trained_order(df) == [0, 2, 3]
    shares = script.get_shares(df, 8)
    assert shares.loc[2, 2] == pytest.approx(1.0)
    assert shares.loc[0, 4] == pytest.approx(0.5)
    assert shares.loc[3, 0] == pytest.approx(0.25)
    sims = script.get_mean_sims(df)
    assert sims.loc[(0, 0), 0] == pytest.approx(205.0)
    assert sims.loc[(0, 8), 0] == pytest.approx(105.0)


def test_corridor_table(script: ModuleType, tmp_path: Path) -> None:
    """Tests that the own corridor's cell is the trained site and other cells
    average the evaluated corridor's sites."""
    _write_probes(tmp_path / "probes")
    table = script.get_corridor_table(script.load_probes(tmp_path / "probes"), 8)
    assert len(table) == 9
    assert table.loc[(0, 0), "saved"] == pytest.approx(100.0)
    assert table.loc[(0, 0), "se"] == pytest.approx(0.0)
    assert table.loc[(1, 0), "share"] == pytest.approx(0.25)
    assert table.loc[(2, 1), "saved"] == pytest.approx(25.0)
    text = script.format_corridor_table(table)
    assert "| blue | 100 ± 0 (1.00) | 25 ± 0 (0.25) | 25 ± 0 (0.25) |" in text


def test_count_units(script: ModuleType) -> None:
    """Tests that units are counted per night and site, skipping empty nights
    and nights past the limit."""
    entries = pd.Series(["0;0;3", np.nan, "3", "1;2"])
    units = script.count_units(entries, 4, nights=3)
    assert units.tolist() == [[2, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]]
    assert script.count_units(pd.Series([2, 1]), 3).tolist() == [[0, 0, 1], [0, 1, 0]]


def test_plots(script: ModuleType, tmp_path: Path) -> None:
    """Tests that the figures are written with and without the overlay."""
    _write_probes(tmp_path / "probes")
    df = script.load_probes(tmp_path / "probes")
    script.plot_landscape(df, tmp_path / "plain.png")
    overlay = script.Overlay(
        units=np.array([[3, 1, 0, 2, 2, 0], [0, 4, 0, 0, 4, 0]]),
        groups=[0, 0, 1, 1, 2, 2],
        schedule=np.full(6, 1 / 6),
    )
    script.plot_landscape(df, tmp_path / "overlay.png", overlay)
    script.plot_transfer(df, [1, 8], tmp_path / "transfer.png")
    for name in ("plain", "overlay", "transfer"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_load_overlay_skips_without_column(script: ModuleType, tmp_path: Path) -> None:
    """Tests that a run without a training_sites column gives no overlay."""
    pd.DataFrame({"cycle": [0, 1]}).to_csv(tmp_path / "cycle_results.csv")
    assert script.load_overlay(tmp_path, nights=None) is None


def test_load_overlay(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that a run's units are counted and its schedule and groups are read
    from the domain its config makes for its seed."""
    OmegaConf.save({"seed": 1, "env": {"domain": {"x": 0}}}, tmp_path / "config.yaml")
    pd.DataFrame({"cycle": range(3), "training_sites": ["0;2;2", "1", ""]}).to_csv(
        tmp_path / "cycle_results.csv", index=False
    )
    tasks = [[Task(initial_state=g, goal=None)] for g in range(2)]
    routines = SiteRoutineTaskDistribution(tasks, [0, 1, 1], [0.5, 0.3, 0.2])

    def instantiate(cfg: DictConfig, seed: int) -> SimpleNamespace:
        assert cfg.x == 0 and seed == 1
        return SimpleNamespace(task_distribution=routines)

    monkeypatch.setattr(script.hydra.utils, "instantiate", instantiate)
    overlay = script.load_overlay(tmp_path, nights=2)
    assert overlay.units.tolist() == [[1, 0, 2], [0, 1, 0]]
    assert overlay.groups == [0, 1, 1]
    assert overlay.schedule.tolist() == pytest.approx([0.5, 0.3, 0.2])


def test_main_with_legacy_probes(tmp_path: Path) -> None:
    """Tests the script on probe CSVs from before the train_site column."""
    _write_probes(tmp_path / "probes", legacy=True)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--probe_dir",
            str(tmp_path / "probes"),
            "--out_dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "site_landscape.png").exists()
    assert (tmp_path / "out" / "site_transfer.png").exists()
