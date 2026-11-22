"""Tests for experiments/probe_site_transfer.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from able.envs.multi_site import SiteRoutineTaskDistribution
from able.structs import Task

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "probe_site_transfer.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("probe_site_transfer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_focus_routines() -> None:
    """Tests that focusing keeps every site's routine and puts the whole schedule
    on the chosen site."""
    script = _load_script()
    routines = [
        [Task(initial_state=(g, i), goal=None) for i in range(2)] for g in range(4)
    ]
    original = SiteRoutineTaskDistribution(routines, [0, 0, 1, 2, 3], np.full(5, 0.2))
    focused = script.focus_routines(original, 3)
    assert focused.schedule.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    for site in range(5):
        assert focused.get_group(site) == original.get_group(site)
        assert [t.inner for t in focused.get_routine(site)] == [
            t.inner for t in original.get_routine(site)
        ]
    rng = np.random.default_rng(0)
    assert {focused.sample(rng).site for _ in range(20)} == {3}


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
