"""Tests for experiments/run_landscape.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from able.beliefs.interval import IntervalBelief
from able.beliefs.particles import ParticleBelief
from able.beliefs.range_of_motion import RangeOfMotionBelief

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "run_landscape.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_landscape", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(out_dir: Path, *overrides: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), f"out_dir={out_dir}", *overrides],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_make_point_belief() -> None:
    """Tests that point beliefs of each type are certain of the truth."""
    script = _load_script()
    truth = np.array([0.5, 0.7])
    interval = script.make_point_belief(IntervalBelief([[0, 1], [0, 1]]), truth)
    np.testing.assert_allclose(interval.intervals, [[0.5, 0.5], [0.7, 0.7]])
    prior = ParticleBelief(np.random.default_rng(0).uniform(size=(10, 2)), 0, 1)
    particles = script.make_point_belief(prior, truth)
    assert particles.particles.shape == (10, 2)
    np.testing.assert_allclose(particles.std, 0.0, atol=1e-9)
    motion = RangeOfMotionBelief.from_prior(0.9, 0.15, 0.7, 1.2, num_motions=2)
    point = script.make_point_belief(motion, truth)
    np.testing.assert_allclose(point.mean, truth)
    assert np.all(point.stddev < 0.01)
    with pytest.raises(TypeError):
        script.make_point_belief(object(), truth)


def test_known_landscape(tmp_path: Path) -> None:
    """Tests that a known landscape on the sampled arm saves every checkpoint and
    evaluation unit, that each run is evaluated on every context at its cross
    depth and on its own context otherwise, and that a rerun skips finished
    work."""
    out_dir = tmp_path / "known"
    overrides = (
        "env=two_link_arm_sampled",
        "landscape.num_truths=3",
        "landscape.num_tasks=2",
        "landscape.depths=[1,2]",
        "landscape.cross_depths=[2]",
        "landscape.contexts_per_unit=2",
        "approach.model.num_steps=20",
    )
    _run(out_dir, *overrides)
    contexts = pd.read_csv(out_dir / "contexts.csv")
    assert list(contexts["cycle"]) == [-2, -2, -2]
    assert np.all(contexts.filter(like="feature_4") == 0.0)
    models = sorted(p.name for p in (out_dir / "models").glob("*.pkl"))
    assert len(models) == 3 * 2 and "context2_r0_d2.pkl" in models
    units = sorted(p.stem for p in (out_dir / "evals").glob("*.csv"))
    assert units == sorted(
        ["base_c0", "base_c2"]
        + [f"diag_context{j}_r0" for j in range(3)]
        + [f"cross_context{j}_r0_d2_c{c}" for j in range(3) for c in (0, 2)]
    )
    rows = pd.concat(pd.read_csv(p) for p in (out_dir / "evals").glob("*.csv"))
    base = rows[rows["train_context"] == -1]
    assert sorted(set(base["eval_context"])) == [0, 1, 2]
    run = rows[rows["train_context"] == 0]
    assert set(run.loc[run["depth"] == 1, "eval_context"]) == {0}
    assert set(run.loc[run["depth"] == 2, "eval_context"]) == {0, 1, 2}
    # Depth 1 on its own context, depth 2 on its own context, and depth 2 on all
    # three contexts, with two tasks each.
    assert len(run) == 2 + 2 + 2 * 3
    modified = (out_dir / "evals" / "base_c0.csv").stat().st_mtime
    _run(out_dir, *overrides)
    assert (out_dir / "evals" / "base_c0.csv").stat().st_mtime == modified


def test_stages_in_turn(tmp_path: Path) -> None:
    """Tests that running the stages one at a time, with two shards, gives the
    same evaluations as running them together."""
    overrides = (
        "env=two_link_arm",
        "env.domain.num_goals=2",
        "landscape.num_truths=2",
        "landscape.depths=[1]",
        "landscape.cross_depths=[1]",
    )
    _run(tmp_path / "together", *overrides)
    _run(tmp_path / "apart", *overrides, "stage=contexts")
    for stage in ("train", "eval"):
        for shard in (0, 1):
            _run(
                tmp_path / "apart",
                *overrides,
                f"stage={stage}",
                f"shard={shard}",
                "num_shards=2",
            )
    for name in ("base_c0", "diag_context1_r0", "cross_context0_r0_d1_c0"):
        together = pd.read_csv(tmp_path / "together" / "evals" / f"{name}.csv")
        apart = pd.read_csv(tmp_path / "apart" / "evals" / f"{name}.csv")
        columns = ["eval_context", "task", "success", "stat_num_samples"]
        pd.testing.assert_frame_equal(together[columns], apart[columns])


def test_visited_landscape_with_a_plan_cache(tmp_path: Path) -> None:
    """Tests that visited contexts share one prior context under a fixed truth,
    and that with an exact-match plan cache a run leaves the cost of a context
    with a different cache key as the untrained model left it, because every
    model sees the same random draws."""
    out_dir = tmp_path / "visited"
    _run(
        out_dir,
        "env=two_link_arm",
        "env.domain.num_goals=2",
        "landscape.contexts=visited",
        "landscape.fixed_truth=true",
        "landscape.num_truths=2",
        "landscape.cycles=[-1,3]",
        "landscape.depths=[1]",
        "landscape.cross_depths=[1]",
    )
    contexts = pd.read_csv(out_dir / "contexts.csv")
    assert list(zip(contexts["trajectory"], contexts["cycle"])) == [
        (0, -1),
        (0, 3),
        (1, 3),
    ]
    beliefs = [c.belief for c in _load_script().load_contexts(out_dir)]
    assert beliefs[0].get_discretized_key(0.1) != beliefs[1].get_discretized_key(0.1)
    evals = out_dir / "evals"
    base = pd.read_csv(evals / "base_c0.csv").set_index(["eval_context", "task"])
    run = pd.read_csv(evals / "cross_context1_r0_d1_c0.csv")
    other = run[run["eval_context"] == 0].set_index(["eval_context", "task"])
    pd.testing.assert_series_equal(
        other["stat_num_samples"], base.loc[other.index, "stat_num_samples"]
    )
