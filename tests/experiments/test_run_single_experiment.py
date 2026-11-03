"""Tests for experiments/run_single_experiment.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from able.envs.base import FiniteTaskDistribution, TaskDistribution
from able.structs import Task

SCRIPT = (
    Path(__file__).resolve().parents[2] / "experiments" / "run_single_experiment.py"
)


class _UniformGoalDistribution(TaskDistribution[Task[float, float]]):
    """An infinite distribution over goals in [0, 1]."""

    def sample(self, rng: np.random.Generator) -> Task[float, float]:
        return Task(initial_state=0.0, goal=float(rng.uniform()))


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_single_experiment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_eval_tasks_finite() -> None:
    """Tests that a finite distribution evaluates every task, in order."""
    tasks = [Task(initial_state=0.0, goal=float(goal)) for goal in range(3)]
    eval_tasks = _load_script().get_eval_tasks(
        FiniteTaskDistribution(tasks), num_eval_tasks=6, seed=0
    )
    assert eval_tasks == tasks


def test_get_eval_tasks_infinite() -> None:
    """Tests that an infinite distribution gets a fixed sample of tasks per seed,
    drawn from its own random stream."""
    get_eval_tasks = _load_script().get_eval_tasks
    distribution = _UniformGoalDistribution()
    goals = [t.goal for t in get_eval_tasks(distribution, num_eval_tasks=6, seed=0)]
    assert len(goals) == 6
    assert len(set(goals)) == 6
    assert goals == [
        t.goal for t in get_eval_tasks(distribution, num_eval_tasks=6, seed=0)
    ]
    assert goals != [
        t.goal for t in get_eval_tasks(distribution, num_eval_tasks=6, seed=1)
    ]
    # The cycles draw from np.random.default_rng(seed), which is a different stream.
    assert goals != list(np.random.default_rng(0).uniform(size=6))


def test_run_single_experiment(tmp_path: Path) -> None:
    """Tests that each approach runs for two short cycles and writes its results."""
    for approach in ("scratch", "prior_only", "current_belief", "dreaming"):
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


def test_record_videos(tmp_path: Path) -> None:
    """Tests that recording writes a video for each stage with something to show."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "approach=dreaming",
            "num_cycles=1",
            "budgets.sleep.max_iterations=1",
            "env.domain.num_goals=2",
            "record_videos=true",
            f"hydra.run.dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # There are no queries, so there is no video of information gathering.
    assert sorted(p.name for p in (tmp_path / "videos").iterdir()) == [
        "cycle_000_1_sleep.mp4",
        "cycle_000_3_execution.mp4",
        "cycle_000_4_evaluation.mp4",
    ]


def test_limb(tmp_path: Path) -> None:
    """Tests one short cycle in the limb environment, with videos."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "env=limb",
            "approach=limb_scratch",
            "num_cycles=1",
            "num_eval_tasks=1",
            "env.domain.horizon=10",
            "approach.num_rollouts=4",
            "approach.num_particles=4",
            "budgets.information_gathering.max_iterations=1",
            "record_videos=true",
            f"hydra.run.dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    eval_df = pd.read_csv(tmp_path / "eval_results.csv")
    assert list(eval_df.task_id) == [0]
    cycle_df = pd.read_csv(tmp_path / "cycle_results.csv")
    assert list(cycle_df.num_queries) == [1]
    assert list(cycle_df.num_executed_tasks) == [1]
    assert sorted(p.name for p in (tmp_path / "videos").iterdir()) == [
        "cycle_000_2_queries.mp4",
        "cycle_000_3_execution.mp4",
        "cycle_000_4_evaluation.mp4",
    ]
