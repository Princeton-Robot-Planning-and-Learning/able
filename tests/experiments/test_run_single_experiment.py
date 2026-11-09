"""Tests for experiments/run_single_experiment.py."""

import dataclasses
import importlib.util
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Callable

import hydra
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from able.approaches import make_scratch_robot
from able.cycle import CycleBudgets, CycleResult, run_cycles
from able.envs.base import Domain, FiniteTaskDistribution, TaskDistribution
from able.envs.two_link_arm import make_domain
from able.robot import Robot
from able.structs import Budget, Task

SCRIPT = (
    Path(__file__).resolve().parents[2] / "experiments" / "run_single_experiment.py"
)
STAGE_TIME_COLUMNS = (
    "sleep_wall_time",
    "gathering_wall_time",
    "execution_wall_time",
    "evaluation_wall_time",
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


APPROACHES = ("scratch", "prior_only", "current_belief", "dreaming", "belief_coverage")


@pytest.mark.parametrize("env", ["limb", "friction", "human_reach"])
def test_every_approach_config_builds_its_robot(env: str) -> None:
    """Tests that every approach's config for the environment builds its robot, as
    run_single_experiment.py does, without running any cycles. Short runs of each
    environment test the cycles."""
    with initialize_config_dir(
        config_dir=str(SCRIPT.parent / "conf"), version_base=None
    ):
        configs = [
            compose(config_name="config", overrides=[f"env={env}", f"approach={a}"])
            for a in APPROACHES
        ]
    domain = hydra.utils.instantiate(configs[0].env.domain, seed=0)
    assert isinstance(domain, Domain)
    for cfg in configs:
        make_robot = hydra.utils.instantiate(cfg.approach, _partial_=True)
        robot = make_robot(domain=domain, recorder=None)
        assert isinstance(robot, Robot)
        if env == "human_reach":
            params = robot.planner_params
            assert params.placement == "continuous" and params.delta == 0.02
    assert isinstance(hydra.utils.instantiate(configs[0].budgets), CycleBudgets)


def test_della_launcher() -> None:
    """Tests that the Della launcher config runs jobs through Slurm on one node type,
    with a CPU and its memory per worker process."""
    with initialize_config_dir(
        config_dir=str(SCRIPT.parent / "conf"), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=["hydra/launcher=della"],
            return_hydra_config=True,
        )
        workers_cfg = compose(
            config_name="config",
            overrides=["hydra/launcher=della", "num_workers=6"],
            return_hydra_config=True,
        )
    launcher = cfg.hydra.launcher
    assert launcher["_target_"].endswith("SlurmLauncher")
    assert launcher.constraint == "genoa"
    assert launcher.cpus_per_task == 1
    assert launcher.mem_per_cpu == "8G" and launcher.mem_gb is None
    assert workers_cfg.hydra.launcher.cpus_per_task == 6


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

        # Two cycles of 12 evaluation tasks each.
        eval_df = pd.read_csv(run_dir / "eval_results.csv")
        assert len(eval_df) == 24
        assert list(eval_df.cycle.unique()) == [0, 1]
        for column in (
            "success",
            "planning_wall_time",
            "total_cost",
            "num_samples",
            "failure",
        ):
            assert column in eval_df.columns
        # Random shooting has no time limit here.
        assert not eval_df.timed_out.any()

        cycle_df = pd.read_csv(run_dir / "cycle_results.csv")
        assert list(cycle_df.cycle) == [0, 1]
        assert list(cycle_df.num_executed_tasks) == [1, 1]
        for column in STAGE_TIME_COLUMNS:
            assert (cycle_df[column] >= 0).all()
        stage_total = cycle_df[list(STAGE_TIME_COLUMNS)].sum(axis=1)
        assert (cycle_df.cycle_wall_time >= stage_total).all()

        # One progress line per cycle, and no temporary files left behind.
        log = (run_dir / "run_single_experiment.log").read_text()
        assert "Cycle 0 done (1/2)" in log
        assert "Cycle 1 done (2/2)" in log
        assert not list(run_dir.glob("*.tmp"))


def _run_arm_cycles(
    num_cycles: int, on_cycle_end: Callable[[CycleResult], None] | None = None
) -> list[CycleResult]:
    """Run the scratch robot on a two-link arm with two goals."""
    domain = make_domain(seed=0, num_goals=2)
    task_distribution = domain.task_distribution
    eval_tasks = _load_script().get_eval_tasks(task_distribution, 2, seed=0)
    budgets = CycleBudgets(
        sleep=Budget(max_iterations=1),
        information_gathering=Budget(max_iterations=0),
        execution=Budget(max_iterations=1),
        planning=Budget(),
    )
    return run_cycles(
        make_scratch_robot(domain),
        domain.real_env,
        task_distribution,
        eval_tasks,
        num_cycles,
        budgets,
        np.random.default_rng(0),
        on_cycle_end=on_cycle_end,
    )


def test_writer_keeps_finished_cycles(tmp_path: Path) -> None:
    """Tests that a run that dies partway keeps the results of its finished
    cycles."""
    writer = _load_script().CycleResultsWriter(
        3,
        tmp_path / "eval_results.csv",
        tmp_path / "cycle_results.csv",
        time.perf_counter(),
    )

    def on_cycle_end(result: CycleResult) -> None:
        writer(result)
        if result.cycle == 1:
            raise RuntimeError("Hit the Slurm time limit")

    with pytest.raises(RuntimeError):
        _run_arm_cycles(3, on_cycle_end)
    assert list(pd.read_csv(tmp_path / "cycle_results.csv").cycle) == [0, 1]
    assert list(pd.read_csv(tmp_path / "eval_results.csv").cycle) == [0, 0, 1, 1]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "cycle_results.csv",
        "eval_results.csv",
    ]


def test_format_progress() -> None:
    """Tests the progress line, with a finish estimate until the last cycle."""
    format_progress = _load_script().format_progress
    results = [
        dataclasses.replace(
            result,
            sleep_wall_time=962.26,
            gathering_wall_time=0.0,
            execution_wall_time=210.4,
            evaluation_wall_time=450.0,
            cycle_wall_time=1623.0,
            evaluations=[
                dataclasses.replace(
                    result.evaluations[0], success=True, planning_wall_time=200.0
                ),
                dataclasses.replace(
                    result.evaluations[1], success=False, planning_wall_time=100.0
                ),
            ],
        )
        for result in _run_arm_cycles(2)
    ]
    now = datetime(2026, 11, 4, 20, 0)
    stages = (
        "success rate 0.50, mean planning time 150 s | sleep 0:16:02, "
        "gathering 0.0s, execution 0:03:30, evaluation 0:07:30, cycle 0:27:03"
    )
    # Nineteen cycles of 1623 s are left.
    assert format_progress(results[:1], 20, 1700.0, now) == (
        f"Cycle 0 done (1/20) | {stages} | elapsed 0:28:20, "
        "about 8:33:57 left (finishes around 2026-11-05 04:33)"
    )
    assert format_progress(results, 2, 3400.0, now) == (
        f"Cycle 1 done (2/2) | {stages} | elapsed 0:56:40"
    )
    # A timed-out task is left out of the mean planning time and counted.
    last = results[-1]
    timed_out = dataclasses.replace(last.evaluations[1], timed_out=True)
    last = dataclasses.replace(last, evaluations=[last.evaluations[0], timed_out])
    assert format_progress([last], 1, 60.0, now).startswith(
        "Cycle 1 done (1/1) | success rate 0.50, mean planning time 200 s, "
        "1 timed out | "
    )


def test_format_duration() -> None:
    """Tests seconds under a minute and hours, minutes, and seconds above."""
    format_duration = _load_script().format_duration
    assert format_duration(4.24) == "4.2s"
    assert format_duration(3725.4) == "1:02:05"
    assert format_duration(90000) == "25:00:00"


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
            "approach=scratch",
            "num_cycles=1",
            "num_eval_tasks=1",
            "env.domain.horizon=1",
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


def test_friction(tmp_path: Path) -> None:
    """Tests one short cycle in the friction environment, with videos."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "env=friction",
            "approach=scratch",
            "num_cycles=1",
            "num_eval_tasks=1",
            "env.domain.horizon=2",
            "env.domain.num_particles=20",
            "approach.num_sims=20",
            "approach.max_depth=2",
            "budgets.information_gathering.max_iterations=1",
            "record_videos=true",
            f"hydra.run.dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(pd.read_csv(tmp_path / "eval_results.csv").task_id) == [0]
    cycle_df = pd.read_csv(tmp_path / "cycle_results.csv")
    assert list(cycle_df.num_queries) == [1]
    assert list(cycle_df.num_executed_tasks) == [1]
    assert sorted(p.name for p in (tmp_path / "videos").iterdir()) == [
        "cycle_000_2_queries.mp4",
        "cycle_000_3_execution.mp4",
        "cycle_000_4_evaluation.mp4",
    ]


def test_worker_processes(tmp_path: Path) -> None:
    """Tests runs with worker processes: belief coverage in the friction
    environment, whose sleeps and evaluations run in the workers, and the arm with
    videos, which runs without them."""
    common = ["num_workers=2", "num_cycles=2", "budgets.sleep.max_iterations=2"]
    friction = [
        "env=friction",
        "approach=belief_coverage",
        "num_eval_tasks=3",
        "env.domain.horizon=2",
        "env.domain.num_particles=20",
        "approach.num_sims=20",
        "approach.max_depth=2",
        "approach.model.num_steps=5",
    ]
    arm = ["approach=dreaming", "env.domain.num_goals=2", "record_videos=true"]
    for name, overrides in (("friction", friction), ("arm", arm)):
        run_dir = tmp_path / name
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                *common,
                *overrides,
                f"hydra.run.dir={run_dir}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        eval_df = pd.read_csv(run_dir / "eval_results.csv")
        assert list(eval_df.cycle) == (
            [0, 0, 0, 1, 1, 1] if name == "friction" else [0, 0, 1, 1]
        )
        log = (run_dir / "run_single_experiment.log").read_text()
        if name == "friction":
            assert "Starting 2 worker processes" in log
        else:
            assert "running without worker processes" in log


def test_human_reach(tmp_path: Path) -> None:
    """Tests one short cycle in the human reach environment. The other environments'
    tests record videos; the human reach renderer has its own tests."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "env=human_reach",
            "approach=scratch",
            "num_cycles=1",
            "num_eval_tasks=1",
            "env.domain.num_requested=1",
            "env.domain.num_clutter=0",
            "env.domain.num_particles=50",
            # Every bedside pose is usable, so planning is quick.
            "approach.delta=1.0",
            "approach.max_time=60",
            "budgets.information_gathering.max_iterations=1",
            f"hydra.run.dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(pd.read_csv(tmp_path / "eval_results.csv").task_id) == [0]
    cycle_df = pd.read_csv(tmp_path / "cycle_results.csv")
    assert list(cycle_df.num_queries) == [1]
    assert list(cycle_df.num_executed_tasks) == [1]
