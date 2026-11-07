"""Run one approach in one environment for a number of cycles.

See the README for examples.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from able.cycle import CycleBudgets, CycleResult, run_cycles
from able.envs.base import Domain, FiniteTaskDistribution, TaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.recording.video import VideoRecorder
from able.robot import Robot

# Separates the random stream for sampling evaluation tasks from the cycles' stream.
_EVAL_TASK_STREAM = 1


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()
    logging.info(
        f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
    )
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logging.info(OmegaConf.to_yaml(resolved_cfg))
    OmegaConf.save(resolved_cfg, cfg.config_file)

    domain = hydra.utils.instantiate(cfg.env.domain, seed=cfg.seed)
    assert isinstance(domain, Domain)
    task_distribution = domain.task_distribution
    eval_tasks = get_eval_tasks(task_distribution, cfg.num_eval_tasks, cfg.seed)
    recorder = _make_video_recorder(cfg, domain) if cfg.record_videos else None
    # The domain and recorder are objects rather than configs, so they are passed
    # after instantiation.
    make_robot = hydra.utils.instantiate(cfg.approach, _partial_=True)
    robot = make_robot(domain=domain, recorder=recorder)
    assert isinstance(robot, Robot)
    budgets = hydra.utils.instantiate(cfg.budgets)
    assert isinstance(budgets, CycleBudgets)

    # The same evaluation tasks are used in every cycle.
    writer = CycleResultsWriter(
        cfg.num_cycles,
        Path(cfg.eval_results_file),
        Path(cfg.cycle_results_file),
        start_time,
    )
    run_cycles(
        robot,
        domain.real_env,
        task_distribution,
        eval_tasks,
        cfg.num_cycles,
        budgets,
        np.random.default_rng(cfg.seed),
        recorder,
        on_cycle_end=writer,
    )
    logging.info(
        f"Finished in {format_duration(time.perf_counter() - start_time)}. Wrote "
        f"{cfg.eval_results_file} and {cfg.cycle_results_file}"
    )


class CycleResultsWriter:
    """After each cycle, rewrites both results files with every cycle so far and
    logs the cycle's progress line.

    A run that dies therefore keeps its finished cycles, and a sweep can be
    plotted while it runs. Each file is written to a temporary file first and then
    renamed, so a run killed while writing never leaves a partial file. The start
    time is the run's, from time.perf_counter().
    """

    def __init__(
        self,
        num_cycles: int,
        eval_results_file: Path,
        cycle_results_file: Path,
        start_time: float,
    ) -> None:
        self._num_cycles = num_cycles
        self._eval_results_file = eval_results_file
        self._cycle_results_file = cycle_results_file
        self._start_time = start_time
        self._results: list[CycleResult] = []

    def __call__(self, result: CycleResult) -> None:
        self._results.append(result)
        _write_csv(pd.DataFrame(_get_eval_rows(self._results)), self._eval_results_file)
        _write_csv(
            pd.DataFrame(_get_cycle_rows(self._results)), self._cycle_results_file
        )
        elapsed = time.perf_counter() - self._start_time
        logging.info(
            format_progress(self._results, self._num_cycles, elapsed, datetime.now())
        )


def format_progress(
    results: Sequence[CycleResult], num_cycles: int, elapsed: float, now: datetime
) -> str:
    """One line on the latest cycle: its evaluation, the wall time of each stage,
    the run's elapsed time, and when the run should finish at the mean cycle time
    so far."""
    result = results[-1]
    parts = [f"Cycle {result.cycle} done ({len(results)}/{num_cycles})"]
    if result.evaluations:
        success_rate = np.mean([e.success for e in result.evaluations])
        planning_time = np.mean([e.planning_wall_time for e in result.evaluations])
        parts.append(
            f"success rate {success_rate:.2f}, "
            f"mean planning time {planning_time:.4g} s"
        )
    stages = [
        ("sleep", result.sleep_wall_time),
        ("gathering", result.gathering_wall_time),
        ("execution", result.execution_wall_time),
        ("evaluation", result.evaluation_wall_time),
        ("cycle", result.cycle_wall_time),
    ]
    parts.append(", ".join(f"{name} {format_duration(t)}" for name, t in stages))
    num_left = num_cycles - len(results)
    if num_left > 0:
        mean_cycle_time = np.mean([r.cycle_wall_time for r in results])
        left = float(mean_cycle_time * num_left)
        finish = (now + timedelta(seconds=left)).strftime("%Y-%m-%d %H:%M")
        parts.append(
            f"elapsed {format_duration(elapsed)}, "
            f"about {format_duration(left)} left (finishes around {finish})"
        )
    else:
        parts.append(f"elapsed {format_duration(elapsed)}")
    return " | ".join(parts)


def format_duration(seconds: float) -> str:
    """Seconds under a minute, such as 4.2s, and H:MM:SS above."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def get_eval_tasks(
    task_distribution: TaskDistribution, num_eval_tasks: int, seed: int
) -> list[Any]:
    """Every task of a finite distribution, or num_eval_tasks tasks sampled once
    for the seed.

    The tasks are sampled from their own random stream, so sampling them does not
    change the random streams of the cycles.
    """
    if isinstance(task_distribution, FiniteTaskDistribution):
        return list(task_distribution.get_tasks())
    rng = np.random.default_rng([seed, _EVAL_TASK_STREAM])
    return [task_distribution.sample(rng) for _ in range(num_eval_tasks)]


def _make_video_recorder(cfg: DictConfig, domain: Domain) -> VideoRecorder:
    real_env = domain.real_env
    assert isinstance(real_env, SimulatedEnvironment)
    make_renderer = hydra.utils.instantiate(cfg.env.renderer, _partial_=True)
    cycles = None if cfg.video_cycles is None else list(cfg.video_cycles)
    logging.info(f"Recording videos to {cfg.video_dir}")
    return VideoRecorder(
        make_renderer(domain.env_model),
        real_env.hidden_params,
        Path(cfg.video_dir),
        fps=cfg.video_fps,
        cycles=cycles,
    )


def _get_eval_rows(results: list[CycleResult]) -> list[dict[str, Any]]:
    """One row per evaluation task per cycle, including planner statistics."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for task_id, evaluation in enumerate(result.evaluations):
            rows.append(
                {
                    "cycle": result.cycle,
                    "task_id": task_id,
                    "success": evaluation.success,
                    "planning_wall_time": evaluation.planning_wall_time,
                    "total_cost": evaluation.total_cost,
                    "planning_failed": evaluation.failure is not None,
                    **evaluation.planning_stats,
                }
            )
    return rows


def _get_cycle_rows(results: list[CycleResult]) -> list[dict[str, Any]]:
    """One row per cycle, with the wall time of each stage and of the whole cycle,
    and what the robot did between sleeping and evaluation."""
    return [
        {
            "cycle": result.cycle,
            "sleep_wall_time": result.sleep_wall_time,
            "gathering_wall_time": result.gathering_wall_time,
            "execution_wall_time": result.execution_wall_time,
            "evaluation_wall_time": result.evaluation_wall_time,
            "cycle_wall_time": result.cycle_wall_time,
            "num_queries": len(result.gathered_data),
            "num_executed_tasks": len(result.executions),
        }
        for result in results
    ]


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
