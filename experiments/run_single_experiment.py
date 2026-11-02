"""Run one approach in one environment for a number of cycles.

See the README for examples.
"""

import logging
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from able.cycle import CycleBudgets, CycleResult, run_cycles
from able.envs.base import Domain, FiniteTaskDistribution
from able.robot import Robot


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(
        f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
    )
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logging.info(OmegaConf.to_yaml(resolved_cfg))
    OmegaConf.save(resolved_cfg, cfg.config_file)

    domain = hydra.utils.instantiate(cfg.env, seed=cfg.seed)
    assert isinstance(domain, Domain)
    task_distribution = domain.task_distribution
    assert isinstance(task_distribution, FiniteTaskDistribution)
    # The domain is an object rather than a config, so it is passed after
    # instantiation.
    make_robot = hydra.utils.instantiate(cfg.approach, _partial_=True)
    robot = make_robot(domain=domain)
    assert isinstance(robot, Robot)
    budgets = hydra.utils.instantiate(cfg.budgets)
    assert isinstance(budgets, CycleBudgets)

    # Every task in the distribution is evaluated in every cycle.
    results = run_cycles(
        robot,
        domain.real_env,
        task_distribution,
        list(task_distribution.get_tasks()),
        cfg.num_cycles,
        budgets,
        np.random.default_rng(cfg.seed),
    )

    eval_df = pd.DataFrame(_get_eval_rows(results))
    for cycle, cycle_df in eval_df.groupby("cycle"):
        logging.info(
            f"Cycle {cycle}: success rate {cycle_df.success.mean():.2f}, "
            f"mean planning time {cycle_df.planning_wall_time.mean():.4f}s"
        )
    eval_df.to_csv(cfg.eval_results_file, index=False)
    logging.info(f"Wrote evaluation results to {cfg.eval_results_file}")
    cycle_df = pd.DataFrame(_get_cycle_rows(results))
    cycle_df.to_csv(cfg.cycle_results_file, index=False)
    logging.info(f"Wrote cycle results to {cfg.cycle_results_file}")


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
    """One row per cycle, with the time spent sleeping and what the robot did between
    sleeping and evaluation."""
    return [
        {
            "cycle": result.cycle,
            "sleep_wall_time": result.sleep_wall_time,
            "num_queries": len(result.gathered_data),
            "num_executed_tasks": len(result.executions),
        }
        for result in results
    ]


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
