"""Map planning cost over contexts and over models trained under one context each.

A context is the true hidden parameters, the belief the robot plans under, and a
fixed set of evaluation tasks drawn given the truth. Known contexts plan under a
point belief at the truth. Visited contexts plan under the beliefs a robot holds
after some cycles of real experience, with its sleep turned off. Contexts with the
same truth share their tasks.

Each training context gets a run: a model trained from the untrained model under
that context's belief only, checkpointed at each depth (sleep iterations so far).
Every checkpoint is evaluated on its own context, and the checkpoints at the
cross depths are evaluated on every context. The untrained model is evaluated on
every context. Evaluation uses common random numbers: a context's tasks, planning
seeds, and environment seeds are the same for every model.

The work runs in three stages, so that a cluster can spread each one over many
jobs (see shard and num_shards):
1. contexts: make the contexts (contexts.pkl, and a summary in contexts.csv);
2. train: train each run and save its checkpoints (models/);
3. eval: evaluate models on chunks of contexts, with a CSV per unit of work and a
   row per evaluation task (evals/).
Stage all runs the three in turn. A unit whose output exists is skipped, so a
stopped stage resumes where it left off. See experiments/conf/landscape.yaml for
the settings and experiments/analysis/plot_landscape.py for the figures.
"""

import dataclasses
import functools
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.beliefs.particles import ParticleBelief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.cycle import CycleBudgets, CycleResult, run_cycles
from able.envs.base import Domain, FiniteTaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.learners.belief_features import make_moment_features
from able.planners.base import Planner
from able.robot import Robot
from able.structs import Budget, PlanningFailure, PlanningTimeout

# Separate random streams, so that changing one setting does not shift the draws
# made for another.
_TRUTH_STREAM = 1000
_TASK_STREAM = 13
_CYCLE_STREAM = 17
_TRAIN_STREAM = 19
_PLAN_STREAM = 23
_ENV_STREAM = 29

# The cycle of a known context, and of a visited context under the prior.
KNOWN_CYCLE = -2
PRIOR_CYCLE = -1

# The training context of the untrained model's rows.
BASE_CONTEXT = -1

STAGES = ("all", "contexts", "train", "eval")


@dataclass(frozen=True)
class Context:
    """A truth, a belief to plan under, and evaluation tasks.

    The trajectory numbers the truth the context came from, and the cycle is the
    cycle after which a visited context's belief was taken (PRIOR_CYCLE for the
    prior, KNOWN_CYCLE for a known context). A model is trained under the belief
    only if train is set. The features are the belief's moment features.
    """

    index: int
    trajectory: int
    cycle: int
    truth: Any
    belief: Belief[Any]
    tasks: list[Any]
    train: bool
    features: Any


@dataclass(frozen=True)
class EvalUnit:
    """Models to evaluate on contexts, written to one CSV.

    The kind is base (the untrained model), diag (every checkpoint of a run on
    its own context), or cross (one checkpoint on a chunk of contexts).
    """

    kind: str
    contexts: tuple[int, ...]
    train_context: int = BASE_CONTEXT
    replicate: int = 0
    depth: int = 0

    @property
    def name(self) -> str:
        """The unit's file name, without the extension."""
        if self.kind == "base":
            return f"base_c{self.contexts[0]}"
        run = f"context{self.train_context}_r{self.replicate}"
        if self.kind == "diag":
            return f"diag_{run}"
        return f"cross_{run}_d{self.depth}_c{self.contexts[0]}"


def make_point_belief(prior: Belief[Any], truth: Any) -> Belief[Any]:
    """A belief of the prior's type that is certain of the truth, or as nearly
    certain as the type allows."""
    truth = np.asarray(truth, dtype=np.float64)
    if isinstance(prior, IntervalBelief):
        return IntervalBelief(np.stack([truth, truth], axis=1))
    if isinstance(prior, ParticleBelief):
        particles = np.tile(truth, (len(prior.particles), 1))
        return ParticleBelief(particles, prior.low, prior.high)
    if isinstance(prior, RangeOfMotionBelief):
        stddev = 1e-3 * (prior.high - prior.low)
        return dataclasses.replace(prior, mean=truth, stddev=stddev)
    raise TypeError(f"No point belief for {type(prior).__name__}")


def get_truth(domain: Domain) -> Any:
    """The hidden parameters of the domain's simulated real environment."""
    real_env = domain.real_env
    assert isinstance(real_env, SimulatedEnvironment)
    return real_env.hidden_params


class LandscapeBuilder:
    """Builds domains and robots from the configuration.

    Every domain shares the world seed and differs only in its true hidden
    parameters, given through the environment's argument for them.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._truth_arg = cfg.truth_args[cfg.env_name]

    def make_domain(self, truth: Any = None, seed: int | None = None) -> Domain:
        """The domain with the given truth, or with its own draw from the prior
        for the seed (the world seed by default) if no truth is given."""
        seed = self._cfg.landscape.world_seed if seed is None else seed
        kwargs = {} if truth is None else {self._truth_arg: np.asarray(truth)}
        domain = hydra.utils.instantiate(self._cfg.env.domain, seed=seed, **kwargs)
        assert isinstance(domain, Domain)
        return domain

    def make_robot(self, domain: Domain) -> Robot:
        """The approach's robot for the domain."""
        make_robot = hydra.utils.instantiate(self._cfg.approach, _partial_=True)
        robot = make_robot(domain=domain, recorder=None)
        assert isinstance(robot, Robot)
        return robot

    def draw_truth(self, trajectory: int) -> Any:
        """The truth for a trajectory: the world's truth if the truth is fixed,
        and otherwise a draw from the environment's prior."""
        if self._cfg.landscape.fixed_truth:
            return get_truth(self.make_domain())
        return get_truth(self.make_domain(seed=_TRUTH_STREAM + trajectory))

    def draw_tasks(self, domain: Domain, trajectory: int) -> list[Any]:
        """Every task of a finite distribution, or num_tasks tasks drawn for the
        trajectory."""
        distribution = domain.task_distribution
        if isinstance(distribution, FiniteTaskDistribution):
            return list(distribution.get_tasks())
        rng = np.random.default_rng([trajectory, _TASK_STREAM])
        return [distribution.sample(rng) for _ in range(self._cfg.landscape.num_tasks)]


def make_trajectory_records(
    cfg_dict: dict[str, Any], trajectory: int
) -> list[dict[str, Any]]:
    """The fields of one trajectory's contexts, without their index and
    features."""
    cfg = OmegaConf.create(cfg_dict)
    builder = LandscapeBuilder(cfg)
    truth = np.asarray(builder.draw_truth(trajectory))
    domain = builder.make_domain(truth)
    tasks = builder.draw_tasks(domain, trajectory)
    prior = domain.initial_belief
    records = []

    def add(cycle: int, belief: Belief, train: bool) -> None:
        records.append(
            {
                "trajectory": trajectory,
                "cycle": cycle,
                "truth": truth,
                "belief": belief,
                "tasks": tasks,
                "train": train,
            }
        )

    if cfg.landscape.contexts == "known":
        add(KNOWN_CYCLE, make_point_belief(prior, truth), True)
        return records
    cycles = sorted(cfg.landscape.cycles)
    if PRIOR_CYCLE in cycles and (trajectory == 0 or not cfg.landscape.fixed_truth):
        # Every trajectory starts at the prior, so one model trained under it
        # serves them all.
        add(PRIOR_CYCLE, prior, trajectory == 0)
    for cycle, belief in visit_beliefs(cfg, builder, domain, trajectory, cycles):
        add(cycle, belief, True)
    return records


def visit_beliefs(
    cfg: DictConfig,
    builder: LandscapeBuilder,
    domain: Domain,
    trajectory: int,
    cycles: list[int],
) -> list[tuple[int, Belief]]:
    """The beliefs after the given cycles of a robot that never sleeps, with the
    configured queries and task executions in the domain."""
    wanted = [c for c in cycles if c >= 0]
    if not wanted:
        return []
    robot = builder.make_robot(domain)
    budgets = hydra.utils.instantiate(cfg.budgets)
    assert isinstance(budgets, CycleBudgets)
    budgets = dataclasses.replace(budgets, sleep=Budget(max_iterations=0))
    beliefs: list[tuple[int, Belief]] = []

    def collect(result: CycleResult) -> None:
        if result.cycle in wanted:
            beliefs.append((result.cycle, result.belief))

    run_cycles(
        robot,
        domain.real_env,
        domain.task_distribution,
        [],
        max(wanted) + 1,
        budgets,
        np.random.default_rng([trajectory, _CYCLE_STREAM]),
        on_cycle_end=collect,
    )
    return beliefs


def make_contexts(cfg: DictConfig, cfg_dict: dict[str, Any]) -> list[Context]:
    """The known or visited contexts the configuration asks for, with each
    trajectory made by one of num_workers processes."""
    assert cfg.landscape.contexts in ("known", "visited"), "Unknown contexts"
    trajectories = list(range(cfg.landscape.num_truths))
    records = [
        record
        for records in run_parallel(
            make_trajectory_records,
            cfg_dict,
            trajectories,
            cfg.num_workers,
            "trajectory",
        )
        for record in records
    ]
    prior = LandscapeBuilder(cfg).make_domain().initial_belief
    get_features = make_moment_features(prior)
    return [
        Context(index=index, features=get_features(record["belief"]), **record)
        for index, record in enumerate(records)
    ]


def evaluate_context(
    planner: Planner,
    params: Any,
    env_model: Any,
    context: Context,
    budget: Budget,
) -> list[dict[str, Any]]:
    """Plan and run each of the context's tasks with the parameters, and return a
    row per task.

    Each task's planning and environment seeds depend only on the context and
    the task, so every model sees the same draws.
    """
    rows = []
    for task_index, task in enumerate(context.tasks):
        rng = np.random.default_rng([context.index, task_index, _PLAN_STREAM])
        env_rng = np.random.default_rng([context.index, task_index, _ENV_STREAM])
        env = SimulatedEnvironment(env_model, context.truth, env_rng)
        start_time = time.perf_counter()
        failure, timed_out = "", False
        try:
            policy = planner.plan(task, context.belief, params, budget, rng)
            trajectory = run_episode(env, task, policy)
            success = env.check_goal(task, trajectory.states[-1])
            stats = policy.get_planning_stats()
        except PlanningFailure as error:
            success, stats = False, dict(error.stats)
            failure, timed_out = str(error), isinstance(error, PlanningTimeout)
        row: dict[str, Any] = {
            "eval_context": context.index,
            "task": task_index,
            "success": bool(success),
            "timed_out": timed_out,
            "failure": failure,
            "wall_time": time.perf_counter() - start_time,
        }
        row.update({f"stat_{k}": v for k, v in stats.items()})
        rows.append(row)
    return rows


def get_train_units(cfg: DictConfig, contexts: list[Context]) -> list[tuple[int, int]]:
    """The (training context, replicate) pair of every run."""
    return [
        (c.index, replicate)
        for replicate in range(cfg.landscape.num_replicates)
        for c in contexts
        if c.train
    ]


def get_eval_units(cfg: DictConfig, contexts: list[Context]) -> list[EvalUnit]:
    """The units of evaluation work, with at most contexts_per_unit contexts in
    each base and cross unit."""
    size = cfg.landscape.contexts_per_unit
    indices = [c.index for c in contexts]
    chunks = [tuple(indices[i : i + size]) for i in range(0, len(indices), size)]
    units = [EvalUnit("base", chunk) for chunk in chunks]
    for train_context, replicate in get_train_units(cfg, contexts):
        units.append(EvalUnit("diag", (train_context,), train_context, replicate))
        for depth in sorted(cfg.landscape.cross_depths):
            units.extend(
                EvalUnit("cross", chunk, train_context, replicate, depth)
                for chunk in chunks
            )
    return units


def get_model_path(out_dir: Path, train_context: int, replicate: int, depth: int):
    """Where a run's checkpoint at a depth is saved."""
    return out_dir / "models" / f"context{train_context}_r{replicate}_d{depth}.pkl"


def train_run(cfg_dict: dict[str, Any], out_dir: str, unit: tuple[int, int]) -> None:
    """Train one run under its context's belief and save a checkpoint at each
    depth, with the training time and data so far."""
    _limit_threads()
    cfg = OmegaConf.create(cfg_dict)
    train_context, replicate = unit
    context = load_contexts(Path(out_dir))[train_context]
    builder = LandscapeBuilder(cfg)
    robot = builder.make_robot(builder.make_domain(context.truth))
    params = robot.planner_params
    rng = np.random.default_rng([train_context, replicate, _TRAIN_STREAM])
    trained, train_time = 0, 0.0
    for depth in sorted(d for d in cfg.landscape.depths if d > 0):
        start_time = time.perf_counter()
        params = robot.learner.learn(
            context.belief, params, Budget(max_iterations=depth - trained), rng
        )
        train_time += time.perf_counter() - start_time
        trained = depth
        dataset = getattr(robot.learner, "dataset", ())
        checkpoint = {"params": params, "train_time": train_time, "rows": len(dataset)}
        _write_atomically(
            get_model_path(Path(out_dir), train_context, replicate, depth),
            functools.partial(_dump, checkpoint),
        )


def evaluate_unit(cfg_dict: dict[str, Any], out_dir: str, unit: EvalUnit) -> None:
    """Evaluate a unit's models on its contexts and write a row per task."""
    _limit_threads()
    cfg = OmegaConf.create(cfg_dict)
    contexts = load_contexts(Path(out_dir))
    builder = LandscapeBuilder(cfg)
    source = contexts[unit.contexts[0] if unit.kind == "base" else unit.train_context]
    domain = builder.make_domain(source.truth)
    robot = builder.make_robot(domain)
    budgets = hydra.utils.instantiate(cfg.budgets)
    if unit.kind == "base":
        models = [(0, robot.planner_params, 0.0)]
    else:
        depths = [unit.depth]
        if unit.kind == "diag":
            depths = sorted(d for d in cfg.landscape.depths if d > 0)
        models = []
        for depth in depths:
            path = get_model_path(
                Path(out_dir), unit.train_context, unit.replicate, depth
            )
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
            models.append((depth, checkpoint["params"], checkpoint["train_time"]))
    rows: list[dict[str, Any]] = []
    for depth, params, train_time in models:
        for index in unit.contexts:
            for row in evaluate_context(
                robot.planner,
                params,
                domain.env_model,
                contexts[index],
                budgets.planning,
            ):
                row.update(
                    train_context=unit.train_context,
                    replicate=unit.replicate,
                    depth=depth,
                    train_time=train_time,
                )
                rows.append(row)
    _write_atomically(
        Path(out_dir) / "evals" / f"{unit.name}.csv",
        lambda f: pd.DataFrame(rows).to_csv(f, index=False),
        binary=False,
    )


def _dump(value: Any, f: Any) -> None:
    pickle.dump(value, f)


def _write_atomically(
    path: Path, write: Callable[[Any], None], binary: bool = True
) -> None:
    """Write to a temporary file and then rename it, so that a stopped job never
    leaves a partial file."""
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "wb" if binary else "w") as f:
        write(f)
    temp_path.replace(path)


def _limit_threads() -> None:
    """One thread per process, so that parallel jobs do not compete for cores."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    torch.set_num_threads(1)


def load_contexts(out_dir: Path) -> list[Context]:
    """The contexts saved by write_contexts(), which saves each one's fields
    rather than the Context itself, so that they load in any module."""
    with open(out_dir / "contexts.pkl", "rb") as f:
        saved = pickle.load(f)
    return [c if isinstance(c, Context) else Context(**c) for c in saved]


def write_contexts(contexts: list[Context], out_dir: Path) -> None:
    """Save the contexts for the later stages and a readable summary of them."""
    _write_atomically(
        out_dir / "contexts.pkl",
        functools.partial(_dump, [vars(c) for c in contexts]),
    )
    rows = []
    for c in contexts:
        row: dict[str, Any] = {
            "index": c.index,
            "trajectory": c.trajectory,
            "cycle": c.cycle,
            "train": c.train,
            "num_tasks": len(c.tasks),
        }
        row.update({f"truth_{i}": v for i, v in enumerate(np.ravel(c.truth))})
        row.update({f"feature_{i}": v for i, v in enumerate(c.features)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "contexts.csv", index=False)


def run_parallel(
    function: Callable[..., Any],
    cfg_dict: dict[str, Any],
    items: list[Any],
    num_workers: int,
    label: str,
    *args: Any,
) -> list[Any]:
    """function(cfg_dict, *args, item) for each item, in num_workers processes,
    with a progress line after each. Returns the results in the items' order.

    An item that raises is logged and does not stop the others; once every item
    has finished, a RuntimeError names the items that raised.
    """
    start_time = time.perf_counter()
    results: list[Any] = [None] * len(items)
    failed: list[Any] = []

    def log(num_done: int, item: Any) -> None:
        elapsed = time.perf_counter() - start_time
        left = elapsed / num_done * (len(items) - num_done)
        logging.info(
            f"{label} {item} done ({num_done}/{len(items)}), elapsed "
            f"{elapsed / 60:.1f} min, about {left / 60:.1f} min left"
        )

    if num_workers <= 1:
        for num_done, (k, item) in enumerate(enumerate(items), 1):
            try:
                results[k] = function(cfg_dict, *args, item)
            except Exception:  # pylint: disable=broad-exception-caught
                logging.exception(f"{label} {item} failed")
                failed.append(item)
            log(num_done, item)
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(num_workers, mp_context=context) as pool:
            futures = {
                pool.submit(function, cfg_dict, *args, item): k
                for k, item in enumerate(items)
            }
            for num_done, future in enumerate(as_completed(futures), 1):
                k = futures[future]
                try:
                    results[k] = future.result()
                except Exception:  # pylint: disable=broad-exception-caught
                    logging.exception(f"{label} {items[k]} failed")
                    failed.append(items[k])
                log(num_done, items[k])
    if failed:
        raise RuntimeError(f"{len(failed)} items failed: {failed}")
    return results


@hydra.main(version_base=None, config_name="landscape", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    assert cfg.stage in STAGES, f"stage must be one of {STAGES}"
    assert set(cfg.landscape.cross_depths) <= set(cfg.landscape.depths)
    out_dir = Path(cfg.out_dir)
    for name in ("models", "evals"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(resolved, dict)
    cfg_dict = {str(key): value for key, value in resolved.items()}
    if cfg.stage in ("all", "contexts"):
        # Only these stages write the settings, so that shards running at once
        # never write the same file.
        OmegaConf.save(cfg_dict, out_dir / "landscape_config.yaml")
        if not (out_dir / "contexts.pkl").exists():
            write_contexts(make_contexts(cfg, cfg_dict), out_dir)
    contexts = load_contexts(out_dir)
    if cfg.stage in ("all", "train"):
        units = get_train_units(cfg, contexts)[cfg.shard :: cfg.num_shards]
        max_depth = max(cfg.landscape.depths)
        units = [
            u for u in units if not get_model_path(out_dir, *u, max_depth).exists()
        ]
        logging.info(f"{len(units)} runs to train with {cfg.num_workers} workers")
        run_parallel(train_run, cfg_dict, units, cfg.num_workers, "Run", str(out_dir))
    if cfg.stage in ("all", "eval"):
        eval_units = get_eval_units(cfg, contexts)[cfg.shard :: cfg.num_shards]
        eval_units = [
            u for u in eval_units if not (out_dir / "evals" / f"{u.name}.csv").exists()
        ]
        logging.info(f"{len(eval_units)} units to evaluate")
        run_parallel(
            evaluate_unit, cfg_dict, eval_units, cfg.num_workers, "Unit", str(out_dir)
        )


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
