"""Run independent pieces of a cycle in worker processes.

Evaluation tasks never change the robot, and the dreams and training plans of a
sleep whose model is refit once at its end all plan with the same parameters, so
each can run on its own CPU. A worker holds its own copy of the robot's parts,
built once from the run's configuration, and each job brings the belief, planner
parameters, and random generator it needs.

A job is a module-level function called as fn(context, *args), so that worker
processes can find it by name.
"""

import multiprocessing
import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import hydra
from omegaconf import OmegaConf

# Each worker uses one thread for numpy and torch, so that the workers together use
# the CPUs they were given.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class WorkerContext:
    """The parts of a robot that jobs use: its domain (for the environment model
    and the true hidden parameters of evaluation), planner, and learner."""

    domain: Any
    planner: Any
    learner: Any


class WorkerPool(ABC):
    """Runs jobs, each called as fn(context, *args), and returns their results in
    order."""

    @abstractmethod
    def map(self, fn: Callable[..., Any], jobs: Sequence[tuple[Any, ...]]) -> list[Any]:
        """Run fn(context, *args) for each job's args."""

    def close(self) -> None:
        """Release the pool's workers."""


class SerialPool(WorkerPool):
    """Runs jobs one after another in this process, with a context of the robot's
    own parts.

    Results match those of a ProcessPool built from the same configuration, so it
    tests the parallel code paths without worker processes.
    """

    def __init__(self, context: WorkerContext) -> None:
        self._context = context

    def map(self, fn: Callable[..., Any], jobs: Sequence[tuple[Any, ...]]) -> list[Any]:
        return [fn(self._context, *args) for args in jobs]


class ProcessPool(WorkerPool):
    """Runs jobs in num_workers processes, each with a context from make_context.

    make_context runs once in each worker and must be picklable, for example
    functools.partial(build_context, config). The workers start fresh ("spawn"),
    so they share no simulator or thread state with this process. If a worker dies
    (for example, killed for running out of memory), map() raises WorkerDied
    instead of waiting for its job forever.
    """

    def __init__(
        self, num_workers: int, make_context: Callable[[], WorkerContext]
    ) -> None:
        assert num_workers > 0, "Need at least one worker"
        # Workers start as jobs arrive and copy the environment then, so the thread
        # limits stay set until the pool closes.
        self._saved_environment = {
            name: os.environ.get(name) for name in _THREAD_VARIABLES
        }
        os.environ.update({name: "1" for name in _THREAD_VARIABLES})
        self._executor = ProcessPoolExecutor(
            num_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(make_context,),
        )

    def map(self, fn: Callable[..., Any], jobs: Sequence[tuple[Any, ...]]) -> list[Any]:
        try:
            return list(self._executor.map(_run_job, [fn] * len(jobs), jobs))
        except BrokenProcessPool as error:
            raise WorkerDied(
                "A worker process died, for example killed for running out of memory"
            ) from error

    def close(self) -> None:
        self._executor.shutdown()
        for name, value in self._saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class WorkerDied(RuntimeError):
    """Raised when a worker process dies while the pool has jobs for it."""


def build_context(config: dict[str, Any]) -> WorkerContext:
    """Build a robot's parts from a run's resolved configuration, as
    run_single_experiment.py does: the domain for the configuration's seed, and the
    approach's robot without a recorder."""
    cfg = OmegaConf.create(config)
    domain = hydra.utils.instantiate(cfg.env.domain, seed=cfg.seed)
    make_robot = hydra.utils.instantiate(cfg.approach, _partial_=True)
    robot = make_robot(domain=domain, recorder=None)
    return WorkerContext(domain=domain, planner=robot.planner, learner=robot.learner)


_CONTEXT: WorkerContext | None = None


def _init_worker(make_context: Callable[[], WorkerContext]) -> None:
    global _CONTEXT  # pylint: disable=global-statement
    # Imported here so that importing this module does not load torch.
    import torch  # pylint: disable=import-outside-toplevel

    torch.set_num_threads(1)
    _CONTEXT = make_context()


def _run_job(fn: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    assert _CONTEXT is not None, "The worker has no context"
    return fn(_CONTEXT, *args)
