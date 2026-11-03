"""Solving PDDLStream problems, with each stream evaluation counted."""

import os
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from typing import Callable, Iterator

from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test


@contextmanager
def private_cwd() -> Iterator[None]:
    """Run in a fresh temp dir: Fast Downward writes ./temp in the cwd, so parallel
    runs would collide."""
    old, tmp = os.getcwd(), tempfile.mkdtemp(prefix="pddlstream_")
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)


def counted(fn: Callable, calls: Counter, name: str, generator: bool) -> Callable:
    """`fn` with each evaluation of stream `name` counted in `calls`."""

    def test(*args):
        calls[name] += 1
        return fn(*args)

    def gen(*args):
        outputs = fn(*args)
        while True:
            calls[name] += 1
            try:
                out = next(outputs)
            except StopIteration:
                return
            yield out

    return gen if generator else test


def stream_map(
    gens: dict[str, Callable], tests: dict[str, Callable], calls: Counter
) -> dict:
    """PDDLStream stream map with every evaluation counted."""
    out = {n: from_gen_fn(counted(f, calls, n, True)) for n, f in gens.items()}
    return out | {n: from_test(counted(f, calls, n, False)) for n, f in tests.items()}


def solve_problem(
    problem: PDDLProblem,
    max_time: float,
    algorithm: str = "adaptive",
    search_sample_ratio: float = 1.0,
    max_complexity: float = float("inf"),
    verbose: bool = False,
):
    """The plan (list of actions) or None."""
    # pddlstream walks its binding tree recursively; long plans exceed the default limit.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    with private_cwd():
        solution = solve(
            problem,
            algorithm=algorithm,
            unit_costs=True,
            max_time=max_time,
            search_sample_ratio=search_sample_ratio,
            max_complexity=max_complexity,
            verbose=verbose,
        )
    if verbose:
        print_solution(solution)
    return solution[0]
