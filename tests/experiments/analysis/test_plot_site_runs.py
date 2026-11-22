"""Tests for experiments/analysis/plot_site_runs.py."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from able.approaches import make_scratch_robot
from able.cycle import CycleBudgets, run_cycles
from able.envs.base import FiniteTaskDistribution
from able.envs.friction.layout import make_world
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR
from able.envs.two_link_arm import ArmTask, make_domain
from able.structs import Budget

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"
LEVELS = np.array([0.03, 0.08, 0.15])


def _load() -> ModuleType:
    sys.path.insert(0, str(ANALYSIS_DIR))
    try:
        import plot_site_runs  # pylint: disable=import-outside-toplevel
    finally:
        sys.path.pop(0)
    return plot_site_runs


def _domain(module: ModuleType, num_cycles: int = 3) -> Any:
    """Three sites in groups [1, 2] on the corridors maze: group 0 uses floors
    0 and 1, group 1 floors 3 to 5. Site 2 differs from site 1 on floor 3."""
    truths = np.tile(LEVELS[[0, 1, 2, 0, 1, 2, 0, 1, 2]], (3, 1))
    truths[2, 3] = 0.15
    usage = np.zeros((num_cycles, 3, 9))
    usage[:, 0, :2] = 0.5
    usage[:, 1:, 3:6] = 1 / 3
    return module.SiteDomain(
        group_of=np.array([0, 1, 1]),
        schedule=np.array([0.5, 0.25, 0.25]),
        truths=truths,
        prior_mean=np.full((3, 9), 0.08),
        prior_std=np.full((3, 9), 0.05),
        usage=usage,
        routine_size=2,
        levels=LEVELS,
        world=make_world(2, 0.2, 9, "corridors"),
    )


def _run(module: ModuleType, approach: str, seed: int, domain: Any) -> Any:
    """Three cycles in which the belief means move to the truths and the
    simulations fall."""
    rows = []
    for cycle, step in enumerate([0.0, 0.5, 1.0]):
        means = (1 - step) * 0.08 + step * domain.truths
        row = {"cycle": cycle, "training_sites": ["0;1;2", "1", ""][cycle]}
        for i, (mean, std) in enumerate(
            zip(means.ravel(), np.full(27, 0.05 - step / 40))
        ):
            row[f"belief_mean_{i}"] = mean
            row[f"belief_std_{i}"] = std
        rows.append(row)
    evals = pd.DataFrame(
        {
            "cycle": np.repeat(range(3), 6),
            "task_id": np.tile(range(6), 3),
            "num_sims": np.repeat([300.0, 200.0, 100.0], 6),
        }
    )
    return module.SiteRun(approach, seed, pd.DataFrame(rows), evals, domain)


class _DrawnDays(FiniteTaskDistribution[ArmTask]):
    """One task a day, whose goal is drawn from the day's stream."""

    def __init__(self) -> None:
        super().__init__([ArmTask(initial_state=np.zeros(2), goal=np.zeros(2))])

    def start_day(self, rng: np.random.Generator) -> None:
        goal = np.array([rng.uniform(), 0.0])
        super().__init__([ArmTask(initial_state=np.zeros(2), goal=goal)])


def test_replay_days_matches_run_cycles() -> None:
    """Tests that replaying the days gives each cycle the tasks it had in
    run_cycles()."""
    module = _load()
    domain = make_domain(seed=0)
    days = _DrawnDays()
    seen: list[float] = []
    budgets = CycleBudgets(
        Budget(), Budget(max_iterations=0), Budget(max_iterations=1), Budget()
    )
    run_cycles(
        make_scratch_robot(domain),
        domain.real_env,
        days,
        days.get_tasks,
        3,
        budgets,
        np.random.default_rng(7),
        on_cycle_end=lambda _: seen.append(days.get_tasks()[0].goal[0]),
    )
    replayed = _DrawnDays()
    goals = [
        replayed.get_tasks()[0].goal[0]
        for _ in module.replay_days(7, 3, domain.real_env, replayed)
    ]
    assert goals == seen and len(set(goals)) == 3


def test_units_shares_and_moments() -> None:
    """Tests counting each night's units per group, the groups' schedule shares,
    and reading each site's belief from the columns."""
    module = _load()
    units = module.count_group_units(["0;1;2", 2, float("nan"), "1;1"], [0, 1, 1], 2)
    assert units.tolist() == [[1, 2], [0, 1], [0, 0], [0, 2]]
    shares = module.get_group_shares(np.array([0.5, 0.2, 0.3]), np.array([0, 1, 1]), 2)
    assert np.allclose(shares, [0.5, 0.5])
    cycles = pd.DataFrame(
        {
            **{f"belief_mean_{i}": [float(i)] for i in range(6)},
            **{f"belief_std_{i}": [-float(i)] for i in range(6)},
        }
    )
    means, stds = module.get_belief_moments(cycles, 2, 3)
    assert means.shape == (1, 2, 3) and means[0, 1, 0] == 3.0 and stds[0, 1, 2] == -5.0


def test_staircase_measures() -> None:
    """Tests the belief error over each routine's floors, the simulations per
    site, the schedule weighting, and the staircase built from them."""
    module = _load()
    values = np.array([[[1.0, 3.0, 9.0]]])
    usage = np.array([[[0.5, 0.5, 0.0]]])
    assert module.mean_over_routine_floors(values, usage).tolist() == [[2.0]]
    evals = pd.DataFrame(
        {"cycle": [0, 0, 0, 0], "task_id": [0, 1, 2, 3], "num_sims": [1, 3, 5, 7]}
    )
    assert module.get_site_sims(evals, 2, 2).tolist() == [[2.0, 6.0]]
    weighted = module.weight_by_schedule(
        np.array([[1.0, 2.0, 4.0]]), np.array([0.5, 0.1, 0.3]), np.array([0, 1, 1]), 2
    )
    assert np.allclose(weighted, [[1.0, 3.5]])
    domain = _domain(module)
    staircase = module.get_staircase(_run(module, "a", 0, domain), "error")
    group_0 = staircase[staircase.group == 0]
    assert np.allclose(group_0.belief, [0.5, 0.25, 0.0])
    assert group_0.sims.tolist() == [300.0, 200.0, 100.0]
    stds = module.get_staircase(_run(module, "a", 0, domain), "std")
    assert np.allclose(stds[stds.group == 1].belief, [1.0, 0.75, 0.5])
    assert np.allclose(module.get_prior_measure(domain, "error"), [0.5, 13 / 15])


def test_step_path_and_levels() -> None:
    """Tests that the staircase moves down before sideways, and that each true
    friction gets its level despite the jitter, as does each group's floor."""
    module = _load()
    x, y = module.step_path(np.array([3.0, 2.0, 1.0]), np.array([9.0, 5.0, 4.0]))
    assert x.tolist() == [3.0, 3.0, 2.0, 2.0, 1.0]
    assert y.tolist() == [9.0, 5.0, 5.0, 4.0, 4.0]
    truths = np.array([0.036, 0.064, 0.096, 0.12, 0.024, 0.18])
    assert module.get_friction_levels(truths, LEVELS).tolist() == [0, 1, 1, 2, 0, 2]
    levels, differ = module.get_group_floor_levels(_domain(module), 1)
    assert levels[:6].tolist() == [0, 1, 2, 0, 1, 2]
    assert differ.tolist() == [False] * 3 + [True] + [False] * 5


def test_plots(tmp_path: Path) -> None:
    """Tests that each figure is written from runs of two approaches and
    seeds."""
    module = _load()
    domain = _domain(module)
    runs = [
        _run(module, approach, seed, domain)
        for approach in ("transfer_practice", "current_belief")
        for seed in (0, 1)
    ]
    module.plot_allocation(runs, tmp_path / "allocation.png")
    module.plot_staircase(runs, "error", tmp_path / "staircase.png")
    module.plot_group_map({0: domain, 1: domain}, tmp_path / "map.png")
    for name in ("allocation", "staircase", "map"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_level_set() -> None:
    """Tests that the site map's levels are the groups' base levels, or the
    prior's levels when each floor draws its own."""
    module = _load()
    based = OmegaConf.create({"group_friction": "base", "base_levels": [0.2, 0.05]})
    assert module.get_level_set(based).tolist() == [0.05, 0.2]
    assert module.get_level_set(OmegaConf.create({})).tolist() == [
        0.025,
        0.05,
        0.1,
        0.2,
    ]
    per_floor = OmegaConf.create({"group_friction": "per_floor"})
    assert module.get_level_set(per_floor).tolist() == list(
        DEFAULT_FRICTION_PRIOR.levels
    )
