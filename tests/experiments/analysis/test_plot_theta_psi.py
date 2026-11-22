"""Tests for experiments/analysis/plot_theta_psi.py."""

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from able.beliefs.particles import ParticleBelief
from able.beliefs.sites import SiteBelief
from able.envs.friction.layout import make_world
from able.envs.friction.pomdp import PuckPushingModel
from able.models.base import Dataset
from able.planners.pomcpow import get_feature_dims

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "analysis"
    / "plot_theta_psi.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_theta_psi", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pomdp() -> PuckPushingModel:
    return PuckPushingModel(make_world(2, 0.2, 12, "corridors4"))


def _near_goal(script: ModuleType, pomdp: PuckPushingModel) -> Any:
    """A puck 0.7 m from a goal in corridor 0 with a clear push to it."""
    world = pomdp.world
    colors = np.asarray(world.puck_map.surface_colors)
    h = world.puck_map.cell_size
    for row in range(world.puck_map.num_rows):
        for col in range(world.puck_map.num_cols):
            goal = ((col + 0.5) * h, (row + 0.5) * h)
            for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                puck = np.array(goal) - 0.7 * np.array(
                    [math.cos(angle), math.sin(angle)]
                )
                surfaces = [world.surface_at(puck), world.surface_at(np.array(goal))]
                if min(surfaces) < 0 or colors[surfaces[0]] != 0:
                    continue
                stances = [
                    s for s in pomdp.stances if world.push_feasible(puck, angle, s)
                ]
                if not stances:
                    continue
                state = script.NearGoal(puck, angle, float(stances[0]), goal)
                if np.isfinite(script.landing_speeds(pomdp, state, [0.05] * 12)[0]):
                    return state
    raise AssertionError("No clear push to a goal")


def _dataset(pomdp: PuckPushingModel, states: list[Any]) -> Dataset:
    x_dim = sum(get_feature_dims(pomdp.world))
    x = np.zeros((len(states), x_dim))
    y = np.full((len(states), 5), np.nan)
    for i, (state, has_push) in enumerate(states):
        x[i, :2], x[i, 2:4] = state.puck, state.goal
        if has_push:
            y[i, :4] = [math.cos(state.heading), math.sin(state.heading), 0.5, 0.0]
    return Dataset(x, y, np.zeros(len(states)), np.arange(len(states)))


def test_scale_belief() -> None:
    """Tests that a rescaled belief has the requested mean and keeps its shape."""
    script = _load_script()
    particles = np.random.default_rng(0).uniform(0.09, 0.11, (50, 3))
    belief = ParticleBelief(particles, np.zeros(3), np.ones(3))
    scaled = script.scale_belief(belief, 0.2)
    assert np.isclose(np.asarray(scaled.particles).mean(), 0.2)
    assert np.allclose(np.asarray(scaled.particles) / particles, 0.2 / particles.mean())


def test_landing_speeds_rise_with_friction() -> None:
    """Tests that more friction needs faster pushes to land the puck."""
    script = _load_script()
    pomdp = _pomdp()
    state = _near_goal(script, pomdp)
    slow = script.landing_speeds(pomdp, state, [0.03] * 12)
    fast = script.landing_speeds(pomdp, state, [0.15] * 12)
    assert slow[0] < slow[1] and fast[0] < fast[1]
    assert slow[0] < fast[0] and slow[1] < fast[1]


def test_near_goal_states_filters_rows() -> None:
    """Tests that only pushes in the corridor and near their goal are kept."""
    script = _load_script()
    pomdp = _pomdp()
    state = _near_goal(script, pomdp)
    far = state._replace(goal=(state.goal[0] + 3.0, state.goal[1]))
    dataset = _dataset(pomdp, [(state, True), (state, False), (far, True)])
    found = script.near_goal_states(dataset, pomdp, corridor=0)
    assert len(found) == 1 and np.allclose(found[0].puck, state.puck)
    assert not script.near_goal_states(dataset, pomdp, corridor=1)


def test_measure_and_plot(tmp_path: Path) -> None:
    """Tests the rows for a psi that always aims at the goal at a landing speed,
    and that the figure is written."""
    script = _load_script()
    pomdp = _pomdp()
    state = _near_goal(script, pomdp)
    theta = 0.05
    speed = float(np.mean(script.landing_speeds(pomdp, state, [theta] * 12)))

    class Aimed:  # pylint: disable=too-few-public-methods
        """Always the push straight at the goal at a landing speed."""

        def sample(self, x: Any, num: int, rng: Any) -> Any:
            """num copies of the push."""
            del x, rng
            row = [math.cos(state.heading), math.sin(state.heading), speed, 0.0]
            return np.tile([*row[:3], state.stance], (num, 1))

    particles = np.full((20, 12), theta)
    beliefs = SiteBelief([ParticleBelief(particles, np.zeros(12), np.ones(12))])
    snapshot = {
        "seed": 0,
        "train_site": 0,
        "truths": np.full((1, 12), theta),
        "beliefs": beliefs,
        "corridors": [0],
        "dataset": _dataset(pomdp, [(state, True)]),
        "snapshots": {0: None, 4: SimpleNamespace(model=Aimed())},
        "pomdp": pomdp,
    }
    rows = script.measure(snapshot, [0.025, 0.1], num_draws=16)
    assert len(rows) == 6 and rows.exact.sum() == 2
    own = rows[rows.exact & (rows.depth == 4)].iloc[0]
    assert np.isclose(own.theta, theta) and own.num_states == 1
    assert np.isclose(own.speed, speed) and own.cos_mean > 0.99
    assert own.landed > 0.5
    assert 0.0 <= rows.landed.min() and rows.landed.max() <= 1.0
    every = rows.assign(train_site=-1, train_theta=np.nan)
    out = tmp_path / "theta_psi.png"
    script.plot(pd.concat([rows, every]), seed=0, out=out)
    assert out.exists() and out.stat().st_size > 0
