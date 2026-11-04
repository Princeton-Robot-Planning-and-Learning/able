"""Tests for envs/human_reach/query.py."""

import numpy as np
import pytest

from able.envs.human_reach.query import (
    PRESENT_TOL,
    QuerySimulator,
    ReachQuery,
    bedside_directions,
    present_conf,
    reach_distance,
)

THETA = np.array([0.7, 0.6, 0.65])


@pytest.fixture(name="exact", scope="module")
def _exact():
    simulator = QuerySimulator(noisy=False, stop_at_miss=False)
    yield simulator
    simulator.close()


def test_directions_and_reach_distance(exact):
    """Directions are unit vectors, and the reach distance lies on the ellipsoid."""
    directions = bedside_directions(exact.config, exact.sim.shoulder)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    r = reach_distance(directions, THETA)
    on_surface = directions * r[:, None] / THETA
    assert np.allclose(np.linalg.norm(on_surface, axis=1), 1.0)


def test_reached_near_not_far_and_skips_unpresentable_points(exact):
    """Near points are reached and far ones are not; points the robot cannot
    present are skipped."""
    sim = exact.sim
    d = bedside_directions(exact.config, sim.shoulder)[0]
    assert reach_distance(d, THETA) == pytest.approx(0.655, abs=0.01)
    response = exact.ask(
        THETA, ReachQuery(d, (0.3, 0.6, 0.8)), np.random.default_rng(0)
    )
    assert [reached for _, reached in response] == [True, False]
    for (point, _), s in zip(response, (0.6, 0.8)):
        assert np.linalg.norm(np.array(point) - (sim.shoulder + s * d)) < PRESENT_TOL
        assert present_conf(sim, np.array(point)) is not None
    # The arm would hit the person.
    assert present_conf(sim, sim.shoulder + 0.3 * d) is None


def test_noisy_answers_stop_at_the_first_miss():
    """Noisy answers stop after the first no."""
    simulator = QuerySimulator()
    d = bedside_directions(simulator.config, simulator.sim.shoulder)[0]
    response = simulator.ask(
        THETA, ReachQuery(d, (0.6, 0.8, 0.85)), np.random.default_rng(0)
    )
    answers = [reached for _, reached in response]
    assert answers and (all(answers) or answers.index(False) == len(answers) - 1)
    simulator.close()
