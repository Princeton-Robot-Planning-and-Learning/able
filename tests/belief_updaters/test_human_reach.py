"""Tests for belief_updaters/human_reach.py."""

import numpy as np

from able.belief_updaters.human_reach import ReachBeliefUpdater, reached_log_likelihood
from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import ReachState
from able.envs.human_reach.query import ReachQuery
from able.structs import QueryResponse

SHOULDER = np.array([0.2, 0.1, 0.9])
LOW, HIGH = np.array([0.45, 0.35, 0.40]), np.array([0.95, 0.80, 0.90])


def _prior(num_particles: int = 500) -> ParticleBelief:
    rng = np.random.default_rng(0)
    return ParticleBelief(rng.uniform(LOW, HIGH, (num_particles, 3)), LOW, HIGH)


def test_likelihood_sign():
    """Reaching a point favors longer reaches, and missing it shorter ones."""
    point = SHOULDER + [0.6, 0.0, 0.0]
    small, large = np.array([[0.5, 0.6, 0.6]]), np.array([[0.9, 0.6, 0.6]])
    for reached in (True, False):
        gap = (
            reached_log_likelihood([point], [reached], large, SHOULDER)[0]
            - reached_log_likelihood([point], [reached], small, SHOULDER)[0]
        )
        assert gap > 0 if reached else gap < 0


def test_queries_and_deliveries_share_the_model():
    """A query's points weigh like separate deliveries of the same points."""
    belief = _prior()
    points = [tuple(SHOULDER + [0.5, 0.0, 0.0]), tuple(SHOULDER + [0.8, 0.0, 0.0])]
    together = reached_log_likelihood(points, [True, False], belief.particles, SHOULDER)
    apart = sum(
        reached_log_likelihood([p], [r], belief.particles, SHOULDER)
        for p, r in zip(points, (True, False))
    )
    assert np.allclose(together, apart)
    updater = ReachBeliefUpdater(SHOULDER)
    data = [
        QueryResponse(
            ReachQuery(np.array([1.0, 0, 0]), (0.5, 0.8)),
            tuple(zip(points, (True, False))),
        )
    ]
    posterior = updater.update_from_queries(belief, data)
    assert isinstance(posterior, ParticleBelief)
    assert 0.5 <= posterior.mean[0] <= 0.8 and posterior.std[0] < belief.std[0]


def test_deliveries_update_once_after_the_plan():
    """Only the step that executes the plan updates the belief, one item at a time."""
    belief = _prior()
    updater = ReachBeliefUpdater(SHOULDER)
    deliveries = (
        ("req0", tuple(SHOULDER + [0.6, 0.0, 0.0]), True),
        ("req1", tuple(SHOULDER + [0.0, 0.0, -0.75]), False),
    )
    before = ReachState(scene=None)
    after = ReachState(scene=None, executed=True, deliveries=deliveries)
    posterior = updater.update_from_transition(belief, before, (), after)
    assert isinstance(posterior, ParticleBelief)
    assert posterior.mean[0] > belief.mean[0] and posterior.mean[2] < belief.mean[2]
    assert updater.update_from_transition(belief, after, (), after) is belief
