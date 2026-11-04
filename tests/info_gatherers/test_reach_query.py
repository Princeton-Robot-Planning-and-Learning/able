"""Tests for info_gatherers/reach_query.py."""

import numpy as np

from able.belief_updaters.human_reach import ReachBeliefUpdater
from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import ReachModel
from able.envs.human_reach.query import reach_distance
from able.envs.human_reach.tasks import make_domain
from able.info_gatherers.reach_query import ReachQueryGatherer
from able.structs import Budget

THETA = np.array([0.5, 0.75, 0.85])  # away from the prior mean


def _get_shoulder(domain) -> np.ndarray:
    model = domain.env_model
    assert isinstance(model, ReachModel)
    return model.sim.shoulder


def _particles(belief) -> np.ndarray:
    assert isinstance(belief, ParticleBelief)
    return belief.particles


def test_queries_narrow_the_belief_toward_the_reach():
    """Five queries narrow the reach along the first query's direction and
    everywhere on average."""
    domain = make_domain(seed=0, num_requested=2, num_clutter=0, true_theta=THETA)
    updater = ReachBeliefUpdater(_get_shoulder(domain))
    gatherer = ReachQueryGatherer(updater)
    belief = domain.initial_belief
    assert isinstance(belief, ParticleBelief)
    first = gatherer.choose(belief)
    assert first is not None and np.all(np.diff(first.distances) > 0)
    data = gatherer.gather(
        belief, domain.real_env, Budget(max_iterations=5), np.random.default_rng(0)
    )
    assert len(data) == 5 and all(d.response for d in data)
    posterior = updater.update_from_queries(belief, data)
    dirs = gatherer.directions

    def reach_error(particles):
        r = reach_distance(dirs[:, None], particles[None])
        return np.abs(r - reach_distance(dirs, THETA)[:, None]).mean()

    before, after = _particles(belief), _particles(posterior)
    r_before = reach_distance(first.direction, before)
    r_after = reach_distance(first.direction, after)
    assert r_after.std() < r_before.std() / 3
    assert abs(r_after.mean() - reach_distance(first.direction, THETA)) < 0.03
    assert reach_error(after) < reach_error(before) / 2
    gatherer.close()


def test_no_query_when_the_reach_is_known():
    """Nothing is asked when every answer is certain, or with no budget."""
    domain = make_domain(seed=0, num_requested=2, num_clutter=0)
    gatherer = ReachQueryGatherer(ReachBeliefUpdater(_get_shoulder(domain)))
    low, high = np.array([0.45, 0.35, 0.40]), np.array([0.95, 0.80, 0.90])
    known = ParticleBelief(np.tile(THETA, (50, 1)), low, high)
    assert gatherer.choose(known) is None
    rng = np.random.default_rng(0)
    assert not gatherer.gather(known, domain.real_env, Budget(max_iterations=3), rng)
    belief = domain.initial_belief
    assert not gatherer.gather(belief, domain.real_env, Budget(max_iterations=0), rng)
    gatherer.close()
