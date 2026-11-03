"""Tests for info_gatherers/probe_floor.py."""

import numpy as np

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.beliefs.particles import ParticleBelief
from able.envs.friction.levels import level_world
from able.envs.friction.tasks import make_domain
from able.info_gatherers.probe_floor import ProbeFloorGatherer
from able.structs import Budget

WORLD = level_world("L2")


def _prior(domain) -> ParticleBelief:
    belief = domain.initial_belief
    assert isinstance(belief, ParticleBelief)
    return belief


def _least_certain(belief) -> int:
    return int(np.argmax(belief.std))


def test_probes_the_least_certain_floor_each_time():
    """Each query probes the floor that is least certain after the earlier ones."""
    domain = make_domain(seed=0, num_particles=100)
    updater = FrictionBeliefUpdater(WORLD)
    gatherer = ProbeFloorGatherer(WORLD, updater)
    belief = _prior(domain)
    data = gatherer.gather(
        belief, domain.real_env, Budget(max_iterations=3), np.random.default_rng(0)
    )
    assert len(data) == 3
    for datum in data:
        assert datum.query.floor == _least_certain(belief)
        assert datum.response.feasible and datum.response.puck == datum.query.puck
        belief = updater.update_from_queries(belief, [datum])


def test_stops_when_every_floor_is_known():
    """No probe is asked when every floor is already narrow, or with no budget."""
    domain = make_domain(seed=0, num_particles=100)
    gatherer = ProbeFloorGatherer(WORLD, FrictionBeliefUpdater(WORLD))
    known = ParticleBelief(np.full((10, 3), 0.08), [0.024] * 3, [0.18] * 3)
    rng = np.random.default_rng(0)
    assert not gatherer.gather(known, domain.real_env, Budget(max_iterations=3), rng)
    belief = _prior(domain)
    assert not gatherer.gather(belief, domain.real_env, Budget(max_iterations=0), rng)
