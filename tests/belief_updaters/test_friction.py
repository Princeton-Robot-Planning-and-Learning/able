"""Tests for belief_updaters/friction.py."""

import math

import numpy as np
import pytest

from able.belief_updaters.friction import FrictionBeliefUpdater, push_log_likelihood
from able.beliefs.particles import ParticleBelief
from able.envs.friction.layout import make_world
from able.envs.friction.model import FrictionState
from able.envs.friction.query import ProbeSimulator, PushRecord, probe_queries
from able.envs.friction.tasks import make_prior_belief
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, GRAVITY, Push
from able.structs import QueryResponse

WORLD = make_world()


def _prior(num_particles: int = 100, seed: int = 0) -> ParticleBelief:
    return make_prior_belief(
        DEFAULT_FRICTION_PRIOR, num_particles, np.random.default_rng(seed)
    )


def _record(puck, push: Push, release_speed: float, stop, fell=False) -> PushRecord:
    return PushRecord(
        puck=(float(puck[0]), float(puck[1])),
        push=push,
        feasible=True,
        release_speed=release_speed,
        stop=(float(stop[0]), float(stop[1])),
        fell_in_hole=fell,
        goal_reached=False,
        drive_distance=0.0,
    )


def _transition(record: PushRecord) -> tuple[FrictionState, FrictionState]:
    state = FrictionState(record.puck, (0.0, 0.0), record.puck, (0.0, 0.0, 0.0))
    next_state = FrictionState(
        record.stop, (0.0, 0.0), record.stop, (0.0, 0.0, 0.0), 1, record
    )
    return state, next_state


def test_push_likelihood_matches_the_math():
    """A clear straight slide along the blue corridor, by hand."""
    phys = WORLD.physics
    puck, speed = np.array([1.0, 3.4]), 0.5
    stop = np.array([1.45, 3.41])
    particles = np.array([[0.10, 0.08, 0.03], [0.06, 0.08, 0.03]])
    record = _record(puck, Push(0.0, speed), speed, stop)
    got = push_log_likelihood(WORLD, particles, record)

    def by_hand(mu):
        length = speed**2 / (2 * mu * GRAVITY)
        along = stop[0] - (puck[0] + phys.accel_distance + length)
        across = stop[1] - puck[1]
        var_along = (
            phys.position_obs_noise**2
            + (2 * length * phys.speed_obs_noise / speed) ** 2
        )
        var_across = phys.position_obs_noise**2 + (length * phys.angle_noise) ** 2
        return -0.5 * (
            along**2 / var_along + across**2 / var_across
        ) - 0.5 * math.log(var_along * var_across)

    assert got == pytest.approx([by_hand(0.10), by_hand(0.06)], abs=1e-9)
    fell = _record(puck, Push(0.0, speed), speed, stop, fell=True)
    assert push_log_likelihood(WORLD, particles, fell) == pytest.approx(
        got + math.log(0.05)
    )
    belief = _prior()
    state, next_state = _transition(record)
    posterior = FrictionBeliefUpdater(WORLD).update_from_transition(
        belief, state, Push(0.0, speed), next_state
    )
    assert isinstance(posterior, ParticleBelief)
    assert posterior.std[0] < belief.std[0]
    assert np.all(posterior.particles >= belief.low)
    assert np.all(posterior.particles <= belief.high)


def test_pushes_concentrate_the_belief():
    """Pushes on tan narrow its coefficient around the truth."""
    updater = FrictionBeliefUpdater(WORLD)
    belief = _prior(num_particles=400)
    prior_std = belief.std.copy()
    true = (0.04, 0.06, 0.15)
    puck = np.array([1.0, 2.0])  # tan
    for angle, speed in [(math.pi / 2, 1.0), (-math.pi / 2, 0.4)]:
        result = WORLD.slide(WORLD.release_point(puck, angle), speed, angle, true)
        record = _record(
            puck, Push(angle, speed), speed, result.stop, result.hole is not None
        )
        belief = updater.update_from_transition(
            belief, *_transition(record)[:1], Push(angle, speed), _transition(record)[1]
        )
        puck = result.stop
    assert isinstance(belief, ParticleBelief)
    assert belief.std[0] < prior_std[0] and belief.std[1] < 0.5 * prior_std[1]
    assert abs(belief.mean[1] - true[1]) < 0.01


def test_three_probes_narrow_each_floor():
    """One probe per floor narrows every coefficient toward the truth."""
    rng = np.random.default_rng(1)
    frictions = np.array([0.15, 0.03, 0.08])
    simulator = ProbeSimulator("2d")
    data = [
        QueryResponse(query, simulator.ask(frictions, query, rng))
        for query in probe_queries(WORLD, DEFAULT_FRICTION_PRIOR.bounds[0])
    ]
    before = _prior(seed=1)
    belief = FrictionBeliefUpdater(WORLD).update_from_queries(before, data)
    assert isinstance(belief, ParticleBelief)
    assert np.all(belief.std < 0.3 * before.std)
    assert np.all(np.abs(belief.mean - frictions) < 0.2 * frictions)
    simulator.close()


def test_steps_without_a_new_feasible_push_show_nothing():
    """Infeasible pushes and steps after the task ended leave the belief."""
    updater = FrictionBeliefUpdater(WORLD)
    belief = _prior()
    record = _record((1.0, 3.4), Push(0.0, 0.5), 0.5, (1.45, 3.41))
    state, next_state = _transition(record)
    assert updater.update_from_transition(
        belief, next_state, Push(0, 0), next_state
    ) is (belief)
    infeasible = PushRecord(
        record.puck, record.push, False, 0.0, record.puck, False, False, 0.0
    )
    _, blocked = _transition(infeasible)
    assert updater.update_from_transition(belief, state, record.push, blocked) is belief
