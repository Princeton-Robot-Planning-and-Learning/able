"""Belief updates for the floors' friction coefficients."""

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.friction.model import FrictionState
from able.envs.friction.query import FrictionQuery, PushRecord
from able.envs.friction.world import PuckWorld, Push
from able.structs import QueryResponse

# Likelihood factor when a particle's predicted hole outcome disagrees.
_HOLE_MISMATCH = 0.05


def push_log_likelihood(
    world: PuckWorld, particles: NDArray[np.float64], record: PushRecord
) -> NDArray[np.float64]:
    """The log likelihood of a push's measurements under each particle.

    Each particle slides the puck at the measured release speed. The stop error is
    Gaussian along and across the slide's final heading, and the hole flag must
    agree.
    """
    phys = world.physics
    puck, angle = np.array(record.puck), record.push.angle
    release = world.release_point(puck, angle)
    station = world.push_station(puck, angle, record.push.stance)
    robot = world.strike_end(station, puck, angle)
    speed = max(record.release_speed, 1e-3)
    slides = world.slide_many(
        release, record.release_speed, angle, particles, robot=robot
    )
    along_dir = np.stack([np.cos(slides.headings), np.sin(slides.headings)], axis=1)
    error = np.array(record.stop) - slides.stops
    along = np.einsum("ij,ij->i", error, along_dir)
    across = error[:, 0] * along_dir[:, 1] - error[:, 1] * along_dir[:, 0]
    length = slides.distances
    var_along = (
        phys.position_obs_noise**2 + (2.0 * length * phys.speed_obs_noise / speed) ** 2
    )
    var_across = phys.position_obs_noise**2 + (length * phys.angle_noise) ** 2
    out = -0.5 * (along**2 / var_along + across**2 / var_across) - 0.5 * np.log(
        var_along * var_across
    )
    mismatch = (slides.holes >= 0) != record.fell_in_hole
    out[mismatch] += math.log(_HOLE_MISMATCH)
    return out


class FrictionBeliefUpdater(
    BeliefUpdater[FrictionState, Push, NDArray[np.float64], FrictionQuery, PushRecord]
):
    """Reweights friction particles by how well they explain each push, and
    resamples.

    A probe is a real push too, so queries and executed pushes share the push
    model. An infeasible push, or a step after the task ended, shows nothing.
    """

    def __init__(self, world: PuckWorld) -> None:
        self._world = world

    def update_from_transition(
        self,
        belief: Belief[NDArray[np.float64]],
        state: FrictionState,
        action: Push,
        next_state: FrictionState,
    ) -> Belief[NDArray[np.float64]]:
        record = next_state.last_push
        if next_state.pushes == state.pushes or record is None or not record.feasible:
            return belief
        return self._update(belief, record)

    def update_from_queries(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[QueryResponse[FrictionQuery, PushRecord]],
    ) -> Belief[NDArray[np.float64]]:
        for datum in data:
            belief = self._update(belief, datum.response)
        return belief

    def _update(
        self, belief: Belief[NDArray[np.float64]], record: PushRecord
    ) -> ParticleBelief:
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        log_weights = push_log_likelihood(self._world, belief.particles, record)
        return belief.resample(log_weights)
