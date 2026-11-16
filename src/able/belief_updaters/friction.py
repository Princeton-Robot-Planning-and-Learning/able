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
from able.envs.friction.world import PuckWorld, Push, SlideBatch
from able.structs import QueryResponse

# Likelihood factor when a particle's predicted hole outcome disagrees.
_HOLE_MISMATCH = 0.05
# How far short of the paddle's travel the 2D simulator releases the puck (m): its
# paddle starts 5 mm behind the puck. Calibrated against the simulator (#152).
RELEASE_SHORTFALL = 0.004


def push_log_likelihood(
    world: PuckWorld, particles: NDArray[np.float64], record: PushRecord
) -> NDArray[np.float64]:
    """The log likelihood of a push's measurements under each particle.

    Each particle slides the puck at the measured release speed along the commanded
    heading. The stop error is Gaussian along and across the slide's final heading. Its
    variance is the stop's measurement noise, the start's if the start was a
    measurement, and the spread from the strike's heading noise and the release speed's
    measurement noise. Each spread is how far the stop moves when the slide starts one
    standard deviation off (a sigma point), which follows bounces and floor changes. The
    hole flag must agree.
    """
    phys = world.physics
    puck, push = np.array(record.puck), record.push
    station = world.push_station(puck, push.angle, push.stance)

    def stops(angle: float, speed: float) -> SlideBatch:
        direction = np.array([math.cos(angle), math.sin(angle)])
        release = world.release_point(puck, angle) - RELEASE_SHORTFALL * direction
        robot = world.strike_end(station, puck, angle)
        return world.slide_many(release, max(speed, 0.0), angle, particles, robot=robot)

    slides = stops(push.angle, record.release_speed)
    along_dir = np.stack([np.cos(slides.headings), np.sin(slides.headings)], axis=1)
    measurements = 2 if record.start_measured else 1
    var_along = np.full(len(particles), measurements * phys.position_obs_noise**2)
    var_across = var_along.copy()
    for d_angle, d_speed in ((phys.angle_noise, 0.0), (0.0, phys.speed_obs_noise)):
        moved = stops(push.angle + d_angle, record.release_speed + d_speed).stops
        spread = moved - slides.stops
        var_along += np.einsum("ij,ij->i", spread, along_dir) ** 2
        var_across += (
            spread[:, 0] * along_dir[:, 1] - spread[:, 1] * along_dir[:, 0]
        ) ** 2
    error = np.array(record.stop) - slides.stops
    along = np.einsum("ij,ij->i", error, along_dir)
    across = error[:, 0] * along_dir[:, 1] - error[:, 1] * along_dir[:, 0]
    out = -0.5 * (along**2 / var_along + across**2 / var_across) - 0.5 * np.log(
        var_along * var_across
    )
    mismatch = (slides.holes >= 0) != record.fell_in_hole
    out[mismatch] += math.log(_HOLE_MISMATCH)
    return out


def crossed_floors(
    world: PuckWorld, record: PushRecord, lowest: NDArray[np.float64]
) -> list[int]:
    """The floors under a push's slide, as far as the lowest frictions would carry it.

    Every slide follows the same path and stops along it (see slide_many), so the push
    says nothing about the other floors.
    """
    puck, angle = np.array(record.puck), record.push.angle
    release = world.release_point(puck, angle)
    station = world.push_station(puck, angle, record.push.stance)
    robot = world.strike_end(station, puck, angle)
    slide = world.slide(release, record.release_speed, angle, lowest, robot=robot)
    floors = {world.surface_at(release)}
    for a, b in zip(slide.path[:-1], slide.path[1:]):
        for t in np.linspace(0.0, 1.0, int(np.linalg.norm(b - a) / 0.02) + 2):
            floors.add(world.surface_at(a + t * (b - a)))
    return sorted(floors - {-1})


class FrictionBeliefUpdater(
    BeliefUpdater[FrictionState, Push, NDArray[np.float64], FrictionQuery, PushRecord]
):
    """Reweights friction particles by how well they explain each push, and resamples
    the floors the push crossed, in steps if the push is sharp.

    A probe is a real push too, so queries and executed pushes share the push model. An
    infeasible push, or a step after the task ended, shows nothing.
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
        floors = crossed_floors(self._world, record, belief.low)
        return belief.reweight(
            lambda particles: push_log_likelihood(self._world, particles, record),
            floors,
        )
