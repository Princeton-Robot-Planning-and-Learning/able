"""Belief updates for the person's reach."""

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import ReachAction, ReachState
from able.envs.human_reach.query import REACH_SCALE, ReachQuery, ReachResponse
from able.envs.human_reach.scene import reach_margin
from able.structs import QueryResponse


def reached_log_likelihood(
    points: ArrayLike,
    reached: ArrayLike,
    particles: NDArray[np.float64],
    shoulder: NDArray[np.float64],
    scale: float = REACH_SCALE,
) -> NDArray[np.float64]:
    """The log likelihood of each particle: sum over points of log sigmoid(+-margin
    / scale), with the sign set by whether the point was reached."""
    m = reach_margin(np.asarray(points, float)[:, None], particles[None], shoulder)
    z = np.where(np.asarray(reached, bool)[:, None], m, -m) / scale
    return -np.logaddexp(0.0, -z).sum(0)


# The chance the robot's record of a delivery disagrees with the true reach: P(reached
# | theta) is 1 - DELIVERY_LAPSE if theta reaches the item and DELIVERY_LAPSE if not.
DELIVERY_LAPSE = 0.01


def delivered_log_likelihood(
    point: ArrayLike,
    reached: bool,
    particles: NDArray[np.float64],
    shoulder: NDArray[np.float64],
    lapse: float = DELIVERY_LAPSE,
) -> NDArray[np.float64]:
    """The log likelihood of each particle for one delivery: log(1 - lapse) if the
    particle agrees with whether the person reached the item, else log(lapse).

    A person reaches a delivered item exactly when it is within their reach, so
    every particle that agrees with the outcome is equally likely, however far
    inside its reach the item lies.
    """
    m = reach_margin(np.asarray(point, float)[None], particles, shoulder)
    agrees = (m >= 0) == bool(reached)
    return np.where(agrees, np.log1p(-lapse), np.log(lapse))


class ReachBeliefUpdater(
    BeliefUpdater[
        ReachState, ReachAction, NDArray[np.float64], ReachQuery, ReachResponse
    ]
):
    """Reweights reach particles by how likely each makes what the person did, and
    resamples.

    Each delivered item is one update, after the action that placed it, weighed by
    delivered_log_likelihood: the person reaches an item exactly when it is within
    their reach, up to delivery_lapse. Each query response is one update with all of
    its points, weighed by P(reached x | theta) = sigmoid(margin(x, theta) / scale),
    since the person answers a query noisily near the edge of their reach.

    Weighing deliveries by the sigmoid would favor longer reaches after every item
    the person reaches, because it expects a person whose reach barely covers an
    item to miss it sometimes, and they never do. The belief then overestimates
    the reach until the robot places an item out of it.
    """

    def __init__(
        self,
        shoulder: NDArray[np.float64],
        scale: float = REACH_SCALE,
        delivery_lapse: float = DELIVERY_LAPSE,
    ):
        self._shoulder = np.asarray(shoulder, dtype=float)
        self._scale = scale
        self._delivery_lapse = delivery_lapse

    def update_from_transition(
        self,
        belief: Belief[NDArray[np.float64]],
        state: ReachState,
        action: ReachAction,
        next_state: ReachState,
    ) -> Belief[NDArray[np.float64]]:
        delivery = next_state.delivery
        if next_state.num_actions == state.num_actions or delivery is None:
            return belief
        _, point, reached = delivery
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        log_weights = delivered_log_likelihood(
            point, reached, belief.particles, self._shoulder, self._delivery_lapse
        )
        return belief.resample(log_weights)

    def update_from_queries(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[QueryResponse[ReachQuery, ReachResponse]],
    ) -> Belief[NDArray[np.float64]]:
        for datum in data:
            points = [point for point, _ in datum.response]
            reached = [answer for _, answer in datum.response]
            belief = self._update(belief, points, reached)
        return belief

    def _update(
        self,
        belief: Belief[NDArray[np.float64]],
        points: list[tuple[float, float, float]],
        reached: list[bool],
    ) -> ParticleBelief:
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        if not points:
            log_weights = np.zeros(len(belief.particles))
        else:
            log_weights = reached_log_likelihood(
                points, reached, belief.particles, self._shoulder, self._scale
            )
        return belief.resample(log_weights)
