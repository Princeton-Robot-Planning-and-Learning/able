"""Belief updates for the person's reach."""

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import ReachPlan, ReachState
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


class ReachBeliefUpdater(
    BeliefUpdater[ReachState, ReachPlan, NDArray[np.float64], ReachQuery, ReachResponse]
):
    """Reweights reach particles by P(reached x | theta) = sigmoid(margin(x, theta) /
    scale), and resamples.

    Queries and deliveries share this model. Each delivered item is one update, and
    each query response is one update with all of its points.
    """

    def __init__(self, shoulder: NDArray[np.float64], scale: float = REACH_SCALE):
        self._shoulder = np.asarray(shoulder, dtype=float)
        self._scale = scale

    def update_from_transition(
        self,
        belief: Belief[NDArray[np.float64]],
        state: ReachState,
        action: ReachPlan,
        next_state: ReachState,
    ) -> Belief[NDArray[np.float64]]:
        if state.executed or not next_state.executed:
            return belief
        for _, point, reached in next_state.deliveries:
            belief = self._update(belief, [point], [reached])
        return belief

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
