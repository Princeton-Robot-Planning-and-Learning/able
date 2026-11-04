"""Tasks and domains for the human reach environment."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.beliefs.particles import ParticleBelief
from able.envs.base import Domain, TaskDistribution
from able.envs.human_reach.model import (
    ReachAction,
    ReachModel,
    ReachState,
    ReachTask,
)
from able.envs.human_reach.query import ReachQuery, ReachResponse
from able.envs.human_reach.scene import BedsideStocking3DEnvConfig
from able.envs.simulated import SimulatedEnvironment

ReachDomain = Domain[
    ReachState,
    ReachAction,
    ReachTask,
    NDArray[np.float64],
    ReachQuery,
    ReachResponse,
]


class ReachTaskDistribution(TaskDistribution[ReachTask]):
    """Scenes from random seeds, which place the small requested items and the
    clutter. The goal is to deliver every requested item."""

    def __init__(self, model: ReachModel) -> None:
        self._model = model

    def sample(self, rng: np.random.Generator) -> ReachTask:
        state = self._model.get_initial_state(int(rng.integers(2**31 - 1)))
        return ReachTask(state, tuple(state.scene.requested_objects))


def make_prior_belief(
    config: BedsideStocking3DEnvConfig, num_particles: int, rng: np.random.Generator
) -> ParticleBelief:
    """Particles drawn uniformly from the config's box of reaches."""
    low, high = np.array(config.reach_theta_lb), np.array(config.reach_theta_ub)
    return ParticleBelief(rng.uniform(low, high, (num_particles, len(low))), low, high)


def make_domain(
    seed: int,
    num_requested: int = 2,
    num_clutter: int = 4,
    horizon: int = 40,
    num_particles: int = 2000,
    config: BedsideStocking3DEnvConfig = BedsideStocking3DEnvConfig(),
    true_theta: ArrayLike | None = None,
) -> ReachDomain:
    """A human reach domain with episodes of horizon actions, and a person whose
    reach is drawn uniformly from the config's box unless given. The robot's
    initial belief is particles from the same box."""
    rng = np.random.default_rng(seed)
    low, high = np.array(config.reach_theta_lb), np.array(config.reach_theta_ub)
    theta = rng.uniform(low, high) if true_theta is None else true_theta
    model = ReachModel(num_requested, num_clutter, config, horizon)
    return Domain(
        env_model=model,
        task_distribution=ReachTaskDistribution(model),
        initial_belief=make_prior_belief(config, num_particles, rng),
        real_env=SimulatedEnvironment(model, np.asarray(theta, np.float64), rng),
    )
