"""Tasks and domains for the friction environment."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from able.beliefs.particles import ParticleBelief
from able.envs.base import Domain, TaskDistribution
from able.envs.friction.model import (
    FrictionModel,
    FrictionState,
    FrictionTask,
    PuckRewards,
)
from able.envs.friction.query import FrictionQuery, PushRecord
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, FrictionPrior, Push
from able.envs.simulated import SimulatedEnvironment

FrictionDomain = Domain[
    FrictionState,
    Push,
    FrictionTask,
    NDArray[np.float64],
    FrictionQuery,
    PushRecord,
]


class FrictionTaskDistribution(TaskDistribution[FrictionTask]):
    """Every task pushes the puck from the map's start to its goal.

    Tasks differ only in the execution and measurement noise of their episodes,
    so each sample is a new task with the same start and goal.
    """

    def __init__(self, model: FrictionModel) -> None:
        self._model = model

    def sample(self, rng: np.random.Generator) -> FrictionTask:
        del rng  # unused because the start and goal are fixed
        goal = self._model.world.puck_map.goal
        return FrictionTask(self._model.get_initial_state(), (goal[0], goal[1]))


def make_prior_belief(
    prior: FrictionPrior, num_particles: int, rng: np.random.Generator
) -> ParticleBelief:
    """Particles drawn from the prior, with a resampling seed from the stream."""
    low, high = prior.bounds
    particles = prior.sample(rng, num_particles)
    return ParticleBelief(particles, low, high, seed=int(rng.integers(2**31)))


def make_domain(
    seed: int,
    sim: str = "2d",
    num_holes: int = 2,
    goal_radius: float = 0.3,
    horizon: int = 15,
    num_particles: int = 500,
    prior: FrictionPrior = DEFAULT_FRICTION_PRIOR,
    rewards: PuckRewards = PuckRewards(),
    true_frictions: ArrayLike | None = None,
) -> FrictionDomain:
    """A friction domain on the maze with num_holes holes and the goal radius, with
    floor frictions drawn from the prior unless given. The robot's initial belief
    is particles from the prior."""
    rng = np.random.default_rng(seed)
    frictions = prior.sample(rng) if true_frictions is None else true_frictions
    model = FrictionModel(sim, num_holes, goal_radius, horizon, rewards)
    return Domain(
        env_model=model,
        task_distribution=FrictionTaskDistribution(model),
        initial_belief=make_prior_belief(prior, num_particles, rng),
        real_env=SimulatedEnvironment(model, np.asarray(frictions, np.float64), rng),
    )
