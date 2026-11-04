"""Tests for envs/friction/tasks.py."""

import numpy as np

from able.beliefs.particles import ParticleBelief
from able.envs.friction.model import FrictionModel
from able.envs.friction.tasks import FrictionTaskDistribution, make_domain
from able.envs.friction.world import DEFAULT_FRICTION_PRIOR, Push
from able.envs.simulated import SimulatedEnvironment


def _get_parts(domain):
    model, real_env = domain.env_model, domain.real_env
    belief, distribution = domain.initial_belief, domain.task_distribution
    assert isinstance(model, FrictionModel)
    assert isinstance(real_env, SimulatedEnvironment)
    assert isinstance(belief, ParticleBelief)
    assert isinstance(distribution, FrictionTaskDistribution)
    return model, real_env, belief, distribution


def test_make_domain():
    """The person's frictions come from the prior, the belief is prior particles,
    and every task starts at the map's start."""
    model, real_env, belief, distribution = _get_parts(
        make_domain(seed=0, num_particles=100)
    )
    ratios = np.sort(real_env.hidden_params) / np.array(DEFAULT_FRICTION_PRIOR.levels)
    assert np.all((0.8 - 1e-9 <= ratios) & (ratios <= 1.2 + 1e-9))
    assert belief.particles.shape == (100, 3)
    assert np.allclose(belief.low, DEFAULT_FRICTION_PRIOR.bounds[0])
    rng = np.random.default_rng(0)
    first, second = distribution.sample(rng), distribution.sample(rng)
    assert first is not second and first.initial_state == second.initial_state
    assert first.goal == model.world.puck_map.goal
    state = real_env.reset(first)
    result = real_env.step(Push(np.pi / 2, 0.5))
    assert result.state.pushes == 1 and not result.terminated
    assert state.pushes == 0
    model.close()


def test_seeds_and_given_frictions():
    """Seeds draw different people, and given frictions override the prior."""
    frictions = [_get_parts(make_domain(seed=s))[1].hidden_params for s in range(3)]
    assert not np.allclose(frictions[0], frictions[1])
    domain = make_domain(seed=0, num_holes=0, true_frictions=[0.1, 0.05, 0.03])
    assert np.allclose(_get_parts(domain)[1].hidden_params, [0.1, 0.05, 0.03])
