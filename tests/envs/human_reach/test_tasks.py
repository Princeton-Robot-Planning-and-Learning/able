"""Tests for envs/human_reach/tasks.py."""

import numpy as np

from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import ReachModel
from able.envs.human_reach.tasks import ReachTaskDistribution, make_domain
from able.envs.simulated import SimulatedEnvironment


def _get_parts(domain):
    model, real_env = domain.env_model, domain.real_env
    belief, distribution = domain.initial_belief, domain.task_distribution
    assert isinstance(model, ReachModel)
    assert isinstance(real_env, SimulatedEnvironment)
    assert isinstance(belief, ParticleBelief)
    assert isinstance(distribution, ReachTaskDistribution)
    return model, real_env, belief, distribution


def _close(model: ReachModel) -> None:
    model.close()


def test_make_domain():
    """The person's reach and the belief's particles lie in the prior box, and
    scenes differ by seed."""
    model, real_env, belief, distribution = _get_parts(
        make_domain(seed=0, num_requested=2, num_clutter=2, num_particles=100)
    )
    low, high = belief.low, belief.high
    assert np.all((low <= real_env.hidden_params) & (real_env.hidden_params <= high))
    assert belief.particles.shape == (100, 3)
    assert np.all((low <= belief.particles) & (belief.particles <= high))
    rng = np.random.default_rng(0)
    first, second = distribution.sample(rng), distribution.sample(rng)
    assert first.goal == ("req0", "req1") and first.initial_state.num_actions == 0
    assert model.horizon == 40
    poses = [
        [t.initial_state.scene.get_object_pose(c).position for c in ("clutter0",)]
        for t in (first, second)
    ]
    assert not np.allclose(poses[0], poses[1])
    _close(model)


def test_given_reach():
    """A given reach overrides the prior."""
    model, real_env, _, _ = _get_parts(
        make_domain(seed=0, num_requested=2, num_clutter=0, true_theta=[0.7, 0.6, 0.65])
    )
    assert np.allclose(real_env.hidden_params, [0.7, 0.6, 0.65])
    _close(model)
