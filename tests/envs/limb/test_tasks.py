"""Tests for envs/limb/tasks.py."""

import numpy as np
import pytest

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import VARIANTS, RangeOfMotionPrior
from able.envs.limb.model import LimbModel
from able.envs.limb.tasks import (
    LimbTaskDistribution,
    TaskBox,
    load_fixed_setup,
    make_domain,
)
from able.envs.simulated import SimulatedEnvironment


def _close(domain) -> None:
    distribution = domain.task_distribution
    assert isinstance(distribution, LimbTaskDistribution)
    distribution.close()
    model = domain.env_model
    assert isinstance(model, LimbModel)
    model.close()


def _get_true_theta(domain) -> np.ndarray:
    real_env = domain.real_env
    assert isinstance(real_env, SimulatedEnvironment)
    return real_env.hidden_params


@pytest.fixture(name="domain", scope="module")
def _domain():
    domain = make_domain(seed=0, horizon=20)
    yield domain
    _close(domain)


def test_fixed_setups():
    """Tests that every variant has a fixed setup."""
    for variant in VARIANTS:
        setup = load_fixed_setup(variant)
        assert setup.variant == variant
        assert setup.base_pose.x == setup.base[0]
    with pytest.raises(KeyError):
        load_fixed_setup("sofa-third-arm")


def test_make_domain(domain):
    """Tests the person, the initial belief, and the real environment."""
    prior = RangeOfMotionPrior()
    theta = _get_true_theta(domain)
    assert theta.shape == (3,)
    assert np.all((prior.low <= theta) & (theta <= prior.high))
    assert domain.initial_belief == RangeOfMotionBelief.from_prior(
        prior.mean, prior.stddev, prior.low, prior.high, 3
    )
    same = make_domain(seed=0)
    given = make_domain(seed=0, true_theta=np.array([0.8, 0.9, 1.0]))
    assert np.array_equal(_get_true_theta(same), theta)
    assert np.array_equal(_get_true_theta(given), [0.8, 0.9, 1.0])
    _close(same)
    _close(given)


def test_sampled_tasks(domain):
    """Tests that tasks start grasped, inside the box, and far enough from the goal,
    and that the robot holds the limb still from the start."""
    model = domain.env_model
    assert isinstance(model, LimbModel)
    box = TaskBox()
    lower, upper = model.variant.get_joint_limits(np.full(3, box.scale))
    rng = np.random.default_rng(0)
    tasks = [domain.task_distribution.sample(rng) for _ in range(2)]
    assert not np.allclose(tasks[0].goal, tasks[1].goal)
    for task in tasks:
        start = task.initial_state
        assert start.coupled.grasp_frame is not None
        for positions in (start.limb_positions, task.goal):
            active = list(model.variant.active_joints)
            assert np.all(lower[active] <= positions[active])
            assert np.all(positions[active] <= upper[active])
        separation = model.get_goal_error(task, start)
        assert box.min_separation <= separation <= box.max_separation

        state = domain.real_env.reset(task)
        assert state is start
        for _ in range(10):
            state = domain.real_env.step(np.zeros(7)).state
        assert not state.failure
        assert (
            model.sim.get_limb_error(state.limb_positions, start.limb_positions) < 0.01
        )
        assert model.get_goal_error(task, state) == pytest.approx(separation, abs=0.01)
