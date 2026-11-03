"""Tests for planners/limb_mpc.py."""

import dataclasses

import numpy as np
import pytest

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import RangeOfMotionPrior, get_variant
from able.envs.limb.model import LimbModel, LimbTask
from able.envs.limb.tasks import LimbTaskDistribution, make_domain
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.planners.limb_mpc import (
    LimbMPCConfig,
    LimbMPCPlanner,
    MPCParams,
    ParticleLimits,
    retarget,
)
from able.planners.predictive_sampling import PredictiveSamplingConfig
from able.structs import Budget, PlanningFailure

VARIANT = get_variant("human-left-arm")
# Few rollouts, for speed.
FAST = MPCParams(
    config=LimbMPCConfig(sampling=PredictiveSamplingConfig(num_rollouts=8)),
    num_particles=8,
)


def _prior() -> RangeOfMotionBelief:
    prior = RangeOfMotionPrior()
    return RangeOfMotionBelief.from_prior(
        prior.mean, prior.stddev, prior.low, prior.high, 3
    )


@pytest.fixture(name="setup", scope="module")
def _setup():
    domain = make_domain(seed=0, horizon=40)
    model = domain.env_model
    assert isinstance(model, LimbModel)
    distribution = domain.task_distribution
    assert isinstance(distribution, LimbTaskDistribution)
    sampled = distribution.sample(np.random.default_rng(0))
    # A nearby goal: bend the elbow a little more.
    goal = sampled.initial_state.limb_positions.copy()
    goal[VARIANT.active_joints[2]] += 0.15 * VARIANT.active_signs[2]
    task = LimbTask(initial_state=sampled.initial_state, goal=goal)
    planner = LimbMPCPlanner(model)
    yield domain, task, planner
    planner.close()
    _close(distribution, model)


def _close(distribution: LimbTaskDistribution, model: LimbModel) -> None:
    distribution.close()
    model.close()


def test_particle_limits():
    """Tests that a configuration is inside if enough particles allow it."""
    particles = np.array([[0.75, 0.9, 0.9], [0.85, 0.9, 0.9], [1.0, 0.9, 0.9]])
    joint, sign = VARIANT.active_joints[0], VARIANT.active_signs[0]
    positions = np.zeros(6)
    positions[joint] = sign * 0.8 * VARIANT.default_magnitudes[0]
    assert ParticleLimits.from_particles(VARIANT, particles, 0.5)(positions)
    assert not ParticleLimits.from_particles(VARIANT, particles, 0.9)(positions)
    limits = ParticleLimits.from_particles(VARIANT, particles, 1.0)
    assert np.isclose(limits.get_quantile_theta()[0], 0.75)


def test_retarget():
    """Tests that a goal beyond the believed limits moves just inside them."""
    particles = np.stack([_prior().sample(np.random.default_rng(i)) for i in range(16)])
    limits = ParticleLimits.from_particles(VARIANT, particles, 0.9)
    joint, sign = VARIANT.active_joints[0], VARIANT.active_signs[0]

    def goal_at(scale: float) -> np.ndarray:
        goal = np.zeros(6)
        goal[joint] = sign * scale * VARIANT.default_magnitudes[0]
        return goal

    inside = goal_at(0.5)
    assert retarget(inside, limits, VARIANT) == (inside, False)
    beyond = goal_at(1.1)
    moved, retargeted = retarget(beyond, limits, VARIANT)
    assert retargeted and limits(moved)
    assert abs(moved[joint]) < abs(beyond[joint])
    assert np.array_equal(np.delete(moved, joint), np.delete(beyond, joint))


def test_plan_reaches_the_goal(setup):
    """Tests that the plan reaches the goal in the real environment and holds it
    there to the end of the horizon."""
    domain, task, planner = setup
    policy = planner.plan(task, _prior(), FAST, Budget(), np.random.default_rng(0))
    stats = policy.get_planning_stats()
    assert stats["num_mpc_steps"] > 0 and stats["num_rollouts"] > 0
    assert stats["retargeted"] == 0.0
    assert stats["replay_error"] < 0.1
    assert len(policy.actions) == domain.env_model.horizon
    real_env = domain.real_env
    assert isinstance(real_env, SimulatedEnvironment)
    trajectory = run_episode(real_env, task, policy)
    assert real_env.check_goal(task, trajectory.states[-1])


def test_planning_failures(setup):
    """Tests that running out of budget, or a goal the belief rules out entirely,
    fails."""
    _, task, planner = setup
    rng = np.random.default_rng(0)
    with pytest.raises(PlanningFailure, match="budget"):
        planner.plan(task, _prior(), FAST, Budget(max_iterations=0), rng)
    short = dataclasses.replace(
        FAST, config=dataclasses.replace(FAST.config, divergence_patience=0)
    )
    with pytest.raises(PlanningFailure, match="Diverged"):
        planner.plan(task, _prior(), short, Budget(), rng)
