"""Tests for planners/mpc.py."""

import dataclasses

import numpy as np
import pytest

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import RangeOfMotionPrior, get_variant
from able.envs.limb.model import LimbModel, LimbTask
from able.envs.limb.tasks import LimbTaskDistribution, make_domain
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.planners.mpc import (
    LimbMPCConfig,
    LimbMPCPlanner,
    MPCParams,
    NoFeasibleControls,
    ParticleLimits,
    PredictiveSampling,
    PredictiveSamplingConfig,
    RolloutProblem,
    get_control_basis,
    retarget,
)
from able.structs import Budget, PlanningFailure

GOAL = np.array([1.0, -0.5])
DT = 0.1


def _simulate(state: np.ndarray, controls: np.ndarray) -> list[np.ndarray]:
    """A 2D point mass: the state is (position, velocity), controls accelerate it."""
    states = []
    for control in controls:
        velocity = state[2:] + DT * control
        state = np.concatenate([state[:2] + DT * velocity, velocity])
        states.append(state)
    return states


class _PointMass(RolloutProblem[np.ndarray]):
    """Reach the goal; rollouts that leave x <= max_x are rejected."""

    def __init__(self, max_x: float = np.inf, extra: list | None = None) -> None:
        self.state = np.zeros(4)
        self.max_x = max_x
        self.extra = [] if extra is None else extra

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_error(self) -> float:
        return float(np.linalg.norm(self.state[:2] - GOAL))

    def rollout(self, state: np.ndarray, controls: np.ndarray) -> tuple[bool, float]:
        states = _simulate(state, controls)
        if any(s[0] > self.max_x for s in states):
            return True, np.inf
        return False, sum(float(np.sum((s[:2] - GOAL) ** 2)) for s in states)

    def get_extra_candidates(self, state, noise_scale, rng) -> list:
        return list(self.extra)


def test_control_basis():
    """Tests that the basis interpolates control points linearly."""
    basis = get_control_basis(horizon=5, num_control_points=3)
    assert basis.shape == (5, 3)
    assert np.allclose(basis.sum(axis=1), 1.0)
    points = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
    assert np.allclose(basis @ points, [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]])


def test_reaches_the_goal():
    """Tests that committing the best controls each step drives the point mass to
    the goal."""
    config = PredictiveSamplingConfig(noise_scale=2.0, noise_reference_error=1.0)
    problem = _PointMass()
    mpc = PredictiveSampling(
        config, problem, np.full(2, -3.0), np.full(2, 3.0), np.random.default_rng(0)
    )
    for _ in range(100):
        controls = mpc.step()
        assert controls.shape == (1, 2)
        assert np.all(np.abs(controls) <= 3.0)
        problem.state = _simulate(problem.state, controls)[-1]
    assert problem.get_error() < 0.1
    assert mpc.num_rollouts == 100 * config.num_rollouts


def test_constraints():
    """Tests that rejected candidates lose and that extra candidates are scored."""
    lower, upper = np.full(2, -3.0), np.full(2, 3.0)
    config = PredictiveSamplingConfig(num_rollouts=4)
    # Moving toward the wall at x = 0, only braking hard avoids it.
    start = np.array([-0.05, 0.0, 0.3, 0.0])
    brake = np.tile([-3.0, 0.0], (4, 1))
    problem = _PointMass(max_x=0.0, extra=[brake])
    problem.state = start
    mpc = PredictiveSampling(config, problem, lower, upper, np.random.default_rng(0))
    assert np.allclose(mpc.step(), [[-3.0, 0.0]])

    problem = _PointMass(max_x=0.0)
    problem.state = start
    mpc = PredictiveSampling(config, problem, lower, upper, np.random.default_rng(0))
    with pytest.raises(NoFeasibleControls):
        mpc.step()


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
    task = _make_nearby_task(model, distribution)
    planner = LimbMPCPlanner(model)
    yield domain, task, planner
    planner.close()
    _close(distribution, model)


def _make_nearby_task(model: LimbModel, distribution: LimbTaskDistribution) -> LimbTask:
    """A sampled task's start, with a goal that bends the elbow a little more."""
    start = distribution.sample(np.random.default_rng(0)).initial_state.coupled
    goal = np.array(start.limb_positions)
    goal[VARIANT.active_joints[2]] += 0.15 * VARIANT.active_signs[2]
    return LimbTask(initial_state=model.get_initial_state(start, goal), goal=goal)


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
    """Tests that the plan reaches the goal in the real environment, which ends the
    task."""
    domain, task, planner = setup
    policy = planner.plan(task, _prior(), FAST, Budget(), np.random.default_rng(0))
    stats = policy.get_planning_stats()
    assert stats["num_mpc_steps"] > 0 and stats["num_rollouts"] > 0
    assert stats["retargeted"] == 0.0
    assert stats["replay_error"] < 0.1
    assert len(policy.actions) == domain.env_model.horizon
    num_controls = sum(not np.isnan(a).all() for a in policy.actions)
    assert all(np.isnan(a).all() for a in policy.actions[num_controls:])
    real_env = domain.real_env
    assert isinstance(real_env, SimulatedEnvironment)
    trajectory = run_episode(real_env, task, policy)
    assert trajectory.states[-1].reached
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
