"""Tests for planners/mpc.py."""

import dataclasses

import numpy as np
import pytest
from pybullet_helpers.geometry import SE2Pose

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.human import RangeOfMotionPrior, get_variant
from able.envs.limb.model import LimbModel, LimbTask
from able.envs.limb.sim import CoupledState
from able.envs.limb.tasks import LimbTaskDistribution, make_domain
from able.envs.simulated import SimulatedEnvironment
from able.episodes import run_episode
from able.models.base import ConditionalModel, Dataset
from able.models.exact_match_cache import ExactMatchCache
from able.planners.mpc import (
    LimbMPCConfig,
    LimbMPCPlanner,
    LimbMPCPolicy,
    MPCParams,
    NoFeasibleControls,
    ParticleLimits,
    PredictiveSampling,
    PredictiveSamplingConfig,
    RolloutProblem,
    SeedConfig,
    _with_start_reach,
    get_control_basis,
    get_control_window,
    get_mpc_features,
    retarget,
)
from able.structs import Budget, PlanningFailure, PlanningTimeout

# The limb tests share a planned task, so they run on one pytest-xdist worker.
pytestmark = pytest.mark.xdist_group("limb_mpc")

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


def _pd_controls(state: np.ndarray, horizon: int, sign: float = 1.0) -> np.ndarray:
    """Per-step controls of a PD controller toward the goal (away if sign < 0)."""
    controls = []
    for _ in range(horizon):
        control = np.clip(sign * 4.0 * (GOAL - state[:2]) - 3.0 * state[2:], -3, 3)
        controls.append(control)
        state = _simulate(state, control[None])[-1]
    return np.array(controls)


class _PDBar(_PointMass):
    """The PD controller toward the goal is the only extra candidate."""

    def get_extra_candidates(self, state, noise_scale, rng) -> list:
        return [_pd_controls(state, 12)[[0, 4, 7, 11]]]


def _run(problem: _PointMass, mpc: PredictiveSampling, num_steps: int) -> list:
    committed = []
    for _ in range(num_steps):
        controls = mpc.step()
        committed.append(controls)
        problem.state = _simulate(problem.state, controls)[-1]
    return committed


def test_seeding():
    """Tests that good proposals are committed with few rollouts, bad ones fall back
    to the full batch, and no proposals change nothing."""
    config = PredictiveSamplingConfig(noise_scale=2.0, noise_reference_error=1.0)
    bounds = (np.full(2, -3.0), np.full(2, 3.0))

    def seeder(state, step, num):
        del step, num
        return [_pd_controls(state, config.horizon)]

    problem = _PointMass()
    mpc = PredictiveSampling(config, problem, *bounds, np.random.default_rng(0), seeder)
    _run(problem, mpc, 60)
    assert problem.get_error() < 0.1
    assert mpc.num_seeded_steps == 60 and mpc.num_rollouts == 60

    # The PD controls are the bar, so proposals that push away are never good enough.
    problem = _PDBar()
    away = PredictiveSampling(
        config,
        problem,
        *bounds,
        np.random.default_rng(0),
        lambda state, step, num: [_pd_controls(state, config.horizon, sign=-1.0)],
        SeedConfig(num_rollouts=1),
    )
    _run(problem, away, 10)
    assert away.num_seeded_steps == 0 and away.num_full_steps == 10

    plain_problem, empty_problem = _PointMass(), _PointMass()
    plain = PredictiveSampling(config, plain_problem, *bounds, np.random.default_rng(0))
    empty = PredictiveSampling(
        config, empty_problem, *bounds, np.random.default_rng(0), lambda *_: []
    )
    for first, second in zip(
        _run(plain_problem, plain, 5), _run(empty_problem, empty, 5)
    ):
        assert np.array_equal(first, second)


VARIANT = get_variant("human-left-arm")
# Few, short rollouts, for speed.
FAST = MPCParams(
    config=LimbMPCConfig(
        sampling=PredictiveSamplingConfig(num_rollouts=4, horizon=4),
        directed_gains=(1.0, 4.0),
    ),
    num_particles=8,
)


def _prior() -> RangeOfMotionBelief:
    prior = RangeOfMotionPrior()
    return RangeOfMotionBelief.from_prior(
        prior.mean, prior.stddev, prior.low, prior.high, 3
    )


def _with_goal(model: LimbModel, task: LimbTask, fraction: float) -> LimbTask:
    """The task with its goal moved the fraction of the way from the start."""
    start = task.initial_state.limb_positions
    goal = start + fraction * (np.asarray(task.goal) - start)
    initial_state = model.get_initial_state(task.initial_state.coupled, goal)
    return LimbTask(initial_state=initial_state, goal=goal)


@pytest.fixture(name="setup", scope="module")
def _setup():
    domain = make_domain(seed=0, horizon=60)
    model = domain.env_model
    assert isinstance(model, LimbModel)
    distribution = domain.task_distribution
    assert isinstance(distribution, LimbTaskDistribution)
    # Sampled tasks hold the grasp along the line to the goal, so a goal 30% of
    # the way is reachable, and fast MPC solves it in about 11 steps.
    task = _with_goal(model, distribution.sample(np.random.default_rng(3)), 0.3)
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


@pytest.fixture(name="plain", scope="module")
def _plain(setup):
    """A plan made with a model that records what the seeder saw and proposes
    nothing, so it plans as if there were no model."""
    _, task, planner = setup
    recorder = _RecordingModel()
    params = dataclasses.replace(FAST, model=recorder)
    policy = planner.plan(task, _prior(), params, Budget(), np.random.default_rng(0))
    return policy, recorder.inputs


def test_plan_reaches_the_goal(setup, plain):
    """Tests that the plan reaches the goal in the real environment, which ends the
    task."""
    domain, task, _ = setup
    policy, _ = plain
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
    with pytest.raises(PlanningFailure, match="budget") as failure:
        planner.plan(task, _prior(), FAST, Budget(max_iterations=0), rng)
    assert not isinstance(failure.value, PlanningTimeout)
    with pytest.raises(PlanningTimeout, match="time"):
        planner.plan(task, _prior(), FAST, Budget(max_wall_time=0.0), rng)
    short = dataclasses.replace(
        FAST, config=dataclasses.replace(FAST.config, divergence_patience=0)
    )
    with pytest.raises(PlanningFailure, match="Diverged"):
        planner.plan(task, _prior(), short, Budget(), rng)


def test_the_start_raises_the_believed_limits(setup):
    """Tests that the believed limits cover the start's reach, so a start beyond
    what the belief's samples allow does not break the believed range of motion
    before any control."""
    _, task, planner = setup
    reach = np.array(task.initial_state.max_reach)
    prior = RangeOfMotionPrior()
    # Every sample allows about 0.72 of each motion's range, short of the start's
    # third motion.
    belief = RangeOfMotionBelief.from_prior(0.72, 0.005, prior.low, prior.high, 3)
    assert reach[2] > 0.75
    raised = _with_start_reach(belief, task.initial_state)
    assert np.allclose(raised.floor, [prior.low, prior.low, reach[2]])
    # Planning gets past the first step and runs out of its two-step budget.
    with pytest.raises(PlanningFailure, match="budget"):
        planner.plan(
            task, belief, FAST, Budget(max_iterations=2), np.random.default_rng(0)
        )


class _RecordingModel(ConditionalModel):
    """Records every input it is asked about and proposes the given output, if any."""

    def __init__(self, output: np.ndarray | None = None) -> None:
        self.inputs: list[np.ndarray] = []
        self.output = output

    def fit(self, dataset: Dataset) -> "_RecordingModel":
        return self

    def sample(self, x, num_samples, rng) -> np.ndarray:
        self.inputs.append(x)
        if self.output is None:
            return np.zeros((0, 0))
        return np.tile(self.output, (num_samples, 1))


def test_features_and_labels():
    """Tests the model's input and the padded control window."""
    belief = _prior()
    coupled = CoupledState(
        SE2Pose(0.0, 0.0, 0.0),
        tuple(range(13)),
        tuple(range(13)),
        tuple(range(6)),
        tuple(range(6)),
    )
    x = get_mpc_features(coupled, np.ones(6), belief, 3, np.full(7, 2.0))
    assert x.shape == (6 + 7 + 6 + 3 + 3 + 1 + 6 + 7 + 7,)
    assert x[25] == 3.0 and np.all(x[-7:] == 2.0)
    controls = [np.full(7, float(i)) for i in range(3)]
    window = get_control_window(controls, 1, 4)
    assert window.shape == (4, 7)
    assert list(window[:, 0]) == [1.0, 2.0, 2.0, 2.0]


def test_plan_examples_match_the_seeder(setup, plain):
    """Tests that the training rows are what the seeder saw, and the labels are the
    controls that followed."""
    _, task, planner = setup
    policy, inputs = plain
    assert isinstance(policy, LimbMPCPolicy)
    stats = policy.get_planning_stats()
    assert stats["num_seeded_steps"] == 0 and stats["fell_back"] == 0
    examples = planner.get_plan_examples(task, _prior(), policy)
    assert len(examples) == len(policy.controls) == len(inputs)
    assert np.allclose(examples.x, np.array(inputs))
    horizon = FAST.config.sampling.horizon
    assert examples.y.shape == (len(examples), horizon * 7)
    assert np.allclose(examples.y[0][:7], policy.controls[0])


def test_a_trained_model_seeds_mpc(setup, plain):
    """Tests that a cache of a plan's rows seeds replanning the same task, with
    fewer rollouts."""
    domain, task, planner = setup
    plain_policy, _ = plain
    examples = planner.get_plan_examples(task, _prior(), plain_policy)
    cache = ExactMatchCache(key_resolution=1e-6).fit(examples)
    seeded_params = dataclasses.replace(
        FAST, model=cache, seed=SeedConfig(num_rollouts=1, tolerance=1.0)
    )
    seeded = planner.plan(
        task, _prior(), seeded_params, Budget(), np.random.default_rng(0)
    )
    stats = seeded.get_planning_stats()
    assert stats["num_seeded_steps"] > 0 and stats["fell_back"] == 0
    assert stats["num_rollouts"] < plain_policy.get_planning_stats()["num_rollouts"]
    real_env = domain.real_env
    assert isinstance(real_env, SimulatedEnvironment)
    trajectory = run_episode(real_env, task, seeded)
    assert real_env.check_goal(task, trajectory.states[-1])


class _FailingWithModel(LimbMPCPlanner):
    """Fails whenever it plans with a model that proposed something."""

    def _plan_once(self, attempt, stats, use_model):
        if use_model:
            attempt.model_proposed = self.model_proposes
            raise PlanningFailure("Stalled with the model")
        return super()._plan_once(attempt, stats, use_model)

    model_proposes = True


def test_fallback_without_the_model(setup):
    """Tests that a failed plan with the model is replanned without it, unless the
    model proposed nothing."""
    domain, task, _ = setup
    model = domain.env_model
    assert isinstance(model, LimbModel)
    # Without the model, a goal at the start is planned without any MPC steps.
    task = _with_goal(model, task, 0.0)
    planner = _FailingWithModel(model)
    try:
        policy = planner.plan(task, _prior(), FAST, Budget(), np.random.default_rng(0))
        assert policy.get_planning_stats()["fell_back"] == 1
        planner.model_proposes = False
        with pytest.raises(PlanningFailure, match="Stalled"):
            planner.plan(task, _prior(), FAST, Budget(), np.random.default_rng(0))
    finally:
        planner.close()
