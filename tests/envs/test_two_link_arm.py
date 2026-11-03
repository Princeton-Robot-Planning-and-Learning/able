"""Tests for envs/two_link_arm.py."""

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmTask,
    JointQuery,
    TwoLinkArmModel,
    make_domain,
    sample_goal_positions,
)


def _sample_limits(rng: np.random.Generator, num: int) -> np.ndarray:
    return rng.uniform(JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 1], size=(num, 4))


def _random_actions(
    model: TwoLinkArmModel, rng: np.random.Generator, num: int
) -> np.ndarray:
    bound = model.max_joint_displacement
    return rng.uniform(-bound, bound, size=(num, model.horizon, 2))


def test_kinematics():
    """Tests forward and inverse kinematics."""
    model = TwoLinkArmModel()
    assert np.allclose(model.forward_kinematics(np.array([0.0, 0.0])), [2.0, 0.0])
    assert np.allclose(model.forward_kinematics(np.array([np.pi / 2, 0.0])), [0.0, 2.0])
    assert np.allclose(model.forward_kinematics(np.array([0.0, np.pi / 2])), [1.0, 1.0])
    batch = np.zeros((3, 5, 2))
    assert model.forward_kinematics(batch).shape == (3, 5, 2)

    rng = np.random.default_rng(0)
    for joints in rng.uniform(-np.pi + 0.01, np.pi - 0.01, size=(50, 2)):
        solutions = model.inverse_kinematics(model.forward_kinematics(joints))
        assert len(solutions) == 2
        for solution in solutions:
            assert np.allclose(
                model.forward_kinematics(solution), model.forward_kinematics(joints)
            )
        assert any(np.allclose(solution, joints) for solution in solutions)
    assert not model.inverse_kinematics(np.array([2.5, 0.0]))


def test_joint_limits_clip_motion():
    """Tests that the joint limits clip each step."""
    model = TwoLinkArmModel()
    limits = np.array([-0.1, 0.2, -0.2, 0.1])
    rng = np.random.default_rng(0)
    next_state = model.sample_next_state(
        np.zeros(2), np.array([0.3, -0.3]), limits, rng
    )
    assert np.allclose(next_state, [0.2, -0.2])
    next_state = model.sample_next_state(
        np.zeros(2), np.array([0.05, 0.05]), limits, rng
    )
    assert np.allclose(next_state, [0.05, 0.05])


def test_joint_queries():
    """Tests that a joint query is answered yes exactly when the angle is within that
    joint's limits."""
    model = TwoLinkArmModel()
    limits = np.array([-0.1, 0.2, -0.2, 0.1])
    rng = np.random.default_rng(0)
    expected = {
        JointQuery(joint=0, angle=0.2): True,
        JointQuery(joint=0, angle=0.21): False,
        JointQuery(joint=0, angle=-0.1): True,
        JointQuery(joint=0, angle=-0.11): False,
        JointQuery(joint=1, angle=0.15): False,
        JointQuery(joint=1, angle=-0.15): True,
    }
    for query, response in expected.items():
        assert model.sample_response(query, limits, rng) is response
    env = SimulatedEnvironment(model, limits, rng)
    assert env.ask(JointQuery(joint=1, angle=0.1))
    assert not env.ask(JointQuery(joint=1, angle=0.11))


def test_check_goal():
    """Tests that the goal is reached within the goal tolerance."""
    model = TwoLinkArmModel(goal_tolerance=0.1)
    joints = np.array([0.3, 0.4])
    task = ArmTask(initial_state=np.zeros(2), goal=model.forward_kinematics(joints))
    limits = JOINT_LIMIT_PRIOR.mean(axis=1)
    assert model.check_goal(task, joints, limits)
    # Moving the elbow by 0.05 moves the end effector by about 0.05.
    assert model.check_goal(task, joints + np.array([0.0, 0.05]), limits)
    assert not model.check_goal(task, joints + np.array([0.0, 0.2]), limits)

    domain = make_domain(seed=0, goal_tolerance=0.2)
    assert domain.real_env.check_goal(task, joints + np.array([0.0, 0.15]))


def test_rollout_batch_matches_stepping():
    """Tests that batched rollouts and costs match stepping one action at a time."""
    model = TwoLinkArmModel()
    rng = np.random.default_rng(0)
    task = ArmTask(initial_state=np.zeros(2), goal=np.array([0.5, 1.2]))
    actions = _random_actions(model, rng, num=4)
    limits = _sample_limits(rng, num=3)
    states = model.rollout_batch(task.initial_state, actions, limits, rng)
    assert states.shape == (4, 3, model.horizon + 1, 2)
    costs = model.get_cost_batch(task, states, actions, limits)
    assert costs.shape == (4, 3)
    for k, action_sequence in enumerate(actions):
        for p, hidden_params in enumerate(limits):
            env = SimulatedEnvironment(model, hidden_params, rng)
            env.reset(task)
            total_cost = 0.0
            for t, action in enumerate(action_sequence):
                result = env.step(action)
                assert np.allclose(result.state, states[k, p, t + 1])
                total_cost += result.cost
            assert result.terminated
            assert np.isclose(total_cost, costs[k, p])


def test_cost_lower_bound():
    """Tests that the cost lower bound is zero for reachable goals and below the cost of
    sampled plans otherwise."""
    model = TwoLinkArmModel()
    rng = np.random.default_rng(0)

    # A goal at the end-effector position of a configuration within the limits.
    limits = np.array([[-1.0, 1.0, -1.0, 1.0]])
    reachable = ArmTask(
        initial_state=np.zeros(2),
        goal=model.forward_kinematics(np.array([0.3, -0.4])),
    )
    assert np.allclose(model.get_cost_lower_bound(reachable, limits), 0.0)

    # A goal beyond the arm's full extension.
    too_far = ArmTask(initial_state=np.zeros(2), goal=np.array([2.5, 0.0]))
    assert np.all(model.get_cost_lower_bound(too_far, limits) >= 0.5 - 1e-9)

    # Random goals and limits: no sampled plan beats the bound by more than the
    # discretization error.
    limits = _sample_limits(rng, num=8)
    for goal in rng.uniform(-2.0, 2.0, size=(10, 2)):
        task = ArmTask(initial_state=np.zeros(2), goal=goal)
        bound = model.get_cost_lower_bound(task, limits)
        assert bound.shape == (8,)
        actions = _random_actions(model, rng, num=200)
        states = model.rollout_batch(task.initial_state, actions, limits, rng)
        best_costs = model.get_cost_batch(task, states, actions, limits).min(axis=0)
        assert np.all(bound <= best_costs + 0.01)


def test_sample_goal_positions():
    """Tests that sampled goals are reachable under the true limits."""
    model = TwoLinkArmModel()
    rng = np.random.default_rng(0)
    for true_limits in _sample_limits(rng, num=10):
        belief = IntervalBelief(np.stack([true_limits - 1e-9, true_limits + 1e-9], 1))
        goals = sample_goal_positions(model, true_limits, num_goals=6, rng=rng)
        assert goals.shape == (6, 2)
        for goal in goals:
            solutions = model.inverse_kinematics(goal)
            lows = belief.intervals[[0, 2], 0]
            highs = belief.intervals[[1, 3], 1]
            assert any(
                np.all(lows <= solution) and np.all(solution <= highs)
                for solution in solutions
            )


def test_make_domain():
    """Tests make_domain()."""
    domain = make_domain(seed=0)
    rng = np.random.default_rng(0)
    tasks = list(
        {
            id(t): t for t in (domain.task_distribution.sample(rng) for _ in range(200))
        }.values()
    )
    assert len(tasks) == 12
    for task in tasks:
        assert np.allclose(task.initial_state, 0.0)
    assert domain.initial_belief == IntervalBelief(JOINT_LIMIT_PRIOR)

    true_limits = np.array([-1.0, 0.6, -1.0, 0.6])
    domain = make_domain(seed=0, true_limits=true_limits)
    env = domain.real_env
    env.reset(tasks[0])
    for _ in range(3):
        result = env.step(np.array([0.3, 0.3]))
    assert np.allclose(result.state, [0.6, 0.6])
