"""Tests for envs/friction/tasks.py."""

import numpy as np
import pytest

from able.beliefs.particles import ParticleBelief
from able.envs.friction.model import FrictionModel
from able.envs.friction.pomdp import PuckPushingModel
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
    and each task starts the puck and the robot where the task says."""
    model, real_env, belief, distribution = _get_parts(
        make_domain(seed=0, num_particles=100)
    )
    levels = np.array(DEFAULT_FRICTION_PRIOR.levels)
    ratios = real_env.hidden_params[:, None] / levels
    assert np.all(np.any((0.8 - 1e-9 <= ratios) & (ratios <= 1.2 + 1e-9), axis=1))
    assert belief.particles.shape == (100, 3)
    assert np.allclose(belief.low, DEFAULT_FRICTION_PRIOR.bounds[0])
    rng = np.random.default_rng(0)
    first, second = distribution.sample(rng), distribution.sample(rng)
    assert first.initial_state != second.initial_state and first.goal != second.goal
    assert first.initial_state.goal == first.goal
    state = real_env.reset(first)
    assert state.puck == first.initial_state.puck and state.goal == first.goal
    result = real_env.step(Push(np.pi / 2, 0.05))
    assert result.state.pushes == 1 and not result.terminated
    assert result.state.goal == first.goal
    model.close()
    model, real_env, belief, _ = _get_parts(
        make_domain(seed=0, num_particles=100, num_floors=8)
    )
    assert real_env.hidden_params.shape == (8,)
    assert belief.particles.shape == (100, 8)
    model.close()


@pytest.mark.parametrize("num_floors", [3, 8])
def test_tasks_vary_and_stay_within_reach(num_floors):
    """Tasks start and end on every floor, each with a pushable path between its
    start and goal of the configured length, clear of the holes, with the robot clear of
    the walls and the puck."""
    domain = make_domain(seed=0, task_distances=(1.5, 3.5), num_floors=num_floors)
    model, _, _, distribution = _get_parts(domain)
    world = model.world
    paths = PuckPushingModel(world)
    phys = world.physics
    rng = np.random.default_rng(1)
    start_floors, goal_floors = set(), set()
    for _ in range(100):
        task = distribution.sample(rng)
        state = task.initial_state
        puck, robot = np.array(state.puck), np.array(state.robot)
        distance = paths.distance_to_goal(puck, task.goal)
        assert 1.5 - 1e-9 <= distance <= 3.5 + 1e-9
        assert world.has_room(puck) and not world.in_goal(puck, task.goal)
        gaps = np.linalg.norm(world.holes - puck, axis=1)
        assert np.all(gaps > phys.hole_radius + phys.puck_radius)
        goal_gaps = np.linalg.norm(world.holes - np.array(task.goal), axis=1)
        assert np.all(goal_gaps > phys.hole_radius + phys.goal_radius)
        assert world.robot_position_free(robot)
        assert world.robot_clear_of_puck(robot, puck)
        assert state.robot_pose == (*state.robot, 0.0)
        start_floors.add(world.surface_at(puck))
        goal_floors.add(world.surface_at(np.array(task.goal)))
    assert start_floors == goal_floors == set(range(num_floors))


def test_tasks_vary_and_stay_within_reach():
    """Tasks start and end all over the maze, each with a pushable path between its
    start and goal of the configured length, clear of the holes, with the robot clear of
    the walls and the puck."""
    domain = make_domain(seed=0, task_distances=(1.5, 3.5))
    model, _, _, distribution = _get_parts(domain)
    world = model.world
    paths = PuckPushingModel(world)
    phys = world.physics
    rng = np.random.default_rng(1)
    start_floors, goal_floors = set(), set()
    for _ in range(100):
        task = distribution.sample(rng)
        state = task.initial_state
        puck, robot = np.array(state.puck), np.array(state.robot)
        distance = paths.distance_to_goal(puck, task.goal)
        assert 1.5 - 1e-9 <= distance <= 3.5 + 1e-9
        assert world.has_room(puck) and not world.in_goal(puck, task.goal)
        gaps = np.linalg.norm(world.holes - puck, axis=1)
        assert np.all(gaps > phys.hole_radius + phys.puck_radius)
        goal_gaps = np.linalg.norm(world.holes - np.array(task.goal), axis=1)
        assert np.all(goal_gaps > phys.hole_radius + phys.goal_radius)
        assert world.robot_position_free(robot)
        assert world.robot_clear_of_puck(robot, puck)
        assert state.robot_pose == (*state.robot, 0.0)
        start_floors.add(world.surface_at(puck))
        goal_floors.add(world.surface_at(np.array(task.goal)))
    assert start_floors == goal_floors == {0, 1, 2}


def test_seeds_and_given_frictions():
    """Seeds draw different people, and given frictions override the prior."""
    frictions = [_get_parts(make_domain(seed=s))[1].hidden_params for s in range(3)]
    assert not np.allclose(frictions[0], frictions[1])
    domain = make_domain(seed=0, num_holes=0, true_frictions=[0.1, 0.05, 0.03])
    assert np.allclose(_get_parts(domain)[1].hidden_params, [0.1, 0.05, 0.03])
