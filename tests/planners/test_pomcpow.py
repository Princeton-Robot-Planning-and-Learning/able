"""Tests for planners/pomcpow.py."""

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pytest

from able.belief_updaters.friction import FrictionBeliefUpdater
from able.envs.friction.model import NO_PUSH, FrictionModel
from able.envs.friction.tasks import make_domain
from able.envs.friction.world import Push
from able.episodes import run_episode
from able.models.base import ConditionalModel, Dataset
from able.planners.pomcpow import (
    POMCPOW,
    X_DIM,
    Y_DIM,
    FrictionPOMCPOWPlanner,
    FrictionPOMCPOWPolicy,
    POMCPOWConfig,
    POMCPOWParams,
    ProposalConfig,
    SearchStats,
    encode_push,
    get_push_features,
)
from able.structs import Budget, PlanningFailure, Task

PARAMS = POMCPOWParams(POMCPOWConfig(num_sims=40, max_depth=3))


@dataclass(frozen=True)
class _State:
    position: float
    target: float  # hidden
    steps: int = 0
    last_reward: float = 0.0


class _GuessModel:
    """Move to a hidden target in [0, 1]; the observation is the noisy target."""

    def step(self, state, action, rng):
        """Move to `action`; the reward is minus the distance to the target."""
        del rng
        reward = -abs(action - state.target)
        nxt = _State(action, state.target, state.steps + 1, reward)
        return nxt, reward

    def reward(self, state, action, next_state):
        """The reward stored on the next state."""
        del state, action
        return next_state.last_reward

    def observe(self, next_state, action, rng):
        """The target, with noise."""
        del action
        return round(next_state.target + float(rng.normal(0.0, 0.05)), 3)

    def likelihood(self, observation, next_state, action):
        """Gaussian density of the observed target."""
        del action
        return float(np.exp(-0.5 * ((observation - next_state.target) / 0.05) ** 2))

    def is_terminal(self, state):
        """Two moves end the episode."""
        return state.steps >= 2

    def propose(self, state, tried: Sequence[float], rng):
        """A uniform random position."""
        del state, tried
        return round(float(rng.uniform(0.0, 1.0)), 4)

    def value(self, state):
        """No leaf estimate."""
        del state
        return 0.0


def test_pomcpow_finds_the_hidden_target():
    """With a known target, the best first action lands near it."""
    particles = [_State(0.0, 0.7)] * 20
    solver = POMCPOW(
        _GuessModel(), POMCPOWConfig(num_sims=2000, max_depth=2, exploration=0.5)
    )
    action, stats = solver.plan(particles)
    assert action is not None and abs(action - 0.7) < 0.1
    assert stats.num_sims == 2000
    assert solver.root is not None and stats.root_actions == len(solver.root.actions)
    assert stats.root_values[0][1] >= stats.root_values[-1][1]


def test_pomcpow_proposer_controls_the_actions():
    """A custom proposer replaces the model's sampler."""
    calls = []

    def proposer(state, states, tried, rng):
        del rng
        calls.append((state, len(states), len(tried)))
        return 0.5 if not tried else None

    particles = [_State(0.0, 0.3), _State(0.0, 0.9)]
    solver = POMCPOW(_GuessModel(), POMCPOWConfig(num_sims=50, max_depth=2), proposer)
    action, stats = solver.plan(particles)
    assert action == 0.5 and stats.root_actions == 1
    assert calls and calls[0][1] == 2


def test_pomcpow_time_budget_and_dead_ends():
    """The time budget stops the search, and a root with no actions returns None."""
    solver = POMCPOW(_GuessModel(), POMCPOWConfig(num_sims=10**9, max_time=0.2))
    action, stats = solver.plan([_State(0.0, 0.5)])
    assert action is not None and stats.seconds < 2.0
    stuck = POMCPOW(_GuessModel(), proposer=lambda *args: None)
    action, stats = stuck.plan([_State(0.0, 0.5)])
    assert action is None and stats.root_actions == 0


def _make_planner(model: FrictionModel) -> FrictionPOMCPOWPlanner:
    return FrictionPOMCPOWPlanner(model, FrictionBeliefUpdater(model.world))


def _close(model: FrictionModel) -> None:
    model.close()


@pytest.fixture(name="setup", scope="module")
def _setup():
    domain = make_domain(seed=0, num_particles=50)
    model = domain.env_model
    assert isinstance(model, FrictionModel)
    planner = _make_planner(model)
    task = domain.task_distribution.sample(np.random.default_rng(0))
    yield domain, planner, task
    _close(model)


def _plan(planner, task, belief, budget=Budget()) -> FrictionPOMCPOWPolicy:
    return planner.plan(task, belief, PARAMS, budget, np.random.default_rng(0))


def test_replans_each_push_under_the_updated_belief(setup):
    """The first push is planned ahead, and later pushes are searched under the
    belief updated with the earlier ones."""
    domain, planner, task = setup
    belief = domain.initial_belief
    policy = _plan(planner, task, belief)
    assert len(policy.searches) == 1
    first = policy.searches[0].push
    world = planner.pomdp.world
    assert world.push_feasible(task.initial_state.puck_array, first.angle, first.stance)
    trajectory = run_episode(domain.real_env, task, policy, max_steps=2)
    assert trajectory.actions[0] == first and len(policy.searches) == 2
    stats = policy.get_planning_stats()
    assert stats["num_sims"] == 80 and stats["num_searches"] == 2
    # An untrained model proposes nothing, so the search is plain POMCPOW.
    assert stats["num_seeded_searches"] == 0
    states = trajectory.states
    expected = planner.belief_updater.update_from_transition(
        belief, states[0], first, states[1]
    )
    # The second search planned under the belief updated with the first push.
    searched = policy.searches[1]
    assert searched.state == states[1]
    assert searched.root_values[0][0] == trajectory.actions[1]
    assert np.array_equal(policy.belief.particles, expected.particles)


def test_no_push_after_the_task_ends(setup):
    """An ended task needs no search."""
    domain, planner, task = setup
    policy = _plan(planner, task, domain.initial_belief)
    ended = replace(task.initial_state, outcome="goal")
    assert policy.get_action(ended) == NO_PUSH and len(policy.searches) == 1


def test_budget_and_dead_ends(setup):
    """Searches share the budget, and planning fails without a feasible push."""
    domain, planner, task = setup
    policy = _plan(planner, task, domain.initial_belief, Budget(max_iterations=30))
    assert policy.get_planning_stats()["num_sims"] == 30
    trajectory = run_episode(domain.real_env, task, policy, max_steps=1)
    with pytest.raises(PlanningFailure, match="budget"):
        policy.get_action(trajectory.states[-1])
    slot = replace(task.initial_state, puck=(2.6, 2.2), true_puck=(2.6, 2.2))
    with pytest.raises(PlanningFailure, match="No feasible push"):
        _plan(planner, Task(slot, task.goal), domain.initial_belief)


class _FixedModel(ConditionalModel):
    """Always proposes the same push."""

    def __init__(self, push: Push) -> None:
        self.push = push

    def fit(self, dataset: Dataset) -> ConditionalModel:
        return self

    def sample(self, x, num_samples, rng):
        return np.array([encode_push(self.push)] * num_samples)


class _FailingWithModel(FrictionPOMCPOWPlanner):
    """Finds no push whenever the model guides the search."""

    def search(self, state, belief, config, rng, proposer=None):
        if proposer is not None:
            return None, SearchStats()
        return super().search(state, belief, config, rng, proposer)


def _make_failing(model: FrictionModel) -> FrictionPOMCPOWPlanner:
    return _FailingWithModel(model, FrictionBeliefUpdater(model.world))


def test_plan_examples(setup):
    """Each search gives rows of its features and well-visited root pushes, with
    the chosen push among them, costed by the negative of their values."""
    domain, planner, task = setup
    belief = domain.initial_belief
    policy = _plan(planner, task, belief)
    run_episode(domain.real_env, task, policy, max_steps=2)
    rows = planner.get_plan_examples(task, belief, policy)
    assert rows.x.shape[1] == X_DIM and rows.y.shape[1] == Y_DIM
    assert sorted(set(rows.groups)) == [0, 1]
    for group, search in enumerate(policy.searches):
        features = get_push_features(
            search.state.puck_array, task.goal, planner.pomdp.world, belief
        )
        mine = rows.groups == group
        assert np.allclose(rows.x[mine], features)
        assert any(np.allclose(y, encode_push(search.push)) for y in rows.y[mine])
        values = [value for _, _, value in search.root_values]
        assert all(np.isclose(values, -cost).any() for cost in rows.costs[mine])


def test_a_trained_model_guides_the_search(setup):
    """The model's push is tried at the root, and a search that fails with the
    model is repeated without it."""
    domain, planner, task = setup
    push = Push(1.5708, 0.6, 0.0)
    model = _FixedModel(push)
    params = planner.set_model(replace(PARAMS, proposal=ProposalConfig(mix=1.0)), model)
    assert planner.get_model(params) is model
    rng = np.random.default_rng(0)
    policy = planner.plan(task, domain.initial_belief, params, Budget(), rng)
    assert push in [p for p, _, _ in policy.searches[0].root_values]
    assert policy.get_planning_stats()["num_seeded_searches"] == 1
    failing = _make_failing(domain.env_model)
    policy = failing.plan(task, domain.initial_belief, params, Budget(), rng)
    stats = policy.get_planning_stats()
    assert stats["num_fallbacks"] == 1 and stats["num_sims"] == 40
