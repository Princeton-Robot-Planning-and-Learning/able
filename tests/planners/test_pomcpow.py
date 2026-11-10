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
    EarlyStopConfig,
    FrictionPOMCPOWPlanner,
    FrictionPOMCPOWPolicy,
    POMCPOWConfig,
    POMCPOWParams,
    ProposalConfig,
    SearchStats,
    _ActionNode,
    _ObservationNode,
    encode_push,
    get_push_features,
)
from able.structs import Budget, PlanningFailure, PlanningTimeout, Task

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


def test_early_stop_ends_a_settled_search():
    """A search whose best action is clear stops early, at a check, with the
    action the full search finds."""
    particles = [_State(0.0, 0.7)] * 20
    config = POMCPOWConfig(num_sims=2000, max_depth=2, exploration=0.5)
    rule = EarlyStopConfig(min_sims=100, check_every=25, epsilon=0.05)
    action, stats = POMCPOW(_GuessModel(), replace(config, early_stop=rule)).plan(
        particles
    )
    assert 100 <= stats.num_sims < 2000 and stats.num_sims % 25 == 0
    assert action is not None and abs(action - 0.7) < 0.1


def _solver_with_root(num_sims: int, rule: EarlyStopConfig, nodes) -> POMCPOW:
    solver = POMCPOW(_GuessModel(), POMCPOWConfig(num_sims=num_sims, early_stop=rule))
    solver.root = _ObservationNode(actions=dict(enumerate(nodes)))
    return solver


def test_early_stop_rules():
    """The search stops once no action can overtake the most visited one in
    visits, or once its lower bound clears every other upper bound minus
    epsilon."""
    every_check = {"min_sims": 1, "check_every": 1}
    # Only the visit lead: no epsilon lets the bounds settle the search.
    lead = EarlyStopConfig(**every_check, epsilon=-1e9)
    nodes = [_ActionNode(visits=60), _ActionNode(visits=10, value=5.0)]
    assert _solver_with_root(120, lead, nodes).is_settled(71)  # 50 > 49 left
    assert not _solver_with_root(120, lead, nodes).is_settled(50)  # not > 70 left

    # Returns with a standard deviation of 1 over 50 visits each: a standard
    # error of 0.14, so bounds 0.28 either side of the mean.
    def node(value):
        return _ActionNode(visits=50, value=value, squared_deviations=49.0)

    for epsilon, best, settled in [
        (0.0, 10.0, True),
        (0.0, 0.5, False),
        (1.0, 0.5, True),
    ]:
        rule = EarlyStopConfig(**every_check, epsilon=epsilon)
        solver = _solver_with_root(1000, rule, [node(best), node(0.0)])
        assert solver.is_settled(100) == settled
    # Without a rule, or at a sim that is not a check, the search goes on.
    solver = _solver_with_root(
        1000, EarlyStopConfig(min_sims=1, check_every=7), [node(10.0), node(0.0)]
    )
    assert not solver.is_settled(100) and solver.is_settled(105)


def test_a_trained_model_proposes_first_and_widens_slowly(setup):
    """With model_first, a node's first push comes from the model even with mix 0,
    and searches the model guides use the proposal's widening."""
    domain, _, task = setup
    planner = _RecordingPlanner(
        domain.env_model, FrictionBeliefUpdater(domain.env_model.world)
    )
    push = Push(1.5708, 0.6, 0.0)
    proposal = ProposalConfig(mix=0.0, model_first=1, action_widening=(1.0, 0.5))
    params = planner.set_model(replace(PARAMS, proposal=proposal), _FixedModel(push))
    policy = planner.plan(
        task, domain.initial_belief, params, Budget(), np.random.default_rng(0)
    )
    assert push in [p for p, _, _ in policy.searches[0].root_values]
    assert planner.widenings == [(1.0, 0.5)]
    # Without a model, the search keeps the default widening.
    planner.plan(
        task, domain.initial_belief, PARAMS, Budget(), np.random.default_rng(0)
    )
    assert planner.widenings[-1] == PARAMS.config.action_widening


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
    assert policy.searches[0].belief is belief
    assert np.array_equal(searched.belief.particles, expected.particles)


def test_no_push_after_the_task_ends(setup):
    """An ended task needs no search."""
    domain, planner, task = setup
    policy = _plan(planner, task, domain.initial_belief)
    ended = replace(task.initial_state, outcome="goal")
    assert policy.get_action(ended) == NO_PUSH and len(policy.searches) == 1


def test_budget_and_dead_ends(setup):
    """Searches share the budget, running out of its simulations or time is a
    timeout, and planning fails without a feasible push. Failures keep the
    statistics so far."""
    domain, planner, task = setup
    policy = _plan(planner, task, domain.initial_belief, Budget(max_iterations=30))
    assert policy.get_planning_stats()["num_sims"] == 30
    trajectory = run_episode(domain.real_env, task, policy, max_steps=1)
    with pytest.raises(PlanningTimeout, match="simulations") as failure:
        policy.get_action(trajectory.states[-1])
    assert failure.value.stats["num_sims"] == 30
    with pytest.raises(PlanningTimeout, match="time"):
        _plan(planner, task, domain.initial_belief, Budget(max_wall_time=0.0))
    slot = replace(task.initial_state, puck=(2.6, 2.2), true_puck=(2.6, 2.2))
    with pytest.raises(PlanningFailure, match="No feasible push") as failure:
        _plan(planner, Task(slot, task.goal), domain.initial_belief)
    assert not isinstance(failure.value, PlanningTimeout)
    assert failure.value.stats["num_sims"] > 0


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


class _RecordingPlanner(FrictionPOMCPOWPlanner):
    """Records the progressive widening of every search."""

    widenings: list[tuple[float, float]] = []

    def search(self, state, belief, config, rng, proposer=None):
        self.widenings.append(config.action_widening)
        return super().search(state, belief, config, rng, proposer)


def _make_failing(model: FrictionModel) -> FrictionPOMCPOWPlanner:
    return _FailingWithModel(model, FrictionBeliefUpdater(model.world))


def test_plan_examples(setup):
    """Each search gives rows of its features under the belief it searched with
    and well-visited root pushes, with the chosen push among them, costed by the
    negative of their values."""
    domain, planner, task = setup
    belief = domain.initial_belief
    policy = _plan(planner, task, belief)
    run_episode(domain.real_env, task, policy, max_steps=2)
    rows = planner.get_plan_examples(task, belief, policy)
    assert rows.x.shape[1] == X_DIM and rows.y.shape[1] == Y_DIM
    assert sorted(set(rows.groups)) == [0, 1]
    # The first push narrowed the belief the second search used.
    assert not np.allclose(policy.searches[1].belief.std, belief.std)
    for group, search in enumerate(policy.searches):
        features = get_push_features(
            search.state.puck_array, task.goal, planner.pomdp.world, search.belief
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
