"""Tests for learners/model_learner.py."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution, TaskDistribution
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmTask,
    TwoLinkArmModel,
)
from able.learners.belief_sources import CurrentBeliefSource, PriorSource
from able.learners.model_learner import ModelLearner
from able.models.base import Dataset
from able.models.exact_match_cache import ExactMatchCache
from able.planners.random_shooting import (
    RandomShootingParams,
    RandomShootingPlanner,
    get_plan_features,
)
from able.structs import Budget

TASKS = [
    ArmTask(initial_state=np.zeros(2), goal=np.array(goal))
    for goal in [(1.0, 1.0), (0.5, 1.5), (-1.0, 1.2)]
]


@dataclasses.dataclass(frozen=True, eq=False)
class _CountingCache(ExactMatchCache):
    """An exact-match cache that counts how often it is fit."""

    fits: list = dataclasses.field(default_factory=list)

    def fit(self, dataset: Dataset) -> "_CountingCache":
        self.fits.append(len(dataset))
        fitted = super().fit(dataset)
        assert isinstance(fitted, _CountingCache)
        return fitted


class _GoalDistribution(TaskDistribution[ArmTask]):
    """An infinite distribution over reachable goals."""

    def sample(self, rng: np.random.Generator) -> ArmTask:
        radius, angle = rng.uniform(0.5, 1.8), rng.uniform(0, np.pi)
        goal = radius * np.array([np.cos(angle), np.sin(angle)])
        return ArmTask(initial_state=np.zeros(2), goal=goal)


class _TimedSource(CurrentBeliefSource[Any, Any]):
    """Trains under the current belief and records the training times observed."""

    def __init__(self) -> None:
        self.observed: list[tuple[Belief[Any], float]] = []

    def observe_training_time(self, belief: Belief[Any], training_time: float) -> None:
        self.observed.append((belief, training_time))


def _cache_entries(params: RandomShootingParams) -> dict:
    assert isinstance(params.model, ExactMatchCache)
    return dict(params.model.entries)


def _assert_same_entries(first: dict, second: dict) -> None:
    assert set(first) == set(second)
    for key, entries in first.items():
        assert [cost for cost, _ in entries] == [cost for cost, _ in second[key]]
        for (_, a), (_, b) in zip(entries, second[key]):
            assert np.array_equal(a, b)


def _make_planner(cache: ExactMatchCache | None = None) -> RandomShootingPlanner:
    cache = ExactMatchCache() if cache is None else cache
    return RandomShootingPlanner(TwoLinkArmModel(), RandomShootingParams(model=cache))


def test_model_learner():
    """Tests that each iteration trains on plans for every task under the training
    belief, and that the data accumulates across sleeps."""
    planner = _make_planner()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    learner: ModelLearner = ModelLearner(
        planner, FiniteTaskDistribution(TASKS), CurrentBeliefSource(), Budget()
    )
    params = planner.get_default_params()
    new_params = learner.learn(
        belief, params, Budget(max_iterations=2), np.random.default_rng(0)
    )
    assert len(learner.dataset) == 2 * len(TASKS)
    assert not _cache_entries(params)
    cache = new_params.model
    assert isinstance(cache, ExactMatchCache)
    assert set(cache.entries) == {
        cache.get_key(get_plan_features(task, belief)) for task in TASKS
    }
    learner.learn(
        belief, new_params, Budget(max_iterations=1), np.random.default_rng(1)
    )
    assert len(learner.dataset) == 3 * len(TASKS)

    with pytest.raises(AssertionError):
        learner.learn(belief, params, Budget(), np.random.default_rng(0))
    with pytest.raises(AssertionError):
        ModelLearner(
            planner,
            FiniteTaskDistribution(TASKS),
            CurrentBeliefSource(),
            Budget(),
            refit="never",
        )


def test_training_times():
    """Tests that the belief source observes each iteration's training time."""
    planner = _make_planner()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    source = _TimedSource()
    learner: ModelLearner = ModelLearner(
        planner, FiniteTaskDistribution(TASKS), source, Budget()
    )
    learner.learn(
        belief,
        planner.get_default_params(),
        Budget(max_iterations=2),
        np.random.default_rng(0),
    )
    assert [b for b, _ in source.observed] == [belief, belief]
    assert all(t > 0 for _, t in source.observed)


def test_refit_timing():
    """Tests that the model is fit after every iteration or once per sleep, on all
    of the data either way."""
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    entries = []
    for refit, expected_fits in (
        ("per_iteration", [3, 6, 9]),
        ("per_sleep", [9]),
    ):
        cache = _CountingCache()
        planner = _make_planner(cache)
        learner: ModelLearner = ModelLearner(
            planner,
            FiniteTaskDistribution(TASKS),
            CurrentBeliefSource(),
            Budget(),
            refit=refit,
        )
        params = learner.learn(
            belief,
            planner.get_default_params(),
            Budget(max_iterations=3),
            np.random.default_rng(0),
        )
        assert cache.fits == expected_fits
        entries.append(set(_cache_entries(params)))
    assert entries[0] == entries[1]


def test_max_sleeps():
    """Tests that the learner stops changing the parameters after max_sleeps."""
    planner = _make_planner()
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    learner: ModelLearner = ModelLearner(
        planner,
        FiniteTaskDistribution(TASKS),
        PriorSource(prior),
        Budget(),
        max_sleeps=1,
    )
    rng = np.random.default_rng(0)
    trained = learner.learn(
        prior, planner.get_default_params(), Budget(max_iterations=1), rng
    )
    assert _cache_entries(trained)
    assert learner.learn(prior, trained, Budget(max_iterations=1), rng) is trained


def test_sampled_training_tasks():
    """Tests that an infinite task distribution trains on new sampled tasks in each
    iteration."""
    planner = _make_planner()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    learner: ModelLearner = ModelLearner(
        planner,
        _GoalDistribution(),
        CurrentBeliefSource(),
        Budget(),
        num_training_tasks=3,
    )
    learner.learn(
        belief,
        planner.get_default_params(),
        Budget(max_iterations=2),
        np.random.default_rng(0),
    )
    goals = learner.dataset.x[:, :2]
    assert len(goals) == 6
    assert len({tuple(goal) for goal in goals}) == 6
    with pytest.raises(AssertionError):
        ModelLearner(
            planner,
            _GoalDistribution(),
            CurrentBeliefSource(),
            Budget(),
            num_training_tasks=0,
        )
