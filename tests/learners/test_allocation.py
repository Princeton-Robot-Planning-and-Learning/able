"""Tests for learners/allocation.py."""

from typing import Any

import numpy as np
import pytest

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.learners.allocation import (
    AllocationModel,
    AllocationSource,
    _Problem,
    allocate,
    get_cliff_gain,
    get_diminishing_gain,
    get_equal_features_transfer,
    get_ramp_gain,
    get_saturating_gain,
    make_moment_transfer,
)
from able.learners.belief_sources import BeliefSource


def _toy_problem(kappa: float, combine: str = "sum") -> _Problem:
    """Five values on a line with need (0.1, 0.25, 0.3, 0.25, 0.1); a value is
    fast with 6 effective units, and a unit counts kappa^d at distance d."""
    distances = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    return _Problem(
        weights=np.array([0.1, 0.25, 0.3, 0.25, 0.1]),
        kernel=kappa**distances,
        gain=get_cliff_gain(6.0),
        combine=combine,
        prior=np.zeros(5),
        run_length=6.0,
    )


def test_transfer_practice_spreads_units_to_use_transfer():
    """Tests that with transfer, the runs rule finds the split that makes the
    three middle values fast, while one unit at a time puts everything on the
    most likely value and ignoring transfer makes only two values fast."""
    problem = _toy_problem(0.5)
    units = allocate(problem, 10, "transfer_practice")
    np.testing.assert_array_equal(units, [0, 4, 2, 4, 0])
    assert problem.objective(units.astype(float)) == pytest.approx(0.8)
    greedy = allocate(problem, 10, "one_step_greedy")
    assert problem.objective(greedy.astype(float)) == pytest.approx(0.3)
    independent = allocate(_toy_problem(0.0), 10, "transfer_practice")
    assert independent[2] == 6
    assert problem.objective(independent.astype(float)) == pytest.approx(0.55)


def test_max_combine_covers_need():
    """Tests that when a value benefits from its best trained neighbor, units go
    where they cover the most uncovered need."""
    distances = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :])
    problem = _Problem(
        weights=np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1]),
        kernel=(distances <= 1).astype(float),
        gain=get_cliff_gain(1.0),
        combine="max",
        prior=np.zeros(6),
    )
    units = allocate(problem, 2, "transfer_practice")
    np.testing.assert_array_equal(units, [0, 1, 0, 0, 1, 0])
    assert problem.objective(units.astype(float)) == pytest.approx(1.0)


def test_units_without_gain_go_to_likely_candidates():
    """Tests that units that buy nothing still go to candidates, by need."""
    problem = _Problem(
        weights=np.array([0.2, 0.8]),
        kernel=np.eye(2),
        gain=get_cliff_gain(1.0),
        combine="max",
        prior=np.ones(2),
    )
    assert allocate(problem, 3, "deliberate_practice").tolist() == [1, 2]


def test_leftover_units_go_toward_unfinished_runs():
    """Tests that units too few to finish a run go where they count toward the
    most need still short of the run length, rather than to need already met."""
    problem = _Problem(
        weights=np.array([0.5, 0.2, 0.2, 0.1]),
        kernel=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0] * 3 + [1.0],
            ]
        ),
        gain=get_cliff_gain(4.0),
        combine="sum",
        prior=np.array([4.0, 0.0, 0.0, 0.0]),
        run_length=4.0,
    )
    units = allocate(problem, 2, "transfer_practice")
    assert units[0] == 0 and units[3] == 0 and units[1] + units[2] == 2


def test_equal_features_transfer():
    """Tests that transfer is 1 between equal features and 0 otherwise."""
    a = np.array([[0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(get_equal_features_transfer(a, a), np.eye(2))


def _point(value: float) -> IntervalBelief:
    return IntervalBelief(np.array([[value, value]]))


class _ListSource(BeliefSource[Any, Any]):
    """Returns the given beliefs in order, cycling."""

    def __init__(self, beliefs: list[Belief[Any]]) -> None:
        self._beliefs = beliefs

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[Any],
        params: Any,
        rng: np.random.Generator,
    ) -> Belief[Any]:
        return self._beliefs[(iteration - 1) % len(self._beliefs)]


def _lower_bound(belief: Belief[Any]) -> np.ndarray:
    assert isinstance(belief, IntervalBelief)
    return belief.intervals[:, 0].copy()


def _source(rule: str, dreams: list[float]) -> AllocationSource[Any, Any]:
    model = AllocationModel(
        get_features=_lower_bound,
        transfer=lambda a, b: (np.abs(a[:, None, 0] - b[None, :, 0]) <= 1).astype(
            float
        ),
        gain=get_cliff_gain(1.0),
        combine="max",
    )
    return AllocationSource(
        _ListSource([_point(v) for v in dreams]),
        model,
        rule,
        num_units=2,
        dreams_per_unit=4,
    )


def test_allocation_source_trains_where_need_is_covered():
    """Tests that a sleep trains under the candidates the rule chooses, that
    candidates are dreams with equal features, and that what earlier sleeps
    trained counts in later ones."""
    dreams = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0]
    source = _source("transfer_practice", dreams)
    rng = np.random.default_rng(0)
    chosen = [
        source.get_training_belief(i, _point(0.0), None, rng).intervals[0, 0]
        for i in (1, 2)
    ]
    # 5.0 covers 3/8 of the dreams; 0.0 or 1.0 covers 4/8 between them.
    assert sorted(chosen) in ([0.0, 5.0], [1.0, 5.0])
    # In the next sleep, only 9.0 is uncovered, and the spare unit goes to the
    # most needed candidate.
    later = [
        source.get_training_belief(i, _point(0.0), None, rng).intervals[0, 0]
        for i in (1, 2)
    ]
    assert sorted(later) == [5.0, 9.0]


def test_deliberate_practice_ignores_transfer():
    """Tests that deliberate practice ranks candidates by their own need."""
    dreams = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5.0, 5.0]
    source = _source("deliberate_practice", dreams)
    schedule = source.choose_all([_point(v) for v in dreams], 2, _point(0.0))
    assert sorted(b.intervals[0, 0] for b in schedule) == [0.0, 1.0]
    transfer = _source("transfer_practice", dreams)
    schedule = transfer.choose_all([_point(v) for v in dreams], 2, _point(0.0))
    assert sorted(b.intervals[0, 0] for b in schedule)[1] == 5.0


def test_current_belief_is_a_candidate():
    """Tests that a sleep can train under the current belief, which no dream
    matches, when its transfer covers the dreams."""
    dreams = [0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0]
    source = _source("transfer_practice", dreams)
    schedule = source.choose_all([_point(v) for v in dreams], 2, _point(1.0))
    assert 1.0 in [b.intervals[0, 0] for b in schedule]
    independent = _source("deliberate_practice", dreams)
    schedule = independent.choose_all([_point(v) for v in dreams], 2, _point(1.0))
    assert 1.0 not in [b.intervals[0, 0] for b in schedule]


def test_gains_and_moment_transfer():
    """Tests the shapes of the gains and the asymmetry of moment transfer."""
    units = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(get_diminishing_gain(0.9, 0.5)(units), [0.0, 0.9, 0.95])
    saturating = get_saturating_gain(1.0, 1.0)(units)
    np.testing.assert_allclose(saturating, [0.0, 0.5, 2.0 / 3.0])
    ramp = get_ramp_gain(1.0, 4.0)(np.array([0.0, 1.0, 2.5, 4.0, 8.0]))
    np.testing.assert_allclose(ramp, [0.0, 0.0, 0.5, 1.0, 1.0])
    transfer = make_moment_transfer(length=1.0, narrowing_rate=1.0)
    narrow, wide = np.array([[0.0, 1.0]]), np.array([[0.0, 2.0]])
    assert transfer(narrow, wide)[0, 0] == pytest.approx(1.0)
    assert transfer(wide, narrow)[0, 0] == pytest.approx(np.exp(-1.0))
    shifted = np.array([[1.0, 1.0]])
    assert transfer(shifted, narrow)[0, 0] == pytest.approx(np.exp(-0.5))


def test_forgetting_and_independent_transfer():
    """Tests that a source without memory ignores earlier sleeps, and that
    deliberate practice uses the model's independent transfer."""
    dreams = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0]
    model = AllocationModel(
        get_features=_lower_bound,
        transfer=lambda a, b: (np.abs(a[:, None, 0] - b[None, :, 0]) <= 1).astype(
            float
        ),
        gain=get_cliff_gain(1.0),
        combine="max",
        independent_transfer=lambda a, b: np.ones((len(a), len(b))),
        include_current_belief=False,
    )
    forgetful = AllocationSource(
        _ListSource([_point(v) for v in dreams]),
        model,
        "transfer_practice",
        num_units=2,
        remember_training=False,
    )
    first = forgetful.choose_all([_point(v) for v in dreams], 2, _point(7.0))
    second = forgetful.choose_all([_point(v) for v in dreams], 2, _point(7.0))
    assert [b.intervals[0, 0] for b in first] == [b.intervals[0, 0] for b in second]
    assert 7.0 not in [b.intervals[0, 0] for b in first]
    # Under independent transfer of 1 everywhere, one unit covers everything.
    independent = AllocationSource(
        _ListSource([_point(v) for v in dreams]), model, "deliberate_practice", 2
    )
    schedule = independent.choose_all([_point(v) for v in dreams], 2, _point(7.0))
    assert len(schedule) == 2
