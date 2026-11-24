"""Tests for horizon.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.learners.belief_sources import CurrentBeliefSource
from able.learners.horizon import (
    DayBelief,
    HorizonAllocationSource,
    HorizonModel,
    HorizonSource,
    LookaheadSource,
    NightsProblem,
    Tier,
    allocate_nights,
)


def _make_belief(place: float) -> IntervalBelief:
    return IntervalBelief([[place, place]])


def _shift(belief, days_ahead):
    return IntervalBelief(belief.intervals + days_ahead)


def _step_transfer(radius):
    def transfer(a, b):
        return (np.abs(a[:, None, 0] - b[None, :, 0]) <= radius + 1e-9).astype(float)

    return transfer


def _make_model(run_length, tiers, between=True):
    return HorizonModel(
        get_features=lambda b: b.intervals.mean(axis=1),
        run_length=run_length,
        tiers=tuple(Tier(_step_transfer(r), v) for r, v in tiers),
        between=(
            (lambda a, b: IntervalBelief((a.intervals + b.intervals) / 2.0))
            if between
            else None
        ),
    )


def test_lookahead_source_forecasts():
    """Trains under the forecast a fixed number of days ahead."""
    source = LookaheadSource(_shift, 3)
    belief = source.get_training_belief(1, _make_belief(0.0), None, None)
    assert np.allclose(belief.intervals, [[3.0, 3.0]])


def test_horizon_source_draws_coming_days():
    """Draws are for days within the horizon, with the forecast of that day."""
    source = HorizonSource(CurrentBeliefSource(), _shift, 3)
    rng = np.random.default_rng(0)
    dreams = [source.draw_day(i, _make_belief(0.0), None, rng) for i in range(60)]
    assert {d.days_ahead for d in dreams} == {0, 1, 2}
    for d in dreams:
        assert np.allclose(d.belief.intervals, [[d.days_ahead, d.days_ahead]])


def _make_problem(kernel, weights, run_length, prior=None):
    num_beliefs = kernel.shape[0]
    return NightsProblem(
        weights=weights,
        kernels=[kernel],
        priors=[np.zeros(num_beliefs) if prior is None else prior],
        values=[1.0],
        run_length=run_length,
    )


def test_nights_anticipate_a_day_one_night_cannot_prepare():
    """Training starts early for a day that needs more than one night's units."""
    weights = np.zeros((2, 2))
    weights[0, 0] = 1.0
    weights[1, 1] = 1.0
    # Today needs four units and has only this night's two, so it cannot be served.
    # Tomorrow has two nights, and needs both for its own four.
    units = allocate_nights(_make_problem(np.eye(2), weights, run_length=4), 2)
    assert units.tolist() == [0, 2]


def test_nights_prepare_days_before_they_come():
    """A unit only serves the days from its own night on."""
    weights = np.zeros((2, 2))
    weights[0, 0] = 1.0
    weights[1, 1] = 1.0
    # Today needs two units and has only this night; tomorrow can wait.
    units = allocate_nights(_make_problem(np.eye(2), weights, 2), 2)
    assert units.tolist() == [2, 0]


def test_nights_count_earlier_training():
    """What earlier sleeps trained is not trained again."""
    weights = np.zeros((1, 2))
    weights[0, :] = 0.5
    problem = _make_problem(np.eye(2), weights, 2, prior=np.array([2.0, 0.0]))
    assert allocate_nights(problem, 2).tolist() == [0, 2]


def test_units_without_need_are_still_spent():
    """Every unit is spent even when every need is met."""
    weights = np.ones((1, 1))
    problem = _make_problem(np.eye(1), weights, 1, prior=np.array([5.0]))
    assert allocate_nights(problem, 3).tolist() == [3]


def _make_dreams(places_by_day, count=4):
    return [
        DayBelief(_make_belief(place), day)
        for day, place in places_by_day
        for _ in range(count)
    ]


def test_transfer_practice_trains_between_days():
    """One run midway between two days serves both, so transfer_practice chooses
    it."""
    # Days at 0 and 2, with transfer to within 1, so training at 1 serves both.
    dreams = _make_dreams([(1, 0.0), (2, 2.0)])
    source = HorizonAllocationSource(
        HorizonSource(CurrentBeliefSource(), _shift, 3),
        _make_model(4, [(1, 1.0)]),
        "transfer_practice",
        4,
        2,
    )
    schedule = source.choose_all(dreams, 4, _make_belief(0.0))
    places = {float(b.intervals[0, 0]) for b in schedule}
    assert places == {1.0}


def test_deliberate_practice_ignores_transfer():
    """deliberate_practice trains only for day types themselves."""
    dreams = _make_dreams([(1, 0.0), (2, 2.0)])
    source = HorizonAllocationSource(
        HorizonSource(CurrentBeliefSource(), _shift, 3),
        _make_model(4, [(1, 1.0)]),
        "deliberate_practice",
        4,
        2,
    )
    schedule = source.choose_all(dreams, 4, _make_belief(0.0))
    assert {float(b.intervals[0, 0]) for b in schedule} <= {0.0, 2.0}


def test_training_is_remembered_across_sleeps():
    """Training a day type makes the next sleep look elsewhere."""
    dreams = _make_dreams([(0, 0.0), (1, 5.0)])
    source = HorizonAllocationSource(
        HorizonSource(CurrentBeliefSource(), _shift, 3),
        _make_model(4, [(1, 1.0)]),
        "deliberate_practice",
        4,
        2,
    )
    first = source.choose_all(dreams, 4, _make_belief(0.0))
    second = source.choose_all(dreams, 4, _make_belief(0.0))
    assert {float(b.intervals[0, 0]) for b in first} == {0.0}
    assert {float(b.intervals[0, 0]) for b in second} == {5.0}


def test_greedy_has_no_horizon():
    """The one-step rule does not plan several nights."""
    with pytest.raises(AssertionError):
        HorizonAllocationSource(
            HorizonSource(CurrentBeliefSource(), _shift, 3),
            _make_model(2, [(1, 1.0)]),
            "one_step_greedy",
            2,
        )
