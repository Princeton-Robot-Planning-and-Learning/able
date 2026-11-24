"""Belief sources that look several days ahead.

A sleep's training helps every later day that its plans still serve, so a robot
that knows how its world will change can train tonight for a day that one night
cannot prepare. The sources here forecast the belief for each of the coming days
and split a sleep's units with the nights after it in mind.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from able.beliefs.base import Belief
from able.learners.allocation import ALLOCATION_RULES, get_equal_features_transfer
from able.learners.belief_sources import BeliefSource
from able.structs import HiddenParamsType, PlannerParamsType

Transfer = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

# A forecast of the robot's belief some days from now, from its belief now.
Forecast = Callable[[Belief[Any], int], Belief[Any]]

# Needs smaller than this share of a day are left out of the plan.
_MIN_NEED = 0.03
# Cap on the search for the coming nights' plan, in branch-and-bound nodes rather
# than seconds so that a plan does not depend on how busy the machine is.
_MAX_PLANNING_NODES = 1000
# Training beliefs considered per sleep, the ones whose units reach the most need.
_MAX_CANDIDATES = 80


class DayBelief(Belief[HiddenParamsType]):
    """A belief for a day that is a number of days from now."""

    def __init__(self, belief: Belief[HiddenParamsType], days_ahead: int) -> None:
        self.belief = belief
        self.days_ahead = days_ahead

    def sample(self, rng: np.random.Generator) -> HiddenParamsType:
        return self.belief.sample(rng)


class LookaheadSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains for one fixed number of days ahead: under the forecast of the
    robot's belief then."""

    def __init__(self, forecast: Forecast, days_ahead: int) -> None:
        assert days_ahead >= 0, "Cannot train for the past"
        self._forecast = forecast
        self._days_ahead = days_ahead

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        del iteration, params, rng  # unused because the day is fixed
        return self._forecast(belief, self._days_ahead)


class HorizonSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Trains under dreams of the coming days: each draw is for a day uniform over
    today and the next horizon - 1 days, and dreams from the forecast of the
    robot's belief for that day with another source."""

    def __init__(
        self,
        source: BeliefSource[HiddenParamsType, PlannerParamsType],
        forecast: Forecast,
        horizon: int,
    ) -> None:
        assert horizon > 0, "Need at least one day"
        self._source = source
        self._forecast = forecast
        self._horizon = horizon

    @property
    def draws_in_workers(self) -> bool:
        """Whether the other source's draws may be made in workers. The forecast
        changes with the day, which only this process knows."""
        return False

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        return self.draw_day(iteration, belief, params, rng).belief

    def draw_day(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> DayBelief[HiddenParamsType]:
        """A dream of a coming day, with how many days ahead it is."""
        days_ahead = int(rng.integers(self._horizon))
        forecast = self._forecast(belief, days_ahead)
        dream = self._source.get_training_belief(iteration, forecast, params, rng)
        return DayBelief(dream, days_ahead)


@dataclass(frozen=True)
class Tier:
    """A way a day can be served: a belief is served with this value, the chance
    that the day goes well, once run_length effective units transfer to it."""

    transfer: Transfer
    value: float


@dataclass(frozen=True)
class HorizonModel:
    """A prediction of what units of sleep compute buy on each of the coming days.

    Each belief has features. A belief is worth its best tier's value once
    run_length units have transferred to it under that tier's transfer, and
    nothing before. A belief's transfer to another is 1 or 0 at each tier, and
    units add over training beliefs.

    independent_transfer is the transfer under deliberate_practice, which treats
    what each unit trains as a separate skill: by default, beliefs with equal
    features.
    """

    get_features: Callable[[Belief[Any]], NDArray[np.float64]]
    run_length: int
    tiers: tuple[Tier, ...]
    # A belief between two, where training can serve both, if the model has one.
    between: Callable[[Belief[Any], Belief[Any]], Belief[Any]] | None = None
    independent_transfer: Transfer = get_equal_features_transfer

    def __post_init__(self) -> None:
        assert self.run_length > 0, "Need at least one unit to serve a day"
        assert len(self.tiers) > 0, "Need a tier"


@dataclass
class NightsProblem:
    """The coming nights' allocation.

    weights[d, i] is the need of belief i on day d. kernels[m][i, j] is how much a
    unit at candidate j counts at belief i under tier m, and priors[m][i] is what
    earlier sleeps' training already counts there. A unit in night n helps days
    d >= n.
    """

    weights: NDArray[np.float64]
    kernels: list[NDArray[np.float64]]
    priors: list[NDArray[np.float64]]
    values: list[float]
    run_length: int


def allocate_nights(problem: NightsProblem, num_units: int) -> NDArray[np.int64]:
    """Tonight's units at each candidate, from the split of every coming night's
    num_units that maximizes the expected number of days served.

    Solved as a mixed integer program over the nights, each with num_units, and
    only the first night's units are returned, since the next sleep replans.
    Units beyond what serving needs go where they count most.
    """
    num_days, num_beliefs = problem.weights.shape
    num_candidates = problem.kernels[0].shape[1]
    num_tiers = len(problem.kernels)
    needs = [
        (d, i)
        for d in range(num_days)
        for i in range(num_beliefs)
        if problem.weights[d, i] > _MIN_NEED
    ]
    assert needs, "Need at least one belief to serve"
    num_units_vars = num_days * num_candidates
    num_vars = num_units_vars + len(needs) * num_tiers
    num_rows = len(needs) * (num_tiers + 1) + num_days
    matrix = lil_matrix((num_rows, num_vars))
    lower = np.full(num_rows, -np.inf)
    upper = np.full(num_rows, np.inf)
    row = 0
    for k, (d, i) in enumerate(needs):
        first_tier = num_units_vars + k * num_tiers
        for m in range(num_tiers):
            for n in range(d + 1):
                matrix[row, n * num_candidates : (n + 1) * num_candidates] = (
                    problem.kernels[m][i]
                )
            matrix[row, first_tier + m] = -(problem.run_length - 1e-6)
            lower[row] = -problem.priors[m][i]
            row += 1
        matrix[row, first_tier : first_tier + num_tiers] = 1.0
        upper[row] = 1.0
        row += 1
    for n in range(num_days):
        matrix[row, n * num_candidates : (n + 1) * num_candidates] = 1.0
        upper[row] = num_units
        row += 1
    need_by_belief = problem.weights.sum(axis=0)
    reach = sum(
        value * (need_by_belief @ kernel)
        for value, kernel in zip(problem.values, problem.kernels)
    )
    assert isinstance(reach, np.ndarray)
    cost = np.zeros(num_vars)
    for n in range(num_days):
        cost[n * num_candidates : (n + 1) * num_candidates] = (
            -1e-6 * (num_days - n) * reach
        )
    for k, (d, i) in enumerate(needs):
        for m in range(num_tiers):
            cost[num_units_vars + k * num_tiers + m] = (
                -problem.weights[d, i] * problem.values[m]
            )
    integrality = np.zeros(num_vars)
    integrality[:num_candidates] = 1
    integrality[num_units_vars:] = 1
    result = milp(
        c=cost,
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        integrality=integrality,
        bounds=Bounds(
            np.zeros(num_vars),
            np.concatenate(
                [
                    np.full(num_units_vars, num_units),
                    np.ones(len(needs) * num_tiers),
                ]
            ),
        ),
        options={"node_limit": _MAX_PLANNING_NODES, "mip_rel_gap": 0.005},
    )
    units = (
        np.zeros(num_candidates)
        if result.x is None
        else np.round(result.x[:num_candidates])
    )
    while units.sum() > num_units:
        units[np.argmax(units)] -= 1
    while units.sum() < num_units:
        units[np.argmax(reach - 1e-3 * units)] += 1
    return units.astype(np.int64)


class HorizonAllocationSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Splits each sleep's units among dreamed beliefs, planning the coming nights
    together.

    At the start of each sleep the source draws num_units * dreams_per_unit
    dreams of coming days from the candidate source, a HorizonSource. Dreams with
    equal features on a day are one belief, whose need is the share of that day's
    dreams it received, and each day has the same weight. Training beliefs are
    these beliefs and, if the model has a between, the beliefs between any two of
    them. The units go to tonight's share of the best plan for the coming nights
    (see allocate_nights()), under the model's tiers for transfer_practice, and
    under deliberate_practice with one tier in which units serve only the belief
    they were spent on. What earlier sleeps trained counts too.
    """

    def __init__(
        self,
        candidate_source: HorizonSource[HiddenParamsType, PlannerParamsType],
        model: HorizonModel,
        rule: str,
        num_units: int,
        dreams_per_unit: int = 16,
    ) -> None:
        assert rule in ALLOCATION_RULES, f"rule must be one of {ALLOCATION_RULES}"
        assert rule != "one_step_greedy", "Greedy looks one sleep ahead"
        assert num_units > 0 and dreams_per_unit > 0, "Need units and dreams"
        self._candidate_source = candidate_source
        self._model = model
        self._rule = rule
        self._num_units = num_units
        self._dreams_per_unit = dreams_per_unit
        self._trained_features: list[NDArray[np.float64]] = []
        self._schedule: list[Belief[HiddenParamsType]] = []

    @property
    def num_draws(self) -> int:
        """The dreams per unit."""
        return self._dreams_per_unit

    @property
    def draws_in_workers(self) -> bool:
        """Whether the candidate source's draws may be made in workers."""
        return self._candidate_source.draws_in_workers

    def draw(
        self,
        index: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        """One dream from the candidate source, with its day."""
        return self._candidate_source.draw_day(index, belief, params, rng)

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[HiddenParamsType],
        params: PlannerParamsType,
        rng: np.random.Generator,
    ) -> Belief[HiddenParamsType]:
        if iteration == 1:
            num_dreams = self._num_units * self._dreams_per_unit
            dreams = [
                self.draw(i, belief, params, rng) for i in range(1, num_dreams + 1)
            ]
            self._schedule = self.choose_all(dreams, self._num_units, belief)
        return self._schedule[(iteration - 1) % len(self._schedule)]

    def choose(
        self, iteration: int, draws: Sequence[Belief[HiddenParamsType]]
    ) -> Belief[HiddenParamsType]:
        raise NotImplementedError("Allocation chooses a whole sleep with choose_all()")

    def choose_all(
        self,
        draws: Sequence[Belief[HiddenParamsType]],
        num_iterations: int,
        belief: Belief[HiddenParamsType],
    ) -> list[Belief[HiddenParamsType]]:
        """Tonight's training belief for every iteration, from all of the dreams of
        the coming days."""
        del belief  # unused because the dreams include today
        dreams: list[DayBelief[HiddenParamsType]] = []
        for dream in draws:
            assert isinstance(dream, DayBelief), "Need dreams with a day"
            dreams.append(dream)
        features = np.array([self._model.get_features(d.belief) for d in dreams])
        nodes, first, inverse = np.unique(
            features, axis=0, return_index=True, return_inverse=True
        )
        inverse = np.asarray(inverse).reshape(-1)
        days = np.array([d.days_ahead for d in dreams])
        weights = np.zeros((int(days.max()) + 1, len(nodes)))
        np.add.at(weights, (days, inverse), 1.0)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        beliefs: list[Belief[HiddenParamsType]] = [dreams[i].belief for i in first]
        candidates = nodes
        if self._rule == "transfer_practice" and self._model.between is not None:
            extra = [
                self._model.between(beliefs[a], beliefs[b])
                for a in range(len(beliefs))
                for b in range(a + 1, len(beliefs))
                if any(
                    t.transfer(nodes[a : a + 1], nodes[b : b + 1])[0, 0] > 0.0
                    or self._has_middle(nodes[a], nodes[b], t)
                    for t in self._model.tiers
                )
            ]
            if extra:
                extra_features = np.array([self._model.get_features(b) for b in extra])
                candidates = np.vstack([nodes, extra_features])
                beliefs = beliefs + extra
                candidates, keep = np.unique(candidates, axis=0, return_index=True)
                beliefs = [beliefs[i] for i in keep]
        if len(candidates) > _MAX_CANDIDATES:
            need = weights.sum(axis=0)
            reach = sum(
                t.value * (need @ t.transfer(nodes, candidates))
                for t in self._model.tiers
            )
            assert isinstance(reach, np.ndarray)
            best = np.sort(np.argsort(-reach, kind="stable")[:_MAX_CANDIDATES])
            candidates = candidates[best]
            beliefs = [beliefs[i] for i in best]
        tiers = self._model.tiers
        if self._rule == "deliberate_practice":
            tiers = (Tier(self._model.independent_transfer, 1.0),)
        trained = np.array(self._trained_features)
        problem = NightsProblem(
            weights=weights,
            kernels=[t.transfer(nodes, candidates) for t in tiers],
            priors=[
                (
                    t.transfer(nodes, trained).sum(axis=1)
                    if len(trained)
                    else np.zeros(len(nodes))
                )
                for t in tiers
            ],
            values=[t.value for t in tiers],
            run_length=self._model.run_length,
        )
        units = allocate_nights(problem, num_iterations)
        schedule: list[Belief[HiddenParamsType]] = []
        for j in np.argsort(-units, kind="stable"):
            for _ in range(int(units[j])):
                self._trained_features.append(candidates[j])
                schedule.append(beliefs[j])
        return schedule

    def _has_middle(
        self, a: NDArray[np.float64], b: NDArray[np.float64], tier: Tier
    ) -> bool:
        """Whether training midway between two beliefs' features would serve both
        under a tier."""
        middle = ((a + b) / 2.0)[None]
        return bool(
            tier.transfer(a[None], middle)[0, 0] > 0.0
            and tier.transfer(b[None], middle)[0, 0] > 0.0
        )
