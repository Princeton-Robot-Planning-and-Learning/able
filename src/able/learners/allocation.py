"""Belief sources that decide how many units of a sleep each belief gets.

A sleep has a budget of units, and each unit trains under one belief. The
sources here predict need by dreaming many times, predict what units of compute
buy at each dreamed belief with an allocation model, split the sleep's units
among the dreamed beliefs, and then train under them in turn. They adapt
Deliberate Practice (Vats, Harithas, Akbulut, Raghunathan, and Konidaris,
arXiv:2608.13415, 2026), which splits a budget of practice episodes among robot
skills by weighing what each skill's practice buys.

The rules differ in how they split the units:
- transfer_practice weighs each belief's need against what units buy there,
  counting what units at one belief buy at the others. It adds whole runs of
  units at a time, then moves units between beliefs while that helps.
- deliberate_practice does the same but assumes that units at one belief buy
  nothing at any other, as Deliberate Practice assumes for skills.
- one_step_greedy gives each unit to the belief where one more unit buys the
  most, counting transfer.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp

from able.beliefs.base import Belief
from able.learners.belief_sources import BeliefSource
from able.structs import HiddenParamsType, PlannerParamsType

ALLOCATION_RULES = ("transfer_practice", "deliberate_practice", "one_step_greedy")
COMBINE_RULES = ("sum", "max")

# Gains closer than this count as equal.
_TOLERANCE = 1e-9


def get_equal_features_transfer(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Transfer of 1 between beliefs with equal features and 0 otherwise."""
    return np.all(a[:, None, :] == b[None, :, :], axis=2).astype(np.float64)


@dataclass(frozen=True)
class AllocationModel:
    """A prediction of what units of sleep compute buy at each belief.

    Each belief has features. transfer(a, b) gives, for rows of features a and
    b, how much a unit at each belief of b counts at each belief of a, which is
    1 for equal features. gain(e) is the predicted fall in planning cost at a
    belief that has e effective units, with gain(0) = 0.

    With combine "sum", effective units add over training beliefs: a belief's
    effective units are the transfer-weighted sum of every training belief's
    units, and its gain is gain() of that sum. With combine "max", a belief
    benefits from its best training belief: its gain is the largest transfer
    times gain() of a training belief's units, as when a planner uses the
    nearest stored solution.

    If run_length is set, gain() must be get_cliff_gain(run_length): a belief
    gains 1 once it has run_length effective units. The runs rules then also
    search over sets of beliefs to make fast.

    independent_transfer is the transfer under deliberate_practice, which treats
    what each unit trains as a separate skill: by default, beliefs with equal
    features. include_current_belief makes the robot's current belief a
    candidate.
    """

    get_features: Callable[[Belief[Any]], NDArray[np.float64]]
    transfer: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
    gain: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    combine: str = "sum"
    run_length: float | None = None
    independent_transfer: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ] = get_equal_features_transfer
    include_current_belief: bool = True

    def __post_init__(self) -> None:
        assert self.combine in COMBINE_RULES, f"combine must be one of {COMBINE_RULES}"


def get_cliff_gain(
    run_length: float,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """A gain of 1 once a belief has run_length effective units and 0 before."""

    def gain(effective_units: NDArray[np.float64]) -> NDArray[np.float64]:
        return (effective_units >= run_length - _TOLERANCE).astype(np.float64)

    return gain


def get_diminishing_gain(
    first: float, decay: float
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """A gain of first after one effective unit that approaches 1 with more:
    1 - (1 - first) * decay^(e - 1) for e >= 1, and 0 for e < 1."""

    def gain(effective_units: NDArray[np.float64]) -> NDArray[np.float64]:
        extra = np.maximum(effective_units - 1.0, 0.0)
        values = 1.0 - (1.0 - first) * decay**extra
        return np.where(effective_units >= 1.0 - _TOLERANCE, values, 0.0)

    return gain


def get_saturating_gain(
    scale: float, power: float
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """A gain that rises smoothly toward 1: 1 - (1 + e / scale)^(-power)."""

    def gain(effective_units: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 - (1.0 + np.maximum(effective_units, 0.0) / scale) ** (-power)

    return gain


def get_ramp_gain(
    start: float, end: float
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """A gain of 0 up to start effective units that rises linearly to 1 at end."""
    assert 0.0 <= start < end, "Need 0 <= start < end"

    def gain(effective_units: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip((effective_units - start) / (end - start), 0.0, 1.0)

    return gain


def make_moment_transfer(
    length: float, narrowing_rate: float
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Transfer between beliefs described by moment features, each parameter's
    mean and then each one's standard deviation. It falls with the distance d
    between the means as exp(-d^2 / (2 length^2)), and by exp(-narrowing_rate *
    s) when the training belief's standard deviations sum to s less than the
    other belief's: training under a narrower belief transfers less, and under a
    wider one transfers as well as under an equal one."""

    def transfer(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        num_params = a.shape[1] // 2
        means = ((a[:, None, :num_params] - b[None, :, :num_params]) ** 2).sum(axis=2)
        widths = (b[None, :, num_params:] - a[:, None, num_params:]).sum(axis=2)
        return np.exp(
            -means / (2.0 * length**2) + narrowing_rate * np.minimum(widths, 0.0)
        )

    return transfer


@dataclass
class _Problem:
    """One sleep's allocation: need weights over candidate beliefs, the transfer
    from each candidate to each candidate, and what earlier sleeps' training
    already buys at each candidate."""

    weights: NDArray[np.float64]
    kernel: NDArray[np.float64]
    gain: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    combine: str
    # Effective units at each candidate for "sum", or its gain for "max".
    prior: NDArray[np.float64]
    # The effective units that make a candidate fast, if the gain is a cliff.
    run_length: float | None = None

    def values(self, units: NDArray[np.float64]) -> NDArray[np.float64]:
        """The gain at each candidate under an allocation."""
        if self.combine == "sum":
            return self.gain(self.prior + self.kernel @ units)
        contributions = self.kernel * self.gain(units)[None, :]
        return np.maximum(self.prior, contributions.max(axis=1, initial=0.0))

    def objective(self, units: NDArray[np.float64]) -> float:
        """The need-weighted gain of an allocation."""
        return float(self.weights @ self.values(units))

    def objectives_after_adding(
        self, units: NDArray[np.float64], amount: int
    ) -> NDArray[np.float64]:
        """The objective after adding amount units to each candidate in turn."""
        if self.combine == "sum":
            effective = self.prior + self.kernel @ units
            values = self.gain(effective[:, None] + amount * self.kernel)
            return self.weights @ values
        current = self.values(units)
        added = self.kernel * self.gain(units + amount)[None, :]
        return self.weights @ np.maximum(current[:, None], added)

    def reach(self) -> NDArray[np.float64]:
        """The need each candidate's units transfer to."""
        return self.weights @ self.kernel


def _allocate_by_runs(problem: _Problem, num_units: int) -> NDArray[np.float64]:
    """Repeatedly add the run of units at one candidate with the best gain per
    unit, then move units between candidates while that raises the objective."""
    units = np.zeros(len(problem.weights))
    reach = problem.reach()
    while units.sum() < num_units:
        remaining = int(num_units - units.sum())
        current = problem.objective(units)
        best: tuple[float, float, float] | None = None
        best_move = (0, 0)
        for length in range(1, remaining + 1):
            gains = problem.objectives_after_adding(units, length) - current
            for j in np.flatnonzero(gains > _TOLERANCE):
                key = (gains[j] / length, gains[j], reach[j])
                if best is None or key > best:
                    best, best_move = key, (int(j), length)
        if best is None:
            break
        units[best_move[0]] += best_move[1]
    return _improve_by_moves(problem, units)


def _improve_by_moves(
    problem: _Problem, units: NDArray[np.float64], max_moves: int = 200
) -> NDArray[np.float64]:
    """Move runs of units from one candidate to another while the best such move
    raises the objective."""
    for _ in range(max_moves):
        current = problem.objective(units)
        best_gain, best_move = _TOLERANCE, None
        for source in np.flatnonzero(units > 0):
            for length in range(1, int(units[source]) + 1):
                reduced = units.copy()
                reduced[source] -= length
                gains = problem.objectives_after_adding(reduced, length) - current
                gains[source] = -np.inf
                target = int(np.argmax(gains))
                if gains[target] > best_gain:
                    best_gain, best_move = gains[target], (int(source), target, length)
        if best_move is None:
            break
        source, target, length = best_move
        units[source] -= length
        units[target] += length
    return units


def _get_min_units(
    problem: _Problem, targets: list[int], num_units: int
) -> NDArray[np.float64] | None:
    """The fewest units that give every target run_length effective units, or
    None if num_units cannot. Solved as a small integer program."""
    assert problem.run_length is not None, "Targets need a cliff"
    rows = problem.kernel[targets]
    needed = problem.run_length - problem.prior[targets] - _TOLERANCE
    result = milp(
        c=np.ones(len(problem.weights)),
        constraints=[
            LinearConstraint(rows, lb=needed),
            LinearConstraint(np.ones((1, len(problem.weights))), ub=num_units),
        ],
        integrality=np.ones(len(problem.weights)),
        bounds=Bounds(0, num_units),
    )
    return None if result.x is None else np.round(result.x)


def _allocate_by_targets(problem: _Problem, num_units: int) -> NDArray[np.float64]:
    """Grow a set of candidates to make fast. Each step adds the candidate that
    buys the most need per extra unit, where the units a set needs come from a
    small covering problem. Only for cliff gains that add over training beliefs."""
    units = np.zeros(len(problem.weights))
    targets: list[int] = []
    while True:
        current = problem.objective(units)
        best: tuple[float, float] | None = None
        best_units, best_target = units, -1
        for i in np.argsort(-problem.weights, kind="stable"):
            if i in targets or problem.values(units)[i] >= 1.0:
                continue
            candidate = _get_min_units(problem, targets + [int(i)], num_units)
            if candidate is None:
                continue
            gain = problem.objective(candidate) - current
            extra = candidate.sum() - units.sum()
            key = (np.inf if extra <= 0 else gain / extra, gain)
            if gain > _TOLERANCE and (best is None or key > best):
                best, best_units, best_target = key, candidate, int(i)
        if best is None:
            return units
        units = best_units
        targets.append(best_target)


def _allocate_one_unit_at_a_time(
    problem: _Problem, num_units: int
) -> NDArray[np.float64]:
    """Give each unit to the candidate where one more unit raises the objective
    most, breaking ties toward the candidate whose units reach the most need,
    including when no unit raises it."""
    units = np.zeros(len(problem.weights))
    reach = problem.reach()
    for _ in range(num_units):
        gains = problem.objectives_after_adding(units, 1) - problem.objective(units)
        best = np.flatnonzero(gains >= gains.max() - _TOLERANCE)
        units[best[np.argmax(reach[best])]] += 1
    return units


def _get_progress(problem: _Problem, units: NDArray[np.float64]) -> NDArray[np.float64]:
    """For each candidate, the need of the candidates still short of the run
    length that one more unit there counts toward, or zeros if the problem has
    no run length to add units toward."""
    if problem.run_length is None or problem.combine != "sum":
        return np.zeros(len(units))
    effective = problem.prior + problem.kernel @ units
    short = effective < problem.run_length - _TOLERANCE
    return (problem.weights * short) @ problem.kernel


def allocate(problem: _Problem, num_units: int, rule: str) -> NDArray[np.int64]:
    """Split num_units among the problem's candidates by the rule. Units that no
    candidate gains from go where they count toward the most need still short of
    the run length, since a later sleep can finish the run, and otherwise to
    candidates in order of need."""
    assert rule in ALLOCATION_RULES, f"rule must be one of {ALLOCATION_RULES}"
    if rule == "one_step_greedy":
        units = _allocate_one_unit_at_a_time(problem, num_units)
    else:
        units = _allocate_by_runs(problem, num_units)
        if problem.run_length is not None and problem.combine == "sum":
            by_targets = _allocate_by_targets(problem, num_units)
            if problem.objective(by_targets) > problem.objective(units) + _TOLERANCE:
                units = by_targets
    order = np.argsort(-problem.weights, kind="stable")
    position = 0
    while units.sum() < num_units:
        progress = _get_progress(problem, units)
        if progress.max() > _TOLERANCE:
            units[np.argmax(progress)] += 1
        else:
            units[order[position % len(order)]] += 1
            position += 1
    return units.astype(np.int64)


class AllocationSource(BeliefSource[HiddenParamsType, PlannerParamsType]):
    """Splits each sleep's units among dreamed beliefs by an allocation rule.

    At the start of each sleep the source dreams num_units * dreams_per_unit
    times. Dreams with equal features are one candidate, whose need is the
    share of dreams it received, and the robot's current belief is a candidate
    with no need. The rule splits the units among the candidates, counting what
    earlier sleeps' training already buys at each one, since the planner keeps
    what it learned, unless remember_training is false, for a planner that
    keeps only the last sleep's training. The sleep then trains under
    each candidate as many times as it has units, the candidates with the most
    units first.

    The candidate source, usually an ImaginedNextBeliefSource, sees
    num_units * dreams_per_unit iterations per sleep, numbered from 1.
    """

    def __init__(
        self,
        candidate_source: BeliefSource[HiddenParamsType, PlannerParamsType],
        model: AllocationModel,
        rule: str,
        num_units: int,
        dreams_per_unit: int = 4,
        remember_training: bool = True,
    ) -> None:
        assert rule in ALLOCATION_RULES, f"rule must be one of {ALLOCATION_RULES}"
        assert num_units > 0 and dreams_per_unit > 0, "Need units and dreams"
        self._candidate_source = candidate_source
        self._model = model
        self._rule = rule
        self._num_units = num_units
        self._dreams_per_unit = dreams_per_unit
        self._remember_training = remember_training
        self._trained_features: list[NDArray[np.float64]] = []
        self._trained_units: list[int] = []
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
        """One dream from the candidate source."""
        return self._candidate_source.get_training_belief(index, belief, params, rng)

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
        """The training belief of every iteration of a sleep, from all of its
        dreams. The robot's current belief is also a candidate, with no need of
        its own, since training under it can help the beliefs that follow it."""
        dreams = list(draws)
        features = np.array([self._model.get_features(d) for d in dreams])
        candidates, first, counts = np.unique(
            features, axis=0, return_index=True, return_counts=True
        )
        weights = counts / len(dreams)
        beliefs = [dreams[i] for i in first]
        if self._model.include_current_belief:
            current = self._model.get_features(belief)
            if not np.any(np.all(candidates == current, axis=1)):
                candidates = np.vstack([candidates, current])
                weights = np.append(weights, 0.0)
                beliefs.append(belief)
        problem = self._make_problem(candidates, weights)
        units = allocate(problem, num_iterations, self._rule)
        order = np.lexsort((-problem.weights, -units))
        schedule = []
        for j in order:
            if units[j] > 0:
                self._trained_features.append(candidates[j])
                self._trained_units.append(int(units[j]))
                schedule.extend([beliefs[j]] * int(units[j]))
        return schedule

    def _transfer(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._rule == "deliberate_practice":
            return self._model.independent_transfer(a, b)
        return self._model.transfer(a, b)

    def _make_problem(
        self, candidates: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> _Problem:
        model = self._model
        prior = np.zeros(len(candidates))
        if not self._remember_training:
            self._trained_features, self._trained_units = [], []
        if self._trained_features:
            trained = np.array(self._trained_features)
            trained_units = np.array(self._trained_units, dtype=np.float64)
            kernel = self._transfer(candidates, trained)
            if model.combine == "sum":
                prior = kernel @ trained_units
            else:
                prior = (kernel * model.gain(trained_units)[None, :]).max(axis=1)
        return _Problem(
            weights=weights,
            kernel=self._transfer(candidates, candidates),
            gain=model.gain,
            combine=model.combine,
            prior=prior,
            run_length=model.run_length,
        )
