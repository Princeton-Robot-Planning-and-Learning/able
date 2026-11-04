"""Tests for learners/belief_sources.py."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, TwoLinkArmModel, make_domain
from able.info_gatherers.base import NullInformationGatherer
from able.learners.belief_sources import (
    BeliefCoverageSource,
    BeliefSource,
    CurrentBeliefSource,
    ImaginedNextBeliefSource,
    NextCallSource,
    PriorSource,
)
from able.planners.random_shooting import RandomShootingPlanner
from able.structs import Budget
from tests.recording.event_recorder import EventRecorder


def _imagined_next_source(domain, recorder=None) -> ImaginedNextBeliefSource:
    model = TwoLinkArmModel()
    return ImaginedNextBeliefSource(
        model,
        domain.task_distribution,
        RandomShootingPlanner(model),
        JointLimitBeliefUpdater(),
        NullInformationGatherer(),
        gathering_budget=Budget(max_iterations=0),
        execution_budget=Budget(max_iterations=1),
        planning_budget=Budget(),
        recorder=recorder,
    )


class _ScriptedSource(BeliefSource[Any, Any]):
    """Returns the given beliefs in order and records the iterations it sees."""

    def __init__(self, beliefs: list[Belief[Any]]) -> None:
        self._beliefs = iter(beliefs)
        self.iterations: list[int] = []

    def get_training_belief(
        self,
        iteration: int,
        belief: Belief[Any],
        params: Any,
        rng: np.random.Generator,
    ) -> Belief[Any]:
        self.iterations.append(iteration)
        return next(self._beliefs)


def _point(value: float) -> IntervalBelief:
    """A belief that one parameter equals the value."""
    return IntervalBelief([[value, value]])


def _get_low(belief: Belief[Any]) -> NDArray[np.float64]:
    assert isinstance(belief, IntervalBelief)
    return belief.intervals[:, 0]


def test_current_belief_source():
    """Tests that the current belief is returned unchanged."""
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    source: CurrentBeliefSource = CurrentBeliefSource()
    rng = np.random.default_rng(0)
    assert source.get_training_belief(1, belief, params, rng) is belief


def test_prior_source():
    """Tests that the prior is returned whatever the current belief is."""
    prior = IntervalBelief(JOINT_LIMIT_PRIOR)
    narrower = IntervalBelief(JOINT_LIMIT_PRIOR / 2)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    source: PriorSource = PriorSource(prior)
    rng = np.random.default_rng(0)
    assert source.get_training_belief(2, narrower, params, rng) is prior


def test_imagined_next_belief_source():
    """Tests that each dream imagines a narrower belief and is recorded."""
    domain = make_domain(seed=0)
    recorder = EventRecorder()
    source = _imagined_next_source(domain, recorder)
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    for iteration in (1, 2):
        imagined = source.get_training_belief(
            iteration, belief, params, np.random.default_rng(iteration)
        )
        assert isinstance(imagined, IntervalBelief)
        assert imagined.get_total_width() < belief.get_total_width()
    episodes = [e for e in recorder.events if e[0] == "start_episode"]
    assert episodes == [
        ("start_episode", "Dream 1 · imagined task 1", True),
        ("start_episode", "Dream 2 · imagined task 1", True),
    ]
    beliefs = [e for e in recorder.events if e[0] == "record_belief"]
    assert beliefs == [
        ("record_belief", "Dream 1 · imagined belief", True),
        ("record_belief", "Dream 2 · imagined belief", True),
    ]
    outcomes = {e[1] for e in recorder.events if e[0] == "end_episode"}
    assert outcomes <= {"imagined success", "imagined failure", "planning failed"}


def test_imagined_belief_is_known_with_known_limits():
    """Tests that a dream from a known belief imagines the same belief."""
    limits = np.array([-1.2, 1.4, -2.0, 0.9])
    domain = make_domain(seed=0, true_limits=limits)
    known = IntervalBelief(np.stack([limits, limits], axis=1))
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    imagined = _imagined_next_source(domain).get_training_belief(
        1, known, params, np.random.default_rng(0)
    )
    assert imagined == known


def test_belief_coverage_source():
    """Tests that each iteration chooses the candidate farthest from its nearest
    chosen belief, including beliefs chosen in earlier sleeps."""
    candidates = [_point(v) for v in (0, 5, 1, 3, 2, 6, 0, 4.5)]
    candidate_source = _ScriptedSource(candidates)
    source = BeliefCoverageSource(candidate_source, _get_low, num_candidates=2)
    belief = _point(0)
    rng = np.random.default_rng(0)
    # With nothing chosen yet, the first candidate is chosen.
    chosen = [source.get_training_belief(i, belief, None, rng) for i in (1, 2, 3)]
    assert chosen == [_point(0), _point(3), _point(6)]
    # In the next sleep, 0 has been chosen before, and 4.5 is 1.5 from 3 and 6.
    assert source.get_training_belief(1, belief, None, rng) == _point(4.5)
    assert candidate_source.iterations == [1, 2, 3, 4, 5, 6, 1, 2]


def test_belief_coverage_with_one_candidate_is_dreaming():
    """Tests that with one candidate, each training belief is the dream."""
    domain = make_domain(seed=0)
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    params = RandomShootingPlanner(TwoLinkArmModel()).get_default_params()
    dreamer = _imagined_next_source(domain)
    coverage = BeliefCoverageSource(
        _imagined_next_source(domain), _get_low, num_candidates=1
    )
    for iteration in (1, 2, 3):
        dream = dreamer.get_training_belief(
            iteration, belief, params, np.random.default_rng(iteration)
        )
        chosen = coverage.get_training_belief(
            iteration, belief, params, np.random.default_rng(iteration)
        )
        assert chosen == dream


def _get_value(belief: Belief[Any]) -> float:
    assert isinstance(belief, IntervalBelief)
    return float(belief.intervals[0, 0])


def test_next_call_source():
    """Tests that each iteration trains in the region with the largest product of
    probability and estimated saving, and that the saving is measured from the
    training times."""
    # Regions 0, 1, and 2 hold half, a third, and a sixth of the first sleep's
    # candidates, and region 3 holds every candidate of the second sleep.
    candidates = [_point(v) for v in (0, 1, 0, 2, 1, 0)] + [_point(3)] * 6
    candidate_source = _ScriptedSource(candidates)
    source = NextCallSource(candidate_source, _get_value, num_candidates=6)
    # In every region, training takes 10 s with no earlier updates, 4 s after
    # one update, and 3 s after more.
    profile = [10.0, 4.0, 3.0]
    num_updates: dict[float, int] = {}
    belief = _point(0)
    rng = np.random.default_rng(0)

    def train(iteration: int) -> float:
        training_belief = source.get_training_belief(iteration, belief, None, rng)
        region = _get_value(training_belief)
        n = num_updates.get(region, 0)
        source.observe_training_time(training_belief, profile[min(n, 2)])
        num_updates[region] = n + 1
        return region

    # With nothing measured, the most probable region comes first, and region 0
    # keeps its lead until its updates measure the first two savings, 6 s and 1
    # s. After that, a first update to a region is worth 6 s times the region's
    # probability, and a later update 1 s or less.
    regions = [train(i) for i in range(1, 9)]
    assert regions == [0, 0, 0, 1, 2, 0, 1, 2]
    assert candidate_source.iterations == [1, 2, 3, 4, 5, 6]
    assert source.estimate_saving(0) == 6.0
    assert source.estimate_saving(1) == 1.0
    # An update after three is assumed to save as much as the third, which saved
    # nothing.
    assert source.estimate_saving(4) == 0.0
    # The next sleep draws new candidates and trains in the new region.
    assert train(1) == 3
    assert candidate_source.iterations[6:] == [1, 2, 3, 4, 5, 6]
