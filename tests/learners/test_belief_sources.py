"""Tests for learners/belief_sources.py."""

import numpy as np

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, TwoLinkArmModel, make_domain
from able.info_gatherers.base import NullInformationGatherer
from able.learners.belief_sources import (
    CurrentBeliefSource,
    ImaginedNextBeliefSource,
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
