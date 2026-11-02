"""Tests for info_gatherers/task_execution.py."""

import dataclasses

import numpy as np
import pytest

from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmTask,
    TwoLinkArmModel,
    make_domain,
)
from able.info_gatherers.task_execution import TaskExecutionGatherer
from able.planners.random_shooting import RandomShootingPlanner
from able.structs import Budget


def test_gather_one_task():
    """Tests that one iteration executes one task for a full episode."""
    domain = make_domain(seed=0)
    model = TwoLinkArmModel()
    planner = RandomShootingPlanner(model)
    gatherer = TaskExecutionGatherer(planner, domain.task_distribution, Budget())
    data = gatherer.gather(
        domain.initial_belief,
        planner.get_default_params(),
        domain.real_env,
        Budget(max_iterations=1),
        np.random.default_rng(0),
    )
    assert len(data) == 1
    assert len(data[0]) == model.horizon

    with pytest.raises(AssertionError):
        gatherer.gather(
            domain.initial_belief,
            planner.get_default_params(),
            domain.real_env,
            Budget(),
            np.random.default_rng(0),
        )


class _RecordingTaskDistribution(FiniteTaskDistribution[ArmTask]):
    """Records every task it samples."""

    def __init__(self, tasks: list[ArmTask]) -> None:
        super().__init__(tasks)
        self.sampled: list[ArmTask] = []

    def sample(self, rng: np.random.Generator) -> ArmTask:
        """Sample a task and record it."""
        task = super().sample(rng)
        self.sampled.append(task)
        return task


def test_task_sequence_does_not_depend_on_planning():
    """Tests that planners consuming different amounts of randomness see the same
    tasks."""
    domain = make_domain(seed=0)
    planner = RandomShootingPlanner(TwoLinkArmModel())
    rng = np.random.default_rng(0)
    tasks = [domain.task_distribution.sample(rng) for _ in range(20)]
    sequences = []
    for params in (
        planner.get_default_params(),
        dataclasses.replace(planner.get_default_params(), max_batches=1),
    ):
        task_distribution = _RecordingTaskDistribution(tasks)
        gatherer = TaskExecutionGatherer(planner, task_distribution, Budget())
        gatherer.gather(
            domain.initial_belief,
            params,
            domain.real_env,
            Budget(max_iterations=5),
            np.random.default_rng(1),
        )
        sequences.append([id(task) for task in task_distribution.sampled])
    assert len(sequences[0]) == 5
    assert sequences[0] == sequences[1]


def test_belief_collapses_with_edge_goals():
    """Tests that solving goals at the edge of the range of motion shrinks the belief
    while keeping the true limits in it."""
    updater = JointLimitBeliefUpdater()
    prior_width = IntervalBelief(JOINT_LIMIT_PRIOR).get_total_width()
    final_widths = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        true_limits = rng.uniform(JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 1])
        domain = make_domain(seed, edge_probability=1.0, true_limits=true_limits)
        planner = RandomShootingPlanner(TwoLinkArmModel())
        gatherer = TaskExecutionGatherer(planner, domain.task_distribution, Budget())
        belief = IntervalBelief(JOINT_LIMIT_PRIOR)
        for _ in range(10):
            data = gatherer.gather(
                belief,
                planner.get_default_params(),
                domain.real_env,
                Budget(max_iterations=1),
                rng,
            )
            new_belief = updater.update(belief, data)
            assert new_belief.contains(true_limits)
            assert new_belief.get_total_width() <= belief.get_total_width() + 1e-12
            belief = new_belief
        final_widths.append(belief.get_total_width())
    assert np.mean(final_widths) < 0.5 * prior_width
