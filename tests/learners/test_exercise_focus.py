"""Tests for exercise_focus.py."""

# pylint: disable=no-member  # Domain fields hold their base classes.

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.envs.calendar_arm import CalendarTaskDistribution, make_calendar_arm_domain
from able.learners.belief_sources import CurrentBeliefSource
from able.learners.exercise_focus import ExerciseModelLearner
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.structs import Budget


def _make_learner(domain, num_training_tasks):
    planner = RandomShootingPlanner(
        domain.env_model, RandomShootingParams(), lambda task, belief: np.zeros(1)
    )
    return ExerciseModelLearner(
        planner,
        domain.task_distribution,
        CurrentBeliefSource(),
        Budget(max_iterations=1),
        num_training_tasks=num_training_tasks,
    )


def test_units_cover_the_routine_in_turn():
    """Units for the same belief plan each exercise before any is repeated, and
    then repeat the one planned longest ago or farthest away."""
    domain = make_calendar_arm_domain(0, routine_size=4)
    learner = _make_learner(domain, 1)
    rng = np.random.default_rng(0)
    belief = domain.initial_belief
    chosen = [learner.draw_training_tasks(belief, rng)[0].exercise for _ in range(4)]
    assert sorted(chosen) == [0, 1, 2, 3]


def test_far_belief_replans_the_exercise_with_the_farthest_plan():
    """An exercise planned only for a far belief is planned for this one."""
    domain = make_calendar_arm_domain(0, routine_size=2)
    tasks = domain.task_distribution
    assert isinstance(tasks, CalendarTaskDistribution)
    assert isinstance(domain.initial_belief, IntervalBelief)
    learner = _make_learner(domain, 1)
    rng = np.random.default_rng(0)
    near = domain.initial_belief
    far = IntervalBelief(near.intervals + 0.5)
    first = learner.draw_training_tasks(near, rng)[0].exercise
    second = learner.draw_training_tasks(far, rng)[0].exercise
    assert {first, second} == {0, 1}
    third = learner.draw_training_tasks(near, rng)[0].exercise
    assert third == second


def test_several_exercises_per_unit():
    """A unit plans different exercises."""
    domain = make_calendar_arm_domain(0, routine_size=4)
    learner = _make_learner(domain, 3)
    tasks = learner.draw_training_tasks(domain.initial_belief, np.random.default_rng(0))
    assert len({t.exercise for t in tasks}) == 3
