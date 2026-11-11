"""Tests for learners/rollout_model_learner.py."""

import numpy as np
import pytest

from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.learners.belief_sources import CurrentBeliefSource
from able.learners.rollout_model_learner import RolloutModelLearner
from able.models.base import Dataset
from able.models.exact_match_cache import ExactMatchCache
from able.planners.base import TrainablePlanner
from able.policies.base import OpenLoopPolicy
from able.structs import Budget, PlanningFailure
from tests.envs.wall_model import WallModel, WallTask

# The wall stands between 2 and 3, so the first goal is reachable and the second is
# not.
BELIEF = IntervalBelief([[2.0, 3.0]])
REACHABLE = WallTask(initial_state=np.zeros(1), goal=1.0)
BLOCKED = WallTask(initial_state=np.zeros(1), goal=5.0)


class _StraightPlanner(TrainablePlanner):
    """Moves straight to the goal, then waits, and records the tasks it gives rows
    for, the policies it plans, and the parameters it plans with."""

    def __init__(self) -> None:
        self.example_tasks: list = []
        self.policies: list = []
        self.planned_params: list = []
        self.training_params = ExactMatchCache()

    def get_default_params(self):
        return ExactMatchCache()

    def get_model(self, params):
        return params

    def set_model(self, params, model):
        return model

    def get_training_params(self, params):
        return self.training_params

    def plan(self, task, belief, params, budget, rng):
        self.planned_params.append(params)
        actions = [np.array([task.goal]), np.zeros(1), np.zeros(1)]
        policy = OpenLoopPolicy(actions)
        self.policies.append(policy)
        return policy

    def get_plan_examples(self, task, belief, policy):
        self.example_tasks.append(task)
        return Dataset(
            x=np.zeros((1, 1)),
            y=np.array([[task.goal]]),
            costs=np.zeros(1),
            groups=np.zeros(1, dtype=np.int64),
        )


def test_only_completed_rollouts_give_rows():
    """Each training plan, made with the planner's training parameters, runs for
    an episode, and only those that reach the goal give rows."""
    planner = _StraightPlanner()
    learner: RolloutModelLearner = RolloutModelLearner(
        planner,
        FiniteTaskDistribution([REACHABLE, BLOCKED]),
        CurrentBeliefSource(),
        Budget(),
        WallModel(),
    )
    params = planner.get_default_params()
    new_params = learner.learn(
        BELIEF, params, Budget(max_iterations=1), np.random.default_rng(0)
    )
    assert planner.example_tasks == [REACHABLE]
    assert planner.planned_params == [planner.training_params] * 2
    assert len(learner.dataset) == 1 and new_params is not params
    # Both plans ran to the end of their episodes.
    for policy in planner.policies:
        with pytest.raises(PlanningFailure):
            policy.get_action(np.zeros(1))


class _AllRolloutsPlanner(_StraightPlanner):
    """Gives a row for every rollout, recording whether it solved its task."""

    def __init__(self) -> None:
        super().__init__()
        self.rollouts: list = []

    def get_rollout_examples(self, task, belief, policy, trajectory, solved):
        self.rollouts.append((task, len(trajectory.actions), solved))
        return self.get_plan_examples(task, belief, policy)


def test_the_planner_can_take_rows_from_every_rollout():
    """Every rollout goes to the planner with its trajectory and whether it solved
    its task."""
    planner = _AllRolloutsPlanner()
    learner: RolloutModelLearner = RolloutModelLearner(
        planner,
        FiniteTaskDistribution([REACHABLE, BLOCKED]),
        CurrentBeliefSource(),
        Budget(),
        WallModel(),
    )
    learner.learn(
        BELIEF,
        planner.get_default_params(),
        Budget(max_iterations=1),
        np.random.default_rng(0),
    )
    assert [(t, s) for t, _, s in planner.rollouts] == [
        (REACHABLE, True),
        (BLOCKED, False),
    ]
    assert all(steps > 0 for _, steps, _ in planner.rollouts)
    assert len(learner.dataset) == 2
