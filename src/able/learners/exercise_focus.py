"""A learner for a routine of exercises whose tasks move with the limits (see
envs/calendar.py)."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.envs.calendar import ExerciseTaskDistribution
from able.learners.model_learner import ModelLearner
from able.structs import HiddenParamsType, PlannerParamsType


class ExerciseModelLearner(ModelLearner[HiddenParamsType, PlannerParamsType]):
    """A model learner whose units each plan one or more exercises under a
    training belief, at its midpoint limits.

    A unit plans the exercises whose nearest plan is farthest from the belief, in
    the tasks' own distance, so units for nearby beliefs
    cover the routine between them, whichever rule chose the beliefs. An exercise
    with no plan is the farthest.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._planned: dict[int, list[NDArray[np.float64]]] = {}

    def draw_training_tasks(
        self, belief: Belief[HiddenParamsType], rng: np.random.Generator
    ) -> list[Any]:
        del rng  # unused because the exercises are chosen, not drawn
        tasks = self._task_distribution
        assert isinstance(tasks, ExerciseTaskDistribution), "Need exercises"
        limits = tasks.get_midpoint(belief)
        chosen = []
        for _ in range(min(self._num_training_tasks, tasks.num_exercises)):
            distances = [
                (
                    -1.0
                    if exercise in chosen
                    else min(
                        (
                            tasks.get_distance(limits, other)
                            for other in self._planned.get(exercise, [])
                        ),
                        default=np.inf,
                    )
                )
                for exercise in range(tasks.num_exercises)
            ]
            chosen.append(int(np.argmax(distances)))
        for exercise in chosen:
            self._planned.setdefault(exercise, []).append(limits)
        return [tasks.get_task(exercise, limits) for exercise in chosen]
