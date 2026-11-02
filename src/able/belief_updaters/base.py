"""Base class for belief updaters."""

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from able.beliefs.base import Belief
from able.structs import ActType, HiddenParamsType, StateType, Trajectory


class BeliefUpdater(ABC, Generic[StateType, ActType, HiddenParamsType]):
    """Computes a new belief from data gathered in an environment.

    Updaters do not modify their inputs or keep state, so learners can apply them
    to imagined data without affecting the robot's actual belief.
    """

    @abstractmethod
    def update(
        self,
        belief: Belief[HiddenParamsType],
        data: Sequence[Trajectory[StateType, ActType]],
    ) -> Belief[HiddenParamsType]:
        """Return the belief after observing the data."""
