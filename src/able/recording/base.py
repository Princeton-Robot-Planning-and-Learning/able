"""Base class for recorders, which observe each stage of each cycle."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, Sequence

from able.beliefs.base import Belief
from able.structs import HiddenParamsType, QueryResponse, StateType, TaskType


class Stage(Enum):
    """The stages of a cycle, in order."""

    SLEEP = "sleep"
    QUERIES = "queries"
    EXECUTION = "execution"
    EVALUATION = "evaluation"


class CycleRecorder(ABC, Generic[StateType, TaskType, HiddenParamsType]):
    """Observes each stage of each cycle, for example to make videos.

    The cycle loop reports stages, episodes, and queries. Learners can report what
    they imagine while the robot sleeps. Recorders only observe; they never change
    the robot or the environment. Work that takes time should wait for
    end_stage(), which is called outside every timed section.
    """

    @abstractmethod
    def start_stage(self, cycle: int, stage: Stage) -> None:
        """A stage of the cycle is starting."""

    @abstractmethod
    def end_stage(self) -> None:
        """The current stage has ended."""

    @abstractmethod
    def start_episode(
        self,
        label: str,
        task: TaskType,
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        """An episode for the task is starting under the belief.

        In dreams, the hypothesis is the sampled hidden parameters of the imagined
        environment.
        """

    @abstractmethod
    def record_step(self, state: StateType, belief: Belief[HiddenParamsType]) -> None:
        """The current episode reached the state, and the belief is now as given."""

    @abstractmethod
    def end_episode(self, outcome: str) -> None:
        """The current episode ended, or planning for it failed, as the outcome
        describes."""

    @abstractmethod
    def record_queries(
        self,
        data: Sequence[QueryResponse[Any, Any]],
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        """Queries were asked and answered, and the belief was updated with the
        responses."""

    @abstractmethod
    def record_belief(
        self,
        label: str,
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        """A belief worth showing outside an episode, such as the belief a dream
        imagined."""


class NullRecorder(CycleRecorder[StateType, TaskType, HiddenParamsType]):
    """Records nothing."""

    def start_stage(self, cycle: int, stage: Stage) -> None:
        pass

    def end_stage(self) -> None:
        pass

    def start_episode(
        self,
        label: str,
        task: TaskType,
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        pass

    def record_step(self, state: StateType, belief: Belief[HiddenParamsType]) -> None:
        pass

    def end_episode(self, outcome: str) -> None:
        pass

    def record_queries(
        self,
        data: Sequence[QueryResponse[Any, Any]],
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        pass

    def record_belief(
        self,
        label: str,
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        pass
