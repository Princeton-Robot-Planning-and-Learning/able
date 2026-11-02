"""A recorder used in tests that logs the events it observes."""

from typing import Any, Sequence

from able.beliefs.base import Belief
from able.recording.base import CycleRecorder, Stage
from able.structs import QueryResponse


class EventRecorder(CycleRecorder[Any, Any, Any]):
    """Logs each event as a tuple of its name and its most useful arguments."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def start_stage(self, cycle: int, stage: Stage) -> None:
        self.events.append(("start_stage", cycle, stage))

    def end_stage(self) -> None:
        self.events.append(("end_stage",))

    def start_episode(
        self,
        label: str,
        task: Any,
        belief: Belief[Any],
        hypothesis: Any = None,
    ) -> None:
        self.events.append(("start_episode", label, hypothesis is not None))

    def record_step(self, state: Any, belief: Belief[Any]) -> None:
        self.events.append(("record_step", belief))

    def end_episode(self, outcome: str) -> None:
        self.events.append(("end_episode", outcome))

    def record_queries(
        self,
        data: Sequence[QueryResponse[Any, Any]],
        belief: Belief[Any],
        hypothesis: Any = None,
    ) -> None:
        self.events.append(("record_queries", len(data)))

    def record_belief(
        self, label: str, belief: Belief[Any], hypothesis: Any = None
    ) -> None:
        self.events.append(("record_belief", label, hypothesis is not None))

    def get_names(self) -> list[str]:
        """The name of each event, in order."""
        return [event[0] for event in self.events]
