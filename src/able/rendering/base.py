"""Base class for renderers, which draw video frames."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.structs import HiddenParamsType, StateType, TaskType


@dataclass(frozen=True)
class Snapshot(Generic[StateType, TaskType, HiddenParamsType]):
    """What one video frame shows.

    The states are those of the current episode so far, and the last one is the
    current state. Frames outside an episode have no task and no states. Videos
    are for people, so frames show the true hidden parameters. In dreams, the
    hypothesis is the sampled value that the imagined environment uses instead.
    """

    caption: str
    belief: Belief[HiddenParamsType]
    true_params: HiddenParamsType
    task: TaskType | None = None
    states: tuple[StateType, ...] = ()
    hypothesis: HiddenParamsType | None = None


class Renderer(ABC, Generic[StateType, TaskType, HiddenParamsType]):
    """Draws snapshots as images of a fixed size."""

    @abstractmethod
    def render(
        self, snapshot: Snapshot[StateType, TaskType, HiddenParamsType]
    ) -> NDArray[np.uint8]:
        """Draw the snapshot as an RGB image with shape (height, width, 3)."""
