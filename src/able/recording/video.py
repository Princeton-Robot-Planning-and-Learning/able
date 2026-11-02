"""Recording one video per stage per cycle."""

from pathlib import Path
from typing import Any, Collection, Sequence

import imageio.v2 as imageio

from able.beliefs.base import Belief
from able.recording.base import CycleRecorder, Stage
from able.rendering.base import Renderer, Snapshot
from able.structs import HiddenParamsType, QueryResponse, StateType, TaskType

STAGE_TITLES = {
    Stage.SLEEP: "Sleep",
    Stage.QUERIES: "Information gathering",
    Stage.EXECUTION: "Task execution",
    Stage.EVALUATION: "Evaluation",
}


def get_video_filename(cycle: int, stage: Stage) -> str:
    """The name of the video for a stage of a cycle.

    Sorting the names of a run's videos puts them in the order they happened.
    """
    index = list(Stage).index(stage) + 1
    return f"cycle_{cycle:03d}_{index}_{stage.value}.mp4"


class VideoRecorder(CycleRecorder[StateType, TaskType, HiddenParamsType]):
    """Writes one video for each stage of each recorded cycle.

    During a stage, the recorder only stores snapshots. It renders them and writes
    the video when the stage ends, so rendering never counts toward planning or
    sleep time. Stages with nothing to show, such as sleep for a robot that does
    not dream, produce no video. The last frame of each episode, each query, and
    each recorded belief is held for hold_seconds. If cycles is given, only those
    cycles are recorded.
    """

    def __init__(
        self,
        renderer: Renderer[StateType, TaskType, HiddenParamsType],
        true_params: HiddenParamsType,
        video_dir: Path,
        fps: int = 10,
        cycles: Collection[int] | None = None,
        hold_seconds: float = 0.5,
    ) -> None:
        self._renderer = renderer
        self._true_params = true_params
        self._video_dir = video_dir
        self._fps = fps
        self._cycles = cycles
        self._hold_frames = max(1, round(hold_seconds * fps))
        self._cycle = 0
        self._stage = Stage.SLEEP
        self._recording = False
        # Each entry is a snapshot and the number of frames it lasts.
        self._frames: list[tuple[Snapshot[StateType, TaskType, HiddenParamsType], int]]
        self._frames = []
        self._label = ""
        self._task: TaskType | None = None
        self._states: list[StateType] = []
        self._belief: Belief[HiddenParamsType] | None = None
        self._hypothesis: HiddenParamsType | None = None

    def start_stage(self, cycle: int, stage: Stage) -> None:
        self._cycle = cycle
        self._stage = stage
        self._recording = self._cycles is None or cycle in self._cycles
        self._frames = []

    def end_stage(self) -> None:
        if self._recording and self._frames:
            self._video_dir.mkdir(parents=True, exist_ok=True)
            path = self._video_dir / get_video_filename(self._cycle, self._stage)
            # The writer's type does not declare append_data().
            writer: Any = imageio.get_writer(path, fps=self._fps)
            with writer:
                for snapshot, num_frames in self._frames:
                    image = self._renderer.render(snapshot)
                    for _ in range(num_frames):
                        writer.append_data(image)
        self._frames = []

    def start_episode(
        self,
        label: str,
        task: TaskType,
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        self._label = label
        self._task = task
        self._states = [task.initial_state]
        self._belief = belief
        self._hypothesis = hypothesis
        self._add_episode_frame("step 0", 1)

    def record_step(self, state: StateType, belief: Belief[HiddenParamsType]) -> None:
        self._states.append(state)
        self._belief = belief
        self._add_episode_frame(f"step {len(self._states) - 1}", 1)

    def end_episode(self, outcome: str) -> None:
        self._add_episode_frame(outcome, self._hold_frames)

    def record_queries(
        self,
        data: Sequence[QueryResponse[Any, Any]],
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        for i, pair in enumerate(data):
            label = f"Query {i + 1} of {len(data)}: {pair.query} → {pair.response}"
            self._add_frame(label, belief, hypothesis, self._hold_frames)

    def record_belief(
        self,
        label: str,
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None = None,
    ) -> None:
        self._add_frame(label, belief, hypothesis, 2 * self._hold_frames)

    def _add_episode_frame(self, detail: str, num_frames: int) -> None:
        assert self._belief is not None, "Call start_episode() first"
        if not self._recording:
            return
        snapshot = Snapshot(
            caption=self._get_caption(f"{self._label} · {detail}"),
            belief=self._belief,
            true_params=self._true_params,
            task=self._task,
            states=tuple(self._states),
            hypothesis=self._hypothesis,
        )
        self._frames.append((snapshot, num_frames))

    def _add_frame(
        self,
        label: str,
        belief: Belief[HiddenParamsType],
        hypothesis: HiddenParamsType | None,
        num_frames: int,
    ) -> None:
        if not self._recording:
            return
        snapshot: Snapshot[StateType, TaskType, HiddenParamsType] = Snapshot(
            caption=self._get_caption(label),
            belief=belief,
            true_params=self._true_params,
            hypothesis=hypothesis,
        )
        self._frames.append((snapshot, num_frames))

    def _get_caption(self, detail: str) -> str:
        return f"Cycle {self._cycle} · {STAGE_TITLES[self._stage]}\n{detail}"
