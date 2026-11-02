"""Tests for recording/video.py."""

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, ArmTask, JointQuery
from able.recording.base import Stage
from able.recording.video import VideoRecorder, get_video_filename
from able.rendering.base import Renderer, Snapshot
from able.structs import QueryResponse


class _CaptionRenderer(Renderer[np.ndarray, ArmTask, np.ndarray]):
    """Draws small solid frames and records the snapshots it was given."""

    def __init__(self) -> None:
        self.snapshots: list[Snapshot] = []

    def render(self, snapshot: Snapshot) -> NDArray[np.uint8]:
        """Return a gray frame."""
        self.snapshots.append(snapshot)
        return np.full((32, 48, 3), 128, dtype=np.uint8)


def _count_frames(path: Path) -> int:
    # The reader's type does not declare count_frames().
    reader: Any = imageio.get_reader(path)
    return reader.count_frames()


def test_get_video_filename():
    """Tests that sorting video names puts the stages of each cycle in order."""
    names = [get_video_filename(c, s) for c in (10, 2) for s in reversed(Stage)]
    assert sorted(names)[:4] == [
        "cycle_002_1_sleep.mp4",
        "cycle_002_2_queries.mp4",
        "cycle_002_3_execution.mp4",
        "cycle_002_4_evaluation.mp4",
    ]


def test_video_recorder(tmp_path: Path):
    """Tests the videos and frames that a recorder writes."""
    renderer = _CaptionRenderer()
    true_params = np.array([-1.0, 1.0, -1.0, 1.0])
    recorder = VideoRecorder(
        renderer, true_params, tmp_path, fps=10, cycles=[1], hold_seconds=0.3
    )
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    task = ArmTask(initial_state=np.zeros(2), goal=np.ones(2))

    # Cycle 0 is not recorded.
    recorder.start_stage(0, Stage.EXECUTION)
    recorder.start_episode("Executed task 1", task, belief)
    recorder.record_step(np.ones(2), belief)
    recorder.end_episode("success")
    recorder.end_stage()
    assert not list(tmp_path.iterdir())

    # One episode of two steps is three frames plus three held frames.
    recorder.start_stage(1, Stage.EXECUTION)
    recorder.start_episode("Executed task 1", task, belief)
    for t in range(2):
        recorder.record_step(np.full(2, t + 1.0), belief)
    recorder.end_episode("success, planning time 1.0 ms")
    recorder.end_stage()
    assert _count_frames(tmp_path / "cycle_001_3_execution.mp4") == 6
    assert len(renderer.snapshots) == 4
    last = renderer.snapshots[-1]
    assert last.caption == (
        "Cycle 1 · Task execution\nExecuted task 1 · success, planning time 1.0 ms"
    )
    assert len(last.states) == 3
    assert last.task is task
    assert last.true_params is true_params

    # A dream's belief is held for twice as long, and the hypothesis is shown.
    hypothesis = np.zeros(4)
    recorder.start_stage(1, Stage.SLEEP)
    recorder.record_belief("Dream 1 · imagined belief", belief, hypothesis)
    recorder.end_stage()
    assert _count_frames(tmp_path / "cycle_001_1_sleep.mp4") == 6
    assert renderer.snapshots[-1].hypothesis is hypothesis
    assert renderer.snapshots[-1].task is None

    # Each query gets a held frame, and a stage with nothing to show has no video.
    recorder.start_stage(1, Stage.QUERIES)
    recorder.record_queries(
        [QueryResponse(JointQuery(joint=0, angle=1.0), True)] * 2, belief
    )
    recorder.end_stage()
    assert _count_frames(tmp_path / "cycle_001_2_queries.mp4") == 6
    recorder.start_stage(1, Stage.EVALUATION)
    recorder.end_stage()
    assert not (tmp_path / "cycle_001_4_evaluation.mp4").exists()
