"""Tests for experiments/combine_videos.py."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "combine_videos.py"


def _write_video(path: Path, num_frames: int, value: int) -> None:
    writer: Any = imageio.get_writer(path, fps=10)
    with writer:
        for _ in range(num_frames):
            writer.append_data(np.full((32, 48, 3), value, dtype=np.uint8))


def test_combine_videos(tmp_path: Path) -> None:
    """Tests that stage videos are combined in order with title cards."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    OmegaConf.save(
        {"approach_name": "dreaming", "env_name": "two_link_arm", "seed": 0},
        tmp_path / "config.yaml",
    )
    # Written out of order; the combined video sorts them.
    _write_video(video_dir / "cycle_001_3_execution.mp4", 2, 50)
    _write_video(video_dir / "cycle_000_1_sleep.mp4", 3, 200)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_dir",
            str(tmp_path),
            "--title_seconds",
            "0.2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    reader: Any = imageio.get_reader(video_dir / "full_pipeline.mp4")
    frames = [frame.astype(float).mean() for frame in reader]
    # Four frames for the run's title card, then two title frames before each
    # stage video.
    assert len(frames) == 4 + (2 + 3) + (2 + 2)
    sleep_frames = frames[6:9]
    execution_frames = frames[11:13]
    assert all(abs(v - 200) < 10 for v in sleep_frames)
    assert all(abs(v - 50) < 10 for v in execution_frames)
