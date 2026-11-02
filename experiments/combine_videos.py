"""Combine a run's stage videos into one video of the whole pipeline.

The run must have been recorded with record_videos=true. Each stage video is
preceded by a title card naming its cycle and stage.

Example:
    python experiments/combine_videos.py --run_dir logs/2026-11-02/16-12-04
"""

import argparse
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from omegaconf import OmegaConf

from able.recording.base import Stage
from able.recording.video import STAGE_TITLES

_VIDEO_NAME = re.compile(r"cycle_(\d+)_\d_([a-z]+)\.mp4")


def _make_title_card(
    title: str, subtitle: str, height: int, width: int
) -> NDArray[np.uint8]:
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    fig.set_facecolor("#fcfcfb")
    canvas = FigureCanvasAgg(fig)
    fig.text(0.5, 0.56, title, ha="center", va="center", fontsize=30, color="#0b0b0b")
    fig.text(
        0.5, 0.42, subtitle, ha="center", va="center", fontsize=18, color="#52514e"
    )
    canvas.draw()  # type: ignore[no-untyped-call]
    rgba = canvas.buffer_rgba()  # type: ignore[no-untyped-call]
    return np.asarray(rgba)[..., :3].copy()


def _main(run_dir: Path, outfile: Path, title_seconds: float) -> None:
    videos = sorted((run_dir / "videos").glob("cycle_*.mp4"))
    assert videos, f"No stage videos in {run_dir / 'videos'}"
    # The readers' and writer's types do not declare the methods used here.
    first_reader: Any = imageio.get_reader(videos[0])
    metadata = first_reader.get_meta_data()
    first_reader.close()
    fps = metadata["fps"]
    width, height = metadata["size"]
    num_title_frames = max(1, round(title_seconds * fps))

    writer: Any = imageio.get_writer(outfile, fps=fps)
    with writer:
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            card = _make_title_card(
                f"Approach: {cfg.approach_name}",
                f"Environment: {cfg.env_name} · seed {cfg.seed}",
                height,
                width,
            )
            for _ in range(2 * num_title_frames):
                writer.append_data(card)
        for video in videos:
            match = _VIDEO_NAME.fullmatch(video.name)
            assert match is not None, f"Unexpected video name {video.name}"
            stage = Stage(match.group(2))
            card = _make_title_card(
                f"Cycle {int(match.group(1))}", STAGE_TITLES[stage], height, width
            )
            for _ in range(num_title_frames):
                writer.append_data(card)
            reader: Any = imageio.get_reader(video)
            with reader:
                for frame in reader:
                    writer.append_data(frame)
    print(f"Combined {len(videos)} stage videos into {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Defaults to videos/full_pipeline.mp4 in the run directory.",
    )
    parser.add_argument("--title_seconds", type=float, default=1.0)
    args = parser.parse_args()
    _main(
        args.run_dir,
        args.outfile or args.run_dir / "videos" / "full_pipeline.mp4",
        args.title_seconds,
    )
