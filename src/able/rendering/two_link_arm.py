"""Video frames for the two-link arm."""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from numpy.typing import NDArray

from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, ArmTask, TwoLinkArmModel
from able.rendering.base import Renderer, Snapshot

ArmArray = NDArray[np.float64]

_BELIEF_COLOR = "#2a78d6"
_HYPOTHESIS_COLOR = "#eb6834"
_GOAL_COLOR = "#1baf7a"
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_PRIOR_COLOR = "#e4e3df"
_SURFACE = "#fcfcfb"

_LIMIT_NAMES = ["Joint 1 lower", "Joint 1 upper", "Joint 2 lower", "Joint 2 upper"]
# The workspace masks are square images with this many pixels per side.
_MASK_RESOLUTION = 221
# Workspace masks are computed from a grid with this many angles per joint.
_ANGLES_PER_JOINT = 400
_MAX_CACHED_MASKS = 256
# Known limits have zero-width intervals, which are drawn this wide (radians).
_MIN_DRAWN_WIDTH = 0.03


class TwoLinkArmRenderer(Renderer[ArmArray, ArmTask, ArmArray]):
    """Draws the arm in its workspace next to the belief over the joint limits.

    The workspace panel shades the positions reachable under some hypothesis in
    the belief and the positions reachable under every hypothesis, and outlines
    the positions reachable under the true limits (and under the sampled limits
    in dreams). The belief panel shows each limit's interval within the prior,
    with ticks at the true and sampled values. Images are width x height pixels,
    and both should be multiples of 16 for video encoding.
    """

    def __init__(
        self,
        model: TwoLinkArmModel,
        width: int = 1280,
        height: int = 560,
        dpi: int = 100,
    ) -> None:
        self._model = model
        self._shape = (height, width, 3)
        self._fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self._fig.set_facecolor(_SURFACE)
        self._canvas = FigureCanvasAgg(self._fig)
        self._half_width = 1.1 * sum(model.link_lengths)
        self._mask_cache: dict[tuple[float, ...], NDArray[np.bool_]] = {}

    def render(
        self, snapshot: Snapshot[ArmArray, ArmTask, ArmArray]
    ) -> NDArray[np.uint8]:
        belief = snapshot.belief
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        self._fig.clear()
        self._fig.text(0.02, 0.97, snapshot.caption, va="top", fontsize=13, color=_INK)
        self._draw_workspace(
            self._fig.add_axes((0.03, 0.17, 0.40, 0.68)), snapshot, belief
        )
        self._draw_belief(
            self._fig.add_axes((0.57, 0.27, 0.40, 0.55)), snapshot, belief
        )
        self._canvas.draw()  # type: ignore[no-untyped-call]
        rgba = self._canvas.buffer_rgba()  # type: ignore[no-untyped-call]
        image = np.asarray(rgba)[..., :3].copy()
        assert image.shape == self._shape
        return image

    def _draw_workspace(
        self,
        ax: Axes,
        snapshot: Snapshot[ArmArray, ArmTask, ArmArray],
        belief: IntervalBelief,
    ) -> None:
        intervals = belief.intervals
        # The limits that allow the most motion under some hypothesis, and the
        # limits that every hypothesis allows.
        outer = intervals[[0, 1, 2, 3], [0, 1, 0, 1]]
        inner = intervals[[0, 1, 2, 3], [1, 0, 1, 0]]
        image = np.zeros((_MASK_RESOLUTION, _MASK_RESOLUTION, 4))
        blue = np.array([0x2A, 0x78, 0xD6]) / 255
        image[self._get_workspace_mask(outer)] = [*blue, 0.18]
        image[self._get_workspace_mask(inner)] = [*blue, 0.45]
        extent = (-self._half_width, self._half_width) * 2
        ax.imshow(image, extent=extent, origin="lower", interpolation="nearest")
        pixels = np.linspace(-self._half_width, self._half_width, _MASK_RESOLUTION)
        outlines = [(snapshot.true_params, _INK, 1.0)]
        if snapshot.hypothesis is not None:
            outlines.append((snapshot.hypothesis, _HYPOTHESIS_COLOR, 1.5))
        for limits, color, linewidth in outlines:
            mask = self._get_workspace_mask(limits).astype(float)
            ax.contour(
                pixels, pixels, mask, levels=[0.5], colors=[color], linewidths=linewidth
            )

        if snapshot.task is not None:
            goal = snapshot.task.goal
            ax.add_patch(
                Circle(
                    (goal[0], goal[1]),
                    self._model.goal_tolerance,
                    fill=False,
                    color=_GOAL_COLOR,
                    linewidth=1.5,
                )
            )
            ax.plot(goal[0], goal[1], "*", color=_GOAL_COLOR, markersize=14)
        if snapshot.states:
            trail = self._model.forward_kinematics(np.array(snapshot.states))
            ax.plot(trail[:, 0], trail[:, 1], color=_SECONDARY_INK, linewidth=1)
            q1, _ = snapshot.states[-1]
            l1, _ = self._model.link_lengths
            joints = [(0.0, 0.0), (l1 * np.cos(q1), l1 * np.sin(q1)), tuple(trail[-1])]
            ax.plot(
                [x for x, _ in joints],
                [y for _, y in joints],
                color=_INK,
                linewidth=3,
                solid_capstyle="round",
                marker="o",
                markersize=6,
                markerfacecolor=_SURFACE,
            )

        ax.set_xlim(-self._half_width, self._half_width)
        ax.set_ylim(-self._half_width, self._half_width)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_PRIOR_COLOR)
        ax.set_title("Workspace", fontsize=11, loc="left", color=_INK)
        handles: list[Any] = [
            Patch(
                color=_BELIEF_COLOR,
                alpha=0.45,
                label="Reachable under every hypothesis",
            ),
            Patch(
                color=_BELIEF_COLOR, alpha=0.18, label="Reachable under some hypothesis"
            ),
            Line2D(
                [], [], color=_INK, linewidth=1, label="Reachable under the true limits"
            ),
            Line2D(
                [], [], color=_GOAL_COLOR, marker="*", linestyle="none", label="Goal"
            ),
        ]
        if snapshot.hypothesis is not None:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=_HYPOTHESIS_COLOR,
                    linewidth=1.5,
                    label="Reachable under the sampled limits",
                )
            )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

    def _draw_belief(
        self,
        ax: Axes,
        snapshot: Snapshot[ArmArray, ArmTask, ArmArray],
        belief: IntervalBelief,
    ) -> None:
        rows = np.arange(len(_LIMIT_NAMES))[::-1]
        for row, (prior_low, prior_high), (low, high), true_value in zip(
            rows, JOINT_LIMIT_PRIOR, belief.intervals, snapshot.true_params
        ):
            ax.barh(
                row,
                prior_high - prior_low,
                left=prior_low,
                height=0.5,
                color=_PRIOR_COLOR,
            )
            width = max(high - low, _MIN_DRAWN_WIDTH)
            center = (low + high) / 2
            ax.barh(
                row, width, left=center - width / 2, height=0.5, color=_BELIEF_COLOR
            )
            ax.plot([true_value] * 2, [row - 0.35, row + 0.35], color=_INK, linewidth=2)
        if snapshot.hypothesis is not None:
            for row, value in zip(rows, snapshot.hypothesis):
                ax.plot(
                    [value] * 2,
                    [row - 0.35, row + 0.35],
                    color=_HYPOTHESIS_COLOR,
                    linewidth=2,
                )
        ax.set_yticks(rows, labels=_LIMIT_NAMES)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks(
            [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi],
            labels=["-π", "-π/2", "0", "π/2", "π"],
        )
        ax.set_xlabel("Angle (rad)", color=_SECONDARY_INK)
        ax.grid(axis="x", color=_PRIOR_COLOR, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            "Belief over the joint limits", fontsize=11, loc="left", color=_INK
        )
        handles: list[Any] = [
            Patch(color=_PRIOR_COLOR, label="Prior"),
            Patch(color=_BELIEF_COLOR, label="Belief"),
            Line2D([], [], color=_INK, linewidth=2, label="True limit"),
        ]
        if snapshot.hypothesis is not None:
            handles.append(
                Line2D(
                    [], [], color=_HYPOTHESIS_COLOR, linewidth=2, label="Sampled limit"
                )
            )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(handles),
            fontsize=9,
            frameon=False,
        )

    def _get_workspace_mask(self, limits: ArmArray) -> NDArray[np.bool_]:
        """Pixels of the workspace image that the end effector can reach within the
        joint limits [lo1, hi1, lo2, hi2]."""
        key = tuple(float(v) for v in np.round(limits, 6))
        if key not in self._mask_cache:
            if len(self._mask_cache) >= _MAX_CACHED_MASKS:
                self._mask_cache.clear()
            q1 = np.linspace(limits[0], limits[1], _ANGLES_PER_JOINT)
            q2 = np.linspace(limits[2], limits[3], _ANGLES_PER_JOINT)
            joints = np.stack(np.meshgrid(q1, q2, indexing="ij"), axis=-1)
            positions = self._model.forward_kinematics(joints.reshape(-1, 2))
            scaled = (positions + self._half_width) / (2 * self._half_width)
            pixels = np.round(scaled * (_MASK_RESOLUTION - 1)).astype(int)
            pixels = np.clip(pixels, 0, _MASK_RESOLUTION - 1)
            mask = np.zeros((_MASK_RESOLUTION, _MASK_RESOLUTION), dtype=bool)
            mask[pixels[:, 1], pixels[:, 0]] = True
            self._mask_cache[key] = mask
        return self._mask_cache[key]
