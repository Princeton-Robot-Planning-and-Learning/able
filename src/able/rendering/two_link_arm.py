"""Video frames for the two-link arm."""

from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.legend import Legend
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
# The workspace image is square with this many pixels per side.
_MASK_RESOLUTION = 301
_MAX_CACHED_MASKS = 512
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

    The axes, labels, and legends are drawn once. Each frame redraws only the
    parts that change.
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
        self._half_width = 1.1 * sum(model.link_lengths)
        centers = np.linspace(-self._half_width, self._half_width, _MASK_RESOLUTION)
        x, y = np.meshgrid(centers, centers)
        self._pixel_radius_sq = x**2 + y**2
        self._pixel_angle = np.arctan2(y, x)
        self._mask_cache: dict[tuple[float, ...], NDArray[np.bool_]] = {}

        self._fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self._fig.set_facecolor(_SURFACE)
        self._canvas = FigureCanvasAgg(self._fig)
        self._caption = self._fig.text(
            0.02, 0.97, "", va="top", fontsize=13, color=_INK, animated=True
        )
        self._setup_workspace()
        self._setup_belief()
        # The static parts of the figure, with and without the dream legends.
        self._backgrounds: dict[bool, Any] = {}

    def render(
        self, snapshot: Snapshot[ArmArray, ArmTask, ArmArray]
    ) -> NDArray[np.uint8]:
        belief = snapshot.belief
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        dream = snapshot.hypothesis is not None
        background = self._get_background(dream)
        self._canvas.restore_region(background)  # type: ignore[no-untyped-call]
        self._caption.set_text(snapshot.caption)
        self._fig.draw_artist(self._caption)
        self._draw_workspace(snapshot, belief)
        self._draw_belief(snapshot, belief)
        rgba = self._canvas.buffer_rgba()  # type: ignore[no-untyped-call]
        image = np.asarray(rgba)[..., :3].copy()
        assert image.shape == self._shape
        return image

    def _setup_workspace(self) -> None:
        ax = self._fig.add_axes((0.03, 0.17, 0.40, 0.68))
        self._workspace_ax = ax
        extent = (-self._half_width, self._half_width) * 2
        self._workspace_image = ax.imshow(
            np.zeros((_MASK_RESOLUTION, _MASK_RESOLUTION, 4)),
            extent=extent,
            origin="lower",
            animated=True,
        )
        self._goal_circle = Circle(
            (0.0, 0.0),
            self._model.goal_tolerance,
            fill=False,
            color=_GOAL_COLOR,
            linewidth=1.5,
            animated=True,
        )
        ax.add_patch(self._goal_circle)
        (self._goal_marker,) = ax.plot(
            [], [], "*", color=_GOAL_COLOR, markersize=14, animated=True
        )
        (self._trail,) = ax.plot(
            [], [], color=_SECONDARY_INK, linewidth=1, animated=True
        )
        (self._arm,) = ax.plot(
            [],
            [],
            color=_INK,
            linewidth=3,
            solid_capstyle="round",
            marker="o",
            markersize=6,
            markerfacecolor=_SURFACE,
            animated=True,
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
            Patch(color=_BELIEF_COLOR, alpha=0.45),
            Patch(color=_BELIEF_COLOR, alpha=0.18),
            Line2D([], [], color=_INK, linewidth=1),
            Line2D([], [], color=_GOAL_COLOR, marker="*", linestyle="none"),
            Line2D([], [], color=_HYPOTHESIS_COLOR, linewidth=1.5),
        ]
        labels = [
            "Reachable under every hypothesis",
            "Reachable under some hypothesis",
            "Reachable under the true limits",
            "Goal",
            "Reachable under the sampled limits",
        ]
        self._workspace_legends = {
            dream: self._add_legend(
                ax, handles[: 5 if dream else 4], labels, (0.5, -0.02), 2, 8
            )
            for dream in (False, True)
        }

    def _setup_belief(self) -> None:
        ax = self._fig.add_axes((0.57, 0.27, 0.40, 0.55))
        self._belief_ax = ax
        self._rows = np.arange(len(_LIMIT_NAMES))[::-1]
        prior_widths = JOINT_LIMIT_PRIOR[:, 1] - JOINT_LIMIT_PRIOR[:, 0]
        ax.barh(
            self._rows,
            prior_widths,
            left=JOINT_LIMIT_PRIOR[:, 0],
            height=0.5,
            color=_PRIOR_COLOR,
        )
        self._belief_bars = list(
            ax.barh(
                self._rows,
                prior_widths,
                left=JOINT_LIMIT_PRIOR[:, 0],
                height=0.5,
                color=_BELIEF_COLOR,
                animated=True,
            )
        )
        self._true_ticks = [
            ax.plot([], [], color=_INK, linewidth=2, animated=True)[0]
            for _ in self._rows
        ]
        self._hypothesis_ticks = [
            ax.plot([], [], color=_HYPOTHESIS_COLOR, linewidth=2, animated=True)[0]
            for _ in self._rows
        ]
        ax.set_yticks(self._rows, labels=_LIMIT_NAMES)
        ax.set_ylim(-0.6, len(_LIMIT_NAMES) - 0.4)
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
            Patch(color=_PRIOR_COLOR),
            Patch(color=_BELIEF_COLOR),
            Line2D([], [], color=_INK, linewidth=2),
            Line2D([], [], color=_HYPOTHESIS_COLOR, linewidth=2),
        ]
        labels = ["Prior", "Belief", "True limit", "Sampled limit"]
        self._belief_legends = {
            dream: self._add_legend(
                ax, handles[: 4 if dream else 3], labels, (0.5, -0.2), 4, 9
            )
            for dream in (False, True)
        }

    @staticmethod
    def _add_legend(
        ax: Any,
        handles: list[Any],
        labels: list[str],
        anchor: tuple[float, float],
        ncol: int,
        fontsize: int,
    ) -> Legend:
        legend = Legend(
            ax,
            handles,
            labels[: len(handles)],
            loc="upper center",
            bbox_to_anchor=anchor,
            ncol=ncol,
            fontsize=fontsize,
            frameon=False,
        )
        ax.add_artist(legend)
        return legend

    def _get_background(self, dream: bool) -> Any:
        if dream not in self._backgrounds:
            for legends in (self._workspace_legends, self._belief_legends):
                legends[dream].set_visible(True)
                legends[not dream].set_visible(False)
            self._canvas.draw()  # type: ignore[no-untyped-call]
            background = self._canvas.copy_from_bbox(  # type: ignore[no-untyped-call]
                self._fig.bbox
            )
            self._backgrounds[dream] = background
        return self._backgrounds[dream]

    def _draw_workspace(
        self, snapshot: Snapshot[ArmArray, ArmTask, ArmArray], belief: IntervalBelief
    ) -> None:
        intervals = belief.intervals
        # The limits that allow the most motion under some hypothesis, and the
        # limits that every hypothesis allows.
        outer = intervals[[0, 1, 2, 3], [0, 1, 0, 1]]
        inner = intervals[[0, 1, 2, 3], [1, 0, 1, 0]]
        image = np.zeros((_MASK_RESOLUTION, _MASK_RESOLUTION, 4))
        image[self._get_workspace_mask(outer)] = (*to_rgb(_BELIEF_COLOR), 0.18)
        image[self._get_workspace_mask(inner)] = (*to_rgb(_BELIEF_COLOR), 0.45)
        true_mask = self._get_workspace_mask(snapshot.true_params)
        image[_get_outline(true_mask)] = (*to_rgb(_INK), 1.0)
        if snapshot.hypothesis is not None:
            hypothesis_mask = self._get_workspace_mask(snapshot.hypothesis)
            image[_get_outline(hypothesis_mask)] = (*to_rgb(_HYPOTHESIS_COLOR), 1.0)
        self._workspace_image.set_data(image)

        task = snapshot.task
        self._goal_circle.set_visible(task is not None)
        if task is not None:
            self._goal_circle.set_center((task.goal[0], task.goal[1]))
            self._goal_marker.set_data([task.goal[0]], [task.goal[1]])
        else:
            self._goal_marker.set_data([], [])
        if snapshot.states:
            trail = self._model.forward_kinematics(np.array(snapshot.states))
            q1 = snapshot.states[-1][0]
            l1 = self._model.link_lengths[0]
            elbow = (l1 * np.cos(q1), l1 * np.sin(q1))
            self._trail.set_data(trail[:, 0], trail[:, 1])
            self._arm.set_data(
                [0.0, elbow[0], trail[-1, 0]], [0.0, elbow[1], trail[-1, 1]]
            )
        else:
            self._trail.set_data([], [])
            self._arm.set_data([], [])
        for artist in (
            self._workspace_image,
            self._goal_circle,
            self._trail,
            self._arm,
            self._goal_marker,
        ):
            self._workspace_ax.draw_artist(artist)

    def _draw_belief(
        self, snapshot: Snapshot[ArmArray, ArmTask, ArmArray], belief: IntervalBelief
    ) -> None:
        for interval_bar, (low, high) in zip(self._belief_bars, belief.intervals):
            width = max(high - low, _MIN_DRAWN_WIDTH)
            interval_bar.set_x((low + high - width) / 2)
            interval_bar.set_width(width)
            self._belief_ax.draw_artist(interval_bar)
        hypothesis = snapshot.hypothesis
        for i, row in enumerate(self._rows):
            ticks = [(self._true_ticks[i], snapshot.true_params[i])]
            if hypothesis is not None:
                ticks.append((self._hypothesis_ticks[i], hypothesis[i]))
            for tick, value in ticks:
                tick.set_data([value, value], [row - 0.35, row + 0.35])
                self._belief_ax.draw_artist(tick)

    def _get_workspace_mask(self, limits: ArmArray) -> NDArray[np.bool_]:
        """Pixels of the workspace image that the end effector can reach within the
        joint limits [lo1, hi1, lo2, hi2].

        A pixel is reachable if one of the two inverse kinematics solutions for
        its center lies within the limits, so the mask has no sampling gaps.
        """
        key = tuple(float(v) for v in limits)
        if key not in self._mask_cache:
            if len(self._mask_cache) >= _MAX_CACHED_MASKS:
                self._mask_cache.clear()
            lo1, hi1, lo2, hi2 = limits
            l1, l2 = self._model.link_lengths
            cos_elbow = (self._pixel_radius_sq - l1**2 - l2**2) / (2 * l1 * l2)
            in_reach = np.abs(cos_elbow) <= 1
            elbow = np.arccos(np.clip(cos_elbow, -1.0, 1.0))
            mask = np.zeros_like(in_reach)
            for q2 in (elbow, -elbow):
                q1 = self._pixel_angle - np.arctan2(
                    l2 * np.sin(q2), l1 + l2 * np.cos(q2)
                )
                q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
                mask |= in_reach & (lo1 <= q1) & (q1 <= hi1) & (lo2 <= q2) & (q2 <= hi2)
            self._mask_cache[key] = mask
        return self._mask_cache[key]


def _get_outline(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The pixels of the mask that border a pixel outside it."""
    padded = np.pad(mask, 1)
    interior = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior
