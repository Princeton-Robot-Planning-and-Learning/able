"""Video frames for the friction environment."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from numpy.typing import NDArray

from able.beliefs.particles import ParticleBelief
from able.envs.friction.model import FrictionModel, FrictionState, FrictionTask
from able.envs.friction.world import (
    GOAL_RGB,
    HOLE_RGB,
    PUCK_RGB,
    SURFACE_NAMES,
    SURFACE_RGB,
    WALL_RGB,
)
from able.rendering.base import Renderer, Snapshot

_BELIEF_COLOR = "#2a78d6"
_HYPOTHESIS_COLOR = "#eb6834"
_TRUTH_COLOR = "#0b0b0b"
_ROBOT_COLOR = "#6b6b6b"
_INK = "#0b0b0b"
_SURFACE = "#fcfcfb"


class FrictionRenderer(Renderer[FrictionState, FrictionTask, NDArray[np.float64]]):
    """Draws the maze from above next to the belief over the floors' friction.

    The left panel shows the floors, walls, holes, and goal, the puck's stops so
    far, and the robot's base, and says how the last push went. The right panel
    shows each floor's particles, their mean, and ticks at the true friction (and
    the sampled friction in dreams). Images are width x height pixels; both should
    be multiples of 16 for video encoding.
    """

    def __init__(
        self, model: FrictionModel, width: int = 1280, height: int = 560, dpi: int = 100
    ) -> None:
        self._world = model.world
        self._shape = (height, width, 3)
        self._fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self._fig.set_facecolor(_SURFACE)
        self._canvas = FigureCanvasAgg(self._fig)
        self._view = self._fig.add_axes((0.02, 0.04, 0.5, 0.8))
        self._belief = self._fig.add_axes((0.62, 0.15, 0.35, 0.65))

    def render(
        self, snapshot: Snapshot[FrictionState, FrictionTask, NDArray[np.float64]]
    ) -> NDArray[np.uint8]:
        self._fig.texts.clear()
        self._fig.text(0.02, 0.97, snapshot.caption, va="top", fontsize=13, color=_INK)
        self._draw_view(snapshot.states)
        belief = snapshot.belief
        assert isinstance(belief, ParticleBelief), "Need a ParticleBelief"
        self._draw_belief(belief, snapshot.true_params, snapshot.hypothesis)
        self._canvas.draw()  # type: ignore[no-untyped-call]
        rgba = self._canvas.buffer_rgba()  # type: ignore[no-untyped-call]
        image = np.asarray(rgba)[..., :3]
        assert image.shape == self._shape, "Width and height must match the figure"
        return image.copy()

    def _draw_view(self, states: tuple[FrictionState, ...]) -> None:
        view, world = self._view, self._world
        puck_map, phys = world.puck_map, world.physics
        view.clear()
        view.set_axis_off()
        view.set_aspect("equal")
        view.set_xlim(0, puck_map.width)
        view.set_ylim(0, puck_map.height)
        for surface, (x0, y0, x1, y1) in puck_map.surface_rects():
            view.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, color=SURFACE_RGB[surface])
            )
        for x0, y0, x1, y1 in puck_map.wall_rects():
            view.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=WALL_RGB))
        for hole in world.holes:
            view.add_patch(Circle(tuple(hole), phys.hole_radius, color=HOLE_RGB))
        view.add_patch(
            Circle(tuple(world.goal), phys.goal_radius, color=GOAL_RGB, alpha=0.6)
        )
        if not states:
            view.set_title("No episode", fontsize=11, color=_INK)
            return
        trail = np.array([s.true_puck for s in states])
        view.plot(trail[:, 0], trail[:, 1], "o-", color=_INK, ms=3, lw=1)
        state = states[-1]
        view.add_patch(Circle(state.true_puck, phys.puck_radius, color=PUCK_RGB))
        robot = state.robot_pose[:2]
        view.add_patch(Circle(robot, phys.robot_radius, color=_ROBOT_COLOR, alpha=0.7))
        view.set_title(_describe(state), fontsize=11, color=_INK)

    def _draw_belief(
        self,
        belief: ParticleBelief,
        true_params: NDArray[np.float64],
        hypothesis: NDArray[np.float64] | None,
    ) -> None:
        axes = self._belief
        axes.clear()
        rows = np.arange(len(SURFACE_NAMES))
        spread = np.linspace(-0.25, 0.25, len(belief.particles))
        for row in rows:
            axes.scatter(
                belief.particles[:, row], row + spread, s=4, color=_BELIEF_COLOR
            )
        axes.scatter(belief.mean, rows, color="white", edgecolor=_INK, zorder=3, s=24)
        axes.vlines(true_params, rows - 0.4, rows + 0.4, color=_TRUTH_COLOR, lw=2)
        if hypothesis is not None:
            axes.vlines(
                hypothesis, rows - 0.4, rows + 0.4, color=_HYPOTHESIS_COLOR, lw=2
            )
        axes.set_yticks(rows, list(SURFACE_NAMES))
        margin = 0.1 * float(np.max(belief.high - belief.low))
        axes.set_xlim(
            float(belief.low.min()) - margin, float(belief.high.max()) + margin
        )
        axes.invert_yaxis()
        axes.set_xlabel("Friction coefficient")
        axes.set_title(
            "Particles (blue), mean (white), truth (black)"
            + (", dream (orange)" if hypothesis is not None else ""),
            fontsize=11,
            color=_INK,
        )


def _describe(state: FrictionState) -> str:
    lines = [f"Push {state.pushes}"]
    record = state.last_push
    if record is not None:
        if not record.feasible:
            lines.append("infeasible push")
        else:
            lines.append(f"release speed {record.release_speed:.2f} m/s")
    if state.outcome == "goal":
        lines.append("reached the goal")
    elif state.outcome == "hole":
        lines.append("fell in a hole")
    return ", ".join(lines)
