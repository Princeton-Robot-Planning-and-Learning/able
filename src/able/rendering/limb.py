"""Video frames for the limb environment."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.limb.model import LimbModel, LimbState, LimbTask
from able.envs.limb.sim import LimbSim
from able.rendering.base import Renderer, Snapshot

_BELIEF_COLOR = "#2a78d6"
_HYPOTHESIS_COLOR = "#eb6834"
_TRUTH_COLOR = "#0b0b0b"
_FLOOR_COLOR = "#1baf7a"
_PRIOR_COLOR = "#e4e3df"
_INK = "#0b0b0b"
_SURFACE = "#fcfcfb"


class LimbRenderer(Renderer[LimbState, LimbTask, NDArray[np.float64]]):
    """Draws the robot and the person next to the belief over their range.

    The left panel is the simulator's camera view of the current state, with the
    task's goal configuration as a translucent green limb, and says whether the
    task has ended. The right
    panel shows, for each motion, the belief's mean plus or minus one standard
    deviation within the prior's range, its floor, and ticks at the true scale
    (and the sampled scale in dreams). The renderer restores the state in a
    simulator of its own. Images are width x height pixels; both should be
    multiples of 16 for video encoding.
    """

    def __init__(
        self, model: LimbModel, width: int = 1280, height: int = 560, dpi: int = 100
    ) -> None:
        self._model = model
        self._sim: LimbSim | None = None
        self._shape = (height, width, 3)
        self._fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self._fig.set_facecolor(_SURFACE)
        self._canvas = FigureCanvasAgg(self._fig)
        self._view = self._fig.add_axes((0.01, 0.02, 0.45, 0.82))
        self._belief = self._fig.add_axes((0.62, 0.15, 0.35, 0.65))

    def _get_sim(self) -> LimbSim:
        if self._sim is None:
            self._sim = LimbSim(self._model.variant, self._model.settings)
        return self._sim

    def close(self) -> None:
        """Disconnect the renderer's simulator."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def render(
        self, snapshot: Snapshot[LimbState, LimbTask, NDArray[np.float64]]
    ) -> NDArray[np.uint8]:
        self._fig.texts.clear()
        self._fig.text(0.02, 0.97, snapshot.caption, va="top", fontsize=13, color=_INK)
        self._draw_view(snapshot)
        belief = snapshot.belief
        assert isinstance(belief, RangeOfMotionBelief), "Need a RangeOfMotionBelief"
        self._draw_belief(belief, snapshot.true_params, snapshot.hypothesis)
        self._canvas.draw()  # type: ignore[no-untyped-call]
        rgba = self._canvas.buffer_rgba()  # type: ignore[no-untyped-call]
        image = np.asarray(rgba)[..., :3]
        assert image.shape == self._shape, "Width and height must match the figure"
        return image.copy()

    def _draw_view(
        self, snapshot: Snapshot[LimbState, LimbTask, NDArray[np.float64]]
    ) -> None:
        view = self._view
        view.clear()
        view.set_axis_off()
        if not snapshot.states:
            view.text(0.5, 0.5, "No episode", ha="center", va="center", color=_INK)
            return
        state = snapshot.states[-1]
        sim = self._get_sim()
        sim.restore(state.coupled)
        goal_limb = getattr(sim.env.scene, "goal_limb", None)
        if goal_limb is not None and snapshot.task is not None:
            goal_limb.set_joints(list(snapshot.task.goal))
        view.imshow(sim.env.render())
        lines = [f"Step {len(snapshot.states) - 1}"]
        if snapshot.task is not None:
            error = sim.get_limb_error(state.limb_positions, snapshot.task.goal)
            lines.append(f"distance to goal {error:.2f}")
        if state.reached:
            lines.append("reached the goal")
        if state.violated:
            lines.append("passed the person's limit")
        if state.failure:
            lines.append(f"failed: {state.failure}")
        view.set_title(", ".join(lines), fontsize=11, color=_INK)

    def _draw_belief(
        self,
        belief: RangeOfMotionBelief,
        true_params: NDArray[np.float64],
        hypothesis: NDArray[np.float64] | None,
    ) -> None:
        axes = self._belief
        axes.clear()
        rows = np.arange(len(belief.mean))
        mean, stddev = belief.truncated_mean, belief.truncated_stddev
        axes.barh(rows, belief.high - belief.low, left=belief.low, color=_PRIOR_COLOR)
        axes.barh(rows, 2 * stddev, left=mean - stddev, height=0.5, color=_BELIEF_COLOR)
        axes.scatter(mean, rows, color="white", zorder=3, s=12)
        axes.vlines(belief.lower, rows - 0.4, rows + 0.4, color=_FLOOR_COLOR, lw=2)
        axes.vlines(true_params, rows - 0.4, rows + 0.4, color=_TRUTH_COLOR, lw=2)
        if hypothesis is not None:
            axes.vlines(
                hypothesis, rows - 0.4, rows + 0.4, color=_HYPOTHESIS_COLOR, lw=2
            )
        names = [m.replace("_", " ") for m in self._model.variant.active_motions]
        axes.set_yticks(rows, names)
        axes.set_xlim(belief.low - 0.05, belief.high + 0.05)
        axes.invert_yaxis()
        axes.set_xlabel("Range-of-motion scale")
        axes.set_title(
            "Belief (blue), floor (green), truth (black)"
            + (", dream (orange)" if hypothesis is not None else ""),
            fontsize=11,
            color=_INK,
        )
