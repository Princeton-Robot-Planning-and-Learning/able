"""Video frames for the human reach environment."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from able.beliefs.particles import ParticleBelief
from able.envs.human_reach.model import ReachModel, ReachState, ReachTask
from able.envs.human_reach.scene import ObjectCentricBedsideStocking3DEnv
from able.rendering.base import Renderer, Snapshot

_AXES = ("forward (x)", "sideways (y)", "up (z)")
_BELIEF_COLOR = "#2a78d6"
_HYPOTHESIS_COLOR = "#eb6834"
_TRUTH_COLOR = "#0b0b0b"
_INK = "#0b0b0b"
_SURFACE = "#fcfcfb"


class HumanReachRenderer(Renderer[ReachState, ReachTask, NDArray[np.float64]]):
    """Draws the scene next to the belief over the person's reach.

    The left panel is the simulator's camera view of the current state, and says
    which requested items the person reached after the plan. The right panel shows
    the particles of each reach semi-axis, their mean, and ticks at the true reach
    (and the sampled reach in dreams). The renderer restores the state in a
    simulator of its own. Images are width x height pixels; both should be
    multiples of 16 for video encoding.
    """

    def __init__(
        self, model: ReachModel, width: int = 1280, height: int = 560, dpi: int = 100
    ) -> None:
        self._model = model
        self._sim: ObjectCentricBedsideStocking3DEnv | None = None
        self._shape = (height, width, 3)
        self._fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self._fig.set_facecolor(_SURFACE)
        self._canvas = FigureCanvasAgg(self._fig)
        self._view = self._fig.add_axes((0.01, 0.02, 0.5, 0.82))
        self._belief = self._fig.add_axes((0.62, 0.15, 0.35, 0.65))

    def _get_sim(self) -> ObjectCentricBedsideStocking3DEnv:
        if self._sim is None:
            model = self._model
            self._sim = ObjectCentricBedsideStocking3DEnv(
                model.num_requested,
                model.num_clutter,
                config=model.config,
                allow_state_access=True,
            )
        return self._sim

    def close(self) -> None:
        """Close the renderer's simulator."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def render(
        self, snapshot: Snapshot[ReachState, ReachTask, NDArray[np.float64]]
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

    def _draw_view(self, states: tuple[ReachState, ...]) -> None:
        view = self._view
        view.clear()
        view.set_axis_off()
        if not states:
            view.text(0.5, 0.5, "No episode", ha="center", va="center", color=_INK)
            return
        state = states[-1]
        sim = self._get_sim()
        sim.reset(options={"init_state": state.scene})
        view.imshow(sim.render())
        view.set_title(_describe(state), fontsize=11, color=_INK)

    def _draw_belief(
        self,
        belief: ParticleBelief,
        true_params: NDArray[np.float64],
        hypothesis: NDArray[np.float64] | None,
    ) -> None:
        axes = self._belief
        axes.clear()
        rows = np.arange(len(_AXES))
        spread = np.linspace(-0.25, 0.25, len(belief.particles))
        for row in rows:
            axes.scatter(
                belief.particles[:, row], row + spread, s=3, color=_BELIEF_COLOR
            )
        axes.scatter(belief.mean, rows, color="white", edgecolor=_INK, zorder=3, s=24)
        axes.vlines(true_params, rows - 0.4, rows + 0.4, color=_TRUTH_COLOR, lw=2)
        if hypothesis is not None:
            axes.vlines(
                hypothesis, rows - 0.4, rows + 0.4, color=_HYPOTHESIS_COLOR, lw=2
            )
        axes.set_yticks(rows, list(_AXES))
        axes.set_xlim(float(belief.low.min()) - 0.05, float(belief.high.max()) + 0.05)
        axes.invert_yaxis()
        axes.set_xlabel("Reach semi-axis (m)")
        axes.set_title(
            "Particles (blue), mean (white), truth (black)"
            + (", dream (orange)" if hypothesis is not None else ""),
            fontsize=11,
            color=_INK,
        )


def _describe(state: ReachState) -> str:
    if not state.executed:
        return "Before the plan"
    reached = [name for name, _, ok in state.deliveries if ok]
    total = len(state.scene.requested_objects)
    outcome = "goal reached" if state.goal_reached else "goal not reached"
    return f"After the plan: {len(reached)} of {total} items within reach, {outcome}"
