"""The human reach environment model.

The robot moves requested items from far furniture to the bedside of a person
sitting up in bed, so that the person can reach them. The hidden parameters theta
are the semi-axes of the person's reach ellipsoid at their right shoulder. One
action is a whole PDDLStream plan (move_base, pick, and place actions), which the
simulator executes with refined motions. Afterwards the robot sees, for each
requested item resting on a surface, whether the person reached it.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.envs.base import EnvironmentModel
from able.envs.human_reach.execute import execute_plan
from able.envs.human_reach.query import QuerySimulator, ReachQuery, ReachResponse
from able.envs.human_reach.scene import (
    BedsideStocking3DEnvConfig,
    BedsideStocking3DObjectCentricState,
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
)
from able.structs import Task

# A PDDLStream plan: (action name, arguments) pairs.
ReachPlan = tuple[tuple[str, tuple[Any, ...]], ...]


@dataclass(frozen=True, eq=False)
class ReachState:
    """The scene, and what executing the plan showed.

    Each delivery is a requested item resting on a surface after the plan, its
    grasp point, and whether the person reached it. Once the plan has run, the
    state stays fixed. num_unreached counts the requested items that the person
    cannot reach, because they are out of reach, blocked, or not on a surface.
    """

    scene: BedsideStocking3DObjectCentricState
    executed: bool = False
    deliveries: tuple[tuple[str, tuple[float, float, float], bool], ...] = ()
    num_unreached: int = 0
    goal_reached: bool = False


# The goal is the requested items' names.
ReachTask = Task[ReachState, tuple[str, ...]]


class ReachModel(
    EnvironmentModel[
        ReachState,
        ReachPlan,
        ReachTask,
        NDArray[np.float64],
        ReachQuery,
        ReachResponse,
    ]
):
    """A PyBullet model of stocking the bedside.

    Every episode is one step, which executes the plan. The task succeeds if every
    requested item ends on a surface, within the person's reach, and unobstructed
    by other items. Steps cost nothing, and the terminal cost is the number of
    requested items the person cannot reach. Queries run in a simulator of their
    own.
    """

    def __init__(
        self,
        num_requested: int = 4,
        num_clutter: int = 4,
        config: BedsideStocking3DEnvConfig = BedsideStocking3DEnvConfig(),
    ) -> None:
        self.num_requested = num_requested
        self.num_clutter = num_clutter
        self.config = config
        self._sim: ObjectCentricBedsideStocking3DEnv | None = None
        self._queries = QuerySimulator(config)

    @property
    def horizon(self) -> int:
        return 1

    @property
    def sim(self) -> ObjectCentricBedsideStocking3DEnv:
        """The simulator, built on first use."""
        if self._sim is None:
            self._sim = ObjectCentricBedsideStocking3DEnv(
                self.num_requested,
                self.num_clutter,
                config=self.config,
                allow_state_access=True,
            )
        return self._sim

    def close(self) -> None:
        """Close the simulators."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        self._queries.close()

    def get_initial_state(self, scene_seed: int) -> ReachState:
        """The scene for a seed, which places the requested items and clutter."""
        scene, _ = self.sim.reset(seed=scene_seed)
        assert isinstance(scene, BedsideStocking3DObjectCentricState)
        return ReachState(scene, num_unreached=self.num_requested)

    def sample_next_state(
        self,
        state: ReachState,
        action: ReachPlan,
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> ReachState:
        if state.executed:
            return state
        sim = self.sim
        sim.reset(options={"init_state": state.scene})
        sim.reach_theta = np.array(hidden_params, dtype=float)
        goal_reached = execute_plan(
            sim, list(action), seed=int(rng.integers(2**31 - 1))
        )
        deliveries, num_unreached = [], 0
        for name in sim.requested_objects:
            # pylint: disable-next=protected-access
            body = sim._object_name_to_pybullet_id(name)
            # pylint: disable-next=protected-access
            if sim._grasped_object == name or not sim._get_surfaces_supporting_object(
                body
            ):
                num_unreached += 1
                continue
            point = sim.grasp_point(name)
            margin = reach_margin(point, sim.reach_theta, sim.shoulder)
            xyz = (float(point[0]), float(point[1]), float(point[2]))
            deliveries.append((name, xyz, bool(margin >= 0)))
            num_unreached += int(not sim.item_reachable(name))
        return ReachState(
            scene=sim._get_obs(),  # pylint: disable=protected-access
            executed=True,
            deliveries=tuple(deliveries),
            num_unreached=num_unreached,
            goal_reached=goal_reached,
        )

    def get_cost(
        self,
        task: ReachTask,
        state: ReachState,
        action: ReachPlan,
        next_state: ReachState,
        hidden_params: NDArray[np.float64],
    ) -> float:
        return 0.0

    def get_terminal_cost(
        self, task: ReachTask, state: ReachState, hidden_params: NDArray[np.float64]
    ) -> float:
        return float(state.num_unreached)

    def check_goal(
        self, task: ReachTask, state: ReachState, hidden_params: NDArray[np.float64]
    ) -> bool:
        return state.goal_reached

    def sample_response(
        self,
        query: ReachQuery,
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> ReachResponse:
        return self._queries.ask(hidden_params, query, rng)
