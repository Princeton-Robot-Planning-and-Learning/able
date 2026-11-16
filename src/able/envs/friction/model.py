"""The friction environment model.

The robot pushes a puck through the threeroutes maze to a goal, from a start and
to a goal that vary by task. Each action is one push: the robot drives behind the
puck and strikes it at a heading and speed, with a stance. The hidden parameters are
the friction coefficients of the floors. A puck that stops too close to a wall for
the robot to push it on is shed to the nearest spot it can push from
(PuckWorld.shed). The task ends when the puck stops in the goal or passes over a
hole, and the state stays fixed afterwards.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.envs.base import EnvironmentModel
from able.envs.friction.layout import make_sim, make_world
from able.envs.friction.query import (
    FrictionQuery,
    ProbeSimulator,
    PushRecord,
    execute_push,
)
from able.envs.friction.world import Push
from able.structs import Task

# The action after the task has ended, which changes nothing.
NO_PUSH = Push(0.0, 0.0)


@dataclass(frozen=True)
class PuckRewards:
    """The rewards of a push, whose negatives are the step costs."""

    push_cost: float = 1.0
    drive_cost: float = 0.5  # per m driven
    goal_reward: float = 20.0
    hole_penalty: float = 50.0
    # Value of a state with no feasible push (e.g. inside the slot), for planners.
    stuck_penalty: float = 30.0

    def get_reward(self, record: PushRecord) -> float:
        """The reward of a push."""
        drive = record.drive_distance if np.isfinite(record.drive_distance) else 0.0
        reward = -self.push_cost - self.drive_cost * drive
        if record.fell_in_hole:
            reward -= self.hole_penalty
        elif record.goal_reached:
            reward += self.goal_reward
        return reward


@dataclass(frozen=True)
class FrictionState:
    """What the robot knows of the puck and itself, the true poses, the task's
    goal, and the last push.

    Planners and belief updaters use only what the robot knows: `puck` is the
    measured stop moved off any wall, and `robot` is where the last push left the
    base. The true puck position and base pose let the model continue the episode
    in a fresh simulator. The goal is the center of the goal circle, which the
    simulator needs to end the episode. The outcome is "goal" or "hole" once the
    task has ended.
    """

    puck: tuple[float, float]
    robot: tuple[float, float]
    true_puck: tuple[float, float]
    robot_pose: tuple[float, float, float]
    goal: tuple[float, float]
    pushes: int = 0
    last_push: PushRecord | None = None
    outcome: str | None = None
    # Whether `puck` is a measured stop, rather than known exactly.
    puck_measured: bool = False

    @property
    def ended(self) -> bool:
        """Whether the puck has stopped in the goal or passed over a hole."""
        return self.outcome is not None

    @property
    def puck_array(self) -> NDArray[np.float64]:
        """The puck as the robot knows it, as an array."""
        return np.array(self.puck, dtype=np.float64)

    @property
    def robot_array(self) -> NDArray[np.float64]:
        """The robot's position as it knows it, as an array."""
        return np.array(self.robot, dtype=np.float64)


# The goal is the center of the goal circle.
FrictionTask = Task[FrictionState, tuple[float, float]]


class FrictionModel(
    EnvironmentModel[
        FrictionState,
        Push,
        FrictionTask,
        NDArray[np.float64],
        FrictionQuery,
        PushRecord,
    ]
):
    """A simulator model of pushing the puck, in 2D (PyMunk) or 3D (PyBullet).

    Each step resets the simulator to the state's true poses with the hidden
    frictions, seeded from the given random stream, and runs the push. The step
    cost is the negative of the push's reward, and nothing is charged after the
    task ends. A task succeeds if the puck stopped in the goal.
    """

    def __init__(
        self,
        sim: str = "2d",
        num_holes: int = 2,
        goal_radius: float = 0.3,
        horizon: int = 15,
        rewards: PuckRewards = PuckRewards(),
        num_floors: int = 3,
    ) -> None:
        self.sim_name = sim
        self.rewards = rewards
        self.world = make_world(num_holes, goal_radius, num_floors)
        self._horizon = horizon
        self._sim: Any = None
        self._probes = ProbeSimulator(sim, num_holes, goal_radius, num_floors)

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def sim(self) -> Any:
        """The simulator, built on first use."""
        if self._sim is None:
            self._sim = make_sim(self.sim_name, self.world)
        return self._sim

    def close(self) -> None:
        """Close the simulators."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        self._probes.close()

    def get_initial_state(
        self,
        puck: tuple[float, float] | None = None,
        robot: tuple[float, float] | None = None,
        goal: tuple[float, float] | None = None,
    ) -> FrictionState:
        """The puck and the robot at their starts, the robot facing +x, for the
        task with the goal. Each defaults to the map's default task."""
        puck_map = self.world.puck_map
        puck = puck_map.puck_start if puck is None else puck
        robot = puck_map.robot_start if robot is None else robot
        goal = puck_map.goal if goal is None else goal
        return FrictionState(puck, robot, puck, (robot[0], robot[1], 0.0), goal)

    def sample_next_state(
        self,
        state: FrictionState,
        action: Push,
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> FrictionState:
        if state.ended:
            return state
        sim = self.sim
        options = {
            "frictions": tuple(float(f) for f in hidden_params),
            "puck": state.true_puck,
            "robot": state.robot_pose,
            "goal": state.goal,
        }
        sim.reset(seed=int(rng.integers(2**31)), options=options)
        terminated, info = execute_push(sim, action)
        record = PushRecord.from_info(info, state.puck, action, state.puck_measured)
        robot = state.robot
        if record.feasible:
            world = self.world
            station = world.push_station(state.puck_array, action.angle, action.stance)
            moved = station + world.physics.accel_distance * action.direction
            robot = (float(moved[0]), float(moved[1]))
        outcome = None
        if terminated:
            outcome = "hole" if record.fell_in_hole else "goal"
        # A noisy measurement can land the puck slightly inside a wall.
        puck = self.world.nearest_free_puck_position(np.array(record.stop))
        true_puck, pose = np.asarray(sim.puck_position), sim.robot_pose
        measured = True
        if outcome is None:
            rest = self.world.shed(true_puck, state.goal)
            if not np.array_equal(rest, true_puck):
                # The robot sees where a shed puck comes to rest.
                puck, true_puck, measured = rest, rest, False
        return FrictionState(
            puck=(float(puck[0]), float(puck[1])),
            robot=robot,
            true_puck=(float(true_puck[0]), float(true_puck[1])),
            robot_pose=(float(pose[0]), float(pose[1]), float(pose[2])),
            goal=state.goal,
            pushes=state.pushes + 1,
            last_push=record,
            outcome=outcome,
            puck_measured=measured,
        )

    def get_cost(
        self,
        task: FrictionTask,
        state: FrictionState,
        action: Push,
        next_state: FrictionState,
        hidden_params: NDArray[np.float64],
    ) -> float:
        if state.ended or next_state.last_push is None:
            return 0.0
        return -self.rewards.get_reward(next_state.last_push)

    def get_terminal_cost(
        self,
        task: FrictionTask,
        state: FrictionState,
        hidden_params: NDArray[np.float64],
    ) -> float:
        return 0.0

    def check_goal(
        self,
        task: FrictionTask,
        state: FrictionState,
        hidden_params: NDArray[np.float64],
    ) -> bool:
        return state.outcome == "goal"

    def sample_response(
        self,
        query: FrictionQuery,
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> PushRecord:
        return self._probes.ask(hidden_params, query, rng)
