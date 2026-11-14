"""The limb repositioning environment model.

The robot holds the person's limb and moves it to a goal configuration. Each action
is a robot torque correction (on top of the torque that holds the system still),
applied for action_repeat simulator steps. The hidden parameters theta scale the
person's active range of motion. While the robot moves the limb, the person relaxes:
their joints hold up the limb and put stiff walls past their limits (see person.py),
so theta changes the physics only past the limits. Moving the limb past the person's
limit fails the task, and the person signals where it happened. As in kinder's
environment, the task ends when the limb reaches the goal. A broken safety
constraint, or the robot stopping (the STOP action) before the goal, fails the task.
Either way the state stays fixed afterwards.
"""

import dataclasses
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from able.envs.base import EnvironmentModel
from able.envs.limb.constraints import (
    HumanTorqueLimits,
    arm_in_collision,
    exceeds_robot_torque_limits,
    get_resting_penetration,
    get_robot_motion_violation,
    limb_in_collision,
)
from able.envs.limb.human import LimbVariant
from able.envs.limb.query import LimbQuery, QueryModel, QueryOutcome, QuerySimulator
from able.envs.limb.sim import CoupledState, LimbSim, LimbSimSettings
from able.structs import Task

# The action that ends the robot's motion, for example after a plan that stops short
# of the goal.
STOP = np.full(7, np.nan)
STOP.setflags(write=False)


@dataclass(frozen=True, eq=False)
class LimbState:
    """The coupled robot and limb state, and what the episode has shown so far.

    The goal is the task's goal configuration, and reached says whether the limb
    has come within the goal tolerance. For each active motion, max_reach is the
    farthest the limb has gone toward the limit, as a scale of the normal range,
    and signalled is the joint angle magnitude where the person first signalled
    that the limb passed their limit (None if they have not). A nonempty failure
    names the broken constraint. The resting penetration is the limb's contact
    with the person at the start of the task, which collision checks allow.
    """

    coupled: CoupledState
    goal: NDArray[np.float64]
    max_reach: tuple[float, ...]
    signalled: tuple[float | None, ...]
    resting_penetration: tuple[tuple[int, float], ...]
    reached: bool = False
    failure: str = ""

    @property
    def limb_positions(self) -> NDArray[np.float64]:
        """The limb's joint positions."""
        return np.array(self.coupled.limb_positions)

    @property
    def violated(self) -> bool:
        """Whether the limb has passed the person's limit."""
        return any(s is not None for s in self.signalled)


# The goal is the limb's goal joint positions, which the initial state also holds.
LimbTask = Task[LimbState, NDArray[np.float64]]


class LimbModel(
    EnvironmentModel[
        LimbState,
        NDArray[np.float64],
        LimbTask,
        NDArray[np.float64],
        LimbQuery,
        QueryOutcome,
    ]
):
    """A PyBullet model of the robot moving the person's limb.

    The model steps one simulator, restoring it only when asked to step from a
    state other than the one it produced last. The task ends when the limb is
    within the environment's goal tolerance at the end of an action. Each step
    costs nothing, and the terminal cost is the distance from the limb to the goal.
    A task succeeds if the limb reached the goal, no constraint broke, and the limb
    never passed the person's limit.
    """

    def __init__(
        self,
        variant: LimbVariant,
        settings: LimbSimSettings = LimbSimSettings(),
        horizon: int = 300,
        action_repeat: int = 12,
        max_robot_velocity: float = 2.0,
        query_model: QueryModel = QueryModel(),
    ) -> None:
        self.variant = variant
        self.settings = settings
        self.action_repeat = action_repeat
        self.max_robot_velocity = max_robot_velocity
        self.query_model = query_model
        self.torque_limits = HumanTorqueLimits.for_limb(variant.limb_name)
        self._horizon = horizon
        self._sim: LimbSim | None = None
        self._sim_state: LimbState | None = None
        self._queries = QuerySimulator(variant, query_model, settings.person)

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def sim(self) -> LimbSim:
        """The simulator, built on first use."""
        if self._sim is None:
            self._sim = LimbSim(self.variant, self.settings)
        return self._sim

    def close(self) -> None:
        """Disconnect the simulators."""
        if self._sim is not None:
            self._sim.close()
        self._sim, self._sim_state = None, None
        self._queries.close()

    def get_initial_state(
        self, coupled: CoupledState, goal: NDArray[np.float64]
    ) -> LimbState:
        """The state at the start of a task with the goal, with the limb's resting
        contact."""
        sim = self.sim
        sim.restore(coupled)
        positions = np.array(coupled.limb_positions)
        goal = np.array(goal, dtype=np.float64)
        goal.setflags(write=False)
        state = LimbState(
            coupled=coupled,
            goal=goal,
            max_reach=tuple(self.get_reach(positions)),
            signalled=(None,) * len(self.variant.active_motions),
            resting_penetration=tuple(sorted(get_resting_penetration(sim).items())),
            reached=sim.get_limb_error(positions, goal) < sim.goal_tolerance,
        )
        self._sim_state = state
        return state

    def get_reach(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """How far each active motion has gone, as a scale of its normal range."""
        variant = self.variant
        reach = np.array(variant.active_signs) * positions[list(variant.active_joints)]
        return reach / variant.default_magnitudes

    def sample_next_state(
        self,
        state: LimbState,
        action: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> LimbState:
        if state.failure or state.reached:
            return state
        if np.isnan(action).all():
            return dataclasses.replace(state, failure="stopped before the goal")
        sim = self.sim
        if state is not self._sim_state:
            sim.restore(state.coupled)
        resting: Mapping[int, float] = dict(state.resting_penetration)
        limits = np.abs(self.variant.get_bounds(hidden_params))
        person_limits = self.variant.get_joint_limits(hidden_params)
        magnitudes = self.variant.default_magnitudes
        max_reach = np.array(state.max_reach)
        signalled = list(state.signalled)
        failure = ""
        for _ in range(self.action_repeat):
            commanded = sim.get_commanded_torque(action)
            if exceeds_robot_torque_limits(sim, commanded):
                failure = "robot torque limits"
                break
            if self.torque_limits.are_exceeded(sim.advance(commanded, person_limits)):
                failure = "human torque limits"
            failure = failure or get_robot_motion_violation(
                sim, self.max_robot_velocity
            )
            if not failure and (
                arm_in_collision(sim) or limb_in_collision(sim, resting)
            ):
                failure = "collision"
            reach = self.get_reach(np.array(sim.env.limb.get_joint_positions()))
            max_reach = np.maximum(max_reach, reach)
            for motion, scale in enumerate(reach):
                if (
                    signalled[motion] is None
                    and scale * magnitudes[motion] > limits[motion]
                ):
                    signalled[motion] = float(scale * magnitudes[motion])
            if failure:
                break
        coupled = sim.capture()
        error = sim.get_limb_error(coupled.limb_positions, state.goal)
        next_state = dataclasses.replace(
            state,
            coupled=coupled,
            max_reach=tuple(max_reach),
            signalled=tuple(signalled),
            reached=not failure and error < sim.goal_tolerance,
            failure=failure,
        )
        self._sim_state = next_state
        return next_state

    def get_cost(
        self,
        task: LimbTask,
        state: LimbState,
        action: NDArray[np.float64],
        next_state: LimbState,
        hidden_params: NDArray[np.float64],
    ) -> float:
        return 0.0

    def get_goal_error(self, task: LimbTask, state: LimbState) -> float:
        """The distance from the limb to the goal configuration."""
        return self.sim.get_limb_error(state.limb_positions, task.goal)

    def get_terminal_cost(
        self, task: LimbTask, state: LimbState, hidden_params: NDArray[np.float64]
    ) -> float:
        return self.get_goal_error(task, state)

    def check_goal(
        self, task: LimbTask, state: LimbState, hidden_params: NDArray[np.float64]
    ) -> bool:
        return state.reached and not state.failure and not state.violated

    def sample_response(
        self,
        query: LimbQuery,
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> QueryOutcome:
        return self._queries.ask(hidden_params, query, rng)
