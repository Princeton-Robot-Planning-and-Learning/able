"""A robot that assists several people, each with their own unknown joint limits.

Each person is a two-link arm (see two_link_arm.py) with their own limits. The
hidden parameters are every person's limits, [lo1, hi1, lo2, hi2] for person 0,
then person 1, and so on. A state is the person's joint angles followed by the
person's index, so the dynamics, the planner, and the belief updater all know
whose arm is moving. Tasks come from a schedule: a person is drawn with their
share of the robot's time, then a goal within that person's limits.

People come from a few wards. People in the same ward have similar limits, so
plans for one transfer to the others, and people in different wards do not.
"""

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from able.belief_updaters.base import BeliefUpdater
from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import Domain, FiniteTaskDistribution, TaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import (
    JOINT_LIMIT_PRIOR,
    ArmTask,
    ArmTaskDistribution,
    JointQuery,
    TwoLinkArmModel,
)
from able.structs import QueryResponse, Task

# The entries of a person's limits within the hidden parameters.
_NUM_LIMITS = 4


def get_person(state: NDArray[np.float64]) -> int:
    """The index of the person whose arm a state describes."""
    return int(state[2])


def get_block(person: int) -> slice:
    """The entries of a person's limits within the hidden parameters."""
    return slice(_NUM_LIMITS * person, _NUM_LIMITS * (person + 1))


class MultiPersonArmModel(TwoLinkArmModel):
    """Two-link arms for several people, whose motion is clipped by the limits
    of the person in the state.

    States are [q1, q2, person], actions are joint displacements with shape (2,),
    and hidden parameters are every person's joint limits, with shape
    (4 * num_people,). Rollouts return joint angles only. Queries are not
    supported.
    """

    def __init__(self, num_people: int, goal_tolerance: float = 0.1) -> None:
        super().__init__(goal_tolerance=goal_tolerance)
        self.num_people = num_people

    def sample_next_state(
        self,
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        limits = hidden_params[get_block(get_person(state))]
        joints = super().sample_next_state(state[:2], action, limits, rng)
        return np.append(joints, state[2])

    def get_cost(
        self,
        task: ArmTask,
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        next_state: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> float:
        return self.action_cost_weight * float(np.abs(action).sum())

    def get_terminal_cost(
        self,
        task: ArmTask,
        state: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> float:
        return float(np.linalg.norm(self.forward_kinematics(state[:2]) - task.goal))

    def check_goal(
        self,
        task: ArmTask,
        state: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> bool:
        distance = np.linalg.norm(self.forward_kinematics(state[:2]) - task.goal)
        return bool(distance <= self.goal_tolerance)

    def sample_response(
        self,
        query: JointQuery,
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> bool:
        raise NotImplementedError("The multi-person arm takes no queries")

    def rollout_batch(
        self,
        initial_state: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        limits = hidden_params[:, get_block(get_person(initial_state))]
        return super().rollout_batch(initial_state[:2], action_sequences, limits, rng)

    def get_cost_lower_bound(
        self, task: ArmTask, hidden_params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        limits = hidden_params[:, get_block(get_person(task.initial_state))]
        return super().get_cost_lower_bound(task, limits)


class ScheduleTaskDistribution(TaskDistribution[ArmTask]):
    """Draws a person with their share of the schedule, then a goal within that
    person's limits that leans toward them (see ArmTaskDistribution). Every task
    starts with the person's arm at q = 0."""

    def __init__(
        self,
        model: TwoLinkArmModel,
        limits: NDArray[np.float64],
        schedule: Sequence[float] | NDArray[np.float64],
        goal_lean: float = 4.0,
    ) -> None:
        self._model = model
        self._limits = np.asarray(limits, dtype=np.float64)
        self._schedule = np.asarray(schedule, dtype=np.float64) / np.sum(schedule)
        self._goal_lean = goal_lean

    @property
    def schedule(self) -> NDArray[np.float64]:
        """Each person's share of the tasks."""
        return self._schedule

    @property
    def depends_on_hidden_params(self) -> bool:
        return True

    def given(self, hidden_params: Any) -> "ScheduleTaskDistribution":
        """Goals within these limits for every person."""
        return ScheduleTaskDistribution(
            self._model, hidden_params, self._schedule, self._goal_lean
        )

    def for_person(self, person: int) -> "ScheduleTaskDistribution":
        """Tasks for one person only."""
        schedule = np.zeros(len(self._schedule))
        schedule[person] = 1.0
        return ScheduleTaskDistribution(
            self._model, self._limits, schedule, self._goal_lean
        )

    def sample(self, rng: np.random.Generator) -> ArmTask:
        person = int(rng.choice(len(self._schedule), p=self._schedule))
        goals = ArmTaskDistribution(
            self._model, self._limits[get_block(person)], self._goal_lean
        )
        goal = goals.sample(rng).goal
        return Task(initial_state=np.array([0.0, 0.0, float(person)]), goal=goal)


class RoutineTaskDistribution(FiniteTaskDistribution[ArmTask]):
    """Tasks from fixed routines: each ward has a list of target poses, and every
    person in the ward practices them. A task draws a person with their share of
    the schedule, then one of their ward's goals uniformly. Every task starts
    with the person's arm at q = 0."""

    def __init__(
        self,
        routines: Sequence[Sequence[NDArray[np.float64]]],
        ward_of: Sequence[int],
        schedule: Sequence[float] | NDArray[np.float64],
    ) -> None:
        self._ward_of = list(ward_of)
        self._schedule = np.asarray(schedule, dtype=np.float64) / np.sum(schedule)
        self._routines = [
            [
                Task(initial_state=np.array([0.0, 0.0, float(person)]), goal=goal)
                for goal in routines[ward]
            ]
            for person, ward in enumerate(self._ward_of)
        ]
        super().__init__([task for tasks in self._routines for task in tasks])

    @property
    def schedule(self) -> NDArray[np.float64]:
        """Each person's share of the tasks."""
        return self._schedule

    def get_ward(self, person: int) -> int:
        """The ward a person belongs to."""
        return self._ward_of[person]

    def get_routine(self, person: int) -> list[ArmTask]:
        """A person's tasks, one per goal of their ward's routine, in order."""
        return list(self._routines[person])

    def sample(self, rng: np.random.Generator) -> ArmTask:
        person = int(rng.choice(len(self._schedule), p=self._schedule))
        routine = self._routines[person]
        return routine[int(rng.integers(len(routine)))]


class MultiPersonLimitUpdater(
    BeliefUpdater[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], JointQuery, bool
    ]
):
    """Narrows the belief over the limits of the person whose arm moved, as
    JointLimitBeliefUpdater does for one person, and leaves the others alone."""

    def __init__(self, tolerance: float = 1e-6) -> None:
        self._one_person = JointLimitBeliefUpdater(tolerance)

    def update_from_transition(
        self,
        belief: Belief[NDArray[np.float64]],
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        next_state: NDArray[np.float64],
    ) -> IntervalBelief:
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        block = get_block(get_person(state))
        intervals = belief.intervals.copy()
        updated = self._one_person.update_from_transition(
            IntervalBelief(intervals[block]), state[:2], action, next_state[:2]
        )
        intervals[block] = updated.intervals
        return IntervalBelief(intervals)

    def update_from_queries(
        self,
        belief: Belief[NDArray[np.float64]],
        data: Sequence[QueryResponse[JointQuery, bool]],
    ) -> Belief[NDArray[np.float64]]:
        assert not data, "The multi-person arm takes no queries"
        return belief


def get_person_plan_features(
    task: ArmTask, belief: IntervalBelief
) -> NDArray[np.float64]:
    """The features of planning for a task: the goal, then the midpoint and width
    of each of the person's intervals. People with the same belief and goal get
    the same features."""
    intervals = belief.intervals[get_block(get_person(task.initial_state))]
    return np.concatenate(
        [task.goal, intervals.mean(axis=1), intervals[:, 1] - intervals[:, 0]]
    )


MultiPersonDomain = Domain[
    NDArray[np.float64],
    NDArray[np.float64],
    ArmTask,
    NDArray[np.float64],
    JointQuery,
    bool,
]


def make_multi_person_domain(
    seed: int,
    num_wards: int = 3,
    people_per_ward: int = 2,
    ward_spread: float = 0.15,
    intake_width: float = 0.6,
    goal_lean: float = 4.0,
    goal_tolerance: float = 0.1,
    routine_size: int | None = None,
    ward_sizes: Sequence[int] | None = None,
) -> MultiPersonDomain:
    """A domain with num_wards * people_per_ward people, or with ward_sizes[w]
    people in ward w if ward_sizes is given.

    Each ward's limits are drawn from the arm's prior, and each person's limits
    are their ward's plus a uniform draw from [-ward_spread, ward_spread] per
    limit, within the prior's range. The schedule gives each person a random
    share of the robot's time. The robot starts knowing each person's limits to
    within intake_width, as after an intake assessment: each interval has that
    width and contains the true limit at a random position.

    If routine_size is set, each ward practices a fixed routine of that many
    goals (see RoutineTaskDistribution), drawn as for the ward's limits narrowed
    by ward_spread on every side so that every person in the ward can reach
    them. Otherwise each task draws a new goal within the person's limits.
    """
    rng = np.random.default_rng(seed)
    sizes = [people_per_ward] * num_wards if ward_sizes is None else list(ward_sizes)
    num_wards, num_people = len(sizes), sum(sizes)
    lows, highs = JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 1]
    wards = rng.uniform(lows, highs, size=(num_wards, _NUM_LIMITS))
    people = np.repeat(wards, sizes, axis=0)
    people = np.clip(
        people + rng.uniform(-ward_spread, ward_spread, people.shape), lows, highs
    )
    truth = people.reshape(-1)
    offsets = rng.uniform(0.0, intake_width, truth.shape)
    tiled_lows, tiled_highs = np.tile(lows, num_people), np.tile(highs, num_people)
    starts = np.clip(truth - offsets, tiled_lows, tiled_highs - intake_width)
    intervals = np.stack([starts, starts + intake_width], axis=1)
    intervals[:, 0] = np.minimum(intervals[:, 0], truth)
    intervals[:, 1] = np.maximum(intervals[:, 1], truth)
    schedule = rng.dirichlet(np.ones(num_people))
    model = MultiPersonArmModel(num_people, goal_tolerance=goal_tolerance)
    task_distribution: TaskDistribution[ArmTask] = ScheduleTaskDistribution(
        model, truth, schedule, goal_lean
    )
    if routine_size is not None:
        narrowed = wards + np.array([ward_spread, -ward_spread] * 2)
        routines = [
            [
                ArmTaskDistribution(model, ward, goal_lean).sample(rng).goal
                for _ in range(routine_size)
            ]
            for ward in narrowed
        ]
        ward_of = [w for w, size in enumerate(sizes) for _ in range(size)]
        task_distribution = RoutineTaskDistribution(routines, ward_of, schedule)
    return Domain(
        env_model=model,
        task_distribution=task_distribution,
        initial_belief=IntervalBelief(intervals),
        real_env=SimulatedEnvironment(model, truth, rng),
    )
