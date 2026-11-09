"""Core data structures shared across planners, learners, and environments."""

from dataclasses import dataclass
from typing import Generic, TypeVar

StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
GoalType = TypeVar("GoalType")
HiddenParamsType = TypeVar("HiddenParamsType")
PlannerParamsType = TypeVar("PlannerParamsType")
QueryType = TypeVar("QueryType")
ResponseType = TypeVar("ResponseType")


@dataclass(frozen=True, eq=False)
class Task(Generic[StateType, GoalType]):
    """A task: reach the goal, starting from the initial state.

    The environment model's costs define what reaching the goal means. Tasks
    compare by identity, because states and goals are often arrays.
    """

    initial_state: StateType
    goal: GoalType


TaskType = TypeVar("TaskType", bound=Task)


@dataclass(frozen=True)
class Trajectory(Generic[StateType, ActType]):
    """A sequence of states, the actions between them, and the cost of each step."""

    states: tuple[StateType, ...]
    actions: tuple[ActType, ...]
    costs: tuple[float, ...]

    def __post_init__(self) -> None:
        assert (
            len(self.states) == len(self.actions) + 1
        ), "Need one more state than actions"
        assert len(self.costs) == len(self.actions), "Need one cost per action"

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def total_cost(self) -> float:
        """The sum of the step costs."""
        return sum(self.costs)


@dataclass(frozen=True)
class QueryResponse(Generic[QueryType, ResponseType]):
    """A query about the hidden parameters and the environment's response to it.

    Each environment defines its own query and response types. Robots learn about the
    hidden parameters from these pairs and from the transitions they observe while
    executing tasks.
    """

    query: QueryType
    response: ResponseType


@dataclass(frozen=True)
class StepResult(Generic[StateType]):
    """The outcome of taking one action in an environment."""

    state: StateType
    cost: float
    terminated: bool


@dataclass(frozen=True)
class Budget:
    """Compute limits for one call to a component.

    Each component defines what counts as one iteration (for example, a batch of sampled
    action sequences for a planner, or one imagined belief update for a learner). A
    limit of None means unlimited.
    """

    max_wall_time: float | None = None
    max_iterations: int | None = None

    def is_exhausted(self, elapsed_time: float, num_iterations: int) -> bool:
        """Check whether either limit has been reached."""
        if self.max_wall_time is not None and elapsed_time >= self.max_wall_time:
            return True
        if self.max_iterations is not None and num_iterations >= self.max_iterations:
            return True
        return False


class PlanningFailure(Exception):
    """Raised when a planner or policy cannot produce an action within its budget.

    stats, if given, are the planner's statistics up to the failure, so that
    failed plans can be told apart, for example by which try ran out of time.
    """

    def __init__(
        self, message: str = "", stats: dict[str, float] | None = None
    ) -> None:
        super().__init__(message)
        self.stats: dict[str, float] = dict(stats or {})


class PlanningTimeout(PlanningFailure):
    """Raised when planning fails because a time limit ran out, rather than for
    another reason, so timeouts can be counted apart from other failures."""
