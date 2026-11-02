"""Running policies in environments."""

from typing import Any, Callable

from able.envs.base import Environment
from able.policies.base import Policy
from able.structs import ActType, StateType, TaskType, Trajectory


def run_episode(
    env: Environment[StateType, ActType, TaskType, Any, Any],
    task: TaskType,
    policy: Policy[StateType, ActType],
    max_steps: int | None = None,
    on_step: Callable[[StateType, ActType, StateType], None] | None = None,
) -> Trajectory[StateType, ActType]:
    """Reset the environment to the task and run the policy until the episode ends.

    The episode ends when the environment reports termination or after max_steps.
    If given, on_step is called with the state, action, and next state after every
    step, so callers can update a belief during execution. A PlanningFailure
    raised by the policy propagates to the caller.
    """
    states = [env.reset(task)]
    actions: list[ActType] = []
    costs: list[float] = []
    while max_steps is None or len(actions) < max_steps:
        action = policy.get_action(states[-1])
        result = env.step(action)
        if on_step is not None:
            on_step(states[-1], action, result.state)
        states.append(result.state)
        actions.append(action)
        costs.append(result.cost)
        if result.terminated:
            break
    return Trajectory(tuple(states), tuple(actions), tuple(costs))
