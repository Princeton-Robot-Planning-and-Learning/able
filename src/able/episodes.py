"""Running policies in environments."""

from able.envs.base import Environment
from able.policies.base import Policy
from able.structs import ActType, StateType, TaskType, Trajectory


def run_episode(
    env: Environment[StateType, ActType, TaskType],
    task: TaskType,
    policy: Policy[StateType, ActType],
    max_steps: int | None = None,
) -> Trajectory[StateType, ActType]:
    """Reset the environment to the task and run the policy until the episode ends.

    The episode ends when the environment reports termination or after max_steps.
    A PlanningFailure raised by the policy propagates to the caller.
    """
    states = [env.reset(task)]
    actions: list[ActType] = []
    costs: list[float] = []
    while max_steps is None or len(actions) < max_steps:
        action = policy.get_action(states[-1])
        result = env.step(action)
        states.append(result.state)
        actions.append(action)
        costs.append(result.cost)
        if result.terminated:
            break
    return Trajectory(tuple(states), tuple(actions), tuple(costs))
