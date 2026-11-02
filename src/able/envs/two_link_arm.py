"""A two-link planar arm with unknown joint limits.

The arm stands in for a person whose range of motion the robot is learning. An
action is a joint displacement applied in one step, and the hidden joint limits clip
the motion. The hidden parameters are the joint limits [lo1, hi1, lo2, hi2].
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from able.beliefs.interval import IntervalBelief
from able.envs.base import BatchRolloutModel, Domain, FiniteTaskDistribution
from able.envs.simulated import SimulatedEnvironment
from able.structs import Task

# Prior over the joint limits, one [low, high] row per limit in the order
# [lo1, hi1, lo2, hi2]. Each lower limit is in [-pi, -0.5] and each upper limit is
# in [0.5, pi], so the initial configuration q = 0 is always within the limits.
JOINT_LIMIT_PRIOR = np.array(
    [[-np.pi, -0.5], [0.5, np.pi], [-np.pi, -0.5], [0.5, np.pi]]
)

# Number of points per edge when discretizing the boundary of the joint-limit box.
_NUM_BOUNDARY_POINTS = 48


@dataclass(frozen=True, eq=False)
class ArmTask(Task):
    """Move the end effector to the goal position."""

    goal: NDArray[np.float64]


class TwoLinkArmModel(BatchRolloutModel[ArmTask]):
    """A kinematic two-link arm whose joint motion is clipped by the joint limits.

    States are joint angles with shape (2,), actions are joint displacements with
    shape (2,), and hidden parameters are joint limits with shape (4,). Each step
    costs a small penalty on the action magnitude. The terminal cost is the
    distance from the end effector to the goal.
    """

    def __init__(
        self,
        link_lengths: tuple[float, float] = (1.0, 1.0),
        horizon: int = 15,
        max_joint_displacement: float = 0.3,
        action_cost_weight: float = 0.001,
    ) -> None:
        self.link_lengths = link_lengths
        self.max_joint_displacement = max_joint_displacement
        self.action_cost_weight = action_cost_weight
        self._horizon = horizon

    @property
    def horizon(self) -> int:
        return self._horizon

    def forward_kinematics(self, joints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map joint angles with shape (..., 2) to end-effector positions with
        shape (..., 2)."""
        l1, l2 = self.link_lengths
        t1 = joints[..., 0]
        t12 = joints[..., 0] + joints[..., 1]
        return np.stack(
            [l1 * np.cos(t1) + l2 * np.cos(t12), l1 * np.sin(t1) + l2 * np.sin(t12)],
            axis=-1,
        )

    def inverse_kinematics(
        self, position: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """All joint configurations that place the end effector at the position.

        Returns the two elbow solutions, with the first joint angle in [-pi, pi),
        or an empty list if the position is out of reach.
        """
        l1, l2 = self.link_lengths
        cos_elbow = (float(position @ position) - l1**2 - l2**2) / (2 * l1 * l2)
        if abs(cos_elbow) > 1:
            return []
        solutions = []
        for q2 in (np.arccos(cos_elbow), -np.arccos(cos_elbow)):
            q1 = np.arctan2(position[1], position[0]) - np.arctan2(
                l2 * np.sin(q2), l1 + l2 * np.cos(q2)
            )
            q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
            solutions.append(np.array([q1, q2]))
        return solutions

    def sample_next_state(
        self,
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return np.clip(state + action, hidden_params[[0, 2]], hidden_params[[1, 3]])

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
        return float(np.linalg.norm(self.forward_kinematics(state) - task.goal))

    def rollout_batch(
        self,
        initial_state: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        num_seqs, length = action_sequences.shape[:2]
        lows = hidden_params[:, [0, 2]]
        highs = hidden_params[:, [1, 3]]
        states = np.empty((num_seqs, len(hidden_params), length + 1, 2))
        states[:, :, 0] = initial_state
        for t in range(length):
            states[:, :, t + 1] = np.clip(
                states[:, :, t] + action_sequences[:, None, t], lows, highs
            )
        return states

    def get_cost_batch(
        self,
        task: ArmTask,
        state_sequences: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        hidden_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        final_positions = self.forward_kinematics(state_sequences[:, :, -1])
        distances = np.linalg.norm(final_positions - task.goal, axis=-1)
        action_costs = self.action_cost_weight * np.abs(action_sequences).sum(
            axis=(1, 2)
        )
        return distances + action_costs[:, None]

    def get_cost_lower_bound(
        self, task: ArmTask, hidden_params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The distance from the goal to the nearest reachable end-effector position
        under each value of the joint limits.

        The distance is zero if an inverse kinematics solution lies within the
        limits. Otherwise the nearest reachable position is the image of a point on
        the boundary of the joint-limit box or of a singular configuration (elbow
        straight or folded). The boundary is discretized, so the bound can exceed
        the true minimum distance by a small amount.
        """
        lo1, hi1, lo2, hi2 = hidden_params.T
        reachable = np.zeros(len(hidden_params), dtype=bool)
        for q1, q2 in self.inverse_kinematics(task.goal):
            reachable |= (lo1 <= q1) & (q1 <= hi1) & (lo2 <= q2) & (q2 <= hi2)

        fractions = np.linspace(0.0, 1.0, _NUM_BOUNDARY_POINTS)[None]
        q1_edge = lo1[:, None] + (hi1 - lo1)[:, None] * fractions
        q2_edge = lo2[:, None] + (hi2 - lo2)[:, None] * fractions

        def _constant(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.broadcast_to(values[:, None], q1_edge.shape)

        curves = [
            np.stack([q1_edge, _constant(lo2)], axis=-1),
            np.stack([q1_edge, _constant(hi2)], axis=-1),
            np.stack([_constant(lo1), q2_edge], axis=-1),
            np.stack([_constant(hi1), q2_edge], axis=-1),
        ]
        for singular_elbow in (0.0, np.pi, -np.pi):
            q2 = np.clip(np.full_like(lo2, singular_elbow), lo2, hi2)
            curves.append(np.stack([q1_edge, _constant(q2)], axis=-1))
        configs = np.concatenate(curves, axis=1)
        distances = np.linalg.norm(
            self.forward_kinematics(configs) - task.goal, axis=-1
        ).min(axis=1)
        return np.where(reachable, 0.0, distances)


def sample_goal_positions(
    model: TwoLinkArmModel,
    true_limits: NDArray[np.float64],
    num_goals: int,
    rng: np.random.Generator,
    edge_probability: float = 0.5,
) -> NDArray[np.float64]:
    """Sample end-effector goals that are reachable under the true joint limits.

    Each goal is the end-effector position of a joint configuration sampled within
    the limits, and within 80% of the distance a joint can travel in one episode.
    With probability edge_probability, each joint angle is moved to one of its
    limits. Those goals sit at the edge of the range of motion, and reaching for
    them reveals the limits. Returns an array with shape (num_goals, 2).
    """
    max_travel = 0.8 * model.max_joint_displacement * model.horizon
    lows = np.maximum(true_limits[[0, 2]], -max_travel)
    highs = np.minimum(true_limits[[1, 3]], max_travel)
    joints = rng.uniform(lows, highs, size=(num_goals, 2))
    at_edge = rng.random((num_goals, 2)) < edge_probability
    at_lower = rng.random((num_goals, 2)) < 0.5
    joints = np.where(at_edge, np.where(at_lower, lows, highs), joints)
    return model.forward_kinematics(joints)


ArmDomain = Domain[
    NDArray[np.float64], NDArray[np.float64], ArmTask, NDArray[np.float64]
]


def make_domain(
    seed: int,
    num_goals: int = 6,
    edge_probability: float = 0.5,
    true_limits: NDArray[np.float64] | None = None,
) -> ArmDomain:
    """Create an arm domain.

    The true joint limits are drawn from the prior unless given. The goals are
    sampled from the workspace reachable under the true limits, and every task
    starts at q = 0.
    """
    rng = np.random.default_rng(seed)
    model = TwoLinkArmModel()
    if true_limits is None:
        true_limits = rng.uniform(JOINT_LIMIT_PRIOR[:, 0], JOINT_LIMIT_PRIOR[:, 1])
    goals = sample_goal_positions(model, true_limits, num_goals, rng, edge_probability)
    tasks = [ArmTask(initial_state=np.zeros(2), goal=goal) for goal in goals]
    return Domain(
        env_model=model,
        task_distribution=FiniteTaskDistribution(tasks),
        initial_belief=IntervalBelief(JOINT_LIMIT_PRIOR),
        real_env=SimulatedEnvironment(model, true_limits, rng),
    )
