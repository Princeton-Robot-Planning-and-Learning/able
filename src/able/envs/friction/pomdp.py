"""The friction environment as a POMDP for POMCPOW.

The hidden state includes the floor frictions, and transitions use the analytic
slide. Observations are the noisy release speed and stop. Rewards are the
environment's push rewards; a hole ends the episode, and so does a stop from which
the robot cannot push the puck on (stuck), at the cost of the stuck penalty, which
running out of pushes also costs.
"""

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from able.envs.friction.model import PuckRewards
from able.envs.friction.world import (
    DEFAULT_FRICTION_PRIOR,
    FrictionPrior,
    PuckWorld,
    Push,
)

# STUCK: the puck stopped where no push is feasible, which ends the real episode.
ONGOING, GOAL, HOLE, STUCK = 0, 1, 2, 3
# Evenly spaced headings that decide whether a stop is stuck, and that the proposer
# sweeps when its random tries find no push; the sweep includes the check's
# headings.
STUCK_CHECK_HEADINGS = 32
SWEEP_HEADINGS = 96
# Stops are checked for being stuck on a grid of this spacing (m), and cached. The
# stuck strips along walls are about 1 cm wide.
STUCK_CHECK_RESOLUTION = 0.002


@dataclass(frozen=True)
class PuckState:
    """A POMDP state; positions are rounded so states hash stably."""

    puck: tuple[float, float]
    robot: tuple[float, float]
    frictions: tuple[float, ...]
    outcome: int = ONGOING
    pushes: int = 0
    # Actual speed off the paddle on the push that led here.
    release_speed: float = 0.0
    # Reward of the transition that led here.
    last_reward: float = 0.0

    @property
    def puck_array(self) -> NDArray[np.float64]:
        """The puck as an array."""
        return np.asarray(self.puck, dtype=np.float64)

    @property
    def robot_array(self) -> NDArray[np.float64]:
        """The robot as an array."""
        return np.asarray(self.robot, dtype=np.float64)


# (release speed, stop x, stop y, fell in a hole), rounded to keep them hashable.
PuckObservation = tuple[float, float, float, bool]


@dataclass
class PuckPushingModel:
    """The POMDP: analytic transitions, measurements, rewards, proposals and
    greedy-rollout leaf values."""

    world: PuckWorld
    rewards: PuckRewards = field(default_factory=PuckRewards)
    friction_prior: FrictionPrior = DEFAULT_FRICTION_PRIOR
    max_pushes: int = 15
    max_proposal_tries: int = 64
    # "grid": every stance_step rad in +-max_stance; "continuous": uniform; "auto":
    # the robot picks.
    stance_mode: str = "grid"
    stance_step: float = math.radians(15)
    value_resolution: float = 0.1
    # Rollout clearance beyond the hole radius (m).
    hole_margin: float = 0.05
    # Greedy pushes per leaf rollout (None: until the episode ends), so that a
    # rollout sees the goal, a hole, getting stuck or running out of pushes.
    rollout_pushes: int | None = None
    # Path cells ahead checked so a rollout can push on from where it stops.
    onward_lookahead: int = 4
    # Frictions the rollout assumes when choosing speeds (None: prior mean).
    rollout_frictions: tuple[float, ...] | None = None
    _stuck_cache: dict[tuple[int, int], bool] = field(
        default_factory=dict, init=False, repr=False
    )

    def initial_state(
        self,
        puck: NDArray[np.float64],
        robot: NDArray[np.float64],
        frictions: Sequence[float],
        pushes: int = 0,
    ) -> PuckState:
        """A root state with the frictions."""
        return PuckState(
            puck=_round_point(puck),
            robot=_round_point(robot),
            frictions=tuple(float(f) for f in frictions),
            pushes=pushes,
        )

    def is_terminal(self, state: PuckState) -> bool:
        """Goal, hole, stuck, or out of pushes."""
        return state.outcome != ONGOING or state.pushes >= self.max_pushes

    def step(
        self, state: PuckState, action: Push, rng: np.random.Generator
    ) -> tuple[PuckState, float]:
        """Sample the push's execution noise and slide the puck.

        The robot plans its next push from the measured stop, moved off any wall, as
        the environment gives it. A puck that stops just clear of a wall can be
        measured where no push is feasible, which ends the real episode, so the
        stop is stuck if a sampled measurement of it is.
        """
        phys = self.world.physics
        angle = action.angle + float(rng.normal(0.0, phys.angle_noise))
        speed = max(
            action.speed * (1.0 + float(rng.normal(0.0, phys.speed_noise))), 0.0
        )
        next_state = self.transition(state, action, angle, speed)
        if not self.is_terminal(next_state) and next_state.release_speed > 0:
            noise = rng.normal(0.0, phys.position_obs_noise, size=2)
            measured = self.world.nearest_free_puck_position(
                next_state.puck_array + noise
            )
            if self.is_stuck(measured):
                next_state = replace(
                    next_state,
                    outcome=STUCK,
                    last_reward=next_state.last_reward - self.rewards.stuck_penalty,
                )
        return next_state, next_state.last_reward

    def transition(
        self, state: PuckState, action: Push, angle: float, speed: float
    ) -> PuckState:
        """Deterministic result of executing `action` as (angle, speed)."""
        next_state = self._transition(state, action, angle, speed)
        if next_state.outcome == ONGOING and next_state.pushes >= self.max_pushes:
            # Running out of pushes fails the task, as getting stuck does.
            penalty = self.rewards.stuck_penalty
            next_state = replace(
                next_state, last_reward=next_state.last_reward - penalty
            )
        return next_state

    def _transition(
        self, state: PuckState, action: Push, angle: float, speed: float
    ) -> PuckState:
        world, rewards = self.world, self.rewards
        puck = state.puck_array
        if not world.push_feasible(puck, action.angle, action.stance):
            # Infeasible push: costs a push, changes nothing.
            return replace(
                state,
                pushes=state.pushes + 1,
                release_speed=0.0,
                last_reward=-rewards.push_cost,
            )
        station = world.push_station(puck, angle, action.stance)
        drive = world.drive_distance(
            world.nav_node_of(state.robot_array), station, puck
        )
        if not math.isfinite(drive):
            drive = 0.0
        strike_end = world.strike_end(station, puck, angle)
        result = world.slide(
            world.release_point(puck, angle),
            speed,
            angle,
            state.frictions,
            robot=strike_end,
        )
        robot = strike_end.base
        reward = -rewards.push_cost - rewards.drive_cost * drive
        stop = _round_point(result.stop)
        if result.hole is not None:
            outcome = HOLE
            reward -= rewards.hole_penalty
        elif world.in_goal(result.stop):
            outcome = GOAL
            reward += rewards.goal_reward
        elif self.is_stuck(np.array(stop)):
            outcome = STUCK
            reward -= rewards.stuck_penalty
        else:
            outcome = ONGOING
        return PuckState(
            puck=stop,
            robot=_round_point(robot),
            frictions=state.frictions,
            outcome=outcome,
            pushes=state.pushes + 1,
            release_speed=round(speed, 4),
            last_reward=reward,
        )

    def is_stuck(self, puck: NDArray[np.float64]) -> bool:
        """Whether no push is feasible from the puck, on STUCK_CHECK_HEADINGS evenly
        spaced headings with the robot's own stance. The answer is cached on a grid
        of STUCK_CHECK_RESOLUTION.

        Before this check, a stuck stop was only valued by `value`. But POMCPOW
        pools the stops of similar observations in one node, so a stuck stop drawn
        from a node with pushes tried from other stops took those pushes, each
        infeasible and costing only a push, and deeper searches undervalued getting
        stuck.
        """
        res = STUCK_CHECK_RESOLUTION
        key = (int(round(puck[0] / res)), int(round(puck[1] / res)))
        if key not in self._stuck_cache:
            point = np.array(key, dtype=np.float64) * res
            angles = np.linspace(
                -math.pi, math.pi, STUCK_CHECK_HEADINGS, endpoint=False
            )
            self._stuck_cache[key] = not any(
                self.world.push_feasible(point, float(a)) for a in angles
            )
        return self._stuck_cache[key]

    def reward(self, state: PuckState, action: Push, next_state: PuckState) -> float:
        """Reward stored on `next_state` by its transition."""
        del state, action
        return next_state.last_reward

    def observe(
        self, next_state: PuckState, action: Push, rng: np.random.Generator
    ) -> PuckObservation:
        """Noisy release speed and stop position."""
        del action
        phys = self.world.physics
        speed = next_state.release_speed + float(rng.normal(0.0, phys.speed_obs_noise))
        stop = next_state.puck_array + rng.normal(0.0, phys.position_obs_noise, size=2)
        return (
            round(max(speed, 0.0), 4),
            round(float(stop[0]), 4),
            round(float(stop[1]), 4),
            next_state.outcome == HOLE,
        )

    def likelihood(
        self, observation: PuckObservation, next_state: PuckState, action: Push
    ) -> float:
        """Gaussian density of the measurements; the hole flag must match."""
        del action
        phys = self.world.physics
        if observation[3] != (next_state.outcome == HOLE):
            return 0.0
        speed_err = (observation[0] - next_state.release_speed) / phys.speed_obs_noise
        dx = (observation[1] - next_state.puck[0]) / phys.position_obs_noise
        dy = (observation[2] - next_state.puck[1]) / phys.position_obs_noise
        return math.exp(-0.5 * (speed_err**2 + dx**2 + dy**2))

    def propose(
        self, state: PuckState, tried: Sequence[Push], rng: np.random.Generator
    ) -> Push | None:
        """Uniformly random feasible push: random heading, then a fitting stance, then
        speed. If max_proposal_tries random headings find none, the SWEEP_HEADINGS
        evenly spaced headings are tried in a random order, so a push is found
        wherever only a few headings are feasible."""
        del tried
        puck = state.puck_array
        for _ in range(self.max_proposal_tries):
            push = self._push_at(puck, float(rng.uniform(-math.pi, math.pi)), rng)
            if push is not None:
                return push
        angles = np.linspace(-math.pi, math.pi, SWEEP_HEADINGS, endpoint=False)
        for angle in rng.permutation(angles):
            push = self._push_at(puck, float(angle), rng)
            if push is not None:
                return push
        return None

    def _push_at(
        self, puck: NDArray[np.float64], angle: float, rng: np.random.Generator
    ) -> Push | None:
        """A push along the heading with a fitting stance and a random speed, or None
        if no stance fits."""
        phys = self.world.physics
        if self.stance_mode == "auto":
            if not self.world.push_feasible(puck, angle):
                return None
            speed = float(rng.uniform(phys.min_speed, phys.max_speed))
            return Push(round(angle, 4), round(speed, 4))
        stance = self._sample_stance(puck, angle, rng)
        if stance is None:
            return None
        speed = float(rng.uniform(phys.min_speed, phys.max_speed))
        # Round for hashing but stay within the limit.
        limit = phys.max_stance
        stance = min(max(round(stance, 4), -limit), limit)
        return Push(round(angle, 4), round(speed, 4), stance)

    @cached_property
    def stances(self) -> NDArray[np.float64]:
        """Stance grid (rad) for "grid" mode."""
        limit = self.world.physics.max_stance
        n = int(math.floor(limit / self.stance_step + 1e-9))
        return self.stance_step * np.arange(-n, n + 1, dtype=np.float64)

    def _sample_stance(
        self, puck: NDArray[np.float64], angle: float, rng: np.random.Generator
    ) -> float | None:
        """A random stance from which the robot can push along `angle`, or None."""
        world = self.world
        if self.stance_mode == "continuous":
            limit = world.physics.max_stance
            for _ in range(len(world.physics.base_offsets)):
                stance = float(rng.uniform(-limit, limit))
                if world.push_feasible(puck, angle, stance):
                    return stance
            return None
        assert self.stance_mode == "grid", f"Unknown stance mode {self.stance_mode!r}"
        for stance in rng.permutation(self.stances):
            if world.push_feasible(puck, angle, float(stance)):
                return float(stance)
        return None

    @cached_property
    def _goal_paths(self) -> tuple[NDArray[np.float64], NDArray[np.int64], int, int]:
        """Dijkstra to the goal over pushable free cells: (dist, next cell, nx, ny),
        where cell = j * nx + i."""
        world, res = self.world, self.value_resolution
        margin = world.physics.hole_radius + self.hole_margin
        nx = int(round(world.puck_map.width / res))
        ny = int(round(world.puck_map.height / res))
        free = np.zeros((ny, nx), dtype=bool)
        for j in range(ny):
            for i in range(nx):
                center = np.array([(i + 0.5) * res, (j + 0.5) * res])
                near_hole = len(world.holes) > 0 and bool(
                    np.any(np.linalg.norm(world.holes - center, axis=1) <= margin)
                )
                free[j, i] = (
                    world.puck_position_free(center)
                    and not near_hole
                    and self._pushable(center)
                )
        rows, cols, lengths = [], [], []
        for j, i in zip(*np.nonzero(free)):
            for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
                jj, ii = j + dj, i + di
                if not (0 <= jj < ny and 0 <= ii < nx and free[jj, ii]):
                    continue
                if dj and di and not (free[j, ii] and free[jj, i]):
                    continue
                length = res * math.hypot(dj, di)
                rows += [j * nx + i, jj * nx + ii]
                cols += [jj * nx + ii, j * nx + i]
                lengths += [length, length]
        goal_i = min(int(world.goal[0] // res), nx - 1)
        goal_j = min(int(world.goal[1] // res), ny - 1)
        graph = csr_matrix((lengths, (rows, cols)), shape=(nx * ny, nx * ny))
        dist, pred = dijkstra(  # type: ignore[call-overload]
            graph, indices=goal_j * nx + goal_i, return_predecessors=True
        )
        return dist, pred, nx, ny

    def _pushable(self, point: NDArray[np.float64], num_angles: int = 16) -> bool:
        """Whether some push is feasible from `point` on evenly spaced headings."""
        angles = np.linspace(-math.pi, math.pi, num_angles, endpoint=False)
        return any(self.world.push_feasible(point, float(a)) for a in angles)

    def path_to_goal(self, point: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """Grid points of a shortest puck path to the goal, ending at the goal (empty
        if none)."""
        dist, pred, nx, ny = self._goal_paths
        res = self.value_resolution
        i = min(max(int(point[0] // res), 0), nx - 1)
        j = min(max(int(point[1] // res), 0), ny - 1)
        cell = j * nx + i
        if not math.isfinite(dist[cell]):
            return []
        path = []
        while cell >= 0:
            path.append(np.array([(cell % nx + 0.5) * res, (cell // nx + 0.5) * res]))
            cell = int(pred[cell])
        path[-1] = self.world.goal.copy()
        return path

    def speed_to_travel(
        self,
        puck: NDArray[np.float64],
        angle: float,
        distance: float,
        frictions: Sequence[float],
        iterations: int = 14,
    ) -> float | None:
        """Speed that slides the puck `distance` on a clear line (None if max speed
        falls short)."""
        phys, world = self.world.physics, self.world
        release = world.release_point(puck, angle)
        target = distance - phys.accel_distance
        if target <= 0:
            return phys.min_speed
        if world.slide(release, phys.max_speed, angle, frictions).distance < target:
            return None
        low, high = phys.min_speed, phys.max_speed
        for _ in range(iterations):
            mid = (low + high) / 2
            if world.slide(release, mid, angle, frictions).distance < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def greedy_push(
        self, puck: NDArray[np.float64], frictions: Sequence[float]
    ) -> Push | None:
        """Push to stop at the farthest straight-line-reachable path point that can be
        pushed on from."""
        world = self.world
        path = self.path_to_goal(puck)
        clearance = world.physics.hole_radius + self.hole_margin
        for k in range(len(path) - 1, -1, -1):
            target = path[k]
            offset = target - puck
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6:
                continue
            if not world.puck_path_clear(puck, target, clearance):
                continue
            angle = math.atan2(offset[1], offset[0])
            if not world.push_feasible(puck, angle):
                continue
            if k < len(path) - 1:
                onward = path[min(k + self.onward_lookahead, len(path) - 1)] - target
                if not world.push_feasible(target, math.atan2(onward[1], onward[0])):
                    continue
            speed = self.speed_to_travel(puck, angle, distance, frictions)
            if speed is not None:
                return Push(angle, speed)
        return None

    def value(self, state: PuckState) -> float:
        """Greedy-rollout return; speeds use `rollout_frictions` but slides use the
        state's, so misjudging costs."""
        rewards = self.rewards
        if state.outcome != ONGOING:
            return 0.0
        frictions = self.rollout_frictions or tuple(
            float(f) for f in self.friction_prior.mean
        )
        total = 0.0
        current = state
        steps = self.max_pushes - state.pushes
        if self.rollout_pushes is not None:
            steps = min(self.rollout_pushes, steps)
        for _ in range(steps):
            push = self.greedy_push(current.puck_array, frictions)
            if push is None:
                return total - rewards.stuck_penalty
            current = self.transition(current, push, push.angle, push.speed)
            total += current.last_reward
            if current.outcome != ONGOING:
                return total
        return total


def _round_point(point: NDArray[np.float64]) -> tuple[float, float]:
    return (round(float(point[0]), 4), round(float(point[1]), 4))
