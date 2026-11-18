"""Tests for the arm's bisecting gatherer and the gatherers that pick queries."""

import numpy as np
import pytest

from able.approaches import make_current_belief_robot, make_dreaming_robot
from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.beliefs.interval import IntervalBelief
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, JointQuery, make_domain
from able.info_gatherers.bisect_limits import BisectLimitGatherer
from able.info_gatherers.selection import (
    FixedFirstQueryGatherer,
    JointSleepGatherer,
    OracleQueryGatherer,
    PlanningAwareGatherer,
    RandomQueryGatherer,
)
from able.learners.belief_sources import (
    Dream,
    ImaginedNextBeliefSource,
    PlannedSource,
)
from able.learners.model_learner import ModelLearner
from able.parallel import SerialPool, WorkerContext
from able.query_methods import with_query_method
from able.structs import Budget

ONE = Budget(max_iterations=1)


def test_bisect_limits():
    """Tests that bisection asks at the widest interval's midpoint, offers every
    fraction of unknown limits, and stops once all are known."""
    gatherer = BisectLimitGatherer(JointLimitBeliefUpdater())
    belief = IntervalBelief([[-3.0, -0.5], [0.5, 1.0], [-1.0, -1.0], [0.5, 2.5]])
    assert gatherer.choose(belief) == JointQuery(0, -1.75)
    candidates = gatherer.get_candidates(belief)
    assert len(candidates) == 9
    assert JointQuery(1, 1.0) in candidates
    known = IntervalBelief([[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]])
    assert gatherer.choose(known) is None

    domain = make_domain(0)
    rng = np.random.default_rng(0)
    data = gatherer.gather(
        domain.initial_belief, domain.real_env, Budget(max_iterations=3), rng
    )
    assert len(data) == 3
    belief = JointLimitBeliefUpdater().update_from_queries(domain.initial_belief, data)
    assert isinstance(belief, IntervalBelief)
    assert (
        belief.get_total_width() < IntervalBelief(JOINT_LIMIT_PRIOR).get_total_width()
    )


def test_fixed_and_random_first_queries():
    """Tests that a fixed gatherer asks its query first and then bisects, and that
    a random one asks one of the candidates."""
    base = BisectLimitGatherer(JointLimitBeliefUpdater())
    domain = make_domain(0)
    rng = np.random.default_rng(0)
    query = JointQuery(1, 2.0)
    data = FixedFirstQueryGatherer(base, query).gather(
        domain.initial_belief, domain.real_env, Budget(max_iterations=2), rng
    )
    assert data[0].query == query
    assert data[1].query == base.choose(
        base.belief_updater.update_from_queries(domain.initial_belief, data[:1])
    )
    candidates = base.get_candidates(domain.initial_belief)
    for _ in range(5):
        data = RandomQueryGatherer(base).gather(
            domain.initial_belief, domain.real_env, ONE, rng
        )
        assert data[0].query in candidates


def test_planning_aware_scores_against_the_sleep():
    """Tests that a dreaming robot's scores differ by candidate and the winner is
    committed and asked, and that a current-belief robot's ties fall back to
    bisection."""
    domain = make_domain(0)
    rng = np.random.default_rng(0)
    budget = Budget(max_iterations=4)
    for method, make in [("joint", "dreaming"), ("sleep_blind", "dreaming")]:
        robot = make_dreaming_robot(domain, ONE, ONE, Budget())
        robot = with_query_method(
            robot, domain, method, ONE, ONE, Budget(), num_eval_dreams=8
        )
        gatherer = robot.info_gatherer
        assert isinstance(gatherer, PlanningAwareGatherer), make
        source = robot.learner.belief_source  # type: ignore[attr-defined]
        assert isinstance(source, ImaginedNextBeliefSource)
        robot.sleep(budget, rng)
        assert len(gatherer.scores) == 12
        candidates = gatherer.get_candidates(domain.initial_belief, rng)
        best = candidates[int(np.argmin(gatherer.scores))]
        assert gatherer.get_commitment() == best
        assert source.get_draw_state() == best
        data = robot.gather_information(domain.real_env, ONE, rng)
        assert data[0].query == best
        if method == "sleep_blind":
            assert gatherer.scores == [1.0] * 12

    robot = make_current_belief_robot(domain, Budget())
    robot = with_query_method(robot, domain, "joint", ONE, ONE, Budget())
    robot.sleep(budget, rng)
    gatherer = robot.info_gatherer
    assert isinstance(gatherer, PlanningAwareGatherer)
    assert gatherer.scores == [1.0] * 12
    assert gatherer.get_commitment() == JointQuery(0, (-np.pi - 0.5) / 2)


def test_oracle_commits_the_best_branch():
    """Tests that the oracle scores every candidate's branch and commits the one
    with the best success rate, then the lowest planning time."""
    domain = make_domain(0)
    robot = make_dreaming_robot(domain, ONE, ONE, Budget())
    distribution = domain.task_distribution
    assert isinstance(distribution, FiniteTaskDistribution)
    tasks = list(distribution.get_tasks())[:3]  # pylint: disable=no-member
    robot = with_query_method(robot, domain, "oracle", ONE, ONE, Budget(), tasks)
    gatherer = robot.info_gatherer
    assert isinstance(gatherer, OracleQueryGatherer)
    robot.sleep(Budget(max_iterations=2), np.random.default_rng(0))
    assert len(gatherer.scores) == 12
    best = max(gatherer.scores, key=lambda s: (s[0], -s[1]))
    index = gatherer.scores.index(best)
    candidates = gatherer.get_candidates(domain.initial_belief, np.random.default_rng())
    assert gatherer.get_commitment() == candidates[index]


def test_query_methods():
    """Tests that the default leaves the robot alone and that other methods
    replace the robot's and its dreams' gatherer."""
    domain = make_domain(0)
    robot = make_dreaming_robot(domain, ONE, ONE, Budget())
    gatherer = robot.info_gatherer
    assert with_query_method(robot, domain).info_gatherer is gatherer
    robot = with_query_method(robot, domain, "base", ONE, ONE, Budget())
    assert isinstance(robot.info_gatherer, BisectLimitGatherer)
    robot = with_query_method(robot, domain, "random", ONE, ONE, Budget())
    assert isinstance(robot.info_gatherer, RandomQueryGatherer)
    with pytest.raises(AssertionError):
        with_query_method(robot, domain, "unknown", ONE, ONE, Budget())


def test_workers_adopt_the_commitment():
    """Tests that a sleep in workers passes the main gatherer's commitment to the
    workers' dreams."""
    domain = make_domain(0)
    robots = [
        with_query_method(
            make_dreaming_robot(domain, ONE, ONE, Budget()),
            domain,
            "random",
            ONE,
            ONE,
            Budget(),
        )
        for _ in range(2)
    ]
    learners = [
        ModelLearner(
            r.planner,
            domain.task_distribution,
            r.learner.belief_source,  # type: ignore[attr-defined]
            Budget(),
            refit="per_sleep",
        )
        for r in robots
    ]
    query = JointQuery(1, 2.0)
    robots[0].info_gatherer.set_commitment(query)
    pool = SerialPool(WorkerContext(domain, robots[1].planner, learners[1]))
    learners[0].learn(
        domain.initial_belief,
        robots[0].planner_params,
        Budget(max_iterations=2),
        np.random.default_rng(0),
        pool,
    )
    assert robots[1].info_gatherer.get_commitment() == query


class _ScriptedLearner:
    """A learner that has trained under nothing."""

    training_beliefs: list[IntervalBelief] = []


def _scripted_gatherer(
    dreams: dict[int, list[Dream[np.ndarray]]],
    uncertainty_weight: float,
    planning_weight: float,
) -> tuple[JointSleepGatherer, PlannedSource[np.ndarray, object]]:
    domain = make_domain(0)
    base = BisectLimitGatherer(JointLimitBeliefUpdater())
    learner = make_dreaming_robot(domain, ONE, ONE, Budget()).learner
    assert isinstance(learner, ModelLearner)
    dreamer = learner.belief_source
    assert isinstance(dreamer, ImaginedNextBeliefSource)
    source: PlannedSource[np.ndarray, object] = PlannedSource()

    class Scripted(JointSleepGatherer):
        """Two candidates with scripted dreams."""

        def get_candidates(self, belief, rng):  # type: ignore[no-untyped-def]
            return [JointQuery(0, -1.0), JointQuery(1, 1.0)]

        def _dream(self, query, belief, params, rng):  # type: ignore[no-untyped-def]
            return dreams[query.joint]

    gatherer = Scripted(
        base,
        dreamer,
        source,
        lambda b: b.intervals[:, 0].copy(),  # type: ignore[attr-defined]
        lambda b, h: float(np.abs(b.intervals.mean(axis=1) - h).mean()),  # type: ignore[attr-defined]
        uncertainty_weight=uncertainty_weight,
        planning_weight=planning_weight,
        cost_floor=10.0,
    )
    return gatherer, source


def test_joint_sleep_trades_planning_against_uncertainty():
    """Tests that plans rank clusters by probability times saving and are scored
    on the held-out half, and that the uncertainty weight prefers the query whose
    dreams end nearer their truths."""
    prior = IntervalBelief([[-2.0, 2.0]] * 4)
    slow = IntervalBelief([[-1.0, 1.0]] * 4)
    near = [IntervalBelief([[1.0 - 0.1 * i, 1.0 + 0.1 * i]] * 4) for i in range(1, 5)]
    dreams = {
        # Uninformative but predictable: every dream lands in one costly cluster.
        0: [Dream(np.ones(4), slow, 1.0, 110.0) for _ in range(4)],
        # Informative: each dream ends at its truth, in its own cheap cluster.
        1: [Dream(np.ones(4), b, 1.0, 20.0) for b in near],
    }
    rng = np.random.default_rng(0)
    for uncertainty_weight, best in [(0.0, 0), (10.0, 1)]:
        gatherer, source = _scripted_gatherer(dreams, uncertainty_weight, 1.0)
        gatherer.prepare(prior, None, _ScriptedLearner(), Budget(max_iterations=2), rng)
        mean_cost = (4 * 110.0 + 4 * 20.0) / 8
        assert gatherer.planning_terms == pytest.approx([100.0 / mean_cost, 0.0])
        assert gatherer.uncertainty_terms == pytest.approx([0.0, 1.0])
        assert gatherer.get_commitment() == JointQuery(best, -1.0 if best == 0 else 1.0)
        assert source.plan == ([slow, slow] if best == 0 else near[:2])


def test_joint_sleep_on_the_arm():
    """Tests that joint_sleep commits a candidate and trains under its plan, and
    that cluster_sleep commits nothing and asks a random candidate."""
    domain = make_domain(0)
    rng = np.random.default_rng(0)
    for method in ("joint_sleep", "cluster_sleep"):
        robot = make_dreaming_robot(domain, ONE, ONE, Budget())
        robot = with_query_method(
            robot, domain, method, ONE, ONE, Budget(), dreams_per_query=4
        )
        gatherer = robot.info_gatherer
        assert isinstance(gatherer, JointSleepGatherer)
        learner = robot.learner
        assert isinstance(learner, ModelLearner)
        source = learner.belief_source
        assert isinstance(source, PlannedSource)
        robot.sleep(Budget(max_iterations=3), rng)
        assert len(source.plan) == 3
        assert learner.training_beliefs == source.plan
        candidates = gatherer.get_candidates(domain.initial_belief, rng)
        data = robot.gather_information(domain.real_env, ONE, rng)
        if method == "joint_sleep":
            assert len(gatherer.scores) == 12
            best = candidates[int(np.argmax(gatherer.scores))]
            assert gatherer.get_commitment() == best
            assert data[0].query == best
        else:
            assert gatherer.get_commitment() is None
            assert data[0].query in candidates


def test_planned_source_cycles_its_plan():
    """Tests that the planned source trains under its plan in order, cycling, and
    under the current belief without one."""
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    plan = [IntervalBelief([[-1.0, -0.5]] * 4), IntervalBelief([[-2.0, -0.5]] * 4)]
    source: PlannedSource[np.ndarray, object] = PlannedSource()
    rng = np.random.default_rng(0)
    assert source.get_training_belief(1, belief, None, rng) is belief
    source.set_draw_state(plan)
    assert [source.get_training_belief(i, belief, None, rng) for i in (1, 2, 3)] == [
        plan[0],
        plan[1],
        plan[0],
    ]
    assert source.get_draw_state() == plan
