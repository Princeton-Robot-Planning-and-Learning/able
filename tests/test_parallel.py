"""Tests for parallel.py and the parallel paths of the cycle and the model
learner."""

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from able.approaches import get_cache_key_features
from able.belief_updaters.joint_limits import JointLimitBeliefUpdater
from able.cycle import CycleBudgets, run_cycles
from able.envs.base import FiniteTaskDistribution
from able.envs.two_link_arm import TwoLinkArmModel, make_domain
from able.info_gatherers.base import NullInformationGatherer
from able.learners.belief_sources import BeliefCoverageSource, ImaginedNextBeliefSource
from able.learners.model_learner import ModelLearner
from able.models.exact_match_cache import ExactMatchCache
from able.parallel import (
    ProcessPool,
    SerialPool,
    WorkerContext,
    WorkerPool,
    build_context,
)
from able.planners.random_shooting import RandomShootingParams, RandomShootingPlanner
from able.robot import Robot
from able.structs import Budget
from tests.experiments.test_run_single_experiment import SCRIPT

NUM_CANDIDATES = 2


def _make_arm_robot() -> tuple:
    """A two-link arm robot that caches plans for dreamed beliefs, chosen by
    belief coverage, and refits once per sleep, so that its sleeps run in
    parallel."""
    domain = make_domain(seed=0, num_goals=4)
    env_model = domain.env_model
    assert isinstance(env_model, TwoLinkArmModel)
    cache = ExactMatchCache(max_per_key=4)
    planner = RandomShootingPlanner(
        env_model, RandomShootingParams(num_proposals=4, model=cache)
    )
    belief_updater = JointLimitBeliefUpdater()
    gatherer: NullInformationGatherer = NullInformationGatherer()
    dreamer: ImaginedNextBeliefSource = ImaginedNextBeliefSource(
        env_model,
        domain.task_distribution,
        planner,
        belief_updater,
        gatherer,
        gathering_budget=Budget(max_iterations=0),
        execution_budget=Budget(max_iterations=1),
        planning_budget=Budget(),
    )
    source: BeliefCoverageSource = BeliefCoverageSource(
        dreamer, get_cache_key_features(cache.key_resolution), NUM_CANDIDATES
    )
    learner: ModelLearner = ModelLearner(
        planner, domain.task_distribution, source, Budget(), refit="per_sleep"
    )
    robot = Robot(
        planner=planner,
        learner=learner,
        belief_updater=belief_updater,
        info_gatherer=gatherer,
        initial_belief=domain.initial_belief,
    )
    return domain, robot


def _make_arm_context() -> WorkerContext:
    """A worker's copy of the arm robot's parts."""
    domain, robot = _make_arm_robot()
    return WorkerContext(domain=domain, planner=robot.planner, learner=robot.learner)


def _run_arm(pool_kind: str) -> tuple:
    """Three cycles of the arm robot with a serial pool of its own parts or two
    worker processes, and its learner's data."""
    domain, robot = _make_arm_robot()
    pool: WorkerPool
    if pool_kind == "serial":
        pool = SerialPool(
            WorkerContext(domain=domain, planner=robot.planner, learner=robot.learner)
        )
    else:
        pool = ProcessPool(2, _make_arm_context)
    budgets = CycleBudgets(
        sleep=Budget(max_iterations=3),
        information_gathering=Budget(max_iterations=0),
        execution=Budget(max_iterations=1),
        planning=Budget(),
    )
    task_distribution = domain.task_distribution
    assert isinstance(task_distribution, FiniteTaskDistribution)
    eval_tasks = list(task_distribution.get_tasks())  # pylint: disable=no-member
    try:
        results = run_cycles(
            robot,
            domain.real_env,
            domain.task_distribution,
            eval_tasks,
            num_cycles=3,
            budgets=budgets,
            rng=np.random.default_rng(0),
            pool=pool,
        )
    finally:
        pool.close()
    learner = robot.learner
    assert isinstance(learner, ModelLearner)
    return results, learner.dataset


def test_worker_processes_match_the_serial_pool():
    """Tests that worker processes give exactly the results of the same parallel
    code run in this process: every evaluation and every training row."""
    serial_results, serial_data = _run_arm("serial")
    process_results, process_data = _run_arm("process")
    assert len(serial_data) > 0
    assert np.array_equal(serial_data.x, process_data.x)
    assert np.array_equal(serial_data.y, process_data.y)
    for serial, process in zip(serial_results, process_results, strict=True):
        assert len(serial.evaluations) == 4
        for s, p in zip(serial.evaluations, process.evaluations, strict=True):
            assert s.success == p.success
            assert s.total_cost == p.total_cost
            assert s.planning_stats == p.planning_stats
        assert serial.belief.get_total_width() == process.belief.get_total_width()
    # Plans for dreamed beliefs warm-start later evaluations.
    used = [e.planning_stats["used_warm_start"] for e in serial_results[-1].evaluations]
    assert any(used)


def test_parallel_sleep_chooses_like_a_serial_sleep():
    """Tests that a parallel sleep chooses each training belief from its own
    candidates, in order, as a serial sleep does."""
    domain, robot = _make_arm_robot()
    learner = robot.learner
    assert isinstance(learner, ModelLearner)
    source = learner.belief_source
    assert source.num_draws == NUM_CANDIDATES
    pool = SerialPool(
        WorkerContext(domain=domain, planner=robot.planner, learner=learner)
    )
    params = learner.learn(
        robot.belief,
        robot.planner_params,
        Budget(max_iterations=3),
        np.random.default_rng(0),
        pool,
    )
    # Every goal was planned under each of the 3 training beliefs.
    assert len(learner.dataset) == 3 * 4
    assert params is not robot.planner_params
    # A wall-time budget cannot be split up front, so the sleep runs serially.
    learner.learn(
        robot.belief,
        params,
        Budget(max_iterations=1, max_wall_time=60.0),
        np.random.default_rng(1),
        pool,
    )
    assert len(learner.dataset) == 4 * 4


def test_a_pool_cannot_be_used_with_a_recorder():
    """Tests that run_cycles refuses a pool and a recorder together."""
    domain, robot = _make_arm_robot()
    pool = SerialPool(
        WorkerContext(domain=domain, planner=robot.planner, learner=robot.learner)
    )
    budgets = CycleBudgets(Budget(), Budget(), Budget(max_iterations=1), Budget())
    with pytest.raises(AssertionError, match="recorder"):
        run_cycles(
            robot,
            domain.real_env,
            domain.task_distribution,
            [],
            num_cycles=1,
            budgets=budgets,
            rng=np.random.default_rng(0),
            recorder=object(),  # type: ignore[arg-type]
            pool=pool,
        )


def test_build_context():
    """Tests that a worker's context is built from a run's configuration, as the
    run builds its robot."""
    with initialize_config_dir(
        config_dir=str(SCRIPT.parent / "conf"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=["approach=current_belief"])
    # The parts a worker uses; the run passes its whole resolved configuration.
    config = {
        "seed": cfg.seed,
        "env": OmegaConf.to_container(cfg.env, resolve=True),
        "approach": OmegaConf.to_container(cfg.approach, resolve=True),
    }
    context = build_context(config)
    assert isinstance(context.planner, RandomShootingPlanner)
    assert isinstance(context.learner, ModelLearner)
    assert len(list(context.domain.task_distribution.get_tasks())) == 12
