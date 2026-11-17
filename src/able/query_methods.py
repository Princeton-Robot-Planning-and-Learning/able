"""Give a robot a rule for choosing its queries, for robots and their dreams."""

from typing import Any, Sequence

from able.approaches import get_cache_key_features
from able.envs.simulated import SimulatedEnvironment
from able.envs.two_link_arm import TwoLinkArmModel
from able.info_gatherers.base import NullInformationGatherer, SequentialGatherer
from able.info_gatherers.bisect_limits import BisectLimitGatherer
from able.info_gatherers.selection import (
    OracleQueryGatherer,
    PlanningAwareGatherer,
    RandomQueryGatherer,
)
from able.learners.belief_features import make_moment_features
from able.learners.belief_sources import ImaginedNextBeliefSource
from able.models.exact_match_cache import ExactMatchCache
from able.robot import Robot
from able.structs import Budget

QUERY_METHODS = ("default", "base", "random", "joint", "sleep_blind", "oracle")


def with_query_method(
    robot: Robot[Any, Any, Any, Any, Any, Any, Any],
    domain: Any,
    method: str = "default",
    gathering_budget: Budget | None = None,
    execution_budget: Budget | None = None,
    planning_budget: Budget | None = None,
    eval_tasks: Sequence[Any] = (),
    num_eval_dreams: int = 32,
    tolerance: float = 0.1,
) -> Robot[Any, Any, Any, Any, Any, Any, Any]:
    """The robot, asking queries by the method, in reality and in its dreams.

    "default" leaves the robot as built. The others start from the robot's own
    gatherer, or bisection of the joint limits on the arm, and pick the first
    query of each gathering as follows: "base" as that gatherer does; "random"
    uniformly among its candidates; "joint" by PlanningAwareGatherer;
    "sleep_blind" likewise without tonight's training; "oracle" by
    OracleQueryGatherer. On the arm, beliefs match when their plan cache keys do;
    elsewhere when their moment features are within tolerance.
    """
    assert method in QUERY_METHODS, f"method must be one of {QUERY_METHODS}"
    if method == "default":
        return robot
    assert gathering_budget is not None and execution_budget is not None
    assert planning_budget is not None
    env_model = domain.env_model
    base = robot.info_gatherer
    if isinstance(base, NullInformationGatherer) and isinstance(
        env_model, TwoLinkArmModel
    ):
        base = BisectLimitGatherer(robot.belief_updater)  # type: ignore[arg-type]
    assert isinstance(base, SequentialGatherer), "Need a gatherer with candidates"
    gatherer: SequentialGatherer[Any, Any, Any] = base
    if method == "random":
        gatherer = RandomQueryGatherer(base)
    elif method in ("joint", "sleep_blind"):
        dreamer: ImaginedNextBeliefSource[Any, Any] = ImaginedNextBeliefSource(
            env_model,
            domain.task_distribution,
            robot.planner,
            robot.belief_updater,
            base,
            gathering_budget,
            execution_budget,
            planning_budget,
        )
        model = getattr(robot.planner.get_default_params(), "model", None)
        if isinstance(model, ExactMatchCache):
            get_features = get_cache_key_features(model.key_resolution)
            tolerance = 0.0
        else:
            get_features = make_moment_features(domain.initial_belief)
        gatherer = PlanningAwareGatherer(
            base,
            dreamer,
            get_features,
            tolerance=tolerance,
            num_eval_dreams=num_eval_dreams,
            sleep_aware=method == "joint",
        )
    elif method == "oracle":
        real_env = domain.real_env
        assert isinstance(real_env, SimulatedEnvironment), "Need a simulated real env"
        gatherer = OracleQueryGatherer(
            base,
            robot.planner,
            env_model,
            domain.task_distribution,
            real_env.hidden_params,
            eval_tasks,
            execution_budget,
            planning_budget,
        )
    robot.set_info_gatherer(gatherer)
    source = getattr(robot.learner, "belief_source", None)
    if source is not None:
        source.set_info_gatherer(gatherer)
    return robot
