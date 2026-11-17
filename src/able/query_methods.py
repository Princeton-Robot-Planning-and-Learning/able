"""Replace a robot's query rule, in reality and in its dreams."""

from typing import Any

from able.envs.friction.model import FrictionModel
from able.envs.human_reach.model import ReachModel
from able.envs.limb.model import LimbModel
from able.envs.two_link_arm import TwoLinkArmModel
from able.info_gatherers.random_queries import (
    ArmQueries,
    FiniteQueryDistribution,
    FrictionQueries,
    InfiniteQueryDistribution,
    LimbQueries,
    QueryDistribution,
    QuerySpace,
    RandomGatherer,
    ReachQueries,
)
from able.robot import Robot

QUERY_METHODS = ("default", "random")
QUERY_DISTRIBUTIONS = ("finite", "infinite")


def make_query_space(env_model: Any) -> QuerySpace[Any]:
    """The environment's queries for random gathering."""
    if isinstance(env_model, TwoLinkArmModel):
        return ArmQueries()
    if isinstance(env_model, LimbModel):
        return LimbQueries(env_model.variant.default_magnitudes)
    if isinstance(env_model, ReachModel):
        return ReachQueries(env_model.config)
    if isinstance(env_model, FrictionModel):
        return FrictionQueries(env_model.world)
    raise NotImplementedError(f"No query space for {type(env_model).__name__}")


def with_query_method(
    robot: Robot[Any, Any, Any, Any, Any, Any, Any],
    domain: Any,
    method: str = "default",
    distribution: str = "finite",
) -> Robot[Any, Any, Any, Any, Any, Any, Any]:
    """The robot as built ("default"), or asking random queries ("random") from the
    finite or infinite distribution, in reality and in its dreams."""
    assert method in QUERY_METHODS, f"method must be one of {QUERY_METHODS}"
    assert distribution in QUERY_DISTRIBUTIONS, f"Need one of {QUERY_DISTRIBUTIONS}"
    if method == "default":
        return robot
    space = make_query_space(domain.env_model)
    dist: QueryDistribution[Any] = (
        FiniteQueryDistribution(space)
        if distribution == "finite"
        else InfiniteQueryDistribution(space)
    )
    gatherer: RandomGatherer[Any, Any] = RandomGatherer(dist, robot.belief_updater)
    robot.set_info_gatherer(gatherer)
    source = getattr(robot.learner, "belief_source", None)
    if source is not None:
        source.set_info_gatherer(gatherer)
    return robot
