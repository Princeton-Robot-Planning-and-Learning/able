"""Tests for info_gatherers/random_queries.py and query_methods.py."""

import numpy as np
import pytest

from able.approaches import make_dreaming_robot
from able.beliefs.interval import IntervalBelief
from able.beliefs.range_of_motion import RangeOfMotionBelief
from able.envs.friction.layout import make_world
from able.envs.friction.tasks import make_domain as make_friction_domain
from able.envs.limb.human import RangeOfMotionPrior, get_variant
from able.envs.two_link_arm import make_domain as make_arm_domain
from able.info_gatherers.random_queries import (
    FiniteQueryDistribution,
    FrictionQueries,
    InfiniteQueryDistribution,
    LimbQueries,
)
from able.query_methods import with_query_method
from able.structs import Budget


@pytest.mark.parametrize("distribution", ["finite", "infinite"])
def test_random_queries_on_the_arm(distribution):
    """Tests that a random arm robot asks the budget's queries, each inside the
    belief's interval for its limit, and stays consistent with the truth."""
    true_limits = np.array([-1.0, 1.5, -2.0, 1.0])
    domain = make_arm_domain(seed=0, true_limits=true_limits)
    budget = Budget(max_iterations=1)
    robot = make_dreaming_robot(domain, budget, budget, Budget())
    robot = with_query_method(robot, domain, "random", distribution)
    rng = np.random.default_rng(0)
    for _ in range(5):
        belief = robot.belief
        assert isinstance(belief, IntervalBelief)
        data = robot.gather_information(domain.real_env, budget, rng)
        assert len(data) == 1
        query = data[0].query
        rows = belief.intervals[2 * query.joint : 2 * query.joint + 2]
        assert np.any((rows[:, 0] <= query.angle) & (query.angle <= rows[:, 1]))
        robot.update_belief_from_queries(data)
        assert robot.belief.contains(true_limits)
    assert robot.belief != domain.initial_belief


def test_limb_and_friction_queries_stay_in_range():
    """Tests that limb targets lie between the scaled lower and upper bounds and
    friction probes run no faster than the floor's probe."""
    rng = np.random.default_rng(0)
    variant = get_variant("human-left-arm")
    mean, stddev, low, high = RangeOfMotionPrior().get_parameters(3).T
    belief = RangeOfMotionBelief.from_prior(mean, stddev, low, high, 3)
    space = LimbQueries(variant.default_magnitudes)
    lo = belief.lower * variant.default_magnitudes
    hi = belief.high * variant.default_magnitudes
    for dist in (FiniteQueryDistribution(space), InfiniteQueryDistribution(space)):
        for _ in range(20):
            query = dist.sample(belief, rng)
            assert lo[query.motion] <= query.target <= hi[query.motion]

    world = make_world()
    friction_belief = make_friction_domain(seed=0, num_particles=100).initial_belief
    friction_space = FrictionQueries(world)
    probes = friction_space._probes(friction_belief)  # pylint: disable=protected-access
    assert len(friction_space.get_queries(friction_belief)) == 3 * len(probes)
    for _ in range(20):
        query = friction_space.sample(friction_belief, rng)
        probe = probes[query.floor]
        assert query.puck == probe.puck
        assert world.physics.min_speed <= query.speed <= probe.speed
