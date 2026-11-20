"""Tests for envs/limb/modes.py."""

import numpy as np
import pytest

from able.beliefs.sites import SiteBelief
from able.envs.limb.human import get_variant
from able.envs.limb.model import LimbModel
from able.envs.limb.modes import (
    MotionExerciseDistribution,
    close_mode_domain,
    get_mode_scales,
    make_mode_domain,
    make_transition,
)
from able.envs.limb.sim import LimbSimSettings
from able.envs.limb.tasks import TaskBox, load_fixed_setup
from able.envs.multi_site import MarkovSiteTaskDistribution


@pytest.fixture(name="domain", scope="module")
def _domain():
    domain = make_mode_domain(seed=0, stiff_motions=((), (2,)), horizon=20)
    yield domain
    close_mode_domain(domain)


def test_mode_scales(domain):
    """Tests that a mode's stiff motions are stiffness less than on a good day
    and its other motions the same."""
    scales = get_mode_scales(domain)
    assert scales.shape == (2, 3)
    np.testing.assert_allclose(scales[0, :2], scales[1, :2])
    assert scales[0, 2] - scales[1, 2] == pytest.approx(0.2)
    assert np.all((scales[0] >= 0.9) & (scales[0] <= 1.1))
    belief = domain.initial_belief
    assert isinstance(belief, SiteBelief) and belief.num_sites == 2
    for mode in range(2):
        mean = belief.beliefs[mode].mean
        assert np.all(np.abs(mean - scales[mode]) < 0.05)


def test_exercises(domain):
    """Tests that each exercise moves its motion within the mode's range, keeps
    the other joints within every mode's range, and that modes share the
    exercises of motions that are equally stiff in them."""
    routines = domain.task_distribution
    assert isinstance(routines, MarkovSiteTaskDistribution)
    variant = get_variant("human-left-arm")
    scales = get_mode_scales(domain)
    good, stiff = routines.get_routine(0), routines.get_routine(1)
    assert [len(good), len(stiff)] == [3, 3]
    assert good[0].inner is stiff[0].inner and good[1].inner is stiff[1].inner
    assert good[2].inner is not stiff[2].inner
    assert routines.get_block_ids(0)[:2] == routines.get_block_ids(1)[:2]
    shared_low, shared_high = variant.get_joint_limits(scales.min(axis=0))
    joints = list(variant.active_joints)
    for mode, routine in enumerate((good, stiff)):
        low, high = variant.get_joint_limits(scales[mode])
        for motion, task in enumerate(routine):
            start = np.array(task.inner.initial_state.coupled.limb_positions)
            goal = task.goal
            assert 0.5 <= np.abs(goal - start)[joints].sum() <= 0.8
            joint = joints[motion]
            assert low[joint] <= goal[joint] <= high[joint] - 0.3
            for other in joints:
                if other != joint:
                    for positions in (start, goal):
                        assert shared_low[other] <= positions[other]
                        assert positions[other] <= shared_high[other]


def test_exercises_that_hold_the_other_joints():
    """Tests that with hold_others, an exercise moves only its motion's joint,
    the task box's separation short of its goal."""
    model = LimbModel(get_variant("human-left-arm"), LimbSimSettings(), horizon=20)
    exercises = MotionExerciseDistribution(
        model,
        load_fixed_setup("human-left-arm"),
        np.full(3, 0.75),
        TaskBox(min_separation=0.3, max_separation=0.5),
        goal_margin=0.3,
        hold_others=True,
    )
    task = exercises.sample_exercise(1, 0.95, np.random.default_rng(0))
    start = np.array(task.initial_state.coupled.limb_positions)
    moved = np.flatnonzero(~np.isclose(start, task.goal))
    joint = model.variant.active_joints[1]
    assert list(moved) == [joint]
    assert 0.3 <= abs(task.goal[joint] - start[joint]) <= 0.5
    exercises.close()
    model.close()


def test_transition():
    """Tests that each mode stays with probability stay and that rows sum to
    one."""
    transition = make_transition(4, 0.6, np.random.default_rng(0))
    np.testing.assert_allclose(np.diag(transition), 0.6)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    assert np.all(transition >= 0.0)
