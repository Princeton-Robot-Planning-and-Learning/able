"""Tests for recording/base.py."""

import numpy as np

from able.beliefs.interval import IntervalBelief
from able.envs.two_link_arm import JOINT_LIMIT_PRIOR, ArmTask
from able.recording.base import NullRecorder, Stage


def test_null_recorder():
    """Tests that NullRecorder accepts every event."""
    recorder: NullRecorder = NullRecorder()
    belief = IntervalBelief(JOINT_LIMIT_PRIOR)
    task = ArmTask(initial_state=np.zeros(2), goal=np.ones(2))
    recorder.start_stage(0, Stage.EXECUTION)
    recorder.start_episode("Executed task 1", task, belief)
    recorder.record_step(np.ones(2), belief)
    recorder.end_episode("success")
    recorder.record_queries([], belief)
    recorder.record_belief("Belief", belief, hypothesis=np.zeros(4))
    recorder.end_stage()
    assert [stage.value for stage in Stage] == [
        "sleep",
        "queries",
        "execution",
        "evaluation",
    ]
