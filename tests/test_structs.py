"""Tests for structs.py."""

import pytest

from able.structs import Budget, QueryResponse, StepResult, Task, Trajectory


def test_task():
    """Tests for Task."""
    task = Task(initial_state=(0, 0), goal=(1, 2))
    assert task.initial_state == (0, 0)
    assert task.goal == (1, 2)
    with pytest.raises(AttributeError):
        task.goal = (3, 4)  # type: ignore[misc]
    same_fields = Task(initial_state=(0, 0), goal=(1, 2))
    assert task != same_fields
    assert len({task, same_fields}) == 2


def test_trajectory():
    """Tests for Trajectory."""
    traj = Trajectory(states=(0, 1, 3), actions=(1, 2), costs=(0.5, 1.5))
    assert len(traj) == 2
    assert traj.total_cost == 2.0

    empty = Trajectory(states=(0,), actions=(), costs=())
    assert len(empty) == 0
    assert empty.total_cost == 0.0

    with pytest.raises(AssertionError):
        Trajectory(states=(0, 1), actions=(1, 2), costs=(0.5, 1.5))
    with pytest.raises(AssertionError):
        Trajectory(states=(0, 1, 3), actions=(1, 2), costs=(0.5,))


def test_query_response():
    """Tests for QueryResponse."""
    pair = QueryResponse(query="Can you lift your arm?", response=True)
    assert pair.query == "Can you lift your arm?"
    assert pair.response
    with pytest.raises(AttributeError):
        pair.response = False  # type: ignore[misc]


def test_step_result():
    """Tests for StepResult."""
    result = StepResult(state=3, cost=1.0, terminated=False)
    assert result.state == 3
    assert not result.terminated


def test_budget():
    """Tests for Budget."""
    unlimited = Budget()
    assert not unlimited.is_exhausted(elapsed_time=1e9, num_iterations=10**9)

    timed = Budget(max_wall_time=1.0)
    assert not timed.is_exhausted(elapsed_time=0.5, num_iterations=10**9)
    assert timed.is_exhausted(elapsed_time=1.0, num_iterations=0)

    counted = Budget(max_iterations=3)
    assert not counted.is_exhausted(elapsed_time=1e9, num_iterations=2)
    assert counted.is_exhausted(elapsed_time=0.0, num_iterations=3)

    both = Budget(max_wall_time=1.0, max_iterations=3)
    assert both.is_exhausted(elapsed_time=2.0, num_iterations=0)
    assert both.is_exhausted(elapsed_time=0.0, num_iterations=5)
    assert not both.is_exhausted(elapsed_time=0.5, num_iterations=2)
