"""Tests for experiments/analysis/check_cycle_shape.py."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import check_cycle_shape  # noqa: E402  # pylint: disable=wrong-import-position

ARGS = argparse.Namespace(
    window=2, min_gap=0.15, max_start=0.8, min_belief_drop=0.05, max_query_error=0.2
)


def _eval_rows(approach: str, success: list[float], time: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "approach": approach,
            "seed": 0,
            "cycle": range(len(success)),
            "success": success,
            "planning_wall_time": time,
        }
    )


def _checks(rows: list[dict]) -> dict[tuple[str, str], str]:
    return {(r["approach"], r["check"]): r["pass"] for r in rows}


def test_check_graph() -> None:
    """A learner that rises and plans faster passes; a flat-high one fails."""
    df = pd.concat(
        [
            _eval_rows("scratch", [0.5, 0.5, 0.5, 0.6], [10, 10, 10, 10]),
            _eval_rows("current_belief", [0.5, 0.5, 0.9, 0.9], [10, 10, 4, 4]),
            _eval_rows("dreaming", [0.9, 0.9, 0.9, 0.9], [10, 10, 10, 10]),
        ]
    )
    checks = _checks(check_cycle_shape.check_graph(df, ARGS))
    assert checks[("scratch", "cycle-0 success")] == "PASS"
    assert checks[("current_belief", "success rise")] == "PASS"
    assert checks[("current_belief", "late gap to baselines")] == "PASS"
    assert checks[("current_belief", "planning time, late / scratch")] == "PASS"
    assert checks[("dreaming", "success rise")] == "FAIL"
    assert checks[("dreaming", "planning time, late / early")] == "FAIL"


def test_check_belief() -> None:
    """The error falls but not the std; with queries it doesn't get near zero."""
    df = pd.DataFrame(
        {
            "approach": "scratch",
            "seed": 0,
            "cycle": [-1, 0, 1, 2],
            "error_ratio": [1.0, 0.9, 0.5, 0.1],
            "std_ratio": [1.0, 1.0, 1.0, 1.0],
        }
    )
    no_queries = _checks(check_cycle_shape.check_belief(df, 0, ARGS))
    assert no_queries[("scratch", "belief error_ratio, late / prior")] == "PASS"
    assert no_queries[("scratch", "belief std_ratio, late / prior")] == "FAIL"
    queries = _checks(check_cycle_shape.check_belief(df, 1, ARGS))
    assert queries[("scratch", "belief error, late")] == "FAIL"
