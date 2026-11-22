"""Tests for experiments/analysis/fit_site_allocation.py."""

import importlib.util
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).parents[2] / "experiments/analysis/fit_site_allocation.py"
    spec = importlib.util.spec_from_file_location("fit_site_allocation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_transfer_recovers_the_rate():
    """A curve made by a known transfer through the ramp fits back to it."""
    module = _load()
    units = np.array(module.DEPTHS, dtype=np.float64)
    curve = module.get_ramp(units * 0.5, 1.0, 3.0)
    assert module.fit_transfer(curve, 1.0, 3.0) == 0.5
