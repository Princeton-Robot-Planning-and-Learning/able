"""Shared configuration for the tests."""

import os

import pytest
from pybullet_helpers.ikfast.load import install_ikfast_if_required
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot

if "PYTEST_XDIST_WORKER" in os.environ:
    # Parallel workers share the CPUs, so each worker, and each experiment it runs
    # in a subprocess, computes with one thread.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    import torch

    torch.set_num_threads(1)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(config: pytest.Config, specs: list) -> None:
    """Build the IKFast solver of the limb and friction environments' Kinova arm
    before parallel workers start.

    pybullet_helpers builds it on first use, which takes minutes after a fresh
    install, and workers that build it at the same time can import a half-written
    module.
    """
    del config, specs  # unused because the build does not depend on them
    ikfast_info = KinovaGen3RobotiqGripperPyBulletRobot.ikfast_info()
    assert ikfast_info is not None
    install_ikfast_if_required(ikfast_info)
