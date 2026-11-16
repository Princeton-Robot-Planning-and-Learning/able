"""Shared configuration for the tests."""

import os
import warnings
from typing import Iterator

import pybullet as p
import pytest
from pybullet_helpers.ikfast.load import install_ikfast_if_required
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot

# PyBullet's limit on clients per process; it reuses the lowest free id.
MAX_PYBULLET_CLIENTS = 1024

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


def _connected_pybullet_clients() -> set[int]:
    return {i for i in range(MAX_PYBULLET_CLIENTS) if p.isConnected(physicsClientId=i)}


@pytest.fixture(autouse=True, scope="module")
def _disconnect_leftover_pybullet_clients(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Disconnect, with a warning, the PyBullet clients a test module leaves
    connected once its tests and fixtures are done.

    Each client holds its world, a few hundred MB, until it is disconnected, and a
    worker runs many modules, so leftovers pile up until the CI runner runs out of
    memory and stops responding. Tests should close what they build; this keeps a
    missed one from taking the run down.
    """
    before = _connected_pybullet_clients()
    yield
    leftovers = _connected_pybullet_clients() - before
    for client in leftovers:
        p.disconnect(physicsClientId=client)
    if leftovers:
        warnings.warn(
            f"{request.module.__name__} left {len(leftovers)} PyBullet clients "
            "connected; close the simulators the tests build"
        )
