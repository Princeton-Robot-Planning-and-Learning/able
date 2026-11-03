"""Shared configuration for the tests."""

import os

if "PYTEST_XDIST_WORKER" in os.environ:
    # Parallel workers share the CPUs, so each worker, and each experiment it runs
    # in a subprocess, computes with one thread.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    import torch

    torch.set_num_threads(1)
