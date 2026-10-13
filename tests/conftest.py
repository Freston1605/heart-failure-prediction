"""Shared test-session configuration for the heart_failure test suite.

Loaded by pytest before any test module is imported, so the environment
stitching done here happens before numpy/BLAS initialise.

Why the BLAS thread caps: the suite runs under pytest-xdist (``-n auto``).
Each worker process that loads numpy/pandas/sklearn would otherwise spawn one
OpenMP/BLAS thread per core, multiplying workers x cores into hundreds of
competing kernel threads. Measured on this host: full suite ~85-98s with
default threading vs ~65s with single-threaded BLAS at the same worker count,
with far less variance (no memory-thrash). ``setdefault`` keeps the cap
overridable by an explicitly set environment for debugging.
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP threads BEFORE numpy/pandas/sklearn are imported by any
# test module (conftest is imported first by pytest).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Headless rendering: every matplotlib figure in the suite goes to PNG via the
# Agg backend. Setting MPLBACKEND as a fallback guard alongside each module's
# own matplotlib.use("Agg", force=True) guard.
os.environ.setdefault("MPLBACKEND", "Agg")
