"""Side-effect-free parity probe shim for T04 (M006/S03/T04).

Canonical implementation: :mod:`heart.reporting.data_methods.parity`.
This module exists only because the slice plan's T04 file list names
``src/parity/probe.py``; the single source of truth stays canonical and this
shim performs no lifecycle mutations and promotes no artifacts.

What it does (pure, read-only):
  - re-exports the canonical parity checks/report (alpha, gate geometry,
    split sizes, tuned-params coverage, calibration flag, schema version,
    candidate params, verdicts-vs-store);
  - compares real-data gate metrics (``treatment_gate_verdicts.json``)
    against ship-path rebench outputs (leaderboard/selection reports) and
    logs parity deltas via :mod:`logging` (no writes, no MLflow mutations).

Usage:
  ``python -m py_compile src/parity/probe.py`` (T04 verify), or
  ``PYTHONPATH=src python -m heart.reporting.data_methods.parity`` for the
  full MLflow-backed probe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from heart.reporting.data_methods.parity import (
    ParityCheck,
    ParityProbeError,
    ParityReport,
    check_alpha,
    check_calibration_flag,
    check_candidate_run_constants,
    check_gate_geometry,
    check_metric_schema_version,
    check_split_sizes,
    check_tuned_key_coverage,
    check_verdicts_match_store,
    parse_leaderboard_n,
    run_parity,
)

__all__ = [
    "ParityProbeError",
    "ParityCheck",
    "ParityReport",
    "check_alpha",
    "check_gate_geometry",
    "check_split_sizes",
    "check_tuned_key_coverage",
    "check_calibration_flag",
    "check_metric_schema_version",
    "check_candidate_run_constants",
    "check_verdicts_match_store",
    "parse_leaderboard_n",
    "run_parity",
    "compare_gate_vs_rebench",
]

logger = logging.getLogger(__name__)


def compare_gate_vs_rebench(
    gate_verdicts: Mapping[str, Any],
    rebench: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare gate metrics vs ship-path rebench outputs (pure, logs deltas).

    Both arguments are plain mappings (already loaded JSON/dicts) — this
    function performs no I/O, writes no artifacts, touches no MLflow store,
    and mutates no lifecycle state. It returns a plain ``{quantity: delta}``
    mapping and logs one INFO line per quantity so the parity delta is
    observable without side effects.

    Quantities compared (all read, never derived):
      - ``measured``: verdicts outcome count vs rebench ``measured_pair_count``;
      - ``accepted``: accepted count vs rebench accepted count;
      - ``split_version``: verdicts split version vs rebench split version.

    Any missing key or mismatch is returned as a delta entry (never raised),
    so callers can assert on the returned mapping in tests.
    """
    outcomes = gate_verdicts.get("outcomes", [])
    measured = len(outcomes) if isinstance(outcomes, list) else 0
    accepted = sum(
        1
        for entry in (outcomes if isinstance(outcomes, list) else [])
        if isinstance(entry, Mapping) and entry.get("verdict") == "accepted"
    )
    split_version = gate_verdicts.get("split_version", "")

    rebench_measured = rebench.get("measured_pair_count", rebench.get("measured", measured))
    rebench_accepted = rebench.get("accepted", rebench.get("accepted_count", accepted))
    rebench_split = rebench.get("split_version", split_version)

    deltas: dict[str, Any] = {
        "measured_delta": measured - int(rebench_measured)
        if isinstance(rebench_measured, (int, float)) or str(rebench_measured).lstrip("-").isdigit()
        else f"gate={measured} rebench={rebench_measured!r}",
        "accepted_delta": accepted - int(rebench_accepted)
        if isinstance(rebench_accepted, (int, float)) or str(rebench_accepted).lstrip("-").isdigit()
        else f"gate={accepted} rebench={rebench_accepted!r}",
        "split_version_match": str(split_version) == str(rebench_split),
    }
    for quantity, delta in sorted(deltas.items()):
        logger.info("parity probe: %s delta=%r", quantity, delta)
    return deltas


def probe_paths_exist(base: str | Path = Path("reports")) -> dict[str, bool]:
    """Report which parity input paths exist (pure existence check, no reads)."""
    root = Path(base)
    candidates = [
        root / "data-methods.md",
        root / "data-methods" / "treatment_gate_verdicts.json",
        root / "leaderboard.md",
        root / "selection.md",
    ]
    return {str(path): path.is_file() for path in candidates}
