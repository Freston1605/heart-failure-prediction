#!/usr/bin/env python3
"""Preflight and launch the MLflow UI against the local tracking store.

The portfolio's tracking backend is deliberately server-less (SQLite plus a
local artifact root, see ``heart.tracking.mlflow_store``), so "viewing the
runs" means launching MLflow's own UI pointed at that store. This script is
that one documented entry point:

* ``--status`` — report whether the store exists and how many runs it holds
  (final battery runs versus all runs), then exit 0.
* ``--launch`` — preflight the store first, then exec ``mlflow ui`` bound to
  the SQLite store on ``MLFLOW_UI_PORT`` (default 5000).

Exit codes (machine-readable for `make` / CI):

* ``0`` — the requested action completed (status printed, or UI exited
  normally);
* ``2`` — the tracking store is missing or has no runs: the UI would open on
  an empty server, which is exactly the failure this script exists to name;
* ``10`` — unexpected internal error (bug, not data).

The preflight is deliberately strict about *zero runs*: an MLflow server that
opens on an empty experiment is the reported bug ("the server doesn't show the
logged training runs"). Rather than opening a blank UI, the launcher says so
and names the command that populates the store.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HEART_EXIT = {
    "ok": 0,
    "empty_store": 2,
    "unexpected": 10,
}

#: Default port for the MLflow UI (MLflow's own default).
DEFAULT_UI_PORT: int = 5000

#: Fallback experiment name if ``heart`` is not importable in this environment.
_FALLBACK_EXPERIMENT: str = "heart-failure-prediction"

#: Environment variable overriding the UI port.
PORT_ENV_VAR: str = "MLFLOW_UI_PORT"


def _experiment_name() -> str:
    """The portfolio experiment name; falls back when ``heart`` is absent."""
    try:
        from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT

        return DEFAULT_EXPERIMENT
    except ImportError:  # pragma: no cover - env, not logic
        return _FALLBACK_EXPERIMENT


@dataclass(frozen=True)
class StoreStatus:
    """The result of preflighting the local tracking store."""

    store_exists: bool
    db_path: Path
    experiment_found: bool
    n_runs_total: int
    n_runs_final: int
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the store exists, the experiment is found, and runs exist."""
        return self.store_exists and self.experiment_found and self.n_runs_total > 0

    def status_line(self) -> str:
        if not self.store_exists:
            return f"tracking store missing: {self.db_path} does not exist"
        if not self.experiment_found:
            return (
                f"tracking store exists at {self.db_path} but has no "
                f"'{_experiment_name()}' experiment yet"
            )
        if self.n_runs_total == 0:
            return (
                f"tracking store exists at {self.db_path} but contains "
                "zero logged runs"
            )
        return (
            f"tracking store ok: {self.db_path} — "
            f"{self.n_runs_total} run(s), {self.n_runs_final} final"
        )

    def fix_hint(self) -> str:
        return (
            "No logged training runs are available to show. Populate the "
            "store with: make reproduce   (or directly: PYTHONPATH=src "
            "python -m heart.models.run_battery)"
        )


def preflight(tracking_dir: str | Path | None = None) -> StoreStatus:
    """Check the local store and count its runs without mutating anything.

    A missing store, a missing experiment, and a zero-run experiment are all
    *reported states*, not exceptions — the caller decides how to render them.
    """
    from heart.tracking.mlflow_store import (
        DEFAULT_EXPERIMENT,
        TRACKING_DB_FILENAME,
        default_tracking_dir,
    )

    directory = (
        Path(tracking_dir) if tracking_dir is not None else default_tracking_dir()
    )
    db_path = directory / TRACKING_DB_FILENAME
    if not db_path.exists():
        return StoreStatus(
            store_exists=False,
            db_path=db_path,
            experiment_found=False,
            n_runs_total=0,
            n_runs_final=0,
            problems=(f"no SQLite store at {db_path}",),
        )

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - env, not logic
        return StoreStatus(
            store_exists=True,
            db_path=db_path,
            experiment_found=False,
            n_runs_total=0,
            n_runs_final=0,
            problems=(f"MLflow is not importable: {exc}",),
        )

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    if experiment is None:
        return StoreStatus(
            store_exists=True,
            db_path=db_path,
            experiment_found=False,
            n_runs_total=0,
            n_runs_final=0,
            problems=(f"no experiment {DEFAULT_EXPERIMENT!r} in the store",),
        )

    n_total = 0
    n_final = 0
    for run in client.search_runs(
        experiment_ids=[experiment.experiment_id], max_results=50000
    ):
        n_total += 1
        if run.data.tags.get("run_kind") == "final":
            n_final += 1

    # Artifact-root existence is advisory: runs render fine without artifacts.
    artifact_root = directory / "artifacts"
    problems: list[str] = []
    if not artifact_root.exists():
        problems.append(f"artifact root missing: {artifact_root}")

    return StoreStatus(
        store_exists=True,
        db_path=db_path,
        experiment_found=True,
        n_runs_total=n_total,
        n_runs_final=n_final,
        problems=tuple(problems),
    )


def resolve_port() -> int:
    """Resolve the UI port from ``MLFLOW_UI_PORT`` or the default."""
    raw = os.environ.get(PORT_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_UI_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{PORT_ENV_VAR} must be an integer, got {raw!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{PORT_ENV_VAR} must be in 1..65535, got {port}")
    return port


def launch(tracking_dir: str | Path | None = None, *, port: int | None = None) -> int:
    """Preflight, then exec the MLflow UI bound to the local store.

    Only returns a non-zero exit code on preflight failure; on success the
    MLflow UI runs in the foreground until interrupted.
    """
    status = preflight(tracking_dir)
    print(status.status_line())
    if not status.ok:
        print(f"ERROR: {status.fix_hint()}", file=sys.stderr)
        return HEART_EXIT["empty_store"]
    for problem in status.problems:
        print(f"warning: {problem}")

    resolved_port = port if port is not None else resolve_port()
    from heart.tracking.mlflow_store import default_artifact_root, default_tracking_uri

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        default_tracking_uri(),
        "--default-artifact-root",
        default_artifact_root().resolve().as_uri(),
        "--port",
        str(resolved_port),
    ]
    print(f"launching: {' '.join(cmd[2:])}")
    print(f"MLflow UI will be available at http://localhost:{resolved_port}")
    try:
        result = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        return HEART_EXIT["ok"]
    return result.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/mlflow_ui.py",
        description=(
            "Preflight and launch the MLflow UI against the local SQLite "
            "tracking store (experiments/mlruns/mlflow.db)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--status",
        action="store_true",
        help="Report store existence and run counts, then exit.",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="Preflight the store, then launch the MLflow UI (default).",
    )
    parser.add_argument(
        "--tracking-dir",
        default=None,
        help="Override the tracking directory (default: experiments/mlruns).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.status:
        status = preflight(args.tracking_dir)
        print(status.status_line())
        if status.ok:
            for problem in status.problems:
                print(f"warning: {problem}")
        if not status.ok:
            print(status.fix_hint())
            return HEART_EXIT["empty_store"]
        return HEART_EXIT["ok"]
    try:
        return launch(args.tracking_dir)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return HEART_EXIT["unexpected"]


if __name__ == "__main__":
    sys.exit(main())
