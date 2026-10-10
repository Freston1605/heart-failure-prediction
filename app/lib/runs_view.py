"""Runs-view contract for the dashboard's Runs page (M002/S02/T01).

This module owns *what the Runs page shows* so the page itself stays thin
and testable, exactly mirroring the :mod:`app.lib.plots` /
:mod:`app.lib.artifact_loader` pattern: every load attempt returns a frozen
value object carrying either data or a *named* error kind plus a plain
language message — never an exception.

D019 data source (frozen decision)
----------------------------------
The page reuses :func:`heart.reporting.leaderboard.select_leaderboard_rows`
and :mod:`heart.tracking.mlflow_store` directly. There is deliberately **no
page-local MLflow query layer**: the ranked table the dashboard renders is
produced by the same selection function that renders
``reports/leaderboard.md``, so the two viewers show identical numbers by
construction. Re-deriving runs through a second query path would let the
report and the dashboard drift apart. No new logging is performed here
either (D011/D012 frozen) — this module is read-only over the store.

Never-raise boundary
--------------------
:func:`load_runs_view` and :func:`run_details` never raise. A missing or
empty tracking store, an unreadable directory, an unknown run id, or any
MLflow internal failure is mapped to a named ``error_kind``:

* ``store-missing`` — the store directory is absent or unreadable;
* ``store-empty`` — the store exists but has no recorded final runs for
  the default experiment (or the experiment itself is absent);
* ``runs-unknown`` — any other selection failure (data errors, config
  errors, MLflow internals);
* ``drilldown-unknown`` — the per-run read-back could not be completed.

Every message is plain language: what is missing plus a concrete next
action naming ``make reproduce`` — the same populate command the
``scripts/mlflow_ui.py`` launcher hints at on exit 2 and the README's
"Viewing the training runs" section documents, so all three viewers
report the *same* empty state.

The ``RUNS_TRACKING_DIR`` environment variable is the testability seam
(mirroring ``EXPLORE_DATA_PATH`` in :mod:`app.lib.plots`): AppTest render
tests point the page at a tmp store without touching the real one.

This module is plain Python — no ``streamlit`` imports — so all logic is
pytest-testable without a Streamlit runtime (MEM047).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from heart.reporting.leaderboard import (
    LeaderboardConfig,
    LeaderboardExperimentNotFoundError,
    LeaderboardRow,
    NoBenchmarkedModelsError,
    select_leaderboard_rows,
)
from heart.tracking.mlflow_store import (
    default_tracking_dir,
    resolve_tracking_uri,
)

__all__ = [
    "RUNS_TRACKING_DIR_ENV",
    "RunsView",
    "RunDetails",
    "load_runs_view",
    "run_details",
]

#: Environment override for the tracking store directory (testability seam).
RUNS_TRACKING_DIR_ENV: str = "RUNS_TRACKING_DIR"

#: Named error kinds surfaced by this module's contract.
ERROR_KIND_STORE_MISSING: str = "store-missing"
ERROR_KIND_STORE_EMPTY: str = "store-empty"
ERROR_KIND_RUNS_UNKNOWN: str = "runs-unknown"
ERROR_KIND_DRILLDOWN_UNKNOWN: str = "drilldown-unknown"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunsView:
    """Result of one leaderboard-load attempt; never raises.

    Attributes:
        rows: Ranked final-run rows in ROC-AUC order (empty on failure).
        n_runs_considered: Finished final runs matched before dedup.
        tracking_uri: The resolved ``sqlite:///`` tracking URI attempted.
        error_kind: Named error bucket (``None`` on success).
        error_message: Friendly, plain-language failure message.
        ok: Derived — rows present and no error kind recorded.
    """

    rows: tuple[LeaderboardRow, ...]
    n_runs_considered: int
    tracking_uri: str
    error_kind: str | None
    error_message: str | None

    @property
    def ok(self) -> bool:
        """True when ranked rows are available and no error was recorded."""
        return self.error_kind is None and bool(self.rows)


@dataclass(frozen=True)
class RunDetails:
    """Per-run drilldown read back from the store; never raises.

    Attributes:
        run_id: The MLflow run id requested.
        run_name: The run's recorded name (``mlflow.runName``).
        params: Hyperparameters logged on the run.
        tags: Tags logged on the run (including MLflow system tags).
        metrics: Every flattened scalar metric logged on the run.
        artifact_names: Artifact file names attached to the run.
        error_kind: Named error bucket (``None`` on success).
        error_message: Friendly, plain-language failure message.
        ok: Derived — no error kind recorded.
    """

    run_id: str
    run_name: str
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, float]
    artifact_names: tuple[str, ...]
    error_kind: str | None
    error_message: str | None

    @property
    def ok(self) -> bool:
        """True when the run was read back with no error recorded."""
        return self.error_kind is None


# ---------------------------------------------------------------------------
# Resolution + message mapping
# ---------------------------------------------------------------------------


def _resolve_tracking_dir(
    tracking_dir: str | Path | None,
) -> Path:
    """Resolve the store directory: explicit arg, env override, then default.

    The ``RUNS_TRACKING_DIR`` environment variable exists so AppTest render
    tests can point the page at a missing/empty/populated tmp store without
    touching the install — the same seam ``EXPLORE_DATA_PATH`` provides for
    the Explore page.
    """
    if tracking_dir is not None:
        return Path(tracking_dir)
    override = os.environ.get(RUNS_TRACKING_DIR_ENV)
    if override:
        return Path(override)
    return default_tracking_dir()


def _make_reproduce_hint() -> str:
    return (
        "Run `make reproduce` to populate the store with recorded training "
        "runs, then reload this page."
    )


def _store_missing_message(directory: Path) -> str:
    return (
        f"The MLflow tracking store directory `{directory}` does not exist "
        "yet, so there are no recorded training runs to show. "
        + _make_reproduce_hint()
    )


def _store_unreadable_message(directory: Path, detail: str) -> str:
    return (
        f"The MLflow tracking store directory `{directory}` exists but could "
        f"not be read ({detail}), so the recorded runs are unavailable. Check "
        "the directory's permissions under experiments/. "
        + _make_reproduce_hint()
    )


def _store_empty_message(experiment: str, uri: str) -> str:
    return (
        f"The tracking store at `{uri}` exists but has no recorded final "
        f"runs for the experiment `{experiment}` yet. "
        + _make_reproduce_hint()
    )


def _runs_unknown_message(detail: str) -> str:
    return (
        "An unexpected problem occurred while reading the recorded training "
        "runs from the MLflow store. The dashboard refuses to display "
        f"possibly-wrong figures. Detail: {detail}. " + _make_reproduce_hint()
    )


def _drilldown_unknown_message(run_id: str, detail: str) -> str:
    return (
        f"The run `{run_id}` could not be read back from the MLflow store "
        f"(it may have been deleted, or the store moved). Detail: {detail}. "
        + _make_reproduce_hint()
    )


# ---------------------------------------------------------------------------
# Contract functions
# ---------------------------------------------------------------------------


def _load_attempt(tracking_dir: Path) -> RunsView:
    """One leaderboard-load attempt; maps every exception to the contract.

    ``select_leaderboard_rows`` raises named leaderboard errors on data
    problems; this boundary swallows everything (BLE001) because the
    dashboard boundary must never leak a stack trace — any unpredicted
    failure lands in ``runs-unknown`` rather than white-screening the page.
    """
    resolved_uri = resolve_tracking_uri(tracking_dir=tracking_dir)

    if not tracking_dir.exists():
        return RunsView((), 0, resolved_uri, ERROR_KIND_STORE_MISSING,
                        _store_missing_message(tracking_dir))
    if not tracking_dir.is_dir():
        return RunsView((), 0, resolved_uri, ERROR_KIND_STORE_MISSING,
                        _store_unreadable_message(tracking_dir, "not a directory"))

    try:
        rows, n_considered = select_leaderboard_rows(
            config=LeaderboardConfig(), tracking_dir=tracking_dir
        )
    except LeaderboardExperimentNotFoundError:
        return RunsView((), 0, resolved_uri, ERROR_KIND_STORE_EMPTY,
                        _store_empty_message(LeaderboardConfig().experiment_name,
                                             resolved_uri))
    except NoBenchmarkedModelsError:
        return RunsView((), 0, resolved_uri, ERROR_KIND_STORE_EMPTY,
                        _store_empty_message(LeaderboardConfig().experiment_name,
                                             resolved_uri))
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return RunsView((), 0, resolved_uri, ERROR_KIND_STORE_MISSING,
                        _store_unreadable_message(tracking_dir, str(exc)[:200]))
    except Exception as exc:  # noqa: BLE001 - the dashboard boundary swallows all
        return RunsView((), 0, resolved_uri, ERROR_KIND_RUNS_UNKNOWN,
                        _runs_unknown_message(f"{type(exc).__name__}: {str(exc)[:200]}"))

    if not rows and n_considered == 0:
        return RunsView((), 0, resolved_uri, ERROR_KIND_STORE_EMPTY,
                        _store_empty_message(LeaderboardConfig().experiment_name,
                                             resolved_uri))

    return RunsView(tuple(rows), int(n_considered), resolved_uri, None, None)


def load_runs_view(*, tracking_dir: str | Path | None = None) -> RunsView:
    """Load the ranked final runs for the Runs page, or the friendly error.

    Never raises. The store directory is resolved from the explicit
    ``tracking_dir`` argument, else the ``RUNS_TRACKING_DIR`` environment
    override, else the project default
    (``heart.tracking.mlflow_store.default_tracking_dir``).

    On success ``rows`` holds the leaderboard rows produced by
    :func:`heart.reporting.leaderboard.select_leaderboard_rows` — the same
    ranked selection that renders ``reports/leaderboard.md``, so the two
    viewers agree by construction (D019). On failure ``ok`` is ``False``
    with a named ``error_kind`` and a plain-language message naming
    ``make reproduce``.
    """
    try:
        return _load_attempt(_resolve_tracking_dir(tracking_dir))
    except Exception as exc:  # noqa: BLE001 - even resolution must not raise
        directory = _resolve_tracking_dir(tracking_dir)
        return RunsView((), 0, str(directory), ERROR_KIND_RUNS_UNKNOWN,
                        _runs_unknown_message(f"{type(exc).__name__}: {str(exc)[:200]}"))


def run_details(
    run_id: str, *, tracking_dir: str | Path | None = None
) -> RunDetails:
    """Read one run's params, tags, metrics and artifacts; never raises.

    The tracking URI is configured explicitly on the client — MLflow 3.x
    silently defaults to ``./mlflow.db`` without an explicit URI, so a page
    that skipped this would drill into whatever database the working
    directory happens to hold. Resolution honours the same seam as
    :func:`load_runs_view` (explicit argument, then ``RUNS_TRACKING_DIR``,
    then the project default) so a page pointing the loader at a tmp store
    drills into the *same* store. Any failure (unknown run id, store error)
    maps to ``drilldown-unknown`` with a plain-language message; ``ok``
    stays ``False``.
    """
    try:
        resolved_uri = resolve_tracking_uri(
            tracking_dir=_resolve_tracking_dir(tracking_dir)
        )
        client = MlflowClient(tracking_uri=resolved_uri)
        run = client.get_run(run_id)
        data = run.data
        artifacts = tuple(
            sorted(info.path for info in client.list_artifacts(run_id))
        )
        return RunDetails(
            run_id=str(run_id),
            run_name=str(data.tags.get("mlflow.runName", "")),
            params={str(k): str(v) for k, v in dict(data.params).items()},
            tags={str(k): str(v) for k, v in dict(data.tags).items()},
            metrics={str(k): float(v) for k, v in dict(data.metrics).items()},
            artifact_names=artifacts,
            error_kind=None,
            error_message=None,
        )
    except Exception as exc:  # noqa: BLE001 - the dashboard boundary swallows all
        kind = type(exc).__name__
        if isinstance(exc, MlflowException):
            detail = str(exc).splitlines()[0][:200]
        else:
            detail = f"{kind}: {str(exc)[:200]}"
        safe_id = str(run_id)[:64]
        return RunDetails(
            run_id=safe_id,
            run_name="",
            params={},
            tags={},
            metrics={},
            artifact_names=(),
            error_kind=ERROR_KIND_DRILLDOWN_UNKNOWN,
            error_message=_drilldown_unknown_message(safe_id, detail),
        )
