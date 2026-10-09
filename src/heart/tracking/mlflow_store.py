"""Local MLflow backend configuration (SQLite + local artifact root, no server).

Every model in the portfolio logs through one MLflow experiment. Standing the
backend up in one place means the storage location, the experiment name, and
the experiment's artifact root are decided once here rather than re-derived by
each model script.

Backend choice
--------------
A **SQLite tracking store plus a local filesystem artifact root**. No MLflow
server, no network, no Docker:

* tracking store  -> ``<experiments>/mlruns/mlflow.db`` (``sqlite:///`` URI)
* artifact root   -> ``<experiments>/mlruns/artifacts`` (``file://`` URI)

SQLite is the durable local backend MLflow recommends for single-machine use;
the run metadata (params, metrics, tags, run names) lives in the DB, while
artifacts (the ``metrics.json`` snapshot) live on disk. Everything is
reproducible from the repository alone and nothing leaves the machine.

The experiment
--------------
:data:`DEFAULT_EXPERIMENT` (``heart-failure-prediction``) is the single
experiment every run belongs to. :func:`configure_tracking` resolves it,
creating it with the declared artifact root on first use, and is idempotent:
calling it twice returns the same ``experiment_id``.

Observability
-------------
:func:`configure_tracking` logs the resolved tracking URI, artifact root, and
experiment id at ``INFO``. :func:`describe_tracking_convention` renders the
active store and naming convention for diagnostics. Backend failures raise
named subclasses of :class:`TrackingError` instead of leaking a bare
``MlflowException``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - exercised by the environment, not by logic
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "heart.tracking requires MLflow, which is not installed. Install the "
        'tracking extra with: pip install -e ".[ml]"'
    ) from exc

from heart.config import EXPERIMENTS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared backend layout and naming
# ---------------------------------------------------------------------------

#: The one experiment every portfolio run is recorded under.
DEFAULT_EXPERIMENT: str = "heart-failure-prediction"

#: Human-readable description attached to the experiment on creation.
EXPERIMENT_DESCRIPTION: str = (
    "Heart-failure prediction portfolio: one comparable run per model per "
    "split version, all scored through heart.eval.evaluate."
)

#: Directory (under ``experiments/``) that holds the SQLite store and artifacts.
TRACKING_DIRNAME: str = "mlruns"

#: SQLite database filename inside the tracking directory.
TRACKING_DB_FILENAME: str = "mlflow.db"

#: Artifact-root sub-directory inside the tracking directory.
ARTIFACT_ROOT_DIRNAME: str = "artifacts"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class TrackingError(Exception):
    """Base class for every tracking-store failure."""


class TrackingConfigError(TrackingError):
    """The requested tracking configuration is invalid."""


class TrackingBackendError(TrackingError):
    """The MLflow backend could not be reached, created, or written to."""


class ExperimentResolutionError(TrackingError):
    """The experiment could not be resolved or created."""


# ---------------------------------------------------------------------------
# Paths and URIs
# ---------------------------------------------------------------------------


def default_tracking_dir() -> Path:
    """Return the default directory holding the MLflow store and artifacts."""
    return Path(EXPERIMENTS_DIR) / TRACKING_DIRNAME


def default_artifact_root() -> Path:
    """Return the default filesystem artifact root for this project."""
    return default_tracking_dir() / ARTIFACT_ROOT_DIRNAME


def sqlite_uri(directory: str | Path) -> str:
    """Build the SQLite tracking URI for ``directory``."""
    return f"sqlite:///{Path(directory) / TRACKING_DB_FILENAME}"


def default_tracking_uri() -> str:
    """Return the default SQLite tracking URI for this project."""
    return sqlite_uri(default_tracking_dir())


# ---------------------------------------------------------------------------
# Resolved configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingConfig:
    """The resolved, active MLflow configuration.

    ``created_experiment`` records whether :func:`configure_tracking` had to
    create the experiment on this call, which makes first-run versus reuse
    observable in tests and logs.
    """

    tracking_uri: str
    artifact_root: Path
    experiment_name: str
    experiment_id: str
    created_experiment: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "tracking_uri": self.tracking_uri,
            "artifact_root": str(self.artifact_root),
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "created_experiment": self.created_experiment,
        }


def _validate_experiment_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise TrackingConfigError(
            f"experiment_name must be a non-empty string, got {name!r}."
        )
    if "/" in name or "\x00" in name:
        raise TrackingConfigError(
            f"experiment_name may not contain '/' or a NUL byte, got {name!r}."
        )
    return name.strip()


def resolve_tracking_uri(
    *, tracking_uri: str | None = None, tracking_dir: str | Path | None = None
) -> str:
    """Resolve the tracking URI from an explicit URI or a directory.

    An explicit ``tracking_uri`` wins; otherwise the SQLite database is placed
    inside ``tracking_dir`` (or the project default).
    """
    if tracking_uri is not None:
        if not isinstance(tracking_uri, str) or not tracking_uri.strip():
            raise TrackingConfigError(
                f"tracking_uri must be a non-empty string, got {tracking_uri!r}."
            )
        return tracking_uri
    directory = Path(tracking_dir) if tracking_dir is not None else default_tracking_dir()
    return sqlite_uri(directory)


def resolve_artifact_root(
    *,
    artifact_root: str | Path | None = None,
    tracking_dir: str | Path | None = None,
) -> Path:
    """Resolve the artifact root from an explicit path or the tracking directory."""
    if artifact_root is not None:
        return Path(artifact_root)
    directory = Path(tracking_dir) if tracking_dir is not None else default_tracking_dir()
    return directory / ARTIFACT_ROOT_DIRNAME


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------


def configure_tracking(
    *,
    tracking_uri: str | None = None,
    artifact_root: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_dir: str | Path | None = None,
) -> TrackingConfig:
    """Point MLflow at the local store and make sure the experiment exists.

    Creates the tracking directory and artifact root if missing, then resolves
    the named experiment, creating it with the declared artifact root on first
    use. Idempotent: repeated calls reuse the experiment.

    Raises
    ------
    TrackingConfigError
        An invalid experiment name was supplied.
    TrackingBackendError
        The tracking directory or artifact root could not be created, or the
        SQLite store could not be opened.
    ExperimentResolutionError
        The experiment could not be created or resolved.
    """
    resolved_name = _validate_experiment_name(experiment_name)
    resolved_uri = resolve_tracking_uri(
        tracking_uri=tracking_uri, tracking_dir=tracking_dir
    )
    resolved_artifact_root = resolve_artifact_root(
        artifact_root=artifact_root, tracking_dir=tracking_dir
    )

    try:
        resolved_artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrackingBackendError(
            f"Could not create artifact root {resolved_artifact_root}: {exc}"
        ) from exc

    try:
        mlflow.set_tracking_uri(resolved_uri)
        client = MlflowClient()
    except (MlflowException, OSError) as exc:
        raise TrackingBackendError(
            f"Could not open the MLflow store at {resolved_uri}: {exc}"
        ) from exc

    created = False
    try:
        experiment = client.get_experiment_by_name(resolved_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                resolved_name,
                artifact_location=resolved_artifact_root.resolve().as_uri(),
                tags={
                    "pipeline": resolved_name,
                    "managed_by": "heart.tracking",
                    "mlflow.note.content": EXPERIMENT_DESCRIPTION,
                },
            )
            created = True
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(resolved_name)
    except (MlflowException, OSError) as exc:
        raise ExperimentResolutionError(
            f"Could not resolve or create MLflow experiment {resolved_name!r} "
            f"in {resolved_uri}: {exc}"
        ) from exc

    config = TrackingConfig(
        tracking_uri=resolved_uri,
        artifact_root=resolved_artifact_root,
        experiment_name=resolved_name,
        experiment_id=experiment_id,
        created_experiment=created,
    )
    logger.info(
        "Configured MLflow tracking: uri=%s artifact_root=%s experiment=%s "
        "(id=%s, created=%s)",
        config.tracking_uri,
        config.artifact_root,
        config.experiment_name,
        config.experiment_id,
        config.created_experiment,
    )
    return config


def current_tracking_uri() -> str:
    """Return the process-wide MLflow tracking URI currently set."""
    return str(mlflow.get_tracking_uri())


def describe_tracking_convention() -> str:
    """Render the fixed store and naming convention as a human-readable report."""
    return "\n".join(
        [
            f"tracking backend: sqlite (no server) at "
            f"{default_tracking_dir() / TRACKING_DB_FILENAME}",
            f"artifact root: {default_artifact_root()}",
            f"experiment: {DEFAULT_EXPERIMENT}",
            "run name: <model-slug>-<split-version> (see heart.tracking.run)",
            "artifacts: metrics.json, run_config.json",
        ]
    )
