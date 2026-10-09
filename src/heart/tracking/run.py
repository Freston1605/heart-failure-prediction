"""The fixed MLflow run-logging convention for every portfolio model.

One function, :func:`log_evaluation_run`, records a model's full metric dict,
its parameters, and its reproducibility artifacts into the local MLflow store
configured by :mod:`heart.tracking.mlflow_store`. Because the convention is
fixed here, logistic regression, kNN, SVM, forests, XGBoost, and the MLP are
all directly comparable in the MLflow UI and in the leaderboard.

The convention (frozen in S02/T02)
----------------------------------
* **Experiment** — :data:`heart.tracking.mlflow_store.DEFAULT_EXPERIMENT`
  (``heart-failure-prediction``). Every run belongs to it.
* **Run name** — ``<model-slug>-<split-version>``, for example
  ``logistic-regression-v1``. :func:`build_run_name` is the only builder; the
  model name is slugified (lowercase, non-alphanumerics collapsed to ``-``) so
  the name is stable and filesystem/URL safe.
* **Tags** — ``model_name``, ``model_slug``, ``split_version``,
  ``metric_schema_version``, ``primary_metric``, ``n_samples``, and
  ``source`` (always ``heart.tracking.run``), plus any caller tags.
* **Params** — every model hyperparameter, after
  :func:`normalise_param_value` renders it as a short string.
* **Metrics** — every scalar leaf of the canonical metric dict, flattened by
  :func:`heart.eval.contract.flatten_metrics` and written with dotted names
  (``roc_auc``, ``confusion_matrix.tp``, ``calibration.bins.0.count``).
* **Artifacts** — ``metrics.json`` (the complete nested metric dict, including
  the full reliability curve and confusion matrix) and ``run_config.json``
  (naming convention, params, tags, and the declared metric schema).

Why the nested dict is logged as an artifact
--------------------------------------------
MLflow metrics are scalars. The calibration reliability curve and the
confusion-matrix block are structured, so a lossy scalar-only log is not
auditable. The scalar leaves serve dashboards; ``metrics.json`` preserves the
exact evaluated dict for anyone re-deriving a number.

Observability
-------------
:func:`log_evaluation_run` logs the experiment, run name, run id, model, and
primary metric at ``INFO`` (and again at ``INFO`` after the run closes, so a
crash mid-log is visible as a missing completion line). :func:`load_run`
reads the run back; :func:`describe_tracking_convention` prints the frozen
convention.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:  # pragma: no cover - exercised by the environment, not by logic
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "heart.tracking requires MLflow, which is not installed. Install the "
        'tracking extra with: pip install -e ".[ml]"'
    ) from exc

from heart.config import RANDOM_SEED
from heart.eval.contract import METRIC_KEYS, METRIC_SCHEMA_VERSION, PRIMARY_METRIC
from heart.eval.contract import evaluate, flatten_metrics, validate_metric_dict
from heart.eval.contract import EvaluationSplit
from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    TrackingConfig,
    TrackingError,
    configure_tracking,
    describe_tracking_convention,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen naming convention
# ---------------------------------------------------------------------------

#: Artifact holding the complete nested metric dict.
METRICS_ARTIFACT: str = "metrics.json"

#: Artifact holding the naming convention, params, and tags of a run.
RUN_CONFIG_ARTIFACT: str = "run_config.json"

#: Separator between the model slug and the split version in a run name.
RUN_NAME_SEPARATOR: str = "-"

#: Tag recording which code path produced the run.
SOURCE_TAG: str = "heart.tracking.run"

#: Maximum length of a logged MLflow param value.
MAX_PARAM_VALUE_LENGTH: int = 6000

#: Characters permitted in an MLflow metric key (MLflow's own restriction).
_MLFLOW_METRIC_KEY_RE = re.compile(r"^[A-Za-z0-9_\-./ ]+$")

#: Characters collapsed to a single ``-`` when slugifying a model name.
_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class TrackingValueError(TrackingError):
    """A value handed to the run logger is malformed for MLflow."""


class RunNotFoundError(TrackingError):
    """The requested MLflow run id does not exist in the store."""


class ArtifactNotFoundError(TrackingError):
    """The requested artifact is not attached to the run."""


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def slugify(name: object) -> str:
    """Return a lowercase, hyphen-separated slug for ``name``.

    Raises :class:`TrackingValueError` when nothing slug-worthy remains, so a
    blank model name fails loudly rather than producing a nameless run.
    """
    if not isinstance(name, str) or not name.strip():
        raise TrackingValueError(
            f"model_name must be a non-empty string, got {name!r}."
        )
    slug = _SLUG_NON_ALNUM_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise TrackingValueError(
            f"model_name {name!r} contains no alphanumeric characters to slugify."
        )
    return slug


def build_run_name(model_name: str, split_version: str) -> str:
    """Build the fixed ``<model-slug>-<split-version>`` run name."""
    if not isinstance(split_version, str) or not split_version.strip():
        raise TrackingValueError(
            f"split_version must be a non-empty string, got {split_version!r}."
        )
    return f"{slugify(model_name)}{RUN_NAME_SEPARATOR}{split_version.strip()}"


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------


def _json_default(value: object) -> object:
    """JSON fallback that normalises numpy scalars and rejects the rest."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def normalise_param_value(value: object) -> str:
    """Render a parameter value as an MLflow-safe string.

    Scalars keep their natural rendering, ``None`` becomes ``"null"``, and
    dicts/lists are JSON-encoded so structured hyperparameters (for example a
    layer list) survive as one readable param.
    """
    if isinstance(value, str):
        rendered = value
    elif isinstance(value, bool):
        rendered = "True" if value else "False"
    elif value is None:
        rendered = "null"
    elif isinstance(value, (int, float, np.integer, np.floating)):
        rendered = str(value.item() if isinstance(value, np.generic) else value)
    elif isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, default=_json_default, sort_keys=True)
    else:
        rendered = str(value)
    if len(rendered) > MAX_PARAM_VALUE_LENGTH:
        raise TrackingValueError(
            f"Parameter value exceeds {MAX_PARAM_VALUE_LENGTH} characters "
            f"({len(rendered)}); MLflow params are scalars, not blobs. Log the "
            "full value as an artifact instead."
        )
    return rendered


def normalise_params(params: Mapping[str, object] | None) -> dict[str, str]:
    """Normalise a params mapping into MLflow-safe string key/value pairs."""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TrackingValueError(
            f"params must be a mapping, got {type(params).__name__}."
        )
    normalised: dict[str, str] = {}
    for key, value in params.items():
        name = str(key)
        if not name:
            raise TrackingValueError("Parameter names may not be empty.")
        normalised[name] = normalise_param_value(value)
    return normalised


def ensure_safe_metric_keys(flat_metrics: Mapping[str, object]) -> None:
    """Raise :class:`TrackingValueError` if any metric key is MLflow-illegal."""
    unsafe = [
        key
        for key in flat_metrics
        if not isinstance(key, str) or not _MLFLOW_METRIC_KEY_RE.match(key)
    ]
    if unsafe:
        raise TrackingValueError(
            f"Metric key(s) {unsafe} contain characters MLflow rejects; allowed "
            "characters are alphanumerics, '_', '-', '.', '/', and space."
        )


# ---------------------------------------------------------------------------
# Result / record value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Everything needed to locate and read back a logged run."""

    run_id: str
    experiment_id: str
    experiment_name: str
    run_name: str
    tracking_uri: str
    metrics_logged: int
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tracking_uri": self.tracking_uri,
            "metrics_logged": self.metrics_logged,
            "tags": dict(self.tags),
        }


@dataclass(frozen=True)
class RunRecord:
    """A run read back from the MLflow store."""

    run_id: str
    run_name: str
    experiment_id: str
    experiment_name: str
    status: str
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]
    artifact_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "tags": dict(self.tags),
            "artifact_paths": list(self.artifact_paths),
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _build_run_config(
    *,
    model_name: str,
    split_version: str,
    run_name: str,
    config: TrackingConfig,
    params: Mapping[str, str],
    tags: Mapping[str, str],
) -> dict[str, object]:
    return {
        "experiment_name": config.experiment_name,
        "run_name": run_name,
        "model_name": model_name,
        "model_slug": slugify(model_name),
        "split_version": split_version,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "metric_keys": list(METRIC_KEYS),
        "params": dict(params),
        "tags": dict(tags),
        "tracking_uri": config.tracking_uri,
        "source": SOURCE_TAG,
    }


def log_evaluation_run(
    metrics: Mapping[str, object],
    *,
    model_name: str,
    split_version: str,
    params: Mapping[str, object] | None = None,
    tags: Mapping[str, object] | None = None,
    run_name: str | None = None,
    artifacts: Sequence[str | Path] | None = None,
    n_samples: int | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    config: TrackingConfig | None = None,
    tracking_dir: str | Path | None = None,
    validate: bool = True,
) -> RunResult:
    """Log one evaluated model to MLflow under the fixed convention.

    Parameters
    ----------
    metrics:
        The canonical metric dict from :func:`heart.eval.contract.evaluate`.
        Schema-checked by default, so a malformed dict fails before any run is
        created.
    model_name:
        Human-readable model name; slugified into the run name.
    split_version:
        The split version the metrics were computed on (for example ``v1``).
    params:
        Model hyperparameters (normalised to strings).
    tags:
        Extra MLflow tags merged over the convention tags.
    run_name:
        Override for the convention run name (rarely needed; the convention
        name is the default).
    artifacts:
        Extra existing files to log alongside ``metrics.json`` and
        ``run_config.json`` (logged at the run's artifact root).
    n_samples:
        Evaluated sample count; defaults to ``confusion_matrix.n_samples``.
    config:
        A pre-resolved :class:`TrackingConfig`. Resolved from
        ``tracking_dir``/``experiment_name`` when omitted.

    Returns
    -------
    RunResult
        The run id and resolved naming metadata.
    """
    if not isinstance(metrics, Mapping):
        raise TrackingValueError(
            f"metrics must be a mapping, got {type(metrics).__name__}."
        )
    metric_dict = dict(metrics)
    if validate:
        validate_metric_dict(metric_dict)

    flat_metrics = flatten_metrics(metric_dict)
    ensure_safe_metric_keys(flat_metrics)

    normalised_params = normalise_params(params)

    if n_samples is None:
        matrix = metric_dict.get("confusion_matrix")
        if isinstance(matrix, Mapping) and "n_samples" in matrix:
            n_samples = int(matrix["n_samples"])  # type: ignore[arg-type]

    resolved_run_name = run_name or build_run_name(model_name, split_version)

    run_tags: dict[str, str] = {
        "model_name": str(model_name),
        "model_slug": slugify(model_name),
        "split_version": str(split_version),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "source": SOURCE_TAG,
    }
    if n_samples is not None:
        run_tags["n_samples"] = str(int(n_samples))
    if tags is not None:
        if not isinstance(tags, Mapping):
            raise TrackingValueError(
                f"tags must be a mapping, got {type(tags).__name__}."
            )
        run_tags.update({str(key): str(value) for key, value in tags.items()})

    config = config or configure_tracking(
        experiment_name=experiment_name, tracking_dir=tracking_dir
    )

    for artifact in artifacts or ():
        if not Path(artifact).exists():
            raise TrackingValueError(
                f"Artifact to log does not exist: {artifact}"
            )

    run_config = _build_run_config(
        model_name=str(model_name),
        split_version=str(split_version),
        run_name=resolved_run_name,
        config=config,
        params=normalised_params,
        tags=run_tags,
    )

    try:
        with mlflow.start_run(
            run_name=resolved_run_name, experiment_id=config.experiment_id
        ) as run:
            run_id = run.info.run_id
            mlflow.log_params(normalised_params)
            mlflow.set_tags(run_tags)
            mlflow.log_metrics(flat_metrics)
            mlflow.log_dict(metric_dict, METRICS_ARTIFACT)
            mlflow.log_dict(run_config, RUN_CONFIG_ARTIFACT)
            for artifact in artifacts or ():
                mlflow.log_artifact(str(artifact))
    except (MlflowException, OSError) as exc:
        raise TrackingError(
            f"MLflow refused to log run {resolved_run_name!r}: {exc}"
        ) from exc

    logger.info(
        "Logged run '%s' (id=%s) to experiment '%s': model=%s split=%s "
        "metrics=%d primary(%s)=%.4f",
        resolved_run_name,
        run_id,
        config.experiment_name,
        model_name,
        split_version,
        len(flat_metrics),
        PRIMARY_METRIC,
        float(metric_dict[PRIMARY_METRIC]),
    )
    logger.info("Run '%s' closed successfully (id=%s)", resolved_run_name, run_id)

    return RunResult(
        run_id=run_id,
        experiment_id=config.experiment_id,
        experiment_name=config.experiment_name,
        run_name=resolved_run_name,
        tracking_uri=config.tracking_uri,
        metrics_logged=len(flat_metrics),
        tags=run_tags,
    )


# ---------------------------------------------------------------------------
# Read-back
# ---------------------------------------------------------------------------


def _client(config: TrackingConfig | None = None) -> MlflowClient:
    if config is not None:
        mlflow.set_tracking_uri(config.tracking_uri)
    return MlflowClient()


def load_run(
    run_id: str, *, config: TrackingConfig | None = None
) -> RunRecord:
    """Read a logged run (params, metrics, tags, artifacts) back from the store."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise TrackingValueError(f"run_id must be a non-empty string, got {run_id!r}.")
    client = _client(config)
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        raise RunNotFoundError(
            f"No MLflow run with id {run_id!r} in the configured store: {exc}"
        ) from exc
    artifacts = tuple(sorted(info.path for info in client.list_artifacts(run_id)))
    return RunRecord(
        run_id=run_id,
        run_name=str(run.data.tags.get("mlflow.runName", "")),
        experiment_id=run.info.experiment_id,
        experiment_name=str(client.get_experiment(run.info.experiment_id).name),
        status=str(run.info.status),
        params=dict(run.data.params),
        metrics=dict(run.data.metrics),
        tags=dict(run.data.tags),
        artifact_paths=artifacts,
    )


def load_run_metrics(
    run_id: str, *, config: TrackingConfig | None = None
) -> dict[str, float]:
    """Return only the scalar metrics logged on ``run_id``."""
    return dict(load_run(run_id, config=config).metrics)


def load_run_artifact(
    run_id: str,
    path: str = METRICS_ARTIFACT,
    *,
    config: TrackingConfig | None = None,
) -> object:
    """Download a JSON artifact attached to ``run_id`` and return its content."""
    if not isinstance(path, str) or not path.strip():
        raise TrackingValueError(f"path must be a non-empty string, got {path!r}.")
    client = _client(config)
    if path not in {info.path for info in client.list_artifacts(run_id)}:
        raise ArtifactNotFoundError(
            f"Run {run_id!r} has no artifact {path!r} at its root. "
            f"Present: {sorted(info.path for info in client.list_artifacts(run_id))}."
        )
    try:
        local = client.download_artifacts(run_id, path)
    except (MlflowException, OSError) as exc:
        raise ArtifactNotFoundError(
            f"Could not download artifact {path!r} from run {run_id!r}: {exc}"
        ) from exc
    return json.loads(Path(local).read_text(encoding="utf-8"))


def load_metrics_artifact(
    run_id: str, *, config: TrackingConfig | None = None
) -> dict[str, object]:
    """Return the complete nested metric dict logged as ``metrics.json``."""
    content = load_run_artifact(run_id, METRICS_ARTIFACT, config=config)
    if not isinstance(content, dict):
        raise TrackingError(
            f"{METRICS_ARTIFACT} for run {run_id!r} is not a JSON object."
        )
    return content


# ---------------------------------------------------------------------------
# Smoke run (proves the store round-trips end to end)
# ---------------------------------------------------------------------------


def run_smoke(
    *,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> RunResult:
    """Train a tiny model, evaluate it, and log it through the full convention.

    This is the executable proof that the tracking convention works: it builds
    a small held-out split, scores a logistic regression through the shared
    :func:`heart.eval.contract.evaluate`, and logs params, the flattened
    metrics, and the ``metrics.json`` / ``run_config.json`` artifacts.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(RANDOM_SEED)
    n = 240
    train_matrix = rng.normal(size=(n, 3))
    logit = 1.3 * train_matrix[:, 0] - 0.9 * train_matrix[:, 1]
    train_labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    test_matrix = rng.normal(size=(n, 3))
    test_logit = 1.3 * test_matrix[:, 0] - 0.9 * test_matrix[:, 1]
    test_labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-test_logit))).astype(int)

    import pandas as pd

    train = pd.DataFrame(train_matrix, columns=["a", "b", "c"])
    test = pd.DataFrame(test_matrix, columns=["a", "b", "c"])
    split = EvaluationSplit(X_test=test, y_test=test_labels, name="smoke/test")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(train, train_labels)
    metrics = evaluate(model, split)

    return log_evaluation_run(
        metrics,
        model_name="smoke-logistic-regression",
        split_version="smoke",
        params={"max_iter": 1000, "random_state": RANDOM_SEED, "n_features": 3},
        tags={"purpose": "tracking-smoke-test"},
        experiment_name=experiment_name,
        tracking_dir=tracking_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.tracking.run",
        description="Log a smoke run through the fixed MLflow convention.",
    )
    parser.add_argument(
        "--tracking-dir",
        default=None,
        help="Directory for the SQLite store and artifacts "
        "(default: experiments/mlruns).",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    print(describe_tracking_convention())
    result = run_smoke(
        tracking_dir=args.tracking_dir, experiment_name=args.experiment
    )
    record = load_run(result.run_id)
    payload = load_metrics_artifact(result.run_id)
    print(f"run id: {result.run_id}")
    print(f"experiment: {result.experiment_name} ({result.experiment_id})")
    print(f"run name: {result.run_name}")
    print(f"metrics logged: {result.metrics_logged}")
    print(f"artifacts: {list(record.artifact_paths)}")
    print(f"read-back roc_auc: {record.metrics.get('roc_auc')}")
    print(f"read-back metrics.json keys: {sorted(payload.keys())}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
