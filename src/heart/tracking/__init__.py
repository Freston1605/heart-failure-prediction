"""MLflow tracking for the heart-failure prediction portfolio.

Two modules:

* :mod:`heart.tracking.mlflow_store` — the local SQLite + filesystem backend
  and the single experiment every run belongs to.
* :mod:`heart.tracking.run` — the frozen run-logging convention
  (:func:`~heart.tracking.run.log_evaluation_run`) and read-back helpers.

Import the public surface from here.
"""

from heart.tracking.mlflow_store import (
    ARTIFACT_ROOT_DIRNAME,
    DEFAULT_EXPERIMENT,
    TRACKING_DB_FILENAME,
    TRACKING_DIRNAME,
    ExperimentResolutionError,
    TrackingBackendError,
    TrackingConfig,
    TrackingConfigError,
    TrackingError,
    configure_tracking,
    current_tracking_uri,
    default_artifact_root,
    default_tracking_dir,
    default_tracking_uri,
    describe_tracking_convention,
)
from heart.tracking.run import (
    MAX_PARAM_VALUE_LENGTH,
    METRICS_ARTIFACT,
    RUN_CONFIG_ARTIFACT,
    RUN_NAME_SEPARATOR,
    SOURCE_TAG,
    ArtifactNotFoundError,
    RunNotFoundError,
    RunRecord,
    RunResult,
    TrackingValueError,
    build_run_name,
    ensure_safe_metric_keys,
    load_metrics_artifact,
    load_run,
    load_run_artifact,
    load_run_metrics,
    log_evaluation_run,
    normalise_param_value,
    normalise_params,
    run_smoke,
    slugify,
)

__all__ = [
    # backend
    "DEFAULT_EXPERIMENT",
    "TRACKING_DIRNAME",
    "TRACKING_DB_FILENAME",
    "ARTIFACT_ROOT_DIRNAME",
    "TrackingConfig",
    "configure_tracking",
    "current_tracking_uri",
    "default_tracking_dir",
    "default_artifact_root",
    "default_tracking_uri",
    "describe_tracking_convention",
    # convention
    "METRICS_ARTIFACT",
    "RUN_CONFIG_ARTIFACT",
    "RUN_NAME_SEPARATOR",
    "SOURCE_TAG",
    "MAX_PARAM_VALUE_LENGTH",
    "RunResult",
    "RunRecord",
    "log_evaluation_run",
    "load_run",
    "load_run_metrics",
    "load_run_artifact",
    "load_metrics_artifact",
    "run_smoke",
    "build_run_name",
    "slugify",
    "normalise_param_value",
    "normalise_params",
    "ensure_safe_metric_keys",
    # errors
    "TrackingError",
    "TrackingConfigError",
    "TrackingBackendError",
    "ExperimentResolutionError",
    "TrackingValueError",
    "RunNotFoundError",
    "ArtifactNotFoundError",
]
