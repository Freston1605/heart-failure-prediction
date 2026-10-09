"""Tests for the MLflow tracking convention (S02/T02).

Two contracts are under test:

1. **Backend** — :mod:`heart.tracking.mlflow_store` stands up a local SQLite
   store with a local artifact root, and one shared experiment. No server, no
   network. Configuration is idempotent.
2. **Convention** — :func:`heart.tracking.run.log_evaluation_run` records
   params, the full flattened metric dict, and the ``metrics.json`` /
   ``run_config.json`` artifacts under a fixed experiment/run naming
   convention that every later model reuses.

Every store-backed test runs against a fresh ``tmp_path`` SQLite database, so
the suite never touches the repository's ``experiments/mlruns`` directory.
Negative tests assert the named failure paths (invalid metric dict, illegal
metric key, over-long param, missing artifact, unknown run id, absent
artifact).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from heart.eval.contract import (
    METRIC_KEYS,
    METRIC_SCHEMA_VERSION,
    PRIMARY_METRIC,
    MissingMetricError,
    compute_metric_dict,
    flatten_metrics,
)
from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    TrackingConfigError,
    configure_tracking,
    current_tracking_uri,
    default_artifact_root,
    default_tracking_dir,
    default_tracking_uri,
    describe_tracking_convention,
)
from heart.tracking.run import (
    METRICS_ARTIFACT,
    RUN_CONFIG_ARTIFACT,
    SOURCE_TAG,
    ArtifactNotFoundError,
    RunNotFoundError,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracking_dir(tmp_path):
    """An isolated SQLite store + artifact root for one test."""
    return tmp_path / "mlruns"


@pytest.fixture
def config(tracking_dir):
    """A configured local store with the shared experiment ready."""
    return configure_tracking(tracking_dir=tracking_dir)


def _sample_metrics(n_bins: int = 5) -> dict[str, object]:
    """A complete, schema-valid metric dict built without fitting a model."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 4)
    probabilities = np.clip(0.18 + 0.64 * labels, 0.0, 1.0)
    predictions = (probabilities >= 0.5).astype(int)
    return compute_metric_dict(labels, probabilities, predictions, n_bins=n_bins)


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------


def test_configure_tracking_creates_the_declared_experiment(config):
    assert config.experiment_name == DEFAULT_EXPERIMENT
    assert config.experiment_id
    assert config.created_experiment is True
    assert config.artifact_root.exists()
    client = MlflowClient()
    experiment = client.get_experiment(config.experiment_id)
    assert experiment.name == DEFAULT_EXPERIMENT


def test_configure_tracking_is_idempotent_and_reuses_experiment(tracking_dir):
    first = configure_tracking(tracking_dir=tracking_dir)
    second = configure_tracking(tracking_dir=tracking_dir)
    assert first.experiment_id == second.experiment_id
    assert second.created_experiment is False
    assert MlflowClient().get_experiment(first.experiment_id).name == DEFAULT_EXPERIMENT


def test_configure_tracking_rejects_blank_experiment_name(tracking_dir):
    with pytest.raises(TrackingConfigError):
        configure_tracking(tracking_dir=tracking_dir, experiment_name="   ")


def test_configure_tracking_rejects_slash_in_experiment_name(tracking_dir):
    with pytest.raises(TrackingConfigError):
        configure_tracking(tracking_dir=tracking_dir, experiment_name="a/b")


def test_configuration_tracks_the_active_uri(config):
    assert current_tracking_uri() == config.tracking_uri
    assert config.tracking_uri.startswith("sqlite:///")
    assert config.tracking_uri.endswith("mlflow.db")


def test_default_paths_live_under_the_experiments_directory():
    assert default_tracking_dir().name == "mlruns"
    assert default_artifact_root().parent == default_tracking_dir()
    assert default_tracking_uri().startswith("sqlite:///")
    assert default_tracking_uri().endswith("mlflow.db")


def test_describe_tracking_convention_names_the_store_and_artifacts():
    report = describe_tracking_convention()
    assert DEFAULT_EXPERIMENT in report
    assert METRICS_ARTIFACT in report
    assert RUN_CONFIG_ARTIFACT in report
    assert "sqlite" in report


def test_tracking_config_serialises(config):
    payload = config.to_dict()
    assert payload["experiment_name"] == DEFAULT_EXPERIMENT
    assert payload["experiment_id"] == config.experiment_id
    assert payload["created_experiment"] is True


# ---------------------------------------------------------------------------
# Naming convention and value normalisation
# ---------------------------------------------------------------------------


def test_build_run_name_is_model_slug_plus_split_version():
    assert build_run_name("logistic regression", "v1") == "logistic-regression-v1"
    assert build_run_name("LogisticRegression", "v1") == "logisticregression-v1"
    assert build_run_name("Random Forest", "v2") == "random-forest-v2"


def test_slugify_collapses_non_alphanumerics():
    assert slugify("  XGBoost (tuned) ") == "xgboost-tuned"
    assert slugify("kNN") == "knn"


def test_build_run_name_rejects_blank_model_name():
    with pytest.raises(TrackingValueError):
        build_run_name("   ", "v1")


def test_build_run_name_rejects_blank_split_version():
    with pytest.raises(TrackingValueError):
        build_run_name("logistic regression", "  ")


def test_normalise_param_value_renders_scalars_and_structures():
    assert normalise_param_value(8) == "8"
    assert normalise_param_value(0.5) == "0.5"
    assert normalise_param_value(True) == "True"
    assert normalise_param_value(None) == "null"
    assert normalise_param_value("radial") == "radial"
    assert json.loads(normalise_param_value({"a": 1})) == {"a": 1}
    assert json.loads(normalise_param_value([64, 32])) == [64, 32]
    assert normalise_param_value(np.int64(7)) == "7"
    assert normalise_param_value(np.float64(1.5)) == "1.5"


def test_normalise_param_value_rejects_overlong_value():
    from heart.tracking.run import MAX_PARAM_VALUE_LENGTH

    with pytest.raises(TrackingValueError):
        normalise_param_value("x" * (MAX_PARAM_VALUE_LENGTH + 1))


def test_normalise_params_rejects_non_mapping():
    with pytest.raises(TrackingValueError):
        normalise_params(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_ensure_safe_metric_keys_accepts_canonical_keys():
    ensure_safe_metric_keys(flatten_metrics(_sample_metrics()))


def test_ensure_safe_metric_keys_rejects_illegal_key():
    with pytest.raises(TrackingValueError):
        ensure_safe_metric_keys({"roc_auc": 0.9, "bad key!": 0.1})


# ---------------------------------------------------------------------------
# Logging and read-back round trip
# ---------------------------------------------------------------------------


def test_log_evaluation_run_writes_params_metrics_and_artifacts(config):
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        params={"max_iter": 2000, "C": 1.0},
        config=config,
    )
    record = load_run(result.run_id, config=config)

    # params
    assert record.params["max_iter"] == "2000"
    assert record.params["C"] == "1.0"
    # metrics
    assert "roc_auc" in record.metrics
    assert "confusion_matrix.tp" in record.metrics
    assert record.metrics["calibration.n_bins"] == 5.0
    # artifacts
    assert METRICS_ARTIFACT in record.artifact_paths
    assert RUN_CONFIG_ARTIFACT in record.artifact_paths
    # read the nested artifact back
    payload = load_metrics_artifact(result.run_id, config=config)
    assert set(payload.keys()) == set(METRIC_KEYS)


def test_run_name_and_experiment_follow_the_convention(config):
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        config=config,
    )
    assert result.run_name == "logistic-regression-v1"
    assert result.experiment_name == DEFAULT_EXPERIMENT
    record = load_run(result.run_id, config=config)
    assert record.run_name == "logistic-regression-v1"
    assert record.experiment_name == DEFAULT_EXPERIMENT


def test_convention_tags_are_recorded(config):
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        tags={"owner": "baseline"},
        config=config,
    )
    record = load_run(result.run_id, config=config)
    assert record.tags["model_name"] == "Logistic Regression"
    assert record.tags["model_slug"] == "logistic-regression"
    assert record.tags["split_version"] == "v1"
    assert record.tags["metric_schema_version"] == METRIC_SCHEMA_VERSION
    assert record.tags["primary_metric"] == PRIMARY_METRIC
    assert record.tags["source"] == SOURCE_TAG
    assert record.tags["n_samples"] == "40"
    assert record.tags["owner"] == "baseline"


def test_flattened_scalar_metrics_match_the_nested_artifact(config):
    metrics = _sample_metrics()
    result = log_evaluation_run(
        metrics,
        model_name="Logistic Regression",
        split_version="v1",
        config=config,
    )
    record = load_run(result.run_id, config=config)
    flat = flatten_metrics(metrics)
    assert result.metrics_logged == len(flat)
    for key, value in flat.items():
        assert record.metrics[key] == pytest.approx(value)


def test_metrics_artifact_round_trips_the_complete_dict(config):
    metrics = _sample_metrics()
    result = log_evaluation_run(
        metrics,
        model_name="Logistic Regression",
        split_version="v1",
        config=config,
    )
    payload = load_metrics_artifact(result.run_id, config=config)
    assert payload["confusion_matrix"] == metrics["confusion_matrix"]
    assert payload["calibration"]["bins"] == metrics["calibration"]["bins"]


def test_run_config_artifact_records_the_naming_convention(config):
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        params={"max_iter": 1000},
        config=config,
    )
    run_config = load_run_artifact(result.run_id, RUN_CONFIG_ARTIFACT, config=config)
    assert run_config["experiment_name"] == DEFAULT_EXPERIMENT
    assert run_config["run_name"] == "logistic-regression-v1"
    assert run_config["model_slug"] == "logistic-regression"
    assert run_config["split_version"] == "v1"
    assert run_config["primary_metric"] == PRIMARY_METRIC
    assert run_config["metric_keys"] == list(METRIC_KEYS)
    assert run_config["params"]["max_iter"] == "1000"
    assert run_config["source"] == SOURCE_TAG


def test_load_run_metrics_returns_logged_scalars(config):
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        config=config,
    )
    scalars = load_run_metrics(result.run_id, config=config)
    assert scalars["roc_auc"] == pytest.approx(1.0)
    assert scalars["accuracy"] == pytest.approx(1.0)


def test_extra_artifact_is_logged_and_listed(config, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("baseline run notes\n", encoding="utf-8")
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        artifacts=[notes],
        config=config,
    )
    record = load_run(result.run_id, config=config)
    assert "notes.txt" in record.artifact_paths


def test_two_runs_share_the_convention_name(config):
    first = log_evaluation_run(
        _sample_metrics(), model_name="Logistic Regression", split_version="v1", config=config
    )
    second = log_evaluation_run(
        _sample_metrics(), model_name="Logistic Regression", split_version="v1", config=config
    )
    assert first.run_name == second.run_name == "logistic-regression-v1"
    assert first.run_id != second.run_id


def test_run_smoke_round_trips_end_to_end(tracking_dir):
    result = run_smoke(tracking_dir=tracking_dir)
    record = load_run(result.run_id)
    payload = load_metrics_artifact(result.run_id)

    assert result.run_name == "smoke-logistic-regression-smoke"
    assert record.params["n_features"] == "3"
    assert "roc_auc" in record.metrics
    assert METRICS_ARTIFACT in record.artifact_paths
    assert RUN_CONFIG_ARTIFACT in record.artifact_paths
    assert set(payload.keys()) == set(METRIC_KEYS)
    assert record.tags["purpose"] == "tracking-smoke-test"


# ---------------------------------------------------------------------------
# Negative paths
# ---------------------------------------------------------------------------


def test_log_evaluation_run_rejects_missing_metric_without_creating_a_run(config):
    metrics = _sample_metrics()
    del metrics["roc_auc"]
    with pytest.raises(MissingMetricError):
        log_evaluation_run(
            metrics,
            model_name="Logistic Regression",
            split_version="v1",
            config=config,
        )
    runs = MlflowClient().search_runs(experiment_ids=[config.experiment_id])
    assert runs == []


def test_log_evaluation_run_rejects_non_mapping_metrics(config):
    with pytest.raises(TrackingValueError):
        log_evaluation_run(
            ["not", "a", "mapping"],  # type: ignore[arg-type]
            model_name="Logistic Regression",
            split_version="v1",
            config=config,
        )


def test_log_evaluation_run_rejects_missing_artifact_path(config, tmp_path):
    with pytest.raises(TrackingValueError):
        log_evaluation_run(
            _sample_metrics(),
            model_name="Logistic Regression",
            split_version="v1",
            artifacts=[tmp_path / "does-not-exist.txt"],
            config=config,
        )


def test_log_evaluation_run_rejects_non_mapping_tags(config):
    with pytest.raises(TrackingValueError):
        log_evaluation_run(
            _sample_metrics(),
            model_name="Logistic Regression",
            split_version="v1",
            tags=["not", "a", "mapping"],  # type: ignore[arg-type]
            config=config,
        )


def test_load_run_unknown_id_raises(config):
    with pytest.raises(RunNotFoundError):
        load_run("00000000000000000000000000000000", config=config)


def test_load_run_rejects_blank_id(config):
    with pytest.raises(TrackingValueError):
        load_run("", config=config)


def test_load_artifact_missing_raises(config):
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        config=config,
    )
    with pytest.raises(ArtifactNotFoundError):
        load_run_artifact(result.run_id, "nope.json", config=config)


def test_load_artifact_rejects_blank_path(config):
    result = log_evaluation_run(
        _sample_metrics(),
        model_name="Logistic Regression",
        split_version="v1",
        config=config,
    )
    with pytest.raises(TrackingValueError):
        load_run_artifact(result.run_id, "  ", config=config)


def test_named_errors_share_a_base():
    from heart.tracking.mlflow_store import TrackingError

    assert issubclass(TrackingValueError, TrackingError)
    assert issubclass(RunNotFoundError, TrackingError)
    assert issubclass(ArtifactNotFoundError, TrackingError)
    assert issubclass(TrackingConfigError, TrackingError)
