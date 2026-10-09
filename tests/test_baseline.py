"""Tests for the logistic-regression baseline (S02/T03).

The baseline is the portfolio's honesty anchor, so these tests pin three
things:

1. It trains on the S01 training rows and predicts on the held-out test rows
   with correct shapes and valid probabilities.
2. It scores through the shared :func:`heart.eval.contract.evaluate`, producing
   the complete canonical metric dict with sane values.
3. Running it logs an MLflow run under the frozen convention and publishes a
   report containing the actual numbers.

Read-only split frames come from the versioned S01 artifacts
(``data/processed/splits/v1``); every write goes to ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, SplitNotFoundError, load_split_frames
from heart.eval.contract import (
    METRIC_KEYS,
    PRIMARY_METRIC,
    evaluate,
    evaluation_split_from_frames,
)
from heart.models.baseline import (
    BASELINE_MODEL_NAME,
    BASELINE_PARAMS,
    BaselineDataError,
    BaselineReportError,
    BaselineResult,
    build_baseline_model,
    render_baseline_report,
    run_baseline,
    train_baseline,
)
from heart.tracking.run import (
    METRICS_ARTIFACT,
    load_metrics_artifact,
    load_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split_frames():
    """The versioned S01 train/test frames (read-only)."""
    return load_split_frames(SPLIT_VERSION)


@pytest.fixture(scope="module")
def fitted_model(split_frames):
    train, _ = split_frames
    return train_baseline(train)


@pytest.fixture(scope="module")
def evaluated(fitted_model, split_frames):
    _, test = split_frames
    split = evaluation_split_from_frames(test, name=f"{SPLIT_VERSION}/test")
    return test, evaluate(fitted_model, split)


# ---------------------------------------------------------------------------
# Training and prediction on the S01 split
# ---------------------------------------------------------------------------


def test_baseline_trains_and_predicts_on_the_s01_split(split_frames, fitted_model):
    train, test = split_frames
    assert len(train) == 734
    assert len(test) == 184

    probabilities = fitted_model.predict_proba(test[list(FEATURE_COLUMNS)])
    predictions = fitted_model.predict(test[list(FEATURE_COLUMNS)])

    assert probabilities.shape == (len(test), 2)
    assert predictions.shape == (len(test),)
    assert np.all(np.isfinite(probabilities))
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    assert set(np.unique(predictions)).issubset({0, 1})


def test_evaluate_returns_the_complete_metric_dict(evaluated):
    _, metrics = evaluated
    assert tuple(metrics.keys()) == METRIC_KEYS
    assert set(metrics.keys()) == set(METRIC_KEYS)


def test_baseline_metrics_are_sane(evaluated):
    _, metrics = evaluated
    assert 0.7 <= metrics[PRIMARY_METRIC] <= 1.0
    assert 0.6 <= metrics["accuracy"] <= 1.0
    for key in ("precision", "recall", "f1", "pr_auc", "specificity", "npv"):
        assert 0.0 <= metrics[key] <= 1.0
    matrix = metrics["confusion_matrix"]
    assert matrix["n_samples"] == 184
    assert matrix["n_samples"] == matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"]


def test_baseline_evaluation_is_deterministic(fitted_model, split_frames):
    _, test = split_frames
    split = evaluation_split_from_frames(test, name=f"{SPLIT_VERSION}/test")
    assert evaluate(fitted_model, split) == evaluate(fitted_model, split)


def test_build_baseline_model_uses_the_declared_params():
    model = build_baseline_model()
    classifier = model.named_steps["clf"]
    assert classifier.C == BASELINE_PARAMS["C"]
    assert classifier.max_iter == BASELINE_PARAMS["max_iter"]
    assert classifier.random_state == BASELINE_PARAMS["random_state"]
    assert classifier.solver == BASELINE_PARAMS["solver"]


def test_train_baseline_rejects_a_missing_target():
    frame = pd.DataFrame({column: [1, 2, 3] for column in FEATURE_COLUMNS})
    with pytest.raises(BaselineDataError):
        train_baseline(frame)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_report_contains_the_actual_numbers(evaluated):
    _, metrics = evaluated
    result = BaselineResult(
        model_name=BASELINE_MODEL_NAME,
        split_version=SPLIT_VERSION,
        train_rows=734,
        test_rows=184,
        params=dict(BASELINE_PARAMS),
        metrics=metrics,
        run_id=None,
        experiment_name="heart-failure-prediction",
        report_path=None,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    report = render_baseline_report(result)

    for heading in (
        "# Logistic-Regression Baseline",
        "## Headline",
        "## Confusion matrix",
        "## Calibration (reliability curve)",
        "## Provenance",
        "## Reproduce",
    ):
        assert heading in report
    assert f"**{float(metrics[PRIMARY_METRIC]):.4f}**" in report
    assert f"| accuracy | {float(metrics['accuracy']):.4f} |" in report
    assert f"- split: `{SPLIT_VERSION}`" in report


# ---------------------------------------------------------------------------
# End-to-end run: MLflow run + report
# ---------------------------------------------------------------------------


def test_run_baseline_logs_the_complete_metric_dict_to_mlflow(tmp_path):
    report = tmp_path / "baseline.md"
    result = run_baseline(
        tracking_dir=tmp_path / "mlruns", report_path=report
    )

    assert result.run_id is not None
    assert result.report_path == report

    record = load_run(result.run_id)
    assert record.run_name == "logistic-regression-v1"
    assert record.tags["model_name"] == BASELINE_MODEL_NAME
    assert record.tags["split_version"] == SPLIT_VERSION
    assert record.params["max_iter"] == str(BASELINE_PARAMS["max_iter"])
    assert record.params["solver"] == BASELINE_PARAMS["solver"]

    # every scalar leaf of the canonical dict is present in the MLflow run
    for key in ("roc_auc", "accuracy", "confusion_matrix.tp", "calibration.n_bins"):
        assert key in record.metrics

    artifact = load_metrics_artifact(result.run_id)
    assert set(artifact.keys()) == set(METRIC_KEYS)
    assert METRICS_ARTIFACT in record.artifact_paths


def test_run_baseline_publishes_the_report(tmp_path):
    report = tmp_path / "nested" / "baseline.md"
    result = run_baseline(
        tracking_dir=tmp_path / "mlruns", report_path=report
    )
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert f"**{float(result.metrics[PRIMARY_METRIC]):.4f}**" in text
    assert f"| accuracy | {float(result.metrics['accuracy']):.4f} |" in text
    assert "| **ROC-AUC (primary)** |" in text


def test_run_baseline_can_skip_mlflow_and_report(tmp_path):
    result = run_baseline(
        tracking_dir=tmp_path / "mlruns",
        log_to_mlflow=False,
        write_report=False,
        report_path=None,
    )
    assert result.run_id is None
    assert result.report_path is None
    assert set(result.metrics.keys()) == set(METRIC_KEYS)


def test_run_baseline_unknown_split_version_raises(tmp_path):
    with pytest.raises(SplitNotFoundError):
        run_baseline(
            split_version="v-does-not-exist",
            tracking_dir=tmp_path / "mlruns",
            write_report=False,
            report_path=None,
        )


def test_run_baseline_report_write_failure_raises(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(BaselineReportError):
        run_baseline(
            tracking_dir=tmp_path / "mlruns",
            log_to_mlflow=False,
            report_path=blocker / "baseline.md",
        )


def test_baseline_errors_share_a_base():
    from heart.models.baseline import BaselineError

    assert issubclass(BaselineDataError, BaselineError)
    assert issubclass(BaselineReportError, BaselineError)
