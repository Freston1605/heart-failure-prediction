"""Tests for log-loss in the canonical metric contract (M006/S02/T01).

Log-loss joins the metric schema as schema version ``2``:

* it is a **loss**, not a rate — a finite, non-negative number with no upper
  bound — so it must live in :data:`LOSS_METRIC_KEYS` and never in
  :data:`RATE_METRIC_KEYS` (whose validation enforces a ``[0, 1]`` domain);
* :func:`compute_metric_dict` computes it with
  :func:`sklearn.metrics.log_loss` on ``(y_true, y_proba)`` with
  ``labels=[0, 1]``, and this suite cross-checks the value against a direct
  call on the same inputs;
* :func:`validate_metric_dict` accepts any finite ``>= 0`` value and rejects
  negative, non-finite, missing, and misnamed entries with the named
  :class:`MetricValueError` / schema-error subclasses;
* the end-to-end :func:`heart.eval.contract.evaluate` path returns a dict
  that validates cleanly, so every downstream consumer (MLflow logging via
  ``heart.tracking.run.log_evaluation_run``) sees the new key for free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss

from heart.eval import LOSS_METRIC_KEYS, METRIC_KEYS, SCALAR_METRIC_KEYS
from heart.eval.contract import (
    EvaluationSplit,
    MetricSchemaError,
    MetricValueError,
    MissingMetricError,
    UnexpectedMetricError,
    evaluate,
    validate_metric_dict,
)
from heart.eval.metrics import (
    LOSS_METRIC_KEYS as LOSS_FROM_METRICS,
    METRIC_KEYS as METRIC_KEYS_FROM_METRICS,
    METRIC_SCHEMA_VERSION,
    RATE_METRIC_KEYS,
    SCALAR_METRIC_KEYS as SCALAR_FROM_METRICS,
    compute_metric_dict,
)

RANDOM_SEED = 7


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _inline_problem(n: int = 200, *, seed: int = 3):
    """A small self-contained binary problem; returns (model, split)."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, 3))
    logit = 1.4 * matrix[:, 0] - 0.9 * matrix[:, 1] + 0.3 * matrix[:, 2]
    labels = (rng.random(n) < expit(logit)).astype(int)
    order = rng.permutation(n)
    train_idx, test_idx = order[: int(0.7 * n)], order[int(0.7 * n) :]
    columns = ["a", "b", "c"]
    X_train = pd.DataFrame(matrix[train_idx], columns=columns)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, labels[train_idx])
    split = EvaluationSplit(
        X_test=pd.DataFrame(matrix[test_idx], columns=columns),
        y_test=labels[test_idx],
        name="inline/test",
    )
    return model, split


@pytest.fixture(scope="module")
def inline_eval():
    return _inline_problem()


@pytest.fixture(scope="module")
def inline_model(inline_eval):
    return inline_eval[0]


@pytest.fixture(scope="module")
def inline_split(inline_eval):
    return inline_eval[1]


def _hand_built_metrics(log_loss_value: float) -> dict[str, object]:
    """A fully hand-built canonical dict (not produced by the contract).

    Four samples, labels ``[0, 1, 0, 1]``, probabilities
    ``[0.2, 0.8, 0.3, 0.6]``, predictions ``[0, 1, 0, 1]``: tn=2, fp=0, fn=0,
    tp=2. Two calibration bins; the hand-derived scalars below mirror what
    the producer would emit so only the ``log_loss`` value under test varies.
    """
    return {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "pr_auc": 1.0,
        "specificity": 1.0,
        "npv": 1.0,
        "prevalence": 0.5,
        "brier_score": 0.0825,
        "log_loss": log_loss_value,
        "confusion_matrix": {
            "tn": 2,
            "fp": 0,
            "fn": 0,
            "tp": 2,
            "n_samples": 4,
            "n_positive": 2,
            "n_negative": 2,
        },
        "calibration": {
            "brier_score": 0.0825,
            "expected_calibration_error": 0.275,
            "n_bins": 2,
            "bins": [
                {
                    "bin": 0,
                    "lower": 0.0,
                    "upper": 0.5,
                    "count": 2,
                    "mean_predicted": 0.25,
                    "fraction_positive": 0.0,
                    "gap": 0.25,
                },
                {
                    "bin": 1,
                    "lower": 0.5,
                    "upper": 1.0,
                    "count": 2,
                    "mean_predicted": 0.7,
                    "fraction_positive": 1.0,
                    "gap": 0.3,
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# Schema declaration
# ---------------------------------------------------------------------------


def test_log_loss_is_declared_exactly_once():
    assert LOSS_METRIC_KEYS == ("log_loss",)
    assert METRIC_KEYS.count("log_loss") == 1
    assert SCALAR_METRIC_KEYS.count("log_loss") == 1


def test_log_loss_is_a_loss_key_not_a_rate_key():
    assert "log_loss" not in RATE_METRIC_KEYS
    assert SCALAR_METRIC_KEYS == RATE_METRIC_KEYS + LOSS_METRIC_KEYS
    assert METRIC_KEYS == SCALAR_METRIC_KEYS + (
        "confusion_matrix",
        "calibration",
    )


def test_schema_version_is_bumped_to_two():
    assert METRIC_SCHEMA_VERSION == "2"


def test_exports_resolve_from_the_package():
    assert LOSS_FROM_METRICS == LOSS_METRIC_KEYS == ("log_loss",)
    assert METRIC_KEYS_FROM_METRICS == METRIC_KEYS
    assert SCALAR_FROM_METRICS == SCALAR_METRIC_KEYS


# ---------------------------------------------------------------------------
# Arithmetic cross-check against the direct sklearn call
# ---------------------------------------------------------------------------


def test_compute_metric_dict_matches_direct_sklearn_log_loss(inline_model, inline_split):
    labels = inline_split.y_test
    probabilities = inline_model.predict_proba(inline_split.X_test)[:, 1]
    predictions = np.asarray(inline_model.predict(inline_split.X_test)).ravel()
    metrics = compute_metric_dict(labels, probabilities, predictions)
    assert metrics["log_loss"] == pytest.approx(
        float(sk_log_loss(labels, probabilities, labels=[0, 1]))
    )


def test_computed_log_loss_is_non_negative_and_finite(inline_model, inline_split):
    probabilities = inline_model.predict_proba(inline_split.X_test)[:, 1]
    predictions = np.asarray(inline_model.predict(inline_split.X_test)).ravel()
    metrics = compute_metric_dict(
        inline_split.y_test, probabilities, predictions
    )
    value = float(metrics["log_loss"])
    assert np.isfinite(value)
    assert value >= 0.0


def test_hand_built_log_loss_alone_matches_sklearn_on_its_own_inputs():
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.2, 0.8, 0.3, 0.6])
    expected = float(sk_log_loss(labels, probabilities, labels=[0, 1]))
    assert _hand_built_metrics(expected)["log_loss"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Validation: acceptance and named rejections
# ---------------------------------------------------------------------------


def test_validate_accepts_hand_built_dict_with_correct_log_loss():
    expected = float(sk_log_loss(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.3, 0.6]), labels=[0, 1]))
    metrics = _hand_built_metrics(expected)
    assert validate_metric_dict(metrics) is metrics


def test_validate_accepts_a_large_but_finite_loss():
    # No upper bound: a confident-wrong loss is finite and in-domain.
    metrics = _hand_built_metrics(36.04365338911715)
    assert validate_metric_dict(metrics) is metrics


def test_validate_accepts_zero_loss():
    assert validate_metric_dict(_hand_built_metrics(0.0)) is not None


@pytest.mark.parametrize("bad_value", [-1e-12, -0.5, -100.0])
def test_negative_log_loss_is_rejected(bad_value):
    with pytest.raises(MetricValueError) as excinfo:
        validate_metric_dict(_hand_built_metrics(bad_value))
    assert "log_loss" in str(excinfo.value)


def test_nan_log_loss_is_rejected():
    with pytest.raises(MetricValueError):
        validate_metric_dict(_hand_built_metrics(float("nan")))


def test_positive_inf_log_loss_is_rejected():
    with pytest.raises(MetricValueError):
        validate_metric_dict(_hand_built_metrics(float("inf")))


def test_negative_inf_log_loss_is_rejected():
    with pytest.raises(MetricValueError):
        validate_metric_dict(_hand_built_metrics(float("-inf")))


def test_non_numeric_log_loss_is_rejected():
    with pytest.raises(MetricValueError):
        validate_metric_dict(_hand_built_metrics("low"))


def test_bool_log_loss_is_rejected():
    with pytest.raises(MetricValueError):
        validate_metric_dict(_hand_built_metrics(True))


def test_missing_log_loss_is_rejected():
    metrics = _hand_built_metrics(0.5)
    del metrics["log_loss"]
    with pytest.raises(MissingMetricError) as excinfo:
        validate_metric_dict(metrics)
    assert "log_loss" in str(excinfo.value)


def test_misnamed_log_loss_is_rejected():
    # Renaming log_loss makes the declared key missing AND adds an unknown
    # key; the validator reports the missing declared key first, so the
    # declared base (MetricSchemaError) is what a rename guarantees.
    metrics = _hand_built_metrics(0.5)
    metrics["logloss"] = metrics.pop("log_loss")
    with pytest.raises(MetricSchemaError) as excinfo:
        validate_metric_dict(metrics)
    assert "log_loss" in str(excinfo.value)


def test_unexpected_extra_metric_key_is_rejected_as_misnamed():
    metrics = _hand_built_metrics(0.5)
    metrics["cross_entropy"] = 0.5
    with pytest.raises(UnexpectedMetricError) as excinfo:
        validate_metric_dict(metrics)
    assert "cross_entropy" in str(excinfo.value)


def test_rejection_errors_share_the_declared_base():
    assert issubclass(MissingMetricError, MetricSchemaError)
    assert issubclass(UnexpectedMetricError, MetricSchemaError)
    assert issubclass(MetricValueError, MetricSchemaError)


# ---------------------------------------------------------------------------
# End-to-end: evaluate() returns a dict that validates cleanly
# ---------------------------------------------------------------------------


def test_evaluate_returns_dict_containing_log_loss_that_validates(
    inline_model, inline_split
):
    metrics = evaluate(inline_model, inline_split)
    assert tuple(metrics.keys()) == METRIC_KEYS
    assert "log_loss" in metrics
    assert validate_metric_dict(metrics) is metrics
    value = float(metrics["log_loss"])
    assert np.isfinite(value) and value >= 0.0


def test_evaluate_log_loss_is_discriminative_not_degenerate(inline_model, inline_split):
    """A useful model must beat constant-0.5 guessing on log-loss."""
    metrics = evaluate(inline_model, inline_split)
    labels = inline_split.y_test
    constant = float(sk_log_loss(labels, np.full(labels.size, 0.5), labels=[0, 1]))
    assert float(metrics["log_loss"]) < constant
