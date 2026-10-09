"""Tests for the shared evaluation contract (S02/T01).

The contract is two-part:

1. :func:`heart.eval.contract.evaluate` is the single evaluation path. Any
   object with ``predict``/``predict_proba`` works through it, and it returns
   the canonical metric dict keyed exactly by ``METRIC_KEYS``.
2. The returned dict is **schema-checked**: a missing, misnamed, non-numeric,
   or out-of-range metric fails loudly instead of being logged.

The model-agnostic requirement is exercised with two structurally different
scikit-learn estimators (a linear model and a tree) plus a hand-rolled
duck-typed class that implements only ``predict``/``predict_proba``. The
negative tests below assert each failure path raises its named exception.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from heart.config import RANDOM_SEED
from heart.data.load import load_dataset
from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import create_split
from heart.eval.contract import (
    EvaluationSplit,
    MetricSchemaError,
    MetricValueError,
    MissingMetricError,
    MissingPredictError,
    MissingPredictProbaError,
    ModelInterfaceError,
    PredictionShapeError,
    SplitContractError,
    UnexpectedMetricError,
    describe_metric_schema,
    evaluate,
    evaluation_split_from_data_split,
    evaluation_split_from_frames,
    flatten_metrics,
    validate_metric_dict,
)
from heart.eval.metrics import (
    CALIBRATION_BIN_KEYS,
    CALIBRATION_KEYS,
    CALIBRATION_KEY,
    CONFUSION_MATRIX_KEYS,
    CONFUSION_MATRIX_KEY,
    METRIC_KEYS,
    PRIMARY_METRIC,
    CalibrationConfigError,
    EmptyEvaluationError,
    InvalidLabelError,
    InvalidProbabilityError,
    SingleClassError,
    calibration_report,
    compute_metric_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _inline_dataset(n: int = 400, *, seed: int = 11):
    """A small, self-contained binary classification problem."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, 3))
    logit = 1.6 * matrix[:, 0] - 1.1 * matrix[:, 1] + 0.4 * matrix[:, 2]
    labels = (rng.random(n) < expit(logit)).astype(int)
    order = rng.permutation(n)
    cut = int(0.7 * n)
    train_idx, test_idx = order[:cut], order[cut:]
    columns = ["a", "b", "c"]
    X_train = pd.DataFrame(matrix[train_idx], columns=columns)
    X_test = pd.DataFrame(matrix[test_idx], columns=columns)
    split = EvaluationSplit(
        X_test=X_test, y_test=labels[test_idx], name="inline/test"
    )
    return X_train, labels[train_idx], split


@pytest.fixture(scope="module")
def inline():
    return _inline_dataset()


@pytest.fixture(scope="module")
def inline_split(inline):
    return inline[2]


@pytest.fixture(scope="module")
def logreg_model(inline):
    X_train, y_train, _ = inline
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def tree_model(inline):
    X_train, y_train, _ = inline
    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


class DuckModel:
    """A model that is *not* a scikit-learn estimator, only the contract."""

    def __init__(self, coefficients, intercept):
        self._coefficients = np.asarray(coefficients, dtype=float).ravel()
        self._intercept = float(intercept)

    def predict_proba(self, X):
        scores = X.to_numpy() @ self._coefficients + self._intercept
        positive = expit(scores)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@pytest.fixture(scope="module")
def duck_model(logreg_model):
    return DuckModel(logreg_model.coef_[0], logreg_model.intercept_[0])


@pytest.fixture(scope="module")
def s01_eval_split():
    dataset = load_dataset()
    data_split = create_split(dataset.frame)
    return (
        data_split,
        dataset.frame,
        evaluation_split_from_data_split(data_split, dataset.frame),
    )


@pytest.fixture(scope="module")
def s01_pipeline_model(s01_eval_split):
    data_split, frame, _ = s01_eval_split
    train = data_split.train_frame(frame)
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        ]
    )
    model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    return model


# ---------------------------------------------------------------------------
# Canonical dict shape and model-agnosticism
# ---------------------------------------------------------------------------


def test_evaluate_returns_exactly_the_declared_keys(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    assert tuple(metrics.keys()) == METRIC_KEYS
    assert set(metrics.keys()) == set(METRIC_KEYS)


def test_two_structurally_different_models_produce_identical_key_sets(
    logreg_model, tree_model, inline_split
):
    linear = evaluate(logreg_model, inline_split)
    tree = evaluate(tree_model, inline_split)
    assert set(linear.keys()) == set(tree.keys()) == set(METRIC_KEYS)
    assert tuple(linear.keys()) == tuple(tree.keys())
    for metrics in (linear, tree):
        validate_metric_dict(metrics)


def test_duck_typed_model_satisfies_the_contract(duck_model, inline_split):
    metrics = evaluate(duck_model, inline_split)
    assert set(metrics.keys()) == set(METRIC_KEYS)
    validate_metric_dict(metrics)


def test_primary_metric_is_roc_auc_and_present(logreg_model, inline_split):
    assert PRIMARY_METRIC == "roc_auc"
    metrics = evaluate(logreg_model, inline_split)
    assert isinstance(metrics[PRIMARY_METRIC], float)
    assert 0.0 <= metrics[PRIMARY_METRIC] <= 1.0


def test_evaluate_is_deterministic(logreg_model, inline_split):
    first = evaluate(logreg_model, inline_split)
    second = evaluate(logreg_model, inline_split)
    assert first == second


# ---------------------------------------------------------------------------
# Schema validation: missing / misnamed / malformed metrics
# ---------------------------------------------------------------------------


def test_valid_metric_dict_passes_validation(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    assert validate_metric_dict(metrics) is metrics


def test_missing_metric_key_raises(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    del metrics["roc_auc"]
    with pytest.raises(MissingMetricError) as excinfo:
        validate_metric_dict(metrics)
    assert "roc_auc" in str(excinfo.value)


def test_misnamed_metric_raises(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    metrics["auroc"] = metrics.pop("roc_auc")
    with pytest.raises(MetricSchemaError):
        validate_metric_dict(metrics)


def test_unexpected_metric_key_raises(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    metrics["made_up_metric"] = 0.5
    with pytest.raises(UnexpectedMetricError) as excinfo:
        validate_metric_dict(metrics)
    assert "made_up_metric" in str(excinfo.value)


def test_non_numeric_metric_raises(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    metrics["accuracy"] = "high"
    with pytest.raises(MetricValueError):
        validate_metric_dict(metrics)


def test_non_finite_metric_raises(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    metrics["accuracy"] = float("nan")
    with pytest.raises(MetricValueError):
        validate_metric_dict(metrics)


def test_out_of_range_metric_raises(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    metrics["accuracy"] = 1.5
    with pytest.raises(MetricValueError):
        validate_metric_dict(metrics)


def test_confusion_matrix_inconsistent_margins_raise(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    metrics[CONFUSION_MATRIX_KEY]["n_samples"] += 1
    with pytest.raises(MetricValueError):
        validate_metric_dict(metrics)


def test_calibration_bin_count_mismatch_raises(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    metrics[CALIBRATION_KEY]["bins"] = metrics[CALIBRATION_KEY]["bins"][:3]
    with pytest.raises(MetricValueError):
        validate_metric_dict(metrics)


# ---------------------------------------------------------------------------
# Model interface failures
# ---------------------------------------------------------------------------


class _OnlyPredict:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _OnlyProba:
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


class _OneColumnProba:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.full((len(X), 1), 0.5)


class _MultiClassProba:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.full((len(X), 3), 1.0 / 3.0)


def test_model_without_predict_proba_raises(inline_split):
    with pytest.raises(MissingPredictProbaError):
        evaluate(_OnlyPredict(), inline_split)


def test_model_without_predict_raises(inline_split):
    with pytest.raises(MissingPredictError):
        evaluate(_OnlyProba(), inline_split)


def test_one_column_predict_proba_raises(inline_split):
    with pytest.raises(PredictionShapeError):
        evaluate(_OneColumnProba(), inline_split)


def test_multiclass_predict_proba_raises(inline_split):
    with pytest.raises(PredictionShapeError):
        evaluate(_MultiClassProba(), inline_split)


def test_model_interface_errors_share_a_base(inline_split):
    assert issubclass(MissingPredictError, ModelInterfaceError)
    assert issubclass(MissingPredictProbaError, ModelInterfaceError)
    assert issubclass(PredictionShapeError, ModelInterfaceError)


# ---------------------------------------------------------------------------
# Split contract failures
# ---------------------------------------------------------------------------


def test_evaluate_rejects_a_non_split(logreg_model):
    with pytest.raises(SplitContractError):
        evaluate(logreg_model, {"X": 1})


def test_empty_split_raises():
    with pytest.raises(SplitContractError):
        EvaluationSplit(X_test=pd.DataFrame({"a": []}), y_test=np.array([]))


def test_misaligned_split_raises():
    with pytest.raises(SplitContractError):
        EvaluationSplit(
            X_test=pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
            y_test=np.array([0, 1]),
        )


def test_non_dataframe_split_raises():
    with pytest.raises(SplitContractError):
        EvaluationSplit(X_test=np.zeros((3, 2)), y_test=np.array([0, 1, 1]))


def test_split_from_frame_requires_target():
    frame = pd.DataFrame({column: [1.0, 2.0] for column in FEATURE_COLUMNS})
    with pytest.raises(SplitContractError):
        evaluation_split_from_frames(frame)


def test_split_from_frame_selects_declared_features():
    values = np.arange(4, dtype=float)
    frame = pd.DataFrame({column: values for column in FEATURE_COLUMNS})
    frame[TARGET_COLUMN] = np.array([0, 1, 0, 1])
    frame["irrelevant_extra_column"] = 99.0
    rebuilt = evaluation_split_from_frames(frame)
    assert list(rebuilt.X_test.columns) == list(FEATURE_COLUMNS)
    np.testing.assert_array_equal(rebuilt.y_test, np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# Label / probability computation failures
# ---------------------------------------------------------------------------


def test_single_class_labels_raise():
    with pytest.raises(SingleClassError):
        compute_metric_dict(
            np.ones(10, dtype=int), np.full(10, 0.9), np.ones(10, dtype=int)
        )


def test_non_binary_labels_raise():
    with pytest.raises(InvalidLabelError):
        compute_metric_dict(
            np.array([0, 1, 2, 0, 1]),
            np.array([0.1, 0.9, 0.4, 0.2, 0.8]),
            np.array([0, 1, 0, 0, 1]),
        )


def test_empty_labels_raise():
    with pytest.raises(EmptyEvaluationError):
        compute_metric_dict(np.array([]), np.array([]), np.array([]))


def test_probability_above_one_raises():
    with pytest.raises(InvalidProbabilityError):
        compute_metric_dict(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 1.7, 0.4, 0.8]),
            np.array([0, 1, 0, 1]),
        )


def test_non_finite_probability_raises():
    with pytest.raises(InvalidProbabilityError):
        compute_metric_dict(
            np.array([0, 1, 0, 1]),
            np.array([0.1, float("nan"), 0.4, 0.8]),
            np.array([0, 1, 0, 1]),
        )


def test_misaligned_probabilities_raise():
    with pytest.raises(InvalidProbabilityError):
        compute_metric_dict(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9]),
            np.array([0, 1, 0, 1]),
        )


def test_invalid_calibration_bins_raise():
    with pytest.raises(CalibrationConfigError):
        calibration_report(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.8]), n_bins=0)


def test_evaluate_propagates_invalid_bin_config(logreg_model, inline_split):
    with pytest.raises(CalibrationConfigError):
        evaluate(logreg_model, inline_split, n_bins=0)


# ---------------------------------------------------------------------------
# Metric arithmetic
# ---------------------------------------------------------------------------


def test_metrics_match_independent_reference(logreg_model, inline_split):
    labels = inline_split.y_test
    predictions = np.asarray(logreg_model.predict(inline_split.X_test)).ravel()
    probabilities = logreg_model.predict_proba(inline_split.X_test)[:, 1]
    metrics = evaluate(logreg_model, inline_split)

    assert metrics["accuracy"] == pytest.approx(np.mean(predictions == labels))
    assert metrics["prevalence"] == pytest.approx(np.mean(labels == 1))
    assert metrics["confusion_matrix"]["n_samples"] == labels.size
    assert (
        metrics["confusion_matrix"]["tn"] + metrics["confusion_matrix"]["tp"]
        == int(np.sum(predictions == labels))
    )
    assert metrics["brier_score"] == pytest.approx(
        float(np.mean((probabilities - labels) ** 2))
    )


def test_confusion_matrix_partitions_all_samples(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    matrix = metrics[CONFUSION_MATRIX_KEY]
    assert set(matrix.keys()) == set(CONFUSION_MATRIX_KEYS)
    assert matrix["n_samples"] == inline_split.n_samples
    assert matrix["n_positive"] + matrix["n_negative"] == inline_split.n_samples
    assert matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"] == inline_split.n_samples


def test_calibration_bins_partition_samples_and_match_brier(
    logreg_model, inline_split
):
    metrics = evaluate(logreg_model, inline_split, n_bins=5)
    calibration = metrics[CALIBRATION_KEY]
    assert set(calibration.keys()) == set(CALIBRATION_KEYS)
    assert calibration["n_bins"] == 5
    assert len(calibration["bins"]) == 5
    assert calibration["brier_score"] == pytest.approx(metrics["brier_score"])
    total = sum(bucket["count"] for bucket in calibration["bins"])
    assert total == inline_split.n_samples
    for bucket in calibration["bins"]:
        assert set(bucket.keys()) == set(CALIBRATION_BIN_KEYS)


def test_empty_calibration_bins_use_none_placeholders():
    # All probabilities land in the first bin, so later bins are empty.
    report = calibration_report(
        np.array([0, 0, 1, 1]), np.array([0.05, 0.1, 0.15, 0.2]), n_bins=10
    )
    empty = report.bins[5]
    assert empty.count == 0
    assert empty.mean_predicted is None
    assert empty.fraction_positive is None
    assert empty.gap is None


def test_specificity_and_npv_are_defined(logreg_model, inline_split):
    metrics = evaluate(logreg_model, inline_split)
    matrix = metrics[CONFUSION_MATRIX_KEY]
    if matrix["n_negative"]:
        assert metrics["specificity"] == pytest.approx(
            matrix["tn"] / matrix["n_negative"]
        )
    if matrix["tn"] + matrix["fn"]:
        assert metrics["npv"] == pytest.approx(matrix["tn"] / (matrix["tn"] + matrix["fn"]))


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------


def test_describe_metric_schema_lists_every_key():
    report = describe_metric_schema()
    for key in METRIC_KEYS:
        assert key in report
    assert PRIMARY_METRIC in report


def test_flatten_metrics_emits_scalar_leaves(logreg_model, inline_split):
    flat = flatten_metrics(evaluate(logreg_model, inline_split))
    assert all(isinstance(value, float) for value in flat.values())
    assert "roc_auc" in flat
    assert flat["confusion_matrix.tp"] >= 0.0
    assert "calibration.expected_calibration_error" in flat
    assert "calibration.bins.0.count" in flat
    # None placeholders from empty bins must not become floats.
    assert all(value == value for value in flat.values())


def test_split_contract_error_is_named(inline_split):
    assert issubclass(MissingMetricError, MetricSchemaError)
    assert issubclass(UnexpectedMetricError, MetricSchemaError)
    assert issubclass(MetricValueError, MetricSchemaError)


# ---------------------------------------------------------------------------
# Integration with the real S01 split artifacts
# ---------------------------------------------------------------------------


def test_evaluate_on_real_s01_split(s01_pipeline_model, s01_eval_split):
    data_split, _, split = s01_eval_split
    metrics = evaluate(s01_pipeline_model, split)
    assert set(metrics.keys()) == set(METRIC_KEYS)
    validate_metric_dict(metrics)
    assert split.n_samples == data_split.test_rows
    assert metrics[CONFUSION_MATRIX_KEY]["n_samples"] == data_split.test_rows
    assert 0.0 <= metrics[PRIMARY_METRIC] <= 1.0
    assert metrics["prevalence"] == pytest.approx(
        split.class_balance()["1"] / split.n_samples
    )


def test_s01_split_contains_no_training_rows(s01_eval_split):
    data_split, frame, split = s01_eval_split
    test_indices = set(int(i) for i in data_split.test_indices)
    train_indices = set(int(i) for i in data_split.train_indices)
    assert test_indices.isdisjoint(train_indices)
    assert split.n_samples == len(test_indices)


def test_evaluate_does_not_mutate_the_split(logreg_model, inline_split):
    before_X = inline_split.X_test.copy(deep=True)
    before_y = inline_split.y_test.copy()
    evaluate(logreg_model, inline_split)
    pd.testing.assert_frame_equal(inline_split.X_test, before_X)
    np.testing.assert_array_equal(inline_split.y_test, before_y)


def test_schema_version_is_declared():
    from heart.eval.metrics import METRIC_SCHEMA_VERSION

    assert METRIC_SCHEMA_VERSION == "1"
    assert isinstance(copy.deepcopy(METRIC_KEYS), tuple)
