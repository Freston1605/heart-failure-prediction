"""The single shared evaluation contract (``evaluate(model, split)``).

Every model in the portfolio — logistic regression, Naive Bayes, kNN, SVM,
Random Forest, XGBoost, MLP — is scored through this one function. The point
is comparability: one prediction path, one metric producer, one schema check.
A model-specific evaluation path is how unfair comparisons and transcription
errors enter a leaderboard, so there is exactly one here.

The contract
------------
:func:`evaluate` accepts

* ``model`` — **any** object exposing ``predict(X)`` and ``predict_proba(X)``
  (fitted scikit-learn estimators, pipelines, and duck-typed classes alike), and
* ``split`` — an :class:`EvaluationSplit` holding the held-out features and
  labels.

and returns the canonical metric dict produced by
:func:`heart.eval.metrics.compute_metric_dict`, after verifying it against
:data:`heart.eval.metrics.METRIC_KEYS`. A missing, misnamed, non-numeric, or
out-of-range metric raises a named subclass of :class:`MetricContractError`
instead of being silently recorded.

Building a split
----------------
:func:`evaluation_split_from_data_split` converts the S01
:class:`~heart.data.split.DataSplit` (train/test row indices) plus the loaded
frame into an :class:`EvaluationSplit`; :func:`evaluation_split_from_frames`
does the same from already-materialised train/test frames. Both resolve the
held-out rows only — the evaluation path never sees training labels.

Observability
-------------
:func:`evaluate` logs the split name, sample count, and primary metric at
``INFO``. :func:`describe_metric_schema` renders the expected keys, and
:func:`flatten_metrics` turns the nested dict into ``key -> float`` pairs for
scalar-only sinks such as MLflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.eval.metrics import (
    CALIBRATION_BIN_KEYS,
    CALIBRATION_KEYS,
    CALIBRATION_KEY,
    CONFUSION_MATRIX_KEYS,
    CONFUSION_MATRIX_KEY,
    DEFAULT_CALIBRATION_BINS,
    METRIC_KEYS,
    METRIC_SCHEMA_VERSION,
    PRIMARY_METRIC,
    RATE_METRIC_KEYS,
    SCALAR_METRIC_KEYS,
    STRUCTURED_METRIC_KEYS,
    compute_metric_dict,
)

logger = logging.getLogger(__name__)

__all__ = [
    "METRIC_KEYS",
    "METRIC_SCHEMA_VERSION",
    "PRIMARY_METRIC",
    "SCALAR_METRIC_KEYS",
    "STRUCTURED_METRIC_KEYS",
    "MetricContractError",
    "MetricSchemaError",
    "MissingMetricError",
    "UnexpectedMetricError",
    "MetricValueError",
    "SplitContractError",
    "ModelInterfaceError",
    "MissingPredictError",
    "MissingPredictProbaError",
    "PredictionShapeError",
    "EvaluationSplit",
    "evaluation_split_from_data_split",
    "evaluation_split_from_frames",
    "validate_metric_dict",
    "describe_metric_schema",
    "flatten_metrics",
    "evaluate",
]


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class MetricContractError(Exception):
    """Base class for every metric-contract or model-interface violation."""


class MetricSchemaError(MetricContractError):
    """The metric dict does not match the declared schema."""


class MissingMetricError(MetricSchemaError):
    """A required metric key is absent from the dict."""


class UnexpectedMetricError(MetricSchemaError):
    """The dict carries a key that is not part of the declared schema."""


class MetricValueError(MetricSchemaError):
    """A metric value is non-numeric, non-finite, or outside its domain."""


class SplitContractError(MetricContractError):
    """The evaluation split is empty or malformed."""


class ModelInterfaceError(MetricContractError):
    """The model does not satisfy the predict/predict_proba interface."""


class MissingPredictError(ModelInterfaceError):
    """The model exposes no callable ``predict``."""


class MissingPredictProbaError(ModelInterfaceError):
    """The model exposes no callable ``predict_proba``."""


class PredictionShapeError(ModelInterfaceError):
    """``predict_proba`` did not return a two-column probability matrix."""


# ---------------------------------------------------------------------------
# The split abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSplit:
    """Held-out features and labels handed to :func:`evaluate`.

    ``X_test`` is a feature frame and ``y_test`` is the aligned binary label
    vector. The split deliberately carries no training data: evaluation must
    never need it.
    """

    X_test: pd.DataFrame
    y_test: np.ndarray
    name: str = "test"

    def __post_init__(self) -> None:
        if not isinstance(self.X_test, pd.DataFrame):
            raise SplitContractError(
                f"X_test must be a pandas.DataFrame, got "
                f"{type(self.X_test).__name__}."
            )
        if self.X_test.empty:
            raise SplitContractError(
                "X_test is empty; evaluation needs at least one held-out row."
            )
        labels = np.asarray(self.y_test)
        if labels.ndim != 1:
            raise SplitContractError(
                f"y_test must be one-dimensional, got shape {labels.shape}."
            )
        if labels.size != len(self.X_test):
            raise SplitContractError(
                f"X_test has {len(self.X_test)} rows but y_test has "
                f"{labels.size} label(s); they must align."
            )
        object.__setattr__(self, "y_test", labels)

    @property
    def n_samples(self) -> int:
        return int(self.y_test.size)

    def class_balance(self) -> dict[str, int]:
        values, counts = np.unique(self.y_test, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(values, counts)}

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "class_balance": self.class_balance(),
        }


def _feature_frame(
    frame: pd.DataFrame, feature_columns: tuple[str, ...]
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise SplitContractError(
            f"Frame is missing feature column(s) {missing}; present columns are "
            f"{list(frame.columns)}."
        )
    return frame[list(feature_columns)].copy()


def evaluation_split_from_data_split(
    data_split: object,
    frame: pd.DataFrame,
    *,
    name: str | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> EvaluationSplit:
    """Resolve the held-out rows of an S01 ``DataSplit`` into an eval split.

    ``data_split`` is anything exposing ``test_frame(frame)`` — the S01
    :class:`~heart.data.split.DataSplit` and its duck-typed equivalents. The
    returned split contains the test features and labels only.
    """
    test_frame = data_split.test_frame(frame)
    if target_column not in test_frame.columns:
        raise SplitContractError(
            f"Held-out frame has no target column {target_column!r}; cannot "
            "evaluate without labels."
        )
    features = _feature_frame(test_frame, feature_columns)
    labels = test_frame[target_column].to_numpy()
    resolved_name = name or f"{getattr(data_split, 'version', 'split')}/test"
    return EvaluationSplit(X_test=features, y_test=labels, name=resolved_name)


def evaluation_split_from_frames(
    test_frame: pd.DataFrame,
    *,
    name: str = "test",
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> EvaluationSplit:
    """Build an :class:`EvaluationSplit` directly from a held-out frame."""
    if not isinstance(test_frame, pd.DataFrame):
        raise SplitContractError(
            f"test_frame must be a pandas.DataFrame, got "
            f"{type(test_frame).__name__}."
        )
    if target_column not in test_frame.columns:
        raise SplitContractError(
            f"Held-out frame has no target column {target_column!r}; cannot "
            "evaluate without labels."
        )
    features = _feature_frame(test_frame, feature_columns)
    labels = test_frame[target_column].to_numpy()
    return EvaluationSplit(X_test=features, y_test=labels, name=name)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _require_exact_keys(
    mapping: object, expected: tuple[str, ...], *, context: str
) -> dict:
    if not isinstance(mapping, dict):
        raise MetricValueError(
            f"{context} must be a dict, got {type(mapping).__name__}."
        )
    missing = [key for key in expected if key not in mapping]
    if missing:
        raise MissingMetricError(
            f"{context} is missing key(s) {missing}; expected exactly "
            f"{list(expected)}."
        )
    unexpected = [key for key in mapping if key not in expected]
    if unexpected:
        raise UnexpectedMetricError(
            f"{context} has unexpected key(s) {unexpected}; expected exactly "
            f"{list(expected)}. A misnamed metric is a schema violation, not a "
            "new field."
        )
    return mapping


def _validate_rate(value: object, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise MetricValueError(
            f"Metric {key!r} must be a number, got {type(value).__name__} "
            f"({value!r})."
        )
    number = float(value)
    if not np.isfinite(number):
        raise MetricValueError(f"Metric {key!r} is non-finite ({value!r}).")
    if number < -1e-9 or number > 1.0 + 1e-9:
        raise MetricValueError(
            f"Metric {key!r} = {number!r} is outside [0, 1]; rate metrics must "
            "be probabilities."
        )
    return number


def _validate_count(value: object, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MetricValueError(
            f"Confusion-matrix field {key!r} must be an integer, got "
            f"{type(value).__name__} ({value!r})."
        )
    number = int(value)
    if number < 0:
        raise MetricValueError(
            f"Confusion-matrix field {key!r} = {number} cannot be negative."
        )
    return number


def validate_metric_dict(metrics: object) -> dict[str, object]:
    """Verify ``metrics`` matches the declared schema exactly, or raise.

    Checks, in order:

    1. top-level keys are exactly :data:`METRIC_KEYS` (missing -> 
       :class:`MissingMetricError`; extra/misnamed ->
       :class:`UnexpectedMetricError`);
    2. every scalar rate metric is a finite number in ``[0, 1]``;
    3. ``confusion_matrix`` has exactly its declared integer key set;
    4. ``calibration`` has exactly its declared key set, a non-negative brier
       score and ECE in ``[0, 1]``, a positive integer ``n_bins``, and a
       ``bins`` list of exactly that length whose entries have the declared
       keys and counts that sum to ``n_samples``.

    Returns the validated dict so callers can use it inline.
    """
    mapping = _require_exact_keys(metrics, METRIC_KEYS, context="metric dict")

    for key in RATE_METRIC_KEYS:
        _validate_rate(mapping[key], key=key)

    matrix = _require_exact_keys(
        mapping[CONFUSION_MATRIX_KEY],
        CONFUSION_MATRIX_KEYS,
        context=CONFUSION_MATRIX_KEY,
    )
    for key in CONFUSION_MATRIX_KEYS:
        _validate_count(matrix[key], key=key)
    if matrix["n_samples"] != (
        matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"]
    ):
        raise MetricValueError(
            f"{CONFUSION_MATRIX_KEY}.n_samples = {matrix['n_samples']} does not "
            "equal tn + fp + fn + tp."
        )
    if matrix["n_positive"] != matrix["tp"] + matrix["fn"]:
        raise MetricValueError(
            f"{CONFUSION_MATRIX_KEY}.n_positive does not equal tp + fn."
        )
    if matrix["n_negative"] != matrix["tn"] + matrix["fp"]:
        raise MetricValueError(
            f"{CONFUSION_MATRIX_KEY}.n_negative does not equal tn + fp."
        )

    calibration = _require_exact_keys(
        mapping[CALIBRATION_KEY], CALIBRATION_KEYS, context=CALIBRATION_KEY
    )
    brier = calibration["brier_score"]
    if isinstance(brier, bool) or not isinstance(brier, (int, float, np.floating)):
        raise MetricValueError(
            f"{CALIBRATION_KEY}.brier_score must be a number, got "
            f"{type(brier).__name__}."
        )
    if not np.isfinite(float(brier)) or not -1e-9 <= float(brier) <= 1.0 + 1e-9:
        raise MetricValueError(
            f"{CALIBRATION_KEY}.brier_score = {brier!r} is not a finite value "
            "in [0, 1]."
        )
    ece = calibration["expected_calibration_error"]
    if isinstance(ece, bool) or not isinstance(ece, (int, float, np.floating)):
        raise MetricValueError(
            f"{CALIBRATION_KEY}.expected_calibration_error must be a number, got "
            f"{type(ece).__name__}."
        )
    if not np.isfinite(float(ece)) or not -1e-9 <= float(ece) <= 1.0 + 1e-9:
        raise MetricValueError(
            f"{CALIBRATION_KEY}.expected_calibration_error = {ece!r} is not a "
            "finite value in [0, 1]."
        )
    n_bins = calibration["n_bins"]
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise MetricValueError(
            f"{CALIBRATION_KEY}.n_bins must be a positive integer, got {n_bins!r}."
        )
    bins = calibration["bins"]
    if not isinstance(bins, (list, tuple)):
        raise MetricValueError(
            f"{CALIBRATION_KEY}.bins must be a list, got {type(bins).__name__}."
        )
    if len(bins) != int(n_bins):
        raise MetricValueError(
            f"{CALIBRATION_KEY}.bins has {len(bins)} entries but n_bins is "
            f"{n_bins}; the reliability curve must be exactly n_bins long."
        )
    bin_total = 0
    for index, bucket in enumerate(bins):
        entry = _require_exact_keys(
            bucket, CALIBRATION_BIN_KEYS, context=f"{CALIBRATION_KEY}.bins[{index}]"
        )
        count = _validate_count(
            entry["count"], key=f"{CALIBRATION_KEY}.bins[{index}].count"
        )
        bin_total += count
        for field in ("mean_predicted", "fraction_positive", "gap"):
            value = entry[field]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.floating)
            ):
                raise MetricValueError(
                    f"{CALIBRATION_KEY}.bins[{index}].{field} must be a number "
                    f"or None, got {type(value).__name__}."
                )
            if not np.isfinite(float(value)):
                raise MetricValueError(
                    f"{CALIBRATION_KEY}.bins[{index}].{field} is non-finite."
                )
        if count == 0 and (entry["mean_predicted"] is not None):
            raise MetricValueError(
                f"{CALIBRATION_KEY}.bins[{index}] is empty but has a "
                "mean_predicted value."
            )
    if bin_total != matrix["n_samples"]:
        raise MetricValueError(
            f"Calibration bins sum to {bin_total} samples but the confusion "
            f"matrix counts {matrix['n_samples']}."
        )

    return mapping  # type: ignore[return-value]


def describe_metric_schema() -> str:
    """Render the declared metric schema as a human-readable report."""
    lines = [
        f"metric schema version: {METRIC_SCHEMA_VERSION}",
        f"primary metric: {PRIMARY_METRIC}",
        f"top-level keys ({len(METRIC_KEYS)}): {', '.join(METRIC_KEYS)}",
        f"scalar metrics: {', '.join(SCALAR_METRIC_KEYS)}",
        f"{CONFUSION_MATRIX_KEY}: {', '.join(CONFUSION_MATRIX_KEYS)}",
        f"{CALIBRATION_KEY}: {', '.join(CALIBRATION_KEYS)}",
        f"  bin: {', '.join(CALIBRATION_BIN_KEYS)}",
    ]
    return "\n".join(lines)


def flatten_metrics(
    metrics: dict[str, object], *, prefix: str = ""
) -> dict[str, float]:
    """Flatten a metric dict into ``dotted.key -> float`` scalar pairs.

    Nested confusion-matrix fields, calibration scalars, and per-bin fields are
    expanded (for example ``confusion_matrix.tp`` and
    ``calibration.bins.0.count``). ``None`` placeholders from empty calibration
    bins are skipped because they carry no numeric value.
    """
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(
                        flatten_metrics(item, prefix=f"{name}.{index}.")
                    )
                elif item is not None:
                    flat[f"{name}.{index}"] = float(item)
        elif value is None:
            continue
        else:
            flat[name] = float(value)
    return flat


# ---------------------------------------------------------------------------
# Prediction and evaluation
# ---------------------------------------------------------------------------


def _predict(model: object, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_pred, y_proba)`` from a fitted model, or raise loudly."""
    if not hasattr(model, "predict") or not callable(model.predict):
        raise MissingPredictError(
            f"Model {type(model).__name__} exposes no callable predict(); the "
            "evaluation contract requires predict(X)."
        )
    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        raise MissingPredictProbaError(
            f"Model {type(model).__name__} exposes no callable predict_proba(); "
            "the contract requires calibrated probabilities, not just hard "
            "labels. Wrap or replace the model with one that yields P(y=1|x)."
        )

    predictions = np.asarray(model.predict(features)).ravel()
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise PredictionShapeError(
            "predict_proba must return an (n_samples, 2) matrix of class "
            f"probabilities; got shape {probabilities.shape}. For multi-class "
            "models, reduce to the positive class before evaluating."
        )
    positive_scores = probabilities[:, 1]
    if predictions.size != len(features) or positive_scores.size != len(features):
        raise PredictionShapeError(
            f"predict produced {predictions.size} and predict_proba produced "
            f"{positive_scores.size} rows for {len(features)} samples."
        )
    return predictions, positive_scores


def evaluate(
    model: object,
    split: EvaluationSplit,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    validate: bool = True,
) -> dict[str, object]:
    """Evaluate any ``predict``/``predict_proba`` model on a held-out split.

    Parameters
    ----------
    model:
        Any fitted object exposing ``predict(X)`` and ``predict_proba(X)``.
    split:
        The :class:`EvaluationSplit` to score (held-out rows only).
    n_bins:
        Number of equal-width bins in the reliability curve.
    validate:
        When ``True`` (default) the returned dict is schema-checked and a
        malformed metric raises. Leave this on everywhere except tests that
        deliberately exercise the validator.

    Returns
    -------
    dict
        The canonical metric dict, keyed exactly by
        :data:`heart.eval.metrics.METRIC_KEYS`.

    Raises
    ------
    MetricContractError
        Any named subclass: :class:`SplitContractError`,
        :class:`ModelInterfaceError`, :class:`MetricSchemaError`.
    """
    if not isinstance(split, EvaluationSplit):
        raise SplitContractError(
            f"split must be an EvaluationSplit, got {type(split).__name__}. "
            "Build one with evaluation_split_from_data_split() or "
            "evaluation_split_from_frames()."
        )

    predictions, probabilities = _predict(model, split.X_test)
    metrics = compute_metric_dict(
        split.y_test, probabilities, predictions, n_bins=n_bins
    )
    if validate:
        validate_metric_dict(metrics)

    logger.info(
        "Evaluated %s model on %s split (n=%d): %s=%.4f, accuracy=%.4f",
        type(model).__name__,
        split.name,
        split.n_samples,
        PRIMARY_METRIC,
        float(metrics[PRIMARY_METRIC]),
        float(metrics["accuracy"]),
    )
    return metrics
