"""Canonical metric computation for the shared evaluation contract.

This module owns the **shape** and the **arithmetic** of every metric the
portfolio reports. :func:`compute_metric_dict` is the single producer of the
metric dict consumed by :mod:`heart.eval.contract`; nothing else is allowed to
assemble metrics by hand. Keeping production here means a metric can only be
added or renamed in one place, so every later model (logistic regression,
tree, XGBoost, MLP) is scored and logged identically.

Canonical metric dict
---------------------
The returned dict has **exactly** these top-level keys (see
:data:`METRIC_KEYS`)::

    accuracy          float in [0, 1]
    precision         float in [0, 1]   (positive class = 1)
    recall            float in [0, 1]
    f1                float in [0, 1]
    roc_auc           float in [0, 1]   (PRIMARY_METRIC; R007)
    pr_auc            float in [0, 1]   (average precision)
    specificity       float in [0, 1]   (true-negative rate)
    npv               float in [0, 1]   (negative predictive value)
    prevalence        float in [0, 1]   (observed positive rate)
    brier_score       float in [0, 1]   (probability calibration error)
    confusion_matrix  dict with tn/fp/fn/tp/n_samples/n_positive/n_negative
    calibration       dict with brier_score/expected_calibration_error/n_bins/bins

``calibration.bins`` is a fixed-length (``n_bins``) equal-width reliability
curve: each bin records its count, mean predicted probability, observed
positive fraction, and the gap between them. Empty bins carry ``None`` for the
three undefined fields so the bin count never changes with the data.

Named exceptions
----------------
Every invalid input raises a self-describing subclass of
:class:`MetricComputationError`: :class:`EmptyEvaluationError`,
:class:`InvalidLabelError`, :class:`SingleClassError`,
:class:`InvalidProbabilityError`, and :class:`CalibrationConfigError`. None of
these are silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared constants (the canonical schema)
# ---------------------------------------------------------------------------

#: Schema version. Bump when a key is added, removed, or renamed.
METRIC_SCHEMA_VERSION: str = "1"

#: Positive class label for this binary prediction task.
POSITIVE_LABEL: int = 1

#: Negative class label.
NEGATIVE_LABEL: int = 0

#: Primary model-selection metric (R007): ROC-AUC.
PRIMARY_METRIC: str = "roc_auc"

#: Default number of equal-width bins in the reliability curve.
DEFAULT_CALIBRATION_BINS: int = 10

#: Scalar metrics that must be finite floats in [0, 1].
RATE_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "specificity",
    "npv",
    "prevalence",
    "brier_score",
)

#: Every scalar (MLflow-loggable) metric key, in canonical order.
SCALAR_METRIC_KEYS: tuple[str, ...] = RATE_METRIC_KEYS

#: Top-level key holding the confusion-matrix breakdown.
CONFUSION_MATRIX_KEY: str = "confusion_matrix"

#: Top-level key holding the calibration / reliability data.
CALIBRATION_KEY: str = "calibration"

#: Structured (non-scalar) top-level keys.
STRUCTURED_METRIC_KEYS: tuple[str, ...] = (CONFUSION_MATRIX_KEY, CALIBRATION_KEY)

#: Top-level keys of the confusion-matrix breakdown.
CONFUSION_MATRIX_KEYS: tuple[str, ...] = (
    "tn",
    "fp",
    "fn",
    "tp",
    "n_samples",
    "n_positive",
    "n_negative",
)

#: Top-level keys of the calibration block.
CALIBRATION_KEYS: tuple[str, ...] = (
    "brier_score",
    "expected_calibration_error",
    "n_bins",
    "bins",
)

#: Keys of one reliability-curve bin.
CALIBRATION_BIN_KEYS: tuple[str, ...] = (
    "bin",
    "lower",
    "upper",
    "count",
    "mean_predicted",
    "fraction_positive",
    "gap",
)

#: The complete, exact set of top-level keys in a metric dict.
METRIC_KEYS: tuple[str, ...] = SCALAR_METRIC_KEYS + STRUCTURED_METRIC_KEYS


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class MetricComputationError(Exception):
    """Base class for every metric-computation failure."""


class EmptyEvaluationError(MetricComputationError):
    """The evaluation set contains no rows, so no metric is defined."""


class InvalidLabelError(MetricComputationError):
    """The label vector is not a one-dimensional binary 0/1 vector."""


class SingleClassError(MetricComputationError):
    """The label vector has only one class; ROC-AUC is undefined."""


class InvalidProbabilityError(MetricComputationError):
    """A probability vector is malformed or outside [0, 1]."""


class CalibrationConfigError(MetricComputationError):
    """The requested calibration bin count is invalid."""


# ---------------------------------------------------------------------------
# Input coercion / validation
# ---------------------------------------------------------------------------


def as_label_vector(y_true: object) -> np.ndarray:
    """Coerce ``y_true`` to a validated, one-dimensional binary 0/1 array.

    Raises :class:`EmptyEvaluationError` for an empty vector,
    :class:`InvalidLabelError` for non-numeric or non-binary values, and
    :class:`SingleClassError` when only one class is present (ROC-AUC and
    PR-AUC are undefined without both classes).
    """
    array = np.asarray(y_true)
    if array.ndim != 1:
        raise InvalidLabelError(
            f"y_true must be one-dimensional, got shape {array.shape}."
        )
    if array.size == 0:
        raise EmptyEvaluationError(
            "y_true is empty; evaluation needs at least one labelled row."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidLabelError(
            f"y_true has dtype {array.dtype!r}; expected a numeric binary 0/1 "
            "vector."
        )
    labels = np.unique(array)
    unexpected = sorted(set(labels.tolist()) - {NEGATIVE_LABEL, POSITIVE_LABEL})
    if unexpected:
        raise InvalidLabelError(
            f"y_true contains non-binary label(s) {unexpected}; this contract "
            f"supports only {NEGATIVE_LABEL}/POSITIVE_LABEL (negative/positive)."
        )
    if labels.size < 2:
        raise SingleClassError(
            f"y_true contains only label {int(labels[0])}; ROC-AUC and PR-AUC "
            "are undefined without both classes. Evaluate on a split that "
            "contains positives and negatives."
        )
    return array.astype(np.int64, copy=False)


def as_probability_vector(y_proba: object) -> np.ndarray:
    """Coerce ``y_proba`` to a validated, finite probability vector in [0, 1].

    A tiny tolerance (``1e-9``) absorbs floating-point drift from ``expit``;
    values are clipped back into range after validation. Anything outside the
    tolerance is a programming error and raises
    :class:`InvalidProbabilityError`.
    """
    array = np.asarray(y_proba, dtype=float)
    if array.ndim != 1:
        raise InvalidProbabilityError(
            f"y_proba must be one-dimensional, got shape {array.shape}."
        )
    if array.size == 0:
        raise EmptyEvaluationError(
            "y_proba is empty; evaluation needs at least one predicted row."
        )
    if not np.all(np.isfinite(array)):
        raise InvalidProbabilityError(
            "y_proba contains non-finite value(s); predicted probabilities must "
            "be finite."
        )
    lower, upper = float(array.min()), float(array.max())
    if lower < -1e-9 or upper > 1.0 + 1e-9:
        raise InvalidProbabilityError(
            f"y_proba spans [{lower:g}, {upper:g}], outside [0, 1]. "
            "predict_proba must return probabilities, not logits or raw scores."
        )
    return np.clip(array, 0.0, 1.0)


def _as_prediction_vector(y_pred: object, n_samples: int) -> np.ndarray:
    array = np.asarray(y_pred)
    if array.ndim != 1:
        array = array.ravel()
    if array.size != n_samples:
        raise InvalidLabelError(
            f"y_pred has {array.size} entries but the label vector has "
            f"{n_samples}; predictions and labels must align."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidLabelError(
            f"y_pred has dtype {array.dtype!r}; expected numeric 0/1 predictions."
        )
    unexpected = sorted(set(np.unique(array).tolist()) - {NEGATIVE_LABEL, POSITIVE_LABEL})
    if unexpected:
        raise InvalidLabelError(
            f"y_pred contains non-binary prediction(s) {unexpected}; expected "
            f"only {NEGATIVE_LABEL}/{POSITIVE_LABEL}."
        )
    return array.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion-matrix counts plus their margins."""

    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def n_samples(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    @property
    def n_positive(self) -> int:
        return self.tp + self.fn

    @property
    def n_negative(self) -> int:
        return self.tn + self.fp

    def to_dict(self) -> dict[str, int]:
        return {
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tp": int(self.tp),
            "n_samples": int(self.n_samples),
            "n_positive": int(self.n_positive),
            "n_negative": int(self.n_negative),
        }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    """Count true/false positives/negatives for binary labels and predictions."""
    tn = int(np.sum((y_true == NEGATIVE_LABEL) & (y_pred == NEGATIVE_LABEL)))
    fp = int(np.sum((y_true == NEGATIVE_LABEL) & (y_pred == POSITIVE_LABEL)))
    fn = int(np.sum((y_true == POSITIVE_LABEL) & (y_pred == NEGATIVE_LABEL)))
    tp = int(np.sum((y_true == POSITIVE_LABEL) & (y_pred == POSITIVE_LABEL)))
    return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator`` or ``0.0`` when undefined."""
    return float(numerator) / float(denominator) if denominator else 0.0


# ---------------------------------------------------------------------------
# Calibration / reliability curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBin:
    """One equal-width bin of a reliability curve."""

    bin: int
    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    fraction_positive: float | None
    gap: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bin": int(self.bin),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "count": int(self.count),
            "mean_predicted": self.mean_predicted,
            "fraction_positive": self.fraction_positive,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Brier score, expected calibration error, and the reliability curve."""

    brier_score: float
    expected_calibration_error: float
    n_bins: int
    bins: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "brier_score": float(self.brier_score),
            "expected_calibration_error": float(self.expected_calibration_error),
            "n_bins": int(self.n_bins),
            "bins": [b.to_dict() for b in self.bins],
        }


def calibration_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationReport:
    """Build the Brier score, ECE, and equal-width reliability curve.

    Bins are equal width on ``[0, 1]``; a predicted probability of exactly 1.0
    falls in the final bin. Empty bins are retained with ``None`` for their
    undefined mean/fraction/gap so the curve length is always ``n_bins``.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise CalibrationConfigError(
            f"n_bins must be a positive integer, got {n_bins!r}."
        )
    n_bins = int(n_bins)
    n_samples = int(y_true.size)

    # Equal-width bin index, with 1.0 folded into the last bin.
    indices = np.clip((y_proba * n_bins).astype(np.int64), 0, n_bins - 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[CalibrationBin] = []
    ece = 0.0
    for b in range(n_bins):
        mask = indices == b
        count = int(np.count_nonzero(mask))
        if count:
            mean_predicted = float(np.mean(y_proba[mask]))
            fraction_positive = float(np.mean(y_true[mask]))
            gap = abs(fraction_positive - mean_predicted)
            ece += (count / n_samples) * gap
        else:
            mean_predicted = None
            fraction_positive = None
            gap = None
        bins.append(
            CalibrationBin(
                bin=b,
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_predicted=mean_predicted,
                fraction_positive=fraction_positive,
                gap=gap,
            )
        )

    return CalibrationReport(
        brier_score=float(brier_score_loss(y_true, y_proba)),
        expected_calibration_error=float(ece),
        n_bins=n_bins,
        bins=tuple(bins),
    )


# ---------------------------------------------------------------------------
# Full metric dict
# ---------------------------------------------------------------------------


def classification_scores(
    y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Threshold-free and threshold-based discrimination metrics.

    Precision/recall/F1 use the positive class (label 1) with
    ``zero_division=0`` so a degenerate prediction returns ``0.0`` instead of
    raising or emitting a warning.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def compute_metric_dict(
    y_true: object,
    y_proba: object,
    y_pred: object,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, object]:
    """Compute the complete canonical metric dict for one evaluated split.

    This is the only sanctioned producer of the metric dict. It validates its
    inputs (binary labels, in-range probabilities, aligned predictions),
    computes every metric, and returns a dict whose keys are exactly
    :data:`METRIC_KEYS`.
    """
    labels = as_label_vector(y_true)
    probabilities = as_probability_vector(y_proba)
    predictions = _as_prediction_vector(y_pred, labels.size)
    if probabilities.size != labels.size:
        raise InvalidProbabilityError(
            f"y_proba has {probabilities.size} entries but y_true has "
            f"{labels.size}; they must align."
        )

    matrix = confusion_counts(labels, predictions)
    scores = classification_scores(labels, probabilities, predictions)
    calibration = calibration_report(labels, probabilities, n_bins=n_bins)

    metrics: dict[str, object] = {
        "accuracy": scores["accuracy"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "roc_auc": scores["roc_auc"],
        "pr_auc": scores["pr_auc"],
        "specificity": _safe_rate(matrix.tn, matrix.n_negative),
        "npv": _safe_rate(matrix.tn, matrix.tn + matrix.fn),
        "prevalence": _safe_rate(matrix.n_positive, matrix.n_samples),
        "brier_score": calibration.brier_score,
        CONFUSION_MATRIX_KEY: matrix.to_dict(),
        CALIBRATION_KEY: calibration.to_dict(),
    }
    return metrics
