"""Probability calibration for the S06 winner (S06/T03).

A repetitive-CV winner is chosen on ROC-AUC, which is threshold-free — but the
app serves a hard decision, so the winner's raw probabilities must be
trustworthy on the ``[0, 1]`` scale too. This module:

1. **Fits a calibration wrapper** on a dedicated calibration split that was
   never used to fit the winner, using :func:`sklearn.frozen.FrozenEstimator`
   inside :class:`sklearn.calibration.CalibratedClassifierCV` so the winner's
   internal parameters stay frozen and the wrapper only learns the probability
   map (isotonic regression or Platt sigmoid).

2. **Produces the calibration evidence**: Brier score and equal-width
   reliability curve **before** and **after** calibration, plus the Brier
   delta. The curve data is delegated to
   :func:`heart.eval.metrics.calibration_report` — the same schema the S03
   metric contract publishes. This module does not re-implement the
   arithmetic; it wires, compares, and names failures.

Failure contract
----------------
Ceremonial inputs raise named errors: malformed vectors raise
:class:`CalibrationDataError`; an unusable configuration (unknown method,
``n_bins < 1``) raises :class:`CalibrationConfigError`. Aligned label/probability
pairing is validated here so calibration failures surface at the call site,
not in the app's serving boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from heart.eval.metrics import (
    DEFAULT_CALIBRATION_BINS,
    CalibrationReport,
    calibration_report,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CALIBRATION_METHOD",
    "CalibrationError",
    "CalibrationConfigError",
    "CalibrationDataError",
    "CalibrationComparison",
    "fit_calibration_wrapper",
    "calibration_comparison",
]


#: S6 default: isotonic regression (ample calibration rows, no shape forcing).
DEFAULT_CALIBRATION_METHOD: str = "isotonic"

_VALID_METHODS: tuple[str, ...] = ("isotonic", "sigmoid")


class CalibrationError(Exception):
    """Base class for every probability-calibration failure."""


class CalibrationConfigError(CalibrationError):
    """The requested calibration configuration is invalid."""


class CalibrationDataError(CalibrationError):
    """The supplied calibration inputs are malformed and cannot be used."""


def _as_label_vector(values: object, *, label: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise CalibrationDataError(
            f"{label} is a {type(values).__name__}, not a numeric vector."
        )
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CalibrationDataError(
            f"{label} could not be coerced to numeric: {exc}"
        ) from exc
    if vector.ndim != 1:
        raise CalibrationDataError(
            f"{label} must be 1-D, got shape {vector.shape}."
        )
    return vector


def _validate_probabilities(proba: np.ndarray, *, label: str) -> np.ndarray:
    if not np.all(np.isfinite(proba)):
        raise CalibrationDataError(
            f"{label} contains non-finite values; calibration requires "
            "well-formed probabilities."
        )
    if proba.min() < 0.0 or proba.max() > 1.0:
        raise CalibrationDataError(
            f"{label} leaves [0, 1] (min={proba.min():.6f}, "
            f"max={proba.max():.6f}); these are not probabilities."
        )
    return proba


def _validate_aligned(
    y_true: np.ndarray, proba: np.ndarray, *, label_pair: tuple[str, str]
) -> None:
    labels_name, proba_name = label_pair
    if y_true.size != proba.size:
        raise CalibrationDataError(
            f"{labels_name} ({y_true.size} rows) and {proba_name} "
            f"({proba.size} rows) do not align; calibration is row-paired and "
            "a silent zip would fabricate the curve."
        )


def fit_calibration_wrapper(
    model: object,
    X_calib,
    y_calib: object,
    *,
    method: str = DEFAULT_CALIBRATION_METHOD,
) -> CalibratedClassifierCV:
    """Fit a probability-calibration wrapper around a frozen fitted model.

    Parameters
    ----------
    model:
        An *already fitted* classifier exposing ``predict_proba``. Its
        parameters are frozen via :class:`~sklearn.frozen.FrozenEstimator`, so
        calibration never refits on the data the model already saw.
    X_calib, y_calib:
        The dedicated calibration split (rows unseen during winner training).
    method:
        ``"isotonic"`` or ``"sigmoid"`` (Platt). Unknown values raise
        :class:`CalibrationConfigError` rather than falling back silently.

    Raises
    ------
    CalibrationConfigError
        Unknown calibration method.
    CalibrationDataError
        Malformed calibration split (empty, non-numeric labels).
    """
    if not isinstance(method, str) or method not in _VALID_METHODS:
        raise CalibrationConfigError(
            f"Unknown calibration method {method!r}; valid methods are "
            f"{list(_VALID_METHODS)}."
        )
    y = _as_label_vector(y_calib, label="y_calib").astype(int)
    if y.size == 0:
        raise CalibrationDataError("Calibration split is empty; cannot fit.")
    if not hasattr(model, "predict_proba"):
        raise CalibrationDataError(
            f"Model of type {type(model).__name__} does not expose "
            "predict_proba; only probabilistic classifiers calibrate."
        )
    wrapped = CalibratedClassifierCV(
        estimator=FrozenEstimator(model), method=method
    )
    try:
        wrapped.fit(X_calib, y)
    except Exception as exc:  # noqa: BLE001 - re-raise named
        raise CalibrationDataError(
            f"Calibration wrapper failed to fit on the calibration split "
            f"({type(exc).__name__}): {exc}"
        ) from exc
    logger.info(
        "Calibrated winner probabilities with %s wrapper on %d rows.",
        method,
        y.size,
    )
    return wrapped


@dataclass(frozen=True)
class CalibrationComparison:
    """Calibration evidence around the winner: before vs after the wrapper."""

    brier_score_raw: float
    brier_score_calibrated: float
    brier_delta: float
    """``raw - calibrated``; positive means calibration improved sharpness."""

    expected_calibration_error_raw: float
    expected_calibration_error_calibrated: float

    report_raw: CalibrationReport
    report_calibrated: CalibrationReport

    @property
    def is_improved(self) -> bool:
        return self.brier_delta > 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "brier_score_raw": round(self.brier_score_raw, 6),
            "brier_score_calibrated": round(self.brier_score_calibrated, 6),
            "brier_delta": round(self.brier_delta, 6),
            "expected_calibration_error_raw": round(
                self.expected_calibration_error_raw, 6
            ),
            "expected_calibration_error_calibrated": round(
                self.expected_calibration_error_calibrated, 6
            ),
            "is_improved": self.is_improved,
            "report_raw": self.report_raw.to_dict(),
            "report_calibrated": self.report_calibrated.to_dict(),
        }

    def summary(self) -> str:
        direction = "improved" if self.is_improved else "did not improve"
        return (
            f"Brier {self.brier_score_raw:.4f} -> {self.brier_score_calibrated:.4f} "
            f"({direction}); ECE {self.expected_calibration_error_raw:.4f} -> "
            f"{self.expected_calibration_error_calibrated:.4f}"
        )


def calibration_comparison(
    y_true: object,
    proba_raw: object,
    proba_calibrated: object,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationComparison:
    """Compare calibration quality before and after the calibration wrapper.

    Both probability vectors must pair with the same ``y_true`` row-for-row;
    module delegates the actual curve/Brier arithmetic to
    :func:`heart.eval.metrics.calibration_report` so the schema stays shared.

    Raises
    ------
    CalibrationDataError
        Mismatched lengths, non-vector inputs, non-finite or out-of-[0, 1]
        probabilities.
    CalibrationConfigError
        ``n_bins`` below 1.
    """
    y = _as_label_vector(y_true, label="y_true")
    raw = _as_label_vector(proba_raw, label="proba_raw")
    after = _as_label_vector(proba_calibrated, label="proba_calibrated")
    if y.size == 0:
        raise CalibrationDataError("calibration_comparison needs at least one row.")
    _validate_aligned(y, raw, label_pair=("y_true", "proba_raw"))
    _validate_aligned(y, after, label_pair=("y_true", "proba_calibrated"))
    raw = _validate_probabilities(raw, label="proba_raw")
    after = _validate_probabilities(after, label="proba_calibrated")
    if n_bins < 1:
        raise CalibrationConfigError(
            f"n_bins must be positive, got {n_bins!r}."
        )

    report_before = calibration_report(y, raw, n_bins=n_bins)
    report_after = calibration_report(y, after, n_bins=n_bins)
    comparison = CalibrationComparison(
        brier_score_raw=report_before.brier_score,
        brier_score_calibrated=report_after.brier_score,
        brier_delta=report_before.brier_score - report_after.brier_score,
        expected_calibration_error_raw=report_before.expected_calibration_error,
        expected_calibration_error_calibrated=(
            report_after.expected_calibration_error
        ),
        report_raw=report_before,
        report_calibrated=report_after,
    )
    logger.info("Calibration comparison — %s", comparison.summary())
    return comparison
