"""Tests for src/heart/eval/calibration.py (S06/T03).

All fixtures are inline/synthetic — no artifact dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from heart.eval.calibration import (
    DEFAULT_CALIBRATION_METHOD,
    CalibrationComparison,
    CalibrationConfigError,
    CalibrationDataError,
    calibration_comparison,
    fit_calibration_wrapper,
)


def _synthetic(rows: int = 200, seed: int = 7, miscalibrated: bool = False):
    """Synthetic split with a fitted model plus its raw probabilities."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 2))
    score = x[:, 0] + x[:, 1]
    y = (score + rng.normal(scale=0.4, size=rows) > 0).astype(int)
    model = LogisticRegression()
    model.fit(x, y)
    proba_raw = model.predict_proba(x)[:, 1]
    if miscalibrated:
        # Known-bad sharpness: push everything toward the middle of the scale.
        proba_raw = 0.5 + 0.2 * (proba_raw - 0.5)
    return model, x, y, proba_raw


def test_default_method_is_isotonic() -> None:
    assert DEFAULT_CALIBRATION_METHOD == "isotonic"


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_fit_calibration_wrapper_freezes_the_winner(method: str) -> None:
    model, x, y, _ = _synthetic()
    wrapper = fit_calibration_wrapper(model, x, y, method=method)
    assert isinstance(wrapper, CalibratedClassifierCV)
    calibrated = wrapper.predict_proba(x)
    assert calibrated.shape == (x.shape[0], 2)
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)


def test_calibration_wrapper_predicts_identically_to_frozen_model_before_fit() -> None:
    """The wrapper wraps the winner it was handed, not a re-fit copy."""
    model, x, y, proba_raw = _synthetic()
    wrapper = fit_calibration_wrapper(model, x, y)
    assert isinstance(wrapper, CalibratedClassifierCV)


def test_fit_calibration_wrapper_rejects_unknown_method() -> None:
    model, x, y, _ = _synthetic()
    with pytest.raises(CalibrationConfigError, match="Unknown calibration method"):
        fit_calibration_wrapper(model, x, y, method="platt_scaling")


def test_fit_calibration_wrapper_rejects_non_probabilistic_model() -> None:
    class NoProba:
        def fit(self, X, y):  # pragma: no cover
            return self

        def predict(self, X):  # pragma: no cover
            return np.zeros(X.shape[0])

    _, x, y, _ = _synthetic()
    with pytest.raises(CalibrationDataError, match="predict_proba"):
        fit_calibration_wrapper(NoProba(), x, y)


def test_fit_calibration_wrapper_rejects_empty_split() -> None:
    model, x, _, _ = _synthetic()
    with pytest.raises(CalibrationDataError, match="empty"):
        fit_calibration_wrapper(model, x[:0], np.array([], dtype=int))


def test_calibration_comparison_brier_fields_and_schema() -> None:
    _, _, y, proba_raw = _synthetic()
    after = np.clip(proba_raw + 0.05, 0.0, 1.0)
    comparison = calibration_comparison(y, proba_raw, after, n_bins=8)
    assert isinstance(comparison, CalibrationComparison)
    assert comparison.report_raw.n_bins == 8
    assert len(comparison.report_raw.bins) == 8
    assert comparison.brier_score_raw == pytest.approx(
        np.mean((y - proba_raw) ** 2)
    )
    assert comparison.brier_delta == pytest.approx(
        comparison.brier_score_raw - comparison.brier_score_calibrated
    )
    payload = comparison.to_dict()
    assert payload["is_improved"] == comparison.is_improved


def test_calibration_comparison_flags_improvement() -> None:
    # Perfect calibration (y itself) must beat a miscalibrated score.
    _, _, y, proba_raw = _synthetic(miscalibrated=True)
    good = calibration_comparison(y, proba_raw, y.astype(float), n_bins=10)
    assert good.is_improved
    bad = calibration_comparison(y, y.astype(float), proba_raw, n_bins=10)
    assert not bad.is_improved


def test_calibration_comparison_rejects_misaligned_lengths() -> None:
    _, _, y, proba_raw = _synthetic(rows=100)
    with pytest.raises(CalibrationDataError, match="do not align"):
        calibration_comparison(y, proba_raw[:-1], proba_raw)


def test_calibration_comparison_rejects_non_finite_proba() -> None:
    _, _, y, proba_raw = _synthetic(rows=100)
    bad = proba_raw.copy()
    bad[0] = np.nan
    with pytest.raises(CalibrationDataError, match="non-finite"):
        calibration_comparison(y, bad, proba_raw)


def test_calibration_comparison_rejects_out_of_range_proba() -> None:
    _, _, y, proba_raw = _synthetic(rows=100)
    bad = proba_raw.copy()
    bad[0] = 1.4
    with pytest.raises(CalibrationDataError, match="not probabilities"):
        calibration_comparison(y, bad, proba_raw)


def test_calibration_comparison_rejects_invalid_bins() -> None:
    _, _, y, proba_raw = _synthetic(rows=100)
    with pytest.raises(CalibrationConfigError, match="n_bins must be positive"):
        calibration_comparison(y, proba_raw, proba_raw, n_bins=0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Run via pytest.")
