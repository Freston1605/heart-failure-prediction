"""Validated single-patient prediction input for the Predict page.

This module owns *how the form's widgets turn into a model-ready frame* so
the page stays thin and validation is testable without Streamlit. Every
produced frame is exactly what
:func:`heart.serving.artifact.ServingArtifact.predict_positive_proba`
demands: a one-row :class:`pandas.DataFrame` carrying every embedded
feature column, in the embedded order, with the declared dtypes.

Validation prefers *named* rejections over coercion:

* unknown feature names are a :class:`ValidationError` (they would
  silently shift columns under the model);
* declared-integer features reject fractional values and out-of-range
  numbers — the serving pipeline's preprocessing coerces dtype labels, so
  a bad value that silently NaN-imputed would mean predicting on a
  different patient than the user described;
* categorical features must be one of the declared input levels
  (``FastingBS``/``HeartDisease``-style integer encodings as 0/1, the
  rest as their uppercase source strings from ``heart.csv``); case
  differences are rejected with the allowed set spelled out rather than
  silently "fixed", because guessing a correction is how a typo becomes a
  mislabeled category.

The module never imports Streamlit — it is the plain, page-agnostic
contract, and the page maps a :class:`ValidationError` to a friendly
message instead of a stack trace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from heart.data.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    SPECS_BY_NAME,
)

#: Categorical *features* only — the schema's CATEGORICAL_COLUMNS includes
#: the target (HeartDisease), which no prediction form offers.
FEATURE_CATEGORICAL: tuple[str, ...] = tuple(
    name for name in CATEGORICAL_COLUMNS if name in FEATURE_COLUMNS
)

__all__ = [
    "ValidationError",
    "NUMERIC_RANGES",
    "NUMERIC_DEFAULTS",
    "CATEGORICAL_DISPLAY_LEVELS",
    "build_feature_frame",
]


class ValidationError(Exception):
    """One or more prediction-form values were rejected, with the reasons.

    ``issues`` maps each offending feature name to its plain-language
    reason, so a single submit can surface every problem at once instead of
    one fix per submission.
    """

    def __init__(self, issues: dict[str, str]) -> None:
        self.issues = dict(issues)
        summary = "; ".join(f"{name} {reason}" for name, reason in issues.items())
        super().__init__(f"Invalid prediction input — {summary}")


#: Widget bounds per numeric feature, derived from the pinned dataset's
#: observed span (``data/raw/heart.csv``, 918 rows: RestingBP/Cholesterol
#: zeros dropped as the S01 sentinel audit instructs) with margin for real
#: patients, so "sensible ranges" stay auditable inside this module.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "Age": (18.0, 100.0),
    "RestingBP": (80.0, 240.0),
    "Cholesterol": (85.0, 700.0),
    "MaxHR": (60.0, 220.0),
    "Oldpeak": (-3.0, 7.0),
}


# Form defaults = the pinned dataset's per-column medians (computed once
# against data/raw/heart.csv and pinned here, so the page's defaults are
# the audited source's central values rather than arbitrary guesses).
NUMERIC_DEFAULTS: dict[str, float] = {
    "Age": 54.0,
    "RestingBP": 130.0,
    "Cholesterol": 223.0,
    "MaxHR": 138.0,
    "Oldpeak": 0.6,
}


def _reject_int_feature(name: str, raw: object) -> int:
    """Validate one declared-integer feature; return the cleaned value."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise _single(name, "this value is required.")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _single(name, "must be a number.")
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        raise _single(name, "must be a finite number.")
    integer = int(value)
    if value != integer:
        raise _single(name, "must be a whole number (fractional values are not accepted).")
    minimum, maximum = NUMERIC_RANGES[name]
    if not minimum <= value <= maximum:
        raise _single(name, f"must be between {minimum:g} and {maximum:g}.")
    return integer


def _reject_float_feature(name: str, raw: object) -> float:
    """Validate one declared-float feature; return the cleaned value."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise _single(name, "this value is required.")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _single(name, "must be a number.")
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        raise _single(name, "must be a finite number.")
    minimum, maximum = NUMERIC_RANGES[name]
    if not minimum <= value <= maximum:
        raise _single(name, f"must be between {minimum:g} and {maximum:g}.")
    return value


def _reject_categorical_feature(
    name: str, raw: object, levels: tuple[str, ...]
) -> str:
    """Validate one declared-categorical feature; return the stored value."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise _single(name, "a choice is required.")
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    text = str(raw)
    if text in levels:
        return text
    raise _single(name, f"'{text}' is not an accepted choice — pick one of {', '.join(levels)}.")


def _single(name: str, reason: str) -> ValidationError:
    """A one-issue :class:`ValidationError` (internal convenience)."""
    return ValidationError({name: reason})


def build_feature_frame(values: Mapping[str, object]) -> pd.DataFrame:
    """Build the one validated, model-ready feature frame.

    ``values`` is the per-feature mapping a form submits — string keys so
    ``FastingBS`` and every categorical name survive a streamlit-widget or
    JSON round-trip. Every declared feature must be present; a single
    missing key rejects the whole frame, because "predict without that
    number" is not an operation the model defines.

    Returns a one-row frame with exactly
    :data:`heart.data.schema.FEATURE_COLUMNS` (the embedded artifact's
    order), correct pandas dtypes, and declared input encoding.

    Raises:
        ValidationError: naming each rejected feature (missing key,
            non-numeric value, out-of-range number, non-whole integer, or
            an unknown categorical level). No other exception is raised
            for user-input problems.
    """
    if not isinstance(values, Mapping):
        raise ValidationError(
            {"input": "must be submitted as a mapping of feature values."}
        )

    incoming: dict[str, object] = dict(values)
    issues: dict[str, str] = {}
    for name in FEATURE_COLUMNS:
        if name not in incoming:
            issues[name] = "is required."
    for name in incoming:
        if name not in FEATURE_COLUMNS:
            issues[name] = "is not a feature this model knows."

    cleaned: dict[str, object] = {}
    if not issues:
        for column in FEATURE_COLUMNS:
            spec = SPECS_BY_NAME[column]
            raw = incoming[column]
            try:
                if spec.is_categorical:
                    cleaned[column] = _reject_categorical_feature(
                        column, raw, CATEGORICAL_DISPLAY_LEVELS[column]
                    )
                elif spec.kind == "integer":
                    cleaned[column] = _reject_int_feature(column, raw)
                else:
                    cleaned[column] = _reject_float_feature(column, raw)
            except ValidationError as exc:
                issues.update(exc.issues)

    if issues:
        raise ValidationError(issues)

    frame = pd.DataFrame([{column: cleaned[column] for column in FEATURE_COLUMNS}])
    for column in FEATURE_CATEGORICAL:
        # FastingBS (and any integer-encoded category) must reach the model
        # as int64: its OneHotEncoder was fitted on int64 levels, and a str
        # "0"/"1" would be silently ignored (handle_unknown="ignore"),
        # yielding an all-zero encoding instead of an error.
        if SPECS_BY_NAME[column].storage == "integer":
            frame[column] = frame[column].astype("int64")
        else:
            frame[column] = frame[column].astype(str)
    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].astype("int64" if SPECS_BY_NAME[column].kind == "integer" else "float64")
    return frame


# ---------------------------------------------------------------------------
# Declared categorical display, lifting only from the audited ColumnSpec
# ---------------------------------------------------------------------------

#: Levels exactly as stored in the dataset's cells, keyed by column.
CATEGORICAL_DISPLAY_LEVELS: dict[str, tuple[str, ...]] = {
    name: tuple(str(level) for level in SPECS_BY_NAME[name].levels or ())
    for name in FEATURE_CATEGORICAL
}
