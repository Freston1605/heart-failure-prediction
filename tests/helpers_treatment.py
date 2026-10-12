"""Deterministic synthetic fixture corpus for the M006 treatment-gate tests.

Shared by T03 (variant builders), T04 (gate runner) and T05 (verdict logic).
The fixture mirrors the real dataset shape: it uses the real
``heart.data.schema.FEATURE_COLUMNS`` names/roles, is imbalanced (~15%
minority class), carries scattered genuine NaNs in ONE numeric column
(``Cholesterol`` — the column whose missingness the committed zero policy
already owns), is otherwise zero-free and physiologically plausible, and
includes a categorical feature with three levels (``ChestPainType``). Every
random draw comes from ``numpy.random.default_rng(heart.config.RANDOM_SEED)``
so the corpus is byte-stable across runs and machines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heart.config import RANDOM_SEED
from heart.data.quality import DEFAULT_ZERO_POLICY, ZERO_SENTINEL
from heart.data.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

#: Feature-only categorical columns (the target is categorical in the declared
#: schema but is never a fixture feature).
_FIXTURE_CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(
    column for column in CATEGORICAL_COLUMNS if column != TARGET_COLUMN
)

__all__ = [
    "FIXTURE_ROWS",
    "MINORITY_SHARE",
    "NAN_SHARE",
    "make_treatment_fixture",
    "treatment_train_test_split",
    "fixture_missing_columns",
]

FIXTURE_ROWS: int = 240
MINORITY_SHARE: float = 0.15
NAN_SHARE: float = 0.08

#: The one numeric column carrying scattered genuine NaNs (mirrors the real
#: missingness story: Cholesterol is the zero-as-missing policy column).
NAN_COLUMN: str = "Cholesterol"


def make_treatment_fixture(
    n_rows: int = FIXTURE_ROWS,
    *,
    seed: int = RANDOM_SEED,
    minority_share: float = MINORITY_SHARE,
    nan_share: float = NAN_SHARE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the deterministic imbalanced binary fixture (X, y).

    - 240 float numeric features plus the real categorical domains; every
      numeric column is zero-free and physiologically plausible (the committed
      zero policy therefore runs but has no sentinel to replace).
    - ``Cholesterol`` is lognormal (right-skewed, strictly positive) with
      ``~nan_share`` scattered genuine NaNs, so the power-transform, KNN-
      imputer and missingness-flag variants all have a real mechanism to act
      on.
    - ``Oldpeak`` is exponential-ish (skewed, carries zeros — a legitimate
      measurement for Oldpeak) so the box-cox strictly-positive restriction
      provably excludes a column on the fixture.
    - ``y`` has exactly ``round(minority_share * n_rows)`` minority rows
      (label 1), correlated with the feature signal (not noise), so every
      variant fits against something learnable.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(29.0, 77.0, size=n_rows)
    resting_bp = rng.uniform(94.0, 190.0, size=n_rows)
    # Lognormal cholesterol: strictly positive, right-skewed (skew ~1.0).
    cholesterol = np.exp(rng.normal(5.18, 0.32, size=n_rows))
    max_hr = rng.uniform(92.0, 200.0, size=n_rows)
    # Oldpeak carries legitimate zeros (a real measurement for Oldpeak), so
    # the box-cox strictly-positive restriction provably excludes it.
    oldpeak = rng.exponential(scale=1.1, size=n_rows)
    oldpeak[rng.random(n_rows) < 0.15] = 0.0

    # Scattered genuine NaNs in exactly one numeric column (~8%): exactly
    # round(nan_share*n_rows) cells, chosen seeded, so the share is exact.
    n_missing = int(round(nan_share * n_rows))
    nan_mask = np.zeros(n_rows, dtype=bool)
    nan_mask[rng.choice(n_rows, size=n_missing, replace=False)] = True
    cholesterol = cholesterol.astype("float64")
    cholesterol[nan_mask] = np.nan

    chest_pain = rng.choice(("ASY", "ATA", "NAP"), size=n_rows)
    sex = rng.choice(("M", "F"), size=n_rows)
    fasting_bs = rng.choice(("0", "1"), size=n_rows)
    resting_ecg = rng.choice(("Normal", "ST", "LVH"), size=n_rows)
    exercise_angina = rng.choice(("N", "Y"), size=n_rows)
    st_slope = rng.choice(("Up", "Flat", "Down"), size=n_rows)

    frame = pd.DataFrame(
        {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }
    )
    # Column-set/type sanity: the committed pipeline requires this exact set.
    if list(frame.columns) != list(FEATURE_COLUMNS):
        raise RuntimeError(
            "fixture frame does not match the declared FEATURE_COLUMNS order."
        )
    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].astype("float64")
        zeros = int((frame[column] == ZERO_SENTINEL).sum())
        # Oldpeak may legitimately carry zeros (a real measurement); every
        # OTHER numeric column stays zero-free and physiologically plausible.
        if zeros and column not in (NAN_COLUMN, "Oldpeak"):
            raise RuntimeError(f"fixture numeric column {column} carries zeros.")
    for column in _FIXTURE_CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("object")

    # Deterministic labels: a smooth latent score + noise; the exact top-k
    # rows become the minority class (label 1) for an exact 15% prevalence.
    age_z = (frame["Age"].to_numpy() - 54.0) / 9.0
    maxhr_z = (frame["MaxHR"].to_numpy() - 140.0) / 22.0
    oldpeak_z = (frame["Oldpeak"].to_numpy() - 1.1) / 1.1
    asy = (frame["ChestPainType"].to_numpy() == "ASY").astype("float64")
    score = 0.9 * age_z + 0.8 * oldpeak_z - 0.7 * maxhr_z + 0.6 * asy
    score += rng.normal(0.0, 0.9, size=n_rows)
    n_minority = int(round(minority_share * n_rows))
    threshold_index = np.argsort(score, kind="stable")[::-1][:n_minority]
    y = np.zeros(n_rows, dtype="int64")
    y[threshold_index] = 1
    return frame, y


def treatment_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    test_size: float = 0.25,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/test split of the fixture, seeded and deterministic."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )


def fixture_missing_columns(X: pd.DataFrame) -> tuple[str, ...]:
    """Fixture columns carrying observed missing values."""
    policy_columns = tuple(DEFAULT_ZERO_POLICY.columns)
    missing = tuple(
        column
        for column in X.columns
        if X[column].isna().any()
        or (
            column in policy_columns
            and pd.api.types.is_numeric_dtype(X[column])
            and (X[column] == ZERO_SENTINEL).any()
        )
    )
    return missing
