"""Tests for the four-stage pre-split EDA engine (M005/S01/T01).

The contract under test is the engine itself: four stage functions plus the
:func:`heart.reporting.data_methods.run_eda` orchestrator, each operating on a
raw pandas DataFrame before any train/test split. Positive tests use a small
deterministic inline fixture frame (mirroring the ``tests/test_leakage.py``
style) that deliberately contains:

* sentinel zeros in the declared zero-as-missing columns (``RestingBP``,
  ``Cholesterol``),
* a perfectly collinear numeric column (``CholesterolProxy = Cholesterol + 1``)
  so the VIF auxiliary-regression formula must inflate its value,
* a severity-driven target so empirical-logit decile bins are ordered, and
* injectable NaN cells so the random-vs-systematic missingness classification
  can be separated.

Matplotlib is forced to the headless Agg backend before the engine imports it.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from heart.config import RANDOM_SEED
from heart.data.schema import TARGET_COLUMN
from heart.reporting.data_methods import (
    EmptyDatasetError,
    MissingTargetColumnError,
    NonDataFrameError,
    run_eda,
    run_stage1_hygiene,
    run_stage2_univariate,
    run_stage3_bivariate,
    run_stage4_assumptions,
)


# ---------------------------------------------------------------------------
# Fixture frame
# ---------------------------------------------------------------------------


def make_frame(n: int = 240, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a deterministic fixture frame in the shape of the raw dataset.

    ``severity`` rises linearly with row index and drives both the numeric
    features and the target, so feature-vs-target relationships and empirical
    logit bins are ordered and predictable.
    """
    rng = np.random.default_rng(seed)
    severity = np.linspace(0.0, 1.0, n)
    age = rng.integers(30, 81, size=n)
    maxhr = np.rint(200 - severity * 60 + rng.normal(0.0, 8.0, size=n)).astype(int)
    oldpeak = np.round(severity * 4.0 + np.abs(rng.normal(0.0, 0.3, size=n)), 2)
    restingbp = rng.integers(90, 190, size=n).astype(int)
    restingbp[3] = 0  # declared sentinel: resting blood pressure cannot be 0
    cholesterol = rng.integers(120, 400, size=n).astype(int)
    cholesterol[:5] = 0  # declared sentinels
    collinear = cholesterol.astype(float) + 1.0  # perfectly collinear with Cholesterol
    return pd.DataFrame(
        {
            "Age": age,
            "Sex": rng.choice(["M", "F"], size=n),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], size=n),
            "RestingBP": restingbp,
            "Cholesterol": cholesterol,
            "CholesterolProxy": collinear,
            "FastingBS": rng.integers(0, 2, size=n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], size=n),
            "MaxHR": maxhr,
            "ExerciseAngina": np.where(rng.random(n) < severity, "Y", "N"),
            "Oldpeak": oldpeak,
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], size=n),
            TARGET_COLUMN: (
                rng.random(n) < 1.0 / (1.0 + np.exp(-(severity * 12.0 - 6.0)))
            ).astype(int),
        }
    )


@pytest.fixture(scope="module")
def eda_result() -> dict:
    """One shared run_eda result for the cheap structural assertions."""
    return run_eda(make_frame())


# ---------------------------------------------------------------------------
# Declared keys and JSON-serialisability
# ---------------------------------------------------------------------------


def test_run_eda_returns_declared_keys(eda_result: dict) -> None:
    assert set(eda_result) == {
        "meta",
        "stage1_hygiene",
        "stage2_univariate",
        "stage3_bivariate",
        "stage4_assumptions",
    }
    assert set(eda_result["stage1_hygiene"]) == {
        "shape",
        "dtypes",
        "missingness",
        "exact_duplicates",
        "impossible_values",
        "figures",
    }
    assert set(eda_result["stage2_univariate"]) == {
        "numeric",
        "categorical",
        "target",
        "figures",
    }
    assert set(eda_result["stage3_bivariate"]) == {
        "numeric_vs_target",
        "contingency",
        "correlation",
        "vif",
        "figures",
    }
    assert set(eda_result["stage4_assumptions"]) == {
        "empirical_logit",
        "outlier_impact",
        "figures",
    }
    assert eda_result["meta"]["row_count"] == 240
    assert eda_result["meta"]["target_column"] == TARGET_COLUMN
    assert eda_result["meta"]["seed"] == RANDOM_SEED
    assert "CholesterolProxy" in eda_result["meta"]["numeric_columns"]
    assert "FastingBS" in eda_result["meta"]["categorical_columns"]


def test_result_is_json_serialisable(eda_result: dict) -> None:
    # allow_nan=False rejects non-finite floats: strict JSON, no NaN/Infinity tokens.
    payload = json.dumps(eda_result, allow_nan=False)
    assert json.loads(payload)["meta"]["row_count"] == 240


def test_run_eda_is_deterministic() -> None:
    first = run_eda(make_frame())
    second = run_eda(make_frame())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# ---------------------------------------------------------------------------
# Stage 1 — missingness classification
# ---------------------------------------------------------------------------


def test_missingness_classification_separates_nan_and_sentinel() -> None:
    frame = make_frame()
    frame.loc[100, "Age"] = np.nan  # random missingness, distinct row from sentinels
    stage1 = run_stage1_hygiene(frame)
    missingness = stage1["missingness"]

    # Systematic: declared sentinel zeros counted only in the declared columns.
    assert missingness["systematic_sentinel"] == {"RestingBP": 1, "Cholesterol": 5}
    assert missingness["random_nan"]["Age"] == 1
    assert missingness["random_nan"]["RestingBP"] == 0
    assert missingness["total_sentinel_zero_cells"] == 6
    assert missingness["total_random_nan_cells"] == 1
    assert missingness["rows_with_any_missingness"] == 6  # 5 sentinel rows + row 100 (NaN)


def test_exact_duplicates_reported() -> None:
    frame = make_frame()
    duplicated_row = frame.iloc[[10]]
    frame = pd.concat([frame, duplicated_row], ignore_index=True)
    stage1 = run_stage1_hygiene(frame)
    assert stage1["exact_duplicates"]["duplicate_rows"] == 2
    assert stage1["exact_duplicates"]["duplicate_groups"] == 1
    clean = run_stage1_hygiene(make_frame())
    assert clean["exact_duplicates"]["duplicate_rows"] == 0


def test_impossible_values_reuse_declared_policy() -> None:
    stage1 = run_stage1_hygiene(make_frame())
    zero_counts = stage1["impossible_values"]["zero_counts"]
    assert zero_counts["RestingBP"]["count"] == 1
    assert zero_counts["Cholesterol"]["count"] == 5
    assert stage1["impossible_values"]["policy"]["columns"] == ["RestingBP", "Cholesterol"]


# ---------------------------------------------------------------------------
# Stage 2 — univariate
# ---------------------------------------------------------------------------


def test_univariate_summary_and_target_ratio(eda_result: dict) -> None:
    stage2 = eda_result["stage2_univariate"]
    assert set(stage2["numeric"]) == {
        "Age",
        "RestingBP",
        "Cholesterol",
        "CholesterolProxy",
        "MaxHR",
        "Oldpeak",
    }
    age = stage2["numeric"]["Age"]
    assert len(age["histogram"]["counts"]) == len(age["histogram"]["bin_edges"]) - 1
    assert age["box_outliers"]["count"] >= 0
    assert age["box_outliers"]["q1"] <= age["box_outliers"]["q3"]
    assert "skew" in age and "kurtosis" in age

    assert set(stage2["categorical"]) == {
        "Sex",
        "ChestPainType",
        "FastingBS",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    }
    sex = stage2["categorical"]["Sex"]
    assert sum(sex["frequencies"].values()) == 240

    target = stage2["target"]
    assert sum(target["counts"].values()) == 240
    total_positives = int(make_frame()[TARGET_COLUMN].sum())
    assert target["counts"]["1"] == total_positives
    assert math.isclose(target["prevalence"], total_positives / 240)
    assert target["imbalance_ratio"] > 0


def test_rare_category_detection_uses_prevalence_floor() -> None:
    frame = make_frame()
    # 2/240 = 0.83% of rows: far below the 5% declared floor.
    frame.loc[:1, "RestingECG"] = "RARE_LEVEL"
    stage2 = run_stage2_univariate(frame)
    assert "RARE_LEVEL" in stage2["categorical"]["RestingECG"]["rare_levels"]
    common = run_stage2_univariate(make_frame())
    for entry in common["categorical"].values():
        assert entry["rare_levels"] == []


# ---------------------------------------------------------------------------
# Stage 3 — bivariate / multivariate
# ---------------------------------------------------------------------------


def test_vif_inflated_for_collinear_fixture(eda_result: dict) -> None:
    vif = eda_result["stage3_bivariate"]["vif"]
    assert set(vif) == {
        "Age",
        "RestingBP",
        "Cholesterol",
        "CholesterolProxy",
        "MaxHR",
        "Oldpeak",
    }
    assert all(value >= 1.0 for value in vif.values())
    # Perfect linear dependence (CholesterolProxy = Cholesterol + 1) must blow up.
    assert vif["CholesterolProxy"] > 100.0
    # Independent-ish features stay small.
    assert vif["Age"] < 10.0


def test_collinear_pair_flagged_in_correlation(eda_result: dict) -> None:
    correlation = eda_result["stage3_bivariate"]["correlation"]
    assert correlation["threshold"] == 0.8
    flagged = {
        (pair["a"], pair["b"]) for pair in correlation["high_correlations"]
    }
    assert ("Cholesterol", "CholesterolProxy") in flagged or (
        "CholesterolProxy",
        "Cholesterol",
    ) in flagged
    matrix = correlation["matrix"]
    assert len(matrix) == 6 and all(len(row) == 6 for row in matrix.values())


def test_contingency_tables_and_chi_square(eda_result: dict) -> None:
    contingency = eda_result["stage3_bivariate"]["contingency"]
    exercise = contingency["ExerciseAngina"]
    assert sum(exercise["table"]["N"].values()) + sum(exercise["table"]["Y"].values()) == 240
    assert exercise["chi_square"]["statistic"] is not None
    assert 0.0 <= exercise["chi_square"]["p_value"] <= 1.0
    assert exercise["chi_square"]["dof"] == 1


def test_numeric_vs_target_overlays(eda_result: dict) -> None:
    overlays = eda_result["stage3_bivariate"]["numeric_vs_target"]
    oldpeak = overlays["Oldpeak"]
    assert set(oldpeak["group_box"]) == {"0", "1"}
    assert oldpeak["group_box"]["0"]["n"] + oldpeak["group_box"]["1"]["n"] == 240
    # Oldpeak rises with severity, so the positive class shifts upward.
    assert oldpeak["group_box"]["1"]["median"] > oldpeak["group_box"]["0"]["median"]


# ---------------------------------------------------------------------------
# Stage 4 — assumptions
# ---------------------------------------------------------------------------


def test_empirical_logit_bins_respect_counts(eda_result: dict) -> None:
    logit = eda_result["stage4_assumptions"]["empirical_logit"]
    bins = logit["Oldpeak"]["bins"]
    assert bins, "expected decile bins for Oldpeak"
    assert sum(entry["n"] for entry in bins) == 240
    assert sum(entry["positives"] for entry in bins) == int(make_frame()[TARGET_COLUMN].sum())
    assert sum(entry["negatives"] for entry in bins) == 240 - int(
        make_frame()[TARGET_COLUMN].sum()
    )
    for entry in bins:
        assert entry["n"] == entry["positives"] + entry["negatives"]
        assert math.isfinite(entry["logit"])  # additive smoothing keeps log-odds finite
        assert 0.0 <= entry["positives"] <= entry["n"]
    # Oldpeak is severity-driven, so the log-odds must rise across deciles.
    assert bins[-1]["logit"] > bins[0]["logit"]


def test_outlier_impact_compares_declared_policies(eda_result: dict) -> None:
    impact = eda_result["stage4_assumptions"]["outlier_impact"]
    assert impact["declared_sentinel_columns"] == ["RestingBP", "Cholesterol"]
    cholesterol = impact["columns"]["Cholesterol"]
    assert cholesterol["declared_policy_applies"] is True
    assert cholesterol["declared_median_impute"]["cells_replaced"] == 5
    assert cholesterol["declared_median_impute"]["fill_value"] > 0
    # Trimming drops rows, Winsorizing keeps the same count as raw.
    assert cholesterol["raw"]["n"] == 240
    assert cholesterol["winsorize_iqr"]["n"] == 240
    assert cholesterol["trim_iqr"]["n"] <= 240
    assert cholesterol["trim_iqr"]["rows_dropped"] == 240 - cholesterol["trim_iqr"]["n"]
    # Non-sentinel columns carry no declared median-impute policy.
    assert impact["columns"]["Age"]["declared_policy_applies"] is False
    assert impact["columns"]["Age"]["declared_median_impute"] is None
    # The flat CSV-ready table covers every column x every applicable policy.
    assert len(impact["table"]) >= 6 * 3


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_figures_rendered_as_png_when_artifact_dir_given(tmp_path) -> None:
    result = run_eda(make_frame(), artifact_dir=tmp_path)
    figures = [
        filename
        for stage in (
            "stage1_hygiene",
            "stage2_univariate",
            "stage3_bivariate",
            "stage4_assumptions",
        )
        for filename in result[stage]["figures"]
    ]
    assert figures
    for filename in figures:
        path = tmp_path / filename
        assert path.exists(), f"missing figure artifact {filename}"
        assert path.stat().st_size > 0
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "stage1_missingness_patterns.png" in result["stage1_hygiene"]["figures"]
    assert "stage2_hist_Oldpeak.png" in result["stage2_univariate"]["figures"]
    assert "stage2_bar_Sex.png" in result["stage2_univariate"]["figures"]
    assert "stage3_overlay_Oldpeak.png" in result["stage3_bivariate"]["figures"]
    assert "stage3_correlation_heatmap.png" in result["stage3_bivariate"]["figures"]
    assert "stage4_empirical_logit_Oldpeak.png" in result["stage4_assumptions"]["figures"]


def test_no_figures_rendered_without_artifact_dir(eda_result: dict) -> None:
    for stage in (
        "stage1_hygiene",
        "stage2_univariate",
        "stage3_bivariate",
        "stage4_assumptions",
    ):
        assert eda_result[stage]["figures"] == []


# ---------------------------------------------------------------------------
# Named failure paths
# ---------------------------------------------------------------------------


def test_empty_frame_raises_named_exception() -> None:
    with pytest.raises(EmptyDatasetError):
        run_stage1_hygiene(pd.DataFrame())
    with pytest.raises(EmptyDatasetError):
        run_eda(pd.DataFrame())


def test_missing_target_column_raises_named_exception() -> None:
    frame = make_frame().drop(columns=[TARGET_COLUMN])
    with pytest.raises(MissingTargetColumnError):
        run_stage2_univariate(frame)
    with pytest.raises(MissingTargetColumnError):
        run_eda(frame)


def test_non_dataframe_raises_named_exception() -> None:
    with pytest.raises(NonDataFrameError):
        run_stage3_bivariate({"Age": [1, 2, 3]})  # type: ignore[arg-type]
    with pytest.raises(NonDataFrameError):
        run_eda("not a frame")  # type: ignore[arg-type]


def test_data_methods_error_is_base_family() -> None:
    from heart.reporting.data_methods import DataMethodsError

    for error in (NonDataFrameError, EmptyDatasetError, MissingTargetColumnError):
        assert issubclass(error, DataMethodsError)
