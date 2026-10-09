"""Tests for the engineered feature transforms (S04/T01).

These tests pin the contract that makes feature ablation possible:

1. The registry is internally consistent — every transform declares inputs,
   outputs, a kind, and a rationale, and no two transforms claim the same
   output column.
2. Every transform is **individually toggleable**: enabling exactly one
   transform adds exactly its declared column(s) and nothing else.
3. The transforms compute the clinically expected values, are deterministic,
   and never mutate their input.
4. The negative surface fails loudly: unknown names, bare-string selections,
   empty selections, missing inputs, column collisions, and un-imputed zero
   denominators all raise named errors.
5. The sklearn wrapper behaves like :func:`engineer_features` and composes in
   a real :class:`~sklearn.pipeline.Pipeline`.

Fixtures are synthetic (no gitignored paths); one integration test reads the
git-tracked S01 split artifacts to prove the engine runs on the real schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart.data.schema import FEATURE_COLUMNS
from heart.data.quality import DEFAULT_ZERO_POLICY, ZeroMedianImputer
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.features.engineering import (
    CATEGORICAL_KIND,
    ENGINEERED_COLUMN_NAMES,
    ENGINEERED_TRANSFORMS,
    NUMERIC_KIND,
    TRANSFORM_NAMES,
    EmptyFeatureSelectionError,
    EngineeredColumnCollisionError,
    EngineeredFeatureTransformer,
    FeatureEngineeringError,
    FeatureTransform,
    MissingFeatureColumnError,
    NonFiniteEngineeredFeatureError,
    NotAFeatureFrameError,
    UnknownFeatureTransformError,
    available_transforms,
    describe_transforms,
    engineer_features,
    resolve_selection,
    selected_column_names,
    selected_columns_by_kind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_frame() -> pd.DataFrame:
    """A 5-row frame covering every transform input with known values."""
    return pd.DataFrame(
        {
            "Age": [35, 45, 55, 65, 75],
            "Sex": ["F", "M", "F", "M", "F"],
            "ChestPainType": ["ATA", "ASY", "NAP", "TA", "ASY"],
            "RestingBP": [120, 130, 140, 150, 160],
            "Cholesterol": [180, 200, 220, 240, 260],
            "FastingBS": [0, 1, 0, 1, 0],
            "RestingECG": ["Normal", "ST", "LVH", "Normal", "ST"],
            "MaxHR": [170, 150, 165, 120, 135],
            "ExerciseAngina": ["N", "Y", "N", "Y", "N"],
            "Oldpeak": [0.0, 1.5, 2.5, 0.0, 3.0],
            "ST_Slope": ["Up", "Flat", "Down", "Up", "Flat"],
        }
    )


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """A fresh synthetic feature frame per test (mutation-safe)."""
    return _synthetic_frame()


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_registry_names_and_columns_are_unique():
    assert len(set(TRANSFORM_NAMES)) == len(TRANSFORM_NAMES)
    assert len(set(ENGINEERED_COLUMN_NAMES)) == len(ENGINEERED_COLUMN_NAMES)
    assert len(ENGINEERED_COLUMN_NAMES) == sum(
        len(t.produces) for t in ENGINEERED_TRANSFORMS.values()
    )


def test_every_transform_declares_a_complete_metadata_contract():
    for name, transform in ENGINEERED_TRANSFORMS.items():
        assert transform.name == name
        assert transform.kind in {NUMERIC_KIND, CATEGORICAL_KIND}
        assert transform.inputs, f"{name} declares no inputs"
        assert transform.produces, f"{name} declares no outputs"
        assert transform.description.strip(), f"{name} lacks a description"
        assert transform.rationale.strip(), f"{name} lacks a clinical rationale"
        assert all(column in FEATURE_COLUMNS for column in transform.inputs)


def test_available_transforms_matches_registry_order():
    assert available_transforms() == TRANSFORM_NAMES


def test_every_declared_output_appears_exactly_once():
    for transform in ENGINEERED_TRANSFORMS.values():
        assert tuple(transform.produces) in [
            tuple(other.produces) for other in ENGINEERED_TRANSFORMS.values()
        ]
    # The full engine produces every declared column.
    frame = _synthetic_frame()
    result = engineer_features(frame)
    for column in ENGINEERED_COLUMN_NAMES:
        assert column in result.columns


def test_transform_to_dict_is_serialisable():
    payload = ENGINEERED_TRANSFORMS["hr_reserve"].to_dict()
    assert payload["name"] == "hr_reserve"
    assert payload["kind"] == NUMERIC_KIND
    assert payload["inputs"] == ["Age", "MaxHR"]
    assert payload["produces"] == ["HR_Reserve"]


# ---------------------------------------------------------------------------
# Individually toggleable transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_each_transform_is_individually_enableable(name, feature_frame):
    transform = ENGINEERED_TRANSFORMS[name]
    result = engineer_features(feature_frame, include=[name])

    for column in transform.produces:
        assert column in result.columns
    added = [c for c in result.columns if c not in feature_frame.columns]
    assert added == list(transform.produces)


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_each_transform_produces_expected_column_with_correct_kind(name, feature_frame):
    transform = ENGINEERED_TRANSFORMS[name]
    block = transform.apply(feature_frame)
    assert tuple(block.columns) == transform.produces
    for column in transform.produces:
        if transform.kind == NUMERIC_KIND:
            assert pd.api.types.is_numeric_dtype(block[column])
        else:
            assert block[column].map(type).eq(str).all()


def test_excluding_a_transform_omits_its_columns(feature_frame):
    result = engineer_features(feature_frame, exclude=["age_band"])
    assert "AgeBand" not in result.columns
    # A different transform still runs.
    assert "HR_Reserve" in result.columns
    assert ENGINEERED_TRANSFORMS["age_band"].produces[0] not in result.columns


def test_include_none_selects_every_transform(feature_frame):
    result = engineer_features(feature_frame)
    assert list(result.columns) == list(feature_frame.columns) + list(
        ENGINEERED_COLUMN_NAMES
    )


def test_selection_order_is_preserved(feature_frame):
    result = engineer_features(
        feature_frame, include=["metabolic_risk_count", "age_band"]
    )
    added = [c for c in result.columns if c not in feature_frame.columns]
    assert added == ["Metabolic_Risk_Count", "AgeBand"]


def test_keep_original_false_returns_only_engineered_columns(feature_frame):
    result = engineer_features(feature_frame, include=["hr_reserve"], keep_original=False)
    assert list(result.columns) == ["HR_Reserve"]


def test_resolve_selection_deduplicates_preserving_order():
    selection = resolve_selection(include=["hr_reserve", "age_band", "hr_reserve"])
    assert [t.name for t in selection] == ["hr_reserve", "age_band"]
    assert all(isinstance(t, FeatureTransform) for t in selection)


def test_selected_column_names_and_kinds_agree():
    all_columns = selected_column_names()
    numeric = selected_columns_by_kind(NUMERIC_KIND)
    categorical = selected_columns_by_kind(CATEGORICAL_KIND)
    assert set(numeric) | set(categorical) == set(all_columns)
    assert not set(numeric) & set(categorical)
    assert "AgeBand" in categorical
    assert "HR_Reserve" in numeric


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------


def test_age_band_boundaries(feature_frame):
    result = engineer_features(feature_frame, include=["age_band"])
    assert list(result["AgeBand"]) == [
        "lt40",
        "40-49",
        "50-59",
        "60-69",
        "ge70",
    ]


def test_max_hr_percent_predicted_values(feature_frame):
    result = engineer_features(feature_frame, include=["max_hr_percent_predicted"])
    expected = [170 / 185 * 100, 150 / 175 * 100, 100.0, 120 / 155 * 100, 135 / 145 * 100]
    assert result["MaxHR_Pct_Predicted"].to_numpy() == pytest.approx(expected)


def test_hr_reserve_values(feature_frame):
    result = engineer_features(feature_frame, include=["hr_reserve"])
    assert result["HR_Reserve"].to_numpy() == pytest.approx([15.0, 25.0, 0.0, 35.0, 10.0])


def test_cholesterol_age_ratio_values(feature_frame):
    result = engineer_features(feature_frame, include=["cholesterol_age_ratio"])
    expected = [180 / 35, 200 / 45, 220 / 55, 240 / 65, 260 / 75]
    assert result["Cholesterol_Age_Ratio"].to_numpy() == pytest.approx(expected)


def test_cholesterol_restingbp_ratio_values(feature_frame):
    result = engineer_features(feature_frame, include=["cholesterol_restingbp_ratio"])
    expected = [180 / 120, 200 / 130, 220 / 140, 240 / 150, 260 / 160]
    assert result["Cholesterol_RestingBP_Ratio"].to_numpy() == pytest.approx(expected)


def test_rate_pressure_product_values(feature_frame):
    result = engineer_features(feature_frame, include=["rate_pressure_product"])
    expected = [120 * 170 / 1000, 130 * 150 / 1000, 140 * 165 / 1000, 150 * 120 / 1000, 160 * 135 / 1000]
    assert result["Rate_Pressure_Product"].to_numpy() == pytest.approx(expected)


def test_oldpeak_slope_score_values(feature_frame):
    result = engineer_features(feature_frame, include=["oldpeak_slope_interaction"])
    # Up weight 0, Flat weight 1, Down weight 2.
    assert result["Oldpeak_Slope_Score"].to_numpy() == pytest.approx(
        [0.0, 1.5, 5.0, 0.0, 3.0]
    )


def test_metabolic_risk_count_values(feature_frame):
    result = engineer_features(feature_frame, include=["metabolic_risk_count"])
    assert list(result["Metabolic_Risk_Count"]) == [0, 2, 1, 4, 2]
    assert pd.api.types.is_integer_dtype(result["Metabolic_Risk_Count"])


def test_exercise_ecg_group_values(feature_frame):
    result = engineer_features(feature_frame, include=["exercise_ecg_group"])
    assert list(result["Exercise_ECG_Group"]) == [
        "N_Up",
        "Y_Flat",
        "N_Down",
        "Y_Up",
        "N_Flat",
    ]


def test_age_band_sex_values(feature_frame):
    result = engineer_features(feature_frame, include=["age_band_sex"])
    assert list(result["AgeBand_Sex"]) == [
        "lt40_F",
        "40-49_M",
        "50-59_F",
        "60-69_M",
        "ge70_F",
    ]


# ---------------------------------------------------------------------------
# Determinism, purity, and index handling
# ---------------------------------------------------------------------------


def test_engineering_is_deterministic(feature_frame):
    first = engineer_features(feature_frame)
    second = engineer_features(feature_frame)
    pd.testing.assert_frame_equal(first, second)


def test_engineering_does_not_mutate_the_input(feature_frame):
    snapshot = feature_frame.copy(deep=True)
    engineer_features(feature_frame)
    pd.testing.assert_frame_equal(feature_frame, snapshot)


def test_index_is_preserved(feature_frame):
    indexed = feature_frame.copy()
    indexed.index = [10, 20, 30, 40, 50]
    result = engineer_features(indexed, include=["hr_reserve"])
    assert list(result.index) == [10, 20, 30, 40, 50]


def test_engineering_handles_a_single_row():
    frame = _synthetic_frame().iloc[[0]]
    result = engineer_features(frame)
    assert len(result) == 1
    assert np.isfinite(result["MaxHR_Pct_Predicted"].iloc[0])


# ---------------------------------------------------------------------------
# Negative surface
# ---------------------------------------------------------------------------


def test_non_dataframe_raises(feature_frame):
    with pytest.raises(NotAFeatureFrameError):
        engineer_features(["not", "a", "frame"])


def test_missing_input_column_raises(feature_frame):
    frame = feature_frame.drop(columns=["MaxHR"])
    with pytest.raises(MissingFeatureColumnError):
        engineer_features(frame, include=["hr_reserve"])


def test_unknown_include_name_raises(feature_frame):
    with pytest.raises(UnknownFeatureTransformError):
        engineer_features(feature_frame, include=["not_a_transform"])


def test_unknown_exclude_name_raises(feature_frame):
    with pytest.raises(UnknownFeatureTransformError):
        engineer_features(feature_frame, exclude=["not_a_transform"])


def test_empty_include_selection_raises(feature_frame):
    with pytest.raises(EmptyFeatureSelectionError):
        engineer_features(feature_frame, include=[])


def test_excluding_everything_raises(feature_frame):
    with pytest.raises(EmptyFeatureSelectionError):
        engineer_features(feature_frame, exclude=list(TRANSFORM_NAMES))


def test_bare_string_include_raises(feature_frame):
    with pytest.raises(FeatureEngineeringError):
        engineer_features(feature_frame, include="age_band")


def test_non_string_include_entry_raises(feature_frame):
    with pytest.raises(FeatureEngineeringError):
        engineer_features(feature_frame, include=["age_band", 5])


def test_column_collision_raises(feature_frame):
    frame = feature_frame.copy()
    frame["AgeBand"] = "already_here"
    with pytest.raises(EngineeredColumnCollisionError):
        engineer_features(frame, include=["age_band"])


def test_unknown_kind_raises():
    with pytest.raises(FeatureEngineeringError):
        selected_columns_by_kind("ordinal")


def test_zero_denominator_fails_closed(feature_frame):
    frame = feature_frame.copy()
    frame.loc[0, "RestingBP"] = 0
    with pytest.raises(NonFiniteEngineeredFeatureError):
        engineer_features(frame, include=["cholesterol_restingbp_ratio"])


def test_zero_denominator_can_be_inspected_when_finiteness_check_disabled(
    feature_frame,
):
    frame = feature_frame.copy()
    frame.loc[0, "RestingBP"] = 0
    result = engineer_features(
        frame,
        include=["cholesterol_restingbp_ratio"],
        check_finite=False,
    )
    assert np.isnan(result.loc[0, "Cholesterol_RestingBP_Ratio"])
    assert np.isfinite(result.loc[1, "Cholesterol_RestingBP_Ratio"])


def test_finiteness_check_does_not_reject_categorical_columns(feature_frame):
    # A categorical column is object dtype; the numeric finiteness sweep must
    # skip it rather than treating it as non-finite.
    result = engineer_features(feature_frame, include=["age_band"])
    assert result["AgeBand"].notna().all()


# ---------------------------------------------------------------------------
# Scikit-learn integration
# ---------------------------------------------------------------------------


def test_transformer_matches_functional_engineer_features(feature_frame):
    transformer = EngineeredFeatureTransformer()
    transformer.fit(feature_frame)
    transformed = transformer.transform(feature_frame)
    expected = engineer_features(feature_frame)
    pd.testing.assert_frame_equal(transformed, expected)


def test_transformer_fit_validates_inputs(feature_frame):
    frame = feature_frame.drop(columns=["Cholesterol"])
    transformer = EngineeredFeatureTransformer(include=["cholesterol_age_ratio"])
    with pytest.raises(MissingFeatureColumnError):
        transformer.fit(frame)


def test_transformer_selection_is_frozen_at_fit(feature_frame):
    transformer = EngineeredFeatureTransformer(include=["hr_reserve"])
    transformer.fit(feature_frame)
    assert transformer.selected_transforms_ == ("hr_reserve",)
    assert transformer.n_features_in_ == feature_frame.shape[1]


def test_transformer_feature_names_out_lists_base_then_engineered(feature_frame):
    transformer = EngineeredFeatureTransformer(include=["hr_reserve", "age_band"])
    transformer.fit(feature_frame)
    names = list(transformer.get_feature_names_out())
    assert names == [str(c) for c in feature_frame.columns] + [
        "HR_Reserve",
        "AgeBand",
    ]


def test_transformer_composes_in_a_pipeline(feature_frame):
    numeric_transforms = [
        name
        for name, transform in ENGINEERED_TRANSFORMS.items()
        if transform.kind == NUMERIC_KIND
    ]
    pipeline = Pipeline(
        steps=[
            (
                "engineer",
                EngineeredFeatureTransformer(
                    include=numeric_transforms, keep_original=False
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    transformed = pipeline.fit_transform(feature_frame)
    expected_columns = selected_columns_by_kind(NUMERIC_KIND)
    assert transformed.shape == (len(feature_frame), len(expected_columns))
    assert np.all(np.isfinite(transformed))


# ---------------------------------------------------------------------------
# Reports and real-data integration
# ---------------------------------------------------------------------------


def test_describe_transforms_mentions_every_transform():
    report = describe_transforms()
    for name, transform in ENGINEERED_TRANSFORMS.items():
        assert name in report
        assert transform.produces[0] in report
        assert transform.rationale[:40] in report


def test_engineers_the_real_s01_training_frame():
    """Integration: runs on the git-tracked S01 train split after imputation."""
    train_frame, _ = load_split_frames(SPLIT_VERSION)
    # Apply the declared zero-as-missing median policy first so the denominator
    # precondition holds; this mirrors the intended pipeline order
    # (impute -> engineer). The imputer is fit on this training frame only.
    imputer = ZeroMedianImputer(
        columns=tuple(DEFAULT_ZERO_POLICY.columns),
        sentinel=DEFAULT_ZERO_POLICY.sentinel,
    )
    imputed = imputer.fit_transform(train_frame)

    result = engineer_features(imputed)
    assert len(result) == len(train_frame)
    for column in ENGINEERED_COLUMN_NAMES:
        assert column in result.columns
        assert result[column].notna().all()
    numeric = selected_columns_by_kind(NUMERIC_KIND)
    assert np.isfinite(result[list(numeric)].to_numpy(dtype=float)).all()
