"""Tests for the treatment-audit engine (M006/S01/T02).

The contract under test is the audit layer:
:func:`heart.reporting.data_methods.audit.build_treatment_audit` and
:func:`...build_declared_plan` must build the JSON payloads for report parts
B, D, F where

* every extracted value equals the *source constant* it claims to come from
  (heart.data.split, heart.data.load, heart.data.quality,
  heart.data.pipeline, heart.models.registry, heart.eval.treatment_declared)
  — never a fixture literal copied into an expected value,
* every battery member appears exactly once, sorted by model_type,
* linear-family members (D028 scope) carry the deep block and robust members
  carry a non-empty robustness justification,
* the fitted train-side statistics on a fixture frame equal hand-computed
  fixture medians and a freshly fitted pipeline's expanded feature naming,
* payloads round-trip through JSON, and
* building twice (or writing into two different directories) is byte-
  identical, and the atomic writer never leaves a partial artifact behind.

Every write in this module lands under ``tmp_path`` — never the committed
``reports/`` tree — and the fixture frame is the inline ``make_frame`` model
of the raw dataset (the same pattern as ``tests/test_data_methods_report.py``).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from heart.data.load import PINNED_SHA256
from heart.data.pipeline import (
    PIPELINE_CATEGORICAL_COLUMNS,
    PIPELINE_NUMERIC_COLUMNS,
    build_preprocessing_pipeline,
)
from heart.data.quality import DEFAULT_ZERO_POLICY
from heart.data.split import (
    N_SPLITS,
    SPLIT_VERSION,
    TEST_FOLD,
    SplitLeakageError,
    assert_no_group_straddle,
)
from heart.eval.treatment_declared import (
    DECLARED_CANDIDATES,
    DECLARED_CANDIDATE_KEYS,
    IMBALANCE_LADDER,
    LINEAR_FAMILY_SCOPE,
    PAIRED_SIGNIFICANCE_ALPHA,
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    GATE_WINNER_STANDING_RULE,
    gate_text,
)
from heart.models.registry import BATTERY_MODEL_TYPES, BATTERY_SPECS
from heart.reporting.data_methods.audit import (
    AUDIT_FILENAME,
    PLAN_FILENAME,
    ROBUSTNESS_JUSTIFICATIONS,
    TREATMENT_ARTIFACTS,
    TreatmentAuditError,
    build_declared_plan,
    build_treatment_audit,
    write_treatment_artifacts,
)


# ---------------------------------------------------------------------------
# Fixture frame (compact model of the raw dataset)
# ---------------------------------------------------------------------------


def make_frame(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Deterministic fixture frame in the raw dataset's column shape."""
    rng = np.random.default_rng(seed)
    severity = np.linspace(0.0, 1.0, n)
    cholesterol = rng.integers(120, 400, size=n)
    cholesterol[:4] = 0  # declared sentinels (zero-as-missing policy)
    restingbp = rng.integers(90, 190, size=n)
    restingbp[2] = 0  # declared sentinel
    return pd.DataFrame(
        {
            "Age": rng.integers(30, 81, size=n),
            "Sex": rng.choice(["M", "F"], size=n),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], size=n),
            "RestingBP": restingbp,
            "Cholesterol": cholesterol,
            "FastingBS": rng.integers(0, 2, size=n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], size=n),
            "MaxHR": rng.integers(90, 202, size=n),
            "ExerciseAngina": np.where(rng.random(n) < severity, "Y", "N"),
            "Oldpeak": np.round(severity * 4.0, 2),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], size=n),
            "HeartDisease": np.where(rng.random(n) < severity, 1, 0),
        }
    )


@pytest.fixture(scope="module")
def audit():
    """Build the audit payload once for the module's read-only assertions."""
    return build_treatment_audit(make_frame())


@pytest.fixture(scope="module")
def plan():
    """Build the declared-plan payload once."""
    return build_declared_plan()


# ---------------------------------------------------------------------------
# Provenance: split / loader / zero-policy extraction equals source constants
# ---------------------------------------------------------------------------


def test_split_provenance_equals_source_constants(audit):
    assert audit["split"]["version"] == SPLIT_VERSION
    assert audit["split"]["n_splits"] == N_SPLITS
    assert audit["split"]["test_fold"] == TEST_FOLD


def test_group_straddle_policy_names_the_real_callables(audit):
    policy = audit["split"]["group_straddle_policy"]
    # The recorded policy string must name the actual failure function and
    # exception — extracted by attribute, not transcribed.
    assert assert_no_group_straddle.__name__ in policy
    assert SplitLeakageError.__name__ in policy


def test_loader_provenance_equals_pinned_sha256(audit):
    assert audit["loader"]["pinned_sha256"] == PINNED_SHA256


def test_zero_policy_equals_default_policy(audit):
    assert audit["zero_policy"]["action"] == DEFAULT_ZERO_POLICY.action
    assert audit["zero_policy"]["columns"] == list(DEFAULT_ZERO_POLICY.columns)
    assert audit["zero_policy"]["sentinel"] == DEFAULT_ZERO_POLICY.sentinel


def test_pipeline_columns_equal_pipeline_module_constants(audit):
    assert audit["preprocessing"]["numeric_columns"] == list(
        PIPELINE_NUMERIC_COLUMNS
    )
    assert audit["preprocessing"]["categorical_columns"] == list(
        PIPELINE_CATEGORICAL_COLUMNS
    )


# ---------------------------------------------------------------------------
# Provenance: per-member registry extraction
# ---------------------------------------------------------------------------


def test_every_battery_member_present_and_sorted(audit):
    model_types = [entry["model_type"] for entry in audit["models"]]
    assert model_types == sorted(BATTERY_MODEL_TYPES)


def test_member_registry_fields_equal_battery_specs(audit):
    for entry in audit["models"]:
        spec = BATTERY_SPECS[entry["model_type"]]
        assert entry["model_name"] == spec.model_name
        assert entry["family"] == spec.family
        assert entry["preprocessing"] == spec.preprocessing
        assert entry["tuned_space_keys"] == sorted(spec.declared_hyperparameters)
        assert entry["fixed_params"] == {
            str(key): value for key, value in sorted(spec.fixed_params.items())
        }
        assert entry["estimator_description"] == spec.description


def test_one_hot_encoder_params_extracted_from_actual_pipeline(audit):
    """The recorded encoder params must equal the ones the real chain builds."""
    fresh = build_preprocessing_pipeline()
    fresh.fit(make_frame())
    encoder = fresh.named_steps["features"].named_transformers_["categorical"]
    actual = {key: value for key, value in encoder.get_params().items()}
    steps = audit["preprocessing"]["pipeline_steps"]
    features_step = next(step for step in steps if step["step"] == "features")
    categorical = next(
        sub for sub in features_step["sub_transformers"] if sub["name"] == "categorical"
    )
    assert categorical["transformer"] == "OneHotEncoder"
    assert categorical["params"]["handle_unknown"] == actual["handle_unknown"]
    assert categorical["params"]["sparse_output"] == actual["sparse_output"]
    assert categorical["params"]["drop"] == actual["drop"]
    assert categorical["columns"] == [str(c) for c in PIPELINE_CATEGORICAL_COLUMNS]
    assert categorical["columns"] == [str(c) for c in encoder.feature_names_in_]


def test_numeric_scaler_is_standard_scaler(audit):
    steps = audit["preprocessing"]["pipeline_steps"]
    features_step = next(step for step in steps if step["step"] == "features")
    numeric = next(
        sub for sub in features_step["sub_transformers"] if sub["name"] == "numeric"
    )
    assert numeric["transformer"] == "StandardScaler"


def test_pipeline_step_names_match_the_real_chain(audit):
    real_names = [name for name, _ in build_preprocessing_pipeline().steps]
    recorded_names = [step["step"] for step in audit["preprocessing"]["pipeline_steps"]]
    assert recorded_names == real_names


# ---------------------------------------------------------------------------
# Depth tiers (D028)
# ---------------------------------------------------------------------------


def test_linear_family_members_carry_deep_block(audit):
    for entry in audit["models"]:
        if entry["model_type"] in LINEAR_FAMILY_SCOPE:
            assert entry["depth"] == "deep"
            assert [step["step"] for step in entry["pipeline_steps"]] == [
                "zero_policy",
                "features",
            ]
            assert entry["expanded_feature_count"] > 0
            assert entry["expanded_feature_names"]
        else:
            assert entry["depth"] == "light"
            assert "pipeline_steps" not in entry


def test_robust_members_carry_robustness_justification(audit):
    for entry in audit["models"]:
        if entry["model_type"] in LINEAR_FAMILY_SCOPE:
            continue
        justification = entry["robustness_justification"]
        assert isinstance(justification, str)
        assert justification.strip() == justification == justification.strip()
        assert len(justification) > 40
        # The justification must be the declared constant, not an ad-hoc string.
        assert justification == ROBUSTNESS_JUSTIFICATIONS[entry["model_type"]]


def test_depth_tier_partitions_match_declared_scope(audit):
    assert audit["depth_tiers"]["deep"] == sorted(
        model for model in BATTERY_MODEL_TYPES if model in LINEAR_FAMILY_SCOPE
    )
    assert audit["depth_tiers"]["light"] == sorted(
        model for model in BATTERY_MODEL_TYPES if model not in LINEAR_FAMILY_SCOPE
    )


def test_unjustified_robust_member_fails_loudly(monkeypatch):
    """A battery member outside the linear scope without a justification
    must raise the named audit error, not render unaudited."""
    from heart.models import registry as registry_module
    from heart.reporting.data_methods import audit as audit_module

    fake_spec = registry_module.BATTERY_SPECS["knn"]
    monkeypatch.setattr(
        audit_module,
        "BATTERY_MODEL_TYPES",
        (*BATTERY_MODEL_TYPES, "throwaway-robust"),
    )
    monkeypatch.setattr(
        audit_module,
        "BATTERY_SPECS",
        {**audit_module.BATTERY_SPECS, "throwaway-robust": fake_spec},
    )
    with pytest.raises(TreatmentAuditError, match="throwaway-robust"):
        build_treatment_audit(make_frame(n=24))


# ---------------------------------------------------------------------------
# Fitted train-side statistics on the fixture frame
# ---------------------------------------------------------------------------


def test_fitted_medians_equal_hand_computed_fixture_medians(audit):
    frame = make_frame()
    medians = audit["preprocessing"]["fitted"]["zero_medians"]
    for column in DEFAULT_ZERO_POLICY.columns:
        hand_computed = float(
            frame.loc[frame[column] != DEFAULT_ZERO_POLICY.sentinel, column].median()
        )
        assert medians[column] == hand_computed
    assert audit["preprocessing"]["fitted"]["row_count"] == len(frame)


def test_expanded_feature_count_matches_fresh_fit(audit):
    frame = make_frame()
    fresh = build_preprocessing_pipeline()
    fresh.fit(frame)
    expected_count = len(fresh.get_feature_names_out())
    expected_names = [str(name) for name in fresh.get_feature_names_out()]
    assert audit["preprocessing"]["fitted"]["expanded_feature_count"] == expected_count
    for entry in audit["models"]:
        if entry["depth"] == "deep":
            assert entry["expanded_feature_count"] == expected_count
            assert entry["expanded_feature_names"] == expected_names


def test_expanded_feature_count_equals_columns_plus_category_levels(audit):
    """Independent recomputation: numeric width + sum of observed level counts."""
    frame = make_frame()
    expected = len(PIPELINE_NUMERIC_COLUMNS) + sum(
        frame[column].nunique() for column in PIPELINE_CATEGORICAL_COLUMNS
    )
    assert audit["preprocessing"]["fitted"]["expanded_feature_count"] == expected


def test_encoder_categories_match_fixture_levels(audit):
    frame = make_frame()
    recorded = audit["preprocessing"]["fitted"]["encoder_categories"]
    for categories, column in zip(recorded, PIPELINE_CATEGORICAL_COLUMNS):
        assert sorted(str(value) for value in categories) == sorted(
            str(value) for value in frame[column].unique()
        )


def test_scaler_statistics_present_and_finite(audit):
    fitted = audit["preprocessing"]["fitted"]
    assert len(fitted["scaler_mean"]) == len(PIPELINE_NUMERIC_COLUMNS)
    assert len(fitted["scaler_scale"]) == len(PIPELINE_NUMERIC_COLUMNS)
    assert all(np.isfinite(value) for value in fitted["scaler_mean"])
    assert all(np.isfinite(value) for value in fitted["scaler_scale"])


# ---------------------------------------------------------------------------
# Declared plan (parts D and F mirrored from the T01 module)
# ---------------------------------------------------------------------------


def test_declared_plan_marks_antecedence(plan):
    assert plan["declared_before_any_candidate_number"] is True


def test_declared_plan_candidates_mirror_the_module(plan):
    keys = [candidate["variant_key"] for candidate in plan["candidates"]]
    assert keys == list(DECLARED_CANDIDATE_KEYS)
    assert len(plan["candidates"]) == len(DECLARED_CANDIDATES)
    for rendered, candidate in zip(plan["candidates"], DECLARED_CANDIDATES):
        assert rendered["mechanism"] == candidate.mechanism
        assert rendered["group"] == candidate.group
        assert rendered["applies_to"] == list(candidate.applies_to)
        assert rendered["description"] == candidate.description
        assert rendered["order"] == candidate.order
        assert rendered["is_incumbent"] == candidate.is_incumbent


def test_declared_plan_ladder_mirrors_the_module(plan):
    rendered_rungs = [rung["rung"] for rung in plan["imbalance_ladder"]]
    assert rendered_rungs == [rung.rung for rung in IMBALANCE_LADDER]
    for rendered, rung in zip(plan["imbalance_ladder"], IMBALANCE_LADDER):
        assert rendered["label"] == rung.label
        assert rendered["variant_key"] == rung.variant_key
    assert rendered_rungs == list(range(len(IMBALANCE_LADDER)))


def test_declared_plan_gate_constants_mirror_the_module(plan):
    gate = plan["gate"]
    assert gate["metric"] == GATE_METRIC
    assert gate["min_delta_roc_auc"] == GATE_MIN_DELTA_ROC_AUC
    assert gate["paired_significance_alpha"] == PAIRED_SIGNIFICANCE_ALPHA
    assert gate["winner_standing_rule"] == GATE_WINNER_STANDING_RULE
    assert gate["text"] == gate_text()


def test_declared_plan_has_no_timestamp_key(plan):
    """Determinism contract: no timestamp/generated-at key may ever appear."""
    forbidden_markers = ("timestamp", "generated_at", "created_at", "as_of", "run_at")
    for payload in (plan, build_treatment_audit(make_frame(n=24))):
        offenders = {
            key
            for key in payload
            if any(marker in key.lower() for marker in forbidden_markers)
        }
        assert not offenders


# ---------------------------------------------------------------------------
# Determinism and JSON round-trip
# ---------------------------------------------------------------------------


def test_build_treatment_audit_is_deterministic():
    first = build_treatment_audit(make_frame())
    second = build_treatment_audit(make_frame())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_declared_plan_is_deterministic():
    assert build_declared_plan() == build_declared_plan()


def test_payloads_round_trip_through_json():
    for payload in (build_treatment_audit(make_frame()), build_declared_plan()):
        restored = json.loads(json.dumps(payload, sort_keys=True))
        assert restored == payload


def test_writing_twice_into_different_dirs_yields_identical_bytes(tmp_path):
    audit_payload = build_treatment_audit(make_frame())
    plan_payload = build_declared_plan()
    dirs = [tmp_path / "run-a", tmp_path / "run-b"]
    for directory in dirs:
        write_treatment_artifacts(audit_payload, plan_payload, directory)
    for filename in (AUDIT_FILENAME, PLAN_FILENAME):
        assert (dirs[0] / filename).read_bytes() == (dirs[1] / filename).read_bytes()


def test_written_artifacts_are_the_declared_basenames(tmp_path):
    written = write_treatment_artifacts(
        build_treatment_audit(make_frame()), build_declared_plan(), tmp_path
    )
    assert written == [
        TREATMENT_ARTIFACTS["treatment_audit"],
        TREATMENT_ARTIFACTS["treatment_plan"],
    ]
    assert (tmp_path / AUDIT_FILENAME).is_file()
    assert (tmp_path / PLAN_FILENAME).is_file()


# ---------------------------------------------------------------------------
# Negative surface: builder and atomic writer failure paths
# ---------------------------------------------------------------------------


def test_non_frame_argument_raises_named_exception():
    with pytest.raises(TreatmentAuditError, match="pandas.DataFrame"):
        build_treatment_audit([1, 2, 3])


def test_frame_missing_feature_column_raises_named_exception():
    frame = make_frame().drop(columns=["Oldpeak"])
    with pytest.raises(TreatmentAuditError, match="Oldpeak"):
        build_treatment_audit(frame)


def test_all_sentinel_policy_column_raises_named_exception():
    frame = make_frame()
    frame["RestingBP"] = 0  # no non-sentinel value -> no median possible
    with pytest.raises(TreatmentAuditError, match="RestingBP"):
        build_treatment_audit(frame)


def test_non_serializable_payload_raises_and_writes_nothing(tmp_path):
    plan_payload = build_declared_plan()
    plan_payload["unserializable"] = {1, 2, 3}  # a set() breaks json.dumps
    with pytest.raises(TreatmentAuditError, match="JSON-serializable"):
        write_treatment_artifacts(build_treatment_audit(make_frame()), plan_payload, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_leaves_no_tmp_file_when_replace_fails(tmp_path, monkeypatch):
    """A failed os.replace must remove its tmp file and leave no partial artifact."""
    real_replace = os.replace

    def failing_replace(src, dst):
        if dst.name == PLAN_FILENAME:
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_treatment_artifacts(
            build_treatment_audit(make_frame()), build_declared_plan(), tmp_path
        )
    # The first artifact completed; the second never landed and no .tmp
    # residue survives anywhere in the output directory.
    assert (tmp_path / AUDIT_FILENAME).is_file()
    assert not (tmp_path / PLAN_FILENAME).exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_non_dict_payloads_raise_named_exception(tmp_path):
    with pytest.raises(TreatmentAuditError, match="audit payload"):
        write_treatment_artifacts(["not", "a", "dict"], build_declared_plan(), tmp_path)
    with pytest.raises(TreatmentAuditError, match="declared-plan payload"):
        write_treatment_artifacts(build_treatment_audit(make_frame()), None, tmp_path)
    assert list(tmp_path.iterdir()) == []
