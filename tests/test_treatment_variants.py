"""Tests for the declared treatment-candidate variant builders (M006/S02 T03).

Contract pinned here (mirrors the task plan):

1. Every declared non-incumbent candidate key builds without error for every
   model_type in its ``applies_to``; each built object refits on the fixture
   train split and predicts / predicts_proba on the test split with aligned
   shapes.
2. Mechanism/group consistency: the declared mechanism routes to the right
   builder (sampling -> imblearn Pipeline; pipeline-variant -> sklearn
   Pipeline; estimator-param -> committed chain + balanced LR) and the
   declared group matches exactly which chain component is mutated — the
   variant differs from the incumbent ONLY in the declared mechanism.
3. Family peculiarities: sampling candidates are imblearn Pipelines with the
   sampler strictly between preprocess and classifier (golden-rule position);
   drop-first reduces expanded categorical columns vs the committed chain;
   the missingness-flag variant adds exactly one column per missing-carrying
   feature; box-cox provably excludes non-positive numeric columns.
4. Registry battery cross-check: BATTERY_MEMBERS == registry.BATTERY_MODEL_TYPES.
5. Negative surface: unknown variant_key, incumbent-not-built, sampler applied
   outside applies_to, and mechanism/sampler mapping gaps all raise
   TreatmentGateError.

Fixture corpus: tests/helpers_treatment.py (n=240, ~15% minority, ~8% NaNs in
one numeric column, one 3-level categorical feature). No repeated CV here.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS, NUMERIC_COLUMNS
from heart.eval import treatment_variants as tv
from heart.eval.treatment_declared import (
    BATTERY_MEMBERS,
    DECLARED_CANDIDATES,
    INCUMBENT_CANDIDATE,
    TreatmentCandidate,
)
from heart.models.registry import BATTERY_MODEL_TYPES
from tests.helpers_treatment import (
    NAN_COLUMN,
    fixture_missing_columns,
    make_treatment_fixture,
    treatment_train_test_split,
)


@lru_cache(maxsize=1)
def _fixture_split():
    X, y = make_treatment_fixture()
    return treatment_train_test_split(X, y)


def _fixture():
    X_train, X_test, y_train, y_test = _fixture_split()
    return X_train, X_test, y_train, y_test.copy()


# --------------------------------------------------------------------------
# 1. Fixture corpus sanity
# --------------------------------------------------------------------------


def test_fixture_is_deterministic_and_imbalanced() -> None:
    X, y = make_treatment_fixture()
    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert len(X) == 240
    assert int(y.sum()) == 36 and 0.14 <= y.mean() <= 0.16
    X2, y2 = make_treatment_fixture()
    assert np.array_equal(y, y2)
    pd.testing.assert_frame_equal(X, X2)


def test_fixture_carries_nans_in_exactly_one_numeric_column() -> None:
    X, _ = make_treatment_fixture()
    missing = fixture_missing_columns(X)
    assert missing == (NAN_COLUMN,)
    nan_columns = [c for c in NUMERIC_COLUMNS if X[c].isna().any()]
    assert nan_columns == [NAN_COLUMN]
    # ~8% scattered
    assert 0.05 <= X[NAN_COLUMN].isna().mean() <= 0.11


def test_fixture_has_a_three_level_categorical_feature() -> None:
    X, _ = make_treatment_fixture()
    levels = set(X["ChestPainType"].unique())
    assert len(levels) == 3


# --------------------------------------------------------------------------
# 2. Every non-incumbent candidate builds and fits for every applies_to model
# --------------------------------------------------------------------------


def _fit_params(model_type: str) -> dict:
    """Deterministic params for fixture fits that need a variance floor.

    QDA with reg_param=0.0 is singular on any one-hot-expanded feature matrix
    (the level indicators are linearly dependent by construction); the fixture
    fits therefore pin a small reg_param via the builder's params argument.
    """
    return {"reg_param": 0.1} if model_type == "qda" else None


def test_every_declared_candidate_builds_fits_and_predicts() -> None:
    X_train, X_test, y_train, y_test = _fixture()
    built = 0
    for candidate in tv.iter_buildable_candidates():
        for model_type in candidate.applies_to:
            model = tv.build_variant(
                candidate.variant_key, model_type, params=_fit_params(model_type)
            )
            assert hasattr(model, "fit") and hasattr(model, "predict")
            assert hasattr(model, "predict_proba")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)
            assert pred.shape == (len(X_test),), (candidate.variant_key, model_type)
            assert proba.shape == (len(X_test), 2), (candidate.variant_key, model_type)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
            assert set(np.unique(pred)) <= {0, 1}
            built += 1
    # 9 linear-family candidates x 5 models + class-weight-balanced x 3
    assert built == 9 * 5 + 3 == 48


def test_builders_produce_fresh_unfitted_objects() -> None:
    first = tv.build_variant("knn-imputer", "lda")
    second = tv.build_variant("knn-imputer", "lda")
    assert first is not second
    first_preprocess = first.named_steps["preprocess"]
    second_preprocess = second.named_steps["preprocess"]
    assert first_preprocess is not second_preprocess


# --------------------------------------------------------------------------
# 3. Mechanism/group routing
# --------------------------------------------------------------------------


def _iter_transformers(estimator: object):
    stack = [estimator]
    while stack:
        current = stack.pop()
        yield current
        steps = getattr(current, "steps", None)
        if steps:
            stack.extend(step for _, step in steps)
        named = getattr(current, "transformers_", None)
        if named:
            stack.extend(step for _, step, _ in named)


def test_mechanism_routing_by_declared_mechanism() -> None:
    X_train, X_test, y_train, y_test = _fixture()
    for candidate in tv.iter_buildable_candidates():
        model_type = candidate.applies_to[0]
        model = tv.build_variant(candidate.variant_key, model_type)
        if candidate.mechanism == "sampling":
            assert isinstance(model, ImblearnPipeline), candidate.variant_key
            tv.assert_sampler_between_preprocess_and_classifier(
                model, candidate.variant_key
            )
        else:
            assert isinstance(model, SklearnPipeline), candidate.variant_key
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2)


def test_group_routes_to_the_mutated_chain_component() -> None:
    X_train, X_test, y_train, y_test = _fixture()

    # transform group -> PowerTransformer with the declared method.
    for variant_key, method in (
        ("yeo-johnson-power-transform", "yeo-johnson"),
        ("box-cox-power-transform", "box-cox"),
    ):
        model = tv.build_variant(variant_key, "logistic-regression-l2")
        model.fit(X_train, y_train)
        subset = next(
            step
            for step in _iter_transformers(model)
            if isinstance(step, tv._ColumnSubsetPowerTransformer)
        )
        assert subset.method == method, variant_key
        assert isinstance(subset.power_transformer_, PowerTransformer)

    # imputation group (knn) -> KNNImputer replaces the median imputer.
    model = tv.build_variant("knn-imputer", "lda")
    model.fit(X_train, y_train)
    knn = next(
        step
        for step in _iter_transformers(model)
        if isinstance(step, KNNImputer)
    )
    assert knn.n_neighbors == tv.KNN_IMPUTER_NEIGHBORS == 5
    assert not any(
        isinstance(step, type(tv.committed_zero_policy_imputer()))
        for step in _iter_transformers(model)
    ), "knn-imputer variant must REPLACE the committed median imputer"

    # imputation group (flags) -> adder prepended, committed imputation kept.
    model = tv.build_variant("missingness-flag-columns", "lda")
    model.fit(X_train, y_train)
    adder = next(
        step
        for step in _iter_transformers(model)
        if isinstance(step, tv.MissingnessFlagAdder)
    )
    assert adder.flag_columns_ == ("Cholesterol",)
    assert any(
        isinstance(step, type(tv.committed_zero_policy_imputer()))
        for step in _iter_transformers(model)
    ), "missingness-flag variant must keep the committed imputation"
    model.predict_proba(X_test)  # flags route through the CT without error

    # encoding group -> OneHotEncoder(drop="first") inside the CT.
    model = tv.build_variant("one-hot-drop-first", "lda")
    model.fit(X_train, y_train)
    encoder = next(
        step
        for step in _iter_transformers(model)
        if isinstance(step, OneHotEncoder)
    )
    assert encoder.drop == "first" and encoder.handle_unknown == "ignore"

    # imbalance estimator-param -> clf class_weight pinned, chain unchanged.
    model = tv.build_variant("class-weight-balanced", "logistic-regression-l2")
    clf = model.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.get_params()["class_weight"] == "balanced"
    spec = tv.resolve_spec("logistic-regression-l2")
    for fixed_key, fixed_value in spec.fixed_params.items():
        assert clf.get_params()[fixed_key] == fixed_value
    assert type(model.named_steps["preprocess"]) is SklearnPipeline


# --------------------------------------------------------------------------
# 4. Family peculiarities
# --------------------------------------------------------------------------


def _expanded_column_count(chain, X: pd.DataFrame) -> int:
    fitted = chain.fit(X)
    return len(fitted.get_feature_names_out())


def test_drop_first_reduces_expanded_categorical_columns() -> None:
    X_train, _, _, _ = _fixture()
    committed = _expanded_column_count(build_preprocessing_pipeline(), X_train)
    variant_chain = tv.build_variant("one-hot-drop-first", "lda").named_steps[
        "preprocess"
    ]
    drop_first = _expanded_column_count(variant_chain, X_train)
    assert drop_first < committed
    # exactly one column less per categorical feature
    n_categorical = 6
    assert committed - drop_first == n_categorical


def test_missingness_flag_adds_exactly_one_column_per_missing_feature() -> None:
    X_train, _, _, _ = _fixture()
    expected_missing = fixture_missing_columns(X_train)
    variant_chain = tv.build_variant(
        "missingness-flag-columns", "lda"
    ).named_steps["preprocess"]
    fitted = variant_chain.fit(X_train)
    names = [str(name) for name in fitted.get_feature_names_out()]
    flags = [name for name in names if name.startswith("missingness_flags__")]
    assert len(flags) == len(expected_missing) == 1
    assert flags == ["missingness_flags__missing_flag_Cholesterol"]


def test_box_cox_excludes_non_positive_columns_on_the_fixture() -> None:
    X_train, _, y_train, _ = _fixture()
    model = tv.build_variant("box-cox-power-transform", "logistic-regression-l2")
    model.fit(X_train, y_train)
    subset = next(
        step
        for step in _iter_transformers(model)
        if isinstance(step, tv._ColumnSubsetPowerTransformer)
    )
    assert "Oldpeak" not in subset.selected_columns_  # carries zeros
    assert "Cholesterol" in subset.selected_columns_  # strictly positive


def test_power_variants_keep_scaling_and_one_hot_encoding() -> None:
    X_train, _, y_train, _ = _fixture()
    for variant_key in (
        "yeo-johnson-power-transform",
        "box-cox-power-transform",
        "knn-imputer",
    ):
        model = tv.build_variant(variant_key, "logistic-regression-l2")
        model.fit(X_train, y_train)
        preprocess = model.named_steps["preprocess"]
        assert any(
            isinstance(step, StandardScaler) for step in _iter_transformers(preprocess)
        ), variant_key
        assert any(
            isinstance(step, OneHotEncoder) for step in _iter_transformers(preprocess)
        ), variant_key


# --------------------------------------------------------------------------
# 5. Registry cross-check + incumbent handling
# --------------------------------------------------------------------------


def test_battery_members_match_registry() -> None:
    assert BATTERY_MEMBERS == BATTERY_MODEL_TYPES
    tv.validate_variant_registry()  # loud no-op on aligned registries


def test_incumbent_is_documented_not_built() -> None:
    with pytest.raises(tv.TreatmentGateError, match="incumbent"):
        tv.build_variant(INCUMBENT_CANDIDATE.variant_key, "lda")


def test_incumbent_factory_returns_the_committed_pipeline() -> None:
    model = tv.incumbent_factory("logistic-regression-l2")
    assert isinstance(model, SklearnPipeline)
    assert [name for name, _ in model.steps] == ["preprocess", "clf"]
    preprocess = model.named_steps["preprocess"]
    assert [name for name, _ in preprocess.steps] == [
        "zero_policy",
        "features",
    ]


# --------------------------------------------------------------------------
# 6. Negative surface
# --------------------------------------------------------------------------


def test_unknown_variant_key_raises() -> None:
    with pytest.raises(tv.TreatmentGateError, match="Unknown treatment variant"):
        tv.build_variant("not-a-declared-variant", "lda")


def test_sampling_candidate_outside_applies_to_raises() -> None:
    sampling = next(
        candidate
        for candidate in DECLARED_CANDIDATES
        if candidate.mechanism == "sampling"
    )
    for model_type in BATTERY_MODEL_TYPES:
        if model_type not in sampling.applies_to:
            with pytest.raises(tv.TreatmentGateError, match="does not apply to"):
                tv.build_candidate(sampling, model_type)
            return
    pytest.fail("no battery model outside the sampling applies_to")


def test_mechanism_sampler_mapping_gap_raises() -> None:
    # Hand-built declared candidate with a sampling mechanism but no mapped
    # sampler: the build must fail loudly instead of silently mis-measuring.
    phantom = TreatmentCandidate(
        variant_key="smote-svf-neighbors",
        mechanism="sampling",
        group="imbalance",
        applies_to=("lda",),
        description="A phantom sampling candidate with no mapped sampler.",
        mlflow_treatment_variant="smote-svf-neighbors",
        order=99,
    )
    with pytest.raises(tv.TreatmentGateError, match="mapping gap"):
        tv.build_candidate(phantom, "lda")


def test_pipeline_variant_mapping_gap_raises() -> None:
    phantom = TreatmentCandidate(
        variant_key="quantile-transform",
        mechanism="pipeline-variant",
        group="transform",
        applies_to=("lda",),
        description="A phantom pipeline-variant with no mapped mutation.",
        mlflow_treatment_variant="quantile-transform",
        order=100,
    )
    with pytest.raises(tv.TreatmentGateError, match="mapping gap"):
        tv.build_candidate(phantom, "lda")


def test_estimated_param_outside_scope_raises() -> None:
    balanced = next(
        candidate
        for candidate in DECLARED_CANDIDATES
        if candidate.mechanism == "estimator-param"
    )
    with pytest.raises(tv.TreatmentGateError, match="does not apply to"):
        tv.build_candidate(balanced, "knn")


def test_samplers_are_seeded_from_the_project_seed() -> None:
    for key in tv.SAMPLER_FACTORIES:
        sampler = tv.sampler_for_variant(key)
        assert sampler.random_state == 42, key


# --------------------------------------------------------------------------
# 7. Determinism under RANDOM_SEED
# --------------------------------------------------------------------------


def test_variant_rebuilds_are_deterministic() -> None:
    X_train, X_test, y_train, y_test = _fixture()
    for variant_key, model_type in (
        ("knn-imputer", "logistic-regression-l2"),
        ("class-weight-balanced", "logistic-regression-l1"),
        ("smote-oversampling", "lda"),
        ("adasyn-oversampling", "logistic-regression-elasticnet"),
    ):
        predictions = []
        for _ in range(2):
            model = tv.build_variant(variant_key, model_type)
            model.fit(X_train, y_train)
            predictions.append(model.predict_proba(X_test))
        assert np.allclose(predictions[0], predictions[1]), variant_key


# --------------------------------------------------------------------------
# 8. Golden-rule position enforcement
# --------------------------------------------------------------------------


def test_sampler_position_violations_raise() -> None:
    sampler = tv.sampler_for_variant("smote-oversampling")
    lr = LogisticRegression(max_iter=200)
    preprocess = build_preprocessing_pipeline()
    misplaced = ImblearnPipeline(steps=[("sampler", sampler), ("clf", lr)])
    with pytest.raises(tv.TreatmentGateError, match="golden-rule position violated"):
        tv.assert_sampler_between_preprocess_and_classifier(misplaced, "smote-oversampling")
    wrong_middle = SklearnPipeline(
        steps=[
            ("preprocess", preprocess),
            ("scale", StandardScaler()),
            ("clf", lr),
        ]
    )
    with pytest.raises(tv.TreatmentGateError, match="exactly one fit_resample"):
        tv.assert_sampler_between_preprocess_and_classifier(wrong_middle, "smote-oversampling")


# --------------------------------------------------------------------------
# 9. Malformed feature frame
# --------------------------------------------------------------------------


def test_missing_feature_columns_raise_treatment_gate_error() -> None:
    X, _ = make_treatment_fixture()
    broken = X.drop(columns=["Age"])
    model = tv.build_variant("missingness-flag-columns", "lda")
    with pytest.raises(tv.TreatmentGateError, match="missing declared column"):
        model.fit(broken, np.zeros(len(broken), dtype=int))
