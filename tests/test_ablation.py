"""Tests for the feature-set ablation harness (S04/T02).

These tests pin the contract that makes an ablation trustable:

1. **Feature modes** are explicit, named selections over the T01 registry; an
   empty ``include`` is the baseline feature set, and an invalid selection
   fails loudly rather than silently changing the comparison.
2. **The baseline mode is the published baseline**: the same zero-policy,
   column-transformer and logistic-regression parameters, so a "baseline" row
   in the ablation report is directly comparable to ``reports/baseline.md``.
3. **Every mode runs through the shared evaluation contract** and is logged to
   MLflow tagged ``run_kind=ablation`` with the mode, its engineered columns,
   and the input feature count.
4. **The core ablation invariant** — a variant with no distinguishing features
   produces metrics identical to the baseline within tolerance — is asserted
   directly, because it is what proves the harness changes only the feature
   representation and nothing else.
5. **The negative surface fails loudly**: bad modes, duplicate mode names, bad
   frames, and failed builds/fits/evaluations are all recorded (fail-soft) and
   raise under ``strict=True``.

Fixtures are synthetic (no gitignored paths); two integration tests read the
git-tracked S01 split artifacts and use a ``tmp_path`` MLflow store.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.eval.contract import METRIC_KEYS, PRIMARY_METRIC
from heart.features.ablation import (
    ABLATION_RUN_KIND,
    FEATURE_MODE_TAG,
    N_FEATURES_TAG,
    RUN_KIND_TAG,
    AblationConfigError,
    AblationDataError,
    AblationError,
    AblationLedgerError,
    AblationRunError,
    FeatureMode,
    ablation_feature_counts,
    all_features_mode,
    baseline_mode,
    build_ablation_pipeline,
    default_variants,
    leave_one_out_modes,
    render_ablation_report,
    run_ablation,
    single_transform_modes,
    write_ablation_ledger,
)
from heart.features.engineering import ENGINEERED_COLUMN_NAMES, TRANSFORM_NAMES
from heart.models.baseline import build_baseline_model
from heart.tracking.run import (
    METRICS_ARTIFACT,
    RUN_CONFIG_ARTIFACT,
    load_metrics_artifact,
    load_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A synthetic frame covering every schema column and both target classes.

    The target is assigned by rank so it is exactly balanced regardless of the
    latent signal, which guarantees :func:`heart.eval.contract.evaluate` never
    hits the single-class path on the held-out split.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, n)
    sex = rng.choice(["M", "F"], n)
    chest_pain = rng.choice(["ASY", "ATA", "NAP", "TA"], n)
    resting_bp = rng.integers(100, 180, n)
    # A couple of impossible zeros exercise the zero-as-missing imputation.
    resting_bp[:2] = 0
    cholesterol = rng.integers(150, 320, n)
    cholesterol[:3] = 0
    fasting_bs = rng.integers(0, 2, n)
    resting_ecg = rng.choice(["Normal", "ST", "LVH"], n)
    max_hr = rng.integers(90, 190, n)
    exercise_angina = rng.choice(["N", "Y"], n)
    oldpeak = np.round(rng.uniform(0.0, 4.0, n), 1)
    st_slope = rng.choice(["Up", "Flat", "Down"], n)

    score = (
        0.04 * (age - 55)
        - 0.03 * (max_hr - 140)
        + 0.9 * (exercise_angina == "Y")
        - 0.5 * (st_slope == "Up")
        + 0.002 * (cholesterol - 220)
        + rng.normal(0.0, 1.0, n)
    )
    order = np.argsort(score)
    target = np.zeros(n, dtype=int)
    target[order[n // 2:]] = 1

    return pd.DataFrame(
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
            TARGET_COLUMN: target,
        }
    )


@pytest.fixture(scope="module")
def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A deterministic synthetic train/test pair with both classes on each side."""
    frame = _synthetic_frame()
    train = frame.iloc[:300].reset_index(drop=True)
    test = frame.iloc[300:].reset_index(drop=True)
    assert train[TARGET_COLUMN].nunique() == 2
    assert test[TARGET_COLUMN].nunique() == 2
    return train, test


@pytest.fixture(scope="module")
def real_split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The git-tracked S01 v1 split frames (integration tests only)."""
    return load_split_frames(SPLIT_VERSION)


# ---------------------------------------------------------------------------
# FeatureMode contract
# ---------------------------------------------------------------------------


def test_baseline_mode_is_the_raw_feature_set():
    mode = baseline_mode()
    assert mode.is_baseline is True
    assert mode.engineered_transforms == ()
    assert mode.engineered_columns == ()
    assert mode.n_engineered_columns == 0


def test_all_features_mode_enables_every_transform():
    mode = all_features_mode()
    assert mode.is_baseline is False
    assert mode.engineered_transforms == TRANSFORM_NAMES
    assert mode.engineered_columns == ENGINEERED_COLUMN_NAMES


def test_single_transform_modes_cover_the_registry_individually():
    modes = single_transform_modes()
    assert len(modes) == len(TRANSFORM_NAMES)
    for mode, name in zip(modes, TRANSFORM_NAMES):
        assert mode.engineered_transforms == (name,)
        assert mode.name == f"add-{name}"


def test_leave_one_out_modes_drop_exactly_one_transform():
    modes = leave_one_out_modes()
    assert len(modes) == len(TRANSFORM_NAMES)
    for mode, dropped in zip(modes, TRANSFORM_NAMES):
        assert dropped not in mode.engineered_transforms
        assert len(mode.engineered_transforms) == len(TRANSFORM_NAMES) - 1
        assert mode.name == f"without-{dropped}"


def test_default_variants_have_unique_names_and_cover_the_registry():
    modes = default_variants()
    names = [mode.name for mode in modes]
    assert len(names) == len(set(names))
    assert names[0] == "baseline"
    assert names[-1] == "all-engineered"
    assert {mode.engineered_transforms for mode in modes} >= {
        (name,) for name in TRANSFORM_NAMES
    }


def test_feature_mode_from_include_resolves_names_and_columns():
    mode = FeatureMode.from_include(
        "add-two", ["hr_reserve", "age_band"], description="two features"
    )
    assert mode.engineered_transforms == ("hr_reserve", "age_band")
    assert mode.engineered_columns == ("HR_Reserve", "AgeBand")
    assert mode.name == "add-two"


def test_feature_mode_to_dict_is_serialisable():
    payload = FeatureMode.from_include("add", ["hr_reserve"]).to_dict()
    assert payload["engineered_transforms"] == ["hr_reserve"]
    assert payload["engineered_columns"] == ["HR_Reserve"]
    assert payload["is_baseline"] is False
    json.dumps(payload)


def test_feature_mode_rejects_blank_name():
    with pytest.raises(AblationConfigError):
        FeatureMode(name="   ")


def test_feature_mode_rejects_unknown_transform():
    with pytest.raises(AblationConfigError):
        FeatureMode.from_include("bad", ["not-a-transform"])


def test_feature_mode_rejects_exclude_without_include():
    with pytest.raises(AblationConfigError):
        FeatureMode(name="empty", include=(), exclude=("age_band",))


def test_feature_mode_rejects_non_string_members():
    with pytest.raises(AblationConfigError):
        FeatureMode(name="bad", include=(1,))  # type: ignore[arg-type]


def test_leave_one_out_rejects_unknown_transform():
    with pytest.raises(AblationConfigError):
        FeatureMode.leave_one_out("not-a-transform")


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def test_baseline_pipeline_uses_a_passthrough_engineering_step():
    pipeline = build_ablation_pipeline(baseline_mode())
    assert pipeline.named_steps["engineering"] == "passthrough"
    assert list(pipeline.named_steps) == ["zero_policy", "engineering", "features", "clf"]


def test_engineered_pipeline_uses_the_stateless_transformer():
    from heart.features.engineering import EngineeredFeatureTransformer

    pipeline = build_ablation_pipeline(all_features_mode())
    assert isinstance(pipeline.named_steps["engineering"], EngineeredFeatureTransformer)


def test_baseline_pipeline_matches_the_published_baseline(synthetic_frames):
    train, test = synthetic_frames
    ablation = build_ablation_pipeline(baseline_mode())
    published = build_baseline_model()
    ablation.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    published.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    left = ablation.predict_proba(test[list(FEATURE_COLUMNS)])
    right = published.predict_proba(test[list(FEATURE_COLUMNS)])
    np.testing.assert_allclose(left, right, atol=1e-12)


def test_ablation_feature_counts_track_engineered_columns(synthetic_frames):
    train, _ = synthetic_frames
    baseline = build_ablation_pipeline(baseline_mode())
    baseline.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    n_input, n_transformed = ablation_feature_counts(baseline)
    assert n_input == len(FEATURE_COLUMNS)
    assert n_transformed > 0

    engineered = build_ablation_pipeline(all_features_mode())
    engineered.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    eng_input, _ = ablation_feature_counts(engineered)
    assert eng_input == len(FEATURE_COLUMNS) + len(ENGINEERED_COLUMN_NAMES)


def test_ablation_feature_counts_requires_fitted_pipeline():
    with pytest.raises(AblationError):
        ablation_feature_counts(build_ablation_pipeline(baseline_mode()))


def test_build_ablation_pipeline_rejects_non_mode():
    with pytest.raises(AblationConfigError):
        build_ablation_pipeline("baseline")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Harness: core behaviour
# ---------------------------------------------------------------------------


def _run(modes, frames, **kwargs):
    train, test = frames
    return run_ablation(
        train, test, modes=modes, log_to_mlflow=False, **kwargs
    )


def test_run_ablation_scores_baseline_and_variant(synthetic_frames):
    result = _run(
        [baseline_mode(), FeatureMode.from_include("add-hr", ["hr_reserve"])],
        synthetic_frames,
    )
    assert result.n_modes == 2
    assert result.n_succeeded == 2
    assert result.n_failed == 0
    for run in result.runs:
        assert run.succeeded
        assert run.metrics is not None
        assert set(run.metrics) == set(METRIC_KEYS)
    assert result.baseline_run is not None
    assert result.run_for("add-hr") is not None


def test_variant_with_no_distinguishing_features_equals_baseline(synthetic_frames):
    """The core ablation invariant: only the feature representation differs.

    ``no-distinguishing-features`` is a distinct, named variant whose resolved
    engineered selection is empty, so the two pipelines are identical and every
    metric must agree within floating-point tolerance.
    """
    variant = FeatureMode(
        name="no-distinguishing-features",
        include=(),
        description="A named variant that adds no engineered columns.",
    )
    result = _run([baseline_mode(), variant], synthetic_frames)
    baseline = result.baseline_run
    other = result.run_for(variant.name)
    assert baseline is not None and other is not None
    assert other.mode is not baseline.mode
    assert other.mode.engineered_columns == baseline.mode.engineered_columns == ()

    for key in METRIC_KEYS:
        left, right = baseline.metrics[key], other.metrics[key]
        if isinstance(left, dict):
            assert left == right, f"structured metric {key!r} diverged"
        else:
            assert left == pytest.approx(right, abs=1e-12), f"metric {key!r} diverged"

    assert result.deltas()[variant.name] == pytest.approx(0.0, abs=1e-12)


def test_run_ablation_is_deterministic(synthetic_frames):
    modes = [baseline_mode(), FeatureMode.from_include("add-hr", ["hr_reserve"])]
    first = _run(modes, synthetic_frames)
    second = _run(modes, synthetic_frames)
    for left, right in zip(first.runs, second.runs):
        assert left.primary_metric == pytest.approx(right.primary_metric, abs=1e-12)
        assert left.n_transformed_features == right.n_transformed_features


def test_run_ablation_deltas_reference_the_baseline(synthetic_frames):
    result = _run(
        [baseline_mode(), FeatureMode.from_include("add-hr", ["hr_reserve"])],
        synthetic_frames,
    )
    deltas = result.deltas()
    assert deltas["baseline"] == pytest.approx(0.0, abs=1e-12)
    variant = result.run_for("add-hr")
    expected = variant.primary_metric - result.baseline_run.primary_metric
    assert deltas["add-hr"] == pytest.approx(expected, abs=1e-12)


def test_run_ablation_ranking_puts_highest_metric_first(synthetic_frames):
    result = _run(
        [
            baseline_mode(),
            FeatureMode.from_include("add-hr", ["hr_reserve"]),
            FeatureMode.from_include("add-chol", ["cholesterol_age_ratio"]),
        ],
        synthetic_frames,
    )
    ranked = result.ranked()
    values = [run.primary_metric for run in ranked if run.succeeded]
    assert values == sorted(values, reverse=True)


def test_run_ablation_without_mlflow_records_no_run_ids(synthetic_frames):
    result = _run([baseline_mode()], synthetic_frames)
    assert result.runs[0].run_id is None


# ---------------------------------------------------------------------------
# Harness: MLflow logging
# ---------------------------------------------------------------------------


def test_run_ablation_logs_each_mode_as_an_ablation_run(synthetic_frames, tmp_path):
    train, test = synthetic_frames
    tracking_dir = tmp_path / "mlruns"
    modes = [
        baseline_mode(),
        FeatureMode.from_include("add-hr", ["hr_reserve"]),
    ]
    result = run_ablation(
        train,
        test,
        modes=modes,
        tracking_dir=tracking_dir,
        experiment_name="ablation-test",
    )
    assert all(run.run_id for run in result.runs)

    for ablation_run in result.runs:
        record = load_run(ablation_run.run_id)
        assert record.tags[RUN_KIND_TAG] == ABLATION_RUN_KIND
        assert record.tags[FEATURE_MODE_TAG] == ablation_run.mode.name
        assert record.tags[N_FEATURES_TAG] == str(ablation_run.n_input_features)
        assert "baseline_feature_mode" in record.tags
        assert record.metrics[PRIMARY_METRIC] == pytest.approx(
            ablation_run.primary_metric, abs=1e-12
        )
        assert METRICS_ARTIFACT in record.artifact_paths
        assert RUN_CONFIG_ARTIFACT in record.artifact_paths
        payload = load_metrics_artifact(ablation_run.run_id)
        assert set(payload) == set(METRIC_KEYS)

    # Run names embed the mode slug, so modes are distinguishable in the UI.
    names = {load_run(run.run_id).run_name for run in result.runs}
    assert len(names) == len(result.runs)


def test_run_ablation_engineered_run_records_engineered_columns(
    synthetic_frames, tmp_path
):
    train, test = synthetic_frames
    result = run_ablation(
        train,
        test,
        modes=[FeatureMode.from_include("add-hr", ["hr_reserve"])],
        tracking_dir=tmp_path / "mlruns",
        experiment_name="ablation-test",
    )
    run = result.runs[0]
    record = load_run(run.run_id)
    assert record.tags["engineered_transforms"] == "hr_reserve"
    assert record.tags["engineered_features"] == "HR_Reserve"
    assert record.tags["baseline_feature_mode"] == "false"
    assert record.params["feature_mode"] == "add-hr"


# ---------------------------------------------------------------------------
# Harness: ledger / report
# ---------------------------------------------------------------------------


def test_write_ablation_ledger_round_trips(synthetic_frames, tmp_path):
    result = _run(
        [baseline_mode(), FeatureMode.from_include("add-hr", ["hr_reserve"])],
        synthetic_frames,
    )
    destination = tmp_path / "nested" / "ablation_ledger.json"
    written = write_ablation_ledger(result, destination)
    assert written == destination
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["n_modes"] == 2
    assert [run["mode"]["name"] for run in payload["runs"]] == [
        "baseline",
        "add-hr",
    ]
    assert payload["deltas"]["baseline"] == pytest.approx(0.0, abs=1e-12)


def test_run_ablation_ledger_path_is_recorded(synthetic_frames, tmp_path):
    train, test = synthetic_frames
    destination = tmp_path / "ledger.json"
    result = run_ablation(
        train,
        test,
        modes=[baseline_mode()],
        log_to_mlflow=False,
        ledger_path=destination,
    )
    assert result.ledger_path == destination
    assert destination.exists()


def test_render_ablation_report_names_modes_and_reports_deltas(synthetic_frames):
    variant = FeatureMode(
        name="no-distinguishing-features", include=()
    )
    result = _run([baseline_mode(), variant], synthetic_frames)
    report = render_ablation_report(result)
    assert "Feature Ablation Report" in report
    assert "`baseline`" in report
    assert f"`{variant.name}`" in report
    assert "No feature mode beat the baseline on ROC-AUC." in report
    assert "run_kind=ablation" in report


def test_render_ablation_report_rejects_other_types():
    with pytest.raises(AblationError):
        render_ablation_report("not-a-result")  # type: ignore[arg-type]


def test_write_ablation_ledger_rejects_other_types(tmp_path):
    with pytest.raises(AblationLedgerError):
        write_ablation_ledger("not-a-result", tmp_path / "x.json")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Negative surface
# ---------------------------------------------------------------------------


def test_run_ablation_rejects_non_dataframe():
    with pytest.raises(AblationDataError):
        run_ablation(
            ["not", "a", "frame"],  # type: ignore[arg-type]
            ["not", "a", "frame"],  # type: ignore[arg-type]
            log_to_mlflow=False,
        )


def test_run_ablation_rejects_frame_missing_target(synthetic_frames):
    train, test = synthetic_frames
    with pytest.raises(AblationDataError):
        run_ablation(
            train.drop(columns=[TARGET_COLUMN]),
            test,
            log_to_mlflow=False,
        )


def test_run_ablation_rejects_empty_modes(synthetic_frames):
    with pytest.raises(AblationConfigError):
        _run([], synthetic_frames)


def test_run_ablation_rejects_duplicate_mode_names(synthetic_frames):
    with pytest.raises(AblationConfigError):
        _run([baseline_mode(), FeatureMode.baseline(name="baseline")], synthetic_frames)


def test_run_ablation_rejects_bare_string_modes(synthetic_frames):
    with pytest.raises(AblationConfigError):
        _run("baseline", synthetic_frames)  # type: ignore[arg-type]


class _FailingFitModel:
    """A model whose ``fit`` raises, to exercise the fitting failure path."""

    def fit(self, X, y):  # noqa: N803 - sklearn contract
        raise ValueError("synthetic fit failure")

    def predict(self, X):  # noqa: N803
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):  # noqa: N803
        return np.full((len(X), 2), 0.5)


class _MissingProbaModel:
    """A model with no ``predict_proba``, to exercise the evaluation path."""

    def fit(self, X, y):  # noqa: N803
        return self

    def predict(self, X):  # noqa: N803
        return np.zeros(len(X), dtype=int)


def test_run_ablation_fail_soft_records_failed_modes(synthetic_frames):
    train, test = synthetic_frames

    def builder(mode):
        if mode.name == "broken-fit":
            return _FailingFitModel()
        return build_ablation_pipeline(mode)

    result = run_ablation(
        train,
        test,
        modes=[baseline_mode(), FeatureMode(name="broken-fit", include=())],
        log_to_mlflow=False,
        model_builder=builder,
    )
    assert result.n_succeeded == 1
    assert result.n_failed == 1
    failed = result.run_for("broken-fit")
    assert failed is not None and not failed.succeeded
    assert failed.error_category == "fitting"
    assert "synthetic fit failure" in (failed.error or "")


def test_run_ablation_fail_soft_records_evaluation_errors(synthetic_frames):
    train, test = synthetic_frames

    def builder(mode):
        if mode.name == "no-proba":
            return _MissingProbaModel()
        return build_ablation_pipeline(mode)

    result = run_ablation(
        train,
        test,
        modes=[
            baseline_mode(),
            FeatureMode(name="no-proba", include=()),
        ],
        log_to_mlflow=False,
        model_builder=builder,
    )
    failed = result.run_for("no-proba")
    assert failed is not None
    assert failed.error_category == "evaluation"


def test_run_ablation_strict_raises_with_the_result(synthetic_frames):
    train, test = synthetic_frames

    def builder(mode):
        if mode.name == "broken":
            raise RuntimeError("build exploded")
        return build_ablation_pipeline(mode)

    with pytest.raises(AblationRunError) as excinfo:
        run_ablation(
            train,
            test,
            modes=[baseline_mode(), FeatureMode(name="broken", include=())],
            log_to_mlflow=False,
            model_builder=builder,
            strict=True,
        )
    assert excinfo.value.failed_modes == ("broken",)
    assert excinfo.value.result.n_modes == 2
    assert excinfo.value.result.run_for("broken").error_type == "RuntimeError"


def test_ablation_errors_share_a_base():
    for error in (
        AblationConfigError,
        AblationDataError,
        AblationRunError,
        AblationLedgerError,
    ):
        assert issubclass(error, AblationError)


# ---------------------------------------------------------------------------
# Integration: the git-tracked S01 split
# ---------------------------------------------------------------------------


def test_real_split_ablation_end_to_end(real_split_frames, tmp_path):
    train, test = real_split_frames
    result = run_ablation(
        train,
        test,
        modes=[
            baseline_mode(),
            FeatureMode.from_include("add-hr", ["hr_reserve"]),
            all_features_mode(),
        ],
        split_version=SPLIT_VERSION,
        tracking_dir=tmp_path / "mlruns",
        experiment_name="ablation-integration",
    )
    assert result.n_succeeded == 3

    baseline = result.baseline_run
    all_mode = result.run_for("all-engineered")
    assert baseline is not None and all_mode is not None
    assert baseline.n_input_features == len(FEATURE_COLUMNS)
    assert all_mode.n_input_features == len(FEATURE_COLUMNS) + len(
        ENGINEERED_COLUMN_NAMES
    )
    assert all_mode.n_transformed_features > baseline.n_transformed_features
    # The baseline mode reproduces the published baseline on the real split.
    assert 0.5 <= baseline.primary_metric <= 1.0


def test_real_split_report_renders(real_split_frames):
    train, test = real_split_frames
    result = run_ablation(
        train,
        test,
        modes=[
            baseline_mode(),
            FeatureMode.from_include("add-hr", ["hr_reserve"]),
        ],
        log_to_mlflow=False,
    )
    report = render_ablation_report(result)
    assert "Feature Ablation Report" in report
    assert "train 734 / test 184" in report
