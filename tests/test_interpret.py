"""Tests for the interpretability artifacts (S04/T04).

These tests pin the contract that makes the explanation report auditable:

1. **Selected feature space** — the leading-model pipeline is built on the
   selected representation, and the exported artifact dimensions equal the
   transformed feature count the classifier actually saw.
2. **SHAP summary** — per-feature mean ``|SHAP|`` and signed mean SHAP are
   finite, non-negative in importance, ranked correctly, and round-trip through
   the durable CSV.
3. **Global attribution** — tree importances and linear coefficients are
   extracted with the right kind/sign and one row per selected feature; an
   estimator with neither surface fails loudly.
4. **Reproducibility** — the leading-model ledger round-trips, and the report
   renders the setup, SHAP, importance, agreement, and artifact sections.

Fixtures are synthetic inline frames (no gitignored paths); one integration
test uses the git-tracked S01 split and the committed selection ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.features.ablation import FeatureMode
from heart.features.selection import read_selection
from heart.interpret import (
    DEFAULT_REPORT_PATH,
    ImportanceArtifactError,
    ImportanceDataError,
    ImportanceReport,
    InterpretConfigError,
    InterpretabilityContext,
    InterpretabilityReportError,
    LeadingModel,
    LeadingModelError,
    LeadingModelLedgerError,
    SelectedPipelineError,
    ShapArtifactError,
    ShapConfigError,
    ShapDataError,
    ShapDependencyError,
    ShapSummary,
    UnsupportedImportanceError,
    UnsupportedShapModelError,
    agreement_spearman,
    build_selected_pipeline,
    coerce_tuned_params,
    compute_shap_values,
    extract_importance,
    generate_interpretability_report,
    matches_engineered,
    model_input_feature_names,
    read_importance,
    read_leading_model_ledger,
    read_shap_summary,
    read_shap_values,
    render_interpretability_report,
    summarize_shap,
    transformed_feature_names,
    with_engineered_flags,
    write_importance,
    write_leading_model_ledger,
    write_shap_summary,
    write_shap_values,
)
from heart.models.registry import resolve_spec
from heart.models.spaces import default_params

#: The four engineered transforms the S04/T03 selection kept.
SELECTED_TRANSFORMS: tuple[str, ...] = (
    "cholesterol_age_ratio",
    "oldpeak_slope_interaction",
    "exercise_ecg_group",
    "age_band_sex",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _synthetic_frame(n: int = 180, seed: int = 7) -> pd.DataFrame:
    """A schema-valid synthetic frame with a learnable, non-trivial signal."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "Age": rng.integers(30, 80, n),
            "Sex": rng.choice(["F", "M"], n),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], n),
            "RestingBP": rng.integers(90, 200, n),
            "Cholesterol": rng.integers(120, 400, n),
            "FastingBS": rng.integers(0, 2, n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], n),
            "MaxHR": rng.integers(70, 200, n),
            "ExerciseAngina": rng.choice(["N", "Y"], n),
            "Oldpeak": rng.uniform(0.0, 6.0, n),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], n),
        }
    )
    label = ((frame["Oldpeak"] > 2.0) & (frame["ExerciseAngina"] == "Y")).astype(int)
    frame[TARGET_COLUMN] = label.values
    return frame


def _selected_mode() -> FeatureMode:
    return FeatureMode.from_include("selected", SELECTED_TRANSFORMS)


def _leading_model(model_type: str, **overrides: object) -> LeadingModel:
    spec = resolve_spec(model_type)
    tuned = dict(default_params(spec.space))
    tuned.update(overrides)
    return LeadingModel(
        model_type=spec.model_type,
        model_name=spec.model_name,
        family=spec.family,
        run_id="test-run",
        run_name="test-run-name",
        split_version=SPLIT_VERSION,
        primary_metric=0.9,
        raw_params={key: str(value) for key, value in tuned.items()},
        tuned_params=tuned,
    )


def _fit_pipeline(mode: FeatureMode, model: LeadingModel, frame: pd.DataFrame):
    pipeline = build_selected_pipeline(mode, model)
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_COLUMN])
    return pipeline


def _fit_dummy_pipeline(bundle: "_Bundle"):
    """The real selected feature chain with an attribution-less final step."""
    pipeline = build_selected_pipeline(bundle.mode, bundle.model)
    pipeline.set_params(clf=_FitOnly())
    pipeline.fit(bundle.X, bundle.frame[TARGET_COLUMN])
    return pipeline


class _FitOnly:
    """A classifier exposing neither ``coef_`` nor ``feature_importances_``."""

    def fit(self, X, y):  # noqa: N803 - sklearn signature
        return self


class _Bundle:
    def __init__(self) -> None:
        self.frame = _synthetic_frame()
        self.X = self.frame[list(FEATURE_COLUMNS)]
        self.mode = _selected_mode()
        self.model = _leading_model("random-forest", n_estimators=8, max_depth=3)
        self.pipeline = _fit_pipeline(self.mode, self.model, self.frame)
        self.shap_values = compute_shap_values(self.pipeline, self.X)
        self.shap_summary = summarize_shap(
            self.shap_values,
            model_name=self.model.model_name,
            model_type=self.model.model_type,
            split_version=SPLIT_VERSION,
            engineered_columns=self.mode.engineered_columns,
        )
        self.importance = with_engineered_flags(
            extract_importance(self.pipeline), self.mode.engineered_columns
        )

    def context(self) -> InterpretabilityContext:
        return InterpretabilityContext(
            selected_mode=self.mode,
            engineered_columns=tuple(self.mode.engineered_columns),
            n_dropped=6,
            leading_model=self.model,
            split_version=SPLIT_VERSION,
            pipeline=self.pipeline,
            test_frame=self.X,
            train_rows=len(self.X),
            test_rows=len(self.X),
            feature_names=transformed_feature_names(self.pipeline),
            input_feature_names=model_input_feature_names(self.pipeline),
        )


@pytest.fixture(scope="module")
def bundle() -> _Bundle:
    return _Bundle()


# ---------------------------------------------------------------------------
# Feature spaces
# ---------------------------------------------------------------------------


def test_raw_inputs_are_schema_columns(bundle: _Bundle) -> None:
    assert model_input_feature_names(bundle.pipeline) == FEATURE_COLUMNS


def test_transformed_feature_count_is_consistent(bundle: _Bundle) -> None:
    names = transformed_feature_names(bundle.pipeline)
    expected = bundle.pipeline.named_steps["features"].get_feature_names_out()
    assert names == tuple(str(name) for name in expected)
    assert len(names) > len(FEATURE_COLUMNS)


def test_engineered_columns_present_in_transformed_names(bundle: _Bundle) -> None:
    names = bundle.shap_summary.feature_names
    for column in bundle.mode.engineered_columns:
        assert any(matches_engineered(name, (column,)) for name in names), column


def test_engineered_mask_flags_selected_columns(bundle: _Bundle) -> None:
    mask = bundle.context().engineered_mask()
    assert len(mask) == bundle.context().n_features
    assert sum(mask) >= len(bundle.mode.engineered_columns)
    # 2 numeric engineered columns + one-hot variants of the 2 categorical ones.
    assert sum(mask) == sum(bundle.shap_summary.engineered)


def test_matches_engineered_exact_and_prefixed() -> None:
    assert matches_engineered("numeric__Cholesterol_Age_Ratio", ("Cholesterol_Age_Ratio",))
    assert matches_engineered("categorical__AgeBand_Sex_50-59_M", ("AgeBand_Sex",))
    assert not matches_engineered("numeric__Cholesterol", ("Cholesterol_Age_Ratio",))
    assert not matches_engineered("categorical__Sex_M", ("AgeBand_Sex",))


# ---------------------------------------------------------------------------
# SHAP values and summary
# ---------------------------------------------------------------------------


def test_shap_dimensions_match_selected_feature_count(bundle: _Bundle) -> None:
    n_features = len(transformed_feature_names(bundle.pipeline))
    assert bundle.shap_values.values.shape == (len(bundle.X), n_features)
    assert bundle.shap_summary.n_features == n_features
    assert len(bundle.shap_summary.to_rows()) == n_features


def test_shap_summary_values_are_finite_and_nonnegative(bundle: _Bundle) -> None:
    assert all(np.isfinite(v) and v >= 0 for v in bundle.shap_summary.mean_abs_shap)
    assert all(np.isfinite(v) for v in bundle.shap_summary.mean_shap)


def test_shap_ranks_are_sorted_and_start_at_one(bundle: _Bundle) -> None:
    rows = bundle.shap_summary.to_rows()
    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    values = [float(row["mean_abs_shap"]) for row in rows]
    assert values == sorted(values, reverse=True)


def test_shap_top_returns_true_highest(bundle: _Bundle) -> None:
    rows = bundle.shap_summary.to_rows()
    top = bundle.shap_summary.top(3)
    assert [row["feature"] for row in top] == [row["feature"] for row in rows[:3]]
    assert [row["rank"] for row in top] == [1, 2, 3]


def test_shap_max_samples_cap(bundle: _Bundle) -> None:
    capped = compute_shap_values(bundle.pipeline, bundle.X, max_samples=25)
    assert capped.n_samples == 25
    assert capped.n_features == bundle.shap_values.n_features


def test_shap_rejects_negative_max_samples(bundle: _Bundle) -> None:
    with pytest.raises(ShapConfigError):
        compute_shap_values(bundle.pipeline, bundle.X, max_samples=0)


def test_shap_rejects_empty_frame(bundle: _Bundle) -> None:
    with pytest.raises(ShapDataError):
        compute_shap_values(bundle.pipeline, bundle.X.iloc[0:0])


def test_shap_rejects_non_dataframe(bundle: _Bundle) -> None:
    with pytest.raises(ShapDataError):
        compute_shap_values(bundle.pipeline, np.zeros((5, len(FEATURE_COLUMNS))))


def test_shap_rejects_missing_columns(bundle: _Bundle) -> None:
    with pytest.raises(ShapDataError):
        compute_shap_values(bundle.pipeline, bundle.X.drop(columns=["Oldpeak"]))


def test_shap_rejects_unfitted_pipeline(bundle: _Bundle) -> None:
    unfitted = build_selected_pipeline(bundle.mode, bundle.model)
    with pytest.raises(ShapDataError):
        compute_shap_values(unfitted, bundle.X)


def test_unsupported_shap_estimator_raises(bundle: _Bundle) -> None:
    pipeline = _fit_dummy_pipeline(bundle)
    with pytest.raises(UnsupportedShapModelError):
        compute_shap_values(pipeline, bundle.X)


def test_shap_dependency_error_is_raised(monkeypatch) -> None:
    import heart.interpret.shap_export as shap_export

    def _boom():
        raise ShapDependencyError("shap is not installed")

    monkeypatch.setattr(shap_export, "_import_shap", _boom)
    with pytest.raises(ShapDependencyError):
        shap_export.build_explainer(object(), np.zeros((3, 2)))


def test_summarize_rejects_wrong_type() -> None:
    with pytest.raises(ShapDataError):
        summarize_shap("not-shap-values")  # type: ignore[arg-type]


def test_shap_summary_top_rejects_negative_k(bundle: _Bundle) -> None:
    with pytest.raises(ShapConfigError):
        bundle.shap_summary.top(-1)


def test_shap_summary_csv_round_trip(bundle: _Bundle, tmp_path) -> None:
    path = write_shap_summary(bundle.shap_summary, tmp_path / "shap_summary.csv")
    restored = read_shap_summary(path)
    # The artifact is rank-ordered, so compare per-feature values, not order.
    original = {
        row["raw_feature"]: row for row in bundle.shap_summary.to_rows()
    }
    roundtrip = {row["raw_feature"]: row for row in restored.to_rows()}
    assert restored.n_features == bundle.shap_summary.n_features
    assert roundtrip.keys() == original.keys()
    assert sum(restored.engineered) == sum(bundle.shap_summary.engineered)
    for key, row in original.items():
        assert roundtrip[key]["mean_abs_shap"] == pytest.approx(
            row["mean_abs_shap"], rel=1e-9
        )
        assert roundtrip[key]["rank"] == row["rank"]


def test_shap_values_csv_round_trip(bundle: _Bundle, tmp_path) -> None:
    path = write_shap_values(bundle.shap_values, tmp_path / "shap_values.csv")
    frame = read_shap_values(path)
    assert frame.shape == bundle.shap_values.values.shape
    assert list(frame.columns) == list(bundle.shap_values.display_names)
    np.testing.assert_allclose(frame.to_numpy(), bundle.shap_values.values, rtol=1e-9)


def test_read_shap_summary_malformed_raises(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("nonsense,header\n1,2\n", encoding="utf-8")
    with pytest.raises(ShapArtifactError):
        read_shap_summary(bad)


def test_read_shap_values_empty_raises(tmp_path) -> None:
    bad = tmp_path / "empty.csv"
    bad.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ShapArtifactError):
        read_shap_values(bad)


# ---------------------------------------------------------------------------
# Coefficients / importances
# ---------------------------------------------------------------------------


def test_importance_dimensions_match_selected_feature_count(bundle: _Bundle) -> None:
    n_features = len(transformed_feature_names(bundle.pipeline))
    assert bundle.importance.n_features == n_features
    assert len(bundle.importance.to_rows()) == n_features


def test_tree_importances_sum_to_one(bundle: _Bundle) -> None:
    assert bundle.importance.kind == "importance"
    assert not bundle.importance.signed
    assert sum(bundle.importance.values) == pytest.approx(1.0, abs=1e-6)


def test_linear_coefficients_are_signed(bundle: _Bundle) -> None:
    model = _leading_model("logistic-regression-l2")
    pipeline = _fit_pipeline(bundle.mode, model, bundle.frame)
    report = extract_importance(pipeline)
    assert report.kind == "coefficient"
    assert report.signed
    assert report.n_features == len(transformed_feature_names(pipeline))
    assert any(value < 0 for value in report.values)


def test_importance_top_is_ranked(bundle: _Bundle) -> None:
    rows = bundle.importance.to_rows()
    abs_values = [float(row["abs_value"]) for row in rows]
    assert abs_values == sorted(abs_values, reverse=True)
    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))


def test_importance_csv_round_trip(bundle: _Bundle, tmp_path) -> None:
    path = write_importance(bundle.importance, tmp_path / "feature_importance.csv")
    restored = read_importance(path)
    # The artifact is rank-ordered, so compare per-feature values, not order.
    original = {row["raw_feature"]: row for row in bundle.importance.to_rows()}
    roundtrip = {row["raw_feature"]: row for row in restored.to_rows()}
    assert restored.n_features == bundle.importance.n_features
    assert restored.kind == bundle.importance.kind
    assert roundtrip.keys() == original.keys()
    assert sum(restored.engineered) == sum(bundle.importance.engineered)
    for key, row in original.items():
        assert roundtrip[key]["value"] == pytest.approx(row["value"], rel=1e-9)
        assert roundtrip[key]["rank"] == row["rank"]


def test_importance_rejects_unsupported_estimator(bundle: _Bundle) -> None:
    pipeline = _fit_dummy_pipeline(bundle)
    with pytest.raises(UnsupportedImportanceError):
        extract_importance(pipeline)


def test_importance_report_rejects_negative_importance() -> None:
    with pytest.raises(ImportanceDataError):
        ImportanceReport(
            feature_names=("a", "b"),
            display_names=("a", "b"),
            engineered=(False, False),
            values=(0.5, -0.5),
            kind="importance",
            model_name="dummy",
            estimator_class="dummy",
            n_features=2,
        )


def test_importance_report_rejects_mismatched_lengths() -> None:
    with pytest.raises(ImportanceDataError):
        ImportanceReport(
            feature_names=("a", "b"),
            display_names=("a",),
            engineered=(False, False),
            values=(0.5, 0.5),
            kind="importance",
            model_name="dummy",
            estimator_class="dummy",
            n_features=2,
        )


def test_read_importance_malformed_raises(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ImportanceArtifactError):
        read_importance(bad)


def test_with_engineered_flags_sets_flags(bundle: _Bundle) -> None:
    raw = extract_importance(bundle.pipeline)
    assert not any(raw.engineered)
    flagged = with_engineered_flags(raw, bundle.mode.engineered_columns)
    assert sum(flagged.engineered) == sum(bundle.importance.engineered)


# ---------------------------------------------------------------------------
# Leading model + ledger
# ---------------------------------------------------------------------------


def test_coerce_tuned_params_types() -> None:
    spec = resolve_spec("random-forest")
    tuned = coerce_tuned_params(
        spec,
        {
            "n_estimators": "218",
            "max_depth": "31",
            "min_samples_split": "15",
            "min_samples_leaf": "6",
            "max_features": "sqrt",
            "class_weight": "null",
        },
    )
    assert tuned["n_estimators"] == 218
    assert isinstance(tuned["n_estimators"], int)
    assert tuned["max_features"] == "sqrt"
    assert tuned["class_weight"] is None


def test_coerce_tuned_params_missing_raises() -> None:
    spec = resolve_spec("random-forest")
    with pytest.raises(LeadingModelError):
        coerce_tuned_params(spec, {"n_estimators": "10"})


def test_leading_model_ledger_round_trip(bundle: _Bundle, tmp_path) -> None:
    path = write_leading_model_ledger(bundle.model, tmp_path / "leading_model.json")
    restored = read_leading_model_ledger(path)
    assert restored == bundle.model


def test_read_leading_model_ledger_malformed_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(LeadingModelLedgerError):
        read_leading_model_ledger(bad)


def test_leading_model_from_row_unknown_type_raises(bundle: _Bundle) -> None:
    from heart.interpret import leading_model_from_row

    class _Row:
        model_type = "not-a-model"
        model_name = "Nope"
        family = "linear"
        run_id = "r"
        run_name = "n"
        split_version = "v1"
        primary_metric = 0.5
        params: dict[str, str] = {}

    # Registry errors are wrapped into the interpretability error family so the
    # CLI can fail visibly instead of tracebacking.
    with pytest.raises(LeadingModelError):
        leading_model_from_row(_Row())


def test_build_selected_pipeline_rejects_bad_mode(bundle: _Bundle) -> None:
    with pytest.raises(SelectedPipelineError):
        build_selected_pipeline("not-a-mode", bundle.model)  # type: ignore[arg-type]


def test_build_selected_pipeline_unknown_model_type(bundle: _Bundle) -> None:
    broken = LeadingModel(
        model_type="not-a-model",
        model_name="Nope",
        family="linear",
        run_id="r",
        run_name="n",
        split_version="v1",
        primary_metric=0.5,
        raw_params={},
        tuned_params={},
    )
    with pytest.raises(SelectedPipelineError):
        build_selected_pipeline(bundle.mode, broken)


# ---------------------------------------------------------------------------
# Agreement + report
# ---------------------------------------------------------------------------


def test_agreement_is_positive_for_consistent_views(bundle: _Bundle) -> None:
    rho = agreement_spearman(bundle.shap_summary, bundle.importance)
    assert rho is not None
    assert rho > 0.2


def test_agreement_returns_none_when_too_few_shared() -> None:
    shap = ShapSummary(
        feature_names=("a", "b"),
        display_names=("a", "b"),
        engineered=(False, False),
        mean_abs_shap=(0.1, 0.2),
        mean_shap=(0.1, -0.1),
        n_samples=5,
        n_features=2,
        model_name="m",
        model_type="m",
        split_version="v1",
    )
    importance = ImportanceReport(
        feature_names=("a", "b"),
        display_names=("a", "b"),
        engineered=(False, False),
        values=(0.1, 0.2),
        kind="importance",
        model_name="m",
        estimator_class="m",
        n_features=2,
    )
    assert agreement_spearman(shap, importance) is None


def test_agreement_rejects_wrong_types(bundle: _Bundle) -> None:
    with pytest.raises(InterpretabilityReportError):
        agreement_spearman("nope", bundle.importance)  # type: ignore[arg-type]
    with pytest.raises(InterpretabilityReportError):
        agreement_spearman(bundle.shap_summary, "nope")  # type: ignore[arg-type]


def test_report_dimensions_and_sections(bundle: _Bundle, tmp_path) -> None:
    report = generate_interpretability_report(
        context=bundle.context(),
        report_path=tmp_path / "interpretability.md",
        shap_summary_path=tmp_path / "shap_summary.csv",
        shap_values_path=tmp_path / "shap_values.csv",
        importance_path=tmp_path / "feature_importance.csv",
        leading_model_path=tmp_path / "leading_model.json",
        top_n=8,
    )
    assert report.n_features == len(transformed_feature_names(bundle.pipeline))
    assert report.shap_summary.n_features == report.n_features
    assert report.importance.n_features == report.n_features
    markdown = render_interpretability_report(report)
    for section in (
        "# Model Interpretability Report",
        "## Setup",
        "## SHAP summary",
        "## Engineered-feature contributions",
        "## Feature importance",
        "## Agreement",
        "## Artifacts",
        "## Reproduce",
    ):
        assert section in markdown


def test_report_writes_all_artifacts(bundle: _Bundle, tmp_path) -> None:
    report = generate_interpretability_report(
        context=bundle.context(),
        report_path=tmp_path / "interpretability.md",
        shap_summary_path=tmp_path / "shap_summary.csv",
        shap_values_path=tmp_path / "shap_values.csv",
        importance_path=tmp_path / "feature_importance.csv",
        leading_model_path=tmp_path / "leading_model.json",
    )
    for path in (
        report.report_path,
        report.shap_summary_path,
        report.shap_values_path,
        report.importance_path,
        report.leading_model_path,
    ):
        assert path is not None and path.exists()
    assert read_shap_summary(report.shap_summary_path).n_features == report.n_features
    assert read_importance(report.importance_path).n_features == report.n_features


def test_report_dict_is_serialisable(bundle: _Bundle) -> None:
    import json

    report = generate_interpretability_report(
        context=bundle.context(),
        report_path=None,
        shap_summary_path=None,
        shap_values_path=None,
        importance_path=None,
        leading_model_path=None,
    )
    payload = json.dumps(report.to_dict())
    assert "features" in payload


def test_report_rejects_non_context(bundle: _Bundle) -> None:
    with pytest.raises(InterpretabilityReportError):
        generate_interpretability_report(context="nope")  # type: ignore[arg-type]


def test_build_context_rejects_bad_max_train_rows(bundle: _Bundle) -> None:
    from heart.interpret import build_interpretability_context

    with pytest.raises(InterpretConfigError):
        build_interpretability_context(
            selection=None,
            leading_model=bundle.model,
            max_train_rows=1,
        )


def test_default_report_path_is_in_reports() -> None:
    assert DEFAULT_REPORT_PATH.name == "interpretability.md"
    assert DEFAULT_REPORT_PATH.parent.name == "reports"


# ---------------------------------------------------------------------------
# CLI failure paths (no MLflow: bad ledger fails before model resolution)
# ---------------------------------------------------------------------------


def test_cli_main_fail_visible_on_bad_ledger(tmp_path) -> None:
    from heart.interpret.coefficients import main as coefficients_main
    from heart.interpret.report import main as report_main
    from heart.interpret.shap_export import main as shap_main

    missing = tmp_path / "does-not-exist.json"
    for entrypoint in (shap_main, coefficients_main, report_main):
        assert entrypoint(["--selection-ledger", str(missing)]) == 1


def test_cli_parsers_have_defaults() -> None:
    from heart.interpret.coefficients import build_parser as coefficients_parser
    from heart.interpret.report import build_parser as report_parser
    from heart.interpret.shap_export import build_parser as shap_parser

    assert shap_parser().parse_args([]).selection_ledger.endswith(
        "feature_selection.json"
    )
    assert coefficients_parser().parse_args([]).importance_path.endswith(
        "feature_importance.csv"
    )
    assert report_parser().parse_args([]).report_path.endswith("interpretability.md")


# ---------------------------------------------------------------------------
# Integration: committed selection ledger + git-tracked split
# ---------------------------------------------------------------------------


def test_selected_ledger_pipeline_export_dimensions() -> None:
    """The real selected ledger drives an export whose dims match the model."""
    selection = read_selection("reports/feature_selection.json")
    mode = selection.selected_mode
    model = _leading_model("random-forest", n_estimators=8, max_depth=3)
    train, test = load_split_frames(SPLIT_VERSION)
    pipeline = build_selected_pipeline(mode, model)
    pipeline.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])

    names = transformed_feature_names(pipeline)
    shap_values = compute_shap_values(pipeline, test[list(FEATURE_COLUMNS)])
    summary = summarize_shap(
        shap_values, engineered_columns=mode.engineered_columns
    )
    importance = with_engineered_flags(
        extract_importance(pipeline), mode.engineered_columns
    )

    assert summary.n_features == len(names)
    assert importance.n_features == len(names)
    assert shap_values.values.shape == (len(test), len(names))
    # Every selected engineered raw column is represented in the design matrix.
    for column in mode.engineered_columns:
        assert any(matches_engineered(name, (column,)) for name in names), column
    assert sum(summary.engineered) >= selection.n_kept
