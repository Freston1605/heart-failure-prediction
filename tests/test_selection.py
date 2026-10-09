"""Tests for the documented feature selection (S04/T03).

These tests pin the contract that makes the *selection* auditable:

1. **Threshold policy** — a transform is kept exactly when its marginal metric
   delta clears the configured bar, with the boundary treated inclusively.
2. **Reproducibility** — the same ablation evidence and a fixed
   :class:`SelectionConfig` yield a byte-identical
   :meth:`FeatureSelection.fingerprint` and the same ``selected``
   :class:`FeatureMode`.
3. **Parsimony cap** — ``max_features`` keeps the highest-delta transforms and
   drops the rest with an explicit reason.
4. **Fail-visible evidence** — a missing, failed, or metric-less ablation mode
   drops the transform with a reason that names the gap instead of silently
   vanishing.
5. **Durable artifacts** — ``reports/features.md`` lists every keep/drop with
   its number, and the JSON ledger round-trips back into an equivalent
   selection for downstream slices.

Fixtures are synthetic (no gitignored paths); one integration test runs the
real ablation over the git-tracked S01 split.
"""

from __future__ import annotations

import json

import pytest

from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.features.ablation import (
    AblationResult,
    AblationRun,
    FeatureMode,
    build_ablation_pipeline,
    run_ablation,
    write_ablation_ledger,
)
from heart.features.engineering import TRANSFORM_NAMES
from heart.features.selection import (
    ADD_MODE_PREFIX,
    ALL_ENGINEERED_MODE_NAME,
    DECISION_DROPPED,
    DECISION_KEPT,
    DEFAULT_MIN_DELTA,
    DEFAULT_SELECTION_CONFIG,
    SELECTED_MODE_NAME,
    SELECTION_METRIC,
    FeatureSelection,
    FeatureSelectionError,
    SelectionConfig,
    SelectionConfigError,
    SelectionDataError,
    SelectionLedgerError,
    SelectionReportError,
    TransformEvidence,
    describe_selection,
    generate_feature_selection,
    read_ablation_ledger,
    read_selection,
    render_features_report,
    select_features,
    write_features_report,
    write_selection_ledger,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _metrics(roc_auc: float) -> dict[str, object]:
    return {
        "roc_auc": float(roc_auc),
        "accuracy": float(roc_auc),
        "pr_auc": float(roc_auc),
        "f1": float(roc_auc),
    }


def _ablation_run(
    mode: FeatureMode,
    roc_auc: float | None,
    *,
    run_id: str | None = None,
    status: str = "succeeded",
    error: str | None = None,
    error_type: str | None = None,
    metrics_override: dict[str, object] | None = None,
) -> AblationRun:
    metrics = None
    if status == "succeeded":
        metrics = _metrics(roc_auc) if metrics_override is None else metrics_override
    return AblationRun(
        mode=mode,
        status=status,
        duration_seconds=0.01,
        n_engineered_columns=mode.n_engineered_columns,
        n_input_features=11 + mode.n_engineered_columns,
        n_transformed_features=20,
        metrics=metrics,
        run_id=run_id,
        error=error,
        error_type=error_type,
        error_category="fitting" if error else None,
    )


def _result(
    deltas: dict[str, float],
    *,
    baseline_metric: float = 0.90,
    all_delta: float | None = None,
    include_all: bool = True,
    with_baseline: bool = True,
    run_ids: bool = True,
) -> AblationResult:
    runs: list[AblationRun] = []
    if with_baseline:
        runs.append(
            _ablation_run(
                FeatureMode.baseline(),
                baseline_metric,
                run_id="run-baseline" if run_ids else None,
            )
        )
    for name in TRANSFORM_NAMES:
        if name not in deltas:
            continue
        mode = FeatureMode.from_include(f"{ADD_MODE_PREFIX}{name}", [name])
        runs.append(
            _ablation_run(
                mode,
                baseline_metric + deltas[name],
                run_id=f"run-{name}" if run_ids else None,
            )
        )
    if include_all:
        mode = FeatureMode.all_features()
        resolved_all = (
            all_delta if all_delta is not None else max(deltas.values(), default=0.0)
        )
        runs.append(
            _ablation_run(
                mode,
                baseline_metric + resolved_all,
                run_id="run-all" if run_ids else None,
            )
        )
    return AblationResult(
        model_name="logistic-regression",
        split_version="v1",
        experiment_name="selection-test",
        train_rows=734,
        test_rows=184,
        params={},
        runs=tuple(runs),
        generated_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_config_is_the_documented_policy():
    assert DEFAULT_SELECTION_CONFIG.metric == SELECTION_METRIC == "roc_auc"
    assert DEFAULT_SELECTION_CONFIG.min_delta == DEFAULT_MIN_DELTA == 0.0005
    assert DEFAULT_SELECTION_CONFIG.max_features is None
    assert DEFAULT_SELECTION_CONFIG.mode_prefix == "add-"


def test_config_round_trips_through_dict():
    config = SelectionConfig(metric="pr_auc", min_delta=0.01, max_features=3)
    restored = SelectionConfig.from_dict(config.to_dict())
    assert restored == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "  "},
        {"min_delta": float("nan")},
        {"min_delta": True},
        {"max_features": 0},
        {"max_features": -1},
        {"max_features": 2.5},
        {"mode_prefix": ""},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(SelectionConfigError):
        SelectionConfig(**kwargs)


# ---------------------------------------------------------------------------
# Threshold policy
# ---------------------------------------------------------------------------


def test_select_keeps_only_transforms_above_the_bar():
    result = _result(
        {
            "hr_reserve": 0.01,
            "cholesterol_age_ratio": 0.0004,
            "rate_pressure_product": -0.002,
        }
    )
    selection = select_features(result)
    assert selection.kept == ("hr_reserve",)
    assert {
        "cholesterol_age_ratio",
        "rate_pressure_product",
    } <= set(selection.dropped)
    assert selection.selected_mode.name == SELECTED_MODE_NAME
    assert selection.selected_mode.include == ("hr_reserve",)


def test_select_threshold_boundary_is_inclusive():
    # 0.5 and 0.0625 are exact in binary floating point, so this pins the
    # ``delta >= min_delta`` semantics without rounding surprises.
    result = _result(
        {
            "hr_reserve": 0.0625,
            "age_band": 0.0625 - 1e-12,
        },
        baseline_metric=0.5,
    )
    selection = select_features(result, SelectionConfig(min_delta=0.0625))
    assert "hr_reserve" in selection.kept
    assert "age_band" in selection.dropped


def test_select_kept_order_follows_the_registry():
    result = _result(
        {
            "age_band_sex": 0.02,
            "age_band": 0.01,
            "hr_reserve": 0.03,
        }
    )
    selection = select_features(result)
    assert selection.kept == (
        "age_band",
        "hr_reserve",
        "age_band_sex",
    )
    assert selection.kept_columns == (
        "AgeBand",
        "HR_Reserve",
        "AgeBand_Sex",
    )
    assert selection.selected_mode.engineered_columns == selection.kept_columns


def test_select_records_evidence_for_every_registry_transform():
    result = _result({"hr_reserve": 0.01})
    selection = select_features(result)
    assert tuple(item.transform for item in selection.evidence) == TRANSFORM_NAMES
    evidence = selection.evidence_for
    assert evidence["hr_reserve"].decision == DECISION_KEPT
    assert evidence["hr_reserve"].mode_metric == pytest.approx(0.91)
    assert evidence["hr_reserve"].delta == pytest.approx(0.01)
    assert evidence["hr_reserve"].reason.startswith("marginal roc_auc")
    assert evidence["age_band"].decision == DECISION_DROPPED


def test_select_drops_everything_when_nothing_clears_the_bar():
    result = _result({"hr_reserve": 0.0001, "age_band": -0.01})
    selection = select_features(result)
    assert selection.kept == ()
    assert selection.selected_mode.is_baseline is True
    assert selection.kept_columns == ()


# ---------------------------------------------------------------------------
# Reproducibility from a fixed configuration
# ---------------------------------------------------------------------------


def test_selection_is_reproducible_from_a_fixed_configuration():
    """The fixed config is the whole contract: same evidence -> same selection."""
    result = _result(
        {
            "hr_reserve": 0.002,
            "age_band": 0.001,
            "cholesterol_age_ratio": -0.001,
        }
    )
    config = DEFAULT_SELECTION_CONFIG
    first = select_features(result, config, generated_at="fixed")
    second = select_features(result, config, generated_at="fixed")

    assert first.fingerprint() == second.fingerprint()
    assert first.selected_mode == second.selected_mode
    assert first.selected_mode.include == ("age_band", "hr_reserve")

    # And the representation is a first-class FeatureMode, ready downstream.
    pipeline = build_ablation_pipeline(first.selected_mode)
    assert pipeline.named_steps["engineering"] != "passthrough"


def test_selection_fingerprint_ignores_timestamps_and_paths():
    result = _result({"hr_reserve": 0.01})
    left = select_features(result, generated_at="2026-01-01T00:00:00+00:00")
    right = select_features(result, generated_at="2099-12-31T23:59:59+00:00")
    assert left.fingerprint() == right.fingerprint()
    assert left.to_dict()["fingerprint"] == right.to_dict()["fingerprint"]


def test_selection_config_change_changes_the_outcome():
    result = _result({"hr_reserve": 0.001, "age_band": 0.02})
    strict = select_features(result, SelectionConfig(min_delta=0.01))
    lenient = select_features(result, SelectionConfig(min_delta=0.0001))
    assert strict.kept == ("age_band",)
    assert lenient.kept == ("age_band", "hr_reserve")


# ---------------------------------------------------------------------------
# Max-features cap
# ---------------------------------------------------------------------------


def test_max_features_caps_the_kept_set_by_delta():
    result = _result(
        {
            "hr_reserve": 0.003,
            "age_band": 0.002,
            "cholesterol_age_ratio": 0.001,
        }
    )
    selection = select_features(result, SelectionConfig(max_features=2))
    assert selection.kept == ("age_band", "hr_reserve")
    assert "cholesterol_age_ratio" in selection.dropped
    capped = selection.evidence_for["cholesterol_age_ratio"]
    assert "max_features=2" in capped.reason


def test_max_features_tie_break_is_registry_order():
    result = _result(
        {
            "age_band": 0.002,
            "hr_reserve": 0.002,
            "cholesterol_age_ratio": 0.002,
        }
    )
    selection = select_features(result, SelectionConfig(max_features=1))
    # All deltas tie, so the first transform in registry order wins.
    assert selection.kept == ("age_band",)
    assert selection.evidence_for["age_band"].decision == DECISION_KEPT


def test_max_features_no_op_when_below_cap():
    result = _result({"hr_reserve": 0.01, "age_band": 0.02})
    selection = select_features(result, SelectionConfig(max_features=5))
    assert set(selection.kept) == {"hr_reserve", "age_band"}


# ---------------------------------------------------------------------------
# Fail-visible evidence
# ---------------------------------------------------------------------------


def test_missing_ablation_mode_is_dropped_with_reason():
    result = _result({"hr_reserve": 0.01}, include_all=False)
    selection = select_features(result)
    assert "hr_reserve" in selection.kept
    missing = selection.evidence_for["age_band"]
    assert missing.decision == DECISION_DROPPED
    assert missing.delta is None
    assert "no ablation evidence" in missing.reason
    assert selection.all_engineered_metric is None


def test_failed_ablation_mode_is_dropped_with_error_reason():
    result = _result({"hr_reserve": 0.01})
    failed = _ablation_run(
        FeatureMode.from_include("add-age_band", ["age_band"]),
        0.0,
        status="failed",
        error="ValueError: boom",
        error_type="ValueError",
    )
    result = AblationResult(
        **{**result.__dict__, "runs": result.runs + (failed,)}
    )
    selection = select_features(result)
    evidence = selection.evidence_for["age_band"]
    assert evidence.decision == DECISION_DROPPED
    assert "ValueError" in evidence.reason
    assert evidence.delta is None


def test_mode_without_the_metric_is_dropped():
    result = _result({"hr_reserve": 0.01})
    metricless = AblationRun(
        mode=FeatureMode.from_include("add-age_band", ["age_band"]),
        status="succeeded",
        duration_seconds=0.0,
        n_engineered_columns=1,
        metrics={"accuracy": 0.9},
    )
    result = AblationResult(
        **{**result.__dict__, "runs": result.runs + (metricless,)}
    )
    selection = select_features(result)
    evidence = selection.evidence_for["age_band"]
    assert evidence.decision == DECISION_DROPPED
    assert "did not record metric" in evidence.reason


def test_select_requires_a_baseline_run():
    result = _result({"hr_reserve": 0.01}, with_baseline=False)
    with pytest.raises(SelectionDataError):
        select_features(result)


def test_select_requires_baseline_metric():
    result = _result({"hr_reserve": 0.01})
    severity = AblationResult(
        **{
            **result.__dict__,
            "runs": (
                AblationRun(
                    mode=FeatureMode.baseline(),
                    status="succeeded",
                    duration_seconds=0.0,
                    n_engineered_columns=0,
                    metrics={"accuracy": 0.9},
                ),
            ),
        }
    )
    with pytest.raises(SelectionDataError):
        select_features(severity)


def test_select_rejects_non_result():
    with pytest.raises(SelectionDataError):
        select_features("not-a-result")  # type: ignore[arg-type]


def test_select_rejects_non_config():
    result = _result({"hr_reserve": 0.01})
    with pytest.raises(SelectionConfigError):
        select_features(result, config="fast")  # type: ignore[arg-type]


def test_evidence_rejects_unknown_decision():
    with pytest.raises(FeatureSelectionError):
        TransformEvidence(
            transform="hr_reserve",
            columns=("HR_Reserve",),
            mode_name="add-hr_reserve",
            decision="maybe",
            delta=0.1,
            mode_metric=0.91,
            threshold=0.0005,
            reason="",
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_lists_kept_and_dropped_with_evidence():
    result = _result(
        {
            "hr_reserve": 0.003,
            "cholesterol_age_ratio": -0.001,
            "age_band": 0.02,
        },
        all_delta=0.0001,
    )
    selection = select_features(result)
    report = render_features_report(selection)

    assert "# Feature Selection Report" in report
    assert "## Decision policy" in report
    assert "marginal Δ ≥ `0.0005`" in report
    assert "## Selected representation" in report
    assert "`selected`" in report
    assert "## Kept features" in report
    assert "## Dropped features" in report
    assert "`hr_reserve`" in report
    assert "`age_band`" in report
    assert "`cholesterol_age_ratio`" in report
    assert "+0.0030" in report
    assert "-0.0010" in report
    assert "did **not** beat" in report
    assert "## Reproduce" in report
    assert "python -m heart.features.selection" in report


def test_report_handles_empty_kept_set():
    result = _result({"hr_reserve": -0.01})
    report = render_features_report(select_features(result))
    assert "baseline feature set" in report
    assert "## Dropped features" in report


def test_report_rejects_other_types():
    with pytest.raises(SelectionReportError):
        render_features_report("not-a-selection")  # type: ignore[arg-type]


def test_write_features_report_creates_parent_dirs(tmp_path):
    selection = select_features(_result({"hr_reserve": 0.01}))
    destination = tmp_path / "nested" / "features.md"
    written = write_features_report(selection, destination)
    assert written == destination
    assert "Feature Selection Report" in destination.read_text(encoding="utf-8")


def test_describe_selection_summarises_the_decision():
    selection = select_features(_result({"hr_reserve": 0.01, "age_band": -0.01}))
    summary = describe_selection(selection)
    assert "hr_reserve" in summary
    assert "age_band" in summary
    assert "kept 1/10" in summary
    with pytest.raises(FeatureSelectionError):
        describe_selection("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ledger round-trip
# ---------------------------------------------------------------------------


def test_selection_ledger_round_trips(tmp_path):
    selection = select_features(
        _result({"hr_reserve": 0.003, "age_band": 0.002}), generated_at="fixed"
    )
    destination = tmp_path / "nested" / "feature_selection.json"
    written = write_selection_ledger(selection, destination)
    assert written == destination

    restored = read_selection(destination)
    assert isinstance(restored, FeatureSelection)
    assert restored.fingerprint() == selection.fingerprint()
    assert restored.selected_mode.include == selection.selected_mode.include
    assert restored.kept == selection.kept
    assert restored.config == selection.config
    assert restored.generated_at == "fixed"


def test_selection_ledger_rejects_other_types(tmp_path):
    with pytest.raises(SelectionLedgerError):
        write_selection_ledger("nope", tmp_path / "x.json")  # type: ignore[arg-type]


def test_read_selection_rejects_missing_and_malformed(tmp_path):
    with pytest.raises(SelectionLedgerError):
        read_selection(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SelectionLedgerError):
        read_selection(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"kept": []}), encoding="utf-8")
    with pytest.raises(SelectionLedgerError):
        read_selection(empty)


def test_read_ablation_ledger_round_trips(tmp_path):
    result = _result(
        {
            "hr_reserve": 0.003,
            "age_band": 0.002,
            "cholesterol_age_ratio": -0.001,
        }
    )
    destination = tmp_path / "ablation_ledger.json"
    write_ablation_ledger(result, destination)

    loaded = read_ablation_ledger(destination)
    assert loaded.n_modes == result.n_modes
    assert loaded.model_name == result.model_name

    expected = select_features(result)
    actual = select_features(loaded)
    assert actual.fingerprint() == expected.fingerprint()


def test_read_ablation_ledger_rejects_malformed(tmp_path):
    with pytest.raises(SelectionDataError):
        read_ablation_ledger(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "an ablation ledger"}', encoding="utf-8")
    with pytest.raises(SelectionDataError):
        read_ablation_ledger(bad)


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------


def test_generate_feature_selection_from_result_without_writing():
    result = _result({"hr_reserve": 0.01})
    selection = generate_feature_selection(result, write=False)
    assert isinstance(selection, FeatureSelection)
    assert selection.report_path is None
    assert selection.ledger_path is None


def test_generate_feature_selection_writes_artifacts(tmp_path):
    result = _result({"hr_reserve": 0.01})
    selection = generate_feature_selection(
        result,
        report_path=tmp_path / "features.md",
        ledger_path=tmp_path / "feature_selection.json",
    )
    assert selection.report_path == tmp_path / "features.md"
    assert selection.ledger_path == tmp_path / "feature_selection.json"
    assert read_selection(selection.ledger_path).kept == selection.kept


def test_cli_selects_from_an_ablation_ledger(tmp_path):
    from heart.features.selection import main

    result = _result({"hr_reserve": 0.01, "age_band": -0.01})
    ledger = tmp_path / "ablation_ledger.json"
    write_ablation_ledger(result, ledger)
    report = tmp_path / "features.md"
    out_ledger = tmp_path / "feature_selection.json"

    code = main(
        [
            "--ablation-ledger",
            str(ledger),
            "--report-path",
            str(report),
            "--ledger-path",
            str(out_ledger),
        ]
    )
    assert code == 0
    assert report.exists()
    assert read_selection(out_ledger).kept == ("hr_reserve",)


def test_cli_reports_configuration_errors(tmp_path):
    from heart.features.selection import main

    result = _result({"hr_reserve": 0.01})
    ledger = tmp_path / "ablation_ledger.json"
    write_ablation_ledger(result, ledger)
    assert main(["--max-features", "0"]) == 2


# ---------------------------------------------------------------------------
# Integration: the git-tracked S01 split
# ---------------------------------------------------------------------------


def test_selection_over_real_split_ablation(tmp_path):
    train, test = load_split_frames(SPLIT_VERSION)
    result = run_ablation(
        train,
        test,
        split_version=SPLIT_VERSION,
        log_to_mlflow=False,
    )
    selection = select_features(result)

    assert selection.baseline_metric == pytest.approx(0.9286, abs=5e-4)
    assert selection.kept == (
        "cholesterol_age_ratio",
        "oldpeak_slope_interaction",
        "exercise_ecg_group",
        "age_band_sex",
    )
    assert "cholesterol_restingbp_ratio" in selection.dropped
    assert selection.all_engineered_metric is not None

    report = render_features_report(selection)
    assert "`oldpeak_slope_interaction`" in report
    assert "`cholesterol_restingbp_ratio`" in report
    assert "did **not** beat" in report

    # The selected representation is reproducible from the ledger and usable.
    selection = generate_feature_selection(
        result,
        report_path=tmp_path / "features.md",
        ledger_path=tmp_path / "feature_selection.json",
    )
    restored = read_selection(tmp_path / "feature_selection.json")
    assert restored.fingerprint() == selection.fingerprint()
    pipeline = build_ablation_pipeline(restored.selected_mode)
    assert pipeline.named_steps["engineering"] != "passthrough"
