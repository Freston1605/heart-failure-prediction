"""Tests for the zero-as-missing quality audit, policy, and class balance.

Positive tests exercise the real pinned dataset through the loader. Negative
tests drive injected fixtures: all-zero columns, missing columns, unknown
policy actions, and transform-before-fit. The report artifact is checked for
the exact figures the slice verification requires, and for freshness against a
live regeneration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from heart.config import REPORTS_DIR
from heart.data.load import load_dataset
from heart.data.quality import (
    MEDIAN_IMPUTE,
    QUALITY_REPORT_NAME,
    ZERO_AS_MISSING_COLUMNS,
    MissingQualityColumnError,
    NoValidValuesError,
    QualityError,
    UnknownPolicyError,
    ZeroMedianImputer,
    ZeroPolicy,
    apply_zero_policy,
    audit_quality,
    build_quality_audit,
    clean_with_policy,
    compute_class_balance,
    count_affected_rows,
    fit_zero_policy,
    invalid_zero_counts,
    main,
    render_quality_report,
    write_quality_report,
)
from heart.data.schema import TARGET_COLUMN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_frame() -> pd.DataFrame:
    """Small frame: RestingBP zero (1), Cholesterol zeros (2), target present."""
    return pd.DataFrame(
        {
            "RestingBP": [120, 140, 0, 160, 130],
            "Cholesterol": [200, 0, 220, 0, 240],
            "Oldpeak": [0.0, 1.5, 0.0, 2.0, 0.5],
            "HeartDisease": [0, 1, 1, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# Impossible-zero measurement on the real dataset
# ---------------------------------------------------------------------------


def test_zero_counts_on_pinned_dataset() -> None:
    frame = load_dataset().frame
    counts = invalid_zero_counts(frame)

    assert counts["RestingBP"].count == 1
    assert counts["Cholesterol"].count == 172
    assert counts["RestingBP"].share == pytest.approx(1 / 918, abs=1e-6)
    assert counts["Cholesterol"].share == pytest.approx(172 / 918, abs=1e-6)


def test_zero_counts_break_down_by_target() -> None:
    frame = load_dataset().frame
    counts = invalid_zero_counts(frame)

    # The cholesterol zeros are strongly associated with the positive class,
    # which is exactly why dropping them would bias prevalence.
    assert counts["Cholesterol"].by_target == {1: 152, 0: 20}
    assert counts["RestingBP"].by_target == {1: 1}


def test_invalid_zero_counts_detects_injected_zeros() -> None:
    frame = _toy_frame()
    counts = invalid_zero_counts(frame)
    assert counts["RestingBP"].count == 1
    assert counts["Cholesterol"].count == 2


def test_count_affected_rows_is_the_union_not_the_sum() -> None:
    # Row 0 carries zeros in BOTH policy columns: the union is 1 row, while a
    # naive per-column sum would double-count it as 2.
    frame = pd.DataFrame(
        {"RestingBP": [0, 120, 130], "Cholesterol": [0, 200, 210]}
    )
    assert count_affected_rows(frame) == 1


def test_count_affected_rows_on_real_dataset() -> None:
    frame = load_dataset().frame
    # RestingBP's single zero is the same patient whose Cholesterol is zero,
    # so the union is 172 rather than 173.
    assert count_affected_rows(frame) == 172


# ---------------------------------------------------------------------------
# Class balance on the real dataset
# ---------------------------------------------------------------------------


def test_class_balance_on_pinned_dataset() -> None:
    frame = load_dataset().frame
    balance = compute_class_balance(frame)

    assert balance.total == 918
    assert balance.negative_count == 410
    assert balance.positive_count == 508
    assert balance.prevalence == pytest.approx(508 / 918, abs=1e-9)
    assert balance.imbalance_ratio == pytest.approx(508 / 410, abs=1e-9)


def test_class_balance_to_dict_is_json_safe() -> None:
    balance = compute_class_balance(_toy_frame())
    payload = balance.to_dict()
    assert payload["counts"] == {"0": 2, "1": 3}
    assert payload["total"] == 5
    assert 0.0 <= payload["prevalence"] <= 1.0


def test_audit_quality_aggregates_every_figure() -> None:
    frame = load_dataset().frame
    audit = audit_quality(frame)

    assert audit.row_count == 918
    assert audit.affected_rows == 172
    assert audit.policy.action == MEDIAN_IMPUTE
    payload = audit.to_dict()
    assert payload["zero_counts"]["Cholesterol"]["count"] == 172
    assert payload["class_balance"]["counts"] == {"0": 410, "1": 508}


# ---------------------------------------------------------------------------
# Policy application: no sentinel survives as a measurement
# ---------------------------------------------------------------------------


def test_policy_replaces_all_zeros_on_real_dataset() -> None:
    frame = load_dataset().frame
    cleaned = clean_with_policy(frame)

    for column in ZERO_AS_MISSING_COLUMNS:
        assert int((cleaned[column] == 0).sum()) == 0, column
    # Values that were real measurements are untouched.
    bp_mask = frame["RestingBP"] != 0
    chol_mask = frame["Cholesterol"] != 0
    pd.testing.assert_series_equal(
        cleaned.loc[bp_mask, "RestingBP"], frame.loc[bp_mask, "RestingBP"].astype(float),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cleaned.loc[chol_mask, "Cholesterol"],
        frame.loc[chol_mask, "Cholesterol"].astype(float),
        check_names=False,
    )


def test_policy_imputes_the_fit_frame_median() -> None:
    frame = _toy_frame()
    cleaned = clean_with_policy(frame)

    # Non-zero RestingBP is [120, 140, 160, 130] -> median 135.
    assert cleaned.loc[2, "RestingBP"] == 135
    # Non-zero Cholesterol is [200, 220, 240] -> median 220.
    assert cleaned.loc[1, "Cholesterol"] == 220
    assert cleaned.loc[3, "Cholesterol"] == 220


def test_unrelated_zero_is_not_touched() -> None:
    # Oldpeak == 0 is a legitimate measurement (no ST depression), so the
    # policy must leave it alone.
    cleaned = clean_with_policy(_toy_frame())
    assert list(cleaned["Oldpeak"]) == [0.0, 1.5, 0.0, 2.0, 0.5]


def test_policy_is_deterministic() -> None:
    frame = load_dataset().frame
    first = clean_with_policy(frame)
    second = clean_with_policy(frame)
    pd.testing.assert_frame_equal(first, second)


def test_fit_and_apply_are_separable() -> None:
    train = _toy_frame()
    imputer = fit_zero_policy(train)
    out1 = apply_zero_policy(train, imputer)
    out2 = apply_zero_policy(train, imputer)
    pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Leakage safety at the unit level: the transform never refits
# ---------------------------------------------------------------------------


def test_imputation_uses_train_median_not_transform_frame_median() -> None:
    train = pd.DataFrame({"RestingBP": [120, 140, 0, 160], "Cholesterol": [200, 0, 220, 240]})
    # A held-out row whose own values would produce a very different median.
    holdout = pd.DataFrame({"RestingBP": [0], "Cholesterol": [0]})

    imputer = fit_zero_policy(train)
    transformed = imputer.transform(holdout)

    # 140 / 220 come from train; a leaky refit on holdout alone could not.
    assert transformed.loc[0, "RestingBP"] == 140
    assert transformed.loc[0, "Cholesterol"] == 220


def test_imputer_is_sklearn_compatible_and_reusable() -> None:
    train = _toy_frame()
    imputer = ZeroMedianImputer()
    fitted = imputer.fit(train)
    assert fitted is imputer
    out = imputer.fit_transform(train)
    assert out is not None
    names = imputer.get_feature_names_out()
    assert "RestingBP" in list(names)


def test_repr_params_expose_columns() -> None:
    imputer = ZeroMedianImputer()
    assert imputer.get_params()["columns"] == ZERO_AS_MISSING_COLUMNS


# ---------------------------------------------------------------------------
# Observability: the policy is logged, never silent
# ---------------------------------------------------------------------------


def test_applying_policy_logs_each_column(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="heart.data.quality"):
        clean_with_policy(_toy_frame())

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Fitted zero-as-missing policy" in messages
    assert "RestingBP" in messages
    assert "Cholesterol" in messages
    assert "imputed with" in messages


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_missing_policy_column_raises_named_error() -> None:
    frame = _toy_frame().drop(columns=["Cholesterol"])
    with pytest.raises(MissingQualityColumnError) as excinfo:
        invalid_zero_counts(frame)
    assert "Cholesterol" in str(excinfo.value)


def test_missing_target_column_raises_named_error() -> None:
    frame = _toy_frame().drop(columns=[TARGET_COLUMN])
    with pytest.raises(MissingQualityColumnError) as excinfo:
        compute_class_balance(frame)
    assert TARGET_COLUMN in str(excinfo.value)


def test_all_zero_column_cannot_be_imputed() -> None:
    frame = pd.DataFrame(
        {"RestingBP": [0, 0, 0], "Cholesterol": [200, 220, 240]}
    )
    with pytest.raises(NoValidValuesError) as excinfo:
        fit_zero_policy(frame)
    assert "RestingBP" in str(excinfo.value)


def test_transform_before_fit_raises() -> None:
    imputer = ZeroMedianImputer()
    with pytest.raises(QualityError):
        imputer.transform(_toy_frame())


def test_non_dataframe_input_raises_quality_error() -> None:
    imputer = ZeroMedianImputer()
    with pytest.raises(QualityError):
        imputer.fit([[120, 200]])  # type: ignore[arg-type]


def test_unknown_policy_action_raises_on_audit() -> None:
    bad = ZeroPolicy(action="magic")
    with pytest.raises(UnknownPolicyError):
        audit_quality(_toy_frame(), policy=bad)


def test_unknown_policy_action_raises_on_clean() -> None:
    bad = ZeroPolicy(action="drop_rows")
    with pytest.raises(UnknownPolicyError):
        clean_with_policy(_toy_frame(), policy=bad)


def test_transform_rejects_frame_missing_policy_column() -> None:
    imputer = fit_zero_policy(_toy_frame())
    with pytest.raises(MissingQualityColumnError):
        imputer.transform(pd.DataFrame({"RestingBP": [120, 140]}))


# ---------------------------------------------------------------------------
# Report artifact
# ---------------------------------------------------------------------------


def test_report_contains_required_figures() -> None:
    audit, frame, dataset = build_quality_audit()
    report = render_quality_report(audit, dataset=dataset)

    # Actual zero counts.
    assert "`RestingBP` | 1 |" in report
    assert "`Cholesterol` | 172 |" in report
    # Applied policy.
    assert "**action:** `median_impute`" in report
    assert "training rows only" in report
    # Class balance.
    assert "`0` (no disease): **410**" in report
    assert "`1` (disease): **508**" in report
    assert "positive prevalence: **0.5534**" in report


def test_committed_report_is_current() -> None:
    report_path = Path(REPORTS_DIR) / QUALITY_REPORT_NAME
    assert report_path.exists(), "reports/data_quality.md must be committed"
    audit, _frame, dataset = build_quality_audit()
    expected = render_quality_report(audit, dataset=dataset)
    assert report_path.read_text(encoding="utf-8") == expected


def test_write_quality_report_writes_file(tmp_path: Path) -> None:
    audit, _frame, dataset = build_quality_audit()
    dest = write_quality_report(audit, dataset=dataset, path=tmp_path / "q.md")
    assert dest.exists()
    assert "Data-Quality Audit" in dest.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prints_audit_and_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "data_quality.md"
    exit_code = main(["--report", str(dest)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "RestingBP: 1 impossible zero(s)" in out
    assert "Cholesterol: 172 impossible zero(s)" in out
    assert "class balance: 0=410 1=508" in out
    assert "policy: action=median_impute" in out
    assert dest.exists()


def test_cli_no_write_skips_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "should_not_exist.md"
    exit_code = main(["--report", str(dest), "--no-write"])
    assert exit_code == 0
    assert not dest.exists()
    assert "class balance" in capsys.readouterr().out
