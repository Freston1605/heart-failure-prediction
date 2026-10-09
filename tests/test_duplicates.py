"""Tests for near-duplicate detection and the cross-split leakage audit.

Positive tests exercise the real pinned dataset through the loader; negative
tests drive injected fixtures (missing columns, non-numeric tolerances, empty
frames, oversized frames, stale reports). A fixture test proves the detector
finds injected near-duplicates — the slice-level requirement — and a
group-aware-split test proves the leakage guarantee holds structurally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from heart.config import RANDOM_SEED, REPORTS_DIR
from heart.data import duplicates as duplicates_module
from heart.data.duplicates import (
    DEFAULT_TOLERANCES,
    DUPLICATE_CATEGORICAL_COLUMNS,
    EXACT_KEY_ALL_COLUMNS,
    LEAKAGE_REPORT_NAME,
    MAX_PAIRWISE_ROWS,
    ColumnTolerance,
    DuplicateError,
    EmptyDatasetError,
    FrameTooLargeError,
    MissingDuplicateColumnError,
    NonDataFrameError,
    NonNumericToleranceError,
    assign_duplicate_groups,
    audit_naive_split,
    build_leakage_audit,
    detect_duplicates,
    find_exact_duplicates,
    find_near_duplicate_clusters,
    main,
    render_leakage_report,
    validate_tolerances,
    write_leakage_report,
)
from heart.data.load import load_dataset
from heart.data.schema import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    """One full-schema patient row; overrides inject duplicate relationships."""
    base: dict[str, object] = {
        "Age": 50,
        "RestingBP": 130,
        "Cholesterol": 220,
        "MaxHR": 150,
        "Oldpeak": 1.0,
        "Sex": "M",
        "ChestPainType": "ASY",
        "FastingBS": 0,
        "RestingECG": "Normal",
        "ExerciseAngina": "N",
        "ST_Slope": "Flat",
        TARGET_COLUMN: 1,
    }
    base.update(overrides)
    return base


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(EXACT_KEY_ALL_COLUMNS))


def _synthetic_clusters(n_patients: int) -> pd.DataFrame:
    """``n_patients`` distinct patients, each recorded twice within tolerance.

    Patients are spaced ten years apart so a patient's two recordings can only
    cluster with each other, never with a neighbour's.
    """
    rows: list[dict[str, object]] = []
    for patient in range(n_patients):
        target = patient % 2
        age = 30 + patient * 10
        sex = "M" if patient % 2 else "F"
        rows.append(_row(Age=age, RestingBP=130, Cholesterol=220, MaxHR=150, Oldpeak=1.0,
                         Sex=sex, ChestPainType="ASY",
                         FastingBS=0, RestingECG="Normal", ExerciseAngina="N",
                         ST_Slope="Flat", **{TARGET_COLUMN: target}))
        # Second recording: same categorical presentation, numerics within tol.
        rows.append(_row(Age=age + 1, RestingBP=133, Cholesterol=225, MaxHR=148, Oldpeak=1.1,
                         Sex=sex, ChestPainType="ASY",
                         FastingBS=0, RestingECG="Normal", ExerciseAngina="N",
                         ST_Slope="Flat", **{TARGET_COLUMN: target}))
    return _frame(rows)


# ---------------------------------------------------------------------------
# Declared method integrity
# ---------------------------------------------------------------------------


def test_tolerances_cover_every_numeric_feature() -> None:
    assert {t.column for t in DEFAULT_TOLERANCES} == set(NUMERIC_COLUMNS)


def test_categorical_key_is_features_only() -> None:
    assert set(DUPLICATE_CATEGORICAL_COLUMNS) == set(CATEGORICAL_COLUMNS) - {
        TARGET_COLUMN
    }
    assert TARGET_COLUMN not in DUPLICATE_CATEGORICAL_COLUMNS


def test_exact_keys_cover_the_schema() -> None:
    assert set(EXACT_KEY_ALL_COLUMNS) == set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS)
    assert TARGET_COLUMN not in duplicates_module.EXACT_KEY_FEATURES


# ---------------------------------------------------------------------------
# Injected near-duplicates (the slice-level detector requirement)
# ---------------------------------------------------------------------------


def test_finds_injected_near_duplicate_pair() -> None:
    frame = _frame(
        [
            _row(),
            _row(Age=51, RestingBP=133, Cholesterol=225, MaxHR=148, Oldpeak=1.1),
        ]
    )
    clusters = find_near_duplicate_clusters(frame)
    assert len(clusters) == 1
    assert clusters[0].row_indices == (0, 1)
    assert clusters[0].size == 2


def test_finds_injected_exact_duplicates() -> None:
    frame = _frame([_row(), _row(), _row(Age=70, ChestPainType="ATA")])
    report = find_exact_duplicates(frame)
    assert report.duplicate_rows == 2
    assert report.duplicate_groups == 1
    assert report.group_size_histogram == {2: 1}
    assert report.row_indices == (0, 1)
    assert report.has_duplicates


def test_exact_duplicates_on_a_custom_column_subset() -> None:
    frame = _frame(
        [
            _row(Sex="F", Age=61),
            _row(Sex="F", Age=61, Cholesterol=300),  # same Age+Sex, different else
            _row(Sex="M", Age=61),
        ]
    )
    report = find_exact_duplicates(frame, columns=("Age", "Sex"))
    assert report.duplicate_rows == 2
    assert report.duplicate_groups == 1
    assert report.row_indices == (0, 1)


def test_transitive_clustering_merges_a_chain() -> None:
    # A(50)-B(51)-C(52): each adjacent pair is within +/-1, but A and C differ
    # by 2 and are NOT directly near-duplicates. Union-find must still merge
    # all three into one cluster.
    frame = _frame([_row(Age=50), _row(Age=51), _row(Age=52)])
    clusters = find_near_duplicate_clusters(frame)
    assert len(clusters) == 1
    assert clusters[0].row_indices == (0, 1, 2)


def test_label_conflict_is_reported() -> None:
    frame = _frame([_row(**{TARGET_COLUMN: 1}), _row(Age=51, **{TARGET_COLUMN: 0})])
    report = detect_duplicates(frame)
    assert report.cluster_count == 1
    assert report.label_conflicts == 1
    assert report.near_duplicates[0].label_agreement is False


def test_label_agreement_is_true_for_consistent_cluster() -> None:
    frame = _frame([_row(), _row(Age=51)])
    report = detect_duplicates(frame)
    assert report.label_conflicts == 0
    assert report.near_duplicates[0].label_agreement is True


# ---------------------------------------------------------------------------
# Negative controls: distinct rows must NOT be flagged
# ---------------------------------------------------------------------------


def test_clearly_distinct_rows_are_not_flagged() -> None:
    frame = _frame(
        [
            _row(),
            _row(Age=70, RestingBP=180, Cholesterol=400, MaxHR=100, Oldpeak=3.0,
                 Sex="F", ChestPainType="ATA", RestingECG="LVH", ST_Slope="Up"),
        ]
    )
    assert find_near_duplicate_clusters(frame) == ()


def test_different_categorical_presentation_is_not_a_duplicate() -> None:
    frame = _frame([_row(), _row(ChestPainType="ATA")])
    assert find_near_duplicate_clusters(frame) == ()


def test_numeric_difference_beyond_tolerance_is_not_a_duplicate() -> None:
    # Age differs by 5 (> +/-1), everything else identical.
    frame = _frame([_row(), _row(Age=55)])
    assert find_near_duplicate_clusters(frame) == ()


def test_tolerance_boundary_is_inclusive() -> None:
    # Age exactly +/-1 is the declared boundary and must match.
    frame = _frame([_row(Age=50), _row(Age=51)])
    assert len(find_near_duplicate_clusters(frame)) == 1


# ---------------------------------------------------------------------------
# Real dataset
# ---------------------------------------------------------------------------


def test_pinned_dataset_has_no_exact_duplicates() -> None:
    frame = load_dataset().frame
    assert find_exact_duplicates(frame).duplicate_rows == 0
    assert find_exact_duplicates(
        frame, columns=duplicates_module.EXACT_KEY_FEATURES
    ).duplicate_rows == 0


def test_pinned_dataset_near_duplicate_clusters() -> None:
    report = detect_duplicates(load_dataset().frame)
    assert report.row_count == 918
    assert report.cluster_count == 5
    assert report.clustered_rows == 10
    assert report.label_conflicts == 0

    expected = {
        frozenset((65, 232)),
        frozenset((146, 163)),
        frozenset((452, 565)),
        frozenset((560, 597)),
        frozenset((641, 783)),
    }
    observed = {frozenset(c.row_indices) for c in report.near_duplicates}
    assert observed == expected
    # Every discovered cluster is exactly a pair.
    assert all(c.size == 2 for c in report.near_duplicates)


def test_pinned_dataset_positive_cluster_targets() -> None:
    report = detect_duplicates(load_dataset().frame)
    by_pair = {frozenset(c.row_indices): c.target_values for c in report.near_duplicates}
    assert by_pair[frozenset((452, 565))] == (1, 1)
    assert by_pair[frozenset((65, 232))] == (0, 0)


def test_report_to_dict_is_json_safe() -> None:
    report = detect_duplicates(load_dataset().frame)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["row_count"] == 918
    assert payload["near_duplicate_cluster_count"] == 5
    assert payload["exact_all_columns"]["duplicate_rows"] == 0


# ---------------------------------------------------------------------------
# Group labels for downstream group-aware splitting
# ---------------------------------------------------------------------------


def test_group_labels_share_within_cluster_and_unique_for_singletons() -> None:
    report = detect_duplicates(load_dataset().frame)
    labels = report.group_labels()
    assert len(labels) == 918

    clustered: set[int] = set()
    for cluster in report.near_duplicates:
        members = list(cluster.row_indices)
        clustered.update(members)
        assert len(set(labels[members])) == 1

    singletons = [i for i in range(918) if i not in clustered]
    assert len(set(labels[singletons])) == len(singletons)


def test_assign_duplicate_groups_matches_report_labels() -> None:
    frame = load_dataset().frame
    report = detect_duplicates(frame)
    assert (assign_duplicate_groups(frame, report) == report.group_labels()).all()


def test_assign_duplicate_groups_computes_report_when_absent() -> None:
    frame = load_dataset().frame
    labels = assign_duplicate_groups(frame)
    assert len(labels) == 918


def test_assign_duplicate_groups_rejects_stale_report() -> None:
    small = _frame([_row(), _row(Age=51)])
    report = detect_duplicates(small)
    bigger = _frame([_row(), _row(Age=51), _row(Age=70, ChestPainType="ATA")])
    with pytest.raises(DuplicateError):
        assign_duplicate_groups(bigger, report)


# ---------------------------------------------------------------------------
# Cross-split leakage audit
# ---------------------------------------------------------------------------


def test_naive_split_leaks_and_group_aware_does_not_on_pinned_data() -> None:
    frame = load_dataset().frame
    report = detect_duplicates(frame)

    naive = audit_naive_split(frame, report, random_state=RANDOM_SEED)
    grouped = audit_naive_split(
        frame, report, random_state=RANDOM_SEED, group_aware=True
    )

    assert naive.straddling_groups >= 1
    assert naive.leaky_test_rows >= 1
    assert grouped.straddling_groups == 0
    assert grouped.leaky_test_rows == 0
    assert grouped.group_aware is True
    assert naive.stratified is True


def test_group_aware_split_never_straddles_synthetic_clusters() -> None:
    frame = _synthetic_clusters(20)
    report = detect_duplicates(frame)
    assert report.cluster_count == 20

    grouped = audit_naive_split(frame, report, random_state=RANDOM_SEED, group_aware=True)
    assert grouped.straddling_groups == 0
    assert grouped.leaky_test_rows == 0


def test_naive_split_of_identical_rows_leaks_every_test_row() -> None:
    rows = [_row(**{TARGET_COLUMN: i % 2}) for i in range(10)]
    frame = _frame(rows)
    report = detect_duplicates(frame)
    assert report.cluster_count == 1
    assert report.clustered_rows == 10

    naive = audit_naive_split(frame, report, random_state=RANDOM_SEED)
    assert naive.straddling_groups == 1
    assert naive.leaky_test_rows == naive.test_rows


def test_split_audit_serialises() -> None:
    frame = load_dataset().frame
    report = detect_duplicates(frame)
    payload = json.loads(
        json.dumps(audit_naive_split(frame, report).to_dict())
    )
    assert payload["random_state"] == RANDOM_SEED


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_non_dataframe_input_raises() -> None:
    with pytest.raises(NonDataFrameError):
        find_exact_duplicates([[1, 2, 3]])  # type: ignore[arg-type]


def test_empty_frame_raises() -> None:
    with pytest.raises(EmptyDatasetError):
        find_near_duplicate_clusters(pd.DataFrame())


def test_missing_categorical_column_raises() -> None:
    frame = _frame([_row(), _row(Age=51)]).drop(columns=["ChestPainType"])
    with pytest.raises(MissingDuplicateColumnError) as excinfo:
        find_near_duplicate_clusters(frame)
    assert "ChestPainType" in str(excinfo.value)


def test_missing_tolerance_column_raises() -> None:
    frame = _frame([_row(), _row(Age=51)]).drop(columns=["Oldpeak"])
    with pytest.raises(MissingDuplicateColumnError) as excinfo:
        validate_tolerances(frame, DEFAULT_TOLERANCES)
    assert "Oldpeak" in str(excinfo.value)


def test_non_numeric_tolerance_column_raises() -> None:
    frame = _frame([_row(), _row()])
    bad = (ColumnTolerance("Sex", 1.0, "nonsense"),)
    with pytest.raises(NonNumericToleranceError) as excinfo:
        validate_tolerances(frame, bad)
    assert "Sex" in str(excinfo.value)


def test_frame_too_large_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(duplicates_module, "MAX_PAIRWISE_ROWS", 2)
    assert MAX_PAIRWISE_ROWS > 2  # guard the guard
    with pytest.raises(FrameTooLargeError):
        find_near_duplicate_clusters(_frame([_row(), _row(Age=51), _row(Age=52)]))


def test_invalid_block_size_raises() -> None:
    frame = _frame([_row(), _row(Age=51)])
    with pytest.raises(DuplicateError):
        find_near_duplicate_clusters(frame, block_size=0)


def test_invalid_test_size_raises() -> None:
    frame = _frame([_row(), _row(Age=51)])
    report = detect_duplicates(frame)
    with pytest.raises(DuplicateError):
        audit_naive_split(frame, report, test_size=1.5)


def test_audit_rejects_mismatched_report() -> None:
    small = _frame([_row(), _row(Age=51)])
    report = detect_duplicates(small)
    bigger = _frame([_row(), _row(Age=51), _row(Age=70, ChestPainType="ATA")])
    with pytest.raises(DuplicateError):
        audit_naive_split(bigger, report)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_detection_logs_a_summary(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="heart.data.duplicates"):
        detect_duplicates(_frame([_row(), _row(Age=51), _row(Age=70, ChestPainType="ATA")]))
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Duplicate audit" in messages
    assert "near-duplicate" in messages


def test_naive_split_logs_straddle_count(caplog: pytest.LogCaptureFixture) -> None:
    frame = load_dataset().frame
    report = detect_duplicates(frame)
    with caplog.at_level(logging.INFO, logger="heart.data.duplicates"):
        audit_naive_split(frame, report)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "naive split" in messages
    assert "straddling groups" in messages


# ---------------------------------------------------------------------------
# Report artifact
# ---------------------------------------------------------------------------


def test_report_contains_required_figures() -> None:
    report, naive, grouped, dataset = build_leakage_audit()
    text = render_leakage_report(report, naive, grouped, dataset=dataset)

    # Exact-duplicate count.
    assert "all 12 columns (incl. target) | 0 | 0" in text
    # Near-duplicate clusters with the method used.
    assert "**Method:** tolerance-match" in text
    assert "**near-duplicate clusters: 5**" in text
    assert "rows participating in a cluster: **10**" in text
    # Cross-split leakage finding.
    assert "naive (stratified random)" in text
    assert "group-aware" in text
    assert "**Finding:** a naive split straddles" in text
    # Downstream guidance.
    assert "assign_duplicate_groups" in text


def test_committed_report_is_current() -> None:
    report_path = Path(REPORTS_DIR) / LEAKAGE_REPORT_NAME
    assert report_path.exists(), "reports/leakage_audit.md must be committed"
    report, naive, grouped, dataset = build_leakage_audit()
    expected = render_leakage_report(report, naive, grouped, dataset=dataset)
    assert report_path.read_text(encoding="utf-8") == expected


def test_write_leakage_report_writes_file(tmp_path: Path) -> None:
    report, naive, grouped, dataset = build_leakage_audit()
    dest = write_leakage_report(report, naive, grouped, dataset=dataset, path=tmp_path / "l.md")
    assert dest.exists()
    assert "Leakage Audit" in dest.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prints_audit_and_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "leakage_audit.md"
    exit_code = main(["--report", str(dest)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "exact duplicates: all-columns=0 features=0" in out
    assert "near-duplicate clusters: 5" in out
    assert "group-aware split: 0 straddling cluster(s), 0 leaky test row(s)" in out
    assert dest.exists()


def test_cli_no_write_skips_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "should_not_exist.md"
    exit_code = main(["--report", str(dest), "--no-write"])
    assert exit_code == 0
    assert not dest.exists()
    assert "near-duplicate clusters" in capsys.readouterr().out
