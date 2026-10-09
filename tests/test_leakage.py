"""Tests for leakage-safe split creation and the automated leakage detectors.

The slice contract for S01/T05 is two-part:

1. Versioned train/test split artifacts are written and reproducible under a
   fixed seed, and they are **group-aware** (no near-duplicate cluster
   straddles the boundary).
2. An automated leakage test proves no test-row information influenced any
   fitted transform — and the detector must actually **fail** when the fit is
   deliberately sabotaged onto the full dataset.

The sabotage tests below are the executable proof of part 2: they hand the
detectors a pipeline fit on the whole frame and assert :class:`DataLeakageError`
fires. Positive tests use the real pinned dataset; the negative tests use
inline fixtures and deliberate sabotage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heart.config import RANDOM_SEED
from heart.data.duplicates import assign_duplicate_groups, detect_duplicates
from heart.data.load import load_dataset
from heart.data.pipeline import (
    PERTURB_NUMERIC_VALUE,
    DataLeakageError,
    FeatureFrameError,
    PipelineError,
    assert_fit_on_training_only,
    assert_no_test_influence,
    audit_fit_scope,
    build_preprocessing_pipeline,
    fit_on_train,
    fit_preprocessing,
    fitted_statistics,
    influence_probe,
    max_statistic_difference,
    perturb_test_rows,
    statistics_equal,
    transform_features,
)
from heart.data.schema import TARGET_COLUMN
from heart.data.split import (
    MANIFEST_FILENAME,
    N_SPLITS,
    SPLIT_VERSION,
    TEST_FILENAME,
    TRAIN_FILENAME,
    MissingTargetError,
    SplitConfigError,
    SplitIntegrityError,
    SplitLeakageError,
    SplitNotFoundError,
    assert_no_group_straddle,
    build_split,
    create_split,
    group_straddle_count,
    load_split_artifacts,
    load_split_frames,
    main,
    write_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def frame(dataset):
    return dataset.frame


@pytest.fixture(scope="module")
def split(frame):
    return create_split(frame)


@pytest.fixture(scope="module")
def duplicate_labels(frame):
    return assign_duplicate_groups(frame)


# ---------------------------------------------------------------------------
# Split: reproducibility, partition, and class balance
# ---------------------------------------------------------------------------


def test_split_is_reproducible_under_fixed_seed(frame):
    first = create_split(frame)
    second = create_split(frame)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert first.to_dict() == second.to_dict()


def test_split_partitions_every_row_exactly_once(frame, split):
    all_indices = np.sort(np.concatenate([split.train_indices, split.test_indices]))
    np.testing.assert_array_equal(all_indices, np.arange(len(frame)))
    assert split.train_rows + split.test_rows == split.row_count == len(frame)
    assert not set(split.train_indices) & set(split.test_indices)


def test_split_holds_out_roughly_one_fold(frame, split):
    assert split.n_splits == N_SPLITS
    assert 0.10 <= split.test_fraction <= 0.30


def test_split_preserves_class_balance_on_both_sides(frame, split):
    balance = split.class_balance(frame)
    assert set(balance["train"]) == {"0", "1"}
    assert set(balance["test"]) == {"0", "1"}
    overall_positive = balance["overall"]["1"] / split.row_count
    for side in ("train", "test"):
        rows = balance[side]["0"] + balance[side]["1"]
        prevalence = balance[side]["1"] / rows
        assert abs(prevalence - overall_positive) < 0.05


# ---------------------------------------------------------------------------
# Split: group awareness (the core leakage guarantee)
# ---------------------------------------------------------------------------


def test_split_keeps_every_duplicate_cluster_whole(frame, duplicate_labels, split):
    assert (
        group_straddle_count(
            duplicate_labels, split.train_indices, split.test_indices
        )
        == 0
    )
    report = detect_duplicates(frame)
    train = set(int(i) for i in split.train_indices)
    for cluster in report.near_duplicates:
        sides = {i in train for i in cluster.row_indices}
        assert len(sides) == 1, f"cluster {cluster.row_indices} straddles the split"


def test_split_reports_group_count_and_zero_straddle(split):
    assert split.group_count == 913  # 918 rows minus 5 merged duplicate pairs
    assert split.straddling_groups == 0


def test_naive_random_split_is_rejected(frame, duplicate_labels):
    # The naive stratified split (the mistake T04 quantifies) straddles a
    # near-duplicate cluster at the project seed, so the guard must fire.
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=frame[TARGET_COLUMN],
    )
    assert group_straddle_count(duplicate_labels, train_idx, test_idx) >= 1
    with pytest.raises(SplitLeakageError) as excinfo:
        assert_no_group_straddle(duplicate_labels, train_idx, test_idx)
    assert "straddle" in str(excinfo.value)


def test_assert_no_group_straddle_accepts_a_clean_partition():
    labels = np.array([0, 0, 1, 1])
    assert_no_group_straddle(labels, np.array([0, 1]), np.array([2, 3]))


# ---------------------------------------------------------------------------
# Split: artifacts on disk
# ---------------------------------------------------------------------------


def test_write_and_load_split_round_trips(frame, tmp_path):
    split = create_split(frame)
    artifacts = write_split(split, frame, root=tmp_path)

    assert artifacts.directory == tmp_path / SPLIT_VERSION
    for path in (artifacts.train_path, artifacts.test_path, artifacts.manifest_path):
        assert path.exists()

    loaded = load_split_artifacts(root=tmp_path)
    assert loaded.manifest["train_rows"] == split.train_rows
    assert loaded.manifest["test_rows"] == split.test_rows
    assert loaded.manifest["group_straddling"] == 0

    train, test = load_split_frames(root=tmp_path)
    pd.testing.assert_frame_equal(train, split.train_frame(frame))
    pd.testing.assert_frame_equal(test, split.test_frame(frame))


def test_split_artifacts_are_byte_reproducible(frame, tmp_path):
    split = create_split(frame)
    first = write_split(split, frame, root=tmp_path / "a")
    second = write_split(split, frame, root=tmp_path / "b")

    assert first.train_path.read_bytes() == second.train_path.read_bytes()
    assert first.test_path.read_bytes() == second.test_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()


def test_manifest_is_json_safe_and_names_its_digests(frame, tmp_path):
    artifacts = write_split(create_split(frame), frame, root=tmp_path)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["created_by"] == "heart.data.split"
    assert manifest["version"] == SPLIT_VERSION
    assert manifest["random_seed"] == RANDOM_SEED
    assert manifest["group_straddling"] == 0
    assert len(manifest["train_indices"]) == manifest["train_rows"]
    assert len(manifest["test_indices"]) == manifest["test_rows"]
    assert manifest["files"]["train"]["name"] == TRAIN_FILENAME
    assert manifest["files"]["test"]["name"] == TEST_FILENAME
    assert len(manifest["files"]["train"]["sha256"]) == 64


def test_committed_split_artifacts_are_current(tmp_path):
    committed = load_split_artifacts()
    assert committed.version == SPLIT_VERSION
    assert (committed.directory / MANIFEST_FILENAME).exists()

    split, dataset = build_split()
    fresh = write_split(split, dataset.frame, root=tmp_path)

    assert fresh.train_path.read_bytes() == committed.train_path.read_bytes()
    assert fresh.test_path.read_bytes() == committed.test_path.read_bytes()
    assert json.loads(fresh.manifest_path.read_text()) == committed.manifest


# ---------------------------------------------------------------------------
# Split: negative tests
# ---------------------------------------------------------------------------


def test_create_split_requires_target_column(frame):
    with pytest.raises(MissingTargetError) as excinfo:
        create_split(frame.drop(columns=[TARGET_COLUMN]))
    assert TARGET_COLUMN in str(excinfo.value)


def test_invalid_split_configuration_raises(frame):
    with pytest.raises(SplitConfigError):
        create_split(frame, n_splits=1)
    with pytest.raises(SplitConfigError):
        create_split(frame, test_fold=99)


def test_group_length_mismatch_raises(frame):
    with pytest.raises(SplitConfigError) as excinfo:
        create_split(frame, groups=np.array([0, 1, 2]))
    assert "one label per row" in str(excinfo.value)


def test_missing_split_version_raises(tmp_path):
    with pytest.raises(SplitNotFoundError):
        load_split_artifacts("v_absent", root=tmp_path)


def test_tampered_split_artifact_raises(frame, tmp_path):
    write_split(create_split(frame), frame, root=tmp_path)
    (tmp_path / SPLIT_VERSION / TEST_FILENAME).write_bytes(b"tampered\n")
    with pytest.raises(SplitIntegrityError) as excinfo:
        load_split_artifacts(root=tmp_path)
    assert "sha256" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Split: observability
# ---------------------------------------------------------------------------


def test_create_split_logs_its_configuration(frame, caplog):
    with caplog.at_level(logging.INFO, logger="heart.data.split"):
        create_split(frame)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "straddling groups=0" in messages
    assert "seed=42" in messages


# ---------------------------------------------------------------------------
# Pipeline: honest fit is train-only
# ---------------------------------------------------------------------------


def test_pipeline_fit_on_train_matches_a_train_only_refit(frame, split):
    pipeline = fit_on_train(frame, split.train_indices)
    audit = assert_fit_on_training_only(pipeline, frame, split.train_indices)
    assert audit.passed
    assert audit.matches_train_only
    assert audit.train_rows == split.train_rows


def test_train_and_full_fits_are_distinguishable(frame, split):
    # Non-vacuous detector: if the two fits were identical, a full-data fit
    # could never be told apart from a train-only fit.
    train_stats = fitted_statistics(fit_on_train(frame, split.train_indices))
    full_stats = fitted_statistics(fit_preprocessing(frame))
    assert max_statistic_difference(train_stats, full_stats) > 0


def test_audit_reports_observed_statistics_match(frame, split):
    pipeline = fit_on_train(frame, split.train_indices)
    audit = audit_fit_scope(pipeline, frame, split.train_indices)
    assert audit.observed_vs_train_max_diff == 0
    assert statistics_equal(
        fitted_statistics(pipeline),
        fitted_statistics(fit_on_train(frame, split.train_indices)),
    )


def test_transform_features_shape_and_completeness(frame, split):
    pipeline = fit_on_train(frame, split.train_indices)
    out = transform_features(pipeline, split.test_frame(frame))
    assert len(out) == split.test_rows
    assert out.isna().sum().sum() == 0
    assert out.shape[1] > len(split.test_frame(frame).columns)  # one-hot expansion


# ---------------------------------------------------------------------------
# Pipeline: SABOTAGE — the detector must catch a full-data fit
# ---------------------------------------------------------------------------


def test_detector_flags_pipeline_fit_on_full_data(frame, split):
    sabotaged = fit_preprocessing(frame)  # deliberately leaky: whole dataset
    with pytest.raises(DataLeakageError) as excinfo:
        assert_fit_on_training_only(sabotaged, frame, split.train_indices)
    message = str(excinfo.value)
    assert "FULL dataset" in message or "leakage" in message.lower()


def test_sabotaged_pipeline_matches_full_and_not_train(frame, split):
    sabotaged = fit_preprocessing(frame)
    audit = audit_fit_scope(sabotaged, frame, split.train_indices)
    assert not audit.passed
    assert audit.matches_full_data
    assert not audit.matches_train_only


def test_influence_probe_is_stable_for_honest_fit(frame, split):
    probe = assert_no_test_influence(frame, split.train_indices, split.test_indices)
    assert probe.stable
    assert probe.first_difference is None
    assert probe.test_rows_perturbed == split.test_rows


def test_influence_probe_detects_leaky_fit_path(frame, split):
    # Sabotage: a fitting function that ignores the training indices and fits
    # on the whole frame. Perturbing the test rows must move its statistics.
    def leaky_fit_on_indices(_frame: pd.DataFrame, _indices: np.ndarray):
        return fit_preprocessing(_frame)

    probe = influence_probe(
        frame,
        split.train_indices,
        split.test_indices,
        fit_on_indices=leaky_fit_on_indices,
    )
    assert not probe.stable

    with pytest.raises(DataLeakageError) as excinfo:
        assert_no_test_influence(
            frame,
            split.train_indices,
            split.test_indices,
            fit_on_indices=leaky_fit_on_indices,
        )
    assert "Held-out rows influenced" in str(excinfo.value)


def test_perturbation_touches_only_test_rows(frame, split):
    perturbed = perturb_test_rows(frame, split.test_indices)
    train = split.train_indices
    pd.testing.assert_frame_equal(perturbed.iloc[train], frame.iloc[train])
    assert (
        perturbed.iloc[split.test_indices]["Age"] == PERTURB_NUMERIC_VALUE
    ).all()
    assert perturbed.iloc[split.test_indices]["Sex"].eq("<LEAK-PROBE>").all()


# ---------------------------------------------------------------------------
# Pipeline: negative tests
# ---------------------------------------------------------------------------


def test_missing_feature_column_raises(frame):
    with pytest.raises(FeatureFrameError) as excinfo:
        fit_preprocessing(frame.drop(columns=["Age"]))
    assert "Age" in str(excinfo.value)


def test_unfitted_pipeline_statistics_raise():
    with pytest.raises(PipelineError):
        fitted_statistics(build_preprocessing_pipeline())


def test_fit_on_train_rejects_empty_indices(frame):
    with pytest.raises(PipelineError):
        fit_on_train(frame, np.array([], dtype=int))


def test_non_dataframe_input_raises():
    with pytest.raises(FeatureFrameError):
        fit_preprocessing([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prints_split_and_writes_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--output-root", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "dataset shape: (918, 12)" in out
    assert "train rows:" in out and "test rows:" in out
    assert "duplicate groups straddling train/test: 0" in out
    assert (tmp_path / SPLIT_VERSION / TRAIN_FILENAME).exists()
    assert (tmp_path / SPLIT_VERSION / TEST_FILENAME).exists()
    assert (tmp_path / SPLIT_VERSION / MANIFEST_FILENAME).exists()


def test_cli_no_write_skips_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--output-root", str(tmp_path), "--no-write"])
    assert exit_code == 0
    assert not (tmp_path / SPLIT_VERSION).exists()
    assert "class balance" in capsys.readouterr().out
