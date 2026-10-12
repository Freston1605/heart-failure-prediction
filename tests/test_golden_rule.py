"""Tests for the leakage golden-rule enforcement layer (M006 S02 T02).

Covers, per the task plan:

1. positive path -- an honest per-fold lineage passes every assertion;
2. pre-split-resampling negative -- a lineage built as if random
   over-sampling was applied to the FULL dataset before the split must
   raise ``GoldenRuleError`` naming the contaminating ids;
3. synthetic-rows-crossing-folds negative -- reserved-space synthetic ids
   appended into a validation fold's id set must raise ``GoldenRuleError``
   naming them;
4. id-space invariants -- synthetic ids start at ``n_original``, never
   collide with real ids, and kept/duplicated originals keep their ids;
5. the combined :func:`heart.eval.golden_rule.assert_golden_rule` accepts
   the honest lineage and rejects both sabotage shapes;
6. an ImportError-free smoke of ``imbalanced-learn`` proving the pinned
   dependency (imbalanced-learn==0.14.2) is importable alongside the
   sklearn stack.

Fixtures are inline; no test reads ``.gsd/`` or any gitignored path.
"""

from __future__ import annotations

import dataclasses

import pytest

from heart.config import RANDOM_SEED
from heart.eval.golden_rule import (
    GoldenRuleError,
    FoldPartition,
    ResampleLineage,
    allocate_synthetic_ids,
    assert_golden_rule,
    assert_sampling_after_split,
    assert_synthetic_rows_never_cross_folds,
    random_resample_train_rows,
)

N_ORIGINAL = 1000

# Inline fixture: a stratification-free id/label fold with a 4:1 imbalance
# (800 majority class 1, 200 minority class 0) over rows 0..999.
TRAIN_IDS = tuple(range(0, 800))
TRAIN_LABELS = tuple(1 if i % 4 else 0 for i in TRAIN_IDS)
VAL_IDS = tuple(range(800, 900))
TEST_IDS = tuple(range(900, 1000))


def honest_partition() -> FoldPartition:
    return FoldPartition(
        fold_id="fold-0",
        train_ids=TRAIN_IDS,
        validation_ids=VAL_IDS,
        test_ids=TEST_IDS,
    )


def honest_random_under_lineage() -> ResampleLineage:
    return random_resample_train_rows(
        TRAIN_IDS,
        TRAIN_LABELS,
        strategy="under",
        variant_key="random-under-sampler",
        mechanism="RandomUnderSampler",
        n_original=N_ORIGINAL,
        seed=RANDOM_SEED,
    )


def honest_random_over_lineage() -> ResampleLineage:
    return random_resample_train_rows(
        TRAIN_IDS,
        TRAIN_LABELS,
        strategy="over",
        variant_key="random-over-sampler",
        mechanism="RandomOverSampler",
        n_original=N_ORIGINAL,
        seed=RANDOM_SEED,
    )


def honest_smote_lineage() -> ResampleLineage:
    synthetic = allocate_synthetic_ids(N_ORIGINAL, 180, counter=0)
    return ResampleLineage(
        variant_key="smote-rung-1",
        mechanism="SMOTE",
        n_original=N_ORIGINAL,
        original_train_ids=TRAIN_IDS,
        resampled_train_ids=TRAIN_IDS + synthetic,
        synthetic_ids=synthetic,
        synthetic_count=len(synthetic),
    )


# ---------------------------------------------------------------------------
# 1. Positive path: honest per-fold lineage passes all assertions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lineage_factory",
    [honest_random_under_lineage, honest_random_over_lineage, honest_smote_lineage],
    ids=["under", "over", "smote"],
)
def test_honest_lineage_passes_all_assertions(lineage_factory):
    lineage = lineage_factory()
    partition = honest_partition()

    assert_sampling_after_split(lineage, partition)
    assert_synthetic_rows_never_cross_folds(lineage, partition)
    assert_golden_rule(lineage, partition)  # no exception


def test_honest_under_sampling_balances_classes_and_keeps_real_ids():
    lineage = honest_random_under_lineage()
    kept = lineage.resampled_train_ids
    # every kept id is a real original training row id
    assert set(kept) <= set(lineage.original_train_ids)
    assert lineage.synthetic_ids == ()
    assert lineage.synthetic_count == 0
    # balanced 1:1: 200 minority kept, 200 majority sampled down
    minority_kept = [i for i in kept if TRAIN_LABELS[i] == 0]
    majority_kept = [i for i in kept if TRAIN_LABELS[i] == 1]
    assert len(minority_kept) == len(TRAIN_LABELS) // 4  # all 200 minority
    assert len(majority_kept) == len(minority_kept)


def test_honest_over_sampling_duplicates_minority_with_real_ids():
    lineage = honest_random_over_lineage()
    kept = lineage.resampled_train_ids
    # every kept id is a real original id (duplicates repeat entries)
    assert set(kept) <= set(lineage.original_train_ids)
    assert lineage.synthetic_ids == ()
    assert lineage.synthetic_count == 0
    # duplicates: multiset larger than the original fold, same id universe
    assert len(kept) > len(lineage.original_train_ids)
    assert sorted(set(kept)) == sorted(set(TRAIN_IDS))
    assert len(kept) == 2 * (len(TRAIN_IDS) - len(TRAIN_LABELS) // 4)


def test_honest_helpers_are_reproducible():
    a = honest_random_under_lineage()
    b = random_resample_train_rows(
        TRAIN_IDS,
        TRAIN_LABELS,
        strategy="under",
        variant_key="random-under-sampler",
        mechanism="RandomUnderSampler",
        n_original=N_ORIGINAL,
        seed=RANDOM_SEED,
    )
    assert a.resampled_train_ids == b.resampled_train_ids


# ---------------------------------------------------------------------------
# 2. Pre-split-resampling negative test.
# ---------------------------------------------------------------------------


def test_presplit_resampling_raises_and_names_contaminating_ids():
    # Dishonest pipeline: random over-sampling applied to the FULL dataset
    # (rows 0..999) BEFORE the split, so validation and test ids appear in
    # the resampled training ids.
    duplicated_val = VAL_IDS[:6]
    duplicated_test = TEST_IDS[:4]
    sabotage = ResampleLineage(
        variant_key="random-over-sampler",
        mechanism="RandomOverSampler",
        n_original=N_ORIGINAL,
        original_train_ids=TRAIN_IDS + VAL_IDS + TEST_IDS,  # full dataset
        resampled_train_ids=TRAIN_IDS + duplicated_val + duplicated_test,
        synthetic_ids=(),
        synthetic_count=0,
    )
    with pytest.raises(GoldenRuleError) as excinfo:
        assert_sampling_after_split(sabotage, honest_partition())
    message = str(excinfo.value)
    for bad_id in (*duplicated_val, *duplicated_test):
        assert str(bad_id) in message
    assert "sampling after split" in message
    assert "random-over-sampler" in message
    # the combined assertion must catch the same sabotage
    with pytest.raises(GoldenRuleError):
        assert_golden_rule(sabotage, honest_partition())


def test_presplit_resampling_detected_even_when_original_ids_are_clean():
    # A lineage whose ORIGINAL train ids match the fold but whose resampled
    # ids picked up out-of-fold rows still fails (the detector does not
    # trust lineage bookkeeping).
    sabotage = ResampleLineage(
        variant_key="knn-imputer-plus-under",
        mechanism="RandomUnderSampler",
        n_original=N_ORIGINAL,
        original_train_ids=TRAIN_IDS,
        resampled_train_ids=TRAIN_IDS[:700] + VAL_IDS[:5],
        synthetic_ids=(),
        synthetic_count=0,
    )
    with pytest.raises(GoldenRuleError) as excinfo:
        assert_sampling_after_split(sabotage, honest_partition())
    message = str(excinfo.value)
    assert "800" in message  # VAL_IDS[0]
    assert "804" in message  # VAL_IDS[4]
    assert "resampled training ids outside" in message


def test_presplit_error_message_truncates_huge_contaminations():
    # every validation id contaminates: message must truncate but keep a count
    sabotage = ResampleLineage(
        variant_key="sabotage",
        mechanism="RandomOverSampler",
        n_original=N_ORIGINAL,
        original_train_ids=TRAIN_IDS + VAL_IDS,
        resampled_train_ids=TRAIN_IDS + VAL_IDS,
        synthetic_ids=(),
        synthetic_count=0,
    )
    with pytest.raises(GoldenRuleError) as excinfo:
        assert_sampling_after_split(sabotage, honest_partition())
    assert "(+" in str(excinfo.value)  # (+N more) tail


# ---------------------------------------------------------------------------
# 3. Synthetic-rows-crossing-folds negative test.
# ---------------------------------------------------------------------------


def test_synthetic_ids_in_validation_fold_raise_and_are_named():
    honest = honest_smote_lineage()
    leaked = honest.synthetic_ids[:5]
    partition = FoldPartition(
        fold_id="fold-0",
        train_ids=TRAIN_IDS,
        validation_ids=VAL_IDS + leaked,  # sabotage: synthetic rows in val fold
        test_ids=TEST_IDS,
    )
    with pytest.raises(GoldenRuleError) as excinfo:
        assert_synthetic_rows_never_cross_folds(honest, partition)
    message = str(excinfo.value)
    for sid in leaked:
        assert str(sid) in message
    assert "synthetic rows crossing folds" in message
    assert "smote-rung-1" in message
    with pytest.raises(GoldenRuleError):
        assert_golden_rule(honest, partition)


def test_synthetic_ids_in_test_rows_raise_and_are_named():
    honest = honest_smote_lineage()
    leaked = honest.synthetic_ids[100:103]
    partition = FoldPartition(
        fold_id="final-test",
        train_ids=TRAIN_IDS,
        validation_ids=VAL_IDS,
        test_ids=TEST_IDS + leaked,  # sabotage: synthetic rows in test set
    )
    with pytest.raises(GoldenRuleError) as excinfo:
        assert_synthetic_rows_never_cross_folds(honest, partition)
    message = str(excinfo.value)
    for sid in leaked:
        assert str(sid) in message
    assert "test rows" in message


# ---------------------------------------------------------------------------
# 4. Id-space invariants.
# ---------------------------------------------------------------------------


def test_synthetic_ids_start_at_n_original():
    assert allocate_synthetic_ids(N_ORIGINAL, 5) == (
        N_ORIGINAL,
        N_ORIGINAL + 1,
        N_ORIGINAL + 2,
        N_ORIGINAL + 3,
        N_ORIGINAL + 4,
    )


def test_synthetic_ids_with_running_counter_never_repeat():
    first = allocate_synthetic_ids(N_ORIGINAL, 50, counter=0)
    second = allocate_synthetic_ids(N_ORIGINAL, 50, counter=len(first))
    assert not (set(first) & set(second))
    assert min(second) == N_ORIGINAL + len(first)


def test_synthetic_ids_never_equal_any_real_id():
    real_ids = set(range(N_ORIGINAL))
    synthetic = allocate_synthetic_ids(N_ORIGINAL, 200)
    assert not (set(synthetic) & real_ids)
    assert all(sid >= N_ORIGINAL for sid in synthetic)


def test_lineage_constructor_rejects_real_space_synthetic_ids():
    with pytest.raises(GoldenRuleError) as excinfo:
        ResampleLineage(
            variant_key="bad-smote",
            mechanism="SMOTE",
            n_original=N_ORIGINAL,
            original_train_ids=TRAIN_IDS,
            resampled_train_ids=TRAIN_IDS,
            synthetic_ids=(5, N_ORIGINAL + 1),  # 5 is a REAL row id
            synthetic_count=2,
        )
    assert "reserved id space" in str(excinfo.value)
    assert "5" in str(excinfo.value)


def test_lineage_constructor_rejects_synthetic_count_mismatch():
    synthetic = allocate_synthetic_ids(N_ORIGINAL, 3)
    with pytest.raises(GoldenRuleError):
        ResampleLineage(
            variant_key="bad-count",
            mechanism="SMOTE",
            n_original=N_ORIGINAL,
            original_train_ids=TRAIN_IDS,
            resampled_train_ids=TRAIN_IDS,
            synthetic_ids=synthetic,
            synthetic_count=99,
        )


def test_lineage_constructor_rejects_duplicate_synthetic_ids():
    sid = allocate_synthetic_ids(N_ORIGINAL, 1)[0]
    with pytest.raises(GoldenRuleError) as excinfo:
        ResampleLineage(
            variant_key="bad-unique",
            mechanism="SMOTE",
            n_original=N_ORIGINAL,
            original_train_ids=TRAIN_IDS,
            resampled_train_ids=TRAIN_IDS,
            synthetic_ids=(sid, sid),
            synthetic_count=2,
        )
    assert "unique" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 5. Combined assert_golden_rule: accepts honest, rejects both sabotages.
# ---------------------------------------------------------------------------


def test_combined_assertion_accepts_all_honest_lineages():
    partition = honest_partition()
    for lineage in (
        honest_random_under_lineage(),
        honest_random_over_lineage(),
        honest_smote_lineage(),
    ):
        assert_golden_rule(lineage, partition)


def test_combined_assertion_reports_split_order_violation_first():
    # A lineage that violates BOTH halves must report the pre-split
    # violation (the combined rule checks sampling-after-split first).
    synthetic = allocate_synthetic_ids(N_ORIGINAL, 3)
    double_sabotage = ResampleLineage(
        variant_key="double-sabotage",
        mechanism="SMOTE",
        n_original=N_ORIGINAL,
        original_train_ids=TRAIN_IDS + VAL_IDS,
        resampled_train_ids=TRAIN_IDS + VAL_IDS + synthetic,
        synthetic_ids=synthetic,
        synthetic_count=len(synthetic),
    )
    partition = FoldPartition(
        fold_id="fold-0",
        train_ids=TRAIN_IDS,
        validation_ids=VAL_IDS + synthetic,  # also crosses folds
        test_ids=TEST_IDS,
    )
    with pytest.raises(GoldenRuleError) as excinfo:
        assert_golden_rule(double_sabotage, partition)
    assert "sampling after split" in str(excinfo.value)


def test_combined_assertion_accepts_a_lineage_set_for_one_fold():
    # The runner will wire multiple variant lineages per fold; the
    # assertion APIs accept a lineage set.
    partition = honest_partition()
    lineages = [
        honest_random_under_lineage(),
        honest_random_over_lineage(),
        honest_smote_lineage(),
    ]
    assert_golden_rule(lineages, partition)

    # one bad lineage in the set fails the whole set
    bad = dataclasses.replace(honest_random_under_lineage())
    bad_sabotage = ResampleLineage(
        variant_key=bad.variant_key,
        mechanism=bad.mechanism,
        n_original=bad.n_original,
        original_train_ids=bad.original_train_ids,
        resampled_train_ids=bad.resampled_train_ids + TEST_IDS[:2],
        synthetic_ids=(),
        synthetic_count=0,
    )
    with pytest.raises(GoldenRuleError):
        assert_golden_rule([honest_random_under_lineage(), bad_sabotage], partition)


# ---------------------------------------------------------------------------
# 6. Dependency smoke: imbalanced-learn imports cleanly (pinned ==0.14.2).
# ---------------------------------------------------------------------------


def test_imbalanced_learn_smoke_import():
    import imblearn
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    assert imblearn.__version__ == "0.14.2"
    # the sampler classes the gate runner will instantiate exist
    for cls in (SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler):
        assert hasattr(cls, "fit_resample")


def test_default_seed_is_project_seed():
    # the native helper defaults to heart.config.RANDOM_SEED
    import inspect

    signature = inspect.signature(random_resample_train_rows)
    assert signature.parameters["seed"].default == RANDOM_SEED
