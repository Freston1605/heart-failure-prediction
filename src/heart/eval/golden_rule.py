"""Leakage golden-rule enforcement for treatment-candidate sampling (D034).

The golden rule: **resampling is applied inside the training fold only,
never before the split; synthetic rows never cross folds.** This module
turns that sentence into enforceable code with loud, named failures
(:class:`GoldenRuleError` quoting the offending row ids), so the M006 gate
criterion "enforced by negative tests that fail loudly on pre-split
resampling and synthetic rows crossing folds" is satisfied by executable
invariants rather than by review.

How the pieces fit together with imbalanced-learn (D035):

- The gate runner (T04) applies imblearn ``fit_resample`` ONLY on the fold's
  training rows, and only after the fold's ``ColumnTransformer`` has produced
  the fold's numeric feature matrix. The transformer is fitted on the
  training rows as well; nothing about the validation/test rows leaks into
  the fitted pipeline.
- Only real row ids are ever handed around at the id level. Synthetic
  vectors produced by SMOTE/ADASYN have no real row identity; they live in a
  RESERVED ID SPACE (ids ``>= n_original``, allocated by
  :func:`allocate_synthetic_ids` as ``n_original + running_counter``) so a
  synthetic id can never collide with a real row id or a fold id. Synthetic
  row counts are recorded in :class:`ResampleLineage` alongside the reserved
  ids; the numeric vectors themselves are never part of the lineage.
- Random under-/over-sampling duplicates or drops REAL rows only, so the
  native helper :func:`random_resample_train_rows` works at the index level
  (kept ids keep their original ids; synthetic ids stay empty) and negative
  tests are exact at the index level without paying for imblearn.
- The dishonest-pipeline scenario is NOT constructed by any helper here.
  Negative tests build sabotage lineages by hand (e.g. a lineage whose
  ``resampled_train_ids`` were drawn from the full dataset) and feed them
  straight to the assertion APIs, which fail loudly.

Structure rules (mirroring :mod:`heart.eval.treatment_declared`):

1. **Declarative lineage, assertive guards.** :class:`ResampleLineage` only
   records what happened; the assertion APIs are the enforcement layer and
   deliberately do NOT trust the lineage's own bookkeeping.
2. **Reserved id space is structural.** Synthetic ids are validated at
   lineage construction (``>= n_original``, unique, never equal to a real
   id handed in). Membership rules (nothing out-of-fold in the resampled
   training set, no synthetic id in validation/test) live ONLY in the
   assertions, so sabotage lineages can be constructed for negative tests.
3. **Fail loudly with offending ids quoted.** Every assertion failure names
   the contaminating ids in the message (sorted, truncated display).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Union

import numpy as np

from heart.config import RANDOM_SEED

__all__ = [
    "GoldenRuleError",
    "ResampleLineage",
    "FoldPartition",
    "allocate_synthetic_ids",
    "random_resample_train_rows",
    "assert_sampling_after_split",
    "assert_synthetic_rows_never_cross_folds",
    "assert_golden_rule",
]

_MAX_SHOWN_IDS = 10

_SamplingStrategy = Literal["under", "over"]


class GoldenRuleError(Exception):
    """A treatment-candidate sampling step violated the leakage golden rule.

    Raised by the assertion APIs with the offending row ids quoted in the
    message. Callers must treat this as a hard stop: the measured number is
    contaminated and must never reach the gate.
    """


def _format_ids(ids: Iterable[int]) -> str:
    """Render an id collection for an error message.

    Sorted, quoted, truncated at :data:`_MAX_SHOWN_IDS` with a tail count so
    huge contaminations stay readable.
    """
    ordered = sorted(set(int(i) for i in ids))
    if not ordered:
        return "[]"
    shown = ordered[:_MAX_SHOWN_IDS]
    head = "[" + ", ".join(str(i) for i in shown) + "]"
    extra = len(ordered) - len(shown)
    if extra:
        return f"{head} (+{extra} more)"
    return head


def _as_id_tuple(ids: Iterable[int], where: str) -> tuple[int, ...]:
    values = tuple(int(i) for i in ids)
    for i in values:
        if i < 0:
            raise GoldenRuleError(
                f"{where}: row id {i} is negative; ids are non-negative integers."
            )
    return values


@dataclass(frozen=True)
class ResampleLineage:
    """Id-level record of one resampling step on one fold's training rows.

    Fields:

    - ``variant_key``: the declared treatment candidate this lineage belongs
      to (``heart.eval.treatment_declared`` ``TreatmentCandidate.variant_key``).
    - ``mechanism``: mechanism label (e.g. ``"sampling"``, or the sampler
      class name such as ``"SMOTE"`` / ``"ADASYN"`` / ``"RandomUnderSampler"``).
    - ``n_original``: size of the full dataset id space. Real ids are
      ``0 .. n_original - 1``; synthetic ids are ``>= n_original``.
    - ``original_train_ids``: the fold's training row ids BEFORE sampling.
    - ``resampled_train_ids``: real row ids AFTER sampling. Kept and
      duplicated originals keep their ids (duplicates may appear as repeated
      entries, e.g. minority duplication by random over-sampling).
    - ``synthetic_ids``: reserved-space ids (``>= n_original``) of synthetic
      rows created by the step; empty for under-/over-sampling of real rows.
    - ``synthetic_count``: number of synthetic rows created (equal to
      ``len(synthetic_ids)``; kept as an explicit count for samplers that
      only report counts).
    """

    variant_key: str
    mechanism: str
    n_original: int
    original_train_ids: tuple[int, ...]
    resampled_train_ids: tuple[int, ...]
    synthetic_ids: tuple[int, ...]
    synthetic_count: int

    def __post_init__(self) -> None:
        if not self.variant_key or not isinstance(self.variant_key, str):
            raise GoldenRuleError(
                "ResampleLineage: variant_key must be a non-empty string."
            )
        if not self.mechanism or not isinstance(self.mechanism, str):
            raise GoldenRuleError(
                f"ResampleLineage ({self.variant_key}): mechanism must be a "
                "non-empty string."
            )
        if self.n_original < 0:
            raise GoldenRuleError(
                f"ResampleLineage ({self.variant_key}): n_original must be "
                f"non-negative, got {self.n_original}."
            )
        original = _as_id_tuple(self.original_train_ids, "ResampleLineage")
        resampled = _as_id_tuple(self.resampled_train_ids, "ResampleLineage")
        synthetic = _as_id_tuple(self.synthetic_ids, "ResampleLineage")
        # Structural id-space rule: every synthetic id lives in the reserved
        # space (>= n_original) and never collides with a real id. Membership
        # rules (out-of-fold contamination, synthetic ids crossing folds) are
        # NOT checked here -- they are the assertion APIs' job, so negative
        # tests can construct sabotage lineages by hand.
        real_id_space = set(range(self.n_original))
        bad = [i for i in synthetic if i < self.n_original or i in real_id_space]
        if bad:
            raise GoldenRuleError(
                f"ResampleLineage ({self.variant_key}): synthetic ids must live "
                f"in the reserved id space (>= n_original={self.n_original}); "
                f"offending synthetic ids: {_format_ids(bad)}"
            )
        if len(set(synthetic)) != len(synthetic):
            raise GoldenRuleError(
                f"ResampleLineage ({self.variant_key}): synthetic ids must be "
                "unique within a lineage."
            )
        if self.synthetic_count != len(synthetic):
            raise GoldenRuleError(
                f"ResampleLineage ({self.variant_key}): synthetic_count "
                f"({self.synthetic_count}) must equal len(synthetic_ids) "
                f"({len(synthetic)})."
            )
        # Freeze the id tuples against accidental mutation by callers.
        object.__setattr__(self, "original_train_ids", original)
        object.__setattr__(self, "resampled_train_ids", resampled)
        object.__setattr__(self, "synthetic_ids", synthetic)


@dataclass(frozen=True)
class FoldPartition:
    """Id-level partition of one CV fold (or the held-out test rows).

    ``train_ids`` are the rows the fold's pipeline may see; ``validation_ids``
    are the rows scored inside CV; ``test_ids`` are the held-out rows
    (empty for inner folds, populated for the final train/test partition).
    """

    fold_id: str
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    test_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("train_ids", "validation_ids", "test_ids"):
            object.__setattr__(
                self, name, _as_id_tuple(getattr(self, name), f"FoldPartition {name}")
            )
        if not self.fold_id or not isinstance(self.fold_id, str):
            raise GoldenRuleError("FoldPartition: fold_id must be a non-empty string.")


def allocate_synthetic_ids(
    n_original: int, count: int, counter: int = 0
) -> tuple[int, ...]:
    """Draw ``count`` synthetic row ids from the reserved id space.

    Reserved ids start at ``n_original + counter`` and run consecutively; a
    caller that resamples several folds in one run passes the running counter
    forward so ids never repeat across folds. A synthetic id can therefore
    never collide with a real row id (``< n_original``) or a fold id.
    """
    if n_original < 0:
        raise GoldenRuleError(
            f"allocate_synthetic_ids: n_original must be non-negative, got {n_original}."
        )
    if count < 0:
        raise GoldenRuleError(
            f"allocate_synthetic_ids: count must be non-negative, got {count}."
        )
    if counter < 0:
        raise GoldenRuleError(
            f"allocate_synthetic_ids: counter must be non-negative, got {counter}."
        )
    start = n_original + counter
    return tuple(range(start, start + count))


def random_resample_train_rows(
    train_ids: Sequence[int],
    train_labels: Sequence[int],
    *,
    strategy: _SamplingStrategy,
    variant_key: str,
    mechanism: str,
    sampling_ratio: float = 1.0,
    n_original: int | None = None,
    seed: int = RANDOM_SEED,
) -> ResampleLineage:
    """Native index-level random under-/over-sampling helper.

    Mirrors ``imblearn``'s ``RandomUnderSampler`` / ``RandomOverSampler``
    semantics at the index level with ``sampling_strategy=sampling_ratio``:

    - ``strategy="under"`` keeps every minority-class row and under-samples
      the majority class so that ``n_majority_kept / n_minority ==
      sampling_ratio`` (default 1.0 balances the fold 1:1).
    - ``strategy="over"`` keeps every majority row and duplicates minority
      rows (sampled with replacement, seeded) so that
      ``n_minority_after / n_majority == sampling_ratio``.

    Kept/duplicated rows keep their REAL ids, so the returned
    :class:`ResampleLineage` has empty ``synthetic_ids`` and
    ``synthetic_count == 0`` -- random under-/over-sampling never creates
    synthetic rows. Seeded from ``heart.config.RANDOM_SEED`` by default so
    the same fold under the same variant reproduces exactly.

    ``n_original`` defaults to ``max(train_ids) + 1`` when the caller cannot
    supply the full-dataset size; pass the true dataset size from the runner
    so the reserved id space is anchored correctly.
    """
    if strategy not in ("under", "over"):
        raise GoldenRuleError(
            f"random_resample_train_rows ({variant_key}): strategy must be "
            f"'under' or 'over', got {strategy!r}."
        )
    ids = [int(i) for i in train_ids]
    labels = [int(y) for y in train_labels]
    if len(ids) != len(labels):
        raise GoldenRuleError(
            f"random_resample_train_rows ({variant_key}): train_ids and "
            f"train_labels must have equal length ({len(ids)} vs {len(labels)})."
        )
    if not ids:
        raise GoldenRuleError(
            f"random_resample_train_rows ({variant_key}): training fold is empty."
        )
    if sampling_ratio <= 0:
        raise GoldenRuleError(
            f"random_resample_train_rows ({variant_key}): sampling_ratio must "
            f"be positive, got {sampling_ratio}."
        )
    if n_original is None:
        n_original = max(ids) + 1
    if n_original <= max(ids):
        raise GoldenRuleError(
            f"random_resample_train_rows ({variant_key}): n_original "
            f"({n_original}) must exceed every training row id "
            f"(max id {max(ids)}); synthetic id space would collide."
        )
    minority_label = min(set(labels))
    majority_label = max(set(labels))
    if minority_label == majority_label:
        raise GoldenRuleError(
            f"random_resample_train_rows ({variant_key}): fold has a single "
            f"class ({majority_label}); resampling is undefined."
        )
    minority_ids = [i for i, y in zip(ids, labels) if y == minority_label]
    majority_ids = [i for i, y in zip(ids, labels) if y == majority_label]
    rng = np.random.default_rng(seed)
    if strategy == "under":
        target_majority = int(round(sampling_ratio * len(minority_ids)))
        if target_majority >= len(majority_ids):
            kept_majority = list(majority_ids)
        else:
            kept_majority = sorted(
                rng.choice(majority_ids, size=target_majority, replace=False).tolist()
            )
        kept = list(minority_ids) + kept_majority
    else:
        target_minority = int(round(sampling_ratio * len(majority_ids)))
        if target_minority <= len(minority_ids):
            added = []
        else:
            extra = target_minority - len(minority_ids)
            added = rng.choice(minority_ids, size=extra, replace=True).tolist()
        kept = list(ids) + added
    return ResampleLineage(
        variant_key=variant_key,
        mechanism=mechanism,
        n_original=n_original,
        original_train_ids=tuple(ids),
        resampled_train_ids=tuple(kept),
        synthetic_ids=(),
        synthetic_count=0,
    )


def _normalize_lineages(
    lineage: Union[ResampleLineage, Iterable[ResampleLineage]],
) -> list[ResampleLineage]:
    if isinstance(lineage, ResampleLineage):
        return [lineage]
    return list(lineage)


def assert_sampling_after_split(
    lineage: Union[ResampleLineage, Iterable[ResampleLineage]],
    fold_partition: FoldPartition,
) -> None:
    """Fail when resampled training rows contain ids outside the fold's train set.

    The pre-split-resampling detector: resampling applied to the FULL dataset
    before the split leaves validation/test row ids inside the resampled
    training set. Any id in ``resampled_train_ids`` (or in
    ``original_train_ids``) that is not part of ``fold_partition.train_ids``
    raises :class:`GoldenRuleError` quoting the contaminating ids.
    """
    fold_train = set(fold_partition.train_ids)
    for lin in _normalize_lineages(lineage):
        original = set(lin.original_train_ids)
        resampled = set(lin.resampled_train_ids) - set(lin.synthetic_ids)
        contaminating_resampled = resampled - fold_train
        contaminating_original = original - fold_train
        if contaminating_original or contaminating_resampled:
            parts = []
            if contaminating_original:
                parts.append(
                    "original training ids outside the fold train set: "
                    + _format_ids(contaminating_original)
                )
            if contaminating_resampled:
                parts.append(
                    "resampled training ids outside the fold train set: "
                    + _format_ids(contaminating_resampled)
                )
            raise GoldenRuleError(
                "Golden rule violated (sampling after split): variant "
                f"{lin.variant_key!r} ({lin.mechanism}) on fold "
                f"{fold_partition.fold_id!r} contains row ids that are NOT part "
                f"of the fold's training rows -- resampling must run inside the "
                f"training fold only, never before the split. "
                + "; ".join(parts)
            )


def assert_synthetic_rows_never_cross_folds(
    lineage: Union[ResampleLineage, Iterable[ResampleLineage]],
    fold_partition: FoldPartition,
) -> None:
    """Fail when reserved-space synthetic ids reach validation or test rows.

    Synthetic rows (SMOTE/ADASYN vectors in the reserved id space) exist only
    inside the training fold that produced them. Any synthetic id appearing
    in ``fold_partition.validation_ids`` or ``fold_partition.test_ids``
    raises :class:`GoldenRuleError` naming the offending synthetic ids.
    """
    for lin in _normalize_lineages(lineage):
        synthetic = set(lin.synthetic_ids)
        crossed_validation = synthetic & set(fold_partition.validation_ids)
        crossed_test = synthetic & set(fold_partition.test_ids)
        if crossed_validation or crossed_test:
            parts = []
            if crossed_validation:
                parts.append(
                    "synthetic ids found in validation fold: "
                    + _format_ids(crossed_validation)
                )
            if crossed_test:
                parts.append(
                    "synthetic ids found in test rows: " + _format_ids(crossed_test)
                )
            raise GoldenRuleError(
                "Golden rule violated (synthetic rows crossing folds): variant "
                f"{lin.variant_key!r} ({lin.mechanism}) on fold "
                f"{fold_partition.fold_id!r} leaks reserved-space synthetic row "
                "ids outside the training fold that produced them. "
                + "; ".join(parts)
            )


def assert_golden_rule(
    lineage: Union[ResampleLineage, Iterable[ResampleLineage]],
    fold_partition: FoldPartition,
) -> None:
    """Enforce both golden-rule halves for one fold's lineage set.

    Combines :func:`assert_sampling_after_split` and
    :func:`assert_synthetic_rows_never_cross_folds`. The pre-split check runs
    first so a contaminated lineage reports the split-order violation before
    any synthetic-row finding.
    """
    assert_sampling_after_split(lineage, fold_partition)
    assert_synthetic_rows_never_cross_folds(lineage, fold_partition)


# Re-export for callers that want the dataclass replace helper without
# importing dataclasses themselves (keeps sabotage construction terse).
replace = dataclasses.replace
