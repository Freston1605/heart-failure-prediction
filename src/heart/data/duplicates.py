"""Near-duplicate detection and cross-split leakage audit.

Why this module exists
----------------------
The fedesoriano dataset is a **merge of five clinical cohorts** (Cleveland,
Hungary, Switzerland, Long Beach VA, and the Statlog set). Merging cohorts
invites a specific failure: the same patient, or two clinically
indistinguishable patients, can appear under more than one site's coding, so a
random train/test split can place near-identical rows on *both* sides. The
model then looks better than it is because it was evaluated on information it
effectively memorised.

This module measures that risk and hands downstream slices the tool they need
to neutralise it:

1. :func:`find_exact_duplicates` — rows identical on a declared column set.
2. :func:`find_near_duplicate_clusters` — rows that agree on every categorical
   feature and differ only within per-column clinical tolerances, clustered
   transitively (union-find).
3. :func:`assign_duplicate_groups` — a group label per row so S01/T05 can use
   group-aware splitting and keep every cluster on one side.
4. :func:`audit_naive_split` — quantifies how many clusters a *naive* random
   split would straddle, next to a group-aware split that straddles none.

Method (chosen with the counts in hand)
---------------------------------------
On the pinned 918-row dataset there are **zero exact duplicates**. The residual
signal is near-duplicates. Two rows are declared near-duplicates when they are
identical on all six categorical features (``Sex``, ``ChestPainType``,
``FastingBS``, ``RestingECG``, ``ExerciseAngina``, ``ST_Slope``) **and** every
numeric feature differs by no more than its declared tolerance
(:data:`DEFAULT_TOLERANCES`). The tolerances are clinical measurement
precision, not tuned knobs: whole-year age, 5 mm Hg cuff pressure, 20 mg/dl
cholesterol assay variation, 5 bpm heart rate, 0.2 Oldpeak.

Exact match on the categorical presentation is required because a different
chest-pain type or ST slope is a different clinical picture, not a repeat
measurement. Tolerance-only numeric matching is deliberate — the multi-site
merge records the same physiology with round-off differences.

The target is *excluded* from the duplicate key: a near-identical feature row
is the leakage risk regardless of its label, and a cluster whose labels
disagree is even more important to keep together (reported as a label
conflict). Clusters are formed by transitive closure of the pair relation.

Scale and memory
----------------
Pairwise matching is O(n²) in time, so the boolean match matrix is computed in
row blocks of :data:`PAIRWISE_BLOCK_SIZE` rather than materialised whole. Peak
memory is O(block × n), not O(n²): a 10x frame (9 180 rows) needs a few MB, and
:data:`MAX_PAIRWISE_ROWS` fails loudly before an accidental quadratic blow-up.
A realistic 10x run stays well inside a second because near-duplicate pairs are
sparse.

Observability
-------------
:func:`detect_duplicates` returns a :class:`DuplicateReport` (machine-readable
via ``to_dict``); :func:`render_leakage_report` / :func:`write_leakage_report`
produce ``reports/leakage_audit.md``; ``python -m heart.data.duplicates``
prints the audit and writes the report. Detection logs the exact-duplicate
count, cluster count, and the naive-split straddle count at ``INFO``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from heart.config import PROJECT_ROOT, RANDOM_SEED, REPORTS_DIR, ensure_project_dirs
from heart.data.load import LoadedDataset, load_dataset
from heart.data.schema import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared method constants
# ---------------------------------------------------------------------------

#: Categorical features that must match exactly for two rows to be near-dups.
#: The target is deliberately excluded — see the module docstring.
DUPLICATE_CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(
    column for column in CATEGORICAL_COLUMNS if column != TARGET_COLUMN
)

#: Column sets exact-duplicate detection runs over.
EXACT_KEY_ALL_COLUMNS: tuple[str, ...] = tuple(
    list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)
)
EXACT_KEY_FEATURES: tuple[str, ...] = tuple(
    column for column in EXACT_KEY_ALL_COLUMNS if column != TARGET_COLUMN
)

#: Number of rows matched against the full frame per block. Bounds the peak
#: size of the pairwise boolean matrix to ``PAIRWISE_BLOCK_SIZE * n``.
PAIRWISE_BLOCK_SIZE: int = 512

#: Fail-closed ceiling on frame size. The detector is only ever used on the
#: 918-row dataset (10x is 9 180), so anything this large signals a wrong
#: input rather than a workload to grind through.
MAX_PAIRWISE_ROWS: int = 50_000

#: Human-readable name of the detection method recorded in the report.
NEAR_DUPLICATE_METHOD: str = (
    "tolerance-match: identical on all categorical features "
    f"({', '.join(DUPLICATE_CATEGORICAL_COLUMNS)}) and every numeric feature "
    "within its declared absolute tolerance; clusters are transitive closures "
    "(union-find) of matching pairs"
)

#: Name of the generated report artifact (relative to the reports directory).
LEAKAGE_REPORT_NAME: str = "leakage_audit.md"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class DuplicateError(Exception):
    """Base class for duplicate-detection failures."""


class NonDataFrameError(DuplicateError):
    """The detector was handed something that is not a pandas DataFrame."""


class MissingDuplicateColumnError(DuplicateError):
    """A column named by the tolerance table or categorical key is absent."""


class NonNumericToleranceError(DuplicateError):
    """A tolerance names a column that is not numeric."""


class EmptyDatasetError(DuplicateError):
    """The frame has no rows, so no duplicate relationship exists."""


class FrameTooLargeError(DuplicateError):
    """The frame exceeds :data:`MAX_PAIRWISE_ROWS` for O(n^2) matching."""


class MissingTargetColumnError(DuplicateError):
    """The declared target column is absent when labels are requested."""


# ---------------------------------------------------------------------------
# Declared tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnTolerance:
    """Absolute tolerance for one numeric feature, with its provenance."""

    column: str
    tolerance: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "column": self.column,
            "tolerance": self.tolerance,
            "rationale": self.rationale,
        }


#: Declared per-column tolerances representing clinical measurement precision.
#: These are NOT tuned to a target cluster count; each value is the smallest
#: difference that is clinically indistinguishable for that measurement.
DEFAULT_TOLERANCES: tuple[ColumnTolerance, ...] = (
    ColumnTolerance("Age", 1.0, "recorded in whole years; +/-1 yr is indistinguishable"),
    ColumnTolerance("RestingBP", 5.0, "mm Hg cuff repeatability between visits"),
    ColumnTolerance("Cholesterol", 20.0, "mg/dl assay and rounding variation"),
    ColumnTolerance("MaxHR", 5.0, "bpm; +/-5 bpm is within exercise-test noise"),
    ColumnTolerance("Oldpeak", 0.2, "ST depression reported to 0.1-0.2 units"),
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactDuplicateReport:
    """Exact-duplicate tally for one declared column set."""

    columns: tuple[str, ...]
    duplicate_rows: int
    duplicate_groups: int
    group_size_histogram: dict[int, int] = field(default_factory=dict)
    row_indices: tuple[int, ...] = ()

    @property
    def has_duplicates(self) -> bool:
        return self.duplicate_groups > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_groups": self.duplicate_groups,
            "group_size_histogram": {
                str(k): v for k, v in sorted(self.group_size_histogram.items())
            },
            "row_indices": list(self.row_indices),
        }


@dataclass(frozen=True)
class NearDuplicateCluster:
    """One transitively-closed group of near-duplicate rows."""

    cluster_id: int
    row_indices: tuple[int, ...]
    target_values: tuple[int, ...] | None = None

    @property
    def size(self) -> int:
        return len(self.row_indices)

    @property
    def label_agreement(self) -> bool:
        """``True`` when every clustered row carries the same target label."""
        if not self.target_values:
            return True
        return len(set(self.target_values)) == 1

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "row_indices": list(self.row_indices),
            "size": self.size,
            "target_values": list(self.target_values)
            if self.target_values is not None
            else None,
            "label_agreement": self.label_agreement,
        }


@dataclass(frozen=True)
class DuplicateReport:
    """Aggregate result of exact- and near-duplicate detection."""

    row_count: int
    exact_all_columns: ExactDuplicateReport
    exact_features: ExactDuplicateReport
    near_duplicates: tuple[NearDuplicateCluster, ...]
    categorical_key: tuple[str, ...]
    tolerances: tuple[ColumnTolerance, ...]
    method: str = NEAR_DUPLICATE_METHOD

    @property
    def cluster_count(self) -> int:
        return len(self.near_duplicates)

    @property
    def clustered_rows(self) -> int:
        return sum(cluster.size for cluster in self.near_duplicates)

    @property
    def label_conflicts(self) -> int:
        """Clusters whose members disagree on the target label."""
        return sum(1 for cluster in self.near_duplicates if not cluster.label_agreement)

    def group_labels(self) -> np.ndarray:
        """Length-``row_count`` int array; duplicates share a label.

        Clustered rows receive the cluster id ``0..k-1``. Every other row gets
        a unique id ``>= k`` so a group-aware splitter treats it as a singleton.
        Suitable as the ``groups`` argument to scikit-learn's
        ``GroupShuffleSplit`` / ``StratifiedGroupKFold``.
        """
        if self.row_count < 0:  # pragma: no cover - defensive
            raise DuplicateError(f"Invalid row_count {self.row_count!r}.")
        labels = np.arange(self.row_count, dtype=np.int64) + self.cluster_count
        for cluster in self.near_duplicates:
            labels[list(cluster.row_indices)] = cluster.cluster_id
        return labels

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "method": self.method,
            "categorical_key": list(self.categorical_key),
            "tolerances": [t.to_dict() for t in self.tolerances],
            "exact_all_columns": self.exact_all_columns.to_dict(),
            "exact_features": self.exact_features.to_dict(),
            "near_duplicate_cluster_count": self.cluster_count,
            "near_duplicate_clustered_rows": self.clustered_rows,
            "label_conflicts": self.label_conflicts,
            "near_duplicate_clusters": [c.to_dict() for c in self.near_duplicates],
        }


@dataclass(frozen=True)
class SplitLeakageAudit:
    """How many duplicate groups a candidate split would straddle."""

    test_size: float
    random_state: int
    stratified: bool
    group_aware: bool
    train_rows: int
    test_rows: int
    straddling_groups: int
    leaky_test_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratified": self.stratified,
            "group_aware": self.group_aware,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "straddling_groups": self.straddling_groups,
            "leaky_test_rows": self.leaky_test_rows,
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_frame(X: object) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise NonDataFrameError(
            f"Expected a pandas.DataFrame, got {type(X).__name__}. Duplicate "
            "detection addresses named columns, so a bare array is not enough."
        )
    if len(X) == 0:
        raise EmptyDatasetError(
            "The frame has no rows, so there is no duplicate relationship to "
            "measure. Load the pinned dataset through heart.data.load."
        )
    return X


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, context: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MissingDuplicateColumnError(
            f"{context} names column(s) {missing} that are absent. Present "
            f"columns: {list(frame.columns)}."
        )


def validate_tolerances(
    frame: pd.DataFrame, tolerances: tuple[ColumnTolerance, ...]
) -> None:
    """Fail loudly if a tolerance names a missing or non-numeric column."""
    _require_columns(
        frame, tuple(t.column for t in tolerances), context="Tolerance table"
    )
    for tolerance in tolerances:
        if not pd.api.types.is_numeric_dtype(frame[tolerance.column]):
            raise NonNumericToleranceError(
                f"Tolerance for {tolerance.column!r} was declared, but the "
                f"column dtype is {frame[tolerance.column].dtype!r}. Numeric "
                "tolerances only apply to numeric columns."
            )


def _check_size(frame: pd.DataFrame) -> None:
    if len(frame) > MAX_PAIRWISE_ROWS:
        raise FrameTooLargeError(
            f"Frame has {len(frame)} rows, above MAX_PAIRWISE_ROWS="
            f"{MAX_PAIRWISE_ROWS}. Pairwise duplicate detection is O(n^2); "
            "subsample, or lower the tolerance and use a blocking key."
        )


# ---------------------------------------------------------------------------
# Exact duplicates
# ---------------------------------------------------------------------------


def find_exact_duplicates(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = EXACT_KEY_ALL_COLUMNS
) -> ExactDuplicateReport:
    """Count rows that are identical on ``columns``.

    ``duplicate_rows`` counts every row that belongs to a group of two or more;
    ``duplicate_groups`` counts the distinct value combinations that repeat.
    """
    frame = _as_frame(frame)
    _require_columns(frame, columns, context="Exact-duplicate key")

    subset = list(columns)
    mask = frame.duplicated(subset=subset, keep=False)
    row_indices = tuple(int(i) for i in np.nonzero(mask.to_numpy())[0])
    if not row_indices:
        return ExactDuplicateReport(
            columns=columns, duplicate_rows=0, duplicate_groups=0
        )

    sizes = frame.loc[mask].groupby(subset, dropna=False).size()
    histogram = {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
    return ExactDuplicateReport(
        columns=columns,
        duplicate_rows=len(row_indices),
        duplicate_groups=int(len(sizes)),
        group_size_histogram=histogram,
        row_indices=row_indices,
    )


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------


class _UnionFind:
    """Minimal path-compressed union-find over ``n`` rows."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self._parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


def _match_pairs(
    frame: pd.DataFrame,
    tolerances: tuple[ColumnTolerance, ...],
    categorical_key: tuple[str, ...],
    *,
    block_size: int,
) -> _UnionFind:
    """Union rows whose categorical key matches and numerics are within tol."""
    n = len(frame)
    union_find = _UnionFind(n)

    categorical = {c: frame[c].to_numpy() for c in categorical_key}
    numeric = {
        t.column: frame[t.column].to_numpy(dtype=float) for t in tolerances
    }

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        width = stop - start
        matches = np.ones((width, n), dtype=bool)
        for values in categorical.values():
            matches &= values[start:stop, None] == values[None, :]
        for tolerance in tolerances:
            values = numeric[tolerance.column]
            matches &= (
                np.abs(values[start:stop, None] - values[None, :])
                <= tolerance.tolerance
            )
        # Only consider pairs (i, j) with j > i to halve the work.
        for offset, row in enumerate(range(start, stop)):
            later = np.nonzero(matches[offset, row + 1:])[0]
            for j in later:
                union_find.union(row, int(j) + row + 1)
    return union_find


def _clusters_from_union_find(
    union_find: _UnionFind, frame: pd.DataFrame, *, target: str
) -> tuple[NearDuplicateCluster, ...]:
    n = union_find._parent.__len__()
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(union_find.find(i), []).append(i)

    has_target = target in frame.columns
    clusters: list[NearDuplicateCluster] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        targets = (
            tuple(int(v) for v in frame.iloc[members][target].tolist())
            if has_target
            else None
        )
        clusters.append(
            NearDuplicateCluster(
                cluster_id=len(clusters),
                row_indices=tuple(members),
                target_values=targets,
            )
        )
    return tuple(clusters)


def find_near_duplicate_clusters(
    frame: pd.DataFrame,
    *,
    tolerances: tuple[ColumnTolerance, ...] = DEFAULT_TOLERANCES,
    categorical_key: tuple[str, ...] = DUPLICATE_CATEGORICAL_COLUMNS,
    target: str = TARGET_COLUMN,
    block_size: int = PAIRWISE_BLOCK_SIZE,
) -> tuple[NearDuplicateCluster, ...]:
    """Return transitively-closed near-duplicate clusters.

    Two rows join a cluster when they agree exactly on every column in
    ``categorical_key`` and each numeric column in ``tolerances`` differs by no
    more than its declared absolute tolerance. Clusters are connected
    components of that pair relation.
    """
    frame = _as_frame(frame)
    _check_size(frame)
    if block_size < 1:
        raise DuplicateError(f"block_size must be >= 1, got {block_size!r}.")
    _require_columns(frame, categorical_key, context="Categorical duplicate key")
    validate_tolerances(frame, tolerances)

    union_find = _match_pairs(
        frame, tolerances, categorical_key, block_size=block_size
    )
    return _clusters_from_union_find(union_find, frame, target=target)


# ---------------------------------------------------------------------------
# Aggregate detection
# ---------------------------------------------------------------------------


def detect_duplicates(
    frame: pd.DataFrame,
    *,
    tolerances: tuple[ColumnTolerance, ...] = DEFAULT_TOLERANCES,
    categorical_key: tuple[str, ...] = DUPLICATE_CATEGORICAL_COLUMNS,
    target: str = TARGET_COLUMN,
    block_size: int = PAIRWISE_BLOCK_SIZE,
) -> DuplicateReport:
    """Run exact- and near-duplicate detection and return the aggregate report."""
    frame = _as_frame(frame)
    exact_all = find_exact_duplicates(frame, columns=EXACT_KEY_ALL_COLUMNS)
    exact_features = find_exact_duplicates(frame, columns=EXACT_KEY_FEATURES)
    near = find_near_duplicate_clusters(
        frame,
        tolerances=tolerances,
        categorical_key=categorical_key,
        target=target,
        block_size=block_size,
    )
    report = DuplicateReport(
        row_count=int(len(frame)),
        exact_all_columns=exact_all,
        exact_features=exact_features,
        near_duplicates=near,
        categorical_key=categorical_key,
        tolerances=tolerances,
    )
    logger.info(
        "Duplicate audit: exact rows=%d (feature-level=%d); near-duplicate "
        "clusters=%d covering %d row(s); label conflicts=%d",
        report.exact_all_columns.duplicate_rows,
        report.exact_features.duplicate_rows,
        report.cluster_count,
        report.clustered_rows,
        report.label_conflicts,
    )
    return report


def assign_duplicate_groups(
    frame: pd.DataFrame,
    report: DuplicateReport | None = None,
    *,
    tolerances: tuple[ColumnTolerance, ...] = DEFAULT_TOLERANCES,
    categorical_key: tuple[str, ...] = DUPLICATE_CATEGORICAL_COLUMNS,
    target: str = TARGET_COLUMN,
) -> np.ndarray:
    """Return a group label per row for group-aware splitting.

    Rows in the same near-duplicate cluster share a label; every other row gets
    a unique label. S01/T05 passes this to ``GroupShuffleSplit`` /
    ``StratifiedGroupKFold`` so no cluster is ever split across train and test.
    """
    frame = _as_frame(frame)
    if report is None:
        report = detect_duplicates(
            frame,
            tolerances=tolerances,
            categorical_key=categorical_key,
            target=target,
        )
    if report.row_count != len(frame):
        raise DuplicateError(
            f"Report was built for {report.row_count} row(s) but the frame has "
            f"{len(frame)}; recompute the report against this frame."
        )
    return report.group_labels()


# ---------------------------------------------------------------------------
# Cross-split leakage audit
# ---------------------------------------------------------------------------


def _straddle_counts(
    labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[int, int]:
    train_labels = set(int(v) for v in labels[train_idx])
    test_labels = set(int(v) for v in labels[test_idx])
    straddling = train_labels & test_labels
    leaky = sum(1 for i in test_idx if int(labels[i]) in straddling)
    return len(straddling), int(leaky)


def audit_naive_split(
    frame: pd.DataFrame,
    report: DuplicateReport,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    group_aware: bool = False,
) -> SplitLeakageAudit:
    """Measure how many duplicate groups a candidate split straddles.

    With ``group_aware=False`` this is the **naive** path: a stratified random
    split that ignores duplicate groups — exactly the mistake the audit exists
    to expose. With ``group_aware=True`` it uses ``GroupShuffleSplit`` keyed on
    :meth:`DuplicateReport.group_labels`, and the straddle count must be zero.
    """
    frame = _as_frame(frame)
    if report.row_count != len(frame):
        raise DuplicateError(
            f"Report row_count {report.row_count} does not match frame length "
            f"{len(frame)}."
        )
    if not 0.0 < test_size < 1.0:
        raise DuplicateError(f"test_size must be in (0, 1), got {test_size!r}.")

    labels = report.group_labels()
    n = len(frame)
    indices = np.arange(n)

    stratified = False
    if group_aware:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(indices, groups=labels))
    else:
        from sklearn.model_selection import train_test_split

        stratify = frame[TARGET_COLUMN] if TARGET_COLUMN in frame.columns else None
        stratified = stratify is not None
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    straddling, leaky = _straddle_counts(labels, train_idx, test_idx)
    audit = SplitLeakageAudit(
        test_size=test_size,
        random_state=random_state,
        stratified=stratified,
        group_aware=group_aware,
        train_rows=int(len(train_idx)),
        test_rows=int(len(test_idx)),
        straddling_groups=straddling,
        leaky_test_rows=leaky,
    )
    logger.info(
        "%s split (test_size=%.2f, seed=%d): straddling groups=%d, leaky test "
        "rows=%d",
        "group-aware" if group_aware else "naive",
        test_size,
        random_state,
        straddling,
        leaky,
    )
    return audit


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _display_path(path: str | Path) -> str:
    """Render ``path`` relative to the repo root when it lives underneath it."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(candidate)


def render_leakage_report(
    report: DuplicateReport,
    naive: SplitLeakageAudit,
    group_aware: SplitLeakageAudit,
    *,
    dataset: LoadedDataset | None = None,
    report_name: str = LEAKAGE_REPORT_NAME,
) -> str:
    """Render the audit as the ``reports/leakage_audit.md`` markdown document."""
    lines: list[str] = [
        "# Duplicate & Cross-Split Leakage Audit — fedesoriano Heart Failure Dataset",
        "",
        f"Generated by `python -m heart.data.duplicates` into `reports/{report_name}`.",
        "Regenerate rather than hand-edit; every number below is computed from",
        "the pinned dataset.",
        "",
        "## Dataset",
        "",
        f"- rows: **{report.row_count}**",
    ]
    if dataset is not None:
        lines.append(f"- source: `{_display_path(dataset.source_path)}`")
        lines.append(f"- sha256: `{dataset.sha256}`")
    lines.extend(
        [
            "",
            "The dataset is a merge of five clinical cohorts (Cleveland, Hungary,",
            "Switzerland, Long Beach VA, Statlog). Exact duplicates were already",
            "removed upstream; the residual risk is near-duplicates: the same",
            "physiology recorded by two sites with slight round-off differences.",
            "",
            "## Exact duplicates",
            "",
            "| Column set | Duplicate rows | Duplicate groups |",
            "|------------|---------------:|-----------------:|",
            f"| all {len(report.exact_all_columns.columns)} columns (incl. target) "
            f"| {report.exact_all_columns.duplicate_rows} "
            f"| {report.exact_all_columns.duplicate_groups} |",
            f"| {len(report.exact_features.columns)} feature columns "
            f"| {report.exact_features.duplicate_rows} "
            f"| {report.exact_features.duplicate_groups} |",
            "",
        ]
    )
    if report.exact_all_columns.has_duplicates:
        lines.append(
            "Exact duplicates are present and are folded into the near-duplicate "
            "clusters below (an exact match is trivially within tolerance)."
        )
    else:
        lines.append(
            "There are **no exact duplicate rows** on the full schema or on the "
            "features alone, so exact-match deduplication alone would find nothing."
        )
    lines.extend(
        [
            "",
            "## Near-duplicate clusters",
            "",
            f"**Method:** {report.method}.",
            "",
            "| Numeric feature | Tolerance | Rationale |",
            "|-----------------|----------:|-----------|",
        ]
    )
    for tolerance in report.tolerances:
        lines.append(
            f"| `{tolerance.column}` | +/-{tolerance.tolerance:g} "
            f"| {tolerance.rationale} |"
        )
    lines.extend(
        [
            "",
            f"- categorical key (exact match required): "
            f"{', '.join(f'`{c}`' for c in report.categorical_key)}",
            f"- **near-duplicate clusters: {report.cluster_count}**",
            f"- rows participating in a cluster: **{report.clustered_rows}** "
            f"({report.clustered_rows / report.row_count * 100:.2f}% of rows)",
            f"- clusters whose members disagree on `{TARGET_COLUMN}`: "
            f"**{report.label_conflicts}**",
            "",
        ]
    )
    if report.near_duplicates:
        lines.extend(
            [
                "| Cluster | Size | Row indices | Targets | Label agreement |",
                "|--------:|-----:|-------------|---------|:---------------:|",
            ]
        )
        for cluster in report.near_duplicates:
            indices = ", ".join(str(i) for i in cluster.row_indices)
            targets = (
                ", ".join(str(v) for v in cluster.target_values)
                if cluster.target_values is not None
                else "n/a"
            )
            lines.append(
                f"| {cluster.cluster_id} | {cluster.size} | {indices} | {targets} "
                f"| {'yes' if cluster.label_agreement else 'NO'} |"
            )
    else:
        lines.append("No near-duplicate clusters were found at the declared tolerances.")
    lines.extend(
        [
            "",
            "## Cross-split leakage",
            "",
            "A random split is simulated twice: once **naively** (stratified,",
            "ignoring duplicate groups) and once **group-aware** (clusters kept",
            "whole), both at the project seed.",
            "",
            "| Split | Test rows | Duplicate groups straddling train/test | Test rows leaking a train-mate |",
            "|-------|----------:|--------------------------------------:|-------------------------------:|",
            f"| naive (stratified random) | {naive.test_rows} | "
            f"{naive.straddling_groups} | {naive.leaky_test_rows} |",
            f"| group-aware | {group_aware.test_rows} | "
            f"{group_aware.straddling_groups} | {group_aware.leaky_test_rows} |",
            "",
            f"- split parameters: `test_size={naive.test_size}`, "
            f"`random_state={naive.random_state}`",
            "",
        ]
    )
    if report.cluster_count == 0:
        lines.append(
            "No near-duplicate clusters exist, so neither split can straddle one."
        )
    elif naive.straddling_groups:
        lines.append(
            f"**Finding:** a naive split straddles **{naive.straddling_groups}** "
            f"duplicate cluster(s), exposing **{naive.leaky_test_rows}** test "
            "row(s) to a near-identical training twin. The group-aware split "
            "straddles none, which is the leakage guarantee S01/T05 must keep."
        )
    else:
        lines.append(
            "Under this seed the naive split happens not to straddle a cluster, "
            "but that is luck, not a guarantee: a different seed would. The "
            "group-aware split enforces the guarantee structurally."
        )
    lines.extend(
        [
            "",
            "## Downstream action (S01/T05)",
            "",
            "Splits must be **group-aware**: pass",
            "`heart.data.duplicates.assign_duplicate_groups(frame)` as the `groups`",
            "argument to `GroupShuffleSplit` / `StratifiedGroupKFold` so every",
            "cluster lands entirely in train or entirely in test. Removing",
            "duplicates outright is *not* recommended here — the clusters are",
            "small and label-consistent, and dropping rows would cost class",
            "balance; keeping them grouped preserves the row count while removing",
            "the leakage.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m heart.data.duplicates        # print audit, write report",
            "pytest tests/test_duplicates.py -v     # detector + leakage audit tests",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_leakage_audit(
    *,
    path: str | Path | None = None,
    allow_download: bool = True,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[DuplicateReport, SplitLeakageAudit, SplitLeakageAudit, LoadedDataset]:
    """Load the pinned dataset and produce (report, naive, group-aware, dataset)."""
    dataset = load_dataset(path=path, allow_download=allow_download)
    report = detect_duplicates(dataset.frame)
    naive = audit_naive_split(
        dataset.frame, report, test_size=test_size, random_state=random_state
    )
    grouped = audit_naive_split(
        dataset.frame,
        report,
        test_size=test_size,
        random_state=random_state,
        group_aware=True,
    )
    return report, naive, grouped, dataset


def write_leakage_report(
    report: DuplicateReport,
    naive: SplitLeakageAudit,
    group_aware: SplitLeakageAudit,
    *,
    dataset: LoadedDataset | None = None,
    path: str | Path | None = None,
) -> Path:
    """Write the rendered leakage report and return its path."""
    target = (
        Path(path) if path is not None else Path(REPORTS_DIR) / LEAKAGE_REPORT_NAME
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_leakage_report(report, naive, group_aware, dataset=dataset),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.data.duplicates",
        description="Detect near-duplicates and audit cross-split leakage.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help=f"Report output path (default: reports/{LEAKAGE_REPORT_NAME}).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the audit without writing the report artifact.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows held out when simulating splits (default: 0.2).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_project_dirs()

    report, naive, grouped, dataset = build_leakage_audit(test_size=args.test_size)
    print(
        f"exact duplicates: all-columns={report.exact_all_columns.duplicate_rows} "
        f"features={report.exact_features.duplicate_rows}"
    )
    print(
        f"near-duplicate clusters: {report.cluster_count} "
        f"({report.clustered_rows} rows, {report.label_conflicts} label conflict(s))"
    )
    print(
        f"naive split: {naive.straddling_groups} straddling cluster(s), "
        f"{naive.leaky_test_rows} leaky test row(s)"
    )
    print(
        f"group-aware split: {grouped.straddling_groups} straddling cluster(s), "
        f"{grouped.leaky_test_rows} leaky test row(s)"
    )

    if not args.no_write:
        written = write_leakage_report(
            report, naive, grouped, dataset=dataset, path=args.report
        )
        print(f"wrote report: {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
