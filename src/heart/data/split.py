"""Versioned, leakage-safe train/test split creation.

The split is the first data-dependent decision in the pipeline, so it must be
made **before** any fitted transform and it must respect the near-duplicate
structure discovered by :mod:`heart.data.duplicates`. This module produces one
immutable, versioned artifact set:

```
data/processed/splits/v1/
    train.csv       # training rows, original raw values (no transform applied)
    test.csv        # held-out rows, original raw values
    manifest.json   # seed, sizes, class balance, groups, per-file sha256, indices
```

Design guarantees
-----------------
* **Group-aware.** Rows are grouped by
  :func:`heart.data.duplicates.assign_duplicate_groups`, so a near-duplicate
  cluster can never straddle train and test. :func:`create_split` fails closed
  (:class:`SplitLeakageError`) rather than emit a straddling split.
* **Stratified.** :class:`~sklearn.model_selection.StratifiedGroupKFold`
  preserves the ``HeartDisease`` prevalence on both sides while keeping groups
  whole.
* **Raw values only.** The CSVs hold the values exactly as loaded. Every
  imputation/scaling decision belongs to :mod:`heart.data.pipeline`, which is
  fit on the training rows of this split.
* **Reproducible.** Everything is a pure function of the frame, the seed, and
  the declared fold; two runs produce byte-identical CSVs and manifest.
* **Atomic.** Files are written to ``*.part`` and renamed, so a crashed run
  never leaves a truncated artifact that looks valid.

Observability
-------------
:func:`create_split` logs the seed, fold, row counts, and confirmed zero
straddle count at ``INFO``; :class:`DataSplit` and :class:`SplitArtifacts`
serialise via ``to_dict``; ``python -m heart.data.split`` prints the split and
writes the artifacts. :func:`load_split_artifacts` re-verifies each file's
sha256 against the manifest and raises :class:`SplitIntegrityError` on drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from heart.config import PROCESSED_DATA_DIR, RANDOM_SEED, ensure_project_dirs
from heart.data.duplicates import assign_duplicate_groups
from heart.data.load import LoadedDataset, load_dataset, sha256_file
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared split configuration
# ---------------------------------------------------------------------------

#: Current artifact schema/version. Bump to invalidate every downstream split.
SPLIT_VERSION: str = "v1"

#: Sub-directory under ``data/processed`` that holds versioned split folders.
SPLITS_DIRNAME: str = "splits"

#: Artifact filenames inside a version directory.
TRAIN_FILENAME: str = "train.csv"
TEST_FILENAME: str = "test.csv"
MANIFEST_FILENAME: str = "manifest.json"

#: Number of stratified folds. ``n_splits=5`` holds out ~20% of the rows.
N_SPLITS: int = 5

#: Which fold becomes the test set. Fixed so the split is deterministic.
TEST_FOLD: int = 0


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class SplitError(Exception):
    """Base class for split-creation failures."""


class MissingTargetError(SplitError):
    """The frame has no target column, so no stratified split is possible."""


class SplitConfigError(SplitError):
    """The requested split configuration is invalid."""


class SplitLeakageError(SplitError):
    """A duplicate group would (or did) straddle train and test."""


class SplitNotFoundError(SplitError):
    """A requested split version has no manifest on disk."""


class SplitIntegrityError(SplitError):
    """A committed split artifact does not match its manifest digest."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSplit:
    """A deterministic partition of a frame into train/test row indices."""

    version: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    random_state: int
    n_splits: int
    test_fold: int
    row_count: int
    group_count: int
    straddling_groups: int
    source_sha256: str | None = None

    @property
    def train_rows(self) -> int:
        return int(self.train_indices.size)

    @property
    def test_rows(self) -> int:
        return int(self.test_indices.size)

    @property
    def test_fraction(self) -> float:
        return self.test_rows / self.row_count if self.row_count else 0.0

    def train_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.iloc[self.train_indices].reset_index(drop=True)

    def test_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.iloc[self.test_indices].reset_index(drop=True)

    def class_balance(self, frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Per-side ``HeartDisease`` counts, keyed by ``train``/``test``/``overall``."""

        def counts(subset: pd.DataFrame) -> dict[str, int]:
            return {
                str(int(k)): int(v)
                for k, v in sorted(subset[TARGET_COLUMN].value_counts().items())
            }

        return {
            "overall": counts(frame),
            "train": counts(self.train_frame(frame)),
            "test": counts(self.test_frame(frame)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "random_state": self.random_state,
            "n_splits": self.n_splits,
            "test_fold": self.test_fold,
            "row_count": self.row_count,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "test_fraction": self.test_fraction,
            "group_count": self.group_count,
            "straddling_groups": self.straddling_groups,
            "source_sha256": self.source_sha256,
            "train_indices": [int(i) for i in self.train_indices],
            "test_indices": [int(i) for i in self.test_indices],
        }


@dataclass(frozen=True)
class SplitArtifacts:
    """Paths and manifest of a written split version."""

    version: str
    directory: Path
    train_path: Path
    test_path: Path
    manifest_path: Path
    manifest: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "directory": str(self.directory),
            "train_path": str(self.train_path),
            "test_path": str(self.test_path),
            "manifest_path": str(self.manifest_path),
            "manifest": self.manifest,
        }


# ---------------------------------------------------------------------------
# Group handling
# ---------------------------------------------------------------------------


def group_straddle_count(
    labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray
) -> int:
    """Number of group labels appearing on both sides of the split."""
    train_labels = {int(value) for value in labels[train_indices]}
    test_labels = {int(value) for value in labels[test_indices]}
    return len(train_labels & test_labels)


def assert_no_group_straddle(
    labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray
) -> None:
    """Raise :class:`SplitLeakageError` if any group spans train and test."""
    straddling = group_straddle_count(labels, train_indices, test_indices)
    if straddling:
        raise SplitLeakageError(
            f"{straddling} duplicate group(s) straddle the train/test boundary. "
            "Near-identical rows on both sides inflate evaluation scores; use "
            "group-aware splitting keyed on "
            "heart.data.duplicates.assign_duplicate_groups."
        )


def _validate_config(n_splits: int, test_fold: int) -> None:
    if n_splits < 2:
        raise SplitConfigError(f"n_splits must be >= 2, got {n_splits!r}.")
    if not 0 <= test_fold < n_splits:
        raise SplitConfigError(
            f"test_fold must be in [0, {n_splits}), got {test_fold!r}."
        )


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------


def create_split(
    frame: pd.DataFrame,
    *,
    version: str = SPLIT_VERSION,
    random_state: int = RANDOM_SEED,
    n_splits: int = N_SPLITS,
    test_fold: int = TEST_FOLD,
    groups: np.ndarray | None = None,
    source_sha256: str | None = None,
) -> DataSplit:
    """Build a stratified, group-aware train/test split of ``frame``.

    ``groups`` defaults to the near-duplicate group labels from
    :func:`heart.data.duplicates.assign_duplicate_groups`, which keeps every
    cluster whole. The returned :class:`DataSplit` carries sorted indices so
    the artifact is stable run to run.
    """
    if not isinstance(frame, pd.DataFrame):
        raise SplitError(f"Expected a pandas.DataFrame, got {type(frame).__name__}.")
    if TARGET_COLUMN not in frame.columns:
        raise MissingTargetError(
            f"Frame has no {TARGET_COLUMN!r} column; a stratified split needs the "
            "target. Load the dataset through heart.data.load.load_dataset."
        )
    _validate_config(n_splits, test_fold)

    labels = np.asarray(groups) if groups is not None else assign_duplicate_groups(frame)
    if labels.ndim != 1 or labels.size != len(frame):
        raise SplitConfigError(
            f"groups must be one label per row ({len(frame)}), got shape "
            f"{labels.shape}."
        )

    feature_matrix = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    folds = list(splitter.split(feature_matrix, target, labels))
    train_indices, test_indices = folds[test_fold]
    train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
    test_indices = np.sort(np.asarray(test_indices, dtype=np.int64))

    # Fail closed: an emitted split must never straddle a duplicate group.
    assert_no_group_straddle(labels, train_indices, test_indices)

    split = DataSplit(
        version=version,
        train_indices=train_indices,
        test_indices=test_indices,
        random_state=random_state,
        n_splits=n_splits,
        test_fold=test_fold,
        row_count=int(len(frame)),
        group_count=int(np.unique(labels).size),
        straddling_groups=0,
        source_sha256=source_sha256,
    )
    logger.info(
        "Created %s split (seed=%d, n_splits=%d, test_fold=%d): train=%d, "
        "test=%d (%.2f%% held out), groups=%d, straddling groups=0",
        version,
        random_state,
        n_splits,
        test_fold,
        split.train_rows,
        split.test_rows,
        split.test_fraction * 100,
        split.group_count,
    )
    return split


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _split_root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else Path(PROCESSED_DATA_DIR) / SPLITS_DIRNAME


def _build_manifest(
    split: DataSplit, frame: pd.DataFrame, directory: Path
) -> dict[str, object]:
    train_path = directory / TRAIN_FILENAME
    test_path = directory / TEST_FILENAME
    manifest: dict[str, object] = {
        "version": split.version,
        "created_by": "heart.data.split",
        "random_seed": split.random_state,
        "n_splits": split.n_splits,
        "test_fold": split.test_fold,
        "row_count": split.row_count,
        "train_rows": split.train_rows,
        "test_rows": split.test_rows,
        "test_fraction": split.test_fraction,
        "source_sha256": split.source_sha256,
        "target_column": TARGET_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "group_count": split.group_count,
        "group_straddling": split.straddling_groups,
        "class_balance": split.class_balance(frame),
        "files": {
            "train": {
                "name": TRAIN_FILENAME,
                "sha256": sha256_file(train_path),
                "rows": split.train_rows,
            },
            "test": {
                "name": TEST_FILENAME,
                "sha256": sha256_file(test_path),
                "rows": split.test_rows,
            },
        },
        "train_indices": [int(i) for i in split.train_indices],
        "test_indices": [int(i) for i in split.test_indices],
    }
    return manifest


def write_split(
    split: DataSplit, frame: pd.DataFrame, *, root: str | Path | None = None
) -> SplitArtifacts:
    """Write ``train.csv``, ``test.csv`` and ``manifest.json`` for ``split``."""
    directory = _split_root(root) / split.version
    directory.mkdir(parents=True, exist_ok=True)

    train_path = directory / TRAIN_FILENAME
    test_path = directory / TEST_FILENAME
    train_csv = split.train_frame(frame).to_csv(index=False).encode("utf-8")
    test_csv = split.test_frame(frame).to_csv(index=False).encode("utf-8")
    _atomic_write_bytes(train_path, train_csv)
    _atomic_write_bytes(test_path, test_csv)

    manifest = _build_manifest(split, frame, directory)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path = directory / MANIFEST_FILENAME
    _atomic_write_bytes(manifest_path, manifest_bytes)

    artifacts = SplitArtifacts(
        version=split.version,
        directory=directory,
        train_path=train_path,
        test_path=test_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    logger.info(
        "Wrote split artifacts to %s (train sha256=%s, test sha256=%s)",
        directory,
        manifest["files"]["train"]["sha256"],  # type: ignore[index]
        manifest["files"]["test"]["sha256"],  # type: ignore[index]
    )
    return artifacts


def load_split_artifacts(
    version: str = SPLIT_VERSION, *, root: str | Path | None = None
) -> SplitArtifacts:
    """Load a written split, verifying every file against the manifest."""
    directory = _split_root(root) / version
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise SplitNotFoundError(
            f"No split manifest at {manifest_path}. Run "
            "`python -m heart.data.split` to create the versioned split."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    for key, filename in (("train", TRAIN_FILENAME), ("test", TEST_FILENAME)):
        path = directory / filename
        if not path.exists():
            raise SplitNotFoundError(f"Split artifact missing: {path}.")
        expected = files.get(key, {}).get("sha256")
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise SplitIntegrityError(
                f"{path} has sha256 {actual}, but the manifest records {expected}. "
                "The artifact was modified; regenerate the split rather than "
                "trusting it."
            )
    return SplitArtifacts(
        version=version,
        directory=directory,
        train_path=directory / TRAIN_FILENAME,
        test_path=directory / TEST_FILENAME,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_split_frames(
    version: str = SPLIT_VERSION, *, root: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train/test frames of a written split version."""
    artifacts = load_split_artifacts(version, root=root)
    train = pd.read_csv(artifacts.train_path)
    test = pd.read_csv(artifacts.test_path)
    return train, test


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_split(
    *,
    path: str | Path | None = None,
    allow_download: bool = True,
    version: str = SPLIT_VERSION,
    random_state: int = RANDOM_SEED,
    n_splits: int = N_SPLITS,
    test_fold: int = TEST_FOLD,
) -> tuple[DataSplit, LoadedDataset]:
    """Load the pinned dataset and build its declared split."""
    dataset = load_dataset(path=path, allow_download=allow_download)
    split = create_split(
        dataset.frame,
        version=version,
        random_state=random_state,
        n_splits=n_splits,
        test_fold=test_fold,
        source_sha256=dataset.sha256,
    )
    return split, dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.data.split",
        description="Create the versioned, group-aware train/test split.",
    )
    parser.add_argument(
        "--version", default=SPLIT_VERSION, help="Artifact version label."
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the split without writing artifacts.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--n-splits", type=int, default=N_SPLITS, help="Number of stratified folds."
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory that holds version folders (default: data/processed/splits).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_project_dirs()

    split, dataset = build_split(
        version=args.version, random_state=args.seed, n_splits=args.n_splits
    )
    balance = split.class_balance(dataset.frame)
    print(
        f"dataset shape: {dataset.frame.shape} "
        f"(rows={split.row_count}, seed={split.random_state}, "
        f"n_splits={split.n_splits}, test_fold={split.test_fold})"
    )
    print(
        f"train rows: {split.train_rows} | test rows: {split.test_rows} "
        f"({split.test_fraction * 100:.2f}% held out)"
    )
    print(
        f"class balance: overall={balance['overall']} "
        f"train={balance['train']} test={balance['test']}"
    )
    print(
        f"groups: {split.group_count} | duplicate groups straddling train/test: "
        f"{split.straddling_groups}"
    )

    if not args.no_write:
        artifacts = write_split(split, dataset.frame, root=args.output_root)
        print(f"wrote split artifacts: {artifacts.directory}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
