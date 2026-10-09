"""Reproducible loader for the fedesoriano heart-failure dataset.

The loader has one job: turn a pinned source file into a validated
:class:`pandas.DataFrame` and *prove* it is the expected bytes.

Provenance strategy
-------------------
A byte-exact copy of ``heart.csv`` is committed under ``data/raw/`` and its
SHA-256 digest is pinned here. Because the same digest is reachable from
several independent mirrors (Hugging Face copies of the fedesoriano Kaggle
release), the pinned value is a publishable provenance record: if the file on
disk or a fresh download does not hash to :data:`PINNED_SHA256`, the loader
fails closed with :class:`IntegrityChecksumError` rather than feeding a
mutated dataset into the benchmark.

The committed copy is the primary, offline-safe path. If it is absent and
``allow_download=True`` (the default), the loader fetches from
:data:`PINNED_SOURCE_URL`, verifies the digest, and caches it locally.

Observability
-------------
:meth:`LoadedDataset.summary` / :meth:`LoadedDataset.describe` report the row
count, per-column dtypes, and the integrity checksum, and
``python -m heart.data.load`` prints them for a human.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from heart.config import RAW_DATA_DIR, ensure_project_dirs
from heart.data.schema import (
    EXPECTED_ROW_COUNT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Pinned provenance
# ---------------------------------------------------------------------------

#: Filename used for the committed / cached raw dataset.
RAW_FILENAME: str = "heart.csv"

#: Pinned fetch source. Several independent Hugging Face mirrors of the
#: fedesoriano Kaggle release are byte-identical; this one is the canonical
#: URL recorded here. If it ever 404s, any mirror of the release works, but
#: the digest check still has to pass.
PINNED_SOURCE_URL: str = (
    "https://huggingface.co/datasets/imkrish/heart-failure-dataset/resolve/main/heart.csv"
)

#: SHA-256 of the canonical 918-row ``heart.csv``. Verified identical across
#: three independent mirrors of the fedesoriano release.
PINNED_SHA256: str = "948420b084d8a3a0ca42b8419fce9aee175879e43f8aedf712377899a67aa49b"

#: Default network timeout (seconds) for the optional download path.
DEFAULT_TIMEOUT: float = 30.0


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class LoadError(Exception):
    """Base class for dataset loading/integrity failures."""


class SourceNotFoundError(LoadError):
    """The raw dataset file is absent and downloading is not allowed."""


class IntegrityChecksumError(LoadError):
    """The file's SHA-256 digest does not match the pinned value."""


class DownloadError(LoadError):
    """The pinned source could not be fetched (network/HTTP/timeout)."""


# ---------------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of ``path`` (streamed, not slurped)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def raw_data_path() -> Path:
    """Return the on-disk location of the pinned raw dataset."""
    return Path(RAW_DATA_DIR) / RAW_FILENAME


# ---------------------------------------------------------------------------
# Loaded dataset value object
# ---------------------------------------------------------------------------


@dataclass
class LoadedDataset:
    """A validated dataset plus its provenance metadata."""

    frame: pd.DataFrame
    source_path: Path
    sha256: str
    source_url: str | None = None
    downloaded: bool = False

    @property
    def row_count(self) -> int:
        return int(len(self.frame))

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.frame.columns)

    @property
    def dtypes(self) -> dict[str, str]:
        return {str(c): str(self.frame[c].dtype) for c in self.frame.columns}

    def summary(self) -> dict[str, object]:
        """Machine-readable report: rows, columns, dtypes, integrity digest."""
        return {
            "source_path": str(self.source_path),
            "source_url": self.source_url,
            "downloaded": self.downloaded,
            "row_count": self.row_count,
            "expected_row_count": EXPECTED_ROW_COUNT,
            "column_count": len(self.columns),
            "columns": list(self.columns),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "dtypes": self.dtypes,
            "sha256": self.sha256,
            "sha256_verified": self.sha256 == PINNED_SHA256,
        }

    def describe(self) -> str:
        """Human-readable report of row count, dtypes, and integrity checksum."""
        summary = self.summary()
        lines = [
            f"source: {summary['source_path']}"
            + (" (downloaded)" if self.downloaded else ""),
            f"sha256: {summary['sha256']}",
            f"sha256 matches pinned: {summary['sha256_verified']}",
            f"rows: {summary['row_count']} (expected {summary['expected_row_count']})",
            f"columns: {summary['column_count']} "
            f"({len(FEATURE_COLUMNS)} features + target {TARGET_COLUMN!r})",
            "dtypes:",
        ]
        width = max((len(c) for c in self.columns), default=0)
        for column in self.columns:
            lines.append(f"  {column:<{width}}  {self.dtypes[column]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def download_raw(
    dest: str | Path | None = None,
    *,
    url: str = PINNED_SOURCE_URL,
    expected_sha256: str = PINNED_SHA256,
    timeout: float = DEFAULT_TIMEOUT,
    force: bool = False,
) -> Path:
    """Fetch the pinned dataset into ``dest`` and verify its digest.

    The download is written to a temporary sibling and atomically renamed, so
    an interrupted transfer never leaves a truncated file masquerading as the
    dataset. A digest mismatch removes the bad artifact before raising.
    """
    target = Path(dest) if dest is not None else raw_data_path()
    if target.exists() and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise DownloadError(
            f"Failed to download the pinned dataset from {url!r}: HTTP {exc.code} "
            f"{exc.reason}. Retry later, or place a copy at {target} manually."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:  # pragma: no cover
        raise DownloadError(
            f"Failed to download the pinned dataset from {url!r}: {exc}. "
            f"Check network access, or place a copy at {target} manually."
        ) from exc

    actual = sha256_bytes(payload)
    if expected_sha256 is not None and actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise IntegrityChecksumError(
            f"Downloaded dataset digest {actual} does not match the pinned "
            f"digest {expected_sha256} from {url!r}. Refusing to cache a mutated "
            "dataset; the source may have changed."
        )

    tmp.write_bytes(payload)
    os.replace(tmp, target)
    return target


def load_raw(
    path: str | Path | None = None,
    *,
    source_url: str = PINNED_SOURCE_URL,
    expected_sha256: str | None = PINNED_SHA256,
    allow_download: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[pd.DataFrame, str, bool]:
    """Read the raw CSV, verifying the integrity checksum first.

    Returns ``(frame, sha256, downloaded)``. The checksum is computed **before**
    parsing so a corrupted file is rejected without pandas' help.
    """
    target = Path(path) if path is not None else raw_data_path()
    downloaded = False

    if not target.exists():
        if not allow_download:
            raise SourceNotFoundError(
                f"Raw dataset not found at {target}. Run "
                "`python -m heart.data.load --download` (or pass allow_download=True) "
                "to fetch the pinned copy, or commit the file to data/raw/heart.csv."
            )
        target = download_raw(
            target, url=source_url, expected_sha256=expected_sha256, timeout=timeout
        )
        downloaded = True

    digest = sha256_file(target)
    if expected_sha256 is not None and digest != expected_sha256:
        raise IntegrityChecksumError(
            f"Dataset at {target} has digest {digest}, expected {expected_sha256}. "
            "The file was modified or replaced; restore the pinned copy "
            "(or re-download with force=True) before proceeding."
        )

    frame = pd.read_csv(target)
    return frame, digest, downloaded


def load_dataset(
    *,
    path: str | Path | None = None,
    allow_download: bool = True,
    verify_checksum: bool = True,
    validate: bool = True,
    expected_rows: int | None = EXPECTED_ROW_COUNT,
) -> LoadedDataset:
    """Load, integrity-check, and schema-validate the pinned dataset.

    Parameters
    ----------
    path:
        Override the raw CSV location. Defaults to ``data/raw/heart.csv``.
    allow_download:
        When the file is absent, fetch it from the pinned source instead of
        raising :class:`SourceNotFoundError`.
    verify_checksum:
        Enforce the pinned SHA-256 digest (default). Disable only for
        deliberately corrupted fixtures in tests.
    validate:
        Run :func:`heart.data.schema.validate_schema` (default).
    expected_rows:
        Row count enforced by schema validation; ``None`` skips the check.
    """
    expected_sha256 = PINNED_SHA256 if verify_checksum else None
    frame, digest, downloaded = load_raw(
        path, allow_download=allow_download, expected_sha256=expected_sha256
    )

    if validate:
        validate_schema(frame, expected_rows=expected_rows)

    target = Path(path) if path is not None else raw_data_path()
    return LoadedDataset(
        frame=frame,
        source_path=target,
        sha256=digest,
        source_url=PINNED_SOURCE_URL if downloaded else None,
        downloaded=downloaded,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.data.load",
        description="Load and integrity-check the pinned heart-failure dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the pinned dataset if data/raw/heart.csv is missing.",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip the pinned SHA-256 check (diagnostics only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_project_dirs()
    dataset = load_dataset(
        allow_download=args.download,
        verify_checksum=not args.no_checksum,
    )
    print(dataset.describe())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
