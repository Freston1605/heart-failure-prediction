"""SHA-256 provenance manifest for the pre-split EDA artifacts (M005/S01/T02).

Why this module exists
----------------------
M005's headline guarantee (decision D025) is that ``reports/data-methods.md``
part A is **never hand-assembled**: every table and figure in the report must
trace to a generated artifact under ``reports/data-methods/``, and any
hand-edit to a number must fail a pytest provenance test. This module is the
lock that makes that enforceable:

1. :func:`write_manifest` scans an artifact directory and records every file's
   SHA-256 digest in ``manifest.json``.
2. :func:`verify_manifest` recomputes the digests and raises a *named*
   exception the moment an artifact is missing
   (:class:`ProvenanceManifestError`) or altered
   (:class:`ProvenanceDigestMismatchError`).
3. :func:`manifest_markdown` renders the manifest as the report's artifact
   index table, so the committed report itself carries the digests of the
   artifacts it cites.

Both exceptions hang off the :class:`DataMethodsError` family defined by
:mod:`heart.reporting.data_methods.eda`, so callers can catch the whole
data-methods surface with one base class.

Determinism
-----------
Manifest construction is deliberately deterministic: files are listed in
sorted order, digests are computed over raw bytes, and the JSON is serialised
with sorted keys and fixed indentation. Two runs over the same artifact bytes
produce byte-identical ``manifest.json`` — the property the S01/T02
regeneration-determinism test relies on.

Observability
-------------
Every manifest write and verification logs one ``INFO`` line (artifacts
covered, verification outcome). Verification failures raise named exceptions
with the offending artifact and both digests in the message, so a failed run
names the file that drifted without needing a debugger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from heart.reporting.data_methods.eda import DataMethodsError

logger = logging.getLogger(__name__)

#: Name of the manifest file written inside the artifact directory.
MANIFEST_NAME: str = "manifest.json"

#: Digest algorithm recorded in the manifest (future-proofing the schema).
MANIFEST_ALGORITHM: str = "sha256"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class ProvenanceManifestError(DataMethodsError):
    """Raised when the manifest is missing, malformed, or lists a lost artifact."""


class ProvenanceDigestMismatchError(ProvenanceManifestError):
    """Raised when an artifact's recomputed digest differs from the manifest.

    This is the tamper signal: a byte changed in any table or figure under the
    artifact directory surfaces as this exception, not as a silent pass.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(artifact_dir: Path) -> list[Path]:
    """List every artifact file under ``artifact_dir`` (manifest excluded)."""
    return sorted(
        (
            path
            for path in artifact_dir.rglob("*")
            if path.is_file() and path.name != MANIFEST_NAME
        ),
        key=lambda path: path.relative_to(artifact_dir).as_posix(),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialise ``payload`` deterministically and replace ``path`` atomically."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public seam
# ---------------------------------------------------------------------------


def build_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Build the manifest mapping (relative artifact name -> sha256 digest)."""
    directory = Path(artifact_dir)
    if not directory.is_dir():
        raise ProvenanceManifestError(
            f"artifact directory does not exist: {directory}"
        )
    files = _artifact_files(directory)
    if not files:
        raise ProvenanceManifestError(
            f"artifact directory contains no artifacts to manifest: {directory}"
        )
    artifacts = {
        path.relative_to(directory).as_posix(): _sha256_file(path) for path in files
    }
    return {
        "algorithm": MANIFEST_ALGORITHM,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Write ``manifest.json`` into ``artifact_dir`` and return the payload."""
    directory = Path(artifact_dir)
    manifest = build_manifest(directory)
    _atomic_write_json(directory / MANIFEST_NAME, manifest)
    logger.info(
        "provenance manifest: wrote %d artifact digest(s) to %s",
        manifest["artifact_count"],
        directory,
    )
    return manifest


def load_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Read and lightly validate ``manifest.json`` from ``artifact_dir``."""
    path = Path(artifact_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ProvenanceManifestError(f"provenance manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceManifestError(
            f"provenance manifest is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or "artifacts" not in manifest:
        raise ProvenanceManifestError(
            f"provenance manifest is malformed (missing 'artifacts' mapping): {path}"
        )
    return manifest


def verify_manifest(artifact_dir: str | Path) -> dict[str, str]:
    """Recompute every digest in the manifest and return the verified mapping.

    Raises
    ------
    ProvenanceManifestError
        When the manifest is missing/malformed or a listed artifact no longer
        exists.
    ProvenanceDigestMismatchError
        When an artifact's bytes no longer hash to the recorded digest —
        i.e. the artifact was tampered with or stale.
    """
    directory = Path(artifact_dir)
    manifest = load_manifest(directory)
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ProvenanceManifestError(
            f"provenance manifest lists no artifacts: {directory}"
        )
    verified: dict[str, str] = {}
    for name, recorded in sorted(artifacts.items()):
        path = directory / name
        if not path.is_file():
            raise ProvenanceManifestError(
                f"manifested artifact is missing: {path}"
            )
        actual = _sha256_file(path)
        if actual != recorded:
            raise ProvenanceDigestMismatchError(
                f"artifact digest mismatch: {path} "
                f"(manifest {recorded[:12]}…, actual {actual[:12]}…)"
            )
        verified[name] = actual
    logger.info(
        "provenance verification: %d artifact(s) verified clean against %s",
        len(verified),
        directory / MANIFEST_NAME,
    )
    return verified


def manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render the manifest as a markdown artifact-index table."""
    artifacts = manifest.get("artifacts", {})
    lines = [
        "| artifact | sha256 |",
        "|---|---|",
    ]
    for name in sorted(artifacts):
        lines.append(f"| `{name}` | `{artifacts[name]}` |")
    return "\n".join(lines)
