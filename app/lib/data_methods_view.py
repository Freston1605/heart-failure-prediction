"""Data & Methods view contract for the dashboard's Data & Methods page
(M005/S02/T01).

This module owns *what the Data & Methods page shows* so the page itself
stays thin and testable, exactly mirroring the M002 Runs-page pattern
(:mod:`app.lib.runs_view` + ``app/pages/3_Runs.py``): every load attempt
returns a frozen value object carrying either the generated artifacts or a
*named* error kind plus a plain-language message — never an exception (MEM047).

D025 data source (frozen decision)
----------------------------------
The page must render exactly the artifacts the S01 report machinery
generated under ``reports/data-methods/`` — never a re-derivation. This
module therefore re-verifies the SHA-256 provenance manifest through the
S01 engine itself (``heart.reporting.data_methods.provenance.load_manifest``
+ ``verify_manifest``) and reads the stage JSONs written by
``heart.reporting.data_methods.report``. Nothing here recomputes a
statistic; if the artifacts are wrong, the page refuses to render instead
of showing possibly-tampered numbers. No new logging is performed either
(D011/D012 frozen) — this module is read-only over the artifact directory.

Never-raise boundary
--------------------
:func:`load_data_methods_view` never raises. A missing artifact directory,
a missing or unreadable ``manifest.json``/``meta.json``, invalid JSON in a
stage file, a missing expected stage file, a missing PNG figure, or any
provenance verification failure is mapped to a named ``error_kind``:

* ``artifacts-missing`` — the artifact directory, ``manifest.json`` or
  ``meta.json`` is absent or unreadable;
* ``artifacts-corrupt`` — invalid JSON, a missing expected stage file, a
  missing PNG, or a digest mismatch under the manifest (tamper signal);
* ``artifacts-unknown`` — any other unexpected failure.

Every message is plain language: what is missing plus the concrete next
action naming the documented regeneration command
``PYTHONPATH=src python -m heart.reporting.data_methods`` — the same hint
the report's technical appendix prints — mirroring
:mod:`app.lib.runs_view`'s ``make reproduce`` hint, so the page and the
report report the *same* empty state.

The ``DATA_METHODS_DIR`` environment variable is the testability seam
(mirroring ``RUNS_TRACKING_DIR`` in :mod:`app.lib.runs_view` and
``EXPLORE_DATA_PATH`` in :mod:`app.lib.plots`): AppTest render tests point
the page at a tmp artifact directory without touching the committed
``reports/data-methods/`` tree.

This module is plain Python — no ``streamlit`` imports — so all logic is
pytest-testable without a Streamlit runtime (MEM047).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from heart.reporting.data_methods import provenance
from heart.reporting.data_methods.provenance import (
    ProvenanceDigestMismatchError,
    ProvenanceManifestError,
)
from heart.reporting.data_methods.report import (
    AUDIT_FILENAME,
    DATAMETHODS_DIR,
    GATE_VERDICTS_FILENAME,
    PLAN_FILENAME,
    STAGE_ARTIFACTS,
)

__all__ = [
    "DATA_METHODS_DIR_ENV",
    "REGENERATION_COMMAND",
    "DataMethodsView",
    "existing_figure_paths",
    "load_data_methods_view",
]

#: Environment override for the artifact directory (testability seam).
DATA_METHODS_DIR_ENV: str = "DATA_METHODS_DIR"

#: The one documented regeneration command every error message names —
#: the same hint the report's technical appendix prints.
REGENERATION_COMMAND: str = "PYTHONPATH=src python -m heart.reporting.data_methods"

#: Named error kinds surfaced by this module's contract.
ERROR_KIND_ARTIFACTS_MISSING: str = "artifacts-missing"
ERROR_KIND_ARTIFACTS_CORRUPT: str = "artifacts-corrupt"
ERROR_KIND_ARTIFACTS_UNKNOWN: str = "artifacts-unknown"

#: The meta.json filename (STAGE_ARTIFACTS["meta"]).
_META_FILE: str = STAGE_ARTIFACTS["meta"]


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataMethodsView:
    """Result of one artifact-load attempt; never raises.

    Attributes:
        attempted_dir: The artifact directory this attempt resolved to.
        meta: ``meta.json`` payload (dataset provenance: row/column counts,
            declared zero-as-missing columns, seed, target column).
        stages: Stage payloads keyed as in ``report.STAGE_ARTIFACTS``
            (``stage1_hygiene`` … ``stage4_assumptions``), empty on failure.
        figure_paths: Paths of PNG figures that exist under the artifact
            directory (sorted; only existing files).
        manifest: The loaded ``manifest.json`` payload (``None`` on failure).
        verified_digests: The digest mapping returned by
            ``provenance.verify_manifest`` (``None`` unless clean).
        error_kind: Named error bucket (``None`` on success).
        error_message: Friendly, plain-language failure message.
        treatment_audit: ``treatment_audit.json`` payload (per-model audit:
            models/pattern/split/zero_policy) or ``None`` when the committed
            split was absent at generation time (part B degrades to n/a).
        treatment_plan: ``treatment_plan.json`` payload (declared candidates,
            imbalance ladder, gate text).
        gate_verdicts: ``treatment_gate_verdicts.json`` payload (MLflow-derived
            outcomes table) or ``None`` when the verdicts step has not run.
        ok: Derived — no error kind recorded and meta loaded.
    """

    attempted_dir: str
    meta: dict[str, Any] | None = field(default=None)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    figure_paths: tuple[Path, ...] = field(default_factory=tuple)
    manifest: dict[str, Any] | None = field(default=None)
    verified_digests: dict[str, str] | None = field(default=None)
    treatment_audit: dict[str, Any] | None = field(default=None)
    treatment_plan: dict[str, Any] | None = field(default=None)
    gate_verdicts: dict[str, Any] | None = field(default=None)
    error_kind: str | None = field(default=None)
    error_message: str | None = field(default=None)

    @property
    def ok(self) -> bool:
        """True when the artifacts loaded and verified with no error."""
        return self.error_kind is None and self.meta is not None


# ---------------------------------------------------------------------------
# Resolution + message mapping
# ---------------------------------------------------------------------------


def _resolve_artifact_dir(artifact_dir: str | Path | None) -> Path:
    """Resolve the artifact directory: explicit arg, env override, default.

    The ``DATA_METHODS_DIR`` environment variable exists so AppTest render
    tests can point the page at a missing/corrupt/populated tmp artifact
    directory without touching the committed ``reports/data-methods/`` tree
    — the same seam ``RUNS_TRACKING_DIR`` provides for the Runs page.
    """
    if artifact_dir is not None:
        return Path(artifact_dir)
    override = os.environ.get(DATA_METHODS_DIR_ENV)
    if override:
        return Path(override)
    return DATAMETHODS_DIR


def _regen_hint() -> str:
    return (
        f"Run `{REGENERATION_COMMAND}` to regenerate the artifacts, "
        "then reload this page."
    )


def _artifacts_missing_message(directory: Path, what: str) -> str:
    return (
        f"{what} `{directory}` does not exist yet, so the Data & Methods "
        "report has nothing to show. " + _regen_hint()
    )


def _artifacts_unreadable_message(directory: Path, what: str, detail: str) -> str:
    return (
        f"{what} `{directory}` could not be read ({detail}), so the Data & "
        "Methods report is unavailable. " + _regen_hint()
    )


def _artifacts_corrupt_message(what: str, detail: str) -> str:
    return (
        f"{what} {detail} The Data & Methods report refuses to display "
        "possibly-tampered or stale figures. " + _regen_hint()
    )


def _artifacts_unknown_message(detail: str) -> str:
    return (
        "An unexpected problem occurred while reading the generated Data & "
        "Methods artifacts. The dashboard refuses to display possibly-wrong "
        f"figures. Detail: {detail}. " + _regen_hint()
    )


# ---------------------------------------------------------------------------
# Figure helper
# ---------------------------------------------------------------------------


def existing_figure_paths(
    directory: str | Path,
    names: "list[str] | tuple[str, ...] | None" = None,
) -> tuple[Path, ...]:
    """Return PNG figure paths that exist under ``directory``.

    With ``names`` given (e.g. the figure filenames referenced by the stage
    payloads), only those names that exist on disk are returned — a missing
    PNG never leaks a broken path to the page. Without ``names``, every
    ``*.png`` file under the directory is enumerated (sorted). Never raises.
    """
    base = Path(directory)
    try:
        if names is None:
            return tuple(
                sorted(path for path in base.glob("*.png") if path.is_file())
            )
        paths = []
        for name in sorted(names):
            candidate = base / name
            if candidate.is_file():
                paths.append(candidate)
        return tuple(paths)
    except OSError:
        return ()


# ---------------------------------------------------------------------------
# Contract function
# ---------------------------------------------------------------------------


def _load_attempt(artifact_dir: Path) -> DataMethodsView:
    """One artifact-load attempt; maps every exception to the contract.

    The provenance engine raises named errors (``ProvenanceManifestError``,
    ``ProvenanceDigestMismatchError``); this boundary swallows everything
    (BLE001) because the dashboard boundary must never leak a stack trace —
    any unpredicted failure lands in ``artifacts-unknown`` rather than
    white-screening the page.
    """
    display = str(artifact_dir)

    if not artifact_dir.exists():
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_MISSING,
            error_message=_artifacts_missing_message(artifact_dir, "The artifact directory"),
        )
    if not artifact_dir.is_dir():
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_MISSING,
            error_message=_artifacts_unreadable_message(
                artifact_dir, "The artifact directory", "not a directory"
            ),
        )

    manifest_path = artifact_dir / provenance.MANIFEST_NAME
    if not manifest_path.is_file():
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_MISSING,
            error_message=_artifacts_missing_message(
                artifact_dir / provenance.MANIFEST_NAME, "The provenance manifest"
            ),
        )

    meta_path = artifact_dir / _META_FILE
    if not meta_path.is_file():
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_MISSING,
            error_message=_artifacts_missing_message(meta_path, "The dataset metadata file"),
        )

    # --- provenance re-verification through the S01 engine itself ----------
    # (after the explicit existence checks: dir / manifest.json / meta.json
    # are pinned to ``artifacts-missing`` by the contract, while verification
    # failures map to ``artifacts-corrupt``)
    try:
        verified = provenance.verify_manifest(artifact_dir)
        manifest = provenance.load_manifest(artifact_dir)
    except ProvenanceDigestMismatchError as exc:
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
            error_message=_artifacts_corrupt_message(
                "The artifact directory failed its SHA-256 provenance check "
                f"against `{provenance.MANIFEST_NAME}`:",
                f"{str(exc)[:300]}",
            ),
        )
    except ProvenanceManifestError as exc:
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
            error_message=_artifacts_corrupt_message(
                "The provenance manifest could not be verified:",
                f"{str(exc)[:300]}",
            ),
        )

    # --- stage JSONs (per report.STAGE_ARTIFACTS) ---------------------------
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("metadata is not a JSON object")
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
            error_message=_artifacts_corrupt_message(
                f"The dataset metadata file `{_META_FILE}` is unreadable:",
                f"{type(exc).__name__}: {str(exc)[:200]}",
            ),
        )

    stages: dict[str, dict[str, Any]] = {}
    for stage_key, filename in STAGE_ARTIFACTS.items():
        if stage_key == "meta":
            continue
        stage_path = artifact_dir / filename
        if not stage_path.is_file():
            return DataMethodsView(
                attempted_dir=display,
                error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
                error_message=_artifacts_corrupt_message(
                    f"The expected stage artifact `{filename}` is missing from the "
                    "artifact directory.",
                    "",
                ),
            )
        try:
            payload = json.loads(stage_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("stage payload is not a JSON object")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            return DataMethodsView(
                attempted_dir=display,
                error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
                error_message=_artifacts_corrupt_message(
                    f"The stage artifact `{filename}` is unreadable:",
                    f"{type(exc).__name__}: {str(exc)[:200]}",
                ),
            )
        stages[stage_key] = payload

    # --- figures referenced by the stages must exist ------------------------
    referenced = sorted(
        {
            name
            for payload in stages.values()
            for name in payload.get("figures", [])
            if isinstance(name, str)
        }
    )
    missing_figures = [name for name in referenced if not (artifact_dir / name).is_file()]
    if missing_figures:
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
            error_message=_artifacts_corrupt_message(
                "Expected figure artifact(s) "
                + ", ".join(f"`{name}`" for name in missing_figures[:5])
                + " are missing from the artifact directory.",
                "",
            ),
        )

    # --- treatment artifacts (parts B-F; M006/S04/T01) ----------------------
    # ``treatment_plan.json`` is always written by the report generator, so a
    # verified directory without it is stale/corrupt (fail closed). The audit
    # (absent when the committed split is missing — part B degrades to n/a)
    # and the MLflow-derived gate verdicts (written by a separate step) are
    # optional: absent files yield ``None`` so the page can degrade, while a
    # present-but-unreadable file is corrupt. Tampering with any listed file
    # already failed closed above in ``verify_manifest`` (manifest-covered).
    try:
        treatment_plan = json.loads((artifact_dir / PLAN_FILENAME).read_text(encoding="utf-8"))
        if not isinstance(treatment_plan, dict):
            raise ValueError("treatment plan payload is not a JSON object")
    except FileNotFoundError:
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
            error_message=_artifacts_corrupt_message(
                f"The expected treatment artifact `{PLAN_FILENAME}` is missing from the "
                "artifact directory.",
                "",
            ),
        )
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return DataMethodsView(
            attempted_dir=display,
            error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
            error_message=_artifacts_corrupt_message(
                f"The treatment artifact `{PLAN_FILENAME}` is unreadable:",
                f"{type(exc).__name__}: {str(exc)[:200]}",
            ),
        )

    treatment_audit: dict[str, Any] | None = None
    audit_path = artifact_dir / AUDIT_FILENAME
    if audit_path.is_file():
        try:
            treatment_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(treatment_audit, dict):
                raise ValueError("treatment audit payload is not a JSON object")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            return DataMethodsView(
                attempted_dir=display,
                error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
                error_message=_artifacts_corrupt_message(
                    f"The treatment artifact `{AUDIT_FILENAME}` is unreadable:",
                    f"{type(exc).__name__}: {str(exc)[:200]}",
                ),
            )

    gate_verdicts: dict[str, Any] | None = None
    verdicts_path = artifact_dir / GATE_VERDICTS_FILENAME
    if verdicts_path.is_file():
        try:
            gate_verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
            if not isinstance(gate_verdicts, dict):
                raise ValueError("gate verdicts payload is not a JSON object")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            return DataMethodsView(
                attempted_dir=display,
                error_kind=ERROR_KIND_ARTIFACTS_CORRUPT,
                error_message=_artifacts_corrupt_message(
                    f"The treatment artifact `{GATE_VERDICTS_FILENAME}` is unreadable:",
                    f"{type(exc).__name__}: {str(exc)[:200]}",
                ),
            )

    return DataMethodsView(
        attempted_dir=display,
        meta=meta,
        stages=stages,
        treatment_audit=treatment_audit,
        treatment_plan=treatment_plan,
        gate_verdicts=gate_verdicts,
        figure_paths=existing_figure_paths(artifact_dir),
        manifest=manifest,
        verified_digests=verified,
    )


def load_data_methods_view(
    *, artifact_dir: str | Path | None = None
) -> DataMethodsView:
    """Load the generated Data & Methods artifacts, or the friendly error.

    Never raises. The artifact directory is resolved from the explicit
    ``artifact_dir`` argument, else the ``DATA_METHODS_DIR`` environment
    override, else the project default
    (``heart.reporting.data_methods.report.DATAMETHODS_DIR``).

    On success the returned view carries the ``meta.json`` payload, the four
    stage payloads, the treatment payloads (audit/plan/verdicts, with audit
    and verdicts ``None`` when those generation steps did not run), the
    existing PNG figure paths, the loaded manifest and
    the verified digest mapping produced by the S01 engine's
    ``verify_manifest``. On failure ``ok`` is ``False`` with a named
    ``error_kind`` and a plain-language message naming the documented
    regeneration command.
    """
    try:
        return _load_attempt(_resolve_artifact_dir(artifact_dir))
    except Exception as exc:  # noqa: BLE001 - even resolution must not raise
        directory = _resolve_artifact_dir(artifact_dir)
        return DataMethodsView(
            attempted_dir=str(directory),
            error_kind=ERROR_KIND_ARTIFACTS_UNKNOWN,
            error_message=_artifacts_unknown_message(
                f"{type(exc).__name__}: {str(exc)[:200]}"
            ),
        )
