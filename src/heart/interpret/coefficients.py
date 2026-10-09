"""Coefficient and feature-importance artifacts (S04/T04).

What this module answers
------------------------
SHAP (see :mod:`heart.interpret.shap_export`) explains individual predictions.
This module reports the model's own *global* attribution: the signed
coefficients of a linear classifier, or the impurity/permutation importances of
a tree ensemble. Together they let a reviewer check the two views agree — an
engineered feature that SHAP says matters should not be a zero-weight column.

The extraction is deliberately estimator-agnostic:

* an estimator exposing ``coef_`` yields a **signed coefficient** table;
* an estimator exposing ``feature_importances_`` yields a **non-negative
  importance** table;
* anything else (for example Gaussian Naive Bayes, which stores ``theta_`` and
  ``var_``) raises :class:`UnsupportedImportanceError` rather than inventing a
  number.

Durable artifact
----------------
``reports/feature_importance.csv`` — one ranked row per transformed feature
with its value, absolute value, kind (``coefficient`` / ``importance``), and an
``engineered`` flag. The artifact round-trips through
:func:`read_importance`, so its dimensions can be asserted against the selected
feature count without retraining.

Observability
-------------
:func:`extract_importance` logs the estimator class, the attribution kind, and
the top features at ``INFO``. Failures are named subclasses of
:class:`CoefficientExportError`.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heart.config import REPORTS_DIR
from heart.features.selection import DEFAULT_LEDGER_PATH as DEFAULT_SELECTION_LEDGER_PATH
from heart.interpret.model import (
    InterpretError,
    atomic_write_text,
    build_interpretability_context,
    display_feature_name,
    matches_engineered,
    transformed_feature_names,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IMPORTANCE_FILENAME",
    "DEFAULT_IMPORTANCE_PATH",
    "IMPORTANCE_COLUMNS",
    "KIND_COEFFICIENT",
    "KIND_IMPORTANCE",
    "CoefficientExportError",
    "ImportanceConfigError",
    "ImportanceDataError",
    "UnsupportedImportanceError",
    "ImportanceArtifactError",
    "ImportanceReport",
    "extract_importance",
    "render_importance_csv",
    "write_importance",
    "read_importance",
    "build_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Durable global-attribution artifact.
IMPORTANCE_FILENAME: str = "feature_importance.csv"

#: Default artifact location.
DEFAULT_IMPORTANCE_PATH: Path = Path(REPORTS_DIR) / IMPORTANCE_FILENAME

#: Header of the importance CSV (order is part of the artifact contract).
IMPORTANCE_COLUMNS: tuple[str, ...] = (
    "rank",
    "feature",
    "raw_feature",
    "engineered",
    "kind",
    "value",
    "abs_value",
    "model",
)

#: Attribution kinds.
KIND_COEFFICIENT: str = "coefficient"
KIND_IMPORTANCE: str = "importance"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class CoefficientExportError(InterpretError):
    """Base class for every coefficient / importance failure."""


class ImportanceConfigError(CoefficientExportError):
    """An importance configuration value is invalid."""


class ImportanceDataError(CoefficientExportError):
    """The pipeline handed to the extractor is missing or malformed."""


class UnsupportedImportanceError(CoefficientExportError):
    """The estimator exposes neither ``coef_`` nor ``feature_importances_``."""


class ImportanceArtifactError(CoefficientExportError):
    """The importance artifact could not be serialised, written, or read."""


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportanceReport:
    """Global feature attribution for one fitted selected pipeline."""

    feature_names: tuple[str, ...]
    display_names: tuple[str, ...]
    engineered: tuple[bool, ...]
    values: tuple[float, ...]
    kind: str
    model_name: str
    estimator_class: str
    n_features: int

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        for field_name, seq in (
            ("display_names", self.display_names),
            ("engineered", self.engineered),
            ("values", self.values),
        ):
            if len(seq) != n:
                raise ImportanceDataError(
                    f"ImportanceReport.{field_name} has {len(seq)} entry/entries "
                    f"but {n} feature(s) were declared."
                )
        if self.n_features != n:
            raise ImportanceDataError(
                f"ImportanceReport.n_features={self.n_features} does not match "
                f"{n} feature name(s)."
            )
        if self.kind not in (KIND_COEFFICIENT, KIND_IMPORTANCE):
            raise ImportanceDataError(
                f"Unknown attribution kind {self.kind!r}; expected "
                f"{[KIND_COEFFICIENT, KIND_IMPORTANCE]}."
            )
        if self.kind == KIND_IMPORTANCE and any(value < 0 for value in self.values):
            raise ImportanceDataError(
                "importance values must be non-negative; got a negative value."
            )
        for value in self.values:
            if not np.isfinite(value):
                raise ImportanceDataError(
                    f"attribution values must be finite, got {value!r}."
                )

    @property
    def signed(self) -> bool:
        """``True`` for signed coefficients, ``False`` for importances."""
        return self.kind == KIND_COEFFICIENT

    def abs_values(self) -> tuple[float, ...]:
        return tuple(abs(value) for value in self.values)

    def _order(self) -> tuple[int, ...]:
        order = sorted(
            range(self.n_features),
            key=lambda index: (-abs(self.values[index]), index),
        )
        return tuple(order)

    def top(self, k: int) -> tuple[dict[str, object], ...]:
        """The ``k`` highest-magnitude features."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ImportanceConfigError(
                f"k must be a non-negative int, got {k!r}."
            )
        return tuple(self.to_rows()[:k])

    def to_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for rank, index in enumerate(self._order(), start=1):
            rows.append(
                {
                    "rank": rank,
                    "feature": self.display_names[index],
                    "raw_feature": self.feature_names[index],
                    "engineered": bool(self.engineered[index]),
                    "kind": self.kind,
                    "value": float(self.values[index]),
                    "abs_value": float(abs(self.values[index])),
                    "model": self.model_name,
                }
            )
        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "estimator_class": self.estimator_class,
            "kind": self.kind,
            "signed": self.signed,
            "n_features": int(self.n_features),
            "features": self.to_rows(),
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _coefficient_vector(estimator: object) -> np.ndarray | None:
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is None:
        return None
    array = np.asarray(coefficients, dtype=float)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise UnsupportedImportanceError(
                f"Estimator {type(estimator).__name__} has a "
                f"{array.shape[0]}-row coef_ matrix; this exporter reports "
                "binary classifiers only."
            )
        array = array[0]
    if array.ndim != 1:
        raise UnsupportedImportanceError(
            f"Estimator {type(estimator).__name__} has a coef_ of shape "
            f"{array.shape}; expected a single coefficient vector."
        )
    return array


def extract_importance(pipeline) -> ImportanceReport:
    """Extract the fitted classifier's global feature attribution.

    Prefers ``coef_`` (signed) and falls back to ``feature_importances_``
    (non-negative). The result has exactly one entry per transformed feature
    the classifier saw.
    """
    from sklearn.pipeline import Pipeline

    if not isinstance(pipeline, Pipeline) or "clf" not in pipeline.named_steps:
        raise ImportanceDataError(
            "pipeline must be a fitted sklearn Pipeline ending in a 'clf' step."
        )
    estimator = pipeline.named_steps["clf"]
    names = transformed_feature_names(pipeline)

    coefficients = _coefficient_vector(estimator)
    if coefficients is not None:
        values = coefficients
        kind = KIND_COEFFICIENT
    else:
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            raise UnsupportedImportanceError(
                f"Estimator {type(estimator).__name__} exposes neither coef_ "
                "nor feature_importances_; no global attribution can be "
                "exported without inventing one."
            )
        values = np.asarray(importances, dtype=float)
        kind = KIND_IMPORTANCE

    if values.shape[0] != len(names):
        raise ImportanceDataError(
            f"Attribution vector has {values.shape[0]} entry/entries but the "
            f"classifier was fitted on {len(names)} transformed feature(s)."
        )
    if not np.all(np.isfinite(values)):
        raise ImportanceDataError(
            "Attribution vector contains non-finite value(s); refusing to "
            "export a corrupt importance table."
        )

    report = ImportanceReport(
        feature_names=names,
        display_names=tuple(display_feature_name(name) for name in names),
        engineered=tuple(False for _ in names),
        values=tuple(float(value) for value in values),
        kind=kind,
        model_name=type(estimator).__name__,
        estimator_class=type(estimator).__name__,
        n_features=len(names),
    )
    logger.info(
        "Extracted %s attribution for %s: %d feature(s), top=%s",
        kind,
        report.model_name,
        report.n_features,
        ", ".join(
            f"{row['feature']} ({row['abs_value']:.4f})"
            for row in report.top(3)
        ),
    )
    return report


def with_engineered_flags(
    report: ImportanceReport, engineered_columns: tuple[str, ...]
) -> ImportanceReport:
    """Return a copy of ``report`` with the engineered-feature flags set."""
    if not isinstance(report, ImportanceReport):
        raise ImportanceDataError(
            f"report must be an ImportanceReport, got {type(report).__name__}."
        )
    return ImportanceReport(
        feature_names=report.feature_names,
        display_names=report.display_names,
        engineered=tuple(
            matches_engineered(name, engineered_columns)
            for name in report.feature_names
        ),
        values=report.values,
        kind=report.kind,
        model_name=report.model_name,
        estimator_class=report.estimator_class,
        n_features=report.n_features,
    )


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def render_importance_csv(report: ImportanceReport) -> str:
    """Render the importance table as CSV text (ranked by magnitude)."""
    if not isinstance(report, ImportanceReport):
        raise ImportanceArtifactError(
            f"report must be an ImportanceReport, got {type(report).__name__}."
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(IMPORTANCE_COLUMNS))
    writer.writeheader()
    for row in report.to_rows():
        writer.writerow(
            {
                "rank": row["rank"],
                "feature": row["feature"],
                "raw_feature": row["raw_feature"],
                "engineered": str(bool(row["engineered"])),
                "kind": row["kind"],
                "value": f"{float(row['value']):.10g}",
                "abs_value": f"{float(row['abs_value']):.10g}",
                "model": row["model"],
            }
        )
    return buffer.getvalue()


def write_importance(report: ImportanceReport, path: str | Path) -> Path:
    """Atomically write the importance CSV to ``path``."""
    try:
        destination = atomic_write_text(path, render_importance_csv(report))
    except OSError as exc:
        raise ImportanceArtifactError(
            f"Could not write the importance table to {path}: {exc}"
        ) from exc
    logger.info(
        "Wrote importance table to %s (%d feature(s))",
        destination,
        report.n_features,
    )
    return destination


def read_importance(path: str | Path) -> ImportanceReport:
    """Read an importance CSV written by :func:`write_importance`."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ImportanceArtifactError(
                    f"Importance table at {source} is empty."
                )
            missing = [c for c in IMPORTANCE_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ImportanceArtifactError(
                    f"Importance table at {source} is missing column(s) {missing}; "
                    f"expected {list(IMPORTANCE_COLUMNS)}."
                )
            records = list(reader)
    except OSError as exc:
        raise ImportanceArtifactError(
            f"Could not read the importance table at {source}: {exc}"
        ) from exc
    if not records:
        raise ImportanceArtifactError(
            f"Importance table at {source} has no data rows."
        )

    records.sort(key=lambda row: int(row["rank"]))
    try:
        return ImportanceReport(
            feature_names=tuple(str(row["raw_feature"]) for row in records),
            display_names=tuple(str(row["feature"]) for row in records),
            engineered=tuple(
                str(row["engineered"]).strip().lower() == "true" for row in records
            ),
            values=tuple(float(row["value"]) for row in records),
            kind=str(records[0]["kind"]),
            model_name=str(records[0]["model"]),
            estimator_class=str(records[0]["model"]),
            n_features=len(records),
        )
    except (KeyError, TypeError, ValueError, ImportanceDataError) as exc:
        raise ImportanceArtifactError(
            f"Importance table at {source} is malformed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.interpret.coefficients",
        description=(
            "Export the leading model's coefficient / feature-importance table "
            "for the selected feature representation."
        ),
    )
    parser.add_argument("--selection-ledger", default=str(DEFAULT_SELECTION_LEDGER_PATH))
    parser.add_argument("--leading-model-ledger", default=None)
    parser.add_argument("--split-version", default=None)
    parser.add_argument("--tracking-dir", default=None)
    parser.add_argument("--importance-path", default=str(DEFAULT_IMPORTANCE_PATH))
    parser.add_argument("--max-train-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        context = build_interpretability_context(
            selection_ledger_path=args.selection_ledger,
            leading_model_ledger_path=args.leading_model_ledger,
            split_version=args.split_version,
            tracking_dir=args.tracking_dir,
            max_train_rows=args.max_train_rows,
        )
        report = with_engineered_flags(
            extract_importance(context.pipeline), context.engineered_columns
        )
        write_importance(report, args.importance_path)
    except InterpretError as exc:
        print(f"importance export error: {exc}")
        return 1
    for row in report.top(5):
        print(
            f"{row['rank']:>2}. {row['feature']:<34} "
            f"{row['kind']}={row['value']:+.4f}"
            + ("  [engineered]" if row["engineered"] else "")
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
