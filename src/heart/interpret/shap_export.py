"""SHAP summary values for the leading model (S04/T04).

What this module answers
------------------------
The leaderboard in S03 says *how well* each model scored. This module, together
with :mod:`heart.interpret.coefficients`, says *what the winning model actually
learned*: which features moved its predictions, and in which direction.

:func:`compute_shap_values` fits nothing itself — it explains an already-fitted
selected-representation pipeline (built by
:mod:`heart.interpret.model`) on the held-out rows. The result is the
per-feature, per-sample SHAP value matrix for the positive class, reduced by
:func:`summarize_shap` into a durable summary: mean ``|SHAP|`` (importance) and
mean signed SHAP (direction) for every design-matrix column.

Durable artifacts
-----------------
* ``reports/shap_summary.csv`` — one row per transformed feature with its
  mean ``|SHAP|``, mean signed SHAP, engineered flag, and rank.
* ``reports/shap_values.csv`` — the full ``n_samples × n_features`` value
  matrix, so a reviewer can re-derive any summary without retraining.

Explainers
----------
Tree ensembles go through :class:`shap.TreeExplainer`, linear models through
:class:`shap.LinearExplainer`, and anything else through a permutation
explainer over ``predict_proba``. An unsupported model raises
:class:`UnsupportedShapModelError` rather than silently emitting zeros.

Observability
-------------
:func:`compute_shap_values` logs the explained row count, feature width, and
the explainer class at ``INFO``; :func:`summarize_shap` logs the top features.
Failures are named: a missing frame column, an unfitted pipeline, a non-binary
output, a malformed artifact, and a missing ``shap`` install each raise their
own subclass of :class:`ShapExportError`.
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
import pandas as pd

from heart.config import REPORTS_DIR
from heart.features.selection import DEFAULT_LEDGER_PATH as DEFAULT_SELECTION_LEDGER_PATH
from heart.interpret.model import (
    InterpretError,
    SelectedPipelineError,
    atomic_write_text,
    build_interpretability_context,
    display_feature_name,
    matches_engineered,
    model_input_feature_names,
    transformed_feature_names,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SHAP_SUMMARY_FILENAME",
    "SHAP_VALUES_FILENAME",
    "DEFAULT_SHAP_SUMMARY_PATH",
    "DEFAULT_SHAP_VALUES_PATH",
    "ShapExportError",
    "ShapConfigError",
    "ShapDataError",
    "ShapDependencyError",
    "UnsupportedShapModelError",
    "ShapComputationError",
    "ShapArtifactError",
    "ShapValues",
    "ShapSummary",
    "build_explainer",
    "compute_shap_values",
    "summarize_shap",
    "render_shap_summary_csv",
    "read_shap_summary",
    "write_shap_summary",
    "write_shap_values",
    "read_shap_values",
    "build_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Durable per-feature SHAP summary artifact.
SHAP_SUMMARY_FILENAME: str = "shap_summary.csv"

#: Durable full SHAP value matrix artifact.
SHAP_VALUES_FILENAME: str = "shap_values.csv"

#: Default artifact locations.
DEFAULT_SHAP_SUMMARY_PATH: Path = Path(REPORTS_DIR) / SHAP_SUMMARY_FILENAME
DEFAULT_SHAP_VALUES_PATH: Path = Path(REPORTS_DIR) / SHAP_VALUES_FILENAME

#: Header of the summary CSV (order is part of the artifact contract).
SUMMARY_COLUMNS: tuple[str, ...] = (
    "rank",
    "feature",
    "raw_feature",
    "engineered",
    "mean_abs_shap",
    "mean_shap",
    "n_samples",
    "model",
    "split_version",
)

#: The positive class index in a binary explainer output.
POSITIVE_CLASS_INDEX: int = 1

#: Default cap on explained rows; ``None`` means every held-out row.
DEFAULT_MAX_SAMPLES: int | None = None


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class ShapExportError(InterpretError):
    """Base class for every SHAP-export failure."""


class ShapConfigError(ShapExportError):
    """A SHAP-export configuration value is invalid."""


class ShapDataError(ShapExportError):
    """The frame or pipeline handed to the exporter is missing or malformed."""


class ShapDependencyError(ShapExportError):
    """The optional ``shap`` package is not installed."""


class UnsupportedShapModelError(ShapExportError):
    """No SHAP explainer is available for the fitted estimator."""


class ShapComputationError(ShapExportError):
    """A SHAP explainer failed or returned an unexpected shape."""


class ShapArtifactError(ShapExportError):
    """A SHAP artifact could not be serialised, written, or read."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ShapValues:
    """The positive-class SHAP value matrix for one explanation run."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    display_names: tuple[str, ...]
    base_value: float | None
    explainer_class: str
    model_name: str
    model_type: str
    split_version: str

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=float)
        if array.ndim != 2:
            raise ShapDataError(
                f"SHAP values must be two-dimensional (n_samples, n_features), "
                f"got shape {array.shape}."
            )
        if len(self.feature_names) != array.shape[1]:
            raise ShapDataError(
                f"SHAP value matrix has {array.shape[1]} column(s) but "
                f"{len(self.feature_names)} feature name(s) were supplied."
            )
        if len(self.display_names) != array.shape[1]:
            raise ShapDataError(
                "display_names must have one entry per feature column."
            )
        object.__setattr__(self, "values", array)


@dataclass(frozen=True)
class ShapSummary:
    """Per-feature SHAP summary: importance and signed direction."""

    feature_names: tuple[str, ...]
    display_names: tuple[str, ...]
    engineered: tuple[bool, ...]
    mean_abs_shap: tuple[float, ...]
    mean_shap: tuple[float, ...]
    n_samples: int
    n_features: int
    model_name: str
    model_type: str
    split_version: str
    base_value: float | None = None

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        for field_name, seq in (
            ("display_names", self.display_names),
            ("engineered", self.engineered),
            ("mean_abs_shap", self.mean_abs_shap),
            ("mean_shap", self.mean_shap),
        ):
            if len(seq) != n:
                raise ShapDataError(
                    f"ShapSummary.{field_name} has {len(seq)} entry/entries but "
                    f"{n} feature(s) were declared."
                )
        if self.n_features != n:
            raise ShapDataError(
                f"ShapSummary.n_features={self.n_features} does not match "
                f"{n} feature name(s)."
            )
        for value in self.mean_abs_shap:
            if not np.isfinite(value) or value < 0:
                raise ShapDataError(
                    f"mean_abs_shap must be finite and non-negative, got {value!r}."
                )

    def _order(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                range(self.n_features),
                key=lambda index: (-self.mean_abs_shap[index], index),
            )
        )

    def top(self, k: int) -> tuple[dict[str, object], ...]:
        """The ``k`` highest-importance features (by mean ``|SHAP|``)."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ShapConfigError(f"k must be a non-negative int, got {k!r}.")
        return tuple(self.to_rows()[:k])

    def to_rows(self) -> list[dict[str, object]]:
        """One serialisable row per feature, ranked by mean ``|SHAP|``."""
        rows: list[dict[str, object]] = []
        for rank, index in enumerate(self._order(), start=1):
            rows.append(
                {
                    "rank": rank,
                    "feature": self.display_names[index],
                    "raw_feature": self.feature_names[index],
                    "engineered": bool(self.engineered[index]),
                    "mean_abs_shap": float(self.mean_abs_shap[index]),
                    "mean_shap": float(self.mean_shap[index]),
                    "n_samples": int(self.n_samples),
                    "model": self.model_name,
                    "split_version": self.split_version,
                }
            )
        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "model_type": self.model_type,
            "split_version": self.split_version,
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "base_value": None if self.base_value is None else float(self.base_value),
            "features": self.to_rows(),
        }


# ---------------------------------------------------------------------------
# Explainers
# ---------------------------------------------------------------------------


def _import_shap():
    try:  # pragma: no cover - import path exercised by the environment
        import shap
    except ImportError as exc:  # pragma: no cover - exercised when shap absent
        raise ShapDependencyError(
            "SHAP explainability requires the 'shap' package, which is not "
            'installed. Install the ml extra with: pip install -e ".[ml]"'
        ) from exc
    return shap


def build_explainer(estimator: object, background: np.ndarray):
    """Choose the SHAP explainer appropriate to ``estimator``.

    Tree ensembles use :class:`shap.TreeExplainer`, models exposing ``coef_``
    use :class:`shap.LinearExplainer`, and everything else falls back to a
    permutation explainer over ``predict_proba``. A model with no usable
    prediction surface raises :class:`UnsupportedShapModelError`.
    """
    shap = _import_shap()
    if hasattr(estimator, "feature_importances_"):
        return shap.TreeExplainer(estimator)
    if hasattr(estimator, "coef_"):
        return shap.LinearExplainer(estimator, background)
    predictor = getattr(estimator, "predict_proba", None)
    if not callable(predictor):
        raise UnsupportedShapModelError(
            f"No SHAP explainer is available for "
            f"{type(estimator).__name__}: it exposes neither "
            "feature_importances_ nor coef_ nor a callable predict_proba."
        )
    return shap.Explainer(predictor, background)


def _positive_class_values(raw: object, n_features: int) -> np.ndarray:
    """Reduce a binary explainer output to a ``(n_samples, n_features)`` array."""
    array = np.asarray(raw, dtype=float)
    if array.ndim == 3:
        if array.shape[2] != 2:
            raise ShapComputationError(
                "SHAP output has a non-binary class axis of size "
                f"{array.shape[2]}; this exporter explains binary classifiers "
                "only."
            )
        array = array[:, :, POSITIVE_CLASS_INDEX]
    elif array.ndim != 2:
        raise ShapComputationError(
            f"SHAP output must be two- or three-dimensional, got shape "
            f"{array.shape}."
        )
    if array.shape[1] != n_features:
        raise ShapComputationError(
            f"SHAP output has {array.shape[1]} feature column(s) but the model "
            f"was fitted on {n_features} transformed feature(s)."
        )
    if not np.all(np.isfinite(array)):
        raise ShapComputationError(
            "SHAP output contains non-finite value(s); refusing to export a "
            "corrupt explanation."
        )
    return array


def _scalar_base_value(raw: object) -> float | None:
    array = np.asarray(raw, dtype=float)
    if array.ndim == 2 and array.shape[1] == 2:
        value = float(array[0, POSITIVE_CLASS_INDEX])
    elif array.ndim >= 1:
        value = float(array.ravel()[0])
    else:
        value = float(array)
    return value if np.isfinite(value) else None


def compute_shap_values(
    pipeline,
    features: pd.DataFrame,
    *,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
    random_state: int | None = None,
    explainer: object | None = None,
) -> ShapValues:
    """Compute positive-class SHAP values for a fitted selected pipeline.

    ``features`` must contain every raw column the pipeline was fitted on
    (engineered columns are produced *inside* the pipeline). At most
    ``max_samples`` held-out rows are explained, sampled deterministically with
    ``random_state`` when a cap is set — SHAP cost grows with the row count, so
    the cap is the load-shedding control.
    """
    from heart.config import RANDOM_SEED
    from sklearn.pipeline import Pipeline

    if not isinstance(pipeline, Pipeline) or "clf" not in pipeline.named_steps:
        raise ShapDataError(
            "pipeline must be a fitted sklearn Pipeline ending in a 'clf' step."
        )
    if not isinstance(features, pd.DataFrame):
        raise ShapDataError(
            f"features must be a pandas.DataFrame, got {type(features).__name__}."
        )
    if len(features) == 0:
        raise ShapDataError("features is empty; nothing to explain.")

    try:
        input_names = model_input_feature_names(pipeline)
    except SelectedPipelineError as exc:
        raise ShapDataError(str(exc)) from exc
    missing = [name for name in input_names if name not in features.columns]
    if missing:
        raise ShapDataError(
            f"features is missing model input column(s) {missing}. Present "
            f"columns: {list(features.columns)}."
        )

    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise ShapConfigError(
                f"max_samples must be None or an int, got "
                f"{type(max_samples).__name__}."
            )
        if max_samples < 1:
            raise ShapConfigError(
                f"max_samples must be at least 1, got {max_samples}."
            )
        if len(features) > max_samples:
            features = features.sample(
                n=max_samples, random_state=random_state or RANDOM_SEED
            ).sort_index()

    X = features[list(input_names)]
    try:
        transformed = pipeline[:-1].transform(X)
    except Exception as exc:  # noqa: BLE001 - re-raise as a named error
        raise ShapComputationError(
            f"Could not transform the explanation rows: {exc}"
        ) from exc

    estimator = pipeline.named_steps["clf"]
    resolved_explainer = (
        explainer if explainer is not None else build_explainer(estimator, np.asarray(transformed))
    )
    try:
        explanation = resolved_explainer(np.asarray(transformed))
        raw_values = explanation.values
        base = explanation.base_values
    except Exception as exc:  # noqa: BLE001 - re-raise as a named error
        raise ShapComputationError(
            f"SHAP explainer {type(resolved_explainer).__name__} failed: {exc}"
        ) from exc

    try:
        names = transformed_feature_names(pipeline)
    except SelectedPipelineError as exc:
        raise ShapDataError(str(exc)) from exc
    values = _positive_class_values(raw_values, len(names))
    result = ShapValues(
        values=values,
        feature_names=names,
        display_names=tuple(display_feature_name(name) for name in names),
        base_value=_scalar_base_value(base),
        explainer_class=type(resolved_explainer).__name__,
        model_name=getattr(pipeline.named_steps["clf"], "__class__").__name__,
        model_type=getattr(pipeline.named_steps["clf"], "__class__").__name__,
        split_version="",
    )
    logger.info(
        "Computed SHAP values with %s: %d row(s) x %d feature(s)",
        result.explainer_class,
        result.n_samples,
        result.n_features,
    )
    return result


def summarize_shap(
    shap_values: ShapValues,
    *,
    model_name: str | None = None,
    model_type: str | None = None,
    split_version: str | None = None,
    engineered_columns: tuple[str, ...] = (),
) -> ShapSummary:
    """Reduce a SHAP matrix to per-feature importance and signed direction."""
    if not isinstance(shap_values, ShapValues):
        raise ShapDataError(
            f"shap_values must be a ShapValues, got {type(shap_values).__name__}."
        )
    array = np.asarray(shap_values.values, dtype=float)
    summary = ShapSummary(
        feature_names=shap_values.feature_names,
        display_names=shap_values.display_names,
        engineered=tuple(
            matches_engineered(name, engineered_columns)
            for name in shap_values.feature_names
        ),
        mean_abs_shap=tuple(float(value) for value in np.abs(array).mean(axis=0)),
        mean_shap=tuple(float(value) for value in array.mean(axis=0)),
        n_samples=int(array.shape[0]),
        n_features=int(array.shape[1]),
        model_name=model_name or shap_values.model_name,
        model_type=model_type or shap_values.model_type,
        split_version=split_version if split_version is not None else shap_values.split_version,
        base_value=shap_values.base_value,
    )
    top = summary.top(3)
    logger.info(
        "SHAP summary: top features %s",
        ", ".join(f"{row['feature']} ({row['mean_abs_shap']:.4f})" for row in top),
    )
    return summary


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def render_shap_summary_csv(summary: ShapSummary) -> str:
    """Render the summary as CSV text (ranked by mean ``|SHAP|``)."""
    if not isinstance(summary, ShapSummary):
        raise ShapArtifactError(
            f"summary must be a ShapSummary, got {type(summary).__name__}."
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(SUMMARY_COLUMNS))
    writer.writeheader()
    for row in summary.to_rows():
        writer.writerow(
            {
                "rank": row["rank"],
                "feature": row["feature"],
                "raw_feature": row["raw_feature"],
                "engineered": str(bool(row["engineered"])),
                "mean_abs_shap": f"{float(row['mean_abs_shap']):.10g}",
                "mean_shap": f"{float(row['mean_shap']):.10g}",
                "n_samples": row["n_samples"],
                "model": row["model"],
                "split_version": row["split_version"],
            }
        )
    return buffer.getvalue()


def write_shap_summary(summary: ShapSummary, path: str | Path) -> Path:
    """Atomically write the SHAP summary CSV to ``path``."""
    try:
        destination = atomic_write_text(path, render_shap_summary_csv(summary))
    except OSError as exc:
        raise ShapArtifactError(
            f"Could not write the SHAP summary to {path}: {exc}"
        ) from exc
    logger.info("Wrote SHAP summary to %s (%d feature(s))", destination, summary.n_features)
    return destination


def read_shap_summary(path: str | Path) -> ShapSummary:
    """Read a SHAP summary CSV written by :func:`write_shap_summary`."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ShapArtifactError(f"SHAP summary at {source} is empty.")
            missing = [c for c in SUMMARY_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ShapArtifactError(
                    f"SHAP summary at {source} is missing column(s) {missing}; "
                    f"expected {list(SUMMARY_COLUMNS)}."
                )
            records = list(reader)
    except OSError as exc:
        raise ShapArtifactError(
            f"Could not read the SHAP summary at {source}: {exc}"
        ) from exc
    if not records:
        raise ShapArtifactError(f"SHAP summary at {source} has no data rows.")

    records.sort(key=lambda row: int(row["rank"]))
    try:
        n_samples = int(records[0]["n_samples"])
        n_features = len(records)
        return ShapSummary(
            feature_names=tuple(str(row["raw_feature"]) for row in records),
            display_names=tuple(str(row["feature"]) for row in records),
            engineered=tuple(str(row["engineered"]).strip().lower() == "true" for row in records),
            mean_abs_shap=tuple(float(row["mean_abs_shap"]) for row in records),
            mean_shap=tuple(float(row["mean_shap"]) for row in records),
            n_samples=n_samples,
            n_features=n_features,
            model_name=str(records[0]["model"]),
            model_type=str(records[0]["model"]),
            split_version=str(records[0]["split_version"]),
        )
    except (KeyError, TypeError, ValueError, ShapDataError) as exc:
        raise ShapArtifactError(
            f"SHAP summary at {source} is malformed: {exc}"
        ) from exc


def write_shap_values(shap_values: ShapValues, path: str | Path) -> Path:
    """Atomically write the full SHAP value matrix as CSV."""
    if not isinstance(shap_values, ShapValues):
        raise ShapArtifactError(
            f"shap_values must be a ShapValues, got {type(shap_values).__name__}."
        )
    frame = pd.DataFrame(
        shap_values.values, columns=list(shap_values.display_names)
    )
    frame.index.name = "sample"
    try:
        destination = atomic_write_text(path, frame.to_csv())
    except OSError as exc:
        raise ShapArtifactError(
            f"Could not write the SHAP value matrix to {path}: {exc}"
        ) from exc
    logger.info(
        "Wrote SHAP value matrix to %s (%d x %d)",
        destination,
        shap_values.n_samples,
        shap_values.n_features,
    )
    return destination


def read_shap_values(path: str | Path) -> pd.DataFrame:
    """Read a full SHAP value matrix written by :func:`write_shap_values`."""
    source = Path(path)
    try:
        frame = pd.read_csv(source, index_col=0)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ShapArtifactError(
            f"Could not read the SHAP value matrix at {source}: {exc}"
        ) from exc
    if frame.empty:
        raise ShapArtifactError(f"SHAP value matrix at {source} is empty.")
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.interpret.shap_export",
        description=(
            "Compute and export the positive-class SHAP summary for the leading "
            "model on the selected feature representation."
        ),
    )
    parser.add_argument("--selection-ledger", default=str(DEFAULT_SELECTION_LEDGER_PATH))
    parser.add_argument("--leading-model-ledger", default=None)
    parser.add_argument("--split-version", default=None)
    parser.add_argument("--tracking-dir", default=None)
    parser.add_argument(
        "--summary-path", default=str(DEFAULT_SHAP_SUMMARY_PATH)
    )
    parser.add_argument("--values-path", default=str(DEFAULT_SHAP_VALUES_PATH))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Cap the number of held-out rows explained (default: all rows).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Deterministically subsample training rows (smoke runs).",
    )
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
        values = compute_shap_values(
            context.pipeline,
            context.test_frame,
            max_samples=args.max_samples,
        )
        summary = summarize_shap(
            values,
            model_name=context.leading_model.model_name,
            model_type=context.leading_model.model_type,
            split_version=context.split_version,
            engineered_columns=context.engineered_columns,
        )
        write_shap_summary(summary, args.summary_path)
        write_shap_values(values, args.values_path)
    except InterpretError as exc:
        print(f"shap export error: {exc}")
        return 1
    for row in summary.top(5):
        print(
            f"{row['rank']:>2}. {row['feature']:<34} "
            f"mean|SHAP|={row['mean_abs_shap']:.4f} mean={row['mean_shap']:+.4f}"
            + ("  [engineered]" if row["engineered"] else "")
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
