"""Interpretability report orchestration (S04/T04).

This module is the single entry point that turns a trained leading model into
the reviewable ``reports/interpretability.md`` plus its durable CSV artifacts.
It fits the selected-representation pipeline once and feeds it to both the SHAP
exporter (:mod:`heart.interpret.shap_export`) and the coefficient/importance
extractor (:mod:`heart.interpret.coefficients`), so the two explanations are
guaranteed to describe the same fitted model.

What the report contains
------------------------
* **Setup** — the leading model, the MLflow run that produced its tuned
  parameters, the selected feature representation, and the explained row count.
* **SHAP summary** — the top design-matrix features ranked by mean ``|SHAP|``.
* **Engineered-feature contributions** — where every selected engineered column
  lands in both the SHAP and importance rankings.
* **Feature importance** — the model's own coefficient / importance table.
* **Agreement** — Spearman's rho between mean ``|SHAP|`` and absolute
  importance, a cheap cross-check that the two attribution views tell the same
  story.

Observability
-------------
:func:`generate_interpretability_report` logs every artifact path it writes.
Failures from the exporters propagate as their named subclasses; a report that
cannot be written raises :class:`InterpretabilityReportError`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from heart.config import REPORTS_DIR
from heart.features.selection import DEFAULT_LEDGER_PATH as DEFAULT_SELECTION_LEDGER_PATH
from heart.interpret.coefficients import (
    DEFAULT_IMPORTANCE_PATH,
    ImportanceReport,
    extract_importance,
    with_engineered_flags,
    write_importance,
)
from heart.interpret.model import (
    DEFAULT_LEADING_MODEL_PATH,
    InterpretError,
    InterpretabilityContext,
    atomic_write_text,
    build_interpretability_context,
    matches_engineered,
    write_leading_model_ledger,
)
from heart.interpret.shap_export import (
    DEFAULT_SHAP_SUMMARY_PATH,
    DEFAULT_SHAP_VALUES_PATH,
    ShapSummary,
    compute_shap_values,
    summarize_shap,
    write_shap_summary,
    write_shap_values,
)

logger = logging.getLogger(__name__)

__all__ = [
    "INTERPRETABILITY_REPORT_FILENAME",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_TOP_N",
    "InterpretabilityReportError",
    "InterpretabilityReport",
    "agreement_spearman",
    "render_interpretability_report",
    "write_interpretability_report",
    "generate_interpretability_report",
    "describe_report",
    "build_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Durable human-readable interpretability report.
INTERPRETABILITY_REPORT_FILENAME: str = "interpretability.md"

#: Default location of the report.
DEFAULT_REPORT_PATH: Path = Path(REPORTS_DIR) / INTERPRETABILITY_REPORT_FILENAME

#: How many ranked features the markdown tables show.
DEFAULT_TOP_N: int = 15


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class InterpretabilityReportError(InterpretError):
    """The interpretability report could not be rendered or written."""


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterpretabilityReport:
    """The fitted context, both attribution summaries, and the written paths."""

    context: InterpretabilityContext
    shap_summary: ShapSummary
    importance: ImportanceReport
    spearman: float | None
    generated_at: str
    top_n: int
    report_path: Path | None = None
    shap_summary_path: Path | None = None
    shap_values_path: Path | None = None
    importance_path: Path | None = None
    leading_model_path: Path | None = None

    @property
    def n_features(self) -> int:
        return self.context.n_features

    def to_dict(self) -> dict[str, object]:
        return {
            "leading_model": self.context.leading_model.to_dict(),
            "split_version": self.context.split_version,
            "selection_mode": self.context.selected_mode.name,
            "engineered_columns": list(self.context.engineered_columns),
            "n_features": self.n_features,
            "n_test_rows": self.context.test_rows,
            "spearman_mean_abs_shap_vs_importance": self.spearman,
            "generated_at": self.generated_at,
            "shap": self.shap_summary.to_dict(),
            "importance": self.importance.to_dict(),
            "artifacts": {
                "report": _portable(self.report_path),
                "shap_summary": _portable(self.shap_summary_path),
                "shap_values": _portable(self.shap_values_path),
                "feature_importance": _portable(self.importance_path),
                "leading_model": _portable(self.leading_model_path),
            },
        }


def _portable(path: Path | None) -> str | None:
    if path is None:
        return None
    from heart.config import PROJECT_ROOT

    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return str(path)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def agreement_spearman(
    shap_summary: ShapSummary, importance: ImportanceReport
) -> float | None:
    """Spearman's rho between mean ``|SHAP|`` and absolute importance.

    Features are aligned by their transformed (raw) name so the two tables can
    be ranked independently. Returns ``None`` when fewer than three features
    are shared or the correlation is undefined (for example every attribution
    is identical).
    """
    if not isinstance(shap_summary, ShapSummary):
        raise InterpretabilityReportError(
            f"shap_summary must be a ShapSummary, got "
            f"{type(shap_summary).__name__}."
        )
    if not isinstance(importance, ImportanceReport):
        raise InterpretabilityReportError(
            f"importance must be an ImportanceReport, got "
            f"{type(importance).__name__}."
        )
    shap_by_raw = {
        str(row["raw_feature"]): float(row["mean_abs_shap"])
        for row in shap_summary.to_rows()
    }
    importance_by_raw = {
        raw: float(abs(value))
        for raw, value in zip(importance.feature_names, importance.abs_values())
    }
    shared = sorted(set(shap_by_raw) & set(importance_by_raw))
    if len(shared) < 3:
        return None
    from scipy.stats import spearmanr

    rho, _ = spearmanr(
        [shap_by_raw[key] for key in shared],
        [importance_by_raw[key] for key in shared],
    )
    value = float(rho)
    return value if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.{digits}f}"


def render_interpretability_report(report: InterpretabilityReport) -> str:
    """Render the interpretability report as markdown."""
    if not isinstance(report, InterpretabilityReport):
        raise InterpretabilityReportError(
            f"report must be an InterpretabilityReport, got "
            f"{type(report).__name__}."
        )
    context = report.context
    model = context.leading_model

    lines: list[str] = []
    lines.append("# Model Interpretability Report")
    lines.append("")
    lines.append(
        "_Generated by `heart.interpret.report` (S04/T04). SHAP summary values "
        "and coefficient/feature-importance artifacts for the leading S03 model "
        "on the S04/T03 selected feature representation. Do not edit by hand — "
        "regenerate with `python -m heart.interpret.report`._"
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- leading model: **{model.model_name}** (`{model.model_type}`, "
        f"{model.family})"
    )
    lines.append(
        f"- leaderboard ROC-AUC: {_fmt(model.primary_metric)} "
        f"(MLflow run `{model.run_id}`, split `{model.split_version}`)"
    )
    lines.append(
        f"- tuned parameters: `{model.tuned_params}`"
    )
    lines.append(
        f"- selected representation: `{context.selected_mode.name}` "
        f"({context.n_engineered_features} engineered transform(s) kept from "
        f"{context.n_dropped + context.n_engineered_features})"
    )
    lines.append(
        "- engineered columns: "
        + (
            ", ".join(f"`{name}`" for name in context.engineered_columns)
            if context.engineered_columns
            else "_none_"
        )
    )
    lines.append(
        f"- explained rows: {report.shap_summary.n_samples} held-out row(s) "
        f"(train {context.train_rows} / test {context.test_rows})"
    )
    lines.append(
        f"- model input features: {len(context.input_feature_names)} raw "
        f"columns -> **{context.n_features}** scaled/encoded features"
    )
    lines.append(
        f"- SHAP explainer: `{_explainer_label(context)}`"
    )
    lines.append(
        f"- base value (positive class): "
        f"{_fmt(report.shap_summary.base_value)}"
    )
    lines.append("")

    lines.append(f"## SHAP summary (top {report.top_n})")
    lines.append("")
    lines.append("| rank | feature | mean abs SHAP | mean SHAP | engineered |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report.shap_summary.top(report.top_n):
        lines.append(
            f"| {row['rank']} | `{row['feature']}` | "
            f"{float(row['mean_abs_shap']):.4f} | "
            f"{_fmt_signed(row['mean_shap'])} | "
            f"{'yes' if row['engineered'] else ''} |"
        )
    lines.append("")

    lines.append("## Engineered-feature contributions")
    lines.append("")
    shap_rank = {
        str(row["raw_feature"]): (int(row["rank"]), float(row["mean_abs_shap"]))
        for row in report.shap_summary.to_rows()
    }
    importance_rank = {
        str(row["raw_feature"]): (int(row["rank"]), float(row["abs_value"]))
        for row in report.importance.to_rows()
    }
    lines.append(
        "_One-hot variants of a categorical engineered column are aggregated: "
        "rank is the best (lowest) variant rank, value is the sum over its "
        "variants._"
    )
    lines.append("")
    lines.append(
        "| engineered feature | SHAP rank | total mean abs SHAP | "
        "importance rank | total abs importance |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for column in context.engineered_columns:
        # Numeric engineered columns keep their name; categorical ones are
        # one-hot encoded into `<column>_<level>` design-matrix variants.
        candidates = [
            name
            for name in report.shap_summary.feature_names
            if matches_engineered(name, (column,))
        ]
        shap_rows = [shap_rank[c] for c in candidates if c in shap_rank]
        importance_rows = [
            importance_rank[c] for c in candidates if c in importance_rank
        ]
        shap_best = min((row[0] for row in shap_rows), default=None)
        shap_total = sum(row[1] for row in shap_rows) if shap_rows else None
        importance_best = min((row[0] for row in importance_rows), default=None)
        importance_total = (
            sum(row[1] for row in importance_rows) if importance_rows else None
        )
        lines.append(
            f"| `{column}` | "
            f"{'-' if shap_best is None else shap_best} | "
            f"{'-' if shap_total is None else format(shap_total, '.4f')} | "
            f"{'-' if importance_best is None else importance_best} | "
            f"{'-' if importance_total is None else format(importance_total, '.4f')} |"
        )
    lines.append("")

    lines.append(f"## Feature importance (top {report.top_n})")
    lines.append("")
    lines.append(
        f"Attribution kind: `{report.importance.kind}` "
        f"({'signed' if report.importance.signed else 'non-negative'}) from "
        f"`{report.importance.estimator_class}`."
    )
    lines.append("")
    lines.append("| rank | feature | value | abs value | engineered |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report.importance.top(report.top_n):
        lines.append(
            f"| {row['rank']} | `{row['feature']}` | "
            f"{_fmt_signed(row['value'])} | "
            f"{format(float(row['abs_value']), '.4f')} | "
            f"{'yes' if row['engineered'] else ''} |"
        )
    lines.append("")

    lines.append("## Agreement between SHAP and importance")
    lines.append("")
    if report.spearman is None:
        lines.append(
            "_Spearman correlation could not be computed (too few shared "
            "features or a degenerate ranking)._"
        )
    else:
        strength = (
            "strong"
            if abs(report.spearman) >= 0.7
            else "moderate"
            if abs(report.spearman) >= 0.4
            else "weak"
        )
        lines.append(
            f"Spearman's rho between mean abs SHAP and absolute importance: "
            f"**{report.spearman:+.3f}** ({strength} agreement). A high positive "
            "value means the two independent attribution views rank the features "
            "similarly, so the feature story is not an artifact of one method."
        )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for label, path in (
        ("report", report.report_path),
        ("SHAP summary", report.shap_summary_path),
        ("SHAP value matrix", report.shap_values_path),
        ("feature importance", report.importance_path),
        ("leading-model ledger", report.leading_model_path),
    ):
        if path is not None:
            lines.append(f"- {label}: `{_portable(path)}`")
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "- feature representation: `reports/feature_selection.json` (S04/T03)"
    )
    lines.append(
        "- leading model: resolved from the S03 `run_kind=final` leaderboard and "
        "recorded in `reports/leading_model.json`"
    )
    lines.append(
        "- SHAP values: positive-class attributions on the held-out split"
    )
    lines.append(f"- generated at: {report.generated_at}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m heart.interpret.report")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _explainer_label(context: InterpretabilityContext) -> str:
    estimator = context.pipeline.named_steps.get("clf")
    if hasattr(estimator, "feature_importances_"):
        return "TreeExplainer"
    if hasattr(estimator, "coef_"):
        return "LinearExplainer"
    return "PermutationExplainer"


def describe_report(report: InterpretabilityReport) -> str:
    """A short human-readable summary for the CLI."""
    if not isinstance(report, InterpretabilityReport):
        raise InterpretabilityReportError(
            f"report must be an InterpretabilityReport, got "
            f"{type(report).__name__}."
        )
    top = report.shap_summary.top(5)
    lines = [
        f"leading model: {report.context.leading_model.model_name}",
        f"selected mode: {report.context.selected_mode.name} "
        f"({report.context.n_features} transformed features)",
        f"explained rows: {report.shap_summary.n_samples}",
        "top SHAP features:",
    ]
    for row in top:
        lines.append(
            f"  {row['rank']:>2}. {row['feature']:<34} "
            f"mean|SHAP|={float(row['mean_abs_shap']):.4f}"
            + ("  [engineered]" if row["engineered"] else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_interpretability_report(
    report: InterpretabilityReport, path: str | Path
) -> Path:
    """Atomically write the markdown report to ``path``."""
    try:
        destination = atomic_write_text(path, render_interpretability_report(report))
    except OSError as exc:
        raise InterpretabilityReportError(
            f"Could not write the interpretability report to {path}: {exc}"
        ) from exc
    logger.info("Wrote interpretability report to %s", destination)
    return destination


def generate_interpretability_report(
    *,
    context: InterpretabilityContext | None = None,
    selection=None,
    leading_model=None,
    selection_ledger_path: str | Path | None = DEFAULT_SELECTION_LEDGER_PATH,
    leading_model_ledger_path: str | Path | None = None,
    split_version: str | None = None,
    tracking_dir: str | Path | None = None,
    max_train_rows: int | None = None,
    max_samples: int | None = None,
    top_n: int = DEFAULT_TOP_N,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    shap_summary_path: str | Path | None = DEFAULT_SHAP_SUMMARY_PATH,
    shap_values_path: str | Path | None = DEFAULT_SHAP_VALUES_PATH,
    importance_path: str | Path | None = DEFAULT_IMPORTANCE_PATH,
    leading_model_path: str | Path | None = DEFAULT_LEADING_MODEL_PATH,
    generated_at: str | None = None,
) -> InterpretabilityReport:
    """Fit the leading model on the selected representation and export everything.

    Pass an already-fitted ``context`` to skip loading/fitting (tests build one
    from synthetic data). Passing ``report_path``/artifact paths as ``None``
    skips that write, which tests use to produce an in-memory report without
    touching ``reports/``.
    """
    if context is None:
        context = build_interpretability_context(
            selection=selection,
            leading_model=leading_model,
            selection_ledger_path=selection_ledger_path,
            leading_model_ledger_path=leading_model_ledger_path,
            split_version=split_version,
            tracking_dir=tracking_dir,
            max_train_rows=max_train_rows,
        )
    elif not isinstance(context, InterpretabilityContext):
        raise InterpretabilityReportError(
            f"context must be an InterpretabilityContext, got "
            f"{type(context).__name__}."
        )

    shap_values = compute_shap_values(
        context.pipeline, context.test_frame, max_samples=max_samples
    )
    shap_summary = summarize_shap(
        shap_values,
        model_name=context.leading_model.model_name,
        model_type=context.leading_model.model_type,
        split_version=context.split_version,
        engineered_columns=context.engineered_columns,
    )
    importance = with_engineered_flags(
        extract_importance(context.pipeline), context.engineered_columns
    )
    spearman = agreement_spearman(shap_summary, importance)

    report = InterpretabilityReport(
        context=context,
        shap_summary=shap_summary,
        importance=importance,
        spearman=spearman,
        generated_at=generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        top_n=int(top_n),
    )

    if shap_summary_path is not None:
        report = replace(report, shap_summary_path=write_shap_summary(shap_summary, shap_summary_path))
    if shap_values_path is not None:
        report = replace(report, shap_values_path=write_shap_values(shap_values, shap_values_path))
    if importance_path is not None:
        report = replace(report, importance_path=write_importance(importance, importance_path))
    if leading_model_path is not None:
        report = replace(
            report,
            leading_model_path=write_leading_model_ledger(
                context.leading_model, leading_model_path
            ),
        )
    if report_path is not None:
        # Set the destination on the report *before* rendering so the markdown
        # "Artifacts" section lists the report file itself.
        report = replace(report, report_path=Path(report_path))
        written = write_interpretability_report(report, report_path)
        report = replace(report, report_path=written)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.interpret.report",
        description=(
            "Generate reports/interpretability.md plus SHAP and importance "
            "artifacts for the leading model on the selected representation."
        ),
    )
    parser.add_argument("--selection-ledger", default=str(DEFAULT_SELECTION_LEDGER_PATH))
    parser.add_argument("--leading-model-ledger", default=None)
    parser.add_argument("--split-version", default=None)
    parser.add_argument("--tracking-dir", default=None)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--shap-summary-path", default=str(DEFAULT_SHAP_SUMMARY_PATH))
    parser.add_argument("--shap-values-path", default=str(DEFAULT_SHAP_VALUES_PATH))
    parser.add_argument("--importance-path", default=str(DEFAULT_IMPORTANCE_PATH))
    parser.add_argument("--leading-model-path", default=str(DEFAULT_LEADING_MODEL_PATH))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        report = generate_interpretability_report(
            selection_ledger_path=args.selection_ledger,
            leading_model_ledger_path=args.leading_model_ledger,
            split_version=args.split_version,
            tracking_dir=args.tracking_dir,
            max_train_rows=args.max_train_rows,
            max_samples=args.max_samples,
            top_n=args.top_n,
            report_path=args.report_path,
            shap_summary_path=args.shap_summary_path,
            shap_values_path=args.shap_values_path,
            importance_path=args.importance_path,
            leading_model_path=args.leading_model_path,
        )
    except InterpretError as exc:
        print(f"interpretability error: {exc}")
        return 1
    print(describe_report(report))
    print(f"report: {report.report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
