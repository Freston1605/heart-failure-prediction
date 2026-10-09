"""The published logistic-regression baseline for the heart-failure portfolio.

This is the honesty anchor: every later model is compared against these
numbers, so the baseline must be trained and scored exactly like every other
model — through the S01 group-aware split and the shared
:func:`heart.eval.contract.evaluate` contract — and recorded with its full
metric suite in MLflow.

Contract
--------
* **Data** — the versioned S01 split artifacts
  (``data/processed/splits/v1/{train,test}.csv``), which are group-aware and
  leakage-safe. The model is fit on the training rows only and evaluated on the
  held-out test rows.
* **Model** — :func:`build_baseline_model` assembles the declared
  preprocessing chain (zero-as-missing median imputation, numeric scaling, and
  one-hot encoding) plus ``LogisticRegression`` with
  :data:`BASELINE_PARAMS`. No feature engineering, no tuning: a plainly named
  floor.
* **Scoring** — :func:`heart.eval.contract.evaluate`, the single evaluation
  path, produces the canonical metric dict (accuracy, precision, recall, F1,
  ROC-AUC, PR-AUC, specificity, NPV, prevalence, Brier score, confusion matrix,
  and the full reliability curve).
* **Tracking** — the run is logged through
  :func:`heart.tracking.run.log_evaluation_run` under the frozen convention
  (run name ``logistic-regression-v1``), and the human-readable report is
  written to ``reports/baseline.md``.

Observability
-------------
:func:`run_baseline` logs the split sizes, fitted model, run id, and primary
metric at ``INFO``. :func:`render_baseline_report` renders the full report, and
``python -m heart.models.baseline`` prints a summary and writes the report.
Failures are named: a missing split version raises
:class:`~heart.data.split.SplitNotFoundError`; a malformed metric dict raises
the eval-contract error; a tracking failure raises a
:class:`~heart.tracking.TrackingError` subclass.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart.config import RANDOM_SEED, REPORTS_DIR
from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.eval.contract import (
    CALIBRATION_KEY,
    CONFUSION_MATRIX_KEY,
    METRIC_SCHEMA_VERSION,
    PRIMARY_METRIC,
    evaluate,
    evaluation_split_from_frames,
)
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT
from heart.tracking.run import RunResult, log_evaluation_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class BaselineError(Exception):
    """Base class for baseline training, evaluation, and publishing failures."""


class BaselineDataError(BaselineError):
    """A baseline input frame is missing declared columns."""


class BaselineReportError(BaselineError):
    """The baseline report could not be written to disk."""


# ---------------------------------------------------------------------------
# Declared baseline configuration
# ---------------------------------------------------------------------------

#: Human-readable model name; slugified to ``logistic-regression`` in MLflow.
BASELINE_MODEL_NAME: str = "logistic-regression"

#: The declared, untuned hyperparameters. Changing these redefines the baseline.
BASELINE_PARAMS: dict[str, object] = {
    "solver": "lbfgs",
    "max_iter": 2000,
    "C": 1.0,
    "class_weight": None,
    "random_state": RANDOM_SEED,
}

#: Where the human-readable baseline report is published.
DEFAULT_REPORT_PATH: Path = Path(REPORTS_DIR) / "baseline.md"

#: Default split version consumed by the baseline.
BASELINE_SPLIT_VERSION: str = SPLIT_VERSION


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Everything produced by one baseline run."""

    model_name: str
    split_version: str
    train_rows: int
    test_rows: int
    params: dict[str, object]
    metrics: dict[str, object]
    run_id: str | None
    experiment_name: str
    report_path: Path | None
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "split_version": self.split_version,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "report_path": str(self.report_path) if self.report_path else None,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Model construction / training
# ---------------------------------------------------------------------------


def build_baseline_model(params: dict[str, object] | None = None) -> Pipeline:
    """Build the unfitted baseline pipeline: preprocessing + logistic regression."""
    resolved = dict(BASELINE_PARAMS if params is None else params)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", LogisticRegression(**resolved)),
        ]
    )


def train_baseline(
    train_frame: pd.DataFrame, *, params: dict[str, object] | None = None
) -> Pipeline:
    """Fit the baseline on the training rows of the S01 split.

    Only ``train_frame`` is read; the held-out rows are never seen here.
    """
    missing = [
        column
        for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
        if column not in train_frame.columns
    ]
    if missing:
        raise BaselineDataError(
            f"Training frame is missing column(s) {missing}; load the S01 split "
            "with heart.data.split.load_split_frames."
        )
    model = build_baseline_model(params)
    model.fit(train_frame[list(FEATURE_COLUMNS)], train_frame[TARGET_COLUMN])
    logger.info(
        "Trained %s baseline on %d training row(s): %s",
        BASELINE_MODEL_NAME,
        len(train_frame),
        {key: value for key, value in (params or BASELINE_PARAMS).items()},
    )
    return model


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _num(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def render_baseline_report(result: BaselineResult) -> str:
    """Render the published baseline report from an evaluated result."""
    metrics = result.metrics
    matrix = metrics[CONFUSION_MATRIX_KEY]
    calibration = metrics[CALIBRATION_KEY]

    lines: list[str] = []
    lines.append("# Logistic-Regression Baseline")
    lines.append("")
    lines.append(
        "_Published by `heart.models.baseline` (S02/T03). This is the honesty "
        "anchor: every tuned model in the portfolio is measured against these "
        "numbers._"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(
        f"| **ROC-AUC (primary)** | **{_num(metrics[PRIMARY_METRIC])}** |"
    )
    for key in (
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "npv",
        "prevalence",
        "brier_score",
    ):
        lines.append(f"| {key} | {_num(metrics[key])} |")
    lines.append(
        f"| Expected calibration error | "
        f"{_num(calibration['expected_calibration_error'])} |"
    )
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("| | Predicted 0 | Predicted 1 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Actual 0 | {matrix['tn']} (TN) | {matrix['fp']} (FP) |")
    lines.append(f"| Actual 1 | {matrix['fn']} (FN) | {matrix['tp']} (TP) |")
    lines.append("")
    lines.append(
        f"Evaluated on **{matrix['n_samples']}** held-out rows "
        f"({matrix['n_positive']} positive / {matrix['n_negative']} negative)."
    )
    lines.append("")
    lines.append("## Calibration (reliability curve)")
    lines.append("")
    lines.append(
        f"Brier score **{_num(calibration['brier_score'])}**, expected "
        f"calibration error **{_num(calibration['expected_calibration_error'])}** "
        f"over {calibration['n_bins']} equal-width bins."
    )
    lines.append("")
    lines.append("| Bin | Range | Count | Mean predicted | Observed | Gap |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for bucket in calibration["bins"]:
        low, high = _num(bucket["lower"], 2), _num(bucket["upper"], 2)
        if bucket["count"] == 0:
            lines.append(
                f"| {bucket['bin']} | {low}-{high} | 0 | - | - | - |"
            )
        else:
            lines.append(
                f"| {bucket['bin']} | {low}-{high} | {bucket['count']} | "
                f"{_num(bucket['mean_predicted'])} | "
                f"{_num(bucket['fraction_positive'])} | "
                f"{_num(bucket['gap'])} |"
            )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        f"- dataset: fedesoriano combined 5-site collection (918 rows, 11 features)"
    )
    lines.append(
        f"- split: `{result.split_version}` group-aware, leakage-safe "
        f"(train {result.train_rows} / test {result.test_rows})"
    )
    lines.append(
        "- model: zero-as-missing median imputation -> standard scaling + "
        "one-hot encoding -> logistic regression"
    )
    lines.append(f"- params: `{result.params}`")
    lines.append(f"- metric schema: v{METRIC_SCHEMA_VERSION}")
    lines.append(
        f"- MLflow run: `{result.run_id or 'not logged'}` "
        f"(experiment `{result.experiment_name}`)"
    )
    lines.append(f"- generated at: {result.generated_at}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m heart.models.baseline")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_baseline(
    *,
    split_version: str = BASELINE_SPLIT_VERSION,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    log_to_mlflow: bool = True,
    write_report: bool = True,
) -> BaselineResult:
    """Train, evaluate, log, and publish the logistic-regression baseline."""
    train_frame, test_frame = load_split_frames(split_version)

    model = train_baseline(train_frame)
    split = evaluation_split_from_frames(
        test_frame, name=f"{split_version}/test"
    )
    metrics = evaluate(model, split)

    run: RunResult | None = None
    if log_to_mlflow:
        run = log_evaluation_run(
            metrics,
            model_name=BASELINE_MODEL_NAME,
            split_version=split_version,
            params=BASELINE_PARAMS,
            tags={"slice": "S02", "role": "baseline"},
            experiment_name=experiment_name,
            tracking_dir=tracking_dir,
        )

    result = BaselineResult(
        model_name=BASELINE_MODEL_NAME,
        split_version=split_version,
        train_rows=int(len(train_frame)),
        test_rows=int(len(test_frame)),
        params=dict(BASELINE_PARAMS),
        metrics=metrics,
        run_id=run.run_id if run else None,
        experiment_name=experiment_name,
        report_path=None,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    if write_report:
        destination = Path(report_path) if report_path is not None else None
        rendered = render_baseline_report(result)
        if destination is not None:
            try:
                _write_text_atomic(destination, rendered)
            except OSError as exc:
                raise BaselineReportError(
                    f"Could not write baseline report to {destination}: {exc}"
                ) from exc
            logger.info("Wrote baseline report to %s", destination)
            result = replace(result, report_path=destination)

    logger.info(
        "Baseline %s on %s: %s=%.4f accuracy=%.4f (run=%s)",
        BASELINE_MODEL_NAME,
        split_version,
        PRIMARY_METRIC,
        float(metrics[PRIMARY_METRIC]),
        float(metrics["accuracy"]),
        run.run_id if run else "not logged",
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.models.baseline",
        description="Train, evaluate, log, and publish the logistic-regression baseline.",
    )
    parser.add_argument(
        "--split-version", default=BASELINE_SPLIT_VERSION, help="S01 split version."
    )
    parser.add_argument(
        "--tracking-dir",
        default=None,
        help="MLflow store directory (default: experiments/mlruns).",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Where to write the report (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip MLflow logging."
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip writing the report file."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    result = run_baseline(
        split_version=args.split_version,
        tracking_dir=args.tracking_dir,
        experiment_name=args.experiment,
        report_path=args.report_path,
        log_to_mlflow=not args.no_mlflow,
        write_report=not args.no_report,
    )
    metrics = result.metrics
    print(f"model: {result.model_name}")
    print(f"split: {result.split_version} (train {result.train_rows} / test {result.test_rows})")
    print(f"params: {result.params}")
    print(f"MLflow run: {result.run_id}")
    print(f"report: {result.report_path}")
    print("headline metrics:")
    for key in (
        PRIMARY_METRIC,
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "npv",
        "brier_score",
    ):
        print(f"  {key}: {_num(metrics[key])}")
    matrix = metrics[CONFUSION_MATRIX_KEY]
    print(
        f"confusion: tn={matrix['tn']} fp={matrix['fp']} "
        f"fn={matrix['fn']} tp={matrix['tp']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
