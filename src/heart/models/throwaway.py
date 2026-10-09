"""Deliberately throwaway models that prove the evaluation contract is agnostic.

The logistic-regression baseline (:mod:`heart.models.baseline`) and this module
must not share a single line of scoring code — they share only the held-out
split and :func:`heart.eval.contract.evaluate`. If the canonical metric dict
were quietly shaped around logistic regression, a structurally different model
would produce a different shape or a different code path. It does not.

This module ships two throwaway model families:

* ``decision-tree`` (default) — a :class:`~sklearn.tree.DecisionTreeClassifier`
  with no feature scaling and hard axis-aligned splits: a tree, not a linear
  model. It is the "genuinely different" counterpart to the baseline.
* ``dummy`` — a :class:`~sklearn.dummy.DummyClassifier` predicting the class
  prior. It is the degenerate floor: no learning at all, yet still a valid
  ``predict``/``predict_proba`` model that flows through the same contract.

Neither is a portfolio candidate. They exist to demonstrate, executably, that
:func:`~heart.eval.contract.evaluate` and
:func:`~heart.tracking.run.log_evaluation_run` accept any estimator family and
return an identically-shaped metric dict and an identically-shaped MLflow run.

Contract
--------
* **Data** — the versioned S01 split artifacts, exactly as the baseline uses.
  The model is fit on the training rows only and evaluated on the held-out
  test rows.
* **Preprocessing** — the declared leakage-safe chain from
  :func:`heart.data.pipeline.build_preprocessing_pipeline`, fit inside the
  pipeline on the training rows only. A tree does not need scaling, but it does
  need the imputation that maps the dataset's structural zeros to ``NaN``; the
  point is that the data contract is identical while the classifier is not.
* **Scoring** — :func:`heart.eval.contract.evaluate`, never a local re-implementation.
* **Tracking** — :func:`heart.tracking.run.log_evaluation_run` under the frozen
  convention, run name ``<model-slug>-<split-version>`` (for example
  ``throwaway-decision-tree-v1``).

Observability
-------------
:func:`run_throwaway` logs the model type, split sizes, run id, and primary
metric at ``INFO``. :func:`run_all_throwaway` runs every declared model type in
one call and logs each. ``python -m heart.models.throwaway --all`` exercises the
whole proof from the command line. Failures are named: an unknown model type
raises :class:`UnknownThrowawayModelError`; a frame missing declared columns
raises :class:`ThrowawayDataError`; a missing split version raises
:class:`~heart.data.split.SplitNotFoundError`; a tracking failure raises a
:class:`~heart.tracking.TrackingError` subclass.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from heart.config import RANDOM_SEED
from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.eval.contract import PRIMARY_METRIC, evaluate, evaluation_split_from_frames
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT
from heart.tracking.run import RunResult, log_evaluation_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class ThrowawayError(Exception):
    """Base class for throwaway-model training, evaluation, and logging failures."""


class UnknownThrowawayModelError(ThrowawayError):
    """The requested throwaway model type is not declared."""


class ThrowawayDataError(ThrowawayError):
    """A throwaway-model input frame is missing declared columns."""


# ---------------------------------------------------------------------------
# Declared throwaway models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThrowawaySpec:
    """One declared throwaway model family.

    ``classifier`` is the unfitted scikit-learn estimator class; ``params`` are
    the fixed, declared hyperparameters. Changing either redefines the proof.
    """

    model_type: str
    model_name: str
    classifier: type[ClassifierMixin]
    params: dict[str, object]
    description: str


#: Default model type when none is requested.
DEFAULT_MODEL_TYPE: str = "decision-tree"

#: The declared, untuned throwaway models. Keys are the ``--model-type`` values.
THROWAWAY_SPECS: dict[str, ThrowawaySpec] = {
    "decision-tree": ThrowawaySpec(
        model_type="decision-tree",
        model_name="throwaway-decision-tree",
        classifier=DecisionTreeClassifier,
        params={
            "max_depth": 4,
            "min_samples_leaf": 5,
            "random_state": RANDOM_SEED,
        },
        description=(
            "a single pruned decision tree (axis-aligned splits, no linear "
            "decision boundary)"
        ),
    ),
    "dummy": ThrowawaySpec(
        model_type="dummy",
        model_name="throwaway-dummy",
        classifier=DummyClassifier,
        params={"strategy": "prior", "random_state": RANDOM_SEED},
        description=(
            "a DummyClassifier predicting the class prior (no learning at all)"
        ),
    ),
}

#: Declared model types, in registration order.
THROWAWAY_MODEL_TYPES: tuple[str, ...] = tuple(THROWAWAY_SPECS)

#: Backwards-friendly alias for the default model name.
THROWAWAY_MODEL_NAME: str = THROWAWAY_SPECS[DEFAULT_MODEL_TYPE].model_name

#: Backwards-friendly alias for the default model params.
THROWAWAY_PARAMS: dict[str, object] = dict(
    THROWAWAY_SPECS[DEFAULT_MODEL_TYPE].params
)

#: The split version consumed by default (the S01 versioned split).
THROWAWAY_SPLIT_VERSION: str = SPLIT_VERSION


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThrowawayResult:
    """Everything produced by one throwaway-model run."""

    model_type: str
    model_name: str
    split_version: str
    train_rows: int
    test_rows: int
    params: dict[str, object]
    metrics: dict[str, object]
    run_id: str | None
    experiment_name: str
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "split_version": self.split_version,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Model construction / training
# ---------------------------------------------------------------------------


def resolve_spec(model_type: str) -> ThrowawaySpec:
    """Return the declared spec for ``model_type`` or raise loudly."""
    if not isinstance(model_type, str) or not model_type.strip():
        raise UnknownThrowawayModelError(
            f"model_type must be a non-empty string, got {model_type!r}."
        )
    key = model_type.strip()
    try:
        return THROWAWAY_SPECS[key]
    except KeyError as exc:
        raise UnknownThrowawayModelError(
            f"Unknown throwaway model type {key!r}; declared types are "
            f"{list(THROWAWAY_MODEL_TYPES)}."
        ) from exc


def build_throwaway_model(
    model_type: str = DEFAULT_MODEL_TYPE,
    *,
    params: dict[str, object] | None = None,
) -> Pipeline:
    """Build the unfitted throwaway pipeline: shared preprocessing + classifier.

    The preprocessing chain is the same declared, leakage-safe chain the
    baseline uses; only the classifier differs. Passing ``params`` overrides
    the spec's declared hyperparameters (used by tests to pin behaviour).
    """
    spec = resolve_spec(model_type)
    resolved = dict(spec.params if params is None else params)
    classifier = spec.classifier(**resolved)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", classifier),
        ]
    )


def train_throwaway(
    model_type: str = DEFAULT_MODEL_TYPE,
    train_frame: pd.DataFrame | None = None,
    *,
    params: dict[str, object] | None = None,
) -> Pipeline:
    """Fit a throwaway model on the training rows of the S01 split.

    Only ``train_frame`` is read; the held-out rows are never seen here.
    """
    spec = resolve_spec(model_type)
    if train_frame is None:
        raise ThrowawayDataError(
            "train_throwaway requires a training frame; load the S01 split "
            "with heart.data.split.load_split_frames."
        )
    missing = [
        column
        for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
        if column not in train_frame.columns
    ]
    if missing:
        raise ThrowawayDataError(
            f"Training frame is missing column(s) {missing}; load the S01 split "
            "with heart.data.split.load_split_frames."
        )
    model = build_throwaway_model(model_type, params=params)
    model.fit(train_frame[list(FEATURE_COLUMNS)], train_frame[TARGET_COLUMN])
    logger.info(
        "Trained throwaway model %s (%s) on %d training row(s): %s",
        spec.model_name,
        spec.description,
        len(train_frame),
        {key: value for key, value in (params or spec.params).items()},
    )
    return model


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_throwaway(
    *,
    model_type: str = DEFAULT_MODEL_TYPE,
    split_version: str = THROWAWAY_SPLIT_VERSION,
    tracking_dir: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    log_to_mlflow: bool = True,
) -> ThrowawayResult:
    """Train, evaluate, and log one throwaway model through the shared path."""
    spec = resolve_spec(model_type)
    train_frame, test_frame = load_split_frames(split_version)

    model = train_throwaway(model_type, train_frame)
    split = evaluation_split_from_frames(test_frame, name=f"{split_version}/test")
    metrics = evaluate(model, split)

    run: RunResult | None = None
    if log_to_mlflow:
        run = log_evaluation_run(
            metrics,
            model_name=spec.model_name,
            split_version=split_version,
            params=spec.params,
            tags={"slice": "S02", "role": "model-agnostic-proof"},
            experiment_name=experiment_name,
            tracking_dir=tracking_dir,
        )

    result = ThrowawayResult(
        model_type=spec.model_type,
        model_name=spec.model_name,
        split_version=split_version,
        train_rows=int(len(train_frame)),
        test_rows=int(len(test_frame)),
        params=dict(spec.params),
        metrics=metrics,
        run_id=run.run_id if run else None,
        experiment_name=experiment_name,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    logger.info(
        "Throwaway %s on %s: %s=%.4f accuracy=%.4f (run=%s)",
        spec.model_name,
        split_version,
        PRIMARY_METRIC,
        float(metrics[PRIMARY_METRIC]),
        float(metrics["accuracy"]),
        run.run_id if run else "not logged",
    )
    return result


def run_all_throwaway(
    *,
    split_version: str = THROWAWAY_SPLIT_VERSION,
    tracking_dir: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    log_to_mlflow: bool = True,
) -> tuple[ThrowawayResult, ...]:
    """Run **every** declared throwaway model through the identical path.

    This is the executable form of the model-agnostic claim: one loop, one
    call to :func:`evaluate` per model, one MLflow run per model, and — as the
    tests assert — one identical metric-key shape across all of them.
    """
    results = tuple(
        run_throwaway(
            model_type=model_type,
            split_version=split_version,
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
            log_to_mlflow=log_to_mlflow,
        )
        for model_type in THROWAWAY_MODEL_TYPES
    )
    shapes = {tuple(result.metrics.keys()) for result in results}
    if len(shapes) != 1:
        raise ThrowawayError(
            "Model-agnostic proof failed: throwaway model types produced "
            f"differently-shaped metric dicts: {sorted(shapes)}."
        )
    logger.info(
        "Model-agnostic proof: %d throwaway model type(s) produced the same "
        "%d-key metric dict: %s",
        len(results),
        len(next(iter(shapes))),
        list(THROWAWAY_MODEL_TYPES),
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.models.throwaway",
        description=(
            "Run throwaway models through the shared evaluate() contract to "
            "prove it is model-agnostic."
        ),
    )
    parser.add_argument(
        "--model-type",
        default=DEFAULT_MODEL_TYPE,
        choices=list(THROWAWAY_MODEL_TYPES),
        help=f"Throwaway model to run (default: {DEFAULT_MODEL_TYPE}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every declared throwaway model type through the same path.",
    )
    parser.add_argument(
        "--split-version", default=THROWAWAY_SPLIT_VERSION, help="S01 split version."
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
        "--no-mlflow", action="store_true", help="Skip MLflow logging."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.all:
        results = run_all_throwaway(
            split_version=args.split_version,
            tracking_dir=args.tracking_dir,
            experiment_name=args.experiment,
            log_to_mlflow=not args.no_mlflow,
        )
    else:
        results = (
            run_throwaway(
                model_type=args.model_type,
                split_version=args.split_version,
                tracking_dir=args.tracking_dir,
                experiment_name=args.experiment,
                log_to_mlflow=not args.no_mlflow,
            ),
        )

    shapes = {tuple(result.metrics.keys()) for result in results}
    for result in results:
        metrics = result.metrics
        print(f"model: {result.model_name} ({result.model_type})")
        print(f"split: {result.split_version} (train {result.train_rows} / test {result.test_rows})")
        print(f"params: {result.params}")
        print(f"MLflow run: {result.run_id}")
        print(
            f"headline: {PRIMARY_METRIC}={float(metrics[PRIMARY_METRIC]):.4f} "
            f"accuracy={float(metrics['accuracy']):.4f} "
            f"pr_auc={float(metrics['pr_auc']):.4f}"
        )
        print(f"metric-dict keys: {len(metrics)}")
    print(f"identical metric-dict shape across models: {len(shapes) == 1}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
