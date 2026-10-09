"""The generic Optuna tuning runner with MLflow trial logging (S03/T02).

This module owns **what is optimised** for a battery member: a K-fold
cross-validation of the model spec's search space, scored through the single
shared evaluation contract. It is deliberately model-agnostic — it imports
:func:`heart.models.registry.build_pipeline` and never switches on model
identity, so adding a member to the battery requires no runner change.

Protocol
--------
1. :func:`build_cv_folds` splits a training frame into group-aware, leakage-safe
   cross-validation folds using the S01 splitter, so a near-duplicate cluster
   can never straddle a fold boundary.
2. :class:`ClassicalTuningObjective` samples one parameter set per trial and,
   for each fold, fits a fresh pipeline (preprocessing included) on that fold's
   training rows and scores the fold's held-out rows with
   :func:`heart.eval.contract.evaluate`. After each fold it reports the running
   mean primary metric to Optuna and honours ``trial.should_prune()``, so the
   configured pruner sees a real intermediate series.
3. The trial's value is the mean primary metric (ROC-AUC) across folds; the
   concatenated out-of-fold predictions produce one canonical metric dict per
   completed trial, which is logged to MLflow through the frozen convention
   (:func:`heart.tracking.run.log_evaluation_run`).

Recording trials and failures
-----------------------------
Every trial outcome is recorded, never dropped:

* a **complete** trial is logged as an MLflow run carrying its sampled params
  and its canonical out-of-fold metric dict (``metrics.json`` +
  ``run_config.json``), tagged ``run_kind=trial``;
* a **pruned** trial is logged as a convention-tagged run with its params, the
  primary metric of each completed fold, and ``run_kind=pruned``;
* a **failed** trial is logged as a convention-tagged run with its params, the
  exception message, and ``run_kind=failed``.

Trial runs live in a dedicated tuning experiment
(:data:`DEFAULT_TUNING_EXPERIMENT`) so the portfolio experiment keeps exactly
one final run per model for T03/T04; the leaderboard filters on
:data:`RUN_KIND_TAG` == :data:`FINAL_RUN_KIND`. A durable JSON **tuning ledger**
(:func:`write_tuning_ledger`) lists every trial with its state, params, value,
fold scores, and error, and is the audit artifact for a study.

Observability
-------------
:func:`run_study` logs one line per trial (state, value, run id) and a summary
line at the end. The ledger is the persisted failure state; the failed/pruned
MLflow runs carry the same information in the tracking store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised by the environment, not by logic
    import mlflow
    import optuna
    from mlflow.exceptions import MlflowException
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "heart.tuning requires Optuna and MLflow, which are not installed. "
        'Install the tuning extra with: pip install -e ".[ml]"'
    ) from exc

from heart.config import RANDOM_SEED
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, create_split
from heart.eval.contract import (
    METRIC_KEYS,
    PRIMARY_METRIC,
    EvaluationSplit,
    evaluate,
    evaluation_split_from_frames,
    validate_metric_dict,
)
from heart.eval.metrics import compute_metric_dict
from heart.models.registry import (
    ModelSpec,
    RegistryError,
    build_pipeline,
    resolve_spec,
)
from heart.tracking.mlflow_store import TrackingConfig, configure_tracking
from heart.tracking.run import (
    TrackingError,
    build_run_name,
    log_evaluation_run,
    normalise_params,
    slugify,
)
from heart.runtime import atomic_write_text, json_default, write_json_document
from heart.tuning.study import (
    TuningConfig,
    TuningError,
    create_study,
    suggest_params,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CV_FOLDS",
    "DEFAULT_TUNING_EXPERIMENT",
    "RUN_KIND_TAG",
    "RUN_KIND_TRIAL",
    "RUN_KIND_PRUNED",
    "RUN_KIND_FAILED",
    "FINAL_RUN_KIND",
    "LEDGER_FILENAME",
    "TuningRunnerError",
    "TuningDataError",
    "TuningObjectiveError",
    "NoCompletedTrialError",
    "TuningLedgerError",
    "TuningFold",
    "TrialRecord",
    "TuningResult",
    "ClassicalTuningObjective",
    "build_cv_folds",
    "run_study",
    "write_tuning_ledger",
    "tuning_ledger",
    "build_best_pipeline",
]

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Default number of cross-validation folds used for tuning.
DEFAULT_CV_FOLDS: int = 5

#: Dedicated MLflow experiment holding tuning-trial runs (never final runs).
DEFAULT_TUNING_EXPERIMENT: str = "heart-failure-prediction-tuning"

#: Tag distinguishing trial-level runs from final per-model runs.
RUN_KIND_TAG: str = "run_kind"

#: A completed Optuna trial with its out-of-fold metric dict.
RUN_KIND_TRIAL: str = "trial"

#: A trial the pruner stopped early.
RUN_KIND_PRUNED: str = "pruned"

#: A trial whose objective raised.
RUN_KIND_FAILED: str = "failed"

#: The run kind T03/T04 use for the single final run per model.
FINAL_RUN_KIND: str = "final"

#: Conventional MLflow tag naming the model registry key.
MODEL_TYPE_TAG: str = "model_type"

#: Conventional MLflow tag recording the Optuna trial number.
TRIAL_NUMBER_TAG: str = "trial_number"

#: Conventional MLflow tag recording the study name.
STUDY_NAME_TAG: str = "study_name"

#: Conventional MLflow tag recording the trial state.
TRIAL_STATE_TAG: str = "trial_state"

#: Conventional MLflow tag recording a failed trial's exception message.
ERROR_TAG: str = "error"

#: Default ledger filename written next to a study.
LEDGER_FILENAME: str = "tuning_ledger.json"

#: MLflow tag values are capped; truncate long error messages.
MAX_ERROR_TAG_LENGTH: int = 4000

#: Optuna trial-state names mapped to the runner's lowercase vocabulary.
_STATE_NAMES: dict[str, str] = {
    "COMPLETE": "complete",
    "PRUNED": "pruned",
    "FAIL": "failed",
    "RUNNING": "running",
    "WAITING": "waiting",
}


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class TuningRunnerError(TuningError):
    """Base class for every tuning-run failure."""


class TuningDataError(TuningRunnerError):
    """A frame or fold handed to the runner is malformed."""


class TuningObjectiveError(TuningRunnerError):
    """The tuning objective could not build, fit, or score a model."""


class NoCompletedTrialError(TuningRunnerError):
    """The study finished without a single completed trial."""


class TuningLedgerError(TuningRunnerError):
    """The tuning ledger could not be serialised or written."""


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def _require_frame(frame: object, *, role: str, require_target: bool = True) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TuningDataError(
            f"{role} must be a pandas.DataFrame, got {type(frame).__name__}."
        )
    if frame.empty:
        raise TuningDataError(f"{role} is empty; cannot tune on zero rows.")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if require_target and TARGET_COLUMN not in frame.columns:
        missing.append(TARGET_COLUMN)
    if missing:
        raise TuningDataError(
            f"{role} is missing declared column(s) {missing}; load the S01 "
            "split with heart.data.split.load_split_frames."
        )
    return frame


@dataclass(frozen=True)
class TuningFold:
    """One cross-validation fold: training rows plus a held-out eval split."""

    fold: int
    train_frame: pd.DataFrame
    split: EvaluationSplit

    def __post_init__(self) -> None:
        if isinstance(self.fold, bool) or not isinstance(self.fold, int):
            raise TuningDataError(f"fold must be an integer, got {self.fold!r}.")
        if self.fold < 0:
            raise TuningDataError(f"fold must be >= 0, got {self.fold!r}.")
        _require_frame(self.train_frame, role=f"fold {self.fold} train_frame")
        if not isinstance(self.split, EvaluationSplit):
            raise TuningDataError(
                f"fold {self.fold} split must be an EvaluationSplit, got "
                f"{type(self.split).__name__}."
            )

    @property
    def n_train(self) -> int:
        return int(len(self.train_frame))

    @property
    def n_eval(self) -> int:
        return int(self.split.n_samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": int(self.fold),
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "split": self.split.to_dict(),
        }


def build_cv_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = DEFAULT_CV_FOLDS,
    random_state: int = RANDOM_SEED,
    version: str = SPLIT_VERSION,
) -> tuple[TuningFold, ...]:
    """Build group-aware cross-validation folds from a training frame.

    Uses the S01 :func:`heart.data.split.create_split` for each fold, so the
    folds are stratified, near-duplicate groups stay whole, and no held-out row
    leaks into a fold's training rows.
    """
    _require_frame(frame, role="tuning frame")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise TuningDataError(f"n_folds must be an integer >= 2, got {n_folds!r}.")
    if n_folds > len(frame):
        raise TuningDataError(
            f"n_folds ({n_folds}) exceeds the number of rows ({len(frame)})."
        )
    folds: list[TuningFold] = []
    for test_fold in range(n_folds):
        data_split = create_split(
            frame,
            version=f"{version}-fold{test_fold}",
            random_state=random_state,
            n_splits=n_folds,
            test_fold=test_fold,
        )
        train_frame = data_split.train_frame(frame)
        test_frame = data_split.test_frame(frame)
        split = evaluation_split_from_frames(
            test_frame, name=f"{version}/fold{test_fold}"
        )
        folds.append(
            TuningFold(fold=test_fold, train_frame=train_frame, split=split)
        )
    logger.info(
        "Built %d cross-validation fold(s) from %d row(s): eval sizes=%s",
        len(folds),
        len(frame),
        [fold.n_eval for fold in folds],
    )
    return tuple(folds)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


class ClassicalTuningObjective:
    """Optuna objective: K-fold CV of one model spec through the shared contract.

    The objective is callable with an Optuna trial. It returns the mean primary
    metric (ROC-AUC) across folds, records the per-fold scores, and stores the
    canonical out-of-fold metric dict for completed trials in
    :attr:`metric_dicts` so the runner can log it under the frozen convention.
    """

    def __init__(
        self,
        spec: ModelSpec,
        folds: Sequence[TuningFold],
        *,
        primary_metric: str = PRIMARY_METRIC,
        validate: bool = True,
    ) -> None:
        if not isinstance(spec, ModelSpec):
            raise TuningDataError(
                f"spec must be a ModelSpec, got {type(spec).__name__}."
            )
        self.spec = spec
        self.folds: tuple[TuningFold, ...] = tuple(folds)
        if not self.folds:
            raise TuningDataError(
                "The tuning objective needs at least one cross-validation fold."
            )
        for fold in self.folds:
            if not isinstance(fold, TuningFold):
                raise TuningDataError(
                    f"folds must contain TuningFold objects, got "
                    f"{type(fold).__name__}."
                )
        if not isinstance(primary_metric, str) or not primary_metric.strip():
            raise TuningDataError(
                f"primary_metric must be a non-empty string, got "
                f"{primary_metric!r}."
            )
        if primary_metric not in METRIC_KEYS:
            raise TuningDataError(
                f"primary_metric {primary_metric!r} is not part of the canonical "
                f"metric schema {list(METRIC_KEYS)}."
            )
        self.primary_metric = primary_metric
        self.validate = bool(validate)
        #: Canonical out-of-fold metric dict per completed trial number.
        self.metric_dicts: dict[int, dict[str, object]] = {}
        #: Per-fold primary-metric scores per trial number (complete or pruned).
        self.fold_scores: dict[int, tuple[float, ...]] = {}

    # -- internals ---------------------------------------------------------

    def _fit_and_score(
        self, params: Mapping[str, object], fold: TuningFold
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
        try:
            model = build_pipeline(self.spec, params)
            model.fit(
                fold.train_frame[list(FEATURE_COLUMNS)],
                fold.train_frame[TARGET_COLUMN],
            )
            metrics = evaluate(model, fold.split, validate=self.validate)
            predictions = np.asarray(model.predict(fold.split.X_test)).ravel()
            probabilities = np.asarray(
                model.predict_proba(fold.split.X_test), dtype=float
            )[:, 1]
        except (RegistryError, TuningRunnerError):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a named error
            raise TuningObjectiveError(
                f"Failed to build/fit/score {self.spec.model_type!r} on fold "
                f"{fold.fold} with params {dict(params)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return metrics, predictions, probabilities

    def _run(self, trial: object) -> float:
        params = suggest_params(self.spec.space, trial)
        scores: list[float] = []
        out_of_fold_labels: list[np.ndarray] = []
        out_of_fold_pred: list[np.ndarray] = []
        out_of_fold_proba: list[np.ndarray] = []
        number = int(getattr(trial, "number", -1))

        for step, fold in enumerate(self.folds):
            metrics, predictions, probabilities = self._fit_and_score(params, fold)
            scores.append(float(metrics[self.primary_metric]))
            out_of_fold_labels.append(np.asarray(fold.split.y_test))
            out_of_fold_pred.append(predictions)
            out_of_fold_proba.append(probabilities)

            # Intermediate value for the pruner: mean primary metric so far.
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                self.fold_scores[number] = tuple(scores)
                raise optuna.TrialPruned(
                    f"{self.spec.model_type} pruned after {step + 1} fold(s) "
                    f"(mean {self.primary_metric}={np.mean(scores):.4f})."
                )

        self.fold_scores[number] = tuple(scores)

        oof_metrics = compute_metric_dict(
            np.concatenate(out_of_fold_labels),
            np.concatenate(out_of_fold_proba),
            np.concatenate(out_of_fold_pred),
        )
        if self.validate:
            validate_metric_dict(oof_metrics)
        self.metric_dicts[number] = oof_metrics
        trial.set_user_attr("fold_scores", [float(score) for score in scores])
        trial.set_user_attr("mean_score", float(np.mean(scores)))
        return float(np.mean(scores))

    def __call__(self, trial: object) -> float:
        try:
            return self._run(trial)
        except optuna.TrialPruned:
            trial.set_user_attr("folds_completed", len(self.fold_scores.get(trial.number, ())))
            raise
        except Exception as exc:  # noqa: BLE001 - record, then let optuna mark FAIL
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise


# ---------------------------------------------------------------------------
# Trial records / study result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRecord:
    """One recorded Optuna trial outcome (never dropped)."""

    number: int
    state: str
    params: dict[str, object]
    value: float | None
    fold_scores: tuple[float, ...]
    error: str | None
    run_id: str | None
    duration_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "number": int(self.number),
            "state": self.state,
            "params": dict(self.params),
            "value": None if self.value is None else float(self.value),
            "fold_scores": [float(score) for score in self.fold_scores],
            "error": self.error,
            "run_id": self.run_id,
            "duration_seconds": None
            if self.duration_seconds is None
            else float(self.duration_seconds),
        }


@dataclass(frozen=True)
class TuningResult:
    """Everything one tuning study produced."""

    model_type: str
    model_name: str
    study_name: str
    direction: str
    n_trials: int
    best_trial_number: int | None
    best_value: float | None
    best_params: dict[str, object] | None
    metric_dict: dict[str, object] | None
    trials: tuple[TrialRecord, ...]
    run_ids: tuple[str, ...] = ()
    experiment_name: str | None = None

    @property
    def n_complete(self) -> int:
        return sum(1 for trial in self.trials if trial.state == "complete")

    @property
    def n_pruned(self) -> int:
        return sum(1 for trial in self.trials if trial.state == "pruned")

    @property
    def n_failed(self) -> int:
        return sum(1 for trial in self.trials if trial.state == "failed")

    def best_record(self) -> TrialRecord | None:
        """Return the best completed :class:`TrialRecord` (or ``None``)."""
        completed = [trial for trial in self.trials if trial.state == "complete"]
        if not completed:
            return None
        if self.direction == "minimize":
            return min(completed, key=lambda trial: float(trial.value))
        return max(completed, key=lambda trial: float(trial.value))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "study_name": self.study_name,
            "direction": self.direction,
            "n_trials": int(self.n_trials),
            "n_complete": self.n_complete,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "best_trial_number": self.best_trial_number,
            "best_value": None if self.best_value is None else float(self.best_value),
            "best_params": None if self.best_params is None else dict(self.best_params),
            "metric_dict": self.metric_dict,
            "run_ids": list(self.run_ids),
            "experiment_name": self.experiment_name,
            "trials": [trial.to_dict() for trial in self.trials],
        }


# ---------------------------------------------------------------------------
# MLflow trial logging
# ---------------------------------------------------------------------------


def _merged_params(
    spec: ModelSpec, tuned: Mapping[str, object] | None
) -> dict[str, object]:
    """Effective hyperparameters: pinned fixed params plus the tuned ones."""
    merged = dict(spec.fixed_params)
    if tuned:
        merged.update(tuned)
    return merged


def _trial_run_name(spec: ModelSpec, split_version: str, number: int) -> str:
    return f"{build_run_name(spec.model_name, split_version)}-trial-{number:03d}"


def _trial_tags(
    *,
    spec: ModelSpec,
    split_version: str,
    study_name: str,
    number: int,
    run_kind: str,
    state: str,
    n_folds: int,
    error: str | None,
) -> dict[str, str]:
    tags = {
        RUN_KIND_TAG: run_kind,
        TRIAL_STATE_TAG: state,
        MODEL_TYPE_TAG: spec.model_type,
        TRIAL_NUMBER_TAG: str(number),
        STUDY_NAME_TAG: study_name,
        "model_name": spec.model_name,
        "model_slug": slugify(spec.model_name),
        "split_version": str(split_version),
        "cv_folds": str(int(n_folds)),
    }
    if error:
        tags[ERROR_TAG] = str(error)[:MAX_ERROR_TAG_LENGTH]
    return tags


def _log_trial_outcome(
    *,
    config: TrackingConfig,
    spec: ModelSpec,
    split_version: str,
    study_name: str,
    number: int,
    state: str,
    run_kind: str,
    params: Mapping[str, object],
    error: str | None,
    fold_scores: Sequence[float],
    primary_metric: str,
    n_folds: int,
) -> str:
    """Log a failed or pruned trial as a convention-tagged MLflow run.

    These trials have no canonical metric dict (pruning stops evaluation early;
    a failure stops it outright), so they cannot go through
    :func:`log_evaluation_run`'s metric-contract path. They are still recorded
    as first-class runs: the sampled params, the state, the error message, and
    the primary metric of each completed fold.
    """
    run_name = _trial_run_name(spec, split_version, number)
    tags = _trial_tags(
        spec=spec,
        split_version=split_version,
        study_name=study_name,
        number=number,
        run_kind=run_kind,
        state=state,
        n_folds=n_folds,
        error=error,
    )
    partial = {
        f"fold_{index}_{primary_metric}": float(score)
        for index, score in enumerate(fold_scores)
    }
    normalised = normalise_params(_merged_params(spec, params))
    try:
        with mlflow.start_run(
            run_name=run_name, experiment_id=config.experiment_id
        ) as run:
            run_id = run.info.run_id
            if normalised:
                mlflow.log_params(normalised)
            mlflow.set_tags(tags)
            if partial:
                mlflow.log_metrics(partial)
    except (MlflowException, OSError) as exc:
        raise TuningRunnerError(
            f"Could not log {state} trial {number} of {spec.model_type!r} to "
            f"experiment {config.experiment_name!r}: {exc}"
        ) from exc
    logger.info(
        "Recorded %s trial %d of %s (mean=%s run_id=%s)",
        state,
        number,
        spec.model_type,
        f"{float(np.mean(fold_scores)):.4f}" if fold_scores else "n/a",
        run_id,
    )
    return run_id


def _record_trial(
    trial: object,
    *,
    spec: ModelSpec,
    study_name: str,
    split_version: str,
    primary_metric: str,
    n_folds: int,
    objective: ClassicalTuningObjective,
    tracking_config: TrackingConfig | None,
) -> TrialRecord:
    """Build a :class:`TrialRecord`, logging the trial to MLflow when enabled."""
    number = int(trial.number)
    state = _STATE_NAMES.get(trial.state.name, trial.state.name.lower())
    params = dict(trial.params)
    value = None if trial.value is None else float(trial.value)
    fold_scores = objective.fold_scores.get(number, ())
    error = trial.user_attrs.get("error")
    duration = trial.duration.total_seconds() if trial.duration is not None else None
    run_id: str | None = None

    if tracking_config is not None:
        if state == "complete" and number in objective.metric_dicts:
            metrics = objective.metric_dicts[number]
            tags = _trial_tags(
                spec=spec,
                split_version=split_version,
                study_name=study_name,
                number=number,
                run_kind=RUN_KIND_TRIAL,
                state=state,
                n_folds=n_folds,
                error=None,
            )
            try:
                run = log_evaluation_run(
                    metrics,
                    model_name=spec.model_name,
                    split_version=split_version,
                    params=_merged_params(spec, params),
                    tags=tags,
                    run_name=_trial_run_name(spec, split_version, number),
                    config=tracking_config,
                )
            except TrackingError as exc:
                raise TuningRunnerError(
                    f"Could not log completed trial {number} of "
                    f"{spec.model_type!r}: {exc}"
                ) from exc
            run_id = run.run_id
        elif state in {"failed", "pruned"}:
            run_id = _log_trial_outcome(
                config=tracking_config,
                spec=spec,
                split_version=split_version,
                study_name=study_name,
                number=number,
                state=state,
                run_kind=RUN_KIND_FAILED if state == "failed" else RUN_KIND_PRUNED,
                params=params,
                error=error,
                fold_scores=fold_scores,
                primary_metric=primary_metric,
                n_folds=n_folds,
            )

    logger.info(
        "Trial %d of %s: state=%s value=%s folds=%d run_id=%s",
        number,
        spec.model_type,
        state,
        "n/a" if value is None else f"{value:.4f}",
        len(fold_scores),
        run_id or "not logged",
    )
    return TrialRecord(
        number=number,
        state=state,
        params=params,
        value=value,
        fold_scores=tuple(float(score) for score in fold_scores),
        error=None if error is None else str(error),
        run_id=run_id,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_spec(model: str | ModelSpec) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    try:
        return resolve_spec(model)
    except RegistryError:
        raise
    except TypeError as exc:
        raise TuningDataError(
            f"model must be a model_type string or ModelSpec, got "
            f"{type(model).__name__}."
        ) from exc


def run_study(
    model: str | ModelSpec,
    folds: Sequence[TuningFold],
    *,
    config: TuningConfig | None = None,
    primary_metric: str = PRIMARY_METRIC,
    sampler: object | None = None,
    pruner: object | None = None,
    log_to_mlflow: bool = True,
    experiment_name: str = DEFAULT_TUNING_EXPERIMENT,
    tracking_dir: str | Path | None = None,
    tracking_config: TrackingConfig | None = None,
    split_version: str = SPLIT_VERSION,
    ledger_path: str | Path | None = None,
    objective: ClassicalTuningObjective | None = None,
) -> TuningResult:
    """Tune ``model`` over ``folds`` and record every trial.

    Parameters
    ----------
    model:
        A battery ``model_type`` or a :class:`ModelSpec` (fixture models).
    folds:
        The cross-validation folds from :func:`build_cv_folds` (or hand-built).
    config:
        The :class:`~heart.tuning.study.TuningConfig`; defaults to the declared
        configuration.
    sampler / pruner:
        Optional prebuilt Optuna sampler/pruner that override the config.
    log_to_mlflow:
        When ``True`` (default) each trial is logged to the tuning experiment.
    experiment_name / tracking_dir:
        MLflow experiment and store for trial runs. The experiment is created
        on first use and is idempotent thereafter.
    tracking_config:
        A pre-resolved :class:`TrackingConfig` (skips store configuration).
    split_version:
        Label recorded on trial runs; use the portfolio split version.
    ledger_path:
        When given, the durable tuning ledger JSON is written here atomically.
    objective:
        A pre-built objective overriding the default
        :class:`ClassicalTuningObjective` — used by the S05/T03 neural sweep,
        which swaps only the per-fold scoring leaf (:class:`MLPTuningObjective`
        with its torched trainer) while inheriting every other behaviour
        (fold loop, pruning, out-of-fold metric dict, MLflow trial recording).
        The object must behave like :class:`ClassicalTuningObjective` (expose
        ``__call__(trial)`` returning the mean primary metric and accumulate
        ``fold_scores`` / ``metric_dicts`` the trial recorder reads). When
        supplied, ``primary_metric`` is that objective's own concern.

    Returns
    -------
    TuningResult
        Best params, the best trial's out-of-fold metric dict, every trial
        record, and the MLflow run ids.
    """
    spec = _resolve_spec(model)
    resolved_config = config or TuningConfig()
    resolved_folds = tuple(folds)
    if not resolved_folds:
        raise TuningDataError("run_study needs at least one fold.")

    if objective is not None and not isinstance(objective, ClassicalTuningObjective):
        raise TuningDataError(
            f"objective must be a ClassicalTuningObjective (or subclass), got "
            f"{type(objective).__name__}."
        )
    objective = objective or ClassicalTuningObjective(
        spec, resolved_folds, primary_metric=primary_metric
    )
    study = create_study(
        spec, resolved_config, sampler=sampler, pruner=pruner  # type: ignore[arg-type]
    )

    resolved_tracking = tracking_config
    if log_to_mlflow and resolved_tracking is None:
        resolved_tracking = configure_tracking(
            experiment_name=experiment_name, tracking_dir=tracking_dir
        )

    records: list[TrialRecord] = []

    def _on_trial(study: "optuna.Study", trial: object) -> None:
        records.append(
            _record_trial(
                trial,
                spec=spec,
                study_name=study.study_name,
                split_version=split_version,
                primary_metric=primary_metric,
                n_folds=len(resolved_folds),
                objective=objective,
                tracking_config=resolved_tracking,
            )
        )

    study.optimize(
        objective,
        n_trials=resolved_config.n_trials,
        timeout=resolved_config.timeout,
        catch=(Exception,),
        callbacks=[_on_trial],
        gc_after_trial=False,
        show_progress_bar=False,
    )

    best = _best_record(records, direction=resolved_config.direction)
    metric_dict = (
        objective.metric_dicts.get(best.number) if best is not None else None
    )
    result = TuningResult(
        model_type=spec.model_type,
        model_name=spec.model_name,
        study_name=study.study_name,
        direction=resolved_config.direction,
        n_trials=len(records),
        best_trial_number=best.number if best is not None else None,
        best_value=best.value if best is not None else None,
        best_params=dict(best.params) if best is not None else None,
        metric_dict=metric_dict,
        trials=tuple(records),
        run_ids=tuple(
            record.run_id for record in records if record.run_id is not None
        ),
        experiment_name=(
            resolved_tracking.experiment_name if resolved_tracking is not None else None
        ),
    )

    if ledger_path is not None:
        write_tuning_ledger(result, ledger_path)

    logger.info(
        "Study '%s' for %s finished: %d trial(s) — complete=%d pruned=%d "
        "failed=%d best=%s",
        result.study_name,
        spec.model_type,
        result.n_trials,
        result.n_complete,
        result.n_pruned,
        result.n_failed,
        "n/a"
        if result.best_value is None
        else f"#{result.best_trial_number} {primary_metric}={result.best_value:.4f}",
    )
    return result


def _best_record(
    records: Sequence[TrialRecord], *, direction: str
) -> TrialRecord | None:
    completed = [record for record in records if record.state == "complete"]
    if not completed:
        return None
    if direction == "minimize":
        return min(completed, key=lambda record: float(record.value))
    return max(completed, key=lambda record: float(record.value))


# ---------------------------------------------------------------------------
# Ledger + best-model hand-off
# ---------------------------------------------------------------------------


def tuning_ledger(result: TuningResult) -> dict[str, object]:
    """Return the study's durable ledger payload (JSON-serialisable)."""
    if not isinstance(result, TuningResult):
        raise TuningLedgerError(
            f"result must be a TuningResult, got {type(result).__name__}."
        )
    return result.to_dict()


def write_tuning_ledger(
    result: TuningResult, path: str | Path
) -> Path:
    """Atomically write the tuning ledger JSON for ``result`` to ``path``."""
    destination = write_json_document(
        path,
        tuning_ledger(result),
        error_factory=TuningLedgerError,
        label=f"tuning ledger for {result.model_type!r}",
    )
    logger.info(
        "Wrote tuning ledger for %s to %s (%d trial(s))",
        result.model_type,
        destination,
        len(result.trials),
    )
    return destination


def build_best_pipeline(
    result: TuningResult, spec: ModelSpec, train_frame: pd.DataFrame
):
    """Refit the best trial's params on ``train_frame`` and return the pipeline.

    This is the hand-off from tuning to final evaluation: the returned fitted
    pipeline is scored once on the held-out test split (through
    :func:`heart.eval.contract.evaluate`) and logged as the model's final run.
    """
    if not isinstance(result, TuningResult):
        raise TuningDataError(
            f"result must be a TuningResult, got {type(result).__name__}."
        )
    if not isinstance(spec, ModelSpec):
        raise TuningDataError(f"spec must be a ModelSpec, got {type(spec).__name__}.")
    if result.best_params is None:
        raise NoCompletedTrialError(
            f"Study {result.study_name!r} for {result.model_type!r} has no "
            "completed trial, so there is no best parameter set to refit."
        )
    frame = _require_frame(train_frame, role="best-pipeline train_frame")
    pipeline = build_pipeline(spec, result.best_params)
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_COLUMN])
    logger.info(
        "Refit best params of %s on %d row(s): %s",
        spec.model_type,
        len(frame),
        dict(result.best_params),
    )
    return pipeline
