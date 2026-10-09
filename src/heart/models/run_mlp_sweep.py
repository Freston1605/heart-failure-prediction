"""The tuned MLP architecture sweep through the shared protocol (S05/T03).

This module owns the **breadth** of the neural slice: it drives every
configured MLP architecture (:data:`heart.models.mlp_spaces.MLP_ARCHITECTURES`)
through the one shared workflow the classical battery uses — group-aware
cross-validation tuning through :func:`heart.tuning.runner.run_study`, a refit
of the winning parameters on the full training frame, a single evaluation on
the held-out test split via :func:`heart.eval.contract.evaluate` — and logs
exactly **one final MLflow run per architecture** to the portfolio experiment.

Why the existing runner
-----------------------
:class:`MLPTuningObjective` specialises :class:`ClassicalTuningObjective` by
swapping only the per-fold scoring leaf: instead of
``build_pipeline(spec, params)`` it calls a *trainer* (the production
:func:`heart.models.mlp_spaces.mlp_trainer`, or a deterministic stand-in in
tests) that trains the torch MLP on the fold's training rows and scores the
fold's held-out split. Everything else — the fold loop, pruning, the
out-of-fold metric dict, and the MLflow trial logging under the frozen
convention (run kind ``trial|pruned|failed`` in the dedicated tuning
experiment) — is inherited verbatim, so the neural trials are recorded by the
same code that records classical trials (decision D013).

Device is a first-class result
------------------------------
Each architecture's final MLflow run is tagged ``device`` / ``gpu_available``
/ ``cpu_fallback`` from the refitted model's
:class:`~heart.models.train_torch.DeviceResolution`, and the sweep ledger
repeats them. The GPU question — "which compute device did this neural run
use?" — is therefore answerable from the artifacts alone, whether the sweep
hit ``cuda:0`` in the ROCm container or fell back to CPU with a warning.

Fail-soft, never silent
-----------------------
A single architecture failing (torch unavailable on the host, a study with no
completed trial, a tracking error) is **recorded** as a failed
:class:`SweepModelResult` with a categorised error, and the sweep continues
with the next architecture. Pass ``strict=True`` to instead raise
:class:`SweepRunError` after the run is summarised. The sweep never drops an
architecture from its result.

Observability
-------------
Every architecture logs its status, trial counts, primary metric, device, and
MLflow run id at ``INFO`` plus a sweep summary line at the end.
:func:`render_sweep_summary` renders the markdown table the CLI prints, and
:func:`write_sweep_ledger` persists the machine-readable result JSON
atomically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from heart.config import RANDOM_SEED
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, SplitError, load_split_frames
from heart.eval.contract import (
    METRIC_KEYS,
    PRIMARY_METRIC,
    EvaluationSplit,
    MetricContractError,
    evaluate,
    evaluation_split_from_frames,
)
from heart.features.selection import (
    DEFAULT_LEDGER_PATH as DEFAULT_SELECTION_LEDGER_PATH,
)
from heart.models.mlp_spaces import (
    FAMILY_NEURAL,
    MLP_ARCHITECTURES,
    MLP_MODEL_TYPES,
    MLP_SWEEP_SIZE,
    MLPSpaceError,
    MLPSpec,
    mlp_trainer,
    select_mlp_architectures,
    to_model_spec,
)
from heart.models.registry import (
    ModelSpec,
    RegistryError,
)
from heart.models.train_torch import (
    SMOKE_EPOCHS,
    TorchTrainingError,
)
from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    TrackingConfig,
    TrackingError,
    configure_tracking,
)
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import (
    DEFAULT_CV_FOLDS,
    DEFAULT_TUNING_EXPERIMENT,
    FINAL_RUN_KIND,
    MODEL_TYPE_TAG,
    RUN_KIND_TAG,
    ClassicalTuningObjective,
    NoCompletedTrialError,
    TuningDataError,
    TuningFold,
    TuningResult,
    TuningRunnerError,
    build_best_pipeline,
    build_cv_folds,
    run_study,
)
from heart.tuning.study import (
    DEFAULT_DIRECTION,
    DEFAULT_N_STARTUP_TRIALS,
    DEFAULT_N_TRIALS,
    DEFAULT_N_WARMUP_STEPS,
    DEFAULT_PRUNER,
    DEFAULT_SAMPLER,
    TuningConfig,
    TuningError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SWEEP_EXPERIMENT",
    "SWEEP_LEDGER_FILENAME",
    "SMOKE_N_TRIALS",
    "SMOKE_CV_FOLDS",
    "SMOKE_EPOCHS",
    "STATUS_SUCCEEDED",
    "STATUS_FAILED",
    "ERROR_CATEGORY_DEPENDENCY",
    "ERROR_CATEGORY_NO_COMPLETED_TRIAL",
    "ERROR_CATEGORY_TUNING",
    "ERROR_CATEGORY_EVALUATION",
    "ERROR_CATEGORY_TRACKING",
    "ERROR_CATEGORY_REGISTRY",
    "ERROR_CATEGORY_UNEXPECTED",
    "SweepError",
    "SweepConfigError",
    "SweepDataError",
    "SweepRunError",
    "SweepLedgerError",
    "MLPTuningObjective",
    "SweepConfig",
    "SweepModelResult",
    "SweepResult",
    "smoke_sweep_config",
    "run_mlp_sweep",
    "sweep_ledger",
    "write_sweep_ledger",
    "render_sweep_summary",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: The portfolio experiment the final per-architecture runs belong to.
SWEEP_EXPERIMENT: str = DEFAULT_EXPERIMENT

#: Default filename for a persisted sweep ledger.
SWEEP_LEDGER_FILENAME: str = "mlp_sweep.json"

#: Reduced trial budget for the smoke configuration (still >= 1 per arch).
SMOKE_N_TRIALS: int = 2

#: Reduced cross-validation fold count for the smoke configuration.
SMOKE_CV_FOLDS: int = 3

#: Reduced epoch budget the smoke configuration applies to every architecture
#: (reuses train_torch's smoke epoch count so training stays cheap).
SMOKE_EPOCHS: int = 3

#: An architecture whose tuning completed, refit, evaluated, and logged.
STATUS_SUCCEEDED: str = "succeeded"

#: An architecture that failed somewhere in the workflow (still recorded).
STATUS_FAILED: str = "failed"

#: Error categories recorded on a failed architecture result.
ERROR_CATEGORY_DEPENDENCY: str = "dependency"
ERROR_CATEGORY_NO_COMPLETED_TRIAL: str = "no-completed-trial"
ERROR_CATEGORY_TUNING: str = "tuning"
ERROR_CATEGORY_EVALUATION: str = "evaluation"
ERROR_CATEGORY_TRACKING: str = "tracking"
ERROR_CATEGORY_REGISTRY: str = "registry"
ERROR_CATEGORY_UNEXPECTED: str = "unexpected"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class SweepError(Exception):
    """Base class for every MLP-sweep failure."""


class SweepConfigError(SweepError):
    """The requested sweep configuration is invalid."""


class SweepDataError(SweepError):
    """A frame or split handed to the sweep is malformed."""


class SweepRunError(SweepError):
    """Strict mode: at least one architecture failed (the result is attached)."""

    def __init__(self, message: str, result: "SweepResult") -> None:
        super().__init__(message)
        self.result = result

    @property
    def failed_architectures(self) -> tuple[str, ...]:
        return tuple(
            record.model_type
            for record in self.result.results
            if record.status == STATUS_FAILED
        )


class SweepLedgerError(SweepError):
    """The sweep ledger could not be serialised or written."""


# ---------------------------------------------------------------------------
# The neural tuning objective
# ---------------------------------------------------------------------------


class MLPTuningObjective(ClassicalTuningObjective):
    """Classical objective with the torch scoring leaf swapped in.

    Inherits the fold loop, pruning, out-of-fold metric-dict assembly, and
    trial bookkeeping from :class:`~heart.tuning.runner.ClassicalTuningObjective`;
    only :meth:`_fit_and_score` is overridden to train the MLP through the
    injected ``trainer`` callable instead of ``build_pipeline``. The trainer
    receives the *merged* parameter dict (fixed architecture + tuned knobs)
    and returns ``(metrics, predictions, probabilities)`` for one fold, so
    every completed trial still produces the canonical metric dict under the
    exact convention the classical battery uses.
    """

    def __init__(
        self,
        spec: ModelSpec,
        folds: Sequence[TuningFold],
        *,
        trainer: Callable[[Mapping[str, object], TuningFold],
                          tuple[dict[str, object], np.ndarray, np.ndarray]],
        primary_metric: str = PRIMARY_METRIC,
        validate: bool = True,
    ) -> None:
        super().__init__(spec, folds, primary_metric=primary_metric, validate=validate)
        if not callable(trainer):
            raise TuningDataError(
                f"trainer must be callable, got {type(trainer).__name__}."
            )
        self.trainer = trainer

    def _fit_and_score(
        self, params: Mapping[str, object], fold: TuningFold
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
        merged = dict(self.spec.fixed_params)
        merged.update(params)
        try:
            metrics, predictions, probabilities = self.trainer(merged, fold)
        except (RegistryError, TuningRunnerError):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a named error
            raise TuningRunnerError(
                f"MLP trial for {self.spec.model_type!r} failed on fold "
                f"{fold.fold} with params {dict(params)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return metrics, predictions, probabilities


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """The resolved, validated knobs for one MLP sweep run.

    Mirrors the classical :class:`~heart.models.run_battery.BatteryConfig`
    (validation delegated to :class:`TuningConfig`) plus
    ``smoke_epochs``: when set, every architecture trains with that reduced
    epoch budget so a smoke sweep stays cheap off the GPU.
    """

    n_trials: int = DEFAULT_N_TRIALS
    cv_folds: int = DEFAULT_CV_FOLDS
    sampler: str = DEFAULT_SAMPLER
    pruner: str = DEFAULT_PRUNER
    timeout: float | None = None
    direction: str = DEFAULT_DIRECTION
    sampler_seed: int = RANDOM_SEED
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS
    n_warmup_steps: int = DEFAULT_N_WARMUP_STEPS
    smoke_epochs: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cv_folds, bool) or not isinstance(self.cv_folds, int):
            raise SweepConfigError(
                f"cv_folds must be an integer, got {self.cv_folds!r}."
            )
        if self.cv_folds < 2:
            raise SweepConfigError(
                f"cv_folds must be >= 2, got {self.cv_folds!r}."
            )
        if self.smoke_epochs is not None:
            if isinstance(self.smoke_epochs, bool) or not isinstance(
                self.smoke_epochs, int
            ):
                raise SweepConfigError(
                    f"smoke_epochs must be a positive integer or None, got "
                    f"{self.smoke_epochs!r}."
                )
            if self.smoke_epochs < 1:
                raise SweepConfigError(
                    f"smoke_epochs must be >= 1, got {self.smoke_epochs!r}."
                )
        try:
            self.to_tuning_config()
        except TuningError as exc:
            raise SweepConfigError(
                f"Invalid sweep tuning configuration: {exc}"
            ) from exc

    def to_tuning_config(self, *, study_name: str | None = None) -> TuningConfig:
        """Build the :class:`TuningConfig` for one architecture's study."""
        return TuningConfig(
            n_trials=self.n_trials,
            timeout=self.timeout,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            sampler_seed=self.sampler_seed,
            n_startup_trials=self.n_startup_trials,
            n_warmup_steps=self.n_warmup_steps,
            study_name=study_name,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_trials": int(self.n_trials),
            "cv_folds": int(self.cv_folds),
            "sampler": self.sampler,
            "pruner": self.pruner,
            "timeout": None if self.timeout is None else float(self.timeout),
            "direction": self.direction,
            "sampler_seed": int(self.sampler_seed),
            "n_startup_trials": int(self.n_startup_trials),
            "n_warmup_steps": int(self.n_warmup_steps),
            "smoke_epochs": None if self.smoke_epochs is None else int(self.smoke_epochs),
        }


def smoke_sweep_config(
    *,
    n_trials: int = SMOKE_N_TRIALS,
    cv_folds: int = SMOKE_CV_FOLDS,
    smoke_epochs: int = SMOKE_EPOCHS,
) -> SweepConfig:
    """Return the reduced smoke configuration used by tests and the CLI.

    A random sampler and no pruner keep the smoke deterministic and fast; the
    folds are still group-aware and leakage-safe, so the smoke exercises the
    real protocol rather than a shortcut.
    """
    return SweepConfig(
        n_trials=n_trials,
        cv_folds=cv_folds,
        sampler="random",
        pruner="none",
        sampler_seed=RANDOM_SEED,
        n_startup_trials=1,
        n_warmup_steps=1,
        smoke_epochs=smoke_epochs,
    )


# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepModelResult:
    """One architecture's outcome within a sweep run (never dropped)."""

    model_type: str
    model_name: str
    family: str
    status: str
    duration_seconds: float
    run_id: str | None = None
    experiment_name: str | None = None
    metric_dict: dict[str, object] | None = None
    best_params: dict[str, object] | None = None
    n_trials: int = 0
    n_complete: int = 0
    n_pruned: int = 0
    n_failed_trials: int = 0
    device: str | None = None
    device_reason: str | None = None
    error: str | None = None
    error_type: str | None = None
    error_category: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == STATUS_SUCCEEDED

    @property
    def primary_metric(self) -> float | None:
        if self.metric_dict is None:
            return None
        return float(self.metric_dict[PRIMARY_METRIC])

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "family": self.family,
            "status": self.status,
            "duration_seconds": float(self.duration_seconds),
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "metric_dict": self.metric_dict,
            "best_params": None if self.best_params is None else dict(self.best_params),
            "n_trials": int(self.n_trials),
            "n_complete": int(self.n_complete),
            "n_pruned": int(self.n_pruned),
            "n_failed_trials": int(self.n_failed_trials),
            "device": self.device,
            "device_reason": self.device_reason,
            "error": self.error,
            "error_type": self.error_type,
            "error_category": self.error_category,
        }


@dataclass(frozen=True)
class SweepResult:
    """Everything one MLP sweep run produced."""

    sweep_run_id: str
    split_version: str
    experiment_name: str
    tuning_experiment_name: str
    train_rows: int
    test_rows: int
    cv_folds: int
    n_trials: int
    config: SweepConfig
    results: tuple[SweepModelResult, ...]
    generated_at: str
    ledger_path: Path | None = None

    @property
    def n_models(self) -> int:
        return len(self.results)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for record in self.results if record.succeeded)

    @property
    def n_failed(self) -> int:
        return sum(1 for record in self.results if not record.succeeded)

    @property
    def failed_architectures(self) -> tuple[str, ...]:
        return tuple(
            record.model_type for record in self.results if not record.succeeded
        )

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(
            record.run_id for record in self.results if record.run_id is not None
        )

    @property
    def succeeded(self) -> bool:
        return self.n_models > 0 and self.n_failed == 0

    def record_for(self, model_type: str) -> SweepModelResult | None:
        for record in self.results:
            if record.model_type == model_type:
                return record
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "sweep_run_id": self.sweep_run_id,
            "split_version": self.split_version,
            "experiment_name": self.experiment_name,
            "tuning_experiment_name": self.tuning_experiment_name,
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "cv_folds": int(self.cv_folds),
            "n_trials": int(self.n_trials),
            "config": self.config.to_dict(),
            "n_models": self.n_models,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "failed_architectures": list(self.failed_architectures),
            "run_ids": list(self.run_ids),
            "generated_at": self.generated_at,
            "results": [record.to_dict() for record in self.results],
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_frame(frame: object, *, role: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise SweepDataError(
            f"{role} must be a pandas.DataFrame, got {type(frame).__name__}."
        )
    if frame.empty:
        raise SweepDataError(f"{role} is empty; nothing to sweep on.")
    missing = [
        column for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise SweepDataError(
            f"{role} is missing column(s) {missing}; load the S01 split with "
            "heart.data.split.load_split_frames."
        )
    return frame


def _effective_params(
    spec: ModelSpec, best_params: Mapping[str, object] | None
) -> dict[str, object]:
    merged = dict(spec.fixed_params)
    if best_params:
        merged.update(best_params)
    return merged


def _error_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _classify_error(exc: BaseException) -> str:
    if isinstance(exc, NoCompletedTrialError):
        return ERROR_CATEGORY_NO_COMPLETED_TRIAL
    if isinstance(exc, (TuningError, TuningRunnerError)):
        return ERROR_CATEGORY_TUNING
    if isinstance(exc, MetricContractError):
        return ERROR_CATEGORY_EVALUATION
    if isinstance(exc, TrackingError):
        return ERROR_CATEGORY_TRACKING
    if isinstance(exc, RegistryError):
        return ERROR_CATEGORY_REGISTRY
    if isinstance(exc, MLPSpaceError):
        return ERROR_CATEGORY_TUNING
    if isinstance(exc, TorchTrainingError):
        return ERROR_CATEGORY_TUNING
    return ERROR_CATEGORY_UNEXPECTED


def _sweep_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"sweep-{stamp}-{uuid.uuid4().hex[:8]}"


def _spec_for_run(
    arch: MLPSpec,
    *,
    prefer_gpu: bool,
    selection_path: str | Path,
    smoke_epochs: int | None,
) -> ModelSpec:
    """The runner-ready spec for ``arch``, applying the smoke epoch override."""
    spec = to_model_spec(
        arch, prefer_gpu=prefer_gpu, selection_path=selection_path
    )
    if smoke_epochs is None:
        return spec
    return replace(
        spec,
        fixed_params={**dict(spec.fixed_params), "epochs": int(smoke_epochs)},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_one_architecture(
    spec: ModelSpec,
    *,
    folds: Sequence[TuningFold],
    train_frame: pd.DataFrame,
    test_split: EvaluationSplit,
    config: SweepConfig,
    trainer: Callable[[Mapping[str, object], TuningFold],
                      tuple[dict[str, object], np.ndarray, np.ndarray]],
    log_to_mlflow: bool,
    main_config: TrackingConfig | None,
    tuning_config: TrackingConfig | None,
    split_version: str,
    sweep_run_id: str,
    sampler: object | None,
    pruner: object | None,
    refit_factory: Callable[[TuningResult, ModelSpec, pd.DataFrame], object],
) -> SweepModelResult:
    """Tune, refit, evaluate, and log one architecture (fail-soft contract)."""
    start = time.perf_counter()
    counts = {"n_trials": 0, "n_complete": 0, "n_pruned": 0, "n_failed": 0}
    tuning_result: TuningResult | None = None

    def _duration() -> float:
        return round(time.perf_counter() - start, 6)

    try:
        tuning_result = run_study(
            spec,
            folds,
            config=config.to_tuning_config(study_name=f"tune-{spec.model_type}"),
            split_version=split_version,
            log_to_mlflow=log_to_mlflow,
            tracking_config=tuning_config,
            sampler=sampler,
            pruner=pruner,
            objective=MLPTuningObjective(spec, folds, trainer=trainer),
        )
        counts = {
            "n_trials": tuning_result.n_trials,
            "n_complete": tuning_result.n_complete,
            "n_pruned": tuning_result.n_pruned,
            "n_failed": tuning_result.n_failed,
        }

        pipeline = refit_factory(tuning_result, spec, train_frame)
        metrics = evaluate(pipeline, test_split)
        params = _effective_params(spec, tuning_result.best_params)

        # The device is a first-class outcome of the neural run.
        device = getattr(pipeline, "device", None)
        resolution = getattr(pipeline, "device_resolution", None)

        run_id: str | None = None
        if log_to_mlflow and device is not None and resolution is not None:
            run = log_evaluation_run(
                metrics,
                model_name=spec.model_name,
                split_version=split_version,
                params=params,
                tags={
                    RUN_KIND_TAG: FINAL_RUN_KIND,
                    MODEL_TYPE_TAG: spec.model_type,
                    "family": spec.family,
                    "sweep_run_id": sweep_run_id,
                    "slice": "S05",
                    "device": str(device),
                    "gpu_available": str(bool(resolution.gpu_available)).lower(),
                    "cpu_fallback": str(bool(resolution.cpu_fallback)).lower(),
                    "cv_folds": str(config.cv_folds),
                    "n_trials": str(tuning_result.n_trials),
                    "n_complete_trials": str(tuning_result.n_complete),
                    "best_trial_number": str(tuning_result.best_trial_number),
                },
                config=main_config,
            )
            run_id = run.run_id

        record = SweepModelResult(
            model_type=spec.model_type,
            model_name=spec.model_name,
            family=spec.family,
            status=STATUS_SUCCEEDED,
            duration_seconds=_duration(),
            run_id=run_id,
            experiment_name=main_config.experiment_name if main_config else None,
            metric_dict=metrics,
            best_params=params,
            n_trials=counts["n_trials"],
            n_complete=counts["n_complete"],
            n_pruned=counts["n_pruned"],
            n_failed_trials=counts["n_failed"],
            device=None if device is None else str(device),
            device_reason=(
                None
                if resolution is None
                else str(getattr(resolution, "reason", None))
            ),
        )
        logger.info(
            "Sweep architecture %s succeeded: device=%s %s=%.4f trials=%d/%d "
            "run=%s (%.2fs)",
            spec.model_type,
            record.device or "unknown",
            PRIMARY_METRIC,
            float(metrics[PRIMARY_METRIC]),
            counts["n_complete"],
            counts["n_trials"],
            run_id or "not logged",
            record.duration_seconds,
        )
        return record
    except SweepError:
        raise
    except Exception as exc:  # noqa: BLE001 - every architecture failure is recorded
        category = _classify_error(exc)
        error = _error_message(exc)
        if isinstance(exc, NoCompletedTrialError) and tuning_result is not None:
            # Surface the underlying trial failure (e.g. torch unavailable)
            # so the ledger says *why* no trial completed.
            first_error = next(
                (trial.error for trial in tuning_result.trials if trial.error),
                None,
            )
            if first_error:
                error = f"{error}; first trial error: {first_error}"
        record = SweepModelResult(
            model_type=spec.model_type,
            model_name=spec.model_name,
            family=spec.family,
            status=STATUS_FAILED,
            duration_seconds=_duration(),
            metric_dict=None,
            best_params=None,
            n_trials=counts["n_trials"],
            n_complete=counts["n_complete"],
            n_pruned=counts["n_pruned"],
            n_failed_trials=counts["n_failed"],
            device=None,
            error=error,
            error_type=type(exc).__name__,
            error_category=category,
        )
        logger.error(
            "Sweep architecture %s failed [%s]: %s",
            spec.model_type,
            category,
            record.error,
        )
        return record


def run_mlp_sweep(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    architectures: Sequence[str] | None = None,
    config: SweepConfig | None = None,
    trainer: (
        Callable[
            [Mapping[str, object], TuningFold],
            tuple[dict[str, object], np.ndarray, np.ndarray],
        ]
        | None
    ) = None,
    experiment_name: str = SWEEP_EXPERIMENT,
    tuning_experiment: str = DEFAULT_TUNING_EXPERIMENT,
    tracking_dir: str | Path | None = None,
    split_version: str = SPLIT_VERSION,
    ledger_path: str | Path | None = None,
    log_to_mlflow: bool = True,
    strict: bool = False,
    sweep_id: str | None = None,
    prefer_gpu: bool = True,
    selection_path: str | Path | None = None,
    sampler: object | None = None,
    pruner: object | None = None,
    refit_factory: (
        Callable[[TuningResult, ModelSpec, pd.DataFrame], object] | None
    ) = None,
) -> SweepResult:
    """Run the selected MLP architectures through the shared protocol.

    Parameters
    ----------
    train_frame / test_frame:
        The S01 split frames (training rows carry the target; the test rows are
        the held-out evaluation split).
    architectures:
        Optional architecture selection; defaults to every configured
        :data:`~heart.models.mlp_spaces.MLP_ARCHITECTURES` member.
    config:
        The :class:`SweepConfig`; defaults to the declared (full) configuration.
    trainer:
        The per-fold scoring leaf. Defaults to the production
        :func:`heart.models.mlp_spaces.mlp_trainer` (bound to ``prefer_gpu``
        and ``selection_path``); tests inject a deterministic scorer so the
        full sweep is executable without torch.
    experiment_name / tuning_experiment:
        The portfolio and tuning MLflow experiments. Final runs land in the
        portfolio experiment; trial runs in the tuning experiment.
    tracking_dir:
        Local MLflow store directory (defaults to ``experiments/mlruns``).
    split_version:
        Label recorded on every run.
    ledger_path:
        When given, the machine-readable sweep result is written here atomically.
    log_to_mlflow:
        When ``False`` the workflow runs without MLflow.
    strict:
        When ``True`` a failed architecture raises :class:`SweepRunError`
        *after* the result is assembled.
    sweep_id:
        Override for the generated grouping id tagged onto every final run.
    prefer_gpu:
        Whether neural training requests the GPU first (falls back to CPU with
        a logged warning when the GPU is unavailable).
    selection_path:
        The S04 feature-selection ledger the MLP representation reads.
    sampler / pruner:
        Optional prebuilt Optuna sampler/pruner forwarded to every study.
    refit_factory:
        ``(tuning_result, spec, train_frame) -> fitted model``; defaults to
        :func:`heart.tuning.runner.build_best_pipeline`. Injecting a stand-in
        lets tests prove the sweep's run-count/logging contract without torch.
    """
    resolved_config = config or SweepConfig()
    selected = select_mlp_architectures(architectures)
    train = _require_frame(train_frame, role="train_frame")
    test = _require_frame(test_frame, role="test_frame")

    request_id = sweep_id or _sweep_run_id()
    test_split = evaluation_split_from_frames(
        test, name=f"{split_version}/test"
    )

    resolved_trainer = trainer if trainer is not None else mlp_trainer(
        prefer_gpu=prefer_gpu,
        selection_path=(
            selection_path if selection_path is not None
            else DEFAULT_SELECTION_LEDGER_PATH
        ),
    )
    resolved_refit = refit_factory if refit_factory is not None else build_best_pipeline

    main_config: TrackingConfig | None = None
    tuning_config: TrackingConfig | None = None
    if log_to_mlflow:
        main_config = configure_tracking(
            experiment_name=experiment_name, tracking_dir=tracking_dir
        )
        tuning_config = configure_tracking(
            experiment_name=tuning_experiment, tracking_dir=tracking_dir
        )

    try:
        folds = build_cv_folds(train, n_folds=resolved_config.cv_folds)
    except TuningError as exc:
        raise SweepDataError(
            f"Could not build {resolved_config.cv_folds}-fold CV from the "
            f"training frame: {exc}"
        ) from exc

    specs = tuple(
        _spec_for_run(
            arch,
            prefer_gpu=prefer_gpu,
            selection_path=(
                selection_path if selection_path is not None
                else DEFAULT_SELECTION_LEDGER_PATH
            ),
            smoke_epochs=resolved_config.smoke_epochs,
        )
        for arch in selected
    )

    logger.info(
        "Starting MLP sweep %s: %d architecture(s), %d fold(s), %d trial(s) "
        "per architecture, split=%s, device_preference=%s, experiment=%s",
        request_id,
        len(specs),
        resolved_config.cv_folds,
        resolved_config.n_trials,
        split_version,
        "gpu" if prefer_gpu else "cpu",
        experiment_name if log_to_mlflow else "not logged",
    )

    results: list[SweepModelResult] = []
    for spec in specs:
        results.append(
            _run_one_architecture(
                spec,
                folds=folds,
                train_frame=train,
                test_split=test_split,
                config=resolved_config,
                trainer=resolved_trainer,
                log_to_mlflow=log_to_mlflow,
                main_config=main_config,
                tuning_config=tuning_config,
                split_version=split_version,
                sweep_run_id=request_id,
                sampler=sampler,
                pruner=pruner,
                refit_factory=resolved_refit,
            )
        )

    result = SweepResult(
        sweep_run_id=request_id,
        split_version=split_version,
        experiment_name=experiment_name,
        tuning_experiment_name=tuning_experiment,
        train_rows=int(len(train)),
        test_rows=int(len(test)),
        cv_folds=int(resolved_config.cv_folds),
        n_trials=int(resolved_config.n_trials),
        config=resolved_config,
        results=tuple(results),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ledger_path=None,
    )

    destination = Path(ledger_path) if ledger_path is not None else None
    if destination is not None:
        result = _with_ledger_path(
            result, write_sweep_ledger(result, destination)
        )

    logger.info(
        "MLP sweep %s finished: %d/%d architecture(s) succeeded, %d failed%s",
        request_id,
        result.n_succeeded,
        result.n_models,
        result.n_failed,
        f" (failed: {', '.join(result.failed_architectures)})"
        if result.n_failed
        else "",
    )

    if strict and result.n_failed:
        raise SweepRunError(
            f"{result.n_failed} of {result.n_models} architecture(s) failed "
            f"({', '.join(result.failed_architectures)}); see the attached "
            "result for per-architecture errors.",
            result,
        )
    return result


def _with_ledger_path(result: SweepResult, path: Path) -> SweepResult:
    return replace(result, ledger_path=path)


# ---------------------------------------------------------------------------
# Ledger / report
# ---------------------------------------------------------------------------


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def sweep_ledger(result: SweepResult) -> dict[str, object]:
    """Return the sweep's durable, JSON-serialisable ledger payload."""
    if not isinstance(result, SweepResult):
        raise SweepLedgerError(
            f"result must be a SweepResult, got {type(result).__name__}."
        )
    return result.to_dict()


def write_sweep_ledger(result: SweepResult, path: str | Path) -> Path:
    """Atomically write the sweep ledger JSON for ``result`` to ``path``."""
    destination = Path(path)
    try:
        payload = json.dumps(
            sweep_ledger(result), indent=2, sort_keys=True, default=_json_default
        )
    except TypeError as exc:
        raise SweepLedgerError(
            f"Could not serialise the sweep ledger: {exc}"
        ) from exc
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".part")
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, destination)
    except OSError as exc:
        raise SweepLedgerError(
            f"Could not write the sweep ledger to {destination}: {exc}"
        ) from exc
    logger.info(
        "Wrote sweep ledger to %s (%d architecture(s), %d/%d succeeded)",
        destination,
        result.n_models,
        result.n_succeeded,
        result.n_models,
    )
    return destination


def render_sweep_summary(result: SweepResult) -> str:
    """Render the sweep results as a markdown table (CLI + diagnostics)."""
    if not isinstance(result, SweepResult):
        raise SweepLedgerError(
            f"result must be a SweepResult, got {type(result).__name__}."
        )
    lines = [
        f"# MLP sweep {result.sweep_run_id}",
        "",
        f"split `{result.split_version}` — train {result.train_rows} / "
        f"test {result.test_rows}; {result.cv_folds} CV folds; "
        f"{result.n_trials} trial(s)/architecture.",
        "",
        f"**{result.n_succeeded}/{result.n_models} architectures succeeded.**",
        "",
        "| model_type | status | ROC-AUC | device | trials | run_id | error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    ranked = sorted(
        result.results,
        key=lambda record: (
            record.primary_metric is None,
            -(record.primary_metric or 0.0),
        ),
    )
    for record in ranked:
        metric = (
            "-" if record.primary_metric is None else f"{record.primary_metric:.4f}"
        )
        error = (record.error or "").replace("|", "\\|")
        lines.append(
            f"| `{record.model_type}` | {record.status} | {metric} | "
            f"{record.device or '-'} | {record.n_complete}/{record.n_trials} | "
            f"{record.run_id or '-'} | {error} |"
        )
    lines.append("")
    lines.append(f"generated at: {result.generated_at}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_architectures(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    members = [part.strip() for part in raw.split(",") if part.strip()]
    return members or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.models.run_mlp_sweep",
        description=(
            "Tune, evaluate, and log the configured MLP architectures through "
            "the shared protocol, recording the compute device per run."
        ),
    )
    parser.add_argument(
        "--split-version", default=SPLIT_VERSION, help="S01 split version."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Reduced smoke configuration ({SMOKE_N_TRIALS} trials, "
            f"{SMOKE_CV_FOLDS} folds, {SMOKE_EPOCHS} epochs)."
        ),
    )
    parser.add_argument("--trials", type=int, default=None, help="Trials per architecture.")
    parser.add_argument(
        "--folds", type=int, default=None, help="Cross-validation folds per architecture."
    )
    parser.add_argument(
        "--architectures",
        default=None,
        help=(
            "Comma-separated subset of architecture model_type values "
            f"(default: all — {', '.join(MLP_MODEL_TYPES)})."
        ),
    )
    parser.add_argument(
        "--sampler",
        choices=("tpe", "random"),
        default=None,
        help="Optuna sampler (default: tpe, or random under --smoke).",
    )
    parser.add_argument(
        "--pruner",
        choices=("median", "none"),
        default=None,
        help="Optuna pruner (default: median, or none under --smoke).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Request the CPU path (GPU acceleration disabled, logged fallback).",
    )
    parser.add_argument(
        "--tracking-dir",
        default=None,
        help="MLflow store directory (default: experiments/mlruns).",
    )
    parser.add_argument(
        "--experiment",
        default=SWEEP_EXPERIMENT,
        help=f"Portfolio experiment name (default: {SWEEP_EXPERIMENT}).",
    )
    parser.add_argument(
        "--tuning-experiment",
        default=DEFAULT_TUNING_EXPERIMENT,
        help=f"Tuning experiment name (default: {DEFAULT_TUNING_EXPERIMENT}).",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="Optional path for the sweep ledger JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any architecture fails (results are still recorded).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip MLflow logging."
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> SweepConfig:
    base = smoke_sweep_config() if args.smoke else SweepConfig()
    return SweepConfig(
        n_trials=base.n_trials if args.trials is None else args.trials,
        cv_folds=base.cv_folds if args.folds is None else args.folds,
        sampler=base.sampler if args.sampler is None else args.sampler,
        pruner=base.pruner if args.pruner is None else args.pruner,
        timeout=base.timeout,
        direction=base.direction,
        sampler_seed=base.sampler_seed,
        n_startup_trials=base.n_startup_trials,
        n_warmup_steps=base.n_warmup_steps,
        smoke_epochs=base.smoke_epochs if args.smoke else None,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        config = _resolve_config(args)
    except SweepConfigError as exc:
        print(f"configuration error: {exc}")
        return 2

    try:
        train_frame, test_frame = load_split_frames(args.split_version)
    except SplitError as exc:
        print(f"could not load split {args.split_version!r}: {exc}")
        return 1

    try:
        result = run_mlp_sweep(
            train_frame,
            test_frame,
            architectures=_parse_architectures(args.architectures),
            config=config,
            experiment_name=args.experiment,
            tuning_experiment=args.tuning_experiment,
            tracking_dir=args.tracking_dir,
            split_version=args.split_version,
            ledger_path=args.ledger,
            log_to_mlflow=not args.no_mlflow,
            strict=args.strict,
            prefer_gpu=not args.cpu,
        )
    except SweepRunError as exc:
        print(render_sweep_summary(exc.result))
        print(f"sweep failed (strict mode): {exc}")
        return 1
    except SweepError as exc:
        print(f"sweep error: {exc}")
        return 1

    print(render_sweep_summary(result))
    print(f"portfolio experiment: {result.experiment_name}")
    print(f"tuning experiment:    {result.tuning_experiment_name}")
    if result.ledger_path is not None:
        print(f"ledger:               {result.ledger_path}")
    return 0 if result.n_succeeded > 0 else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())