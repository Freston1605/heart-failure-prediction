"""Run the full fixed classical battery through the shared protocol (S03/T03).

This module owns the **breadth** of the portfolio: it drives every member of
:data:`heart.models.registry.BATTERY_MODELS` through the one shared workflow —
group-aware cross-validation tuning (T02), a refit of the winning parameters on
the full training frame, and a single evaluation on the held-out test split via
:func:`heart.eval.contract.evaluate` — and logs exactly **one final MLflow run
per model** to the portfolio experiment.

The shape of a battery run
--------------------------
1. The cross-validation folds are built **once**, from the training frame, and
   reused by every model. Using one fold structure for the whole battery is
   what makes the runs comparable: no model gets a luckier partition.
2. Each spec is tuned with :func:`heart.tuning.runner.run_study`. Trial runs go
   to the dedicated tuning experiment (:data:`DEFAULT_TUNING_EXPERIMENT`), so
   the portfolio experiment keeps exactly one run per model (decision D013).
3. The best trial's parameters are refit on the full training frame
   (:func:`heart.tuning.runner.build_best_pipeline`) and scored once on the
   held-out split.
4. The scored metrics are logged through the frozen convention
   (:func:`heart.tracking.run.log_evaluation_run`) tagged
   ``run_kind=final`` and ``model_type=<registry key>``, so T04's leaderboard
   can select exactly these runs from MLflow rather than trusting a hand-made
   table.

Fail-soft, never silent
-----------------------
A single model failing (a missing XGBoost dependency, a study with no completed
trial, a tracking error) is **recorded** as a failed :class:`BatteryModelResult`
with a categorised error, and the battery continues with the next member. This
is deliberate: the battery's job is breadth, and a dropped model would quietly
shrink the leaderboard. Pass ``strict=True`` to instead raise
:class:`BatteryRunError` after the run is summarised, carrying the collected
result. The battery never drops a model from its result.

Observability
-------------
Every model logs its status, trial counts, primary metric, and MLflow run id at
``INFO``, plus a battery summary line at the end. :func:`render_battery_summary`
renders the markdown table the CLI prints, and :func:`write_battery_ledger`
persists the machine-readable result JSON atomically.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

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
from heart.models.registry import (
    BATTERY_MODELS,
    BATTERY_MODEL_TYPES,
    BATTERY_SIZE,
    ModelDependencyError,
    ModelSpec,
    RegistryError,
    resolve_spec,
)
from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    TrackingConfig,
    TrackingError,
    configure_tracking,
)
from heart.runtime import (
    classify_error,
    error_message,
    merge_effective_params,
    timestamped_run_id,
    write_json_document,
)
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import (
    DEFAULT_CV_FOLDS,
    DEFAULT_TUNING_EXPERIMENT,
    FINAL_RUN_KIND,
    MODEL_TYPE_TAG,
    RUN_KIND_TAG,
    NoCompletedTrialError,
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
    "BATTERY_EXPERIMENT",
    "BATTERY_LEDGER_FILENAME",
    "SMOKE_N_TRIALS",
    "SMOKE_CV_FOLDS",
    "STATUS_SUCCEEDED",
    "STATUS_FAILED",
    "ERROR_CATEGORY_DEPENDENCY",
    "ERROR_CATEGORY_NO_COMPLETED_TRIAL",
    "ERROR_CATEGORY_TUNING",
    "ERROR_CATEGORY_EVALUATION",
    "ERROR_CATEGORY_TRACKING",
    "ERROR_CATEGORY_REGISTRY",
    "ERROR_CATEGORY_UNEXPECTED",
    "BatteryError",
    "BatteryConfigError",
    "BatteryDataError",
    "BatteryRunError",
    "BatteryLedgerError",
    "UnknownModelSelectionError",
    "NoModelsSelectedError",
    "BatteryConfig",
    "BatteryModelResult",
    "BatteryResult",
    "smoke_config",
    "select_specs",
    "run_battery",
    "battery_ledger",
    "write_battery_ledger",
    "render_battery_summary",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: The portfolio experiment the final per-model runs belong to.
BATTERY_EXPERIMENT: str = DEFAULT_EXPERIMENT

#: Default filename for a persisted battery ledger.
BATTERY_LEDGER_FILENAME: str = "battery_ledger.json"

#: Reduced trial budget for the smoke configuration (still >= 1 completed trial).
SMOKE_N_TRIALS: int = 2

#: Reduced cross-validation fold count for the smoke configuration.
SMOKE_CV_FOLDS: int = 3

#: A model whose tuning completed, refit, evaluated, and logged successfully.
STATUS_SUCCEEDED: str = "succeeded"

#: A model that failed somewhere in the workflow (still recorded, never dropped).
STATUS_FAILED: str = "failed"

#: Error categories recorded on a failed model result.
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


class BatteryError(Exception):
    """Base class for every battery-run failure."""


class BatteryConfigError(BatteryError):
    """The requested battery configuration is invalid."""


class BatteryDataError(BatteryError):
    """A frame or split handed to the battery is malformed."""


class BatteryRunError(BatteryError):
    """Strict mode: at least one model failed (the result is attached)."""

    def __init__(self, message: str, result: "BatteryResult") -> None:
        super().__init__(message)
        self.result = result

    @property
    def failed_models(self) -> tuple[str, ...]:
        return tuple(
            record.model_type
            for record in self.result.results
            if record.status == STATUS_FAILED
        )


class BatteryLedgerError(BatteryError):
    """The battery ledger could not be serialised or written."""


class UnknownModelSelectionError(BatteryConfigError):
    """``model_types`` names a model that is not in the fixed battery."""


class NoModelsSelectedError(BatteryConfigError):
    """The resolved selection contains no battery member."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryConfig:
    """The resolved, validated knobs for one battery run.

    It mirrors the tuning configuration (:class:`TuningConfig`) plus the
    number of cross-validation folds the battery uses. Validation is delegated
    to :class:`TuningConfig` so an invalid sampler/pruner/trial budget raises
    the same named error family, wrapped as :class:`BatteryConfigError` with the
    battery's own context.
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

    def __post_init__(self) -> None:
        if isinstance(self.cv_folds, bool) or not isinstance(self.cv_folds, int):
            raise BatteryConfigError(
                f"cv_folds must be an integer, got {self.cv_folds!r}."
            )
        if self.cv_folds < 2:
            raise BatteryConfigError(
                f"cv_folds must be >= 2, got {self.cv_folds!r}."
            )
        # Reuse the tuning config's validation instead of duplicating it.
        try:
            self.to_tuning_config()
        except TuningError as exc:
            raise BatteryConfigError(
                f"Invalid battery tuning configuration: {exc}"
            ) from exc

    def to_tuning_config(self, *, study_name: str | None = None) -> TuningConfig:
        """Build the :class:`TuningConfig` for one model's study."""
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
        }


def smoke_config(
    *, n_trials: int = SMOKE_N_TRIALS, cv_folds: int = SMOKE_CV_FOLDS
) -> BatteryConfig:
    """Return the reduced-trial smoke configuration used by tests and the CLI.

    A random sampler and no pruner keep the smoke deterministic and fast; the
    folds are still group-aware and leakage-safe, so the smoke exercises the
    real protocol rather than a shortcut.
    """
    return BatteryConfig(
        n_trials=n_trials,
        cv_folds=cv_folds,
        sampler="random",
        pruner="none",
        sampler_seed=RANDOM_SEED,
        n_startup_trials=1,
        n_warmup_steps=1,
    )


# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryModelResult:
    """One model's outcome within a battery run (never dropped)."""

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
            "error": self.error,
            "error_type": self.error_type,
            "error_category": self.error_category,
        }


@dataclass(frozen=True)
class BatteryResult:
    """Everything one battery run produced."""

    battery_run_id: str
    split_version: str
    experiment_name: str
    tuning_experiment_name: str
    train_rows: int
    test_rows: int
    cv_folds: int
    n_trials: int
    config: BatteryConfig
    results: tuple[BatteryModelResult, ...]
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
    def failed_models(self) -> tuple[str, ...]:
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

    def record_for(self, model_type: str) -> BatteryModelResult | None:
        for record in self.results:
            if record.model_type == model_type:
                return record
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "battery_run_id": self.battery_run_id,
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
            "failed_models": list(self.failed_models),
            "run_ids": list(self.run_ids),
            "generated_at": self.generated_at,
            "results": [record.to_dict() for record in self.results],
        }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _normalise_selection(values: Sequence[str] | None, *, role: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise BatteryConfigError(
            f"{role} must be a sequence of model_type strings, not a bare "
            f"string {values!r}; pass e.g. ['lda', 'qda']."
        )
    resolved: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise BatteryConfigError(
                f"{role} contains a non-string or blank model_type: {value!r}."
            )
        candidate = value.strip()
        try:
            spec = resolve_spec(candidate)
        except RegistryError as exc:
            raise UnknownModelSelectionError(
                f"{role} names unknown battery model {candidate!r}; declared "
                f"members are {list(BATTERY_MODEL_TYPES)}."
            ) from exc
        if spec.model_type not in resolved:
            resolved.append(spec.model_type)
    return tuple(resolved)


def select_specs(
    model_types: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[ModelSpec, ...]:
    """Resolve the battery members to run, in registry (leaderboard) order.

    ``model_types`` restricts the battery to the named members; ``exclude``
    removes members afterwards. Unknown names raise
    :class:`UnknownModelSelectionError`, and a selection that resolves to
    nothing raises :class:`NoModelsSelectedError`.
    """
    include = set(_normalise_selection(model_types, role="model_types"))
    excluded = set(_normalise_selection(exclude, role="exclude"))
    specs = tuple(
        spec
        for spec in BATTERY_MODELS
        if (not include or spec.model_type in include)
        and spec.model_type not in excluded
    )
    if not specs:
        raise NoModelsSelectedError(
            "The battery selection is empty after applying model_types="
            f"{model_types!r} and exclude={exclude!r}; nothing to run. The "
            f"fixed battery has {BATTERY_SIZE} members: "
            f"{list(BATTERY_MODEL_TYPES)}."
        )
    return specs


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_frame(frame: object, *, role: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise BatteryDataError(
            f"{role} must be a pandas.DataFrame, got {type(frame).__name__}."
        )
    if frame.empty:
        raise BatteryDataError(f"{role} is empty; nothing to run the battery on.")
    missing = [
        column for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise BatteryDataError(
            f"{role} is missing column(s) {missing}; load the S01 split with "
            "heart.data.split.load_split_frames."
        )
    return frame


#: Ordered exception-to-category map for fail-soft model records. First match
#: wins; subclass-before-parent order matters.
ERROR_MAP: tuple[tuple[type[BaseException], str], ...] = (
    (ModelDependencyError, ERROR_CATEGORY_DEPENDENCY),
    (NoCompletedTrialError, ERROR_CATEGORY_NO_COMPLETED_TRIAL),
    (TuningError, ERROR_CATEGORY_TUNING),
    (TuningRunnerError, ERROR_CATEGORY_TUNING),
    (MetricContractError, ERROR_CATEGORY_EVALUATION),
    (TrackingError, ERROR_CATEGORY_TRACKING),
    (RegistryError, ERROR_CATEGORY_REGISTRY),
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class _ModelRunState:
    """Mutable scratch space for one model's run, converted to a frozen record."""

    trial_counts: dict[str, int] = field(
        default_factory=lambda: {"n_trials": 0, "n_complete": 0, "n_pruned": 0, "n_failed": 0}
    )


def _run_one_model(
    spec: ModelSpec,
    *,
    folds,
    train_frame: pd.DataFrame,
    test_split: EvaluationSplit,
    config: BatteryConfig,
    log_to_mlflow: bool,
    main_config: TrackingConfig | None,
    tuning_config: TrackingConfig | None,
    split_version: str,
    battery_run_id: str,
    sampler: object | None,
    pruner: object | None,
) -> BatteryModelResult:
    """Tune, refit, evaluate, and log one battery member (fail-soft contract)."""
    start = time.perf_counter()
    counts = _ModelRunState().trial_counts
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
        )
        counts = {
            "n_trials": tuning_result.n_trials,
            "n_complete": tuning_result.n_complete,
            "n_pruned": tuning_result.n_pruned,
            "n_failed": tuning_result.n_failed,
        }

        pipeline = build_best_pipeline(tuning_result, spec, train_frame)
        metrics = evaluate(pipeline, test_split)
        params = merge_effective_params(spec.fixed_params, tuning_result.best_params)

        run_id: str | None = None
        if log_to_mlflow:
            run = log_evaluation_run(
                metrics,
                model_name=spec.model_name,
                split_version=split_version,
                params=params,
                tags={
                    RUN_KIND_TAG: FINAL_RUN_KIND,
                    MODEL_TYPE_TAG: spec.model_type,
                    "family": spec.family,
                    "battery_run_id": battery_run_id,
                    "slice": "S03",
                    "cv_folds": str(config.cv_folds),
                    "n_trials": str(tuning_result.n_trials),
                    "n_complete_trials": str(tuning_result.n_complete),
                    "best_trial_number": str(tuning_result.best_trial_number),
                },
                config=main_config,
            )
            run_id = run.run_id

        record = BatteryModelResult(
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
        )
        logger.info(
            "Battery model %s succeeded: %s=%.4f trials=%d/%d run=%s (%.2fs)",
            spec.model_type,
            PRIMARY_METRIC,
            float(metrics[PRIMARY_METRIC]),
            counts["n_complete"],
            counts["n_trials"],
            run_id or "not logged",
            record.duration_seconds,
        )
        return record
    except BatteryError:
        raise
    except Exception as exc:  # noqa: BLE001 - every model failure is recorded
        category = classify_error(exc, ERROR_MAP, default=ERROR_CATEGORY_UNEXPECTED)
        record = BatteryModelResult(
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
            error=error_message(exc),
            error_type=type(exc).__name__,
            error_category=category,
        )
        logger.error(
            "Battery model %s failed [%s]: %s",
            spec.model_type,
            category,
            record.error,
        )
        return record


def run_battery(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    model_types: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    config: BatteryConfig | None = None,
    experiment_name: str = BATTERY_EXPERIMENT,
    tuning_experiment: str = DEFAULT_TUNING_EXPERIMENT,
    tracking_dir: str | Path | None = None,
    split_version: str = SPLIT_VERSION,
    ledger_path: str | Path | None = None,
    log_to_mlflow: bool = True,
    strict: bool = False,
    battery_id: str | None = None,
    sampler: object | None = None,
    pruner: object | None = None,
) -> BatteryResult:
    """Run the selected battery members through the shared protocol.

    Parameters
    ----------
    train_frame / test_frame:
        The S01 split frames (training rows carry the target; the test rows are
        the held-out evaluation split).
    model_types / exclude:
        Optional battery-member selection; defaults to the full fixed battery.
    config:
        The :class:`BatteryConfig`; defaults to the declared (full) configuration.
    experiment_name / tuning_experiment:
        The portfolio and tuning MLflow experiments. Final runs land in the
        portfolio experiment; trial runs in the tuning experiment.
    tracking_dir:
        Local MLflow store directory (defaults to ``experiments/mlruns``).
    split_version:
        Label recorded on every run.
    ledger_path:
        When given, the machine-readable battery result is written here atomically.
    log_to_mlflow:
        When ``False`` the workflow runs without MLflow (useful for diagnostics).
    strict:
        When ``True`` a failed model raises :class:`BatteryRunError` *after* the
        result is assembled; the attached ``result`` still lists every model.
    battery_id:
        Override for the generated grouping id tagged onto every final run.
    sampler / pruner:
        Optional prebuilt Optuna sampler/pruner forwarded to every study,
        overriding the configuration. A deterministic sampler makes a smoke
        run cheap and reproducible.
    """
    resolved_config = config or BatteryConfig()
    specs = select_specs(model_types, exclude)
    train = _require_frame(train_frame, role="train_frame")
    test = _require_frame(test_frame, role="test_frame")

    request_id = battery_id or timestamped_run_id("battery")
    test_split = evaluation_split_from_frames(
        test, name=f"{split_version}/test"
    )

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
        raise BatteryDataError(
            f"Could not build {resolved_config.cv_folds}-fold CV from the "
            f"training frame: {exc}"
        ) from exc

    logger.info(
        "Starting battery run %s: %d model(s), %d fold(s), %d trial(s) per "
        "model, split=%s, experiment=%s",
        request_id,
        len(specs),
        resolved_config.cv_folds,
        resolved_config.n_trials,
        split_version,
        experiment_name if log_to_mlflow else "not logged",
    )

    results: list[BatteryModelResult] = []
    for spec in specs:
        results.append(
            _run_one_model(
                spec,
                folds=folds,
                train_frame=train,
                test_split=test_split,
                config=resolved_config,
                log_to_mlflow=log_to_mlflow,
                main_config=main_config,
                tuning_config=tuning_config,
                split_version=split_version,
                battery_run_id=request_id,
                sampler=sampler,
                pruner=pruner,
            )
        )

    result = BatteryResult(
        battery_run_id=request_id,
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
            result, write_battery_ledger(result, destination)
        )

    logger.info(
        "Battery run %s finished: %d/%d model(s) succeeded, %d failed%s",
        request_id,
        result.n_succeeded,
        result.n_models,
        result.n_failed,
        f" (failed: {', '.join(result.failed_models)})"
        if result.n_failed
        else "",
    )

    if strict and result.n_failed:
        raise BatteryRunError(
            f"{result.n_failed} of {result.n_models} battery model(s) failed "
            f"({', '.join(result.failed_models)}); see the attached result for "
            "per-model errors.",
            result,
        )
    return result


def _with_ledger_path(result: BatteryResult, path: Path) -> BatteryResult:
    from dataclasses import replace

    return replace(result, ledger_path=path)


# ---------------------------------------------------------------------------
# Ledger / report
# ---------------------------------------------------------------------------


def battery_ledger(result: BatteryResult) -> dict[str, object]:
    """Return the battery's durable, JSON-serialisable ledger payload."""
    if not isinstance(result, BatteryResult):
        raise BatteryLedgerError(
            f"result must be a BatteryResult, got {type(result).__name__}."
        )
    return result.to_dict()


def write_battery_ledger(result: BatteryResult, path: str | Path) -> Path:
    """Atomically write the battery ledger JSON for ``result`` to ``path``."""
    destination = write_json_document(
        path,
        battery_ledger(result),
        error_factory=BatteryLedgerError,
        label="battery ledger",
    )
    logger.info(
        "Wrote battery ledger to %s (%d model(s), %d/%d succeeded)",
        destination,
        result.n_models,
        result.n_succeeded,
        result.n_models,
    )
    return destination


def _num(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def render_battery_summary(result: BatteryResult) -> str:
    """Render the battery results as a markdown table (CLI + diagnostics)."""
    if not isinstance(result, BatteryResult):
        raise BatteryLedgerError(
            f"result must be a BatteryResult, got {type(result).__name__}."
        )
    lines = [
        f"# Battery {result.battery_run_id}",
        "",
        f"split `{result.split_version}` — train {result.train_rows} / "
        f"test {result.test_rows}; {result.cv_folds} CV folds; "
        f"{result.n_trials} trial(s)/model.",
        "",
        f"**{result.n_succeeded}/{result.n_models} models succeeded.**",
        "",
        "| model_type | status | ROC-AUC | trials | run_id | error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ranked = sorted(
        result.results,
        key=lambda record: (record.primary_metric is None, -(record.primary_metric or 0.0)),
    )
    for record in ranked:
        metric = (
            "-" if record.primary_metric is None else _num(record.primary_metric)
        )
        error = (record.error or "").replace("|", "\\|")
        lines.append(
            f"| `{record.model_type}` | {record.status} | {metric} | "
            f"{record.n_complete}/{record.n_trials} | "
            f"{record.run_id or '-'} | {error} |"
        )
    lines.append("")
    lines.append(f"generated at: {result.generated_at}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_models(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    members = [part.strip() for part in raw.split(",") if part.strip()]
    return members or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.models.run_battery",
        description=(
            "Tune, evaluate, and log every model in the fixed classical "
            "battery through the shared protocol."
        ),
    )
    parser.add_argument(
        "--split-version", default=SPLIT_VERSION, help="S01 split version."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Reduced-trial smoke configuration ({SMOKE_N_TRIALS} trials, "
            f"{SMOKE_CV_FOLDS} folds)."
        ),
    )
    parser.add_argument("--trials", type=int, default=None, help="Trials per model.")
    parser.add_argument(
        "--folds", type=int, default=None, help="Cross-validation folds per model."
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of battery model_type values (default: all).",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated battery model_type values to skip.",
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
        "--tracking-dir",
        default=None,
        help="MLflow store directory (default: experiments/mlruns).",
    )
    parser.add_argument(
        "--experiment",
        default=BATTERY_EXPERIMENT,
        help=f"Portfolio experiment name (default: {BATTERY_EXPERIMENT}).",
    )
    parser.add_argument(
        "--tuning-experiment",
        default=DEFAULT_TUNING_EXPERIMENT,
        help=f"Tuning experiment name (default: {DEFAULT_TUNING_EXPERIMENT}).",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="Optional path for the battery ledger JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any model fails (the result is still recorded).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip MLflow logging."
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> BatteryConfig:
    base = smoke_config() if args.smoke else BatteryConfig()
    return BatteryConfig(
        n_trials=base.n_trials if args.trials is None else args.trials,
        cv_folds=base.cv_folds if args.folds is None else args.folds,
        sampler=base.sampler if args.sampler is None else args.sampler,
        pruner=base.pruner if args.pruner is None else args.pruner,
        timeout=base.timeout,
        direction=base.direction,
        sampler_seed=base.sampler_seed,
        n_startup_trials=base.n_startup_trials,
        n_warmup_steps=base.n_warmup_steps,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        config = _resolve_config(args)
    except BatteryConfigError as exc:
        print(f"configuration error: {exc}")
        return 2

    try:
        train_frame, test_frame = load_split_frames(args.split_version)
    except SplitError as exc:
        print(f"could not load split {args.split_version!r}: {exc}")
        return 1

    try:
        result = run_battery(
            train_frame,
            test_frame,
            model_types=_parse_models(args.models),
            exclude=_parse_models(args.exclude),
            config=config,
            experiment_name=args.experiment,
            tuning_experiment=args.tuning_experiment,
            tracking_dir=args.tracking_dir,
            split_version=args.split_version,
            ledger_path=args.ledger,
            log_to_mlflow=not args.no_mlflow,
            strict=args.strict,
        )
    except BatteryRunError as exc:
        print(render_battery_summary(exc.result))
        print(f"battery failed (strict mode): {exc}")
        return 1
    except BatteryError as exc:
        print(f"battery error: {exc}")
        return 1

    print(render_battery_summary(result))
    print(f"portfolio experiment: {result.experiment_name}")
    print(f"tuning experiment:    {result.tuning_experiment_name}")
    if result.ledger_path is not None:
        print(f"ledger:               {result.ledger_path}")
    return 0 if result.n_succeeded > 0 else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
