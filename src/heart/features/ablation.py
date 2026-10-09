"""Feature-set ablation harness (S04/T02).

What this module is for
-----------------------
:mod:`heart.features.engineering` declares ten individually toggleable
engineered transforms but deliberately decides *nothing* about which of them
help. This module answers that question mechanically: it evaluates the
**baseline feature set** and a family of **feature-mode variants** through the
one shared evaluation contract (:func:`heart.eval.contract.evaluate`) and
records every result in MLflow, so a feature decision can be traced to a
number instead of an opinion.

The unit of comparison is the :class:`FeatureMode` — a named, explicit
selection over the T01 registry. A mode with an empty ``include`` is the
baseline feature set (the raw eleven columns); a mode that adds one transform
measures that transform's marginal contribution; a mode that adds all of them
measures the ceiling. Because every mode is scored on the same held-out split
with the same model and the same declared preprocessing, the only thing that
differs between two runs is the feature representation.

Leakage safety
--------------
The pipeline places feature engineering *inside* the sklearn ``Pipeline``,
after the zero-as-missing median imputation and before scaling/one-hot
encoding::

    ZeroMedianImputer  ->  EngineeredFeatureTransformer  ->  ColumnTransformer  ->  clf

Every transform is fit on the training rows alone and the engineering step is
stateless, so a held-out row can never influence a fitted statistic. The
baseline mode uses a ``passthrough`` engineering step over the identical
zero-policy + column-transformer chain, which makes it numerically equivalent
to :func:`heart.models.baseline.build_baseline_model` — so "baseline" here
means exactly the published baseline.

Tracking
--------
Each variant is logged through the frozen convention
(:func:`heart.tracking.run.log_evaluation_run`) tagged
``run_kind=ablation`` plus ``feature_mode``, ``engineered_transforms``,
``engineered_features``, and ``n_features``. The ablation tag keeps these runs
out of the S03 leaderboard (which selects ``run_kind=final``) while keeping
them in the same experiment, so a feature comparison is directly comparable to
a model run. The run name embeds the mode slug
(``logistic-regression-<mode>-v1``) so modes are distinguishable in the UI.

Fail-soft, never silent
-----------------------
A mode that fails to build, fit, evaluate, or log is **recorded** as a failed
:class:`AblationRun` with a categorised error, and the harness continues. This
mirrors the battery's contract: the point is breadth, and a dropped variant
would quietly shrink the comparison. Pass ``strict=True`` to raise
:class:`AblationRunError` after the result is assembled (the result is still
attached).

Observability
-------------
:func:`run_ablation` logs each mode's status, primary metric, and MLflow run id
at ``INFO`` plus a summary line. :func:`render_ablation_report` renders the
markdown comparison table, and :func:`write_ablation_ledger` persists the
machine-readable result JSON atomically.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import REPORTS_DIR
from heart.runtime import (
    atomic_write_text,
    json_default,
    write_json_document,
)
from heart.data.pipeline import (
    PIPELINE_CATEGORICAL_COLUMNS,
    PIPELINE_NUMERIC_COLUMNS,
)
from heart.data.quality import DEFAULT_ZERO_POLICY, ZeroMedianImputer
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION
from heart.eval.contract import (
    PRIMARY_METRIC,
    MetricContractError,
    evaluate,
    evaluation_split_from_frames,
)
from heart.features.engineering import (
    CATEGORICAL_KIND,
    NUMERIC_KIND,
    TRANSFORM_NAMES,
    EngineeredFeatureTransformer,
    FeatureEngineeringError,
    resolve_selection,
    selected_column_names,
    selected_columns_by_kind,
)
from heart.models.baseline import BASELINE_MODEL_NAME, BASELINE_PARAMS
from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    TrackingConfig,
    TrackingError,
    configure_tracking,
)
from heart.tracking.run import log_evaluation_run

logger = logging.getLogger(__name__)

__all__ = [
    "RUN_KIND_TAG",
    "ABLATION_RUN_KIND",
    "FEATURE_MODE_TAG",
    "ENGINEERED_TRANSFORMS_TAG",
    "ENGINEERED_FEATURES_TAG",
    "N_FEATURES_TAG",
    "BASELINE_MODE_TAG",
    "SLICE_TAG",
    "SLICE_NAME",
    "STATUS_SUCCEEDED",
    "STATUS_FAILED",
    "ABLATION_LEDGER_FILENAME",
    "DEFAULT_REPORT_PATH",
    "AblationError",
    "AblationConfigError",
    "AblationDataError",
    "AblationRunError",
    "AblationLedgerError",
    "FeatureMode",
    "baseline_mode",
    "all_features_mode",
    "single_transform_modes",
    "leave_one_out_modes",
    "default_variants",
    "build_ablation_pipeline",
    "run_ablation",
    "ablation_feature_counts",
    "AblationRun",
    "AblationResult",
    "render_ablation_report",
    "write_ablation_ledger",
    "build_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: MLflow tag distinguishing an ablation run from final/trial runs.
RUN_KIND_TAG: str = "run_kind"

#: The run-kind value every ablation run carries.
ABLATION_RUN_KIND: str = "ablation"

#: Tag naming the feature mode (variant) a run evaluated.
FEATURE_MODE_TAG: str = "feature_mode"

#: Tag listing the engineered transform names enabled in the mode.
ENGINEERED_TRANSFORMS_TAG: str = "engineered_transforms"

#: Tag listing the engineered output columns enabled in the mode.
ENGINEERED_FEATURES_TAG: str = "engineered_features"

#: Tag recording how many columns entered the preprocessing transformer.
N_FEATURES_TAG: str = "n_features"

#: Tag marking whether the run used the baseline (no engineered) feature set.
BASELINE_MODE_TAG: str = "baseline_feature_mode"

#: Tag recording the slice that produced the run.
SLICE_TAG: str = "slice"

#: The slice value written on every ablation run.
SLICE_NAME: str = "S04"

#: A mode whose pipeline built, fit, evaluated, and logged successfully.
STATUS_SUCCEEDED: str = "succeeded"

#: A mode that failed somewhere in the harness (still recorded, never dropped).
STATUS_FAILED: str = "failed"

#: Default filename for a persisted ablation ledger.
ABLATION_LEDGER_FILENAME: str = "ablation_ledger.json"

#: Where the human-readable ablation report is published.
DEFAULT_REPORT_PATH: Path = Path(REPORTS_DIR) / "ablation.md"

#: Error categories recorded on a failed run.
ERROR_CATEGORY_ENGINEERING: str = "engineering"
ERROR_CATEGORY_EVALUATION: str = "evaluation"
ERROR_CATEGORY_TRACKING: str = "tracking"
ERROR_CATEGORY_FITTING: str = "fitting"
ERROR_CATEGORY_UNEXPECTED: str = "unexpected"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class AblationError(Exception):
    """Base class for every feature-ablation failure."""


class AblationConfigError(AblationError):
    """A feature mode or harness configuration is invalid."""


class AblationDataError(AblationError):
    """A frame handed to the harness is malformed."""


class AblationRunError(AblationError):
    """Strict mode: at least one feature mode failed (the result is attached)."""

    def __init__(self, message: str, result: "AblationResult") -> None:
        super().__init__(message)
        self.result = result

    @property
    def failed_modes(self) -> tuple[str, ...]:
        return tuple(
            run.mode.name for run in self.result.runs if not run.succeeded
        )


class AblationLedgerError(AblationError):
    """The ablation ledger could not be serialised or written."""


# ---------------------------------------------------------------------------
# The feature-mode abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureMode:
    """One named feature-set variant, defined over the T01 transform registry.

    A mode with an empty ``include`` is the **baseline feature set** — the raw
    eleven columns with no engineered additions. Any non-empty ``include``
    selects engineered transforms by their stable registry names; ``exclude``
    removes names afterwards. The selection is resolved through
    :func:`heart.features.engineering.resolve_selection`, so an unknown name or
    an empty-after-exclusion selection raises the same named engineering error
    family (wrapped as :class:`AblationConfigError`).

    Parameters
    ----------
    name:
        Stable, human-readable mode label. It is slugified into the MLflow run
        name and recorded in the ``feature_mode`` tag, so it must be unique
        within one ablation run.
    include:
        Engineered transform names to enable, or an empty tuple for the
        baseline feature set.
    exclude:
        Transform names to drop from ``include``.
    description:
        One-sentence rationale for the variant (recorded in the report).
    """

    name: str
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AblationConfigError(
                f"FeatureMode.name must be a non-empty string, got {self.name!r}."
            )
        include = tuple(self.include)
        exclude = tuple(self.exclude)
        if not all(isinstance(item, str) for item in (*include, *exclude)):
            raise AblationConfigError(
                f"FeatureMode {self.name!r} include/exclude must contain only "
                "transform names (strings)."
            )
        if not include and exclude:
            raise AblationConfigError(
                f"FeatureMode {self.name!r} excludes {list(exclude)} without "
                "including anything; an empty include is the baseline feature "
                "set and cannot be narrowed further."
            )
        if include:
            try:
                resolve_selection(list(include), exclude)
            except FeatureEngineeringError as exc:
                raise AblationConfigError(
                    f"FeatureMode {self.name!r} declares an invalid transform "
                    f"selection: {exc}"
                ) from exc
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "include", include)
        object.__setattr__(self, "exclude", exclude)

    @property
    def is_baseline(self) -> bool:
        """``True`` when the mode adds no engineered features."""
        return not self.include

    @property
    def engineered_transforms(self) -> tuple[str, ...]:
        """Enabled transform names, in resolved selection order."""
        if self.is_baseline:
            return ()
        return tuple(
            transform.name
            for transform in resolve_selection(list(self.include), self.exclude)
        )

    @property
    def engineered_columns(self) -> tuple[str, ...]:
        """Enabled engineered output columns, in resolved selection order."""
        if self.is_baseline:
            return ()
        return selected_column_names(list(self.include), self.exclude)

    @property
    def n_engineered_columns(self) -> int:
        return len(self.engineered_columns)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "description": self.description,
            "is_baseline": self.is_baseline,
            "engineered_transforms": list(self.engineered_transforms),
            "engineered_columns": list(self.engineered_columns),
            "n_engineered_columns": self.n_engineered_columns,
        }

    # -- constructors -----------------------------------------------------

    @classmethod
    def baseline(cls, name: str = "baseline") -> "FeatureMode":
        """The raw feature set: no engineered transforms."""
        return cls(
            name=name,
            include=(),
            description="Baseline feature set (raw schema, no engineered columns).",
        )

    @classmethod
    def all_features(cls, name: str = "all-engineered") -> "FeatureMode":
        """Every engineered transform enabled together."""
        return cls(
            name=name,
            include=TRANSFORM_NAMES,
            description="All engineered transforms enabled together.",
        )

    @classmethod
    def from_include(
        cls,
        name: str,
        include: Sequence[str],
        exclude: Sequence[str] = (),
        description: str = "",
    ) -> "FeatureMode":
        """Build a mode from an explicit ``include``/``exclude`` selection."""
        return cls(
            name=name,
            include=tuple(include),
            exclude=tuple(exclude),
            description=description,
        )

    @classmethod
    def leave_one_out(
        cls, dropped: str, *, name: str | None = None
    ) -> "FeatureMode":
        """All engineered transforms except ``dropped`` (ablation in reverse)."""
        if dropped not in TRANSFORM_NAMES:
            raise AblationConfigError(
                f"Cannot build a leave-one-out mode for unknown transform "
                f"{dropped!r}; declared transforms are {list(TRANSFORM_NAMES)}."
            )
        return cls(
            name=name or f"without-{dropped}",
            include=tuple(t for t in TRANSFORM_NAMES if t != dropped),
            description=f"All engineered transforms except {dropped!r}.",
        )


def baseline_mode() -> FeatureMode:
    """The baseline feature-set mode (no engineered columns)."""
    return FeatureMode.baseline()


def all_features_mode() -> FeatureMode:
    """The all-engineered feature-set mode."""
    return FeatureMode.all_features()


def single_transform_modes() -> tuple[FeatureMode, ...]:
    """One mode per declared transform (baseline plus exactly that transform)."""
    return tuple(
        FeatureMode.from_include(
            name=f"add-{name}",
            include=(name,),
            description=f"Baseline plus the {name!r} engineered transform.",
        )
        for name in TRANSFORM_NAMES
    )


def leave_one_out_modes() -> tuple[FeatureMode, ...]:
    """One mode per transform, dropping it from the full engineered set."""
    return tuple(FeatureMode.leave_one_out(name) for name in TRANSFORM_NAMES)


def default_variants() -> tuple[FeatureMode, ...]:
    """The default ablation family: baseline, each single add, and all.

    Every single-transform mode measures one transform's marginal contribution
    over the baseline; the all-engineered mode measures the ceiling. The
    leave-one-out family (:func:`leave_one_out_modes`) is available for the
    reverse ablation when marginal contributions are masked by correlated
    raw columns.
    """
    return (
        baseline_mode(),
        *single_transform_modes(),
        all_features_mode(),
    )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _mode_feature_columns(mode: FeatureMode) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(numeric, categorical)`` columns the mode's transformer sees.

    The declared schema columns are always first; engineered columns are
    appended by kind so the ``ColumnTransformer`` scales the numeric ones and
    one-hot encodes the categorical ones. The column *definitions* stay in
    :mod:`heart.data.pipeline` and :mod:`heart.features.engineering`; this
    function only composes them for one mode.
    """
    numeric = list(PIPELINE_NUMERIC_COLUMNS)
    categorical = list(PIPELINE_CATEGORICAL_COLUMNS)
    if not mode.is_baseline:
        include = list(mode.include)
        numeric.extend(
            selected_columns_by_kind(NUMERIC_KIND, include, mode.exclude)
        )
        categorical.extend(
            selected_columns_by_kind(CATEGORICAL_KIND, include, mode.exclude)
        )
    return tuple(numeric), tuple(categorical)


def build_ablation_pipeline(
    mode: FeatureMode, params: Mapping[str, object] | None = None
) -> Pipeline:
    """Build the unfitted pipeline that scores one feature mode.

    The chain is identical for every mode so the only variable is the feature
    representation::

        zero_policy -> engineering -> features -> clf

    * ``zero_policy`` — :class:`~heart.data.quality.ZeroMedianImputer` over the
      declared ``RestingBP``/``Cholesterol`` sentinel policy (fit on training
      rows only).
    * ``engineering`` — :class:`~heart.features.engineering.EngineeredFeatureTransformer`
      for a non-baseline mode, or ``"passthrough"`` for the baseline mode.
    * ``features`` — standard scaling of numeric columns and one-hot encoding
      of categorical columns, including the engineered members by kind.
    * ``clf`` — ``LogisticRegression`` with the published baseline parameters
      (or the caller's overrides).

    The baseline mode is numerically equivalent to
    :func:`heart.models.baseline.build_baseline_model`: the extra passthrough
    step is a no-op and the zero-policy/transformer/classifier are identical.
    """
    if not isinstance(mode, FeatureMode):
        raise AblationConfigError(
            f"mode must be a FeatureMode, got {type(mode).__name__}."
        )
    resolved_params = dict(BASELINE_PARAMS if params is None else params)
    numeric, categorical = _mode_feature_columns(mode)
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric)),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=float
                ),
                list(categorical),
            ),
        ],
        remainder="drop",
    )
    engineering: object = (
        "passthrough"
        if mode.is_baseline
        else EngineeredFeatureTransformer(
            include=list(mode.include),
            exclude=tuple(mode.exclude),
            keep_original=True,
        )
    )
    return Pipeline(
        steps=[
            (
                "zero_policy",
                ZeroMedianImputer(
                    columns=tuple(DEFAULT_ZERO_POLICY.columns),
                    sentinel=DEFAULT_ZERO_POLICY.sentinel,
                ),
            ),
            ("engineering", engineering),
            ("features", column_transformer),
            ("clf", LogisticRegression(**resolved_params)),
        ]
    )


def ablation_feature_counts(pipeline: Pipeline) -> tuple[int, int]:
    """Return ``(n_input_columns, n_transformed_features)`` for a fitted pipeline.

    ``n_input_columns`` is how many columns reached the ``ColumnTransformer``
    (raw plus engineered); ``n_transformed_features`` is the width of the
    scaled/encoded design matrix the classifier actually saw.
    """
    if not isinstance(pipeline, Pipeline):
        raise AblationError(
            f"pipeline must be a sklearn Pipeline, got {type(pipeline).__name__}."
        )
    transformer = pipeline.named_steps.get("features")
    if transformer is None or not hasattr(transformer, "n_features_in_"):
        raise AblationError(
            "Pipeline is not fitted: the 'features' step has no n_features_in_. "
            "Fit the pipeline before counting features."
        )
    n_input = int(transformer.n_features_in_)
    n_output = int(len(transformer.get_feature_names_out()))
    return n_input, n_output


# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationRun:
    """One feature mode's outcome within an ablation run (never dropped)."""

    mode: FeatureMode
    status: str
    duration_seconds: float
    n_engineered_columns: int
    n_input_features: int | None = None
    n_transformed_features: int | None = None
    metrics: dict[str, object] | None = None
    run_id: str | None = None
    experiment_name: str | None = None
    error: str | None = None
    error_type: str | None = None
    error_category: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == STATUS_SUCCEEDED

    @property
    def primary_metric(self) -> float | None:
        if self.metrics is None:
            return None
        return float(self.metrics[PRIMARY_METRIC])

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.to_dict(),
            "status": self.status,
            "duration_seconds": float(self.duration_seconds),
            "n_engineered_columns": int(self.n_engineered_columns),
            "n_input_features": self.n_input_features,
            "n_transformed_features": self.n_transformed_features,
            "metric_dict": self.metrics,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "error": self.error,
            "error_type": self.error_type,
            "error_category": self.error_category,
        }


@dataclass(frozen=True)
class AblationResult:
    """Everything one ablation run produced."""

    model_name: str
    split_version: str
    experiment_name: str
    train_rows: int
    test_rows: int
    params: dict[str, object]
    runs: tuple[AblationRun, ...]
    generated_at: str
    ledger_path: Path | None = None

    @property
    def n_modes(self) -> int:
        return len(self.runs)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for run in self.runs if run.succeeded)

    @property
    def n_failed(self) -> int:
        return sum(1 for run in self.runs if not run.succeeded)

    @property
    def failed_modes(self) -> tuple[str, ...]:
        return tuple(run.mode.name for run in self.runs if not run.succeeded)

    @property
    def baseline_run(self) -> AblationRun | None:
        """The first succeeded baseline (no engineered) mode, if present."""
        for run in self.runs:
            if run.succeeded and run.mode.is_baseline:
                return run
        return None

    def run_for(self, mode_name: str) -> AblationRun | None:
        for run in self.runs:
            if run.mode.name == mode_name:
                return run
        return None

    def deltas(self, metric: str = PRIMARY_METRIC) -> dict[str, float | None]:
        """Per-mode ``metric - baseline``, keyed by mode name.

        A failed mode or a missing baseline yields ``None`` for that entry
        rather than a fabricated number.
        """
        baseline = self.baseline_run
        baseline_value: float | None = None
        if baseline is not None and baseline.metrics is not None:
            baseline_value = float(baseline.metrics[metric])
        deltas: dict[str, float | None] = {}
        for run in self.runs:
            if not run.succeeded or run.metrics is None or baseline_value is None:
                deltas[run.mode.name] = None
                continue
            deltas[run.mode.name] = float(run.metrics[metric]) - baseline_value
        return deltas

    def ranked(self, metric: str = PRIMARY_METRIC) -> tuple[AblationRun, ...]:
        """Succeeded runs ordered by ``metric`` descending (failed runs last)."""
        return tuple(
            sorted(
                self.runs,
                key=lambda run: (
                    run.primary_metric is None,
                    -(float(run.metrics[metric]) if run.metrics else 0.0),
                ),
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "split_version": self.split_version,
            "experiment_name": self.experiment_name,
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "params": dict(self.params),
            "n_modes": self.n_modes,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "failed_modes": list(self.failed_modes),
            "deltas": self.deltas(),
            "generated_at": self.generated_at,
            "runs": [run.to_dict() for run in self.runs],
        }


# ---------------------------------------------------------------------------
# Harness orchestration
# ---------------------------------------------------------------------------


def _require_frame(frame: object, *, role: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise AblationDataError(
            f"{role} must be a pandas.DataFrame, got {type(frame).__name__}."
        )
    if frame.empty:
        raise AblationDataError(f"{role} is empty; nothing to ablate on.")
    missing = [
        column
        for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise AblationDataError(
            f"{role} is missing column(s) {missing}; load the S01 split with "
            "heart.data.split.load_split_frames."
        )
    return frame


def _resolve_modes(modes: Sequence[FeatureMode] | None) -> tuple[FeatureMode, ...]:
    if modes is None:
        return default_variants()
    if isinstance(modes, (str, bytes)):
        raise AblationConfigError(
            "modes must be a sequence of FeatureMode objects, not a bare string."
        )
    resolved = tuple(modes)
    if not resolved:
        raise AblationConfigError(
            "modes is empty; pass at least one FeatureMode (or omit it for the "
            "default ablation family)."
        )
    seen: set[str] = set()
    for mode in resolved:
        if not isinstance(mode, FeatureMode):
            raise AblationConfigError(
                f"modes contains {type(mode).__name__}; expected FeatureMode."
            )
        if mode.name in seen:
            raise AblationConfigError(
                f"modes contains duplicate mode name {mode.name!r}; mode names "
                "must be unique within one ablation run."
            )
        seen.add(mode.name)
    return resolved


def _mode_model_name(base_model_name: str, mode: FeatureMode) -> str:
    """Compose the MLflow model name so each mode slugifies uniquely."""
    return f"{base_model_name} [{mode.name}]"


def _mode_params(
    mode: FeatureMode, params: Mapping[str, object]
) -> dict[str, object]:
    merged = dict(params)
    merged["feature_mode"] = mode.name
    merged["engineered_transforms"] = list(mode.engineered_transforms)
    merged["n_engineered_columns"] = mode.n_engineered_columns
    return merged


def _mode_tags(
    mode: FeatureMode,
    *,
    base_model_name: str,
    n_input_features: int | None,
    n_transformed_features: int | None,
) -> dict[str, object]:
    tags: dict[str, object] = {
        RUN_KIND_TAG: ABLATION_RUN_KIND,
        FEATURE_MODE_TAG: mode.name,
        ENGINEERED_TRANSFORMS_TAG: ",".join(mode.engineered_transforms),
        ENGINEERED_FEATURES_TAG: ",".join(mode.engineered_columns),
        BASELINE_MODE_TAG: "true" if mode.is_baseline else "false",
        "base_model_name": base_model_name,
        SLICE_TAG: SLICE_NAME,
    }
    if n_input_features is not None:
        tags[N_FEATURES_TAG] = str(int(n_input_features))
    if n_transformed_features is not None:
        tags["n_transformed_features"] = str(int(n_transformed_features))
    return tags


def _classify_error(exc: BaseException, stage: str) -> str:
    if isinstance(exc, FeatureEngineeringError):
        return ERROR_CATEGORY_ENGINEERING
    if isinstance(exc, MetricContractError):
        return ERROR_CATEGORY_EVALUATION
    if isinstance(exc, TrackingError):
        return ERROR_CATEGORY_TRACKING
    if stage == "fit":
        return ERROR_CATEGORY_FITTING
    return ERROR_CATEGORY_UNEXPECTED


ModelBuilder = Callable[[FeatureMode], object]


def _run_one_mode(
    mode: FeatureMode,
    *,
    train_frame: pd.DataFrame,
    test_split: object,
    builder: ModelBuilder,
    params: Mapping[str, object],
    base_model_name: str,
    split_version: str,
    log_to_mlflow: bool,
    config: TrackingConfig | None,
) -> AblationRun:
    """Build, fit, evaluate, and log one feature mode (fail-soft contract)."""
    start = time.perf_counter()

    def _duration() -> float:
        return round(time.perf_counter() - start, 6)

    stage = "build"
    try:
        pipeline = builder(mode)
        stage = "fit"
        pipeline.fit(
            train_frame[list(FEATURE_COLUMNS)],
            train_frame[TARGET_COLUMN],
        )
        stage = "evaluate"
        metrics = evaluate(pipeline, test_split)
        stage = "count"
        n_input, n_transformed = ablation_feature_counts(pipeline)

        run_id: str | None = None
        experiment_name: str | None = config.experiment_name if config else None
        if log_to_mlflow:
            stage = "log"
            run = log_evaluation_run(
                metrics,
                model_name=_mode_model_name(base_model_name, mode),
                split_version=split_version,
                params=_mode_params(mode, params),
                tags=_mode_tags(
                    mode,
                    base_model_name=base_model_name,
                    n_input_features=n_input,
                    n_transformed_features=n_transformed,
                ),
                config=config,
            )
            run_id = run.run_id
            experiment_name = run.experiment_name

        record = AblationRun(
            mode=mode,
            status=STATUS_SUCCEEDED,
            duration_seconds=_duration(),
            n_engineered_columns=mode.n_engineered_columns,
            n_input_features=n_input,
            n_transformed_features=n_transformed,
            metrics=metrics,
            run_id=run_id,
            experiment_name=experiment_name,
        )
        logger.info(
            "Ablation mode %s succeeded: %s=%.4f features=%s run=%s (%.2fs)",
            mode.name,
            PRIMARY_METRIC,
            float(metrics[PRIMARY_METRIC]),
            n_input,
            run_id or "not logged",
            record.duration_seconds,
        )
        return record
    except Exception as exc:  # noqa: BLE001 - every mode failure is recorded
        category = _classify_error(exc, stage)
        record = AblationRun(
            mode=mode,
            status=STATUS_FAILED,
            duration_seconds=_duration(),
            n_engineered_columns=mode.n_engineered_columns,
            error=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            error_category=category,
        )
        logger.error(
            "Ablation mode %s failed [%s @ %s]: %s",
            mode.name,
            category,
            stage,
            record.error,
        )
        return record


def run_ablation(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    modes: Sequence[FeatureMode] | None = None,
    params: Mapping[str, object] | None = None,
    model_name: str = BASELINE_MODEL_NAME,
    split_version: str = SPLIT_VERSION,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_dir: str | Path | None = None,
    log_to_mlflow: bool = True,
    model_builder: ModelBuilder | None = None,
    ledger_path: str | Path | None = None,
    strict: bool = False,
) -> AblationResult:
    """Evaluate the baseline and each feature-mode variant through the contract.

    Parameters
    ----------
    train_frame / test_frame:
        The S01 split frames. Every pipeline is fit on ``train_frame`` only and
        scored once on ``test_frame`` through :func:`heart.eval.contract.evaluate`.
    modes:
        The feature modes to evaluate; defaults to :func:`default_variants`
        (baseline, each single-transform add, and all-engineered). Mode names
        must be unique.
    params:
        Classifier hyperparameters; defaults to the published baseline params.
    model_name:
        Base model label slugified into the run name; the mode name is appended.
    split_version:
        Split version recorded on every run.
    experiment_name / tracking_dir:
        The MLflow experiment and local store (defaults to the portfolio
        experiment and ``experiments/mlruns``).
    log_to_mlflow:
        When ``False`` the harness evaluates without logging (diagnostics/tests).
    model_builder:
        Optional ``mode -> estimator`` override, defaulting to
        :func:`build_ablation_pipeline`. Used by T03/T04 to ablate a different
        model family without changing the harness.
    ledger_path:
        When given, the machine-readable result JSON is written here atomically.
    strict:
        When ``True`` a failed mode raises :class:`AblationRunError` *after* the
        result is assembled; the attached ``result`` still lists every mode.
    """
    resolved_modes = _resolve_modes(modes)
    train = _require_frame(train_frame, role="train_frame")
    test = _require_frame(test_frame, role="test_frame")
    resolved_params = dict(BASELINE_PARAMS if params is None else params)
    builder: ModelBuilder = model_builder or (
        lambda mode: build_ablation_pipeline(mode, resolved_params)
    )

    test_split = evaluation_split_from_frames(
        test, name=f"{split_version}/test"
    )

    config: TrackingConfig | None = None
    if log_to_mlflow:
        config = configure_tracking(
            experiment_name=experiment_name, tracking_dir=tracking_dir
        )

    logger.info(
        "Starting ablation run: %d mode(s), model=%s, split=%s, experiment=%s",
        len(resolved_modes),
        model_name,
        split_version,
        experiment_name if log_to_mlflow else "not logged",
    )

    runs = tuple(
        _run_one_mode(
            mode,
            train_frame=train,
            test_split=test_split,
            builder=builder,
            params=resolved_params,
            base_model_name=model_name,
            split_version=split_version,
            log_to_mlflow=log_to_mlflow,
            config=config,
        )
        for mode in resolved_modes
    )

    result = AblationResult(
        model_name=model_name,
        split_version=split_version,
        experiment_name=experiment_name,
        train_rows=int(len(train)),
        test_rows=int(len(test)),
        params=resolved_params,
        runs=runs,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ledger_path=None,
    )

    destination = Path(ledger_path) if ledger_path is not None else None
    if destination is not None:
        result = replace(
            result, ledger_path=write_ablation_ledger(result, destination)
        )

    logger.info(
        "Ablation run finished: %d/%d mode(s) succeeded, %d failed%s",
        result.n_succeeded,
        result.n_modes,
        result.n_failed,
        f" (failed: {', '.join(result.failed_modes)})"
        if result.n_failed
        else "",
    )

    if strict and result.n_failed:
        raise AblationRunError(
            f"{result.n_failed} of {result.n_modes} feature mode(s) failed "
            f"({', '.join(result.failed_modes)}); see the attached result for "
            "per-mode errors.",
            result,
        )
    return result


# ---------------------------------------------------------------------------
# Report / ledger
# ---------------------------------------------------------------------------


def write_ablation_ledger(result: AblationResult, path: str | Path) -> Path:
    """Atomically write the machine-readable ablation result JSON to ``path``."""
    if not isinstance(result, AblationResult):
        raise AblationLedgerError(
            f"result must be an AblationResult, got {type(result).__name__}."
        )
    destination = write_json_document(
        path,
        result.to_dict(),
        error_factory=AblationLedgerError,
        label="ablation ledger",
    )
    logger.info(
        "Wrote ablation ledger to %s (%d mode(s), %d/%d succeeded)",
        destination,
        result.n_modes,
        result.n_succeeded,
        result.n_modes,
    )
    return destination


def _num(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def render_ablation_report(result: AblationResult) -> str:
    """Render the ablation comparison as a markdown report."""
    if not isinstance(result, AblationResult):
        raise AblationError(
            f"result must be an AblationResult, got {type(result).__name__}."
        )
    deltas = result.deltas()
    baseline = result.baseline_run

    lines: list[str] = []
    lines.append("# Feature Ablation Report")
    lines.append("")
    lines.append(
        "_Generated by `heart.features.ablation` (S04/T02). Every row is one "
        "feature mode evaluated on the same held-out split through the shared "
        "`heart.eval.contract.evaluate` path and logged to MLflow tagged "
        "`run_kind=ablation`._"
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- base model: `{result.model_name}`")
    lines.append(
        f"- split: `{result.split_version}` "
        f"(train {result.train_rows} / test {result.test_rows})"
    )
    lines.append(f"- classifier params: `{result.params}`")
    lines.append(
        f"- modes evaluated: {result.n_modes} "
        f"({result.n_succeeded} succeeded, {result.n_failed} failed)"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| mode | status | ROC-AUC | Δ ROC-AUC | accuracy | pr_auc | "
        "eng. cols | input cols | transformed | run_id |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for run in result.ranked():
        if run.succeeded and run.metrics is not None:
            auc = _num(run.metrics[PRIMARY_METRIC])
            delta = deltas.get(run.mode.name)
            delta_text = "-" if delta is None else f"{delta:+.4f}"
            accuracy = _num(run.metrics["accuracy"])
            pr_auc = _num(run.metrics["pr_auc"])
        else:
            auc = delta_text = accuracy = pr_auc = "-"
        lines.append(
            f"| `{run.mode.name}` | {run.status} | {auc} | {delta_text} | "
            f"{accuracy} | {pr_auc} | {run.n_engineered_columns} | "
            f"{run.n_input_features if run.n_input_features is not None else '-'} | "
            f"{run.n_transformed_features if run.n_transformed_features is not None else '-'} | "
            f"{run.run_id or '-'} |"
        )
    lines.append("")
    if baseline is not None:
        lines.append(
            f"Baseline reference: `{baseline.mode.name}` with "
            f"ROC-AUC {_num(baseline.primary_metric)}."
        )
        lines.append("")
        leaders = [
            run
            for run in result.ranked()
            if run.succeeded
            and (deltas.get(run.mode.name) or 0.0) > 0.0
        ]
        if leaders:
            lines.append(
                "Modes that beat the baseline on ROC-AUC: "
                + ", ".join(
                    f"`{run.mode.name}` ({deltas[run.mode.name]:+.4f})"
                    for run in leaders
                )
                + "."
            )
        else:
            lines.append("No feature mode beat the baseline on ROC-AUC.")
        lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- experiment: `{result.experiment_name}`")
    lines.append("- tags: `run_kind=ablation`, `feature_mode`, "
                 "`engineered_transforms`, `n_features`")
    lines.append(f"- generated at: {result.generated_at}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m heart.features.ablation")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _write_text_atomic(path: Path, content: str) -> None:
    """Thin wrapper delegating to heart.runtime.atomic_write_text."""
    atomic_write_text(path, content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.features.ablation",
        description=(
            "Evaluate the baseline feature set and each engineered-feature "
            "variant through the shared evaluation contract and log them to "
            "MLflow."
        ),
    )
    parser.add_argument(
        "--split-version", default=SPLIT_VERSION, help="S01 split version."
    )
    parser.add_argument(
        "--family",
        choices=("default", "leave-one-out"),
        default="default",
        help="Ablation family: default (baseline + single adds + all) or "
        "leave-one-out (all minus each transform).",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated mode names to run instead of the full family.",
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
        "--ledger",
        default=None,
        help="Optional path for the ablation ledger JSON.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip MLflow logging."
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip writing the report file."
    )
    return parser


def _select_family(args: argparse.Namespace) -> tuple[FeatureMode, ...]:
    family = (
        default_variants()
        if args.family == "default"
        else leave_one_out_modes()
    )
    if args.modes is None:
        return family
    requested = [part.strip() for part in args.modes.split(",") if part.strip()]
    by_name = {mode.name: mode for mode in family}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise AblationConfigError(
            f"Unknown mode name(s) {unknown}; declared modes for family "
            f"'{args.family}' are {list(by_name)}."
        )
    return tuple(by_name[name] for name in requested)


def main(argv: list[str] | None = None) -> int:
    from heart.data.split import SplitError, load_split_frames

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        modes = _select_family(args)
    except AblationConfigError as exc:
        print(f"configuration error: {exc}")
        return 2

    try:
        train_frame, test_frame = load_split_frames(args.split_version)
    except SplitError as exc:
        print(f"could not load split {args.split_version!r}: {exc}")
        return 1

    try:
        result = run_ablation(
            train_frame,
            test_frame,
            modes=modes,
            split_version=args.split_version,
            experiment_name=args.experiment,
            tracking_dir=args.tracking_dir,
            log_to_mlflow=not args.no_mlflow,
            ledger_path=args.ledger,
        )
    except AblationError as exc:
        print(f"ablation error: {exc}")
        return 1

    if not args.no_report:
        destination = Path(args.report_path)
        try:
            _write_text_atomic(destination, render_ablation_report(result))
        except OSError as exc:
            print(f"could not write report to {destination}: {exc}")
            return 1
        print(f"wrote ablation report: {destination}")

    print(f"modes: {result.n_succeeded}/{result.n_modes} succeeded")
    for run in result.ranked():
        metric = "-" if run.primary_metric is None else _num(run.primary_metric)
        print(f"  {run.mode.name}: roc_auc={metric} ({run.status})")
    return 0 if result.n_succeeded > 0 else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
