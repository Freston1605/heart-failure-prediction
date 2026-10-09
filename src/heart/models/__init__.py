"""Portfolio models.

The baseline floor lives in :mod:`heart.models.baseline`; the throwaway models
that prove the evaluation contract is model-agnostic live in
:mod:`heart.models.throwaway`. From S03 on, the fixed classical battery is
declared in :mod:`heart.models.registry` (the specs) and
:mod:`heart.models.spaces` (the Optuna search spaces); the tuning runner and
battery entry point live in ``heart.tuning`` and
:mod:`heart.models.run_battery`.

Import the public surface from here; the registry's :func:`resolve_spec` is
the canonical battery lookup (the throwaway proof keeps its own inside
:mod:`heart.models.throwaway`).
"""

from heart.models.baseline import (
    BASELINE_MODEL_NAME,
    BASELINE_PARAMS,
    BASELINE_SPLIT_VERSION,
    DEFAULT_REPORT_PATH,
    BaselineDataError,
    BaselineError,
    BaselineReportError,
    BaselineResult,
    build_baseline_model,
    render_baseline_report,
    run_baseline,
    train_baseline,
)
from heart.models.registry import (
    BATTERY_MODEL_TYPES,
    BATTERY_MODELS,
    BATTERY_SIZE,
    BATTERY_SPECS,
    MissingHyperparameterError,
    ModelDependencyError,
    ModelSpec,
    RegistryError,
    UnknownHyperparameterError,
    UnknownModelTypeError,
    build_estimator,
    build_pipeline,
    registry_report,
    resolve_spec,
    validate_params,
)
from heart.models.spaces import (
    CATEGORICAL_KIND,
    FLOAT_KIND,
    INT_KIND,
    DuplicateParameterError,
    InvalidParameterSpecError,
    ParamSpec,
    SearchSpace,
    SearchSpaceError,
    UnknownParameterError,
    default_params,
    sample_params,
)
from heart.models.throwaway import (
    DEFAULT_MODEL_TYPE,
    THROWAWAY_MODEL_NAME,
    THROWAWAY_MODEL_TYPES,
    THROWAWAY_PARAMS,
    THROWAWAY_SPECS,
    THROWAWAY_SPLIT_VERSION,
    ThrowawayDataError,
    ThrowawayError,
    ThrowawayResult,
    ThrowawaySpec,
    UnknownThrowawayModelError,
    build_throwaway_model,
    run_all_throwaway,
    run_throwaway,
    train_throwaway,
)

__all__ = [
    # baseline
    "BASELINE_MODEL_NAME",
    "BASELINE_PARAMS",
    "BASELINE_SPLIT_VERSION",
    "DEFAULT_REPORT_PATH",
    "BaselineResult",
    "build_baseline_model",
    "train_baseline",
    "run_baseline",
    "render_baseline_report",
    "BaselineError",
    "BaselineDataError",
    "BaselineReportError",
    # battery registry
    "BATTERY_MODEL_TYPES",
    "BATTERY_MODELS",
    "BATTERY_SIZE",
    "BATTERY_SPECS",
    "ModelSpec",
    "build_estimator",
    "build_pipeline",
    "resolve_spec",
    "validate_params",
    "registry_report",
    "RegistryError",
    "UnknownModelTypeError",
    "UnknownHyperparameterError",
    "MissingHyperparameterError",
    "ModelDependencyError",
    # search spaces
    "ParamSpec",
    "SearchSpace",
    "FLOAT_KIND",
    "INT_KIND",
    "CATEGORICAL_KIND",
    "default_params",
    "sample_params",
    "SearchSpaceError",
    "InvalidParameterSpecError",
    "UnknownParameterError",
    "DuplicateParameterError",
    # throwaway proof
    "DEFAULT_MODEL_TYPE",
    "THROWAWAY_MODEL_NAME",
    "THROWAWAY_MODEL_TYPES",
    "THROWAWAY_PARAMS",
    "THROWAWAY_SPECS",
    "THROWAWAY_SPLIT_VERSION",
    "ThrowawaySpec",
    "ThrowawayResult",
    "build_throwaway_model",
    "train_throwaway",
    "run_throwaway",
    "run_all_throwaway",
    "ThrowawayError",
    "ThrowawayDataError",
    "UnknownThrowawayModelError",
]