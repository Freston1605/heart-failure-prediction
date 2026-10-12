"""Shared evaluation contract.

One metric producer (:mod:`heart.eval.metrics`) and one evaluation entry point
(:func:`heart.eval.contract.evaluate`) serve every model in the portfolio.
Import from here, not from the submodules, unless you need the arithmetic
primitives directly.

The package exports below are resolved lazily (PEP 562 ``__getattr__``): the
declarations module :mod:`heart.eval.treatment_declared` is stdlib-only by
design (D032: the S01 report and the S02 gate runner must be able to read the
declared candidates/gate constants without paying for the sklearn/pandas
stack), so the package import itself must not eagerly pull the numeric stack
in either. Any listed name triggers the import of its owning submodule on
first access; ``from heart.eval.contract import evaluate`` style submodule
imports are unaffected.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static-analysis surface only
    from heart.eval.contract import (
        EvaluationSplit,
        MetricContractError,
        MetricSchemaError,
        MissingMetricError,
        ModelInterfaceError,
        PredictionShapeError,
        SplitContractError,
        UnexpectedMetricError,
        describe_metric_schema,
        evaluate,
        evaluation_split_from_data_split,
        evaluation_split_from_frames,
        flatten_metrics,
        validate_metric_dict,
    )
    from heart.eval.metrics import (
        CalibrationReport,
        ConfusionMatrix,
        DEFAULT_CALIBRATION_BINS,
        LOSS_METRIC_KEYS,
        METRIC_KEYS,
        METRIC_SCHEMA_VERSION,
        MetricComputationError,
        PRIMARY_METRIC,
        RATE_METRIC_KEYS,
        SCALAR_METRIC_KEYS,
        STRUCTURED_METRIC_KEYS,
        calibration_report,
        classification_scores,
        compute_metric_dict,
        confusion_counts,
    )

#: Owned-name map: every historical package export -> the submodule that
#: defines it. Kept verbatim so the lazy resolution can never drift from the
#: previous eager ``__all__``.
_EXPORTED_FROM: dict[str, tuple[str, ...]] = {
    "heart.eval.contract": (
        "EvaluationSplit",
        "MetricContractError",
        "MetricSchemaError",
        "MissingMetricError",
        "ModelInterfaceError",
        "PredictionShapeError",
        "SplitContractError",
        "UnexpectedMetricError",
        "MetricValueError",
        "describe_metric_schema",
        "evaluate",
        "evaluation_split_from_data_split",
        "evaluation_split_from_frames",
        "flatten_metrics",
        "validate_metric_dict",
    ),
    "heart.eval.metrics": (
        "DEFAULT_CALIBRATION_BINS",
        "LOSS_METRIC_KEYS",
        "METRIC_KEYS",
        "METRIC_SCHEMA_VERSION",
        "PRIMARY_METRIC",
        "RATE_METRIC_KEYS",
        "SCALAR_METRIC_KEYS",
        "STRUCTURED_METRIC_KEYS",
        "CalibrationReport",
        "ConfusionMatrix",
        "MetricComputationError",
        "calibration_report",
        "classification_scores",
        "compute_metric_dict",
        "confusion_counts",
    ),
}

__all__ = [
    # schema
    "METRIC_KEYS",
    "METRIC_SCHEMA_VERSION",
    "PRIMARY_METRIC",
    "RATE_METRIC_KEYS",
    "LOSS_METRIC_KEYS",
    "SCALAR_METRIC_KEYS",
    "STRUCTURED_METRIC_KEYS",
    "DEFAULT_CALIBRATION_BINS",
    # computation
    "compute_metric_dict",
    "classification_scores",
    "confusion_counts",
    "calibration_report",
    "ConfusionMatrix",
    "CalibrationReport",
    # contract
    "evaluate",
    "EvaluationSplit",
    "evaluation_split_from_data_split",
    "evaluation_split_from_frames",
    "validate_metric_dict",
    "describe_metric_schema",
    "flatten_metrics",
    # errors
    "MetricComputationError",
    "MetricContractError",
    "MetricSchemaError",
    "MissingMetricError",
    "UnexpectedMetricError",
    "MetricValueError",
    "SplitContractError",
    "ModelInterfaceError",
    "PredictionShapeError",
]


def __getattr__(name: str):
    """Resolve a listed export by importing only the submodule that owns it.

    Names outside :data:`_EXPORTED_FROM` raise the standard
    ``AttributeError``, exactly as the previous eager package did.
    """

    for module_name, owned in _EXPORTED_FROM.items():
        if name in owned:
            import importlib

            module = importlib.import_module(module_name)
            try:
                return getattr(module, name)
            except AttributeError as exc:  # pragma: no cover - defensive
                raise AttributeError(
                    f"module {module_name!r} no longer defines {name!r}; "
                    "update _EXPORTED_FROM in heart/eval/__init__.py."
                ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> "list[str]":
    return sorted(set(globals().keys()) | (__all__ and set(__all__)))
