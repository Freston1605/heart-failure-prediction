"""Shared evaluation contract.

One metric producer (:mod:`heart.eval.metrics`) and one evaluation entry point
(:func:`heart.eval.contract.evaluate`) serve every model in the portfolio.
Import from here, not from the submodules, unless you need the arithmetic
primitives directly.
"""

from heart.eval.contract import (
    EvaluationSplit,
    MetricContractError,
    MetricSchemaError,
    MetricValueError,
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
    DEFAULT_CALIBRATION_BINS,
    METRIC_KEYS,
    METRIC_SCHEMA_VERSION,
    PRIMARY_METRIC,
    SCALAR_METRIC_KEYS,
    STRUCTURED_METRIC_KEYS,
    CalibrationReport,
    ConfusionMatrix,
    MetricComputationError,
    calibration_report,
    classification_scores,
    compute_metric_dict,
    confusion_counts,
)

__all__ = [
    # schema
    "METRIC_KEYS",
    "METRIC_SCHEMA_VERSION",
    "PRIMARY_METRIC",
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
