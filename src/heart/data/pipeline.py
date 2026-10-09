"""Leakage-safe preprocessing pipeline and automated leakage detection.

Why this module exists
----------------------
A split is only leakage-safe if **every data-dependent transform is fit on the
training rows alone**. It is easy to say that and easy to break it: a
``StandardScaler`` or a ``ZeroMedianImputer`` fit on the full frame before the
split silently carries test-set statistics into training. The benchmark then
looks better than it is.

This module does two things:

1. :func:`build_preprocessing_pipeline` assembles the declared transform chain
   (zero-as-missing median imputation, then numeric scaling and one-hot
   encoding of the categorical features) as a single scikit-learn ``Pipeline``
   that is fit on training rows only.
2. :func:`assert_fit_on_training_only` and :func:`assert_no_test_influence`
   **prove** that guarantee automatically. They are the executable form of the
   claim, so a future refactor that re-introduces leakage fails a test instead
   of quietly inflating a score.

Two independent detectors
-------------------------
* **Fit-scope audit** — fit a reference pipeline on the training rows, fit a
  second reference on the full dataset, and compare both against the fitted
  statistics of the pipeline under test. The pipeline must match the
  train-only fit. If it instead matches the full-data fit, the transform saw
  held-out rows.
* **Influence probe** — perturb the *test* rows with extreme values and re-run
  the fitting path on the training indices. An honest path is unchanged; a
  path that reads outside its training indices produces different statistics.
  The probe takes the fitting function as an argument precisely so a
  deliberately leaky path can be handed in and shown to be caught.

Splits are formed **before** any transform is fit: :func:`fit_on_train` takes a
split's training indices and never touches the held-out rows.

Observability
-------------
Fitting logs the row count and the learned medians at ``INFO``. Both audits
return serialisable dataclasses (``to_dict``) and raise :class:`DataLeakageError`
with the differing statistic named in the message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.data.quality import DEFAULT_ZERO_POLICY, ZeroMedianImputer
from heart.data.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared shape of the transform chain
# ---------------------------------------------------------------------------

#: Feature-only categorical columns (the target is never a feature).
PIPELINE_CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(
    column for column in CATEGORICAL_COLUMNS if column != TARGET_COLUMN
)

#: Feature-only numeric columns.
PIPELINE_NUMERIC_COLUMNS: tuple[str, ...] = tuple(NUMERIC_COLUMNS)

#: Value written into numeric test rows by :func:`perturb_test_rows`. It is far
#: outside every real feature range, so it cannot be mistaken for a measurement.
PERTURB_NUMERIC_VALUE: int = 1_000_000

#: Value written into categorical test rows by :func:`perturb_test_rows`.
PERTURB_CATEGORICAL_VALUE: str = "<LEAK-PROBE>"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Base class for preprocessing-pipeline failures."""


class FeatureFrameError(PipelineError):
    """A frame handed to the pipeline is missing declared feature columns."""


class DataLeakageError(PipelineError):
    """A fitted transform was influenced by rows outside the training indices."""


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the declared feature columns."""
    if not isinstance(frame, pd.DataFrame):
        raise FeatureFrameError(
            f"Expected a pandas.DataFrame, got {type(frame).__name__}."
        )
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureFrameError(
            f"Frame is missing feature column(s) {missing}. Present columns: "
            f"{list(frame.columns)}. Load the dataset through "
            "heart.data.load.load_dataset first."
        )
    return frame[list(FEATURE_COLUMNS)].copy()


def build_preprocessing_pipeline(policy=DEFAULT_ZERO_POLICY) -> Pipeline:
    """Build the declared, unfitted preprocessing chain.

    Steps, in order:

    1. ``zero_policy`` — :class:`~heart.data.quality.ZeroMedianImputer` turns
       impossible zeros in ``RestingBP`` / ``Cholesterol`` into missing and
       fills them with the **fit-frame** median.
    2. ``features`` — a :class:`~sklearn.compose.ColumnTransformer` that
       standard-scales the numeric features and one-hot encodes the
       categorical ones (``handle_unknown="ignore"`` so an unseen level at
       predict time does not crash the estimator).

    The returned pipeline is unfitted; feed it training rows only.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(PIPELINE_NUMERIC_COLUMNS)),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=float
                ),
                list(PIPELINE_CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            (
                "zero_policy",
                ZeroMedianImputer(
                    columns=tuple(policy.columns), sentinel=policy.sentinel
                ),
            ),
            ("features", column_transformer),
        ]
    )


def fit_preprocessing(frame: pd.DataFrame, *, policy=DEFAULT_ZERO_POLICY) -> Pipeline:
    """Fit a fresh pipeline on exactly the rows of ``frame``.

    Callers pass training rows; the function never looks anywhere else. That is
    the whole contract: the statistics are a function of ``frame`` only.
    """
    features = _feature_frame(frame)
    pipeline = build_preprocessing_pipeline(policy)
    pipeline.fit(features)
    logger.info(
        "Fitted preprocessing pipeline on %d row(s); zero medians=%s",
        len(features),
        {
            column: round(value, 3)
            for column, value in pipeline.named_steps["zero_policy"].medians_.items()
        },
    )
    return pipeline


def fit_on_train(
    frame: pd.DataFrame,
    train_indices: object,
    *,
    policy=DEFAULT_ZERO_POLICY,
) -> Pipeline:
    """Fit the pipeline on ``frame`` restricted to ``train_indices`` only.

    This is the only sanctioned way to prepare evaluation data: the split is
    resolved first, then the transform is fit on the training side of it.
    """
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise PipelineError(
            f"train_indices must be one-dimensional, got shape {indices.shape}."
        )
    if indices.size == 0:
        raise PipelineError("train_indices is empty; nothing to fit on.")
    return fit_preprocessing(frame.iloc[indices], policy=policy)


def transform_features(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted pipeline, returning a named feature DataFrame."""
    features = _feature_frame(frame)
    transformed = pipeline.transform(features)
    names = [str(name) for name in pipeline.get_feature_names_out()]
    return pd.DataFrame(transformed, columns=names, index=frame.index)


# ---------------------------------------------------------------------------
# Fitted-statistics extraction
# ---------------------------------------------------------------------------


def _jsonable(value: object) -> object:
    """Normalise numpy scalars to plain Python so statistics serialise."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def fitted_statistics(pipeline: Pipeline) -> dict[str, object]:
    """Return every data-dependent parameter the pipeline learned.

    These are exactly the numbers a leaky fit would contaminate: the imputation
    medians, the scaler's centre and spread, and the encoder's observed
    categories. Two pipelines with equal statistics were fit on identical
    information.
    """
    zero_step = pipeline.named_steps.get("zero_policy")
    if zero_step is None or not hasattr(zero_step, "medians_"):
        raise PipelineError(
            "Pipeline is not fitted: the zero_policy step has no medians_. "
            "Fit it on training rows before inspecting its statistics."
        )
    features = pipeline.named_steps.get("features")
    scaler = features.named_transformers_["numeric"]
    encoder = features.named_transformers_["categorical"]
    if not hasattr(scaler, "mean_") or not hasattr(encoder, "categories_"):
        raise PipelineError(
            "Pipeline is not fully fitted: scaler/encoder statistics are absent."
        )
    return {
        "zero_medians": {
            str(column): float(value)
            for column, value in zero_step.medians_.items()
        },
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "scaler_var": [float(value) for value in scaler.var_],
        "encoder_categories": [
            [_jsonable(value) for value in categories]
            for categories in encoder.categories_
        ],
    }


def _iter_leaves(value: object, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, object]] = []
        for key, item in value.items():
            leaves.extend(_iter_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_iter_leaves(item, f"{prefix}[{index}]"))
        return leaves
    return [(prefix, value)]


def max_statistic_difference(left: dict, right: dict) -> float:
    """Largest absolute difference between two fitted-statistics dicts.

    Numeric leaves are compared by absolute difference; categorical leaves
    contribute ``inf`` when they differ and ``0`` when they match. Used both to
    report how far apart two fits are and to prove a train-vs-full fit is
    actually distinguishable (a non-vacuous leakage detector).
    """
    left_leaves = dict(_iter_leaves(left))
    right_leaves = dict(_iter_leaves(right))
    keys = set(left_leaves) | set(right_leaves)
    worst = 0.0
    for key in keys:
        a = left_leaves.get(key)
        b = right_leaves.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            worst = max(worst, abs(float(a) - float(b)))
        elif a != b:
            return float("inf")
    return worst


def statistics_equal(left: dict, right: dict, *, atol: float = 1e-12) -> bool:
    """``True`` when two fitted-statistics dicts are numerically identical."""
    return max_statistic_difference(left, right) <= atol


def _first_difference(left: dict, right: dict) -> str | None:
    left_leaves = dict(_iter_leaves(left))
    right_leaves = dict(_iter_leaves(right))
    for key in sorted(set(left_leaves) | set(right_leaves)):
        a = left_leaves.get(key)
        b = right_leaves.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > 1e-12:
                return f"{key}: {a!r} != {b!r}"
        elif a != b:
            return f"{key}: {a!r} != {b!r}"
    return None


# ---------------------------------------------------------------------------
# Detector 1: fit-scope audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitScopeAudit:
    """Whether a fitted pipeline's statistics match a train-only fit."""

    passed: bool
    matches_train_only: bool
    matches_full_data: bool
    observed_vs_train_max_diff: float
    train_vs_full_max_diff: float
    train_rows: int
    full_rows: int
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "matches_train_only": self.matches_train_only,
            "matches_full_data": self.matches_full_data,
            "observed_vs_train_max_diff": self.observed_vs_train_max_diff,
            "train_vs_full_max_diff": self.train_vs_full_max_diff,
            "train_rows": self.train_rows,
            "full_rows": self.full_rows,
            "message": self.message,
        }


def audit_fit_scope(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    train_indices: object,
    *,
    policy=DEFAULT_ZERO_POLICY,
) -> FitScopeAudit:
    """Compare ``pipeline``'s fitted statistics against train-only and full fits."""
    indices = np.asarray(train_indices, dtype=np.int64)
    train_frame = frame.iloc[indices]
    train_stats = fitted_statistics(fit_preprocessing(train_frame, policy=policy))
    full_stats = fitted_statistics(fit_preprocessing(frame, policy=policy))
    observed = fitted_statistics(pipeline)

    matches_train = statistics_equal(observed, train_stats)
    matches_full = statistics_equal(observed, full_stats)
    train_vs_full = max_statistic_difference(train_stats, full_stats)
    observed_vs_train = max_statistic_difference(observed, train_stats)

    if matches_train:
        message = (
            f"Pipeline statistics match a train-only fit on {len(indices)} row(s); "
            "no held-out row influenced a transform."
        )
    elif matches_full and train_vs_full > 0:
        message = (
            "Pipeline statistics match a fit on the FULL dataset, not the "
            f"training rows ({len(indices)}). The transform was fit on data that "
            "includes the held-out rows — this is leakage."
        )
    else:
        message = (
            "Pipeline statistics match neither the train-only fit nor the "
            "full-data fit. The transform was fit on a different row set; "
            f"max deviation from training statistics = {observed_vs_train:g}."
        )

    audit = FitScopeAudit(
        passed=matches_train,
        matches_train_only=matches_train,
        matches_full_data=matches_full,
        observed_vs_train_max_diff=observed_vs_train,
        train_vs_full_max_diff=train_vs_full,
        train_rows=int(len(indices)),
        full_rows=int(len(frame)),
        message=message,
    )
    logger.info("Fit-scope audit: %s", message)
    return audit


def assert_fit_on_training_only(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    train_indices: object,
    *,
    policy=DEFAULT_ZERO_POLICY,
) -> FitScopeAudit:
    """Raise :class:`DataLeakageError` unless ``pipeline`` matches a train-only fit."""
    audit = audit_fit_scope(pipeline, frame, train_indices, policy=policy)
    if not audit.passed:
        raise DataLeakageError(audit.message)
    return audit


# ---------------------------------------------------------------------------
# Detector 2: influence probe
# ---------------------------------------------------------------------------


def perturb_test_rows(
    frame: pd.DataFrame,
    test_indices: object,
    *,
    numeric_value: int = PERTURB_NUMERIC_VALUE,
    categorical_value: str = PERTURB_CATEGORICAL_VALUE,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with every test row pushed to extreme values.

    Numeric features become :data:`PERTURB_NUMERIC_VALUE`; categorical features
    become :data:`PERTURB_CATEGORICAL_VALUE`. If a fitting path secretly reads
    these rows, its statistics will move; an honest path that is handed only
    training indices cannot see the change at all.
    """
    if not isinstance(frame, pd.DataFrame):
        raise FeatureFrameError(
            f"Expected a pandas.DataFrame, got {type(frame).__name__}."
        )
    indices = np.asarray(test_indices, dtype=np.int64)
    out = frame.copy()
    target_labels = out.index[indices]
    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            out.loc[target_labels, column] = numeric_value
        else:
            out.loc[target_labels, column] = categorical_value
    return out


@dataclass(frozen=True)
class InfluenceProbe:
    """Result of perturbing held-out rows and re-running a fitting path."""

    stable: bool
    test_rows_perturbed: int
    first_difference: str | None
    stats_before: dict[str, object]
    stats_after: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "stable": self.stable,
            "test_rows_perturbed": self.test_rows_perturbed,
            "first_difference": self.first_difference,
            "stats_before": self.stats_before,
            "stats_after": self.stats_after,
        }


#: A fitting path: given a frame and the training indices, return a fitted pipeline.
FitOnIndices = Callable[[pd.DataFrame, np.ndarray], Pipeline]


def fit_on_training_indices(frame: pd.DataFrame, indices: np.ndarray) -> Pipeline:
    """Honest fitting path: subset to ``indices``, then fit."""
    return fit_preprocessing(frame.iloc[np.asarray(indices, dtype=np.int64)])


def influence_probe(
    frame: pd.DataFrame,
    train_indices: object,
    test_indices: object,
    *,
    policy=DEFAULT_ZERO_POLICY,
    fit_on_indices: FitOnIndices | None = None,
) -> InfluenceProbe:
    """Perturb test rows and check whether the fitting path's statistics move."""
    if fit_on_indices is None:
        fit_on_indices = lambda f, idx: fit_preprocessing(  # noqa: E731
            f.iloc[np.asarray(idx, dtype=np.int64)], policy=policy
        )
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    before = fitted_statistics(fit_on_indices(frame, train))
    perturbed = perturb_test_rows(frame, test)
    after = fitted_statistics(fit_on_indices(perturbed, train))
    difference = _first_difference(before, after)
    probe = InfluenceProbe(
        stable=difference is None,
        test_rows_perturbed=int(test.size),
        first_difference=difference,
        stats_before=before,
        stats_after=after,
    )
    logger.info(
        "Influence probe on %d held-out row(s): %s",
        test.size,
        "stable" if probe.stable else f"LEAKAGE at {difference}",
    )
    return probe


def assert_no_test_influence(
    frame: pd.DataFrame,
    train_indices: object,
    test_indices: object,
    *,
    policy=DEFAULT_ZERO_POLICY,
    fit_on_indices: FitOnIndices | None = None,
) -> InfluenceProbe:
    """Raise :class:`DataLeakageError` if perturbing test rows moves the fit."""
    probe = influence_probe(
        frame,
        train_indices,
        test_indices,
        policy=policy,
        fit_on_indices=fit_on_indices,
    )
    if not probe.stable:
        raise DataLeakageError(
            "Held-out rows influenced a fitted transform: perturbing "
            f"{probe.test_rows_perturbed} test row(s) changed the learned "
            f"statistics ({probe.first_difference}). The fitting path must read "
            "training indices only."
        )
    return probe
