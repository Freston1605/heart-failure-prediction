"""Candidate variant builders for the pre-declared treatment gate (D034).

This module is the ONLY place where a declared
:class:`~heart.eval.treatment_declared.TreatmentCandidate` translates into a
buildable, predictable model object (decision D034). For every non-incumbent
declared candidate and every ``model_type`` in its ``applies_to``, a builder
returns a fresh unfitted scikit-learn-style model object exposing
``fit`` / ``predict`` / ``predict_proba``, deterministic under
:const:`heart.config.RANDOM_SEED`.

What is built per declared mechanism
------------------------------------

1. ``pipeline-variant`` — a :class:`sklearn.pipeline.Pipeline` whose
   ``preprocess`` step is the committed chain shape
   (:func:`heart.data.pipeline.build_preprocessing_pipeline`) carrying the
   candidate's declared mutation:

   - ``knn-imputer``: the median/zero-policy *imputation* is replaced by
     :class:`sklearn.impute.KNNImputer` (the plan's "KNeighborsImputer"; the
     committed sklearn class name is ``KNNImputer``) with
     :data:`KNN_IMPUTER_NEIGHBORS` neighbours, applied on the numeric feature
     block before scaling. Sentinel zeros (the declared zero-as-missing
     policy columns) are first converted to true missing values by
     :class:`SentinelToMissing` so the distance-based imputer sees exactly
     what the committed policy calls missing.
   - ``missingness-flag-columns``: :class:`MissingnessFlagAdder` prepends one
     binary indicator column per feature carrying any observed missing value
     (NaN, or the declared zero sentinel in a numeric column) BEFORE the
     committed imputation, so the model sees the missingness itself. The
     committed imputation/scaling/encoding is kept unchanged after it.
   - ``one-hot-drop-first``: the committed ``ColumnTransformer`` is rebuilt
     with ``OneHotEncoder(drop="first", handle_unknown="ignore")``.
   - ``yeo-johnson-power-transform`` / ``box-cox-power-transform``:
     :class:`_ColumnSubsetPowerTransformer` inserts
     :class:`sklearn.preprocessing.PowerTransformer` on the skewed numeric
     columns (yeo-johnson) / the strictly-positive numeric columns
     (box-cox) inside the pipeline, restricted per the declared mechanism and
     fit on training rows only. Standard scaling and one-hot encoding are
     kept, so the variant differs from the incumbent ONLY in the declared
     mechanism.

2. ``estimator-param`` — ``class-weight-balanced`` builds a :class:
   `sklearn.linear_model.LogisticRegression` per logistic spec with the
   spec's ``fixed_params`` and tuned defaults copied, and the candidate's
   declared change ``class_weight="balanced"`` applied LAST (so it can never
   be overridden from outside; the fixed set stays intact). The returned
   object is the committed preprocessing chain followed by this estimator,
   so it is directly comparable against the incumbent pipeline.

3. ``sampling`` — an :class:`imblearn.pipeline.Pipeline` (NOT
   ``sklearn.pipeline.Pipeline``, because it must ``fit_resample`` between
   steps) of [committed preprocessing chain, imblearn sampler step
   (:data:`SAMPLER_FACTORIES`, ``random_state=RANDOM_SEED``), the
   registry-built estimator]. :func:`assert_sampler_between_preprocess_and_
   classifier` records and enforces that the sampler sits strictly between
   preprocess and classifier (the M006 leakage golden rule; the id-level
   lineage recording itself is the gate runner's job via
   :mod:`heart.eval.golden_rule`).

Rung 0 (the documented incumbent median policy) is NOT built here —
:func:`incumbent_factory` returns the committed pipeline unchanged, which is
the gate's rung-0 baseline. Every failure mode (unknown ``variant_key``,
mechanism/sampler mapping gap, model outside ``applies_to``) raises
:class:`TreatmentGateError` loudly so the S02->S03 real-store run never
measures a silently mis-built candidate.
"""

from __future__ import annotations

import logging
from typing import Callable, Final, Mapping

import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from heart.config import RANDOM_SEED
from heart.data.pipeline import (
    PIPELINE_CATEGORICAL_COLUMNS,
    PIPELINE_NUMERIC_COLUMNS,
    build_preprocessing_pipeline,
)
from heart.data.quality import (
    DEFAULT_ZERO_POLICY,
    ZERO_SENTINEL,
    ZeroMedianImputer,
)
from heart.data.schema import FEATURE_COLUMNS
from heart.eval.treatment_declared import (
    BATTERY_MEMBERS,
    DECLARED_CANDIDATES,
    INCUMBENT_CANDIDATE,
    TreatmentCandidate,
)
from heart.models.registry import (
    BATTERY_MODEL_TYPES,
    build_estimator,
    build_pipeline,
    default_params,
    resolve_spec,
    validate_params,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TreatmentGateError",
    "KNN_IMPUTER_NEIGHBORS",
    "SKEWNESS_ABS_THRESHOLD",
    "MISSING_FLAG_PREFIX",
    "SENTINEL_TO_MISSING_COLUMNS",
    "SAMPLER_FACTORIES",
    "CLASS_WEIGHT_BALANCED_FACTORIES",
    "TreatmentGateError",
    "SentinelToMissing",
    "MissingnessFlagAdder",
    "candidate_by_key",
    "iter_buildable_candidates",
    "build_candidate",
    "build_variant",
    "incumbent_factory",
    "sampler_for_variant",
    "assert_sampler_between_preprocess_and_classifier",
    "require_feature_frame",
    "validate_variant_registry",
]

# ---------------------------------------------------------------------------
# Named error + declared build constants
# ---------------------------------------------------------------------------


class TreatmentGateError(Exception):
    """A treatment candidate cannot be built or measured honestly.

    Raised on: an unknown ``variant_key``, a mechanism/sampler mapping gap, a
    model type outside the candidate's declared ``applies_to``, a malformed
    feature frame, or a sampler that is not strictly between preprocess and
    classifier. T04 imports this class from here (D034: this module is the
    only owner of the declared-candidate -> model-object translation).
    """


#: Small odd neighbour count for the KNN imputation variant (declared here so
#: the variant is a fixed, reviewable number, never a hyperparameter).
KNN_IMPUTER_NEIGHBORS: Final[int] = 5

#: Absolute sample-skewness threshold selecting the numeric columns the
#: yeo-johnson variant transforms (computed on training rows only).
SKEWNESS_ABS_THRESHOLD: Final[float] = 0.75

#: Reserved column prefix for missingness indicator columns, so the variant
#: ColumnTransformer can route exactly the added flags (never a real feature).
MISSING_FLAG_PREFIX: Final[str] = "missing_flag_"

#: Sentinel->missing scope of :class:`SentinelToMissing`: the declared
#: zero-as-missing policy columns from :mod:`heart.data.quality`.
SENTINEL_TO_MISSING_COLUMNS: Final[tuple[str, ...]] = tuple(
    DEFAULT_ZERO_POLICY.columns
)

# ---------------------------------------------------------------------------
# Declared scope of the estimator-param candidate
# ---------------------------------------------------------------------------

#: The logistic model types the class-weight-balanced candidate applies to,
#: read from the declared candidate itself (never re-declared).
_CLASS_WEIGHT_BALANCED_KEYS: Final[frozenset[str]] = frozenset(
    next(
        candidate
        for candidate in DECLARED_CANDIDATES
        if candidate.variant_key == "class-weight-balanced"
    ).applies_to
)

# ---------------------------------------------------------------------------
# Sampler mapping (mechanism=sampling; the golden-rule layer's subject)
# ---------------------------------------------------------------------------

SAMPLER_FACTORIES: Final[dict[str, Callable[[], object]]] = {
    "random-undersampling": lambda: RandomUnderSampler(random_state=RANDOM_SEED),
    "random-oversampling": lambda: RandomOverSampler(random_state=RANDOM_SEED),
    "smote-oversampling": lambda: SMOTE(random_state=RANDOM_SEED),
    "adasyn-oversampling": lambda: ADASYN(random_state=RANDOM_SEED),
}


def sampler_for_variant(variant_key: str) -> object:
    """Return a fresh sampler instance for a declared sampling candidate.

    Raises :class:`TreatmentGateError` on a mechanism/sampler mapping gap
    (a sampling candidate whose key has no mapped sampler).
    """
    factory = SAMPLER_FACTORIES.get(variant_key)
    if factory is None:
        raise TreatmentGateError(
            f"Mechanism/sampler mapping gap: sampling candidate "
            f"{variant_key!r} has no mapped imblearn sampler; declared "
            f"sampling keys are {sorted(SAMPLER_FACTORIES)}."
        )
    return factory()


# ---------------------------------------------------------------------------
# Shared feature-frame validation
# ---------------------------------------------------------------------------


def require_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate that ``frame`` carries the declared feature columns.

    The builders are designed for the real dataset shape, so any raw (or
    fixture-equivalent) frame handed to a built model must expose every
    :const:`heart.data.schema.FEATURE_COLUMNS` column. Missing columns raise
    :class:`TreatmentGateError` instead of a deep sklearn error.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TreatmentGateError(
            f"Treatment candidates require a pandas.DataFrame, got "
            f"{type(frame).__name__}."
        )
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise TreatmentGateError(
            f"Treatment-candidate feature frame is missing declared column(s) "
            f"{missing}; present columns: {list(frame.columns)}."
        )
    return frame


# ---------------------------------------------------------------------------
# Custom transformers (the only new chain pieces; everything else is the
# committed chain, reused by calling the existing builders)
# ---------------------------------------------------------------------------


class SentinelToMissing(BaseEstimator, TransformerMixin):
    """Convert declared zero-sentinel values into true missing values (NaN).

    The committed policy (:class:`~heart.data.quality.ZeroMedianImputer`)
    treats impossible zeros as missing AND fills them with the fit-frame
    median. The ``knn-imputer`` variant replaces the *fill* with distance-based
    imputation, so it needs the flagging half only: zeros in the declared
    policy columns become NaN for :class:`KNNImputer` to impute. Fit is
    stateless (no statistic is learned), so nothing can leak.
    """

    def __init__(
        self,
        columns: tuple[str, ...] = SENTINEL_TO_MISSING_COLUMNS,
        sentinel: int = ZERO_SENTINEL,
    ) -> None:
        self.columns = columns
        self.sentinel = sentinel

    def fit(self, X: object, y: object = None) -> "SentinelToMissing":
        # NOTE: this transformer sits inside the ColumnTransformer's numeric
        # branch, so it receives the NUMERIC sub-frame only; requiring the
        # full FEATURE_COLUMNS set here would be wrong.
        if not isinstance(X, pd.DataFrame):
            raise TreatmentGateError(
                f"SentinelToMissing requires a pandas.DataFrame, got "
                f"{type(X).__name__}."
            )
        frame = X
        for column in self.columns:
            if column not in frame.columns:
                raise TreatmentGateError(
                    f"SentinelToMissing: declared zero-policy column {column!r} "
                    "is absent from the feature frame."
                )
        self.n_features_in_ = int(frame.shape[1])
        self.feature_names_in_ = pd.Index([str(c) for c in frame.columns])
        return self

    def transform(self, X: object) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):  # pragma: no cover - fit-guarded
            raise TreatmentGateError(
                f"SentinelToMissing requires a pandas.DataFrame, got "
                f"{type(X).__name__}."
            )
        frame = X
        out = frame.copy()
        for column in self.columns:
            if column not in out.columns:  # pragma: no cover - guarded in fit
                raise TreatmentGateError(
                    f"SentinelToMissing.transform: column {column!r} is absent."
                )
            out[column] = out[column].where(out[column] != self.sentinel)
        return out

    def get_feature_names_out(self, input_features: object = None) -> pd.Index:
        if input_features is not None:
            return pd.Index([str(c) for c in input_features])
        if hasattr(self, "feature_names_in_"):
            return pd.Index(self.feature_names_in_)
        return pd.Index([])


class MissingnessFlagAdder(BaseEstimator, TransformerMixin):
    """Prepend one binary indicator column per feature with observed missing.

    Missing is defined as a genuine NaN, or (numeric columns only) the
    declared zero sentinel — exactly the values the committed imputation
    replaces. The column set is learned from the FIT frame (training rows
    only) and frozen in :attr:`flag_columns_`; at transform time exactly those
    frozen columns are flagged, so the pipeline shape is stable across splits.
    The flags carry the :data:`MISSING_FLAG_PREFIX` so the variant
    ColumnTransformer routes them (they are never a real feature).
    """

    def __init__(self, sentinel: int = ZERO_SENTINEL) -> None:
        self.sentinel = sentinel

    def _missing_mask(self, series: pd.Series, column: str):
        mask = series.isna().to_numpy()
        if (
            pd.api.types.is_numeric_dtype(series)
            and column in SENTINEL_TO_MISSING_COLUMNS
        ):
            mask = mask | (series.to_numpy() == self.sentinel)
        return mask

    def fit(self, X: object, y: object = None) -> "MissingnessFlagAdder":
        frame = require_feature_frame(X)
        flagged = [
            column
            for column in frame.columns
            if self._missing_mask(frame[column], column).any()
        ]
        self.flag_columns_ = tuple(flagged)
        self.n_features_in_ = int(frame.shape[1])
        self.feature_names_in_ = pd.Index([str(c) for c in frame.columns])
        logger.info(
            "MissingnessFlagAdder fit on %d row(s): %d flagged feature(s) %s",
            len(frame),
            len(self.flag_columns_),
            list(self.flag_columns_),
        )
        return self

    def transform(self, X: object) -> pd.DataFrame:
        frame = require_feature_frame(X)
        out = frame.copy()
        for column in self.flag_columns_:
            if column not in out.columns:  # pragma: no cover - schema-locked
                raise TreatmentGateError(
                    f"MissingnessFlagAdder.transform: flagged column {column!r} "
                    "is absent from the frame."
                )
            out[f"{MISSING_FLAG_PREFIX}{column}"] = self._missing_mask(
                out[column], column
            ).astype("float64")
        return out

    def get_feature_names_out(self, input_features: object = None) -> pd.Index:
        if input_features is not None:
            base = [str(c) for c in input_features]
        elif hasattr(self, "feature_names_in_"):  # pragma: no cover
            base = [str(c) for c in self.feature_names_in_]
        else:  # pragma: no cover - unfitted misuse fails loudly
            raise TreatmentGateError(
                "MissingnessFlagAdder must be fit before feature names are "
                "requested."
            )
        return pd.Index(
            base + [f"{MISSING_FLAG_PREFIX}{c}" for c in self.flag_columns_]
        )


class _ColumnSubsetPowerTransformer(BaseEstimator, TransformerMixin):
    """Insert ``PowerTransformer`` on a fit-time-selected numeric column subset.

    The declared restriction (skewed columns for yeo-johnson, strictly-positive
    columns for box-cox) is only observable from data, so the selection happens
    in ``fit`` — on the training rows the pipeline is given, never on the full
    dataset. Selected columns are power-transformed and then (unlike
    ``PowerTransformer(standardize=True)``) scaled by the committed
    ``StandardScaler`` step that follows, so the branch keeps standard scaling;
    unselected numeric columns pass through untouched to the same scaler.

    An empty selection degrades to a pure passthrough: a fold whose training
    rows carry no skewed/strictly-positive column simply has no mechanism to
    apply, and the variant stays fit-safe.
    """

    def __init__(
        self,
        method: str = "yeo-johnson",
        skew_threshold: float = SKEWNESS_ABS_THRESHOLD,
    ) -> None:
        self.method = method
        self.skew_threshold = skew_threshold

    def fit(self, X: object, y: object = None) -> "_ColumnSubsetPowerTransformer":
        if self.method not in ("yeo-johnson", "box-cox"):
            raise TreatmentGateError(
                f"Power-transform variant method {self.method!r} is not one of "
                "('yeo-johnson', 'box-cox')."
            )
        if not isinstance(X, pd.DataFrame):
            raise TreatmentGateError(
                f"_ColumnSubsetPowerTransformer requires a pandas.DataFrame "
                f"(named numeric columns), got {type(X).__name__}."
            )
        columns = list(X.columns)
        if self.method == "box-cox":
            selected = [
                column for column in columns if float(X[column].min()) > 0.0
            ]
        else:
            skews = X.skew()
            selected = [
                column
                for column in columns
                if column in skews.index
                and pd.notna(skews[column])
                and abs(float(skews[column])) >= self.skew_threshold
            ]
        self.selected_columns_ = tuple(selected)
        self.power_transformer_ = None
        if selected:
            self.power_transformer_ = PowerTransformer(
                method=self.method, standardize=False
            ).fit(X[list(selected)])
        logger.info(
            "Power variant %s fit on %d row(s): transformed columns %s",
            self.method,
            len(X),
            list(selected),
        )
        return self

    def transform(self, X: object) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TreatmentGateError(
                f"_ColumnSubsetPowerTransformer requires a pandas.DataFrame, "
                f"got {type(X).__name__}."
            )
        if not hasattr(self, "selected_columns_"):
            raise TreatmentGateError(
                "_ColumnSubsetPowerTransformer.transform called before fit."
            )
        out = X.copy()
        if self.selected_columns_ and self.power_transformer_ is not None:
            out[list(self.selected_columns_)] = self.power_transformer_.transform(
                out[list(self.selected_columns_)]
            )
        return out

    def get_feature_names_out(self, input_features: object = None) -> pd.Index:
        if input_features is None:  # pragma: no cover - defensive
            raise TreatmentGateError(
                "_ColumnSubsetPowerTransformer needs input feature names."
            )
        return pd.Index([str(c) for c in input_features])


# ---------------------------------------------------------------------------
# Preprocessing variants (committed chain + the one declared mutation)
# ---------------------------------------------------------------------------


def _committed_onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)


def _committed_column_transformer() -> ColumnTransformer:
    """The committed CT shape: scaled numerics, one-hot categoricals."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(PIPELINE_NUMERIC_COLUMNS)),
            (
                "categorical",
                _committed_onehot(),
                list(PIPELINE_CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
    )


def _preprocess_knn_imputer() -> Pipeline:
    """knn-imputer variant: sentinel->missing then KNNImputer before scaling."""
    numeric_branch = Pipeline(
        steps=[
            (
                "sentinel_to_missing",
                SentinelToMissing(
                    columns=SENTINEL_TO_MISSING_COLUMNS, sentinel=ZERO_SENTINEL
                ),
            ),
            ("knn_imputer", KNNImputer(n_neighbors=KNN_IMPUTER_NEIGHBORS)),
            ("scale", StandardScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, list(PIPELINE_NUMERIC_COLUMNS)),
            ("categorical", _committed_onehot(), list(PIPELINE_CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("features", column_transformer)])


def _preprocess_power_transform(method: str) -> Pipeline:
    """yeo-johnson / box-cox variant: power transform on the declared subset."""
    numeric_branch = Pipeline(
        steps=[
            ("power", _ColumnSubsetPowerTransformer(method=method)),
            ("scale", StandardScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, list(PIPELINE_NUMERIC_COLUMNS)),
            ("categorical", _committed_onehot(), list(PIPELINE_CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            # the committed imputation stays: only the power transform is added
            ("zero_policy", committed_zero_policy_imputer()),
            ("features", column_transformer),
        ]
    )


def _preprocess_one_hot_drop_first() -> Pipeline:
    """one-hot-drop-first variant: dummy-trap encoding, everything else kept."""
    return Pipeline(
        steps=[
            ("zero_policy", committed_zero_policy_imputer()),
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("numeric", StandardScaler(), list(PIPELINE_NUMERIC_COLUMNS)),
                        (
                            "categorical",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=float,
                            ),
                            list(PIPELINE_CATEGORICAL_COLUMNS),
                        ),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )


def _preprocess_missing_flags() -> Pipeline:
    """missingness-flag variant: flags prepended, committed imputation kept."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(PIPELINE_NUMERIC_COLUMNS)),
            ("categorical", _committed_onehot(), list(PIPELINE_CATEGORICAL_COLUMNS)),
            (
                "missingness_flags",
                StandardScaler(),
                lambda frame: [
                    column
                    for column in frame.columns
                    if str(column).startswith(MISSING_FLAG_PREFIX)
                ],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("missingness_flags", MissingnessFlagAdder(sentinel=ZERO_SENTINEL)),
            ("zero_policy", committed_zero_policy_imputer()),
            ("features", column_transformer),
        ]
    )


def committed_zero_policy_imputer() -> ZeroMedianImputer:
    """The committed zero-policy imputer for the default declared policy."""
    return ZeroMedianImputer(
        columns=tuple(DEFAULT_ZERO_POLICY.columns),
        sentinel=DEFAULT_ZERO_POLICY.sentinel,
    )


#: variant_key -> preprocess-factory, one entry per declared pipeline variant.
_PREPROCESS_VARIANT_FACTORIES: Final[dict[str, Callable[[], Pipeline]]] = {
    "knn-imputer": _preprocess_knn_imputer,
    "missingness-flag-columns": _preprocess_missing_flags,
    "one-hot-drop-first": _preprocess_one_hot_drop_first,
    "yeo-johnson-power-transform": lambda: _preprocess_power_transform("yeo-johnson"),
    "box-cox-power-transform": lambda: _preprocess_power_transform("box-cox"),
}


def _make_balanced_logistic_factory(model_type: str) -> Callable[[object], LogisticRegression]:
    """Explicit per-LR-spec factory for the class-weight-balanced candidate.

    Documented fixed_params flow: the spec's ``fixed_params`` (solver,
    l1_ratio pinning, max_iter, random_state) are copied first and can never
    be overridden; the tuned hyperparameters fall back to their declared
    defaults (``params=None``) or are supplied validated; the candidate's
    declared change ``class_weight="balanced"`` is applied LAST, so it pins
    the tuned class-weight default (``None``) by declaration, not by tuning.
    """
    spec = resolve_spec(model_type)

    def factory(params: object = None) -> LogisticRegression:
        if params is None:
            tuned = default_params(spec.space)
        else:
            tuned = validate_params(spec, params)
        merged = dict(spec.fixed_params)
        merged.update(tuned)
        merged["class_weight"] = "balanced"
        return LogisticRegression(**merged)

    return factory


#: Explicit factory per logistic spec, keyed by model_type.
CLASS_WEIGHT_BALANCED_FACTORIES: Final[
    dict[str, Callable[[object], LogisticRegression]]
] = {
    model_type: _make_balanced_logistic_factory(model_type)
    for model_type in sorted(_CLASS_WEIGHT_BALANCED_KEYS)
}

# ---------------------------------------------------------------------------
# Mechanism dispatch
# ---------------------------------------------------------------------------


def _build_pipeline_variant(
    candidate: TreatmentCandidate, model_type: str, params: object
) -> Pipeline:
    """Build the full model object for a pipeline-variant candidate."""
    factory = _PREPROCESS_VARIANT_FACTORIES.get(candidate.variant_key)
    if factory is None:
        raise TreatmentGateError(
            f"Mechanism/preprocessing mapping gap: pipeline-variant candidate "
            f"{candidate.variant_key!r} has no committed-chain mutation; "
            f"declared pipeline-variant keys are "
            f"{sorted(_PREPROCESS_VARIANT_FACTORIES)}."
        )
    estimator = build_estimator(resolve_spec(model_type), params)
    return Pipeline(steps=[("preprocess", factory()), ("clf", estimator)])


def _build_estimator_param(
    candidate: TreatmentCandidate, model_type: str, params: object
) -> Pipeline:
    """Build the full model object for an estimator-param candidate."""
    factory = CLASS_WEIGHT_BALANCED_FACTORIES.get(model_type)
    if factory is None:
        raise TreatmentGateError(
            f"Mechanism/factory mapping gap: estimator-param candidate "
            f"{candidate.variant_key!r} has no declared factory for model_type "
            f"{model_type!r}; declared estimator-param scopes are "
            f"{sorted(CLASS_WEIGHT_BALANCED_FACTORIES)}."
        )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", factory(params)),
        ]
    )


def _build_sampling(
    candidate: TreatmentCandidate, model_type: str, params: object
) -> ImblearnPipeline:
    """Build the imblearn pipeline for a sampling candidate.

    Steps: [committed preprocessing chain, sampler, registry estimator]. The
    committed chain is REUSED by flattening
    :func:`heart.data.pipeline.build_preprocessing_pipeline`'s own steps (an
    imblearn pipeline forbids nested Pipelines as intermediate steps, so the
    chain is inlined step-for-step, not re-declared). The sampler sits
    strictly between preprocess and classifier, which
    :func:`assert_sampler_between_preprocess_and_classifier` verifies on the
    built object before it is returned.
    """
    sampler = sampler_for_variant(candidate.variant_key)
    committed_steps = list(build_preprocessing_pipeline().steps)
    pipeline = ImblearnPipeline(
        steps=[
            *committed_steps,
            ("sampler", sampler),
            ("clf", build_estimator(resolve_spec(model_type), params)),
        ]
    )
    assert_sampler_between_preprocess_and_classifier(pipeline, candidate.variant_key)
    return pipeline


_MECHANISM_BUILDERS: Final[dict[str, Callable[..., object]]] = {
    "pipeline-variant": _build_pipeline_variant,
    "estimator-param": _build_estimator_param,
    "sampling": _build_sampling,
}


def candidate_by_key(variant_key: str) -> TreatmentCandidate:
    """Return the declared candidate for ``variant_key``; raise on unknown."""
    for candidate in DECLARED_CANDIDATES:
        if candidate.variant_key == variant_key:
            return candidate
    raise TreatmentGateError(
        f"Unknown treatment variant key {variant_key!r}; declared candidate "
        f"keys are {list(c.variant_key for c in DECLARED_CANDIDATES)}."
    )


def iter_buildable_candidates() -> tuple[TreatmentCandidate, ...]:
    """Declared candidates the gate measures: every non-incumbent candidate.

    The incumbent (rung 0, documented median policy) is deliberately skipped —
    it is described, not built; :func:`incumbent_factory` provides the rung-0
    baseline object.
    """
    return tuple(candidate for candidate in DECLARED_CANDIDATES if not candidate.is_incumbent)


def build_candidate(
    candidate: TreatmentCandidate, model_type: str, params: object = None
) -> object:
    """Build a fresh unfitted model object for (candidate, model_type).

    Raises :class:`TreatmentGateError` when ``model_type`` is outside the
    candidate's declared ``applies_to``, when the declared mechanism has no
    mapped builder (mapping gap), or when the candidate is the documented
    incumbent (rung 0 is described, not built).
    """
    if not isinstance(candidate, TreatmentCandidate):
        raise TreatmentGateError(
            f"build_candidate expects a TreatmentCandidate, got "
            f"{type(candidate).__name__}."
        )
    if candidate.variant_key == INCUMBENT_CANDIDATE.variant_key:
        raise TreatmentGateError(
            f"Candidate {candidate.variant_key!r} is the documented incumbent "
            "(rung 0); it is described, not built. Use incumbent_factory for "
            "the committed rung-0 baseline object."
        )
    if model_type not in candidate.applies_to:
        raise TreatmentGateError(
            f"Candidate {candidate.variant_key!r} does not apply to "
            f"{model_type!r}: declared applies_to is {list(candidate.applies_to)}."
        )
    builder = _MECHANISM_BUILDERS.get(candidate.mechanism)
    if builder is None:
        raise TreatmentGateError(
            f"Mechanism mapping gap: candidate {candidate.variant_key!r} "
            f"declares unknown mechanism {candidate.mechanism!r}."
        )
    return builder(candidate, model_type, params)


def build_variant(
    variant_key: str, model_type: str, params: object = None
) -> object:
    """Build the model object for a declared variant key + battery model type.

    Deterministic under :const:`heart.config.RANDOM_SEED`: every randomised
    component (samplers, estimators with ``random_state`` fixed params) is
    seeded from the project seed, and all transforms/imputers are
    deterministic given the fit data.
    """
    candidate = candidate_by_key(variant_key)
    return build_candidate(candidate, model_type, params)


def incumbent_factory(
    model_type: str, params: Mapping[str, object] | None = None
) -> Pipeline | ClassifierMixin:
    """The committed rung-0 baseline: the registry pipeline, unchanged.

    This is the documented incumbent median policy — the same
    :func:`heart.models.registry.build_pipeline` object the battery uses, with
    its default-tuned parameters (or an explicitly supplied committed param
    set, e.g. a variance floor a fixture cannot do without, or the battery
    ledger's committed best params for the real-store gate run). Nothing is
    mutated here.
    """
    return build_pipeline(resolve_spec(model_type), params)


def assert_sampler_between_preprocess_and_classifier(
    pipeline: object, variant_key: str
) -> dict[str, object]:
    """Record/enforce the golden-rule position of the sampling step.

    The imblearn pipeline must contain exactly one ``fit_resample``-capable
    step, and that step must sit strictly after the committed preprocessing
    chain and strictly before the classifier (the M006 leakage golden rule:
    resampling sees only the training fold's rows the preprocessing transform
    was already fit on; id-level lineage recording happens in the gate runner
    via :mod:`heart.eval.golden_rule`).
    """
    steps = getattr(pipeline, "steps", None)
    if not steps:
        raise TreatmentGateError(
            f"Variant {variant_key!r}: the sampling pipeline has no steps."
        )
    names = [name for name, _ in steps]
    sampler_positions = [
        index
        for index, (_, step) in enumerate(steps)
        if hasattr(step, "fit_resample")
    ]
    if len(sampler_positions) != 1:
        raise TreatmentGateError(
            f"Variant {variant_key!r}: golden-rule position violated — the "
            f"pipeline {names} must contain exactly one fit_resample-capable "
            f"sampler step, found {len(sampler_positions)} at {sampler_positions}."
        )
    sampler_index = sampler_positions[0]
    if sampler_index == 0:
        raise TreatmentGateError(
            f"Variant {variant_key!r}: golden-rule position violated — the "
            f"sampler ({names[sampler_index]!r}) runs BEFORE the preprocessing "
            "chain; resampling must sit strictly between preprocess and "
            "classifier (inside the training fold only)."
        )
    if sampler_index >= len(steps) - 1:
        raise TreatmentGateError(
            f"Variant {variant_key!r}: golden-rule position violated — the "
            f"sampler ({names[sampler_index]!r}) is not followed by a "
            "classifier; it must sit strictly between preprocess and "
            "classifier."
        )
    for index, (name, step) in enumerate(steps):
        if index < sampler_index:
            if hasattr(step, "fit_resample") or not hasattr(step, "transform"):
                raise TreatmentGateError(
                    f"Variant {variant_key!r}: golden-rule position violated — "
                    f"step {name!r} before the sampler is neither a plain "
                    "preprocessing transformer nor correctly ordered."
                )
        elif index > sampler_index and hasattr(step, "fit_resample"):
            raise TreatmentGateError(
                f"Variant {variant_key!r}: golden-rule position violated — "
                f"step {name!r} after the sampler is a second resampling step."
            )
    preprocess_name, preprocess = steps[0]
    sampler_name, sampler = steps[sampler_index]
    clf_name, clf = steps[-1]
    if not hasattr(preprocess, "transform") or not hasattr(clf, "predict"):
        raise TreatmentGateError(
            f"Variant {variant_key!r}: the pipeline ends must be a "
            "preprocessing transformer and a classifier."
        )
    summary = {
        "preprocess_step": preprocess_name,
        "sampler_step": sampler_name,
        "sampler_index": sampler_index,
        "clf_step": clf_name,
        "sampler_class": type(sampler).__name__,
    }
    logger.info(
        "Golden-rule position recorded for %s: %s", variant_key, summary
    )
    return summary


# ---------------------------------------------------------------------------
# Registry cross-check
# ---------------------------------------------------------------------------


def validate_variant_registry() -> None:
    """Cross-check the declared candidate set against the real registry.

    Raises :class:`TreatmentGateError` when the declared battery copy drifts
    from :const:`heart.models.registry.BATTERY_MODEL_TYPES`, or when any
    non-incumbent declared candidate lacks its mapped builder (mechanism/
    sampler mapping gap).
    """
    if tuple(BATTERY_MEMBERS) != tuple(BATTERY_MODEL_TYPES):
        raise TreatmentGateError(
            f"Declared battery {BATTERY_MEMBERS} drifted from the registry "
            f"battery {BATTERY_MODEL_TYPES}; fix "
            "heart.eval.treatment_declared.BATTERY_MEMBERS."
        )
    for candidate in iter_buildable_candidates():
        if candidate.mechanism == "pipeline-variant":
            factory = _PREPROCESS_VARIANT_FACTORIES.get(candidate.variant_key)
            if factory is None:
                raise TreatmentGateError(
                    f"Mechanism/preprocessing mapping gap: pipeline-variant "
                    f"candidate {candidate.variant_key!r} has no builder."
                )
        elif candidate.mechanism == "sampling":
            if candidate.variant_key not in SAMPLER_FACTORIES:
                raise TreatmentGateError(
                    f"Mechanism/sampler mapping gap: sampling candidate "
                    f"{candidate.variant_key!r} has no mapped sampler."
                )
        elif candidate.mechanism == "estimator-param":
            for model_type in candidate.applies_to:
                if model_type not in CLASS_WEIGHT_BALANCED_FACTORIES:
                    raise TreatmentGateError(
                        f"Mechanism/factory mapping gap: estimator-param "
                        f"candidate {candidate.variant_key!r} has no factory "
                        f"for model_type {model_type!r}."
                    )


# Import-time guard (mirrors the declared-module guard): a declared candidate
# without a mapped builder, or a battery drift, fails the import loudly.
validate_variant_registry()
