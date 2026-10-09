"""Clinically grounded engineered features, each individually toggleable.

Why this module exists
----------------------
The raw fedesoriano schema carries eleven measurements. Several clinically
meaningful quantities are *not* directly present: how a patient's achieved
maximum heart rate compares with the age-predicted maximum, how lipid burden
relates to age and resting pressure, how the ischemic ST response interacts
with the exercise test, and how many independent cardiometabolic risk factors
a patient carries. Those are precisely the relationships a clinician reasons
about, so they are worth exposing to the models as explicit candidates.

The design constraint that makes this useful for a *portfolio* is
**isolatability**: every candidate is a :class:`FeatureTransform` registered
under a stable name, and :func:`engineer_features` accepts an explicit
``include`` / ``exclude`` selection. A feature can therefore be ablated on its
own — added to or removed from a run without touching any other transform. The
S04/T02 ablation harness consumes this surface; nothing here decides which
features are kept.

Leakage safety
--------------
Every transform here is **row-local and stateless**: its output for a row is a
deterministic function of that row's columns alone. No transform learns a
median, a mean, a category set, or any other fitted statistic, so a row can
never influence another row's engineered values. This is a deliberate property,
not an accident — it means feature engineering adds **zero** leakage surface,
unlike imputation/scaling which must still be fit on training rows only.

Imputation precondition
-----------------------
Derived ratios use ``RestingBP`` and ``Cholesterol`` as denominators, and both
columns use ``0`` as an impossible-value sentinel. A zero denominator is
undefined, so engineering **fails closed** rather than silently emitting an
infinity or a fabricated zero. The intended pipeline order is therefore:

    zero-as-missing median imputation  ->  engineered features  ->  scaling/encoding

:class:`~heart.data.quality.ZeroMedianImputer` (or the S01 preprocessing chain)
must run first. Passing a frame that still contains a zero sentinel raises
:class:`NonFiniteEngineeredFeatureError` naming the offending column.

Observability
-------------
:func:`engineer_features` logs the applied transform names and the produced
columns at ``INFO``; :func:`describe_transforms` renders the full registry
(name, kind, inputs, outputs, rationale) as a markdown table;
:meth:`FeatureTransform.to_dict` gives the machine-readable view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class FeatureEngineeringError(Exception):
    """Base class for every feature-engineering failure."""


class NotAFeatureFrameError(FeatureEngineeringError):
    """The object handed to the engine is not a pandas DataFrame."""


class MissingFeatureColumnError(FeatureEngineeringError):
    """A frame lacks a column a selected transform declares as an input."""


class UnknownFeatureTransformError(FeatureEngineeringError):
    """A requested transform name is not declared in the registry."""


class EmptyFeatureSelectionError(FeatureEngineeringError):
    """The resolved transform selection is empty; nothing would be produced."""


class EngineeredColumnCollisionError(FeatureEngineeringError):
    """An engineered column name collides with a column already in the frame."""


class NonFiniteEngineeredFeatureError(FeatureEngineeringError):
    """An engineered numeric column contains a non-finite value."""


# ---------------------------------------------------------------------------
# Declared kinds
# ---------------------------------------------------------------------------

#: Marker for a numerically-usable engineered column.
NUMERIC_KIND: str = "numeric"

#: Marker for a categorical engineered column (one-hot encoded downstream).
CATEGORICAL_KIND: str = "categorical"

_VALID_KINDS: frozenset[str] = frozenset({NUMERIC_KIND, CATEGORICAL_KIND})


# ---------------------------------------------------------------------------
# The transform contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureTransform:
    """One named, individually toggleable engineered feature.

    Parameters
    ----------
    name:
        Stable registry key used by ``include`` / ``exclude`` selections and
        recorded in ablation artifacts.
    kind:
        :data:`NUMERIC_KIND` or :data:`CATEGORICAL_KIND`; tells the downstream
        preprocessing chain how to treat the produced column.
    inputs:
        Source columns the transform reads. All must be present in the frame.
    produces:
        Output column names. Must be non-empty, unique, and disjoint from
        every other transform's outputs (enforced at import by
        :func:`_validate_registry`).
    function:
        ``frame -> DataFrame`` returning exactly the ``produces`` columns,
        indexed like the input. Must be row-local and stateless.
    description:
        One-sentence human summary.
    rationale:
        The clinical justification for the feature.
    """

    name: str
    kind: str
    inputs: tuple[str, ...]
    produces: tuple[str, ...]
    function: Callable[[pd.DataFrame], pd.DataFrame]
    description: str
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise FeatureEngineeringError(
                f"transform name must be a non-empty string, got {self.name!r}."
            )
        if self.kind not in _VALID_KINDS:
            raise FeatureEngineeringError(
                f"transform {self.name!r} has unknown kind {self.kind!r}; "
                f"expected one of {sorted(_VALID_KINDS)}."
            )
        if not self.inputs:
            raise FeatureEngineeringError(
                f"transform {self.name!r} declares no inputs."
            )
        if not self.produces:
            raise FeatureEngineeringError(
                f"transform {self.name!r} declares no produced columns."
            )
        if not callable(self.function):
            raise FeatureEngineeringError(
                f"transform {self.name!r} has a non-callable function."
            )

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return only this transform's produced columns for ``frame``."""
        result = self.function(frame)
        if not isinstance(result, pd.DataFrame):
            raise FeatureEngineeringError(
                f"transform {self.name!r} returned {type(result).__name__}, "
                "expected a pandas.DataFrame."
            )
        produced = tuple(str(column) for column in result.columns)
        if set(produced) != set(self.produces):
            raise FeatureEngineeringError(
                f"transform {self.name!r} declared outputs {list(self.produces)} "
                f"but produced {list(produced)}."
            )
        return result[list(self.produces)].copy()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "produces": list(self.produces),
            "description": self.description,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Small numeric helpers (row-local; no fitted statistics anywhere)
# ---------------------------------------------------------------------------


def _safe_divide(numerator: object, denominator: object) -> pd.Series:
    """Divide elementwise, yielding NaN where the denominator is zero.

    A zero denominator means the sentinel value slipped past imputation; the
    NaN is surfaced by :func:`_check_finite` rather than hidden as an
    infinity or a fabricated zero.
    """
    num = pd.to_numeric(pd.Series(numerator), errors="coerce").astype(float)
    den = pd.to_numeric(pd.Series(denominator), errors="coerce").astype(float)
    den = den.where(den != 0.0, np.nan)
    return num / den


def _age_band_values(frame: pd.DataFrame) -> pd.Series:
    """Decade bands used by both the band and the band-by-sex interaction."""
    age = pd.to_numeric(frame["Age"], errors="coerce").to_numpy(dtype=float)
    labels = np.select(
        [age < 40.0, age < 50.0, age < 60.0, age < 70.0],
        ["lt40", "40-49", "50-59", "60-69"],
        default="ge70",
    )
    return pd.Series(labels, index=frame.index, dtype="object")


# ---------------------------------------------------------------------------
# Transform implementations
# ---------------------------------------------------------------------------


def _transform_age_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket age into clinically conventional decade bands."""
    return pd.DataFrame({"AgeBand": _age_band_values(frame)}, index=frame.index)


def _transform_max_hr_percent_predicted(frame: pd.DataFrame) -> pd.DataFrame:
    """Achieved MaxHR as a percent of the age-predicted maximum (220 - Age)."""
    predicted = 220.0 - pd.to_numeric(frame["Age"], errors="coerce").astype(float)
    achieved = pd.to_numeric(frame["MaxHR"], errors="coerce").astype(float)
    return pd.DataFrame(
        {"MaxHR_Pct_Predicted": _safe_divide(achieved, predicted) * 100.0},
        index=frame.index,
    )


def _transform_hr_reserve(frame: pd.DataFrame) -> pd.DataFrame:
    """Absolute chronotropic reserve: (220 - Age) - MaxHR."""
    predicted = 220.0 - pd.to_numeric(frame["Age"], errors="coerce").astype(float)
    achieved = pd.to_numeric(frame["MaxHR"], errors="coerce").astype(float)
    return pd.DataFrame(
        {"HR_Reserve": predicted - achieved}, index=frame.index
    )


def _transform_cholesterol_age_ratio(frame: pd.DataFrame) -> pd.DataFrame:
    """Serum cholesterol per year of age."""
    return pd.DataFrame(
        {
            "Cholesterol_Age_Ratio": _safe_divide(
                frame["Cholesterol"], frame["Age"]
            )
        },
        index=frame.index,
    )


def _transform_cholesterol_restingbp_ratio(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Serum cholesterol relative to resting blood pressure."""
    return pd.DataFrame(
        {
            "Cholesterol_RestingBP_Ratio": _safe_divide(
                frame["Cholesterol"], frame["RestingBP"]
            )
        },
        index=frame.index,
    )


def _transform_rate_pressure_product(frame: pd.DataFrame) -> pd.DataFrame:
    """Rate-pressure product (RestingBP * MaxHR / 1000), a cardiac workload index."""
    bp = pd.to_numeric(frame["RestingBP"], errors="coerce").astype(float)
    hr = pd.to_numeric(frame["MaxHR"], errors="coerce").astype(float)
    return pd.DataFrame(
        {"Rate_Pressure_Product": bp * hr / 1000.0}, index=frame.index
    )


def _transform_oldpeak_slope_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Oldpeak weighted by ST-slope severity (Up=0, Flat=1, Down=2)."""
    weights = {"Up": 0.0, "Flat": 1.0, "Down": 2.0}
    slope_weight = (
        frame["ST_Slope"].map(weights).astype(float)
    )
    oldpeak = pd.to_numeric(frame["Oldpeak"], errors="coerce").astype(float)
    return pd.DataFrame(
        {"Oldpeak_Slope_Score": oldpeak * slope_weight}, index=frame.index
    )


#: Clinically conventional cardiometabolic thresholds counted by the risk sum.
_METABOLIC_RISK_CHOLESTEROL: float = 240.0
_METABOLIC_RISK_RESTINGBP: float = 140.0


def _transform_metabolic_risk_count(frame: pd.DataFrame) -> pd.DataFrame:
    """Count of four independent cardiometabolic risk factors present."""
    fasting = (pd.to_numeric(frame["FastingBS"], errors="coerce") == 1).astype(int)
    angina = (frame["ExerciseAngina"].astype(str) == "Y").astype(int)
    cholesterol = (
        pd.to_numeric(frame["Cholesterol"], errors="coerce")
        >= _METABOLIC_RISK_CHOLESTEROL
    ).astype(int)
    resting_bp = (
        pd.to_numeric(frame["RestingBP"], errors="coerce")
        >= _METABOLIC_RISK_RESTINGBP
    ).astype(int)
    total = fasting + angina + cholesterol + resting_bp
    return pd.DataFrame(
        {"Metabolic_Risk_Count": total.astype(int)}, index=frame.index
    )


def _transform_exercise_ecg_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Categorical exercise-angina by ST-slope interaction group."""
    group = (
        frame["ExerciseAngina"].astype(str)
        + "_"
        + frame["ST_Slope"].astype(str)
    )
    return pd.DataFrame({"Exercise_ECG_Group": group}, index=frame.index)


def _transform_age_band_sex(frame: pd.DataFrame) -> pd.DataFrame:
    """Categorical age-band by sex interaction group."""
    group = _age_band_values(frame) + "_" + frame["Sex"].astype(str)
    return pd.DataFrame({"AgeBand_Sex": group}, index=frame.index)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Every declared engineered feature, keyed by stable transform name.
ENGINEERED_TRANSFORMS: dict[str, FeatureTransform] = {
    "age_band": FeatureTransform(
        name="age_band",
        kind=CATEGORICAL_KIND,
        inputs=("Age",),
        produces=("AgeBand",),
        function=_transform_age_band,
        description="Patient age bucketed into clinical decade bands.",
        rationale=(
            "Cardiovascular risk rises non-linearly with age; decade bands let "
            "the model use that shape without over-fitting a single linear age "
            "term, and they anchor the age-by-sex grouping."
        ),
    ),
    "max_hr_percent_predicted": FeatureTransform(
        name="max_hr_percent_predicted",
        kind=NUMERIC_KIND,
        inputs=("Age", "MaxHR"),
        produces=("MaxHR_Pct_Predicted",),
        function=_transform_max_hr_percent_predicted,
        description="Achieved MaxHR as a percent of the age-predicted maximum.",
        rationale=(
            "Chronotropic competence is best judged relative to age rather than "
            "in absolute bpm; the percent-of-predicted value (using the "
            "conventional 220 - Age estimate) is the clinically interpretable "
            "form."
        ),
    ),
    "hr_reserve": FeatureTransform(
        name="hr_reserve",
        kind=NUMERIC_KIND,
        inputs=("Age", "MaxHR"),
        produces=("HR_Reserve",),
        function=_transform_hr_reserve,
        description="Chronotropic reserve: age-predicted max minus achieved MaxHR.",
        rationale=(
            "The absolute shortfall from age-predicted maximum is a standard "
            "exercise-test readout; unlike the percentage it does not explode "
            "when the predicted maximum approaches zero."
        ),
    ),
    "cholesterol_age_ratio": FeatureTransform(
        name="cholesterol_age_ratio",
        kind=NUMERIC_KIND,
        inputs=("Cholesterol", "Age"),
        produces=("Cholesterol_Age_Ratio",),
        function=_transform_cholesterol_age_ratio,
        description="Serum cholesterol normalized by age.",
        rationale=(
            "Lipid burden accumulates with age; the per-year ratio separates a "
            "genuinely elevated cholesterol from the age-expected value."
        ),
    ),
    "cholesterol_restingbp_ratio": FeatureTransform(
        name="cholesterol_restingbp_ratio",
        kind=NUMERIC_KIND,
        inputs=("Cholesterol", "RestingBP"),
        produces=("Cholesterol_RestingBP_Ratio",),
        function=_transform_cholesterol_restingbp_ratio,
        description="Serum cholesterol relative to resting blood pressure.",
        rationale=(
            "Lipid and pressure burdens jointly drive risk; their ratio exposes "
            "discordance (high cholesterol with normal pressure, or the "
            "reverse) that neither column shows alone."
        ),
    ),
    "rate_pressure_product": FeatureTransform(
        name="rate_pressure_product",
        kind=NUMERIC_KIND,
        inputs=("RestingBP", "MaxHR"),
        produces=("Rate_Pressure_Product",),
        function=_transform_rate_pressure_product,
        description="Resting rate-pressure product (BP * HR / 1000).",
        rationale=(
            "The rate-pressure product is a classical index of myocardial "
            "oxygen demand; it captures the hemodynamic workload implied by "
            "pressure and heart rate jointly."
        ),
    ),
    "oldpeak_slope_interaction": FeatureTransform(
        name="oldpeak_slope_interaction",
        kind=NUMERIC_KIND,
        inputs=("Oldpeak", "ST_Slope"),
        produces=("Oldpeak_Slope_Score",),
        function=_transform_oldpeak_slope_score,
        description="Oldpeak weighted by ST-slope severity.",
        rationale=(
            "ST depression and slope are read together in exercise testing: the "
            "same Oldpeak is more ominous with a down-sloping segment, so the "
            "severity-weighted product encodes the interaction the raw columns "
            "leave implicit."
        ),
    ),
    "metabolic_risk_count": FeatureTransform(
        name="metabolic_risk_count",
        kind=NUMERIC_KIND,
        inputs=("FastingBS", "ExerciseAngina", "Cholesterol", "RestingBP"),
        produces=("Metabolic_Risk_Count",),
        function=_transform_metabolic_risk_count,
        description="Count of four conventional cardiometabolic risk factors.",
        rationale=(
            "Risk factors accumulate rather than substitute; a single count of "
            "elevated fasting glucose, exercise angina, high cholesterol, and "
            "high resting pressure summarises comorbidity load parsimoniously."
        ),
    ),
    "exercise_ecg_group": FeatureTransform(
        name="exercise_ecg_group",
        kind=CATEGORICAL_KIND,
        inputs=("ExerciseAngina", "ST_Slope"),
        produces=("Exercise_ECG_Group",),
        function=_transform_exercise_ecg_group,
        description="Exercise-angina by ST-slope categorical group.",
        rationale=(
            "The combination of exercise-induced angina with the ST-slope "
            "response is the core of the exercise test; the joint category lets "
            "the model key on that pattern directly."
        ),
    ),
    "age_band_sex": FeatureTransform(
        name="age_band_sex",
        kind=CATEGORICAL_KIND,
        inputs=("Age", "Sex"),
        produces=("AgeBand_Sex",),
        function=_transform_age_band_sex,
        description="Age-band by sex categorical group.",
        rationale=(
            "Cardiovascular risk is shaped by age and sex together (for example "
            "earlier male risk); the joint category exposes that interaction "
            "instead of forcing the model to discover it from two main effects."
        ),
    ),
}

#: Transform names in registry (declaration) order.
TRANSFORM_NAMES: tuple[str, ...] = tuple(ENGINEERED_TRANSFORMS)

#: Every engineered output column, in registry order.
ENGINEERED_COLUMN_NAMES: tuple[str, ...] = tuple(
    column
    for transform in ENGINEERED_TRANSFORMS.values()
    for column in transform.produces
)


def _validate_registry() -> None:
    """Fail at import if the registry is internally inconsistent."""
    seen: dict[str, str] = {}
    for name, transform in ENGINEERED_TRANSFORMS.items():
        if transform.name != name:
            raise FeatureEngineeringError(
                f"registry key {name!r} does not match transform.name "
                f"{transform.name!r}."
            )
        for column in transform.produces:
            if column in seen:
                raise FeatureEngineeringError(
                    f"engineered column {column!r} is produced by both "
                    f"{seen[column]!r} and {name!r}; outputs must be unique."
                )
            seen[column] = name


_validate_registry()


# ---------------------------------------------------------------------------
# Selection resolution
# ---------------------------------------------------------------------------


def _as_name_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise FeatureEngineeringError(
            f"{label} must be a sequence of transform names, not a bare "
            f"string ({value!r}); pass a list or tuple."
        )
    if not isinstance(value, Iterable):
        raise FeatureEngineeringError(
            f"{label} must be a sequence of transform names, got "
            f"{type(value).__name__}."
        )
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise FeatureEngineeringError(
                f"{label} contains a non-string entry {item!r}."
            )
        names.append(item)
    return tuple(names)


def resolve_selection(
    include: object = None, exclude: object = ()
) -> tuple[FeatureTransform, ...]:
    """Resolve ``include`` / ``exclude`` into an ordered tuple of transforms.

    ``include=None`` selects the whole registry. Unknown names (in either
    argument) raise :class:`UnknownFeatureTransformError`; an empty resolved
    selection raises :class:`EmptyFeatureSelectionError`.
    """
    include_names = (
        TRANSFORM_NAMES if include is None else _as_name_tuple(include, label="include")
    )
    exclude_names = _as_name_tuple(exclude, label="exclude")

    for name in include_names:
        if name not in ENGINEERED_TRANSFORMS:
            raise UnknownFeatureTransformError(
                f"Unknown engineered transform {name!r} in include; declared "
                f"transforms are {list(TRANSFORM_NAMES)}."
            )
    for name in exclude_names:
        if name not in ENGINEERED_TRANSFORMS:
            raise UnknownFeatureTransformError(
                f"Unknown engineered transform {name!r} in exclude; declared "
                f"transforms are {list(TRANSFORM_NAMES)}."
            )

    excluded = set(exclude_names)
    ordered = tuple(dict.fromkeys(include_names))
    selected = tuple(name for name in ordered if name not in excluded)
    if not selected:
        raise EmptyFeatureSelectionError(
            "The resolved engineered-feature selection is empty; include at "
            "least one transform (or call the baseline feature set directly)."
        )
    return tuple(ENGINEERED_TRANSFORMS[name] for name in selected)


def selected_column_names(
    include: object = None, exclude: object = ()
) -> tuple[str, ...]:
    """Output columns produced by the resolved selection, in order."""
    return tuple(
        column
        for transform in resolve_selection(include, exclude)
        for column in transform.produces
    )


def selected_columns_by_kind(
    kind: str, include: object = None, exclude: object = ()
) -> tuple[str, ...]:
    """Output columns of one ``kind`` produced by the resolved selection."""
    if kind not in _VALID_KINDS:
        raise FeatureEngineeringError(
            f"unknown kind {kind!r}; expected one of {sorted(_VALID_KINDS)}."
        )
    return tuple(
        column
        for transform in resolve_selection(include, exclude)
        if transform.kind == kind
        for column in transform.produces
    )


def available_transforms() -> tuple[str, ...]:
    """Every declared transform name, in registry order."""
    return TRANSFORM_NAMES


# ---------------------------------------------------------------------------
# Engineering
# ---------------------------------------------------------------------------


def _require_frame(frame: object) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise NotAFeatureFrameError(
            f"Expected a pandas.DataFrame, got {type(frame).__name__}."
        )
    return frame


def _check_finite(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        non_finite = ~np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        count = int(np.count_nonzero(non_finite))
        if count:
            raise NonFiniteEngineeredFeatureError(
                f"Engineered column {column!r} has {count} non-finite "
                "value(s). The usual cause is a zero sentinel in the "
                "denominator (RestingBP or Cholesterol); apply the "
                "zero-as-missing median imputation before engineering."
            )


def engineer_features(
    frame: pd.DataFrame,
    *,
    include: object = None,
    exclude: object = (),
    keep_original: bool = True,
    validate: bool = True,
    check_finite: bool = True,
) -> pd.DataFrame:
    """Return ``frame`` plus the engineered columns of the resolved selection.

    Parameters
    ----------
    frame:
        The source frame. Must contain every input column of every selected
        transform. The input frame is never mutated.
    include:
        Transform names to apply, or ``None`` for the whole registry.
    exclude:
        Transform names to drop from the selection.
    keep_original:
        When ``True`` (default) the returned frame carries the original
        columns followed by the engineered columns; when ``False`` it carries
        the engineered columns only.
    validate:
        When ``True`` (default) input columns are checked and a name collision
        with an existing column raises :class:`EngineeredColumnCollisionError`.
    check_finite:
        When ``True`` (default) engineered numeric columns are required to be
        finite (no NaN/inf), so a missed imputation fails loudly.

    Returns
    -------
    pandas.DataFrame
        A new frame; the original index is preserved.
    """
    source = _require_frame(frame)
    transforms = resolve_selection(include, exclude)

    if validate:
        for transform in transforms:
            missing = [column for column in transform.inputs if column not in source]
            if missing:
                raise MissingFeatureColumnError(
                    f"Transform {transform.name!r} needs input column(s) "
                    f"{missing}, which are absent. Present columns are "
                    f"{list(source.columns)}."
                )
        collisions = [
            column
            for column in selected_column_names(include, exclude)
            if column in source.columns
        ]
        if collisions:
            raise EngineeredColumnCollisionError(
                f"Engineered column(s) {collisions} already exist in the frame; "
                "drop or rename them before engineering to avoid a silent "
                "overwrite."
            )

    produced: dict[str, pd.Series] = {}
    for transform in transforms:
        block = transform.apply(source)
        for column in transform.produces:
            produced[column] = block[column]

    engineered = pd.DataFrame(produced, index=source.index)
    if check_finite:
        _check_finite(
            engineered,
            selected_columns_by_kind(NUMERIC_KIND, include, exclude),
        )

    result = (
        pd.concat([source, engineered], axis=1)
        if keep_original
        else engineered
    )
    logger.info(
        "Engineered %d feature column(s) from %d transform(s): %s",
        len(produced),
        len(transforms),
        list(produced),
    )
    return result


# ---------------------------------------------------------------------------
# Scikit-learn integration
# ---------------------------------------------------------------------------


class EngineeredFeatureTransformer(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer applying a fixed engineered selection.

    Because the underlying transforms are row-local, ``fit`` only validates
    the declared input columns and returns ``self``; there are no fitted
    statistics and therefore no leakage surface. This wrapper lets an
    ablation pipeline place engineering between imputation and scaling.

    Parameters
    ----------
    include:
        Transform names to apply, or ``None`` for the whole registry.
    exclude:
        Transform names to drop from the selection.
    keep_original:
        Whether to pass the input columns through alongside the engineered
        ones (default ``True``).
    """

    def __init__(
        self,
        include: object = None,
        exclude: object = (),
        keep_original: bool = True,
    ) -> None:
        self.include = include
        self.exclude = exclude
        self.keep_original = keep_original

    def fit(self, X, y=None):  # noqa: N803 - sklearn contract
        frame = _require_frame(X)
        transforms = resolve_selection(self.include, self.exclude)
        for transform in transforms:
            missing = [column for column in transform.inputs if column not in frame]
            if missing:
                raise MissingFeatureColumnError(
                    f"Transform {transform.name!r} needs input column(s) "
                    f"{missing}, which are absent during fit."
                )
        self.selected_transforms_ = tuple(
            transform.name for transform in transforms
        )
        self.feature_names_in_ = np.asarray(
            [str(column) for column in frame.columns], dtype=object
        )
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):  # noqa: N803 - sklearn contract
        frame = _require_frame(X)
        return engineer_features(
            frame,
            include=self.selected_transforms_,
            exclude=(),
            keep_original=self.keep_original,
        )

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = list(getattr(self, "feature_names_in_", []))
        base = [str(name) for name in input_features] if self.keep_original else []
        engineered = list(selected_column_names(self.selected_transforms_))
        return np.asarray(base + engineered, dtype=object)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def describe_transforms() -> str:
    """Render the full transform registry as a markdown table."""
    lines = [
        f"engineered transforms: {len(TRANSFORM_NAMES)} "
        f"({len(ENGINEERED_COLUMN_NAMES)} output column(s))",
        "",
        "| # | transform | kind | inputs | produces |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, transform in enumerate(ENGINEERED_TRANSFORMS.values(), start=1):
        lines.append(
            f"| {index} | `{transform.name}` | {transform.kind} | "
            f"{', '.join(transform.inputs)} | {', '.join(transform.produces)} |"
        )
    lines.append("")
    lines.append("Rationale:")
    for transform in ENGINEERED_TRANSFORMS.values():
        lines.append("")
        lines.append(f"### `{transform.name}`")
        lines.append(f"_{transform.description}_")
        lines.append("")
        lines.append(transform.rationale)
    return "\n".join(lines)


__all__ = [
    "NUMERIC_KIND",
    "CATEGORICAL_KIND",
    "FeatureEngineeringError",
    "NotAFeatureFrameError",
    "MissingFeatureColumnError",
    "UnknownFeatureTransformError",
    "EmptyFeatureSelectionError",
    "EngineeredColumnCollisionError",
    "NonFiniteEngineeredFeatureError",
    "FeatureTransform",
    "ENGINEERED_TRANSFORMS",
    "TRANSFORM_NAMES",
    "ENGINEERED_COLUMN_NAMES",
    "EngineeredFeatureTransformer",
    "available_transforms",
    "resolve_selection",
    "selected_column_names",
    "selected_columns_by_kind",
    "engineer_features",
    "describe_transforms",
]
