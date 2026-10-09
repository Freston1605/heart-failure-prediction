"""Declared schema for the fedesoriano heart-failure dataset.

This module is the single source of truth for the *contract* of the raw data:
the expected row count, the 11 feature columns plus the ``HeartDisease`` target,
each column's storage type, and the full domain of every categorical column.

``validate_schema`` enforces that contract and raises **named** exceptions with
actionable messages. The point is to fail loudly when an upstream data change
introduces schema drift: a silently accepted extra category or a column that
quietly became nullable would invalidate every downstream benchmark.

The dataset is the fedesoriano "Heart Failure Prediction" combined dataset
(918 rows) assembled from five clinical sites (Cleveland, Hungary, Switzerland,
Long Beach VA, and the Statlog heart set).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Expected shape
# ---------------------------------------------------------------------------

#: Number of patient rows in the canonical dataset.
EXPECTED_ROW_COUNT: int = 918

#: Outcome column predicted by every model in the portfolio.
TARGET_COLUMN: str = "HeartDisease"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class SchemaError(Exception):
    """Base class for every dataset schema/integrity violation."""


class MissingColumnsError(SchemaError):
    """One or more declared columns are absent from the frame."""


class UnexpectedColumnsError(SchemaError):
    """The frame carries columns that are not declared in the schema."""


class RowCountError(SchemaError):
    """The frame does not contain the expected number of rows."""


class DtypeMismatchError(SchemaError):
    """A column's pandas dtype kind does not match the declared storage."""


class NullValueError(SchemaError):
    """A column contains null values although the schema forbids them."""


class UnexpectedCategoryError(SchemaError):
    """A categorical column contains a level outside its declared domain."""


# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Declared contract for one column.

    ``kind`` is one of ``"integer"``, ``"float"`` or ``"category"``; the
    ``storage`` field is only meaningful for category columns and records
    whether the level domain is encoded as strings (``"string"``) or as the
    integers ``0``/``1`` (``"integer"``).
    """

    name: str
    kind: str
    storage: str | None = None
    levels: tuple[str, ...] | None = None
    role: str = "feature"
    note: str = ""

    @property
    def is_categorical(self) -> bool:
        return self.kind == "category"


#: Canonical ordering of the 12 columns as they appear in ``heart.csv``.
SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec("Age", "integer", note="Years; continuous integer in the source file."),
    ColumnSpec("Sex", "category", storage="string", levels=("F", "M")),
    ColumnSpec(
        "ChestPainType",
        "category",
        storage="string",
        levels=("ASY", "ATA", "NAP", "TA"),
        note="TA=typical angina, ATA=atypical, NAP=non-anginal, ASY=asymptomatic.",
    ),
    ColumnSpec("RestingBP", "integer", note="mm Hg; 0 is impossible and tracked in S01/T03."),
    ColumnSpec("Cholesterol", "integer", note="mm Hg; 0 is impossible and tracked in S01/T03."),
    ColumnSpec(
        "FastingBS",
        "category",
        storage="integer",
        levels=("0", "1"),
        note="1 if fasting blood sugar > 120 mg/dl.",
    ),
    ColumnSpec("RestingECG", "category", storage="string", levels=("Normal", "ST", "LVH")),
    ColumnSpec("MaxHR", "integer", note="Maximum heart rate achieved (bpm)."),
    ColumnSpec(
        "ExerciseAngina",
        "category",
        storage="string",
        levels=("N", "Y"),
        note="Exercise-induced angina.",
    ),
    ColumnSpec("Oldpeak", "float", note="ST depression induced by exercise, relative to rest."),
    ColumnSpec("ST_Slope", "category", storage="string", levels=("Up", "Flat", "Down")),
    ColumnSpec(
        TARGET_COLUMN,
        "category",
        storage="integer",
        levels=("0", "1"),
        role="target",
        note="1 = heart disease present.",
    ),
)

# ---------------------------------------------------------------------------
# Derived lookups
# ---------------------------------------------------------------------------

#: Every declared column name, in canonical order.
ALL_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in SCHEMA)

#: The 11 feature columns (everything except the target).
FEATURE_COLUMNS: tuple[str, ...] = tuple(s.name for s in SCHEMA if s.role == "feature")

#: Names of categorical columns.
CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(s.name for s in SCHEMA if s.is_categorical)

#: Names of numeric columns.
NUMERIC_COLUMNS: tuple[str, ...] = tuple(s.name for s in SCHEMA if not s.is_categorical)

#: Declared categorical domains keyed by column name.
CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    s.name: s.levels for s in SCHEMA if s.is_categorical and s.levels is not None
}

#: Column specs keyed by name.
SPECS_BY_NAME: dict[str, ColumnSpec] = {spec.name: spec for spec in SCHEMA}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _observed_strings(series: pd.Series) -> set[str]:
    """Return the distinct non-null values of ``series`` rendered as strings."""
    return {str(value) for value in series.dropna().unique()}


def _check_dtype(series: pd.Series, spec: ColumnSpec) -> None:
    if spec.kind == "integer":
        ok = pd.api.types.is_integer_dtype(series)
        expected = "integer dtype"
    elif spec.kind == "float":
        ok = pd.api.types.is_float_dtype(series)
        expected = "floating-point dtype"
    elif spec.is_categorical and spec.storage == "integer":
        ok = pd.api.types.is_integer_dtype(series)
        expected = "integer dtype (levels 0/1)"
    elif spec.is_categorical:
        ok = pd.api.types.is_string_dtype(series)
        expected = "string dtype"
    else:  # pragma: no cover - guards against an invalid SCHEMA edit
        ok = True
        expected = "any dtype"

    if not ok:
        raise DtypeMismatchError(
            f"Column {spec.name!r} should have {expected} but pandas read "
            f"{series.dtype!r}. Fix the source file or update ColumnSpec("
            f"name={spec.name!r}) in heart.data.schema."
        )


def _check_nulls(series: pd.Series, spec: ColumnSpec) -> None:
    null_count = int(series.isna().sum())
    if null_count:
        raise NullValueError(
            f"Column {spec.name!r} contains {null_count} null value(s), but the "
            "schema forbids nulls. The raw dataset is expected to be complete; "
            "investigate the upstream source rather than imputing at load time."
        )


def _check_levels(series: pd.Series, spec: ColumnSpec) -> None:
    if not spec.is_categorical or spec.levels is None:
        return
    allowed = set(spec.levels)
    observed = _observed_strings(series)
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise UnexpectedCategoryError(
            f"Column {spec.name!r} contains unexpected categor{'y' if len(unexpected) == 1 else 'ies'} "
            f"{unexpected}; allowed levels are {sorted(allowed)}. Adjust the data "
            "or, if the domain genuinely changed, update CATEGORICAL_LEVELS in "
            "heart.data.schema and document the change."
        )


def validate_schema(frame: pd.DataFrame, *, expected_rows: int | None = EXPECTED_ROW_COUNT) -> None:
    """Validate ``frame`` against the declared dataset schema.

    Parameters
    ----------
    frame:
        The loaded dataset. Must be a :class:`pandas.DataFrame`.
    expected_rows:
        Required number of rows. Pass ``None`` to skip the row-count check
        (useful for small fixtures in tests).

    Raises
    ------
    SchemaError
        One of the named subclasses, describing exactly what drifted.
    """
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError(f"Expected a pandas.DataFrame, got {type(frame).__name__}.")

    present = set(frame.columns)
    missing = [name for name in ALL_COLUMNS if name not in present]
    if missing:
        raise MissingColumnsError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected exactly {list(ALL_COLUMNS)}."
        )

    declared = set(ALL_COLUMNS)
    unexpected_cols = [name for name in frame.columns if name not in declared]
    if unexpected_cols:
        raise UnexpectedColumnsError(
            f"Dataset has unexpected column(s): {unexpected_cols}. "
            f"Expected exactly {list(ALL_COLUMNS)}. Update heart.data.schema "
            "if the source legitimately added columns."
        )

    if expected_rows is not None and len(frame) != expected_rows:
        raise RowCountError(
            f"Dataset has {len(frame)} rows but {expected_rows} were expected "
            "(the fedesoriano combined dataset has 918 rows). Either the source "
            "changed or the wrong file was loaded."
        )

    for spec in SCHEMA:
        series = frame[spec.name]
        _check_dtype(series, spec)
        _check_nulls(series, spec)
        _check_levels(series, spec)


def schema_report() -> str:
    """Render the declared schema as a human-readable table."""
    lines = [
        f"expected rows: {EXPECTED_ROW_COUNT}",
        f"target column: {TARGET_COLUMN}",
        f"feature columns ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}",
        "columns:",
    ]
    width = max(len(spec.name) for spec in SCHEMA)
    for spec in SCHEMA:
        levels = "" if spec.levels is None else f" levels={list(spec.levels)}"
        lines.append(f"  {spec.name:<{width}}  {spec.kind}{levels}")
    return "\n".join(lines)
