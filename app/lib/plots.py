"""Exploration-data builders for the Explore dashboard page.

This module owns *what the dashboard shows* so the page itself stays thin
and testable. Every insight group is produced here as a plain
:class:`pandas.DataFrame` (or dict of scalars) that both the Streamlit page
and the tests consume, so "what renders" and "what is asserted" cannot
drift apart.

The class-balance and impossible-zero figures are NOT recomputed with
ad-hoc one-liners here: they come from the same functions that generated
the audited S01 report (``reports/data_quality.md``) —
:func:`heart.data.quality.compute_class_balance` and
:func:`heart.data.quality.invalid_zero_counts` — so the dashboard's
displayed numbers match the audited figures by construction, and the test
suite pins that equivalence against the report file itself.

Failure contract at this boundary mirrors the one in
:mod:`app.lib.artifact_loader`: :func:`load_explore_data` never raises.
A missing, corrupt (hash mismatch), schema-mismatched or malformed source
dataset yields a :class:`ExploreData` with ``ok=False`` and a named error
kind so the page can render a friendly message instead of a stack-trace
white screen.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import pandas as pd

from heart.data.load import (
    DownloadError,
    IntegrityChecksumError,
    SourceNotFoundError,
    load_dataset,
)
from heart.data.quality import (
    ZERO_AS_MISSING_COLUMNS,
    ClassBalance,
    invalid_zero_counts,
    compute_class_balance,
    count_affected_rows,
)
from heart.data.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

__all__ = [
    "ExploreData",
    "load_explore_data",
    "clear_explore_cache",
    "class_balance",
    "zero_sentinel_table",
    "numeric_distribution",
    "categorical_distribution",
    "correlation_matrix",
    "feature_insight_table",
    "insight_groups",
]

#: Sharable prefix of the dataset digest shown in the integrity banner.
_DIGEST_DISPLAY_CHARS: int = 12

#: Round factor for correlation displays.
_CORR_DECIMALS: int = 3


class ExploreData:
    """Result of one pinned-dataset load attempt; never raises.

    Attributes:
        ok: True when the validated dataset was loaded.
        frame: The validated :class:`pandas.DataFrame` (only when ``ok``).
        error_kind: Named error bucket (always present when not ``ok``).
        error_message: Friendly, plain-language failure message.
        rows: Row count of the loaded (or empty) frame.
        sha256: Integrity digest from the loader (empty on failure).
    """

    def __init__(
        self,
        frame: pd.DataFrame | None,
        error_kind: str | None,
        error_message: str | None,
        sha256: str = "",
    ) -> None:
        self.frame = frame
        self.error_kind = error_kind
        self.error_message = error_message
        self.sha256 = sha256
        self.ok = frame is not None and error_kind is None
        self.rows = int(len(frame)) if frame is not None else 0

    @property
    def digest_display(self) -> str:
        if not self.sha256:
            return "?"
        return self.sha256[:_DIGEST_DISPLAY_CHARS]


def _friendly_message(exc: Exception) -> tuple[str, str]:
    """Map a dataset-load exception to (kind, plain-language message)."""
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, FileNotFoundError) or isinstance(exc, SourceNotFoundError):
        kind = "data-missing"
        message = (
            "The dataset file could not be found. This project ships the "
            "pinned dataset under data/raw/heart.csv, so a missing "
            "file means the checkout is incomplete. Restore the commit or "
            "re-download the pinned source."
        )
    elif isinstance(exc, PermissionError):
        kind = "data-unreadable"
        message = (
            "The dataset file exists but could not be read with this "
            "process's permissions. Check the file ownership under data/raw/."
        )
    elif isinstance(exc, pd.errors.ParserError):
        kind = "data-corrupt"
        message = "The dataset file could not be parsed. It looks damaged."
    elif isinstance(exc, IntegrityChecksumError):
        kind = "data-checksum"
        message = (
            "The dataset file does not match its pinned SHA-256 digest. It "
            "is damaged or replaced; this app refuses to display data "
            "whose provenance cannot be verified."
        )
    elif isinstance(exc, DownloadError):
        kind = "data-network"
        message = (
            "The pinned dataset was offline and the pinned download source "
            "could not be reached. Restore data/raw/heart.csv or retry once "
            "network access is available."
        )
    elif name in {"SchemaError", "RowCountError", "DtypeMismatchError", "NullValueError",
                  "MissingColumnsError", "UnexpectedColumnsError", "UnexpectedCategoryError"}:
        kind = "data-schema"
        message = (
            "The dataset is present but does not match the schema this app "
            "was built against. A column, row count or category domain "
            "changed; re-run the S01 audit before serving figures."
        )
    else:
        kind = "data-unknown"
        message = (
            "An unexpected problem occurred while loading the dataset. "
            "The dashboard refuses to continue rather than showing "
            "possibly-wrong figures."
        )
    if text:
        message = f"{message} ({name}: {text.splitlines()[0][:200]})"
    return kind, message


def _dataset_path() -> Path | None:
    """Dataset path override (``EXPLORE_DATA_PATH``), for error-path demos.

    Unset means the committed ``data/raw/heart.csv``. The env var exists so
    AppTest-based render tests can point the page at a missing / corrupt
    file without touching the install.
    """
    override = os.environ.get("EXPLORE_DATA_PATH")
    return Path(override) if override else None


def _load_attempt() -> ExploreData:
    """One uncached dataset load attempt; maps every exception to the contract."""
    try:
        # Offline-safe: the app serves the committed copy under data/raw/;
        # a missing file is a user-visible error, not a network fetch.
        dataset = load_dataset(path=_dataset_path(), allow_download=False)
        return ExploreData(dataset.frame, None, None, sha256=dataset.sha256)
    except Exception as exc:  # noqa: BLE001 - the dashboard boundary swallows all
        kind, message = _friendly_message(exc)
        return ExploreData(None, kind, message)


@st.cache_data(show_spinner="Loading the pinned dataset...")
def _cached_load() -> ExploreData:
    """Streamlit-cached dataset load: one validation per session."""
    return _load_attempt()


def load_explore_data() -> ExploreData:
    """Load the pinned dataset for the dashboard, or the friendly error."""
    return _cached_load()


def clear_explore_cache() -> None:
    """Drop the cached dataset load (e.g. after data re-download)."""
    _cached_load.clear()


def _require_frame(df: object) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    return df


# ---------------------------------------------------------------------------
# Insight group 1: distributions
# ---------------------------------------------------------------------------


def numeric_distribution(df: pd.DataFrame, column: str, *, bins: int = 20) -> pd.DataFrame:
    """Value counts over ``bins`` of one numeric feature, ascending by edge.

    Output is a single-column frame whose index is the bin label string
    (``lo–hi``); it is the exact frame the page renders, so tests can pin
    its shape and contents.
    """
    _require_frame(df)
    series = df[column]
    cut = pd.cut(series, bins=bins)
    counts = cut.value_counts().sort_index()
    labels = [f"{interval.left:g}–{interval.right:g}" for interval in counts.index]
    out = pd.DataFrame({column: counts.to_numpy()}, index=labels)
    out.index.name = "bin"
    return out


def categorical_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Level counts of one categorical feature, descending.

    Index is the level and its name is ``<column> (level)`` — a name equal
    to the column itself makes ``st.bar_chart`` fail with "cannot insert
    <column>, already exists", so the renderer needs the distinction.
    """
    _require_frame(df)
    counts = df[column].astype(str).value_counts()
    out = pd.DataFrame({column: counts.to_numpy()}, index=counts.index.astype(str))
    out.index.name = f"{column} (level)"
    return out


# ---------------------------------------------------------------------------
# Insight group 2: correlation view
# ---------------------------------------------------------------------------


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of every numeric feature (including the target).

    Returns the frame the page renders; rounding is applied here so the
    displayed matrix and any test expectations share one rounding rule.
    """
    _require_frame(df)
    columns = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if TARGET_COLUMN in df.columns and TARGET_COLUMN not in columns:
        columns = columns + [TARGET_COLUMN]
    corr = df[columns].corr(numeric_only=True).round(_CORR_DECIMALS)
    return corr


def strongest_correlations(
    df: pd.DataFrame, *, top: int = 10
) -> pd.DataFrame:
    """The most informative feature/target correlations, sorted by |r|.

    Diagonal (1.0) and the trivial target-self pair are excluded so the
    table shows feature-driven insight only.
    """
    _require_frame(df)
    corr = correlation_matrix(df)
    target = TARGET_COLUMN if TARGET_COLUMN in corr.columns else corr.columns[-1]
    series = corr[target].drop(index=target)
    ordered = series.abs().sort_values(ascending=False).head(top)
    out = pd.DataFrame(
        {
            "feature": ordered.index,
            f"corr with {target}": [round(float(series.loc[i]), _CORR_DECIMALS) for i in ordered.index],
        }
    ).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Insight group 3: class balance (S01-figured)
# ---------------------------------------------------------------------------


def class_balance(df: pd.DataFrame) -> ClassBalance:
    """The audited class distribution, reusing the S01 audit code path.

    Consumed both by the page's rendered metrics and by the test asserting
    the dashboard matches ``reports/data_quality.md``.
    """
    _require_frame(df)
    return compute_class_balance(df)


def zero_sentinel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Impossible-zero table rendered by the page, via the S01 audit.

    Columns: ``Column``, ``Zero count``, ``Share of rows``, ``Of which
    positive class`` — the same four columns (and the same numbers) as the
    table in ``reports/data_quality.md``.
    """
    _require_frame(df)
    counts = invalid_zero_counts(df)
    rows = [
        {
            "Column": z.column,
            "Zero count": z.count,
            "Share of rows": f"{z.share:.2%}",
            "Of which positive class": int(z.by_target.get(1, 0)),
        }
        for z in counts.values()
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=["Column", "Zero count", "Share of rows", "Of which positive class"]
        )
    return out


def affected_zero_rows(df: pd.DataFrame) -> int:
    """Rows carrying at least one impossible zero, via the S01 audit."""
    _require_frame(df)
    return count_affected_rows(df, columns=ZERO_AS_MISSING_COLUMNS)


# ---------------------------------------------------------------------------
# Insight group 4: per-feature insight
# ---------------------------------------------------------------------------


def feature_insight_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: type, missing, unique levels, and disease rate.

    For numeric features ``Event rate by value`` is the mean target among
    rows at/above the feature's median vs. below it — an interpretable
    spread summary rather than a bare correlation. For categorical
    features ``Event rate by value`` is stored as ``NaN`` and the actual
    per-level rates are surfaced by :func:`categorical_target_rates`.
    """
    _require_frame(df)
    rows: list[dict[str, object]] = []
    prevalence = class_balance(df).prevalence
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            rows.append(
                {"Feature": feature, "Type": "missing"}
            )
            continue
        series = df[feature]
        row: dict[str, object] = {
            "Feature": feature,
            "Type": "categorical" if feature in CATEGORICAL_COLUMNS else "numeric",
            "Non-null": int(series.notna().sum()),
            "Unique levels": int(series.nunique()),
        }
        if feature in NUMERIC_COLUMNS or feature not in CATEGORICAL_COLUMNS:
            row["Min"] = round(float(series.min()), 3)
            row["Max"] = round(float(series.max()), 3)
            row["Mean"] = round(float(series.mean()), 3)
            if TARGET_COLUMN in df.columns:
                median = float(series.median())
                upper = df.loc[series >= median, TARGET_COLUMN].mean()
                lower = df.loc[series < median, TARGET_COLUMN].mean()
                row["Positive rate (≥ median)"] = round(float(upper), 3)
                row["Positive rate (< median)"] = round(float(lower), 3)
        else:
            top = series.value_counts().idxmax()
            row["Most common"] = str(top)
            row["Most common share"] = round(float((series == top).mean()), 3)
            if TARGET_COLUMN in df.columns:
                rates = df.groupby(feature)[TARGET_COLUMN].mean()
                row["Positive rate (≥ median)"] = round(float(rates.max()), 3)
                row["Positive rate (< median)"] = round(float(rates.min()), 3)
        if TARGET_COLUMN in df.columns:
            row["Overall positive rate"] = round(float(prevalence), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_target_rates(
    df: pd.DataFrame, feature: str
) -> pd.DataFrame | None:
    """Per-level disease rate for one categorical feature, or None.

    Returns the exact frame (level → positive rate) the page renders as a
    mini-bar so the clinical contrast (e.g. ASY vs TA chest pain) is
    visible without leaving the dashboard.
    """
    _require_frame(df)
    if feature not in CATEGORICAL_COLUMNS or feature not in df.columns:
        return None
    if TARGET_COLUMN not in df.columns:
        return None
    rates = df.groupby(feature)[TARGET_COLUMN].mean().round(_CORR_DECIMALS)
    out = pd.DataFrame({f"{feature} → disease rate": rates.to_numpy()}, index=rates.index)
    out.index.name = feature
    return out


def insight_groups(df: pd.DataFrame) -> dict[str, object]:
    """All four insight groups in one dictionary, in render order.

    Group names are stable keys the page and tests share:
    ``distributions``, ``per_feature``, ``correlation``, ``class_balance``.
    """
    _require_frame(df)
    numeric_dists = [numeric_distribution(df, col) for col in NUMERIC_COLUMNS if col in df.columns]
    categorical_dists = [
        categorical_distribution(df, col) for col in CATEGORICAL_COLUMNS if col in df.columns
    ]
    balance = class_balance(df)
    return {
        "distributions": {
            "numeric": numeric_dists,
            "categorical": categorical_dists,
        },
        "per_feature": feature_insight_table(df),
        "correlation": {
            "matrix": correlation_matrix(df),
            "strongest": strongest_correlations(df),
        },
        "class_balance": {
            "balance": balance,
            "negative": balance.negative_count,
            "positive": balance.positive_count,
            "prevalence": round(balance.prevalence, 4),
            "imbalance_ratio": round(balance.imbalance_ratio, 4),
            "zero_table": zero_sentinel_table(df),
            "affected_zero_rows": affected_zero_rows(df),
        },
    }


