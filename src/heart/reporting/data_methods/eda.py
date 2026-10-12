"""Pre-split exploratory data analysis (EDA) engine for M005/S01.

Why this module exists
----------------------
M005's milestone contract is that every number published in
``reports/data-methods.md`` part A is **computed by code that runs entirely
before any train/test split**, on the raw pinned dataset — never hand-written
and never derived from post-split artifacts. This module is the computation
engine: four stage functions over a raw :class:`pandas.DataFrame` plus a
:func:`run_eda` orchestrator that returns a structured, JSON-serialisable
result that the provenance-locked report (S01/T02) and the Streamlit EDA page
(S02) both consume.

The four declared stages
------------------------
1. **Hygiene** (:func:`run_stage1_hygiene`) — shape, dtypes, missingness
   classified as *random* (true NaN) versus *systematic* (the declared
   sentinel zeros from :data:`heart.data.quality.ZERO_AS_MISSING_COLUMNS`),
   exact duplicates via :func:`heart.data.duplicates.find_exact_duplicates`,
   impossible-value counts via :func:`heart.data.quality.audit_quality`, and a
   missingness-pattern PNG.
2. **Univariate** (:func:`run_stage2_univariate`) — per-numeric
   histograms/KDE with skew/kurtosis, IQR box outlier counts, categorical
   frequency tables with rare-level detection
   (:data:`RARE_CATEGORY_FLOOR`), and the target class-distribution ratio.
3. **Bivariate/multivariate** (:func:`run_stage3_bivariate`) —
   feature-vs-target overlaid KDE and stratified box summaries,
   categorical-vs-target contingency tables with chi-square, the Pearson
   correlation matrix with the ``|r| >`` :data:`HIGH_CORRELATION_THRESHOLD`
   flag list, and VIF via the auxiliary-regression formula
   ``1 / (1 - R_j^2)`` on standardized numerics.
4. **Assumptions** (:func:`run_stage4_assumptions`) — empirical logit plots
   (decile bins of each numeric feature against the log-odds of the target)
   and an outlier-impact comparison table measuring *trim* versus *Winsorize*
   versus the declared median-impute policy directly on the raw frame.

Provenance and determinism
--------------------------
The engine is deliberately deterministic: no sampling, no shuffling, no
wall-clock inputs. Two calls with the same frame produce byte-identical table
data, which is what makes the S01/T02 regeneration-determinism test possible.
The engine never mutates the input frame and never writes outside the caller
supplied ``artifact_dir`` (figures only); when ``artifact_dir`` is ``None`` the
figure *data* is still computed but no PNG is rendered, so pure-table callers
stay cheap.

Observability
-------------
Every stage logs one ``INFO`` line (rows examined, patterns found, artifacts
written) in the style of :mod:`heart.data.quality` and
:mod:`heart.reporting.leaderboard`. Failures raise named exceptions from the
:data:`DataMethodsError` family so callers can catch precisely.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402  # headless rendering, before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, gaussian_kde
from sklearn.linear_model import LinearRegression

from heart.config import RANDOM_SEED
from heart.data.duplicates import (
    EXACT_KEY_ALL_COLUMNS,
    find_exact_duplicates,
)
from heart.data.quality import (
    MEDIAN_IMPUTE,
    ZERO_AS_MISSING_COLUMNS,
    ZeroPolicy,
    audit_quality,
)
from heart.data.schema import TARGET_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared engine constants
# ---------------------------------------------------------------------------

#: A categorical level observed in fewer than this share of rows is flagged
#: rare (5 % of the frame — small enough to be noise, large enough to matter).
RARE_CATEGORY_FLOOR: float = 0.05

#: Absolute Pearson |r| above which a feature pair is flagged as collinear.
HIGH_CORRELATION_THRESHOLD: float = 0.8

#: Fixed histogram bin count for per-numeric univariate summaries.
HIST_BINS: int = 20

#: Fixed decile count for empirical-logit binning (qcut drops duplicates).
EMPIRICAL_LOGIT_BINS: int = 10

#: Additive smoothing constant for the empirical logit so bins with 0 positives
#: or 0 negatives yield a finite log-odds instead of +/- infinity.
LOGIT_SMOOTHING: float = 0.5

#: IQR multiplier for the box-outlier fences used in stages 2 and 4.
IQR_FENCE_MULTIPLIER: float = 1.5

#: Numerical guard so a perfect auxiliary regression (R^2 -> 1) yields a large
#: finite VIF instead of a division by zero.
MAX_VIF: float = 1.0e12


# ---------------------------------------------------------------------------
# Named exceptions (DataMethodsError family)
# ---------------------------------------------------------------------------


class DataMethodsError(Exception):
    """Base class for every pre-split EDA engine failure."""


class NonDataFrameError(DataMethodsError):
    """A stage received something other than a pandas.DataFrame."""


class EmptyDatasetError(DataMethodsError):
    """A stage received a structurally valid but empty frame."""


class MissingTargetColumnError(DataMethodsError):
    """The declared target column is absent from the frame."""


class StageComputationError(DataMethodsError):
    """A stage's statistical computation failed on the given data."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_raw_frame(df: object, *, target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Fail loudly and namedly unless ``df`` is a usable raw frame."""
    if not isinstance(df, pd.DataFrame):
        raise NonDataFrameError(
            f"Expected a pandas.DataFrame, got {type(df).__name__}. The EDA "
            "engine computes column statistics, so a bare array or dict is "
            "not sufficient — pass the frame from heart.data.load_dataset."
        )
    if df.empty:
        raise EmptyDatasetError(
            "Received an empty frame with no rows to examine. The pre-split "
            "EDA must run on the real dataset, not a placeholder."
        )
    if target not in df.columns:
        raise MissingTargetColumnError(
            f"Target column {target!r} is absent from the frame; stages 2-4 "
            "measure feature-vs-target relationships and cannot run without "
            f"it. Expected one of {list(df.columns)!r}."
        )
    return df


def _column_roles(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Split non-target columns into numeric and categorical by content.

    A numeric column with two or fewer distinct values is treated as
    categorical (this maps the real schema's binary ``FastingBS`` correctly
    and keeps arbitrary test fixtures working without hardcoding schema
    names). Column order follows the frame, so results are deterministic.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 2:
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def _jsonable(value: Any) -> Any:
    """Recursively convert a stage result into strictly JSON-serialisable data.

    NaN becomes ``None`` (strict JSON has no NaN), floats are rounded to six
    decimals so byte-identical regeneration is stable across runs, and numpy
    scalars/arrays are unwrapped to native Python types.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)  # must precede the int branch: bool subclasses int
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return math.copysign(MAX_VIF, number)
        return round(number, 6)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _save_figure(
    fig: "plt.Figure",
    artifact_dir: Path | None,
    filename: str,
    figures: list[str],
) -> None:
    """Render one figure as PNG under ``artifact_dir`` (no-op when None)."""
    if artifact_dir is None:
        plt.close(fig)
        return
    path = Path(artifact_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)
    figures.append(filename)


def _kde_curve(series: pd.Series) -> dict[str, Any] | None:
    """Return a Gaussian KDE curve for ``series`` or ``None`` if degenerate."""
    values = np.asarray(series.dropna(), dtype=float)
    if values.size < 2 or np.all(values == values[0]):
        return None
    try:
        kde = gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), 128)
        return {"x": grid.tolist(), "y": kde(grid).tolist()}
    except Exception as error:  # singular covariance etc. — named, not silent
        raise StageComputationError(
            f"Gaussian KDE failed on a numeric series: {error}"
        ) from error


def _iqr_fences(series: pd.Series) -> tuple[float, float, float, float]:
    """Return ``(q1, q3, low_fence, high_fence)`` for an IQR outlier box."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1, q3, q1 - IQR_FENCE_MULTIPLIER * iqr, q3 + IQR_FENCE_MULTIPLIER * iqr


def _corr_with_target(
    column: str, frame: pd.DataFrame, target: str
) -> float | None:
    """Pearson correlation of one numeric column with the binary target."""
    pair = frame[[column, target]].dropna()
    if len(pair) < 3 or pair[column].std() == 0 or pair[target].nunique() < 2:
        return None
    correlation = pair[column].astype(float).corr(pair[target].astype(float))
    return None if correlation is None or math.isnan(correlation) else float(correlation)


def _summary_stats(series: pd.Series) -> dict[str, Any]:
    """Count/mean/std summary for one numeric series (JSON-safe)."""
    values = series.dropna().astype(float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
    }


# ---------------------------------------------------------------------------
# Stage 1 — hygiene
# ---------------------------------------------------------------------------


def run_stage1_hygiene(
    df: pd.DataFrame,
    *,
    artifact_dir: str | Path | None = None,
    target: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Stage 1 — hygiene: shape, dtypes, missingness, duplicates, sentinels.

    Missingness is classified into *random* (true NaN, any column) and
    *systematic* (the declared sentinel zeros in
    :data:`heart.data.quality.ZERO_AS_MISSING_COLUMNS`). Impossible-value
    counts reuse :func:`heart.data.quality.audit_quality` and exact
    duplicates reuse :func:`heart.data.duplicates.find_exact_duplicates`,
    falling back to a frame-derived exact-duplicate tally for fixture frames
    that do not carry the full declared schema.
    """
    _validate_raw_frame(df, target=target)
    figures: list[str] = []

    # -- missingness classification: random NaN vs systematic sentinel zeros
    random_nan: dict[str, int] = {
        str(column): int(df[column].isna().sum()) for column in df.columns
    }
    sentinel_columns = [c for c in ZERO_AS_MISSING_COLUMNS if c in df.columns]
    systematic_sentinel: dict[str, int] = {
        str(column): int((df[column] == 0).sum()) for column in sentinel_columns
    }
    missing_cells = int(sum(random_nan.values()))
    sentinel_cells = int(sum(systematic_sentinel.values()))
    any_missing = df.isna().any(axis=1)
    if sentinel_columns:
        any_missing = any_missing | (df[sentinel_columns] == 0).any(axis=1)
    rows_affected = int(any_missing.sum())

    # -- exact duplicates (reuses heart.data.duplicates when the schema is complete)
    if set(EXACT_KEY_ALL_COLUMNS).issubset(df.columns):
        exact_duplicates = find_exact_duplicates(df).to_dict()
    else:
        mask = df.duplicated(keep=False)
        grouped = df[mask].groupby(list(df.columns), dropna=False).groups if mask.any() else {}
        sizes = [len(indices) for indices in grouped.values()]
        histogram: dict[str, int] = {}
        for size in sizes:
            histogram[str(size)] = histogram.get(str(size), 0) + 1
        exact_duplicates = {
            "columns": list(df.columns),
            "duplicate_rows": int(mask.sum()),
            "duplicate_groups": len(grouped),
            "group_size_histogram": histogram,
            "row_indices": [int(i) for i in np.flatnonzero(mask.to_numpy())],
        }

    # -- impossible values under the declared zero-as-missing policy
    if sentinel_columns:
        policy = ZeroPolicy(
            action=MEDIAN_IMPUTE,
            columns=tuple(sentinel_columns),
            rationale="Pre-split EDA hygiene audit under the declared zero-as-missing policy.",
        )
        impossible_values: dict[str, Any] = dict(audit_quality(df, policy=policy).to_dict())
    else:
        impossible_values = {
            "row_count": int(len(df)),
            "affected_rows": 0,
            "zero_counts": {},
            "note": "no declared sentinel column present in this frame",
        }

    # -- missingness-pattern figure (0 = present, 1 = random NaN, 2 = sentinel zero)
    pattern_codes = np.zeros(df.shape, dtype=float)
    for position, column in enumerate(df.columns):
        pattern_codes[:, position] += df[column].isna().to_numpy() * 1.0
        if column in sentinel_columns:
            pattern_codes[:, position] += (df[column] == 0).to_numpy() * 2.0
    fig, axis = plt.subplots(figsize=(8, max(3.0, 0.03 * len(df) + 1.5)))
    axis.imshow(pattern_codes, aspect="auto", interpolation="nearest", vmin=0, vmax=2)
    axis.set_xticks(range(len(df.columns)))
    axis.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("row index")
    axis.set_title("Missingness patterns: random NaN (dark) vs sentinel zero (light)")
    _save_figure(fig, artifact_dir, "stage1_missingness_patterns.png", figures)

    result: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missingness": {
            "classification": {
                "random_nan": "true NaN cells (random missingness)",
                "systematic_sentinel": "declared sentinel zeros (systematic missingness)",
            },
            "random_nan": random_nan,
            "systematic_sentinel": systematic_sentinel,
            "total_random_nan_cells": missing_cells,
            "total_sentinel_zero_cells": sentinel_cells,
            "rows_with_any_missingness": rows_affected,
        },
        "exact_duplicates": exact_duplicates,
        "impossible_values": impossible_values,
        "figures": figures,
    }
    logger.info(
        "stage 1 (hygiene): examined %d rows x %d columns; %d random-NaN cells, "
        "%d sentinel-zero cells across %s, %d exact-duplicate row(s); "
        "artifacts written: %s",
        df.shape[0],
        df.shape[1],
        missing_cells,
        sentinel_cells,
        list(systematic_sentinel) or "none",
        exact_duplicates["duplicate_rows"],
        figures or "none (artifact_dir=None)",
    )
    return _jsonable(result)


# ---------------------------------------------------------------------------
# Stage 2 — univariate
# ---------------------------------------------------------------------------


def run_stage2_univariate(
    df: pd.DataFrame,
    *,
    artifact_dir: str | Path | None = None,
    target: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Stage 2 — univariate: histograms/KDE, IQR outliers, rare levels, target ratio."""
    _validate_raw_frame(df, target=target)
    numeric, categorical = _column_roles(df, target)
    figures: list[str] = []
    total_rows = int(len(df))

    numeric_summary: dict[str, Any] = {}
    for column in numeric:
        series = df[column].dropna().astype(float)
        hist_counts, hist_edges = np.histogram(series, bins=HIST_BINS)
        figures_key = f"stage2_hist_{column}.png"
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.hist(series, bins=HIST_BINS, density=True, alpha=0.5, edgecolor="white")
        kde = _kde_curve(series)
        if kde is not None:
            axis.plot(kde["x"], kde["y"], linewidth=2)
        axis.set_title(f"Distribution of {column} (pre-split, raw frame)")
        axis.set_xlabel(column)
        axis.set_ylabel("density")
        _save_figure(fig, artifact_dir, figures_key, figures)

        q1, q3, low_fence, high_fence = _iqr_fences(series)
        outlier_mask = (series < low_fence) | (series > high_fence)
        numeric_summary[column] = {
            "histogram": {
                "bin_edges": [float(edge) for edge in hist_edges],
                "counts": [int(count) for count in hist_counts],
            },
            "kde": kde,
            "skew": float(series.skew()),
            "kurtosis": float(series.kurt()),
            "box_outliers": {
                "q1": q1,
                "q3": q3,
                "low_fence": low_fence,
                "high_fence": high_fence,
                "count": int(outlier_mask.sum()),
                "share": float(outlier_mask.mean()) if len(series) else None,
            },
        }

    categorical_summary: dict[str, Any] = {}
    for column in categorical:
        series = df[column].dropna()
        counts = series.value_counts()
        ordered = sorted(
            ((str(level), int(count)) for level, count in counts.items()),
            key=lambda item: (-item[1], item[0]),
        )
        rare_levels = [
            level for level, count in ordered if count / total_rows < RARE_CATEGORY_FLOOR
        ]
        figures_key = f"stage2_bar_{column}.png"
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.bar([level for level, _ in ordered], [count for _, count in ordered])
        axis.set_title(f"Frequency of {column} (pre-split, raw frame)")
        axis.set_ylabel("rows")
        axis.tick_params(axis="x", rotation=45)
        _save_figure(fig, artifact_dir, figures_key, figures)
        categorical_summary[column] = {
            "frequencies": dict(ordered),
            "distinct_levels": len(ordered),
            "rare_levels": rare_levels,
            "rare_floor_share": RARE_CATEGORY_FLOOR,
        }

    target_counts = df[target].value_counts()
    ordered_target = sorted(
        ((str(level), int(count)) for level, count in target_counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    positives = int((df[target] == 1).sum()) if pd.api.types.is_numeric_dtype(df[target]) else None
    negatives = int(len(df)) - (positives or 0)
    target_summary: dict[str, Any] = {
        "counts": dict(ordered_target),
        "prevalence": float(positives / total_rows) if positives is not None else None,
        "imbalance_ratio": float(negatives / positives) if positives else None,
    }

    result: dict[str, Any] = {
        "numeric": numeric_summary,
        "categorical": categorical_summary,
        "target": target_summary,
        "figures": figures,
    }
    logger.info(
        "stage 2 (univariate): examined %d numeric and %d categorical columns; "
        "%d IQR box outlier flag(s), %d rare categorical level(s) below the %.0f%% "
        "floor; artifacts written: %s",
        len(numeric),
        len(categorical),
        sum(entry["box_outliers"]["count"] for entry in numeric_summary.values()),
        sum(len(entry["rare_levels"]) for entry in categorical_summary.values()),
        RARE_CATEGORY_FLOOR * 100,
        figures or "none (artifact_dir=None)",
    )
    return _jsonable(result)


# ---------------------------------------------------------------------------
# Stage 3 — bivariate / multivariate
# ---------------------------------------------------------------------------


def run_stage3_bivariate(
    df: pd.DataFrame,
    *,
    artifact_dir: str | Path | None = None,
    target: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Stage 3 — feature-vs-target overlays, chi-square, correlations, VIF."""
    _validate_raw_frame(df, target=target)
    numeric, categorical = _column_roles(df, target)
    figures: list[str] = []
    target_levels = sorted(str(level) for level in df[target].dropna().unique())

    numeric_vs_target: dict[str, Any] = {}
    for column in numeric:
        group_kde: dict[str, Any] = {}
        group_box: dict[str, Any] = {}
        fig, axis = plt.subplots(figsize=(6, 4))
        for level in target_levels:
            subset = df.loc[df[target].astype(str) == level, column].dropna().astype(float)
            label = f"{column} | {target}={level}"
            curve = _kde_curve(subset)
            if curve is not None:
                axis.plot(curve["x"], curve["y"], label=label)
            if subset.size:
                group_box[level] = {
                    "n": int(subset.size),
                    "min": float(subset.min()),
                    "q1": float(subset.quantile(0.25)),
                    "median": float(subset.quantile(0.5)),
                    "q3": float(subset.quantile(0.75)),
                    "max": float(subset.max()),
                    "mean": float(subset.mean()),
                }
            curve = curve or {}
            group_kde[level] = {"x": curve.get("x"), "y": curve.get("y")}
        axis.set_title(f"{column} vs {target} (overlaid KDE, pre-split)")
        axis.set_xlabel(column)
        axis.legend(fontsize=7)
        _save_figure(fig, artifact_dir, f"stage3_overlay_{column}.png", figures)
        numeric_vs_target[column] = {
            "group_kde": group_kde,
            "group_box": group_box,
        }

    contingency: dict[str, Any] = {}
    for column in categorical:
        table = pd.crosstab(df[column].astype(str), df[target].astype(str))
        if table.shape[0] < 2 or table.shape[1] < 2:
            chi_square: dict[str, Any] = {
                "statistic": None,
                "p_value": None,
                "dof": None,
                "note": "degenerate table: a constant feature or target has no chi-square test",
            }
        else:
            statistic, p_value, dof, _expected = chi2_contingency(table.to_numpy())
            chi_square = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "dof": int(dof),
            }
        contingency[column] = {
            "table": {str(row): {str(col): int(cell) for col, cell in row_cells.items()}
                      for row, row_cells in table.to_dict(orient="index").items()},
            "chi_square": chi_square,
        }

    correlation = df[numeric].astype(float).corr(method="pearson")
    correlation_matrix = {
        str(row): {str(col): (None if pd.isna(cell) else float(cell))
                   for col, cell in row_cells.items()}
        for row, row_cells in correlation.to_dict(orient="index").items()
    }
    high_correlations: list[dict[str, Any]] = []
    for index, row_name in enumerate(numeric):
        for column_name in numeric[index + 1 :]:
            coefficient = correlation.loc[row_name, column_name]
            if pd.notna(coefficient) and abs(float(coefficient)) > HIGH_CORRELATION_THRESHOLD:
                high_correlations.append(
                    {
                        "a": row_name,
                        "b": column_name,
                        "r": float(coefficient),
                        "abs_r": abs(float(coefficient)),
                    }
                )

    # -- VIF via the auxiliary-regression formula 1 / (1 - R_j^2)
    design = df[numeric].astype(float)
    design = design.fillna(design.median())
    std = design.std(ddof=0).replace(0, np.nan)
    standardized = design.sub(design.mean()).div(std.where(std.notna(), 1.0))
    standardized = standardized.fillna(0.0)
    vif: dict[str, Any] = {}
    for column in numeric:
        others = standardized.drop(columns=[column])
        if others.shape[1] == 0:
            vif[column] = 1.0
            continue
        response = standardized[column]
        model = LinearRegression().fit(others.to_numpy(), response.to_numpy())
        r_squared = float(model.score(others.to_numpy(), response.to_numpy()))
        r_squared = min(max(r_squared, 0.0), 1.0 - 1.0 / MAX_VIF)
        vif[column] = 1.0 / (1.0 - r_squared)

    if len(numeric) >= 2:
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(numeric)))
        axis.set_xticklabels(numeric, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(numeric)))
        axis.set_yticklabels(numeric, fontsize=8)
        axis.set_title(f"Pearson correlation (|r| > {HIGH_CORRELATION_THRESHOLD} flagged)")
        fig.colorbar(image, ax=axis, shrink=0.8)
        _save_figure(fig, artifact_dir, "stage3_correlation_heatmap.png", figures)

    result: dict[str, Any] = {
        "numeric_vs_target": numeric_vs_target,
        "contingency": contingency,
        "correlation": {
            "matrix": correlation_matrix,
            "threshold": HIGH_CORRELATION_THRESHOLD,
            "high_correlations": high_correlations,
        },
        "vif": vif,
        "figures": figures,
    }
    logger.info(
        "stage 3 (bivariate/multivariate): examined %d numeric and %d categorical "
        "columns vs the target; %d chi-square test(s), %d collinear pair(s) with "
        "|r| > %.1f, max VIF %.1f; artifacts written: %s",
        len(numeric),
        len(categorical),
        sum(1 for entry in contingency.values() if entry["chi_square"]["statistic"] is not None),
        len(high_correlations),
        HIGH_CORRELATION_THRESHOLD,
        max(vif.values()) if vif else 0.0,
        figures or "none (artifact_dir=None)",
    )
    return _jsonable(result)


# ---------------------------------------------------------------------------
# Stage 4 — assumptions
# ---------------------------------------------------------------------------


def run_stage4_assumptions(
    df: pd.DataFrame,
    *,
    artifact_dir: str | Path | None = None,
    target: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Stage 4 — empirical logit plots and the outlier-impact comparison."""
    _validate_raw_frame(df, target=target)
    numeric, _categorical = _column_roles(df, target)
    figures: list[str] = []

    # -- empirical logit: decile bins per numeric feature vs log-odds(target)
    empirical_logit: dict[str, Any] = {}
    for column in numeric:
        series = df[column].dropna().astype(float)
        if series.nunique() < 2:
            empirical_logit[column] = {
                "bins": [],
                "note": "constant feature: no decile binning possible",
            }
            continue
        binned = pd.qcut(series, q=EMPIRICAL_LOGIT_BINS, duplicates="drop")
        analysis = pd.DataFrame({"value": series, "bin": binned, "target": df[target]})
        bins: list[dict[str, Any]] = []
        for interval, group in analysis.groupby("bin", observed=True):
            positives = int((group["target"] == 1).sum())
            negatives = int((group["target"] != 1).sum())
            size = int(len(group))
            log_odds = math.log((positives + LOGIT_SMOOTHING) / (negatives + LOGIT_SMOOTHING))
            bins.append(
                {
                    "bin_low": float(interval.left),
                    "bin_high": float(interval.right),
                    "n": size,
                    "positives": positives,
                    "negatives": negatives,
                    "logit": log_odds,
                }
            )
        fig, axis = plt.subplots(figsize=(6, 4))
        midpoints = [(entry["bin_low"] + entry["bin_high"]) / 2 for entry in bins]
        axis.plot(midpoints, [entry["logit"] for entry in bins], marker="o")
        axis.set_title(f"Empirical logit of {column} (decile bins, pre-split)")
        axis.set_xlabel(f"{column} (bin midpoint)")
        axis.set_ylabel(f"log-odds({target}=1)")
        _save_figure(fig, artifact_dir, f"stage4_empirical_logit_{column}.png", figures)
        empirical_logit[column] = {"bins": bins, "smoothing": LOGIT_SMOOTHING}

    # -- outlier-impact comparison: trim vs Winsorize vs declared median-impute
    sentinel_columns = [c for c in ZERO_AS_MISSING_COLUMNS if c in df.columns]
    impact_columns: dict[str, Any] = {}
    impact_table: list[dict[str, Any]] = []
    for column in numeric:
        base = df[column].dropna().astype(float)
        q1, q3, low_fence, high_fence = _iqr_fences(base)
        outlier_mask = (base < low_fence) | (base > high_fence)
        policies: dict[str, Any] = {"raw": {**_summary_stats(base), "corr_with_target": _corr_with_target(column, df, target)}}

        kept = base[~outlier_mask]
        trim_frame = df.loc[kept.index]
        policies["trim_iqr"] = {
            **_summary_stats(kept),
            "corr_with_target": _corr_with_target(column, trim_frame, target),
            "rows_dropped": int(outlier_mask.sum()),
        }
        winsorized = base.clip(low_fence, high_fence)
        winsor_frame = df.assign(**{column: winsorized})
        policies["winsorize_iqr"] = {
            **_summary_stats(winsorized),
            "corr_with_target": _corr_with_target(column, winsor_frame, target),
        }
        declared_applies = column in sentinel_columns
        if declared_applies:
            nonzero_median = float(base[base != 0].median())
            imputed = base.where(base != 0, nonzero_median)
            imputed_frame = df.assign(**{column: imputed})
            policies["declared_median_impute"] = {
                **_summary_stats(imputed),
                "corr_with_target": _corr_with_target(column, imputed_frame, target),
                "cells_replaced": int((base == 0).sum()),
                "fill_value": nonzero_median,
                "note": (
                    "descriptive raw-frame analogue of the declared policy; "
                    "production imputation fits the training-fold median via "
                    "heart.data.quality.ZeroMedianImputer"
                ),
            }
        else:
            policies["declared_median_impute"] = None
        policies["declared_policy_applies"] = declared_applies
        impact_columns[column] = policies
        for policy_name, metrics in policies.items():
            if policy_name in ("declared_policy_applies",) or metrics is None:
                continue
            baseline = policies["raw"]
            impact_table.append(
                {
                    "column": column,
                    "policy": policy_name,
                    "n": metrics["n"],
                    "mean": metrics["mean"],
                    "delta_mean": (
                        None
                        if metrics["mean"] is None or baseline["mean"] is None
                        else metrics["mean"] - baseline["mean"]
                    ),
                    "std": metrics["std"],
                    "delta_std": (
                        None
                        if metrics["std"] is None or baseline["std"] is None
                        else metrics["std"] - baseline["std"]
                    ),
                    "corr_with_target": metrics.get("corr_with_target"),
                }
            )

    result: dict[str, Any] = {
        "empirical_logit": empirical_logit,
        "outlier_impact": {
            "fence_definition": f"q1 - {IQR_FENCE_MULTIPLIER}*IQR / q3 + {IQR_FENCE_MULTIPLIER}*IQR",
            "declared_sentinel_columns": sentinel_columns,
            "columns": impact_columns,
            "table": impact_table,
        },
        "figures": figures,
    }
    logger.info(
        "stage 4 (assumptions): examined %d numeric column(s); %d empirical-logit "
        "bin set(s) rendered, outlier-impact table computed for trim vs Winsorize "
        "vs the declared median-impute policy on sentinel columns %s; "
        "artifacts written: %s",
        len(numeric),
        sum(1 for entry in empirical_logit.values() if entry["bins"]),
        sentinel_columns or "none",
        figures or "none (artifact_dir=None)",
    )
    return _jsonable(result)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_eda(
    df: pd.DataFrame,
    *,
    artifact_dir: str | Path | None = None,
    target: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Run all four pre-split EDA stages and return a JSON-serialisable result.

    Parameters
    ----------
    df:
        The **raw** dataset frame — the caller must pass the pinned raw data
        (via :func:`heart.data.load_dataset`), never a post-split or
        post-transform frame.
    artifact_dir:
        Directory to render stage PNG figures into. ``None`` computes all
        table data without rendering figures.
    target:
        Target column name (default :data:`heart.data.schema.TARGET_COLUMN`).

    Returns
    -------
    dict
        ``{"meta", "stage1_hygiene", "stage2_univariate", "stage3_bivariate",
        "stage4_assumptions"}`` — strictly JSON-serialisable and deterministic
        for a given input frame.
    """
    _validate_raw_frame(df, target=target)
    numeric, categorical = _column_roles(df, target)
    logger.info(
        "pre-split EDA: starting on %d rows x %d columns (seed %d, target %r); "
        "no train/test split has occurred at any point in this pipeline",
        df.shape[0],
        df.shape[1],
        RANDOM_SEED,
        target,
    )
    result: dict[str, Any] = {
        "meta": {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "target_column": target,
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "declared_zero_as_missing_columns": [
                column for column in ZERO_AS_MISSING_COLUMNS if column in df.columns
            ],
            "seed": RANDOM_SEED,
        },
        "stage1_hygiene": run_stage1_hygiene(df, artifact_dir=artifact_dir, target=target),
        "stage2_univariate": run_stage2_univariate(df, artifact_dir=artifact_dir, target=target),
        "stage3_bivariate": run_stage3_bivariate(df, artifact_dir=artifact_dir, target=target),
        "stage4_assumptions": run_stage4_assumptions(df, artifact_dir=artifact_dir, target=target),
    }
    logger.info(
        "pre-split EDA: all four stages complete; stages covered hygiene, "
        "univariate, bivariate/multivariate, and assumption checks"
    )
    return _jsonable(result)
