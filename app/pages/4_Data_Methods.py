"""Data & Methods page: plain-language rendering of the generated EDA report.

The page is deliberately thin: every number and figure comes from the S01
report machinery's artifacts under ``reports/data-methods/`` via the T01
contract module (:func:`app.lib.data_methods_view.load_data_methods_view`),
which re-verifies the SHA-256 provenance manifest through the S01 engine
itself — so the page and ``reports/data-methods.md`` trace to identical
artifacts by construction, never by coincidence (D025). Nothing here
recomputes a statistic.

Never a stack trace: the contract module never raises (MEM047), and a
missing, unreadable, corrupt or provenance-broken artifact directory renders
the named friendly error with the documented regeneration command — the same
hint the report's technical appendix prints — mirroring the Runs page's
store-empty/store-missing branch. Every table and figure caption cites its
artifact path so the page stays auditable against the report.

This page is deliberately *uncached*: one artifact read per rerun, so a
test-time ``DATA_METHODS_DIR`` env override (the seam in the contract
module) can never serve stale artifacts from a previous generation run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib import data_methods_view  # noqa: E402
from app.lib.data_methods_view import REGENERATION_COMMAND  # noqa: E402
from heart.reporting.data_methods.report import DATAMETHODS_DIR  # noqa: E402

st.set_page_config(
    page_title="4 — Data Methods",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data & Methods — what the dataset looks like")
st.caption(
    "Plain-language summary of the four-stage pre-split exploratory analysis "
    "behind `reports/data-methods.md`. Every table and figure below is read "
    "from the generated artifacts (SHA-256 provenance-verified), never "
    "recomputed. Regenerate them with `PYTHONPATH=src python -m "
    "heart.reporting.data_methods`."
)

view = data_methods_view.load_data_methods_view()

if not view.ok:
    st.error(
        f"**The Data & Methods report is unavailable ({view.error_kind}).** "
        f"{view.error_message}"
    )
    st.caption(f"Attempted artifact directory: `{view.attempted_dir}`.")
    st.stop()

# The figure captions cite the artifact path the way the report does
# ("reports/data-methods/<figure>.png") when the committed default directory
# is in use; a test-time seam directory shows its own path instead.
_FIG_PREFIX = (
    "reports/data-methods"
    if str(Path(view.attempted_dir).resolve()) == str(Path(DATAMETHODS_DIR).resolve())
    else str(view.attempted_dir)
)
_FIGURES: dict[str, Path] = {path.name: path for path in view.figure_paths}

st.subheader("Dataset provenance")
st.caption("Artifact: meta.json · verified against manifest.json (SHA-256)")
assert view.meta is not None  # narrow for the type checker; view.ok pins it
st.info(
    "**{rows:,} rows × {cols} columns** · target column `{target}` · random "
    "seed {seed}. Declared zero-as-missing columns (sentinel `0` means "
    "'value not recorded'): {zeros}. The pinned raw-CSV digest verified "
    "clean before any of these numbers were computed.".format(
        rows=int(view.meta.get("row_count", 0)),
        cols=int(view.meta.get("column_count", 0)),
        target=view.meta.get("target_column", "?"),
        seed=view.meta.get("seed", "?"),
        zeros=", ".join(f"`{c}`" for c in view.meta.get(
            "declared_zero_as_missing_columns", [])) or "_none declared_",
    )
)


def _stage_heading(stage_title: str, artifact: str) -> None:
    st.subheader(stage_title)
    st.caption(f"Artifact: {artifact}")


def _df_table(frame: pd.DataFrame) -> None:
    st.dataframe(frame, width="stretch", hide_index=True)


def _figure(name: str, caption: str) -> None:
    path = _FIGURES.get(name)
    if path is None:
        return  # contract module guarantees referenced figures exist; be safe
    st.image(str(path), caption=f"{_FIG_PREFIX}/{name} — {caption}")


def _fmt(value: object, nd: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


# ---------------------------------------------------------------------------
# Stage 1 — hygiene: missingness, duplicates, impossible values
# ---------------------------------------------------------------------------

stage1 = view.stages["stage1_hygiene"]
_stage_heading("Stage 1 — Data hygiene", "stage1_hygiene.json")

missingness = stage1.get("missingness", {})
random_nan = missingness.get("random_nan", {})
sentinel = missingness.get("systematic_sentinel", {})
st.markdown(
    "The dataset declares some `0` values as *'not recorded'* rather than a "
    "real measurement (a sentinel). Missingness is therefore split in two: "
    "true NaN cells and declared sentinel zeros. This dataset has "
    "**{nan} true-NaN cell(s)** and **{sent} sentinel-zero cell(s)** across "
    "**{rows}** row(s) touched by any missingness.".format(
        nan=missingness.get("total_random_nan_cells", 0),
        sent=missingness.get("total_sentinel_zero_cells", 0),
        rows=missingness.get("rows_with_any_missingness", 0),
    )
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": column,
                "True NaN cells": random_nan.get(column, 0),
                "Sentinel zeros": sentinel.get(column, 0),
            }
            for column in sorted(set(random_nan) | set(sentinel))
        ],
        columns=["Column", "True NaN cells", "Sentinel zeros"],
    ),
    width="stretch",
    hide_index=True,
)

duplicates = stage1.get("exact_duplicates", {})
st.markdown(
    "Exact duplicate rows: **{}** (in {} duplicate group(s)) — duplicates "
    "would let the same patient leak across a future train/test split, so "
    "zero is the healthy number.".format(
        duplicates.get("duplicate_rows", 0),
        duplicates.get("duplicate_groups", 0),
    )
)

impossible = stage1.get("impossible_values", {})
zero_counts = impossible.get("zero_counts", {})
policy = impossible.get("policy", {})
st.markdown(
    "Under the declared zero-as-missing policy, {rows} row(s) are affected "
    "by sentinel zeros. The audit pins each column's zero count (and its "
    "split by target class) before any modelling decision:".format(
        rows=impossible.get("affected_rows", 0),
    )
)
_rows = []
for column, entry in zero_counts.items():
    if not isinstance(entry, dict):
        continue
    by_target = entry.get("by_target", {})
    _rows.append(
        {
            "Column": str(column),
            "Zeros": str(entry.get("count", "")),
            "Share": _fmt(entry.get("share")),
            "Zeros in class 0": str(by_target.get("0", "")) if "0" in by_target else "",
            "Zeros in class 1": str(by_target.get("1", "")) if "1" in by_target else "",
        }
    )
if _rows:
    _df_table(pd.DataFrame(_rows))
if policy:
    st.caption(
        "Declared policy: action `{action}` on {columns} (sentinel value "
        "{sentinel}). {rationale}".format(
            action=policy.get("action", "?"),
            columns=", ".join(f"`{c}`" for c in policy.get("columns", [])) or "_none_",
            sentinel=policy.get("sentinel", "?"),
            rationale=policy.get("rationale", ""),
        )
    )
_figure(
    "stage1_missingness_patterns.png",
    "where missingness lives, split true-NaN vs sentinel zeros",
)

# ---------------------------------------------------------------------------
# Stage 2 — univariate: skew, outliers, category frequencies
# ---------------------------------------------------------------------------

stage2 = view.stages["stage2_univariate"]
_stage_heading("Stage 2 — Per-column distributions", "stage2_univariate.json")

numeric = stage2.get("numeric", {})
st.markdown(
    "Each numeric column was checked for skew and box-plot outliers. Skew "
    "far from 0 hints the column is lopsided; outliers are values beyond "
    "the 1.5×IQR fences."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": column,
                "Skew": _fmt(entry.get("skew")),
                "Box outliers": str(entry.get("box_outliers", {}).get("count", "")),
                "Outlier share": _fmt(entry.get("box_outliers", {}).get("share")),
            }
            for column, entry in numeric.items()
            if isinstance(entry, dict)
        ],
        columns=["Column", "Skew", "Box outliers", "Outlier share"],
    ),
    width="stretch",
    hide_index=True,
)
for column in numeric:
    _figure(f"stage2_hist_{column}.png", f"distribution of `{column}`")

categorical = stage2.get("categorical", {})
st.markdown(
    "Categorical columns were checked for level counts and rare levels "
    "(share below the 5% floor), which a future split would have to handle "
    "consistently."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": column,
                "Distinct levels": str(entry.get("distinct_levels", "")),
                "Rare levels": ", ".join(entry.get("rare_levels", [])) or "_none_",
            }
            for column, entry in categorical.items()
            if isinstance(entry, dict)
        ],
        columns=["Column", "Distinct levels", "Rare levels"],
    ),
    width="stretch",
    hide_index=True,
)
for column in categorical:
    _figure(f"stage2_bar_{column}.png", f"category frequencies of `{column}`")

_figure_target = stage2.get("target", {})
if _figure_target:
    st.caption(
        "Target `{target}`: {counts} · prevalence {prev} · imbalance ratio "
        "{ratio}.".format(
            target=view.meta.get("target_column", "?"),
            counts=_figure_target.get("counts", {}),
            prev=_fmt(_figure_target.get("prevalence")),
            ratio=_fmt(_figure_target.get("imbalance_ratio")),
        )
    )

# ---------------------------------------------------------------------------
# Stage 3 — bivariate: chi-square, correlation, VIF
# ---------------------------------------------------------------------------

stage3 = view.stages["stage3_bivariate"]
_stage_heading("Stage 3 — Column-vs-target relationships", "stage3_bivariate.json")

contingency = stage3.get("contingency", {})
st.markdown(
    "Each categorical column was tested against the target with a "
    "chi-square test: a small p-value means the column's category mix "
    "really differs between the two target classes (a real signal, not "
    "noise)."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": column,
                "Chi-square": _fmt(entry.get("chi_square", {}).get("statistic")),
                "dof": str(entry.get("chi_square", {}).get("dof", "")),
                "p-value": _fmt(entry.get("chi_square", {}).get("p_value")),
            }
            for column, entry in contingency.items()
            if isinstance(entry, dict)
        ],
        columns=["Column", "Chi-square", "dof", "p-value"],
    ),
    width="stretch",
    hide_index=True,
)

correlation = stage3.get("correlation", {})
high = correlation.get("high_correlations", [])
st.markdown(
    (
        "Pairs of numeric columns whose correlation exceeds the declared "
        "threshold (|r| > 0.8) — these are near-duplicates that a future "
        "model could double-count:"
        if high
        else "No pair of numeric columns exceeds the declared high-"
        "correlation threshold (|r| > 0.8): the model would not double-count "
        "any two of them."
    )
)
if high:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Column A": pair.get("a", ""),
                    "Column B": pair.get("b", ""),
                    "r": _fmt(pair.get("r")),
                    "|r|": _fmt(pair.get("abs_r")),
                }
                for pair in high
                if isinstance(pair, dict)
            ],
            columns=["Column A", "Column B", "r", "|r|"],
        ),
        width="stretch",
        hide_index=True,
    )

vif = stage3.get("vif", {})
st.markdown(
    "Variance inflation factors (VIF) for the numeric columns: values near "
    "1 mean the column carries information the others do not; values well "
    "above 10 would signal redundant collinearity."
)
st.dataframe(
    pd.DataFrame(
        [{"Column": column, "VIF": _fmt(value)} for column, value in vif.items()],
        columns=["Column", "VIF"],
    ),
    width="stretch",
    hide_index=True,
)

for column in numeric:
    _figure(
        f"stage3_overlay_{column}.png",
        f"`{column}` distribution split by target class",
    )
_figure("stage3_correlation_heatmap.png", "correlation matrix of the numeric columns")

# ---------------------------------------------------------------------------
# Stage 4 — assumptions: outlier policy impact, empirical logit
# ---------------------------------------------------------------------------

stage4 = view.stages["stage4_assumptions"]
_stage_heading(
    "Stage 4 — Robustness assumptions", "stage4_assumptions.json"
)

outlier_impact = stage4.get("outlier_impact", {}).get("columns", {})
st.markdown(
    "Before trusting any model, the analysis checked how the declared "
    "outlier policy would change the data: raw vs IQR-trimmed vs "
    "IQR-winsorized columns. If a policy barely moves the mean, spread and "
    "target correlation, the choice is not load-bearing."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": column,
                "n (raw)": str(entry.get("raw", {}).get("n", "")),
                "Rows trimmed": str(entry.get("trim_iqr", {}).get("rows_dropped", "")),
                "Mean (raw)": _fmt(entry.get("raw", {}).get("mean")),
                "Mean (winsorized)": _fmt(entry.get("winsorize_iqr", {}).get("mean")),
                "r with target (raw)": _fmt(entry.get("raw", {}).get("corr_with_target")),
                "r (trimmed)": _fmt(entry.get("trim_iqr", {}).get("corr_with_target")),
                "Declared policy applies": "yes"
                if entry.get("declared_policy_applies")
                else "no",
            }
            for column, entry in outlier_impact.items()
            if isinstance(entry, dict)
        ],
        columns=[
            "Column", "n (raw)", "Rows trimmed", "Mean (raw)",
            "Mean (winsorized)", "r with target (raw)", "r (trimmed)",
            "Declared policy applies",
        ],
    ),
    width="stretch",
    hide_index=True,
)

empirical_logit = stage4.get("empirical_logit", {})
st.markdown(
    "The empirical-logit check asks whether each numeric column relates to "
    "the target *linearly* in log-odds — the assumption a logistic-regression "
    "family leans on. Each decile figure plots the observed log-odds of the "
    "target against the column's value bands; a roughly straight line means "
    "the linearity assumption is comfortable, obvious bends mean a "
    "transformed or binned feature would fit better."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": column,
                "Value bands": len(entry.get("bins", [])),
                "Logit range": "{min} … {max}".format(
                    min=_fmt(min((b.get("logit", 0.0) for b in entry.get("bins", [])), default=0.0)),
                    max=_fmt(max((b.get("logit", 0.0) for b in entry.get("bins", [])), default=0.0)),
                ),
            }
            for column, entry in empirical_logit.items()
            if isinstance(entry, dict)
        ],
        columns=["Column", "Value bands", "Logit range"],
    ),
    width="stretch",
    hide_index=True,
)
for column in numeric:
    _figure(
        f"stage4_empirical_logit_{column}.png",
        f"empirical logit of the target across `{column}` bands",
    )

# ---------------------------------------------------------------------------
# Technical appendix: the verified manifest artifact index
# ---------------------------------------------------------------------------

with st.expander("Technical appendix — artifact index"):
    st.caption(
        "The full report lives in `reports/data-methods.md`; this page reads "
        f"the same generated artifacts. Regenerate with `{REGENERATION_COMMAND}`."
    )
    st.markdown("Manifest artifact index (all digests verified clean):")
    st.dataframe(
        pd.DataFrame(
            sorted((view.manifest or {}).get("artifacts", {}).items()),
            columns=["Artifact", "SHA-256"],
        ),
        width="stretch",
        hide_index=True,
    )
