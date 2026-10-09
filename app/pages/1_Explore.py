"""Explore page: dataset insight dashboard for the pinned heart dataset.

Four insight groups render here — (1) feature distributions, (2) the
correlation view, (3) the S01-audited class balance including the
impossible-zero table, and (4) the per-feature insight table. Every
number is produced by :mod:`app.lib.plots`, which reuses the exact S01
audit functions behind ``reports/data_quality.md`` so displayed figures
match the audited report rather than a locally recomputed approximation.

Data-load failures surface as the friendly named error from the
:class:`~app.lib.plots.ExploreData` contract — never a stack trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
import time

import streamlit as st

from app.lib.plots import (
    categorical_target_rates,
    insight_groups,
    load_explore_data,
)
from heart.observability import (
    configure_logging,
    log_event,
    record_metric,
    timed_event,
)

logger = logging.getLogger("heart.app")
configure_logging()

st.set_page_config(
    page_title="1 — Explore",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Explore — dataset insight")
st.caption(
    "Figures are the audited S01 data-quality numbers; the dashboard never "
    "recomputes them with a different formula. Pinned dataset, "
    "checksum-verified."
)

with timed_event(logger, "app.dataset_load", "heart.dashboard.dataset_load_ms") as event_fields:
    data = load_explore_data()
    if data.ok:
        event_fields["sha256"] = data.sha256
        event_fields["rows"] = data.rows

if not data.ok:
    st.error(
        f"**Dataset could not be loaded ({data.error_kind}).** "
        f"{data.error_message}"
    )
    log_event(
        logger, "app.dataset_load", "error", level=logging.ERROR,
        message=f"dataset load failed ({data.error_kind})",
        error_kind=data.error_kind,
    )
    record_metric(logger, "heart.dashboard.dataset_load_failures", 1, kind="counter",
                  error_kind=str(data.error_kind))
    st.stop()

with timed_event(logger, "app.explore_render", "heart.dashboard.render_ms",
                 message="explore groups rendered"):
    groups = insight_groups(data.frame)
balance = groups["class_balance"]

# ---------------------------------------------------------------------------
# 1 — Distributions
# ---------------------------------------------------------------------------

st.header("Distributions")
st.caption("Value counts per feature, straight from the pinned dataset.")

distributions = groups["distributions"]
numeric_tabs = st.tabs(
    [dist.columns.to_list()[0] for dist in distributions["numeric"]]
)
for tab, dist in zip(numeric_tabs, distributions["numeric"]):
    with tab:
        column = dist.columns.to_list()[0]
        st.bar_chart(dist, horizontal=True)
        st.caption(
                f"{column}: {dist[column].sum():.0f} rows binned "
                f"over {len(dist)} intervals."
            )

categorical_tabs = st.tabs(
    [dist.columns.to_list()[0] for dist in distributions["categorical"]]
)
for tab, dist in zip(categorical_tabs, distributions["categorical"]):
    with tab:
        column = dist.columns.to_list()[0]
        st.bar_chart(dist, horizontal=True)
        st.caption(
            f"{column}: {dist[column].sum():.0f} rows over "
            f"{len(dist)} levels."
        )

# ---------------------------------------------------------------------------
# 2 — Correlation view
# ---------------------------------------------------------------------------

st.header("Correlation view")
st.caption(
    "Pearson correlation across continuous features and the target "
    "(HeartDisease). Binary flags appear in the categorical insight below "
    "instead of implying a spurious linear correlation."
)

correlation = groups["correlation"]
left, right = st.columns([2, 1])
with left:
    st.dataframe(correlation["matrix"], use_container_width=True)
with right:
    st.markdown("**Strongest feature↔target correlations**")
    st.dataframe(correlation["strongest"], use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 3 — Class balance (S01-audited)
# ---------------------------------------------------------------------------

st.header("Class balance — audited S01 figures")
st.caption(
    "From heart.data.quality — the same code path that generated "
    "reports/data_quality.md. Numbers here are pinned to that report by the "
    "test suite, not recomputed ad-hoc."
)

b1, b2, b3, b4 = st.columns(4)
b1.metric("No disease (0)", f"{balance['negative']}")
b2.metric("Disease (1)", f"{balance['positive']}")
b3.metric("Positive prevalence", f"{balance['prevalence']:.4f}")
b4.metric("Imbalance ratio", f"{balance['imbalance_ratio']:.4f}")

st.markdown(f"Total rows: **{data.rows}**")

st.subheader("Impossible zeros (S01 sentinel audit)")
st.caption(
    "A resting blood pressure or cholesterol of 0 is not a real "
    "measurement; the dataset uses it as a 'not recorded' sentinel."
)
st.dataframe(balance["zero_table"], use_container_width=True)
st.caption(
    f"Rows carrying at least one impossible zero: "
    f"**{balance['affected_zero_rows']}** — the cost of a naive row-drop."
)

st.divider()

# ---------------------------------------------------------------------------
# 4 — Per-feature insight
# ---------------------------------------------------------------------------

st.header("Per-feature insight")
st.caption(
    "Type, completeness, and how each feature splits the disease rate "
    "(numeric features: rates above/below the median; categorical "
    "features: highest vs. lowest per-level rate)."
)

st.dataframe(groups["per_feature"], use_container_width=True)

categorical_features = [
    feature
    for feature in data.frame.columns
    if categorical_target_rates(data.frame, feature) is not None
]
selected_feature = st.select_slider(
    "Per-level disease rates for categorical feature:",
    options=categorical_features,
    value=categorical_features[0],
)
rates_frame = categorical_target_rates(data.frame, selected_feature)
st.markdown(
    f"Rates for **{selected_feature}** — share of disease-present rows "
    "at each level, with the overall prevalence (55.34%) as baseline."
)
c1, c2 = st.columns([1, 2])
with c1:
    st.bar_chart(rates_frame, horizontal=True)
with c2:
    st.dataframe(rates_frame.T, use_container_width=True)

st.divider()
st.caption(
    "Research portfolio demonstration; not a medical device. Numbers are "
    "for educational use only."
)
