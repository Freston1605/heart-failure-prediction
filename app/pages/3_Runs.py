"""Runs page: the ranked final-run table plus a per-run drilldown, read-only.

The page is deliberately thin: the data is the T01 contract module
(:func:`app.lib.runs_view.load_runs_view`), which reuses
:func:`heart.reporting.leaderboard.select_leaderboard_rows` (D019) — the same
selection function that renders ``reports/leaderboard.md`` — so every number
here is identical to the report by construction, not by coincidence. The
per-run drilldown reads back through :func:`app.lib.runs_view.run_details`.

Never a stack trace: the contract module never raises, and anything the page
does add (the drilldown loop) is wrapped so a failed read-back renders its
named friendly message inside the expander instead of white-screening the
page. An empty or missing store renders the named ``store-empty`` /
``store-missing`` error with the ``make reproduce`` hint — the same state the
``scripts/mlflow_ui.py`` launcher reports on exit 2 (D020: hint-only launch
UX; the caption below points at ``make mlflow-ui`` but never spawns a server).

This page is deliberately *uncached*: one cheap SQLite read per rerun, so a
test-time ``RUNS_TRACKING_DIR`` env override (the seam in the contract
module) can never serve stale data from a previous store.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from app.lib import runs_view
from heart.eval.contract import CALIBRATION_KEY


def _mlflow_ui_port() -> int:
    """The advertised MLflow UI port; malformed overrides fall back to 5000."""
    try:
        return int(os.environ.get("MLFLOW_UI_PORT", "5000"))
    except ValueError:
        return 5000


st.set_page_config(
    page_title="3 — Runs",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 Runs — recorded training results")
st.caption(
    "The same recorded training runs behind reports/leaderboard.md — the "
    "identical selection function, so the numbers match by construction. "
    "Prefer MLflow's own UI for the full run timeline? Run `make mlflow-ui` "
    f"(port: $MLFLOW_UI_PORT, default 5000) and open http://localhost:"
    f"{_mlflow_ui_port()}."
)

view = runs_view.load_runs_view()

if not view.ok:
    st.error(
        f"**No recorded runs to show ({view.error_kind}).** {view.error_message}"
    )
    st.caption(f"Attempted tracking store: `{view.tracking_uri}`.")
    st.stop()

# ---------------------------------------------------------------------------
# Provenance (mirrors the report's Provenance section)
# ---------------------------------------------------------------------------

st.markdown(
    "Experiment `heart-failure-prediction` · run kind `final` · "
    "**{}** model(s) ranked from {} candidate run(s) — latest run per "
    "model.".format(len(view.rows), view.n_runs_considered)
)

# ---------------------------------------------------------------------------
# Ranked final-runs table (columns mirror reports/leaderboard.md)
# ---------------------------------------------------------------------------

_ECE_KEY = f"{CALIBRATION_KEY}.expected_calibration_error"


def _fmt_metric(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _ranked_table(v: runs_view.RunsView) -> pd.DataFrame:
    records: list[dict[str, str | int]] = []
    for rank, row in enumerate(v.rows, start=1):
        # Optional-suite columns render blank ("") when a run did not
        # record them; the required suite is always present by D019.
        records.append(
            {
                "Rank": rank,
                "Model": row.model_name,
                "Type": row.model_type,
                "Family": row.family,
                "Device": row.device or "",
                "ROC-AUC": _fmt_metric(row.metric("roc_auc")),
                "Accuracy": _fmt_metric(row.metric("accuracy")),
                "Precision": _fmt_metric(row.metric("precision")),
                "Recall": _fmt_metric(row.metric("recall")),
                "F1": _fmt_metric(row.metric("f1")),
                "PR-AUC": _fmt_metric(row.metric("pr_auc")),
                "Brier": _fmt_metric(row.metric("brier_score")),
                "ECE": _fmt_metric(
                    row.metric(_ECE_KEY)
                ),
                "N": row.n_samples if row.n_samples is not None else "",
            }
        )
    return pd.DataFrame.from_records(records)


st.dataframe(_ranked_table(view), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Per-run drilldown (one expander per ranked row)
# ---------------------------------------------------------------------------

# The useful tag subset for humans; s6.* selection annotations are shown
# plainly whenever present (the selection flow's statistical evidence).
_TAG_SHOWLIST: tuple[str, ...] = (
    "model_name",
    "model_slug",
    "model_type",
    "run_kind",
    "split_version",
    "primary_metric",
    "n_samples",
    "device",
    "source",
)


def _params_table(params: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(sorted(params.items()), columns=["Param", "Value"])


def _tags_table(tags: dict[str, str]) -> pd.DataFrame:
    shown = {key: tags[key] for key in _TAG_SHOWLIST if key in tags}
    shown.update(
        {key: tags[key] for key in sorted(tags) if key.startswith("s6.")}
    )
    return pd.DataFrame(sorted(shown.items()), columns=["Tag", "Value"])


def _metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(sorted(metrics.items()), columns=["Metric", "Value"])


def _start_label(start_time_ms: int | None) -> str:
    if not start_time_ms:
        return "start time not recorded"
    return datetime.fromtimestamp(
        start_time_ms / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


st.subheader("Per-run details")
st.caption(
    "Expand a row to audit that run exactly as recorded: id, hyperparameters, "
    "tags, every flattened metric, and the logged artifacts."
)

for rank, row in enumerate(view.rows, start=1):
    expander_key = f"run-{(row.run_id[:8] or str(rank)).lower()}"
    with st.expander(f"{rank}. {row.model_name} ({row.model_type})",
                     key=expander_key):
        details = runs_view.run_details(row.run_id)
        if not details.ok:
            st.error(
                f"**Run details unavailable ({details.error_kind}).** "
                f"{details.error_message}"
            )
            continue

        st.code(row.run_id, language=None)
        st.info(
            f"Run **{details.run_name}** · split `{row.split_version}` · "
            f"started {_start_label(row.start_time_ms)}"
        )

        st.markdown("Hyperparameters")
        st.dataframe(_params_table(details.params), hide_index=True)

        st.markdown("Tags")
        st.dataframe(_tags_table(details.tags), hide_index=True)

        st.markdown("Metrics (flattened)")
        st.dataframe(_metrics_table(details.metrics), hide_index=True)

        artifact_line = ", ".join(f"`{name}`" for name in details.artifact_names)
        st.markdown(f"Artifacts: {artifact_line or '_none recorded_'}")
