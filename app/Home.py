"""Home page: multipage scaffold + serving-boundary status banner.

The home page never imports artifact internals; it consumes the Layer 3
contract only. Clean start

    streamlit run app/Home.py          # from the project root

Run from another directory? The page prepends the project root to
``sys.path`` so ``heart`` and ``app.*`` resolve regardless.
"""

from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging

import streamlit as st

from app.lib.artifact_loader import load_winning_artifact
from app.lib.errors import AppError
from heart.observability import (
    configure_logging,
    log_event,
    timed_event,
)

logger = logging.getLogger("heart.app")
configure_logging()

st.set_page_config(
    page_title="Heart Failure Prediction Portfolio",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def show_artifact_status() -> None:
    """Render the serve/don't-serve banner from the validated artifact.

    A failed load renders the named, friendly error and *no* confusing
    model claims — the rest of the app can rely on this banner as the gate.
    """
    with timed_event(logger, "app.artifact_load", "heart.serving.artifact_load_ms") as event_fields:
        result = load_winning_artifact()
        _show_banner(result)
        if result.ok:
            event_fields["path"] = str(result.path)


def _show_banner(result) -> None:
    """Render the success / friendly-failure banner (extracted for logging)."""
    if result.ok and result.artifact is not None:
        metadata = result.artifact.metadata
        st.success("Model artifact loaded and validated. The app can serve predictions.")
        left, middle, right = st.columns(3)
        left.metric("Serving model", str(metadata.model_name))
        middle.metric("Decision threshold", f"{metadata.threshold:g}")
        middle.caption(f"Objective: {metadata.threshold_objective}")
        right.metric("Calibration", metadata.calibration_method)
        right.caption("applied" if metadata.calibration_applied else "not applied")
        with st.expander("Artifact details", expanded=False):
            st.caption(f"Path: `{result.path}` (format v{metadata.format_version})")
            st.write("Embedded feature order:")
            st.write(", ".join(str(col) for col in metadata.feature_columns))
            if metadata.flags:
                st.markdown("**Flagged at selection time:**")
                for flag in metadata.flags:
                    st.markdown(f"- {flag}")
            else:
                st.markdown("_No ship flags raised._")
        return

    error: AppError = result.error or AppError(
        "artifact-unknown", "The model could not be loaded."
    )
    st.error(f"**Model cannot be loaded ({error.kind}).** {error.user_message}")
    st.caption(
        "Tip: after re-training, use the sidebar or reload the page to retry. "
        f"(Attempted path: `{result.path}`.)"
    )
    log_event(
        logger, "app.artifact_load", "error", level=logging.ERROR,
        message=f"artifact load failed ({error.kind})",
        error_kind=error.kind, path=str(result.path),
    )


st.title("Heart Failure Prediction Portfolio")
st.markdown(
    """Welcome. This app has two flows:

- **1 — Explore**: distributions, correlations, class balance and
  per-feature insight for the UCI heart-failure dataset, built from the
  audited data-quality outputs.
- **2 — Predict**: a form that returns a model probability from the exact
  serialized winning model (calibrated Random Forest, tuned threshold).

Every prediction page carries the disclaimer: this is a *portfolio
demonstration*, **not** a medical device, and no output here is medical
advice.
"""
)

with st.sidebar:
    st.header("System status")
    st.markdown(
        "Pages:\n\n- [1 — Explore](/1_Exploratory_Data_Analysis)\n"
        "- [2 — Predict](/2_Predict)"
    )

st.header("Serving boundary")
show_artifact_status()

st.divider()
st.caption(
    "Research portfolio demonstration; not a medical device. Predictions are "
    "for educational use only."
)
