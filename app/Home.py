"""Home page: guided landing story + serving-boundary status banner.

The landing story is static markdown quoted from the committed reports —
what the model is, how it was validated, and the honesty notes — so the
page never reads report files or artifact internals at render time. The
home page never imports artifact internals; it consumes the Layer 3
contract only (the status banner below). Clean start

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
    """Welcome. This page is the guided tour: what the served model is, how
it was validated, and — just as important — what the numbers do *not*
prove. The links at the end open each flow."""
)

st.header("What the model is")
st.markdown(
    """\
- **Model**: a **calibrated random forest** (`random-forest`) — the winner
  of the model-selection battery.
- **Decision threshold**: tuned to **0.33** on validation rows only
  (objective `f1`, 0.860335 at selection time).
- **Calibration**: isotonic calibration is applied, so the served
  probability is the calibrated one, not the raw forest score."""
)

st.header("How it was validated")
st.markdown(
    """\
- **Repeated cross-validation**: the winner's repeated-CV ROC-AUC is
  0.930335 ± 0.018704 (alpha 0.05).
- **Held-out evaluation**: a 184-row test split never touched during
  training or threshold tuning, re-scored through the serving path.
- **Severity/treatment gate**: the pre-declared Part F verdict table
  rejects **all 48 measured pairs**, so the incumbent pipeline stands —
  every candidate treatment had to clear three declared conditions against
  the incumbent, and none did."""
)

st.header("Honesty notes")
st.markdown(
    """\
- **Not significantly better than the baseline**: against the tuned
  logistic-regression-l2 baseline the corrected resampled t-test reads
  p = 0.221 at alpha 0.05 — not significant.
- **Calibration did not improve Brier score**: delta -0.005274.
- **The raw/calibrated gap is spelled out**: Raw RF 0.1200 ->
  isotonic-calibrated 0.1111 (calibration gap -0.0089).

These flags ship with the artifact itself; the leaderboard and the
explainability report carry the full derivations."""
)

st.header("Where to look next")
st.markdown(
    """\
- **[Try a prediction](/Predict)** — the served model behind the
  disclaimer gate.
- **[Data & Methods](/Data_Methods)** — the full transparency report:
  data stages, treatment audit, and the severity-gate verdicts.
- **[Runs](/Runs)** — per-run provenance, parameters and metrics.
- **[Explore](/Explore)** — distributions,
  correlations and per-feature insight."""
)

with st.sidebar:
    st.header("System status")
    st.markdown(
        "Pages:\n\n- [1 — Explore](/Explore)\n"
        "- [2 — Predict](/Predict)\n"
        "- [3 — Runs](/Runs)\n"
        "- [4 — Data & Methods](/Data_Methods)"
    )

st.header("Serving boundary")
show_artifact_status()

st.divider()
st.caption(
    "Research portfolio demonstration; not a medical device. Predictions are "
    "for educational use only."
)
