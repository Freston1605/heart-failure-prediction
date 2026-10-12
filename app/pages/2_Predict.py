"""Predict page: form → validated input → probability from the real artifact.

The page is deliberately thin: artifact loading is the T01 loader contract
(:func:`app.lib.artifact_loader.load_winning_artifact`, never-raising with
a named friendly error), the widget-vs-frame conversion is the
:mod:`app.lib.input_validation` contract (:class:`ValidationError` naming
each rejected field), and the disclaimer wording is the
:mod:`app.lib.disclaimer` contract asserted by tests. A corrupt or missing
artifact therefore renders the same friendly error everywhere in the app,
never a stack-trace white screen.

Predictions are gated behind an explicit acknowledgement checkbox, and the
full disclaimer is re-rendered next to every returned probability.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

import logging

from app.lib.artifact_loader import load_winning_artifact
from heart.observability import (
    configure_logging,
    log_event,
    record_metric,
    timed_event,
)

logger = logging.getLogger("heart.app")
configure_logging()
from app.lib.disclaimer import (
    DISCLAIMER_FULL,
    render_disclaimer,
    require_disclaimer_acknowledgement,
)
from app.lib.errors import to_app_error
from app.lib.input_validation import (
    CATEGORICAL_DISPLAY_LEVELS,
    NUMERIC_DEFAULTS,
    NUMERIC_RANGES,
    ValidationError,
    build_feature_frame,
)
from heart.data.schema import SPECS_BY_NAME

st.set_page_config(
    page_title="2 — Predict",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Predict — risk probability")
st.caption(
    "The form's values are validated field-by-field, then fed to the exact "
    "serialized winning model — the same artifact the Explore banner "
    "describes, no re-fitting and no look-alike copy."
)

with timed_event(logger, "app.artifact_load", "heart.serving.artifact_load_ms") as event_fields:
    result = load_winning_artifact()
    if result.ok:
        event_fields["path"] = str(result.path)

if not result.ok:
    error = result.error
    st.error(
        f"**Prediction is unavailable ({error.kind}).** {error.user_message}"
    )
    st.caption(f"Attempted artifact path: `{result.path}`.")
    log_event(
        logger, "app.artifact_load", "error", level=logging.ERROR,
        message=f"artifact load failed ({error.kind})",
        error_kind=error.kind, path=str(result.path),
    )
    st.stop()

artifact = result.artifact
metadata = artifact.metadata

if not metadata.ship:
    st.warning(
        "**Selection verdict: NO-SHIP.** The winner-selection flow recorded "
        "flags for this artifact, so treat every number below as a "
        "technical demonstration rather than an endorsed operating point."
        + (
            " Flags: " + "; ".join(metadata.flags) if metadata.flags else ""
        )
    )

st.header("Audit trail")
st.caption(
    f"Serving model: {metadata.model_name}; decision threshold "
    f"{metadata.threshold:g} ({metadata.threshold_objective}); calibration "
    f"{metadata.calibration_method} "
    f"({'applied' if metadata.calibration_applied else 'not applied'}). "
    "Predictions are the model's calibrated probability, plus the selected "
    "threshold's flag."
)

# ---------------------------------------------------------------------------
# Disclaimer (prominent, above the form)
# ---------------------------------------------------------------------------

render_disclaimer(short=True)

# ---------------------------------------------------------------------------
# Form widgets: one per declared feature, in the artifact's column order
# ---------------------------------------------------------------------------

with st.form("prediction-form"):
    st.subheader("Patient information")

    numeric_state: dict[str, float] = {}
    categorical_state: dict[str, str] = {}

    for column in metadata.feature_columns:
        spec = SPECS_BY_NAME[column]
        if spec.is_categorical:
            levels = CATEGORICAL_DISPLAY_LEVELS[column]
            value = st.selectbox(
                column,
                options=list(levels),
                index=0,
                help=spec.note or f"Allowed levels: {', '.join(levels)}.",
                key=f"in-{column}",
            )
            categorical_state[column] = value
        else:
            minimum, maximum = NUMERIC_RANGES[column]
            default = NUMERIC_DEFAULTS[column]
            if spec.kind == "integer":
                value = st.number_input(
                    column,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(default),
                    step=1,
                    help=spec.note or None,
                    key=f"in-{column}",
                )
            else:
                value = st.number_input(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=round(default, 1),
                    step=0.1,
                    format="%g",
                    help=spec.note or None,
                    key=f"in-{column}",
                )
            numeric_state[column] = value

    submitted = st.form_submit_button("Predict", type="primary")

# The acknowledgement deliberately sits OUTSIDE the form: a checkbox
# inside a form only commits on submit, so it could not enable the button
# it gates. It commits immediately, and its state is checked at submit
# time; without it the page refuses to compute anything.
acknowledged = require_disclaimer_acknowledgement(key="disclaimer_ack")

if submitted and not acknowledged:
    st.warning(
        "**Please tick the disclaimer acknowledgement above before a "
        "prediction is computed.** Nothing was predicted from this input."
    )
    log_event(
        logger, "app.predict_submit", "skipped", level=logging.INFO,
        message="submit ignored: disclaimer not acknowledged", outcome="not-acknowledged",
    )
    record_metric(logger, "heart.serving.predict_count", 1, kind="counter", outcome="not-acknowledged")
    st.stop()

# ---------------------------------------------------------------------------
# Prediction (only when the acknowledgement gate is closed and submit was clicked)
# ---------------------------------------------------------------------------

if submitted:
    submitted_values: dict[str, object] = {}
    for column in metadata.feature_columns:
        spec = SPECS_BY_NAME[column]
        if spec.is_categorical:
            submitted_values[column] = categorical_state[column]
        else:
            submitted_values[column] = numeric_state[column]

    try:
        frame = build_feature_frame(submitted_values)
    except ValidationError as exc:
        for column, reason in exc.issues.items():
            st.error(f"**{column}**: {reason}")
        st.caption(
            "Fix the marked fields and resubmit; nothing was predicted from "
            "this input."
        )
        log_event(
            logger, "app.predict_submit", "skipped", level=logging.INFO,
            message="submit refused by field validation",
            outcome="validation-blocked",
            rejected_field_count=len(exc.issues),
        )
        record_metric(logger, "heart.serving.predict_count", 1, kind="counter", outcome="validation-blocked")
        st.stop()

    try:
        with timed_event(
            logger, "app.predict_submit", "heart.serving.predict_latency_ms",
            message="prediction served",
        ) as predict_fields:
            probability = float(artifact.predict_positive_proba(frame)[0])
            predict_fields["outcome"] = "served"
    except Exception as exc:  # noqa: BLE001 - serving swallows, maps, explains
        mapped = to_app_error(exc)
        st.error(
            f"**Prediction failed ({mapped.kind}).** {mapped.user_message}"
        )
        log_event(
            logger, "app.predict_submit", "error", level=logging.ERROR,
            message=f"prediction failed ({mapped.kind})",
            outcome="error", error_kind=mapped.kind,
        )
        record_metric(logger, "heart.serving.predict_count", 1, kind="counter", outcome="error")
        st.stop()

    threshold = metadata.threshold
    flag = "at or above" if probability >= threshold else "below"
    left, right = st.columns(2)
    left.metric(
        "Predicted probability of heart disease",
        f"{probability:.4f}",
    )
    right.metric(
        f"Selected threshold ({metadata.threshold_objective})",
        f"{threshold:g}",
        delta=None,
    )
    st.markdown(
        f"The probability is **{flag}** the tuned threshold "
        f"({metadata.threshold:g}), which the selection flow chose by "
        f"'{metadata.threshold_objective}'."
    )

    st.warning(DISCLAIMER_FULL)
    log_event(
        logger, "app.predict_submit", "ok", level=logging.INFO,
        message="prediction served", outcome="served",
    )
    record_metric(logger, "heart.serving.predict_count", 1, kind="counter", outcome="served")
else:
    st.caption(
        "Fill the form and press Predict to produce a probability from the "
        "serialized artifact."
    )
