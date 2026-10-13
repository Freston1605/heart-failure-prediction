"""Predict page: form → validated input → probability from the real artifact.

The page is deliberately thin: artifact loading is the T01 loader contract
(:func:`app.lib.artifact_loader.load_winning_artifact`, never-raising with
a named friendly error), the widget-vs-frame conversion is the
:mod:`app.lib.input_validation` contract (:class:`ValidationError` naming
each rejected field), and the disclaimer wording is the
:mod:`app.lib.disclaimer` contract asserted by tests. A corrupt or missing
artifact therefore renders the same friendly error everywhere in the app,
never a stack-trace white screen. A successful submit also exposes the
calibration seam (M007): the raw pre-calibration RF probability is shown
next to the served calibrated one with the gap between them spelled out,
or — if raw access fails — a friendly named notice while the calibrated
probability is still served.

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

    raw_probability: float | None = None
    calibration_gap: float | None = None
    try:
        with timed_event(
            logger, "app.predict_submit", "heart.serving.predict_latency_ms",
            message="prediction served",
        ) as predict_fields:
            probability = float(artifact.predict_positive_proba(frame)[0])
            predict_fields["outcome"] = "served"
            # The raw (pre-calibration) pair is best-effort: it runs in its
            # own guard so a raw-access failure can never block the
            # calibrated probability already served above.
            try:
                raw_probability = float(
                    artifact.predict_raw_positive_proba(frame)[0]
                )
            except Exception as raw_exc:  # noqa: BLE001 - degradation path
                raw_error = to_app_error(raw_exc)
                log_event(
                    logger, "app.predict_submit", "degraded",
                    level=logging.WARNING,
                    message=f"raw probability unavailable ({raw_error.kind})",
                    outcome="raw-unavailable", error_kind=raw_error.kind,
                )
                st.warning(
                    f"**The raw pre-calibration probability is unavailable "
                    f"({raw_error.kind}).** The calibrated probability was "
                    "served normally below; only the pre-calibration view "
                    "is affected."
                )
                raw_probability = None
            else:
                calibration_gap = probability - raw_probability
                predict_fields["raw_probability"] = raw_probability
                predict_fields["calibration_gap"] = calibration_gap
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

    if raw_probability is not None and calibration_gap is not None:
        # The calibration seam is shown deliberately (M007): the raw
        # pre-calibration RF score beside the served calibrated number,
        # with the gap (calibrated - raw, either sign) spelled out. All
        # formatting stays inside this guard so a rendering failure can
        # never white-screen an already-served prediction.
        try:
            raw_left, raw_right = st.columns(2)
            raw_left.metric(
                "Raw RF probability (pre-calibration)",
                f"{raw_probability:.4f}",
            )
            raw_right.markdown(
                f"Raw RF **{raw_probability:.4f}** -> "
                f"{metadata.calibration_method}-calibrated "
                f"**{probability:.4f}** "
                f"(calibration gap {calibration_gap:+.4f})"
            )
            st.caption(
                "The served probability is the calibrated one; the raw "
                "pre-calibration RF score and the calibration gap are "
                "shown deliberately."
            )
        except Exception as raw_render_exc:  # noqa: BLE001 - degradation path
            render_error = to_app_error(raw_render_exc)
            st.warning(
                f"**The calibration-gap view is unavailable "
                f"({render_error.kind}).** The calibrated probability "
                "above was served normally."
            )
            log_event(
                logger, "app.predict_submit", "degraded",
                level=logging.WARNING,
                message=f"raw pair render failed ({render_error.kind})",
                outcome="raw-unavailable", error_kind=render_error.kind,
            )

    st.warning(DISCLAIMER_FULL)
    served_fields: dict[str, object] = {}
    if raw_probability is not None and calibration_gap is not None:
        served_fields["raw_probability"] = raw_probability
        served_fields["calibration_gap"] = calibration_gap
    log_event(
        logger, "app.predict_submit", "ok", level=logging.INFO,
        message="prediction served", outcome="served",
        **served_fields,
    )
    record_metric(logger, "heart.serving.predict_count", 1, kind="counter", outcome="served")
else:
    st.caption(
        "Fill the form and press Predict to produce a probability from the "
        "serialized artifact."
    )
