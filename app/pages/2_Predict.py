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
probability is still served. A submitted prediction also renders a
per-patient SHAP attribution (M007/S02) explaining the RAW pre-calibration
RF score through :mod:`app.lib.explain`, under the same never-raise rule:
a SHAP failure degrades to a kind-named notice while the served
probability stays.

The page also renders a global cohort-drivers section (M007/S03, the
bottom-guarded block): mean |SHAP| per feature over a deterministic
sample of the pinned 918-row dataset, computed once per session behind
``st.cache_resource`` and explained in the raw (pre-calibration) domain
(D011/MEM118 seam stays honest). It sits deliberately outside the
per-submission flow — it renders pre-submit too — and its failure
degrades exactly like S02's: a kind-named warning, prediction intact.

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

import pandas as pd
import streamlit as st

import logging

from app.lib.artifact_loader import load_winning_artifact
from app.lib.explain import cohort_drivers, explain_patient
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

    # ------------------------------------------------------------------
    # Per-patient SHAP attribution (M007/S02) — its OWN guarded section,
    # deliberately outside the prediction try: the probability above is
    # already served, so an attribution failure can only degrade the
    # explanation (MEM125) — never re-raise, never st.stop. The page also
    # guards the call itself, so even a contract breach escaping
    # explain_patient lands in the same kind-named degradation path.
    # ------------------------------------------------------------------
    explain_outcome = None
    explain_error_kind: str | None = None
    explain_error_message: str | None = None
    try:
        with timed_event(
            logger,
            "app.explain",
            "heart.serving.explain_latency_ms",
            message="explanation computed",
        ) as explain_fields:
            explain_outcome = explain_patient(artifact, frame)
            if explain_outcome.attribution is not None:
                explain_fields["n_features"] = explain_outcome.attribution.n_features
                explain_fields["top_feature"] = explain_outcome.attribution.top_feature
            elif explain_outcome.error is not None:
                explain_fields["outcome"] = "explanation-unavailable"
                explain_fields["error_kind"] = explain_outcome.error.kind
    except Exception:  # noqa: BLE001 - call-site guard, never white-screen
        explain_error_kind = "explanation-failed"
        explain_error_message = (
            "Something went wrong while computing this patient's "
            "explanation, so it was skipped rather than shown incorrectly. "
            "The prediction above was served normally; only the "
            "explanation is affected. Re-running the prediction may "
            "succeed."
        )
    if explain_outcome is not None and explain_outcome.error is not None:
        explain_error_kind = explain_outcome.error.kind
        explain_error_message = explain_outcome.error.user_message

    attribution = explain_outcome.attribution if explain_outcome is not None else None
    if explain_error_kind is None and attribution is not None:
        try:
            # Render inside its own guard too: a chart-formatting failure
            # degrades the same way instead of white-screening the page.
            names = list(attribution.display_names)
            values = [float(v) for v in attribution.values]
            base_value = float(attribution.base_value)
            top8 = sorted(range(len(values)), key=lambda i: -abs(values[i]))[:8]
            top8.sort(key=lambda i: values[i], reverse=True)
            chart = pd.Series(
                [values[i] for i in top8],
                index=[names[i] for i in top8],
                name="SHAP value",
            ).rename_axis("feature")
            up_top = sorted(
                (i for i in range(len(values)) if values[i] > 0),
                key=lambda i: -values[i],
            )[:3]
            down_top = sorted(
                (i for i in range(len(values)) if values[i] < 0),
                key=lambda i: values[i],
            )[:3]
            reproduced_raw = base_value + sum(values)

            st.subheader("Why this score?")
            st.caption(
                "SHAP explains the **raw pre-calibration RF score** "
                f"(raw {reproduced_raw:.4f}, base {base_value:.4f}) — the "
                "calibrated probability served above is what counts."
            )
            st.bar_chart(chart)
            rows = [
                "| Direction | Feature | SHAP value |",
                "| --- | --- | ---: |",
            ]
            rows += [
                f"| pushed up | {names[i]} | {values[i]:.4f} |" for i in up_top
            ]
            rows += [
                f"| pushed down | {names[i]} | {values[i]:.4f} |"
                for i in down_top
            ]
            st.markdown("\n".join(rows))
            st.caption(
                f"base {base_value:.4f} + sum of SHAP values "
                f"{sum(values):+.4f} = {reproduced_raw:.4f} — the raw "
                "score, reproduced exactly (the attribution is additive)."
            )
            log_event(
                logger, "app.explain", "ok", level=logging.INFO,
                message="explanation rendered", outcome="served",
                top_feature=attribution.top_feature,
                n_features=attribution.n_features,
            )
        except Exception:  # noqa: BLE001 - render degradation path
            explain_error_kind = "explanation-failed"
            explain_error_message = (
                "The explanation was computed but could not be drawn "
                "correctly, so it was skipped rather than shown "
                "incorrectly. The prediction above was served normally; "
                "only the explanation is affected. Re-running the "
                "prediction may succeed."
            )

    if explain_error_kind is not None:
        st.warning(
            f"**The explanation is unavailable ({explain_error_kind}).** "
            f"{explain_error_message}"
        )
        log_event(
            logger, "app.explain", "degraded", level=logging.WARNING,
            message=f"explanation degraded ({explain_error_kind})",
            outcome="explanation-unavailable",
            error_kind=explain_error_kind,
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


# ---------------------------------------------------------------------------
# Global cohort drivers (M007/S03) — its OWN guarded section, deliberately
# at the BOTTOM of the page and OUTSIDE the per-submission flow: the cohort
# narrative is global context ("what drives scores across patients"),
# computed once per session behind st.cache_resource, so it renders
# pre-submit too (the page already st.stop()ed above if the artifact
# failed to load). A cohort failure can only degrade this section
# (MEM125 discipline) — never re-raise, never st.stop, never touch
# heart.serving.predict_count.
# ---------------------------------------------------------------------------

cohort_outcome = None
cohort_error_kind: str | None = None
cohort_error_message: str | None = None
try:
    # timed_event: the first session render carries the real SHAP compute
    # cost; later renders are st.cache_resource hits (that skew is by
    # design and documented as a monitoring gap in the slice plan).
    with timed_event(
        logger,
        "app.cohort",
        "heart.serving.cohort_explain_latency_ms",
        message="cohort drivers computed",
    ) as cohort_fields:
        cohort_outcome = cohort_drivers()
        if cohort_outcome.drivers is not None:
            cohort_fields["n_rows"] = cohort_outcome.drivers.n_rows
            cohort_fields["top_feature"] = cohort_outcome.drivers.top_feature
        elif cohort_outcome.error is not None:
            cohort_fields["outcome"] = "cohort-unavailable"
            cohort_fields["error_kind"] = cohort_outcome.error.kind
except Exception:  # noqa: BLE001 - call-site guard, never white-screen
    cohort_error_kind = "cohort-failed"
    cohort_error_message = (
        "Something went wrong while computing the global cohort drivers, "
        "so the section was skipped rather than shown incorrectly. "
        "Re-running the page may succeed."
    )
if cohort_outcome is not None and cohort_outcome.error is not None:
    cohort_error_kind = cohort_outcome.error.kind
    cohort_error_message = cohort_outcome.error.user_message

cohort_data = cohort_outcome.drivers if cohort_outcome is not None else None
if cohort_error_kind is None and cohort_data is not None:
    try:
        # Render inside its own guard too, mirroring S02's attribution
        # render guard: a formatting failure degrades the section, not
        # the page. Charts stay native (matplotlib is NOT deployed).
        cohort_names = list(cohort_data.display_names)
        cohort_mean_abs = [float(value) for value in cohort_data.mean_abs]
        cohort_mean_signed = [float(value) for value in cohort_data.mean_signed]
        cohort_top8 = list(range(min(8, len(cohort_names))))
        cohort_chart = pd.Series(
            [cohort_mean_abs[i] for i in cohort_top8],
            index=[cohort_names[i] for i in cohort_top8],
            name="Mean abs SHAP",
        ).rename_axis("feature")

        st.subheader("What drives scores across patients?")
        st.caption(
            f"Mean |SHAP| across {cohort_data.n_rows} deterministically "
            "sampled rows of the pinned 918-row dataset, computed once "
            "per session from the live serving artifact. It explains the "
            "**raw pre-calibration RF score** — the forest's own vote "
            "before the isotonic calibration mapping — so the "
            "D011/MEM118 seam stays honest; the calibrated probability "
            "served above is what counts."
        )
        st.bar_chart(cohort_chart)
        cohort_rows = [
            "| Feature | Mean abs SHAP | Mean signed SHAP |",
            "| --- | ---: | ---: |",
        ]
        for index in range(min(5, len(cohort_names))):
            signed = cohort_mean_signed[index]
            if signed > 0:
                direction = "pushed up"
            elif signed < 0:
                direction = "pushed down"
            else:
                direction = "neutral"
            cohort_rows.append(
                f"| {cohort_names[index]} | {cohort_mean_abs[index]:.4f} | "
                f"{signed:+.4f} ({direction}) |"
            )
        st.markdown("\n".join(cohort_rows))
        st.caption(
            "Bars (top 8) and table (top 5) rank features by mean |SHAP| "
            "— absolute impact across the sampled cohort; the signed "
            "column shows the average direction of that push."
        )
        log_event(
            logger,
            "app.cohort",
            "ok",
            level=logging.INFO,
            message="cohort drivers rendered",
            outcome="served",
            top_feature=cohort_data.top_feature,
            n_rows=cohort_data.n_rows,
        )
    except Exception:  # noqa: BLE001 - render degradation path
        cohort_error_kind = "cohort-failed"
        cohort_error_message = (
            "The cohort drivers were computed but could not be drawn "
            "correctly, so the section was skipped rather than shown "
            "incorrectly. Re-running the page may succeed."
        )

if cohort_error_kind is not None:
    st.warning(
        f"**Cohort drivers are unavailable ({cohort_error_kind}).** "
        f"{cohort_error_message} The prediction above, if submitted, was "
        "served normally; only the global drivers section is affected."
    )
    log_event(
        logger,
        "app.cohort",
        "degraded",
        level=logging.WARNING,
        message=f"cohort drivers degraded ({cohort_error_kind})",
        outcome="cohort-unavailable",
        error_kind=cohort_error_kind,
    )
