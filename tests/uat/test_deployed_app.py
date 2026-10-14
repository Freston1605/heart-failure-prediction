"""Tests/uat: browser-backed UAT of the deployed app's three flows.

The deployed app is live at the streamlit.app URL below (recorded in
``reports/deployment.md``). These tests make the checks reproducible from a
clean environment:

- *Reachability tests* run pure HTTP against the deployed URL. The Streamlit
  Community Cloud edge gates raw first-hit fetches with a 303 to
  ``share.streamlit.io/-/auth/app``; these tests assert that exact documented
  contract so an assertion never mistakes session gating for an outage.
  Once past the edge, the app itself is probed on its documented surfaces:
  ``{url}/healthz`` (JSON ``{"status": "ok"}``) and the direct page routes
  (``/``, ``/Explore``, ``/Predict``, ``/Data_Methods``) followed through the
  auth chain to a 200. The retired ``/~/+`` iframe prefix is deliberately not
  probed — it is not the documented health/route contract.
- *Contract tests* import the same modules the deployed app serves from
  (``app.lib.disclaimer``, ``app.lib.explain``, ``app.lib.data_methods_view``,
  ``app.Home``, the winning artifact) and pin the wording/values that were
  observed in the recorded browser session (screenshots in
  ``reports/uat-evidence/`` and the per-check table in ``reports/uat.md``).
  Two surfaces get a serving-source pin built on one rationale — Streamlit
  Cloud deploys from a git clone of this repository, so the deployed host
  renders exactly the committed bytes, and an in-process render pins what a
  deployed route serves without a second network path: the Data & Methods
  surface re-verifies the committed ``reports/data-methods`` bundle through
  the same provenance engine the deployed page uses (SHA-256 re-verification
  pins that view), and the Home landing surface renders the committed
  ``app/Home.py`` through ``streamlit.testing.v1.AppTest`` (artifact-status
  banner gate, guided-story honesty annotations, canonical ``/Predict`` and
  ``/Data_Methods`` click-path links) to pin what the deployed ``/`` route
  serves.

The interactive, JavaScript-rendered part of all three flows (dashboard
navigation, prediction submission, the disclaimer acknowledgement gate, and
the explainability surface: raw/calibrated pair, per-patient SHAP
attribution, cohort drivers) was exercised in a real browser session; the
objective evidence for each step is committed under ``reports/uat-evidence/``
and summarised in ``reports/uat.md``. The pytest layer keeps the
machine-checkable half of that evidence reproducible: same URL, same
wording pin, same expected probability from the same serialized artifact,
and the same raw/calibrated pair and attribution signs from the same
artifact.

Every network call uses a short timeout so a dead host fails fast with an
explicit message instead of hanging the suite.

Transient wake-ups: Streamlit Community Cloud's free tier sleeps idle apps,
and the edge occasionally rejects the first request after a sleep with a
placeholder 4xx while the container boots. The served-app facts (200 +
``ok`` health, 200 flow routes) are stable once awake, so network checks
retry short 4xx blips with backoff: a genuine outage still fails every
attempt, but a sub-second cold-start blip no longer flakes the suite.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEPLOYED_URL = "https://heart-failure-prediction-krwy4etmtavrqd233mteym.streamlit.app"
_AUTH_HOST = "share.streamlit.io"
_TIMEOUT_S = 10
# Free-tier cold-start blips: retry a few times with backoff before giving up.
_ATTEMPTS = 5
_BACKOFF_S = 3.0

_T = TypeVar("_T")


def _retry_transient(operation: Callable[[], _T], *, transient: Callable[[object], bool]) -> _T:
    """Run *operation*, retrying only while ``transient(result)`` is true.

    A retry budget bounds a real outage (every attempt 4xx => the last
    result is returned as-is and the caller's assertion fails on it); a
    healthy app resolves on the first try with no delay.
    """
    result = operation()
    for _ in range(_ATTEMPTS - 1):
        if not transient(result):
            return result
        time.sleep(_BACKOFF_S)
        result = operation()
    return result


def _is_transient_response(response: object) -> bool:
    """A 4xx blip (e.g. the edge's 400 placeholder during wake-up)."""
    status = getattr(response, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 404


def _get_ok(url: str) -> requests.Response:
    return _retry_transient(
        lambda: requests.get(url, timeout=_TIMEOUT_S),
        transient=_is_transient_response,
    )


@pytest.fixture(scope="module")
def deployed_url() -> str:
    return _DEPLOYED_URL


# ---------------------------------------------------------------------------
# Flow 0: the app is served and reachable
# ---------------------------------------------------------------------------


def test_root_is_reachable_through_documented_auth_chain(deployed_url: str) -> None:
    """Raw first hit is the documented Cloud-edge auth redirect, then 200.

    Negative/edge semantics asserted explicitly: a *redirect to the auth
    host* is the expected free-tier edge behaviour — never raise-and-500,
    never a plain connection failure.
    """
    raw = requests.get(deployed_url, allow_redirects=False, timeout=_TIMEOUT_S)
    assert raw.status_code == 303, (
        f"expected the documented 303 edge gate, got {raw.status_code}"
    )
    location = raw.headers.get("Location", "")
    assert _AUTH_HOST in location and "/-/auth/app" in location, (
        f"303 must point at the Cloud auth gate, got {location!r}"
    )


def test_followed_root_resolves_to_a_200_page(deployed_url: str) -> None:
    """The redirect chain a browser follows ends at a served page."""
    followed = requests.get(deployed_url, allow_redirects=True, timeout=_TIMEOUT_S)
    assert followed.status_code == 200
    assert "Streamlit" in followed.headers.get("Server", "") or followed.ok


def test_app_health_endpoint_is_ok(deployed_url: str) -> None:
    """The platform ``/healthz`` answers ``ok`` for the running app.

    ``/healthz`` is the nginx-level health probe (direct 200 JSON, no auth
    chain); ``/_stcore/health`` is kept as a fallback because it resolves
    through the documented 303 auth chain to the served app shell. The
    legacy ``/~/+/_stcore/health`` iframe prefix was retired by the edge
    (persistent empty 400 as of 2026-10-12) and is no longer probed.
    """
    health = _get_ok(f"{deployed_url}/healthz")
    if health.status_code != 200:
        health = _get_ok(f"{deployed_url}/_stcore/health")
    assert health.status_code == 200
    assert "ok" in health.text.strip().lower()


def test_app_serves_all_documented_routes(deployed_url: str) -> None:
    """All documented routes are served by the app (no 404 burial).

    Routes are fetched directly (``/{route}``); each follows the
    documented 303 auth chain to the 200 app shell — including the
    multipage ``/Data_Methods`` route. The legacy ``/~/+`` iframe prefix
    was retired by the edge (persistent empty 400 as of 2026-10-12) and
    is no longer used. A 404 is never retried as transient.
    """
    for route in ("/", "/Explore", "/Predict", "/Data_Methods"):
        resp = _get_ok(f"{deployed_url}{route}")
        assert resp.status_code == 200, f"{route} returned {resp.status_code}"


# ---------------------------------------------------------------------------
# Facts the browser session observed, pinned against the serving source
# ---------------------------------------------------------------------------


def test_disclaimer_wording_matches_deployed_serving_module() -> None:
    """The disclaimer text seen rendered on /Predict is exactly the module's.

    The browser evidence (reports/uat-evidence/03-predict-disclaimer.jpg)
    shows this wording in a prominent warning banner on the deployed page;
    this test pins it to :mod:`app.lib.disclaimer` so wording cannot drift.
    """
    from app.lib.disclaimer import DISCLAIMER_FULL, DISCLAIMER_SHORT

    assert "not a medical device" in DISCLAIMER_SHORT
    assert DISCLAIMER_FULL.strip().startswith(
        "**This is not a medical device, and this prediction is not medical "
        "advice.**"
    )
    assert "Consult a qualified clinician." in DISCLAIMER_FULL


def test_home_serving_banner_facts_match_deployed_metadata() -> None:
    """Home's green banner metrics (browser evidence 01) match the artifact.

    Browser evidence observed: serving model random-forest, threshold 0.33,
    calibration isotonic (applied), plus the validated-load banner.
    """
    from app.lib.artifact_loader import load_winning_artifact

    result = load_winning_artifact()
    assert result.ok, f"artifact failed locally: {result.error!r}"
    metadata = result.artifact.metadata
    assert metadata.model_name == "random-forest"
    assert metadata.threshold == pytest.approx(0.33)
    assert metadata.calibration_method == "isotonic"
    assert metadata.calibration_applied is True


def test_default_form_prediction_matches_browser_observed_value() -> None:
    """The 0.1111 probability the browser session received is reproducible.

    Browser evidence (reports/uat-evidence/05-predict-success.jpg) shows
    "Predicted probability of heart disease 0.1111 / threshold 0.33 /
    below" after submitting the pre-filled default form with the
    acknowledgement ticked. The exact same defaults fed to the exact same
    serialized artifact locally must reproduce that number — proving the
    deployed app serves the real tracked artifact, not a stand-in.
    """
    from app.lib.artifact_loader import load_winning_artifact
    from app.lib.input_validation import (
        CATEGORICAL_DISPLAY_LEVELS,
        NUMERIC_DEFAULTS,
        build_feature_frame,
    )
    from heart.data.schema import SPECS_BY_NAME

    result = load_winning_artifact()
    assert result.ok
    values: dict[str, object] = {}
    for column in result.artifact.metadata.feature_columns:
        spec = SPECS_BY_NAME[column]
        if spec.is_categorical:
            values[column] = list(CATEGORICAL_DISPLAY_LEVELS[column])[0]
        else:
            values[column] = NUMERIC_DEFAULTS[column]
    frame = build_feature_frame(values)
    probability = float(result.artifact.predict_positive_proba(frame)[0])
    # browser session rendered the value formatted to 4 dp as 0.1111
    assert round(probability, 4) == pytest.approx(0.1111)


# ---------------------------------------------------------------------------
# Explainability surface (/Predict) — browser-observed facts pinned against
# the serving source (M007/S04). Screenshots 09/10/11 record the session;
# these nodes keep the observed numbers reproducible from the same artifact.
# ---------------------------------------------------------------------------


def _build_default_frame(artifact):
    """The pre-filled default /Predict row as the form serves it.

    Mirrors the value construction in
    ``test_default_form_prediction_matches_browser_observed_value``: every
    categorical takes its first display level, every numeric its default —
    no randomness, so all downstream pins are deterministic.
    """
    from app.lib.input_validation import (
        CATEGORICAL_DISPLAY_LEVELS,
        NUMERIC_DEFAULTS,
        build_feature_frame,
    )
    from heart.data.schema import SPECS_BY_NAME

    values: dict[str, object] = {}
    for column in artifact.metadata.feature_columns:
        spec = SPECS_BY_NAME[column]
        if spec.is_categorical:
            values[column] = list(CATEGORICAL_DISPLAY_LEVELS[column])[0]
        else:
            values[column] = NUMERIC_DEFAULTS[column]
    return build_feature_frame(values)


def test_default_form_raw_and_calibrated_pair_matches_browser_observed_value() -> None:
    """The raw/calibrated pair seen in screenshot 09 reproduces to 4 dp.

    Browser evidence (reports/uat-evidence/09-predict-raw-calibrated-pair.jpg)
    shows, for the default-form submission on the deployed /Predict page:
    calibrated metric ``0.1111`` at threshold ``0.33``, the raw
    ``Raw RF probability (pre-calibration)`` metric ``0.1200``, and the gap
    line ``Raw RF 0.1200 -> isotonic-calibrated 0.1111 (calibration gap
    -0.0089)``. The same default row through the same serialized artifact
    must reproduce raw 0.1200 / calibrated 0.1111 / gap -0.0089 exactly —
    proving the deployed app serves the seam deliberately (M007), not that
    it serves a stand-in.
    """
    from app.lib.artifact_loader import load_winning_artifact

    result = load_winning_artifact()
    assert result.ok, f"artifact failed locally: {result.error!r}"
    frame = _build_default_frame(result.artifact)
    raw = float(result.artifact.predict_raw_positive_proba(frame)[0])
    calibrated = float(result.artifact.predict_positive_proba(frame)[0])
    gap = calibrated - raw
    # screenshot 09 rendered all three to 4 dp
    assert round(raw, 4) == pytest.approx(0.1200)
    assert round(calibrated, 4) == pytest.approx(0.1111)
    assert round(gap, 4) == pytest.approx(-0.0089)


def test_attribution_top_features_match_browser_observed_values() -> None:
    """Attribution signs/names from screenshot 10 reproduce locally.

    Browser evidence (reports/uat-evidence/10-predict-shap-attribution.jpg)
    shows on the deployed /Predict page after the default submission:
    ``Why this score?`` with the raw-score caption (raw 0.1200), and the
    pushed-up/pushed-down table headed by ``pushed up ChestPainType_ASY
    0.0790`` and ``pushed down ST_Slope_Up -0.0985`` — ST_Slope_Up also the
    largest-|SHAP| feature overall. ``explain_patient`` attributes the
    single default row against the pinned artifact with no randomness, so
    the same call must yield the same top features and signs.
    """
    from app.lib.artifact_loader import load_winning_artifact
    from app.lib.explain import explain_patient

    result = load_winning_artifact()
    assert result.ok, f"artifact failed locally: {result.error!r}"
    frame = _build_default_frame(result.artifact)
    outcome = explain_patient(result.artifact, frame)
    assert outcome.error is None, (
        f"explanation degraded locally: {outcome.error.user_message!r}"
    )
    attribution = outcome.attribution
    assert attribution is not None
    names = list(attribution.display_names)
    values = [float(v) for v in attribution.values]
    assert len(names) == len(values) and names
    up = [(names[i], values[i]) for i in range(len(values)) if values[i] > 0]
    down = [(names[i], values[i]) for i in range(len(values)) if values[i] < 0]
    assert up and down, "attribution must push at least one feature each way"
    top_up_name, top_up_value = max(up, key=lambda item: item[1])
    top_down_name, top_down_value = min(down, key=lambda item: item[1])
    assert top_up_name == "ChestPainType_ASY"
    assert top_up_value > 0
    assert top_down_name == "ST_Slope_Up"
    assert top_down_value < 0
    # largest |SHAP| overall is the most-negative driver, per screenshot 10
    largest_name = max(range(len(values)), key=lambda i: abs(values[i]))
    assert names[largest_name] == "ST_Slope_Up"


# ---------------------------------------------------------------------------
# Data & Methods surface (/Data_Methods) — facts pinned against the serving
# source (M008/S01). Streamlit Cloud clones this repo, so re-verifying the
# committed bundle pins what the deployed host renders.
# ---------------------------------------------------------------------------


def test_data_methods_bundle_verifies_clean_for_deployed_host() -> None:
    """The committed D&M bundle re-verifies clean with every rendered surface.

    Rationale: Streamlit Cloud deploys from a git clone of this repository,
    so the deployed host renders exactly the committed bytes; SHA-256
    re-verification of the default committed ``reports/data-methods`` dir
    through :func:`app.lib.data_methods_view.load_data_methods_view` pins
    the deployed host's view without a second network path.
    """
    from app.lib.data_methods_view import load_data_methods_view

    view = load_data_methods_view()
    assert view.ok, f"D&M bundle failed to verify locally: {view.error!r}"
    assert view.verified_digests, "manifest re-verification pinned no digests"
    assert set(view.stages) == {
        "stage1_hygiene",
        "stage2_univariate",
        "stage3_bivariate",
        "stage4_assumptions",
    }
    assert isinstance(view.treatment_plan, dict)
    assert view.treatment_plan.get("imbalance_ladder"), (
        "Part D imbalance ladder must be non-empty"
    )
    assert view.treatment_audit is not None


def test_data_methods_gate_verdicts_pin_the_deployed_table() -> None:
    """Part F's table: 48 measured pairs, every verdict rejected (incumbent stands).

    This is the exact table the real-browser session confirms on the live
    ``/Data_Methods`` page in T02.
    """
    from app.lib.data_methods_view import load_data_methods_view

    view = load_data_methods_view()
    assert view.ok
    verdicts = view.gate_verdicts
    assert verdicts is not None, "gate verdicts missing from the committed bundle"
    assert verdicts["split_version"] == "v1"
    assert verdicts["measured_pair_count"] == 48
    outcomes = verdicts["outcomes"]
    assert len(outcomes) == 48
    assert all(outcome["verdict"] == "rejected" for outcome in outcomes)


def test_explainability_evidence_pack_present() -> None:
    """Screenshots 09/10/11 are committed, non-empty captures.

    The >10 KB floor guards against empty or error-page captures (a blank
    frame or a Streamlit traceback page compresses far below it); the three
    files are the slice's demo evidence cited by ``reports/uat.md`` Flow 3.
    """
    evidence_dir = _PROJECT_ROOT / "reports" / "uat-evidence"
    expected = (
        "09-predict-raw-calibrated-pair.jpg",
        "10-predict-shap-attribution.jpg",
        "11-predict-cohort-drivers.jpg",
    )
    for name in expected:
        path = evidence_dir / name
        assert path.is_file(), f"missing evidence file: {path}"
        size = path.stat().st_size
        assert size > 10 * 1024, (
            f"{path} is {size} bytes — too small to be a real capture"
        )


def test_data_methods_evidence_pack_present() -> None:
    """Screenshots 12/13 are committed, non-empty captures of /Data_Methods.

    The >10 KB floor guards against blank frames or error pages (an empty
    frame or a Streamlit traceback page compresses far below it); the two
    files are the slice's demo evidence cited by ``reports/uat.md`` Flow 4:
    12 shows the dataset-provenance banner with the Stage 1-2 tables and
    figures, 13 the Part E narrative plus the Part F 48-pair verdicts table
    and the technical appendix — both captured from the live deployed route
    in the T02 real-browser session.
    """
    evidence_dir = _PROJECT_ROOT / "reports" / "uat-evidence"
    expected = (
        "12-data-methods-stage-sections.jpg",
        "13-data-methods-gate-verdicts.jpg",
    )
    for name in expected:
        path = evidence_dir / name
        assert path.is_file(), f"missing evidence file: {path}"
        size = path.stat().st_size
        assert size > 10 * 1024, (
            f"{path} is {size} bytes — too small to be a real capture"
        )


# ---------------------------------------------------------------------------
# Home landing surface (/) — guided story pinned against the serving source
# (M008/S02). Streamlit Cloud clones this repo, so rendering the committed
# app/Home.py in-process pins what the deployed / route serves.
# ---------------------------------------------------------------------------


def _render_home():
    """Render ``app/Home.py`` with the artifact cache cleared before and after.

    Mirrors ``tests/test_home_landing.py``'s ``_render`` helper so the UAT
    pin and the local landing test exercise the identical render path.
    """
    import streamlit.testing.v1 as st_testing

    from app.lib.artifact_loader import clear_artifact_cache

    clear_artifact_cache()
    test = st_testing.AppTest.from_file(
        str(_PROJECT_ROOT / "app/Home.py"), default_timeout=300
    )
    try:
        test.run()
    finally:
        clear_artifact_cache()
    return test


def _rendered_markdown(test) -> str:
    """Every markdown body on the rendered page (main + sidebar), joined."""
    chunks = [str(m.value) for m in test.markdown]
    for sidebar_markdown in getattr(test.sidebar, "markdown", []):
        chunks.append(str(sidebar_markdown.value))
    return "\n".join(chunks)


def test_home_landing_markers_pin_the_deployed_page() -> None:
    """Home's guided-story markers pin what the deployed ``/`` route serves.

    Rationale: Streamlit Cloud clones this repository, so this render pins
    what the deployed ``/`` route serves — the same deploy-from-git rationale
    as the D&M pins above, without a second network path. Asserted markers:
    the artifact-status success banner stays the trust gate (no error
    banners), the honesty annotations are quoted verbatim (``p = 0.221``,
    ``0.1200``, ``0.1111``, ``-0.0089``, ``0.005274``), and the click-path
    links target the canonical ``/Predict`` and ``/Data_Methods`` routes.

    The plan text named ``/2_Predict``-style hrefs, but Streamlit strips the
    numeric page prefix when routing (``2_Predict.py`` serves at
    ``/Predict``), so the numbered forms are dead links that fall back onto
    Home — the canonical links are asserted instead and the dead numbered
    hrefs are guarded against as a regression.
    """
    test = _render_home()
    assert not test.exception, str(test.exception)
    success_text = " ".join(str(s.value) for s in test.success)
    assert "loaded and validated" in success_text.lower(), (
        "the artifact-status success banner must stay Home's trust gate"
    )
    assert not list(test.error), "the landing must show no error banners"
    markdown_text = _rendered_markdown(test)
    for marker in ("p = 0.221", "0.1200", "0.1111", "-0.0089", "0.005274"):
        assert marker in markdown_text, (
            f"honesty string missing from rendered markdown: {marker}"
        )
    assert "[Try a prediction](/Predict)" in markdown_text, (
        "the primary CTA must link the canonical /Predict route"
    )
    assert "(/Data_Methods)" in markdown_text, (
        "the Data & Methods link must target the canonical /Data_Methods route"
    )
    for dead in ("(/2_Predict)", "(/4_Data_Methods)"):
        assert dead not in markdown_text, f"dead page href in markdown: {dead}"
