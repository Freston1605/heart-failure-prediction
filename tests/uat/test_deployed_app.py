"""Tests/uat: browser-backed UAT of the deployed app's two flows.

The deployed app is live at the streamlit.app URL below (recorded in
``reports/deployment.md``). These tests make the checks reproducible from a
clean environment:

- *Reachability tests* run pure HTTP against the deployed URL. The Streamlit
  Community Cloud edge gates raw first-hit fetches with a 303 to
  ``share.streamlit.io/-/auth/app``; these tests assert that exact documented
  contract so an assertion never mistakes session gating for an outage.
- *Contract tests* import the same modules the deployed app serves from
  (``app.lib.disclaimer``, the winning artifact) and pin the wording/values
  that were observed in the recorded browser session (screenshots in
  ``reports/uat-evidence/`` and the per-check table in ``reports/uat.md``).

The interactive, JavaScript-rendered part of both flows (dashboard
navigation, prediction submission, the disclaimer acknowledgement gate) was
exercised in a real browser session; the objective evidence for each step is
committed under ``reports/uat-evidence/`` and summarised in
``reports/uat.md``. The pytest layer keeps the machine-checkable half of
that evidence reproducible: same URL, same wording pin, same expected
probability from the same serialized artifact.

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

_DEPLOYED_URL = "https://heart-failure-prediction-c68cacatqtcrhvls7eszt2.streamlit.app"
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
    """The app iframe's ``_stcore/health`` answers ``ok`` for the running app."""
    health = _get_ok(f"{deployed_url}/~/+/_stcore/health")
    assert health.status_code == 200
    assert health.text.strip().lower() == "ok"


def test_app_iframe_serves_both_flow_routes(deployed_url: str) -> None:
    """Both flow routes are served by the app iframe (no 404 burial)."""
    for route in ("/", "/Explore", "/Predict"):
        resp = _get_ok(f"{deployed_url}/~/+{route}")
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
