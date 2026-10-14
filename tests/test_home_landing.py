"""Home landing-story tests (M008/S02/T01).

Pins the guided landing story on ``app/Home.py`` against the REAL committed
artifact, following the established AppTest pattern from
``tests/test_app_e2e.py`` (project-root ``sys.path`` bootstrap,
``AppTest.from_file`` with ``clear_artifact_cache()`` around the run, and a
skipif guard on the committed winner artifact):

1. **Banner gate intact** — no page exception, the success banner
   ``loaded and validated`` plus the ``Serving model`` metric, and zero
   error banners on the healthy path.
2. **Story sections** — every guided-story header renders (what the model
   is / how it was validated / honesty notes / where to look next).
3. **Honesty strings** — the committed report figures are quoted verbatim
   in the rendered markdown: ``p = 0.221``, ``not significant``,
   ``0.1200``, ``0.1111``, ``-0.0089``, ``0.005274``.
4. **Click-path links** — markdown links targeting ``/Predict`` and
   ``/Data_Methods`` (plus Runs and Explore) are present. NOTE: the plan
   text named ``/2_Predict``-style hrefs, but Streamlit strips the numeric
   page prefix (``2_Predict.py`` → url pathname ``/Predict``); the numbered
   forms trigger Streamlit's "page does not seem to exist" fallback onto
   Home — verified in a real local browser session, and the deployed UAT's
   ``test_app_serves_all_documented_routes`` pins the canonical routes.
5. **Disclaimer** — the medical-device disclaimer caption still closes the
   page.

Negative paths (corrupt / missing / incompatible artifact) are NOT
re-tested here: ``tests/test_app_e2e.py`` already renders ``app/Home.py``
against each failure and must stay green with the landing story in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib.artifact_loader import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    clear_artifact_cache,
    project_root,
)

REAL_ARTIFACT = project_root() / DEFAULT_ARTIFACT_PATH

pytestmark = pytest.mark.skipif(
    not REAL_ARTIFACT.is_file(),
    reason="Real serialized artifact missing; run the S06 winner flow first.",
)

STORY_SECTIONS = (
    "What the model is",
    "How it was validated",
    "Honesty notes",
    "Where to look next",
)

HONESTY_STRINGS = (
    "p = 0.221",
    "not significant",
    "0.1200",
    "0.1111",
    "-0.0089",
    "0.005274",
)


def _render():
    """Render app/Home.py with the artifact cache cleared before and after.

    Mirrors ``tests/test_app_e2e.py``'s ``_render`` helper minus the env
    seam (Home reads no test-time environment variables).
    """
    import streamlit.testing.v1 as st_testing

    clear_artifact_cache()
    test = st_testing.AppTest.from_file(
        str(PROJECT_ROOT / "app/Home.py"), default_timeout=300
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


# ---------------------------------------------------------------------------
# 1. Banner gate intact on the healthy path
# ---------------------------------------------------------------------------


def test_home_landing_renders_banner_gate_without_errors() -> None:
    test = _render()
    assert not test.exception, str(test.exception)
    texts = " ".join(t.value for t in test.success)
    assert "loaded and validated" in texts.lower(), (
        "the artifact-status success banner must stay the trust gate"
    )
    metrics = {m.label: m.value for m in test.get("metric")}
    assert metrics.get("Serving model") == "random-forest"
    assert not list(test.error), "the healthy landing must not show error banners"


# ---------------------------------------------------------------------------
# 2. Every guided-story section header renders
# ---------------------------------------------------------------------------


def test_home_landing_renders_every_story_section_header() -> None:
    test = _render()
    assert not test.exception, str(test.exception)
    headers = " ".join(h.value for h in test.header)
    for section in STORY_SECTIONS:
        assert section in headers, f"story section header missing: {section}"


# ---------------------------------------------------------------------------
# 3. Honesty strings quoted verbatim in the rendered markdown
# ---------------------------------------------------------------------------


def test_home_landing_quotes_honesty_strings_verbatim() -> None:
    test = _render()
    assert not test.exception, str(test.exception)
    markdown_text = _rendered_markdown(test)
    for marker in HONESTY_STRINGS:
        assert marker in markdown_text, f"honesty string missing from markdown: {marker}"


# ---------------------------------------------------------------------------
# 4. Click-path links into Predict and Data & Methods
# ---------------------------------------------------------------------------


def test_home_landing_links_into_predict_and_data_methods() -> None:
    test = _render()
    assert not test.exception, str(test.exception)
    markdown_text = _rendered_markdown(test)
    assert "[Try a prediction](/Predict)" in markdown_text, (
        "the primary CTA into /Predict must be a markdown link"
    )
    assert "(/Data_Methods)" in markdown_text, (
        "the Data & Methods link must be present"
    )
    assert "(/Runs)" in markdown_text, "the Runs link must be present"
    assert "(/Explore)" in markdown_text, "the Explore link must be present"
    # The numbered plan-literal hrefs are dead: they fall back to Home.
    for dead in ("(/2_Predict)", "(/4_Data_Methods)", "(/3_Runs)"):
        assert dead not in markdown_text, f"dead page href in markdown: {dead}"


# ---------------------------------------------------------------------------
# 5. Medical-device disclaimer caption still closes the page
# ---------------------------------------------------------------------------


def test_home_landing_keeps_medical_device_disclaimer() -> None:
    test = _render()
    assert not test.exception, str(test.exception)
    captions = " ".join(str(c.value) for c in test.caption)
    assert "not a medical device" in captions.lower(), (
        "the closing disclaimer caption must remain on Home"
    )
