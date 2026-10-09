"""Disclaimer wording and rendering for every prediction the app serves.

The words are owned by this module so page code and tests can never drift:
the page renders the same string the tests assert on, so wording cannot
silently drift between what is displayed and what is checked. The
claim is deliberately short, plain-language and non-negotiable — this app
is a *portfolio demonstration*, not a medical device, and no output is
medical advice. The text must stay exactly as written here (including the
"not a medical device" sentence tests pin); wording changes go through
the test that asserts it renders wherever a prediction appears.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

__all__ = [
    "DISCLAIMER_SHORT",
    "DISCLAIMER_FULL",
    "require_disclaimer_acknowledgement",
    "render_disclaimer",
]


#: Short form shown in the page banner above the form.
DISCLAIMER_SHORT: str = (
    "Research portfolio demonstration — **not a medical device**. "
    "No output here is medical advice."
)

#: Full form shown under/next to each prediction result.
DISCLAIMER_FULL: str = (
    "**This is not a medical device, and this prediction is not medical "
    "advice.** The number above comes from a statistical model trained on "
    "a public research dataset and reuses assumptions from that audit; it "
    "has no regulatory clearance, may be wrong in ways the dashboard "
    "cannot detect, and must never be used to diagnose or make any "
    "clinical decision alone. Consult a qualified clinician."
)


def render_disclaimer(*, short: bool = False) -> None:
    """Render the disclaimer prominently (warning banner, not a footnote).

    ``short=True`` renders the one-liner banner above the form; the default
    renders the fuller text used next to every prediction result.
    """
    text = DISCLAIMER_SHORT if short else DISCLAIMER_FULL
    st.warning(text)


def require_disclaimer_acknowledgement(*, key: str = "disclaimer_ack") -> bool:
    """Checkbox shown before a prediction can be requested; returns consent.

    The acknowledgement is deliberately explicit (a checkbox tick, not
    just a scrolled-past banner): a user must confirm they understand
    the outputs are demonstration-only before the app computes a
    probability.
    """
    return bool(
        st.checkbox(
            "I understand this tool is a research portfolio demonstration, "
            "**not a medical device**, and its output is not medical advice.",
            key=key,
        )
    )
