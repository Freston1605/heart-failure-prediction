"""Data & Methods transformation sections tests (M006/S04/T03).

Covers the S04 slice success criteria for parts B-F:

(a) AppTest render of ``app/pages/4_Data_Methods.py`` against the real
    ``reports/data-methods/`` artifact directory raises zero exceptions and
    shows every transformation section header plus the gate-verdicts table.
(b) Fixture test pointing ``DATA_METHODS_DIR`` at an empty tmp dir renders
    only the named friendly empty-state
    ("Data & Methods report is unavailable" + regeneration hint) with zero
    exceptions and never a stack trace.
(c) Parity test asserting the verdicts table source equals
    ``treatment_gate_verdicts.json`` outcomes (48 pairs) verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (headless: never open a GUI window)

from streamlit.testing.v1 import AppTest  # noqa: E402

from app.lib import data_methods_view as dmv  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "4_Data_Methods.py"
REAL_DIR = Path(__file__).resolve().parents[1] / "reports" / "data-methods"
VERDICTS_PATH = REAL_DIR / "treatment_gate_verdicts.json"

FRIENDLY = "Data & Methods report is unavailable"
REGEN_HINT = "PYTHONPATH=src python -m heart.reporting.data_methods"

PART_HEADERS = [
    "Part B \u2014 Per-model treatment audit",
    "Part C \u2014 Logistic-regression preparation methodology",
    "Part D \u2014 Pre-declared class-imbalance ladder",
    "Part E \u2014 Validation and calibration",
    "Part F \u2014 Severity-gate verdicts",
]


def _page_texts(test: AppTest) -> str:
    """Every readable text on the rendered page, joined for assertions."""
    element_lists = [
        test.markdown,
        test.text,
        test.caption,
        test.error,
        test.warning,
        test.info,
        test.code,
        test.title,
        test.header,
        test.subheader,
    ]
    chunks: list[str] = []
    for el_list in element_lists:
        for el in el_list:
            try:
                chunks.append(str(el.value))
            except Exception:  # pragma: no cover - defensive join
                chunks.append(str(el))
    return "\n".join(chunks)


def _run_page(env_dir: str | None) -> AppTest:
    """Run the Data & Methods page with DATA_METHODS_DIR pinned to env_dir."""
    import os

    if env_dir is None:
        os.environ.pop(dmv.DATA_METHODS_DIR_ENV, None)
    else:
        os.environ[dmv.DATA_METHODS_DIR_ENV] = env_dir
    try:
        test = AppTest.from_file(str(PAGE), default_timeout=300)
        test.run()
        return test
    finally:
        os.environ.pop(dmv.DATA_METHODS_DIR_ENV, None)


def test_transformations_render_against_real_artifacts_zero_exceptions() -> None:
    """Real-dir render shows parts B-F with zero exceptions (T03-a)."""
    assert REAL_DIR.is_dir(), f"real artifact dir missing: {REAL_DIR}"
    assert VERDICTS_PATH.is_file(), f"verdicts artifact missing: {VERDICTS_PATH}"

    test = _run_page(str(REAL_DIR))

    assert not test.exception, f"page raised: {test.exception!r}"
    texts = _page_texts(test)
    assert "Traceback" not in texts
    for header in PART_HEADERS:
        assert header in texts, f"missing transformation section: {header!r}"

    verdict_frames = [
        df for df in test.dataframe if "Verdict" in getattr(df.value, "columns", [])
    ]
    assert verdict_frames, "gate-verdicts dataframe missing from page render"
    assert len(verdict_frames[0].value) == 48


def test_transformations_empty_dir_renders_named_friendly_empty_state(
    tmp_path: Path, monkeypatch=None
) -> None:
    """Empty-dir render shows the named friendly message, never a trace (T03-b)."""
    import os

    empty = tmp_path / "empty-methods"
    empty.mkdir()
    old = os.environ.get(dmv.DATA_METHODS_DIR_ENV)
    os.environ[dmv.DATA_METHODS_DIR_ENV] = str(empty)
    try:
        test = AppTest.from_file(str(PAGE), default_timeout=300)
        test.run()
    finally:
        if old is None:
            os.environ.pop(dmv.DATA_METHODS_DIR_ENV, None)
        else:
            os.environ[dmv.DATA_METHODS_DIR_ENV] = old

    assert not test.exception, f"empty-state render raised: {test.exception!r}"
    texts = _page_texts(test)
    assert FRIENDLY in texts, f"named friendly message missing; got:\n{texts[:2000]}"
    assert REGEN_HINT in texts, "regeneration hint missing from empty-state"
    assert "Traceback" not in texts


def test_verdicts_table_source_equals_artifact_verbatim() -> None:
    """View verdicts payload equals treatment_gate_verdicts.json outcomes (T03-c)."""
    expected = json.loads(VERDICTS_PATH.read_text(encoding="utf-8"))
    assert len(expected.get("outcomes", [])) == 48
    assert expected.get("split_version") == "v1"
    assert expected.get("measured_pair_count") == 48
    assert {o.get("verdict") for o in expected["outcomes"]} == {"rejected"}

    view = dmv.load_data_methods_view(artifact_dir=REAL_DIR)
    assert view.ok, f"real-dir view failed: {view.error_kind} {view.error_message}"
    assert view.gate_verdicts is not None, "gate_verdicts missing from real-dir view"
    assert view.gate_verdicts.get("split_version") == "v1"
    assert view.gate_verdicts.get("outcomes") == expected["outcomes"]
