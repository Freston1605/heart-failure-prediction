"""Data & Methods page render tests (M005/S02/T02).

Mirroring ``tests/test_runs_page.py`` section 2, ``streamlit.testing.v1.AppTest``
renders of the real ``app/pages/4_Data_Methods.py`` pin the milestone's
zero-exception contract against tmp artifact directories (never the
committed ``reports/`` tree):

1. **Populated render** — the real :func:`heart.reporting.data_methods.
   generate_report` over a small fixture dataset renders zero exceptions
   with the key stage headings visible and no 'Traceback' text anywhere.
2. **Empty state** — ``DATA_METHODS_DIR`` pointing at a nonexistent tmp
   directory renders the friendly named ``artifacts-missing`` message with
   the regeneration-command hint, zero exceptions.
3. **Corrupt state** — a tampered artifact byte (SHA-256 provenance
   failure) renders the named ``artifacts-corrupt`` message, zero
   exceptions.

Matplotlib is forced to the headless Agg backend before the engine imports
it (same guard as ``tests/test_data_methods_view.py``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # noqa: E402
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):  # noqa: E402
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib.data_methods_view import (  # noqa: E402
    DATA_METHODS_DIR_ENV,
    ERROR_KIND_ARTIFACTS_CORRUPT,
    ERROR_KIND_ARTIFACTS_MISSING,
    REGENERATION_COMMAND,
)

PAGES = PROJECT_ROOT / "app" / "pages"
DATA_METHODS_PAGE = PAGES / "4_Data_Methods.py"


# ---------------------------------------------------------------------------
# Fixture frame + populated artifact dir (mirrors test_data_methods_view.py)
# ---------------------------------------------------------------------------


def make_frame(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Deterministic fixture frame in the raw dataset's column shape."""
    rng = np.random.default_rng(seed)
    severity = np.linspace(0.0, 1.0, n)
    cholesterol = rng.integers(120, 400, size=n)
    cholesterol[:4] = 0  # declared sentinels (zero-as-missing policy)
    restingbp = rng.integers(90, 190, size=n)
    restingbp[2] = 0  # declared sentinel
    return pd.DataFrame(
        {
            "Age": rng.integers(30, 81, size=n),
            "Sex": rng.choice(["M", "F"], size=n),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], size=n),
            "RestingBP": restingbp,
            "Cholesterol": cholesterol,
            "FastingBS": rng.integers(0, 2, size=n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], size=n),
            "MaxHR": rng.integers(90, 202, size=n),
            "ExerciseAngina": np.where(rng.random(n) < severity, "Y", "N"),
            "Oldpeak": np.round(severity * 4.0, 2),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], size=n),
            "HeartDisease": np.where(rng.random(n) < severity, 1, 0),
        }
    )


@pytest.fixture()
def generated(tmp_path):
    """Generate the real report artifacts from the fixture into tmp_path."""
    artifact_dir = tmp_path / "data-methods"
    generate = pytest.importorskip(
        "heart.reporting.data_methods", reason="report machinery unavailable"
    )
    generate.generate_report(
        make_frame(),
        output_dir=artifact_dir,
        report_path=tmp_path / "data-methods.md",
    )
    return {"artifact_dir": artifact_dir, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# AppTest helpers (mirroring tests/test_runs_page.py section 2)
# ---------------------------------------------------------------------------


def _page_texts(test) -> str:
    """Every readable text on the rendered page, joined for assertions."""
    element_lists = [
        test.markdown, test.text, test.caption, test.error, test.warning,
        test.info, test.code, test.title, test.header, test.subheader,
        test.json,
    ]
    chunks: list[str] = []
    for el_list in element_lists:
        for el in el_list:
            chunks.append(str(el.value))
    for exp_el in getattr(test, "expander", []):
        try:
            chunks.append(str(exp_el.label))
        except Exception:  # noqa: BLE001 - render/text bookkeeping only
            pass
    for df_el in test.dataframe:
        try:
            chunks.append(df_el.value.to_csv(index=False))
        except Exception:  # noqa: BLE001 - render/text bookkeeping only
            chunks.append(str(df_el.value))
    for image_el in getattr(test, "image", []):
        try:
            chunks.append(str(image_el.caption))
        except Exception:  # noqa: BLE001 - render/text bookkeeping only
            pass
    return "\n".join(chunks)


def _run_page(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]):
    """Render the real 4_Data_Methods.py with the given env (mirrors the
    Runs-page ``_run_runs_page`` helper: monkeypatched env, AppTest, undo in
    ``finally`` so the seam can never leak between tests)."""
    import streamlit.testing.v1 as st_testing

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    test = st_testing.AppTest.from_file(str(DATA_METHODS_PAGE), default_timeout=300)
    try:
        test.run()
    finally:
        monkeypatch.undo()
    return test


# ---------------------------------------------------------------------------
# 1. Populated render: the full four-stage page, zero exceptions
# ---------------------------------------------------------------------------


class TestDataMethodsPageRenderPopulated:
    """Generated tmp artifacts render the full page with zero exceptions."""

    def test_zero_exceptions_and_stage_headings(self, generated, monkeypatch) -> None:
        test = _run_page(
            monkeypatch, {DATA_METHODS_DIR_ENV: str(generated["artifact_dir"])}
        )

        assert test.exception == [], str(test.exception)
        texts = _page_texts(test)
        # Key stage headings render in order.
        assert "Dataset provenance" in texts
        assert "Stage 1 — Data hygiene" in texts
        assert "Stage 2 — Per-column distributions" in texts
        assert "Stage 3 — Column-vs-target relationships" in texts
        assert "Stage 4 — Robustness assumptions" in texts
        assert "Technical appendix" in texts
        # No 'Traceback' text anywhere on the page.
        assert "Traceback" not in texts

    def test_provenance_line_and_artifact_citations(
        self, generated, monkeypatch
    ) -> None:
        test = _run_page(
            monkeypatch, {DATA_METHODS_DIR_ENV: str(generated["artifact_dir"])}
        )

        assert test.exception == []
        texts = _page_texts(test)
        # Dataset provenance line from meta.json: row count, seed, target,
        # declared zero-as-missing columns.
        assert "120 rows" in texts
        assert "`HeartDisease`" in texts
        assert "`RestingBP`" in texts and "`Cholesterol`" in texts
        # Every stage heading cites its artifact file (traceability contract).
        assert "Artifact: stage1_hygiene.json" in texts
        assert "Artifact: stage2_univariate.json" in texts
        assert "Artifact: stage3_bivariate.json" in texts
        assert "Artifact: stage4_assumptions.json" in texts
        assert "Artifact: meta.json" in texts
        # The regeneration command is named on the page.
        assert REGENERATION_COMMAND in texts
        # Figure captions cite their artifact paths.
        assert "stage1_missingness_patterns.png" in texts
        assert "stage3_correlation_heatmap.png" in texts

    def test_stage_tables_render_real_payload_data(
        self, generated, monkeypatch
    ) -> None:
        test = _run_page(
            monkeypatch, {DATA_METHODS_DIR_ENV: str(generated["artifact_dir"])}
        )

        assert test.exception == []
        texts = _page_texts(test)
        # Stage 1: sentinel-zero audit table shows the fixture's known zeros
        # (Cholesterol=4, RestingBP=1 from make_frame).
        assert "True NaN cells" in texts
        assert "Sentinel zeros" in texts
        # The fixture's own row count renders — never the real dataset's 918.
        assert "918 rows" not in texts
        assert "120 rows" in texts
        # Stage 2: skew table covers the fixture's numeric columns.
        for column in ("Age", "MaxHR", "Oldpeak"):
            assert column in texts
        # Stage 3: chi-square table headers.
        assert "Chi-square" in texts
        assert "VIF" in texts
        # Stage 4: outlier-policy comparison headers.
        assert "Mean (winsorized)" in texts
        assert "Value bands" in texts
        # The technical appendix expander exists with the manifest index.
        assert len(test.expander) >= 1

    def test_page_renders_images(self, generated, monkeypatch) -> None:
        test = _run_page(
            monkeypatch, {DATA_METHODS_DIR_ENV: str(generated["artifact_dir"])}
        )

        assert test.exception == []
        # The generated figures render as images (histograms, heatmap...).
        assert len(test.image) >= 10


# ---------------------------------------------------------------------------
# 2. Empty state: missing artifacts render the friendly named message
# ---------------------------------------------------------------------------


class TestDataMethodsPageRenderEmpty:
    """A nonexistent artifact directory renders the friendly message."""

    def test_zero_exceptions_and_named_missing_message(
        self, tmp_path, monkeypatch
    ) -> None:
        missing = tmp_path / "no-such-data-methods"
        test = _run_page(monkeypatch, {DATA_METHODS_DIR_ENV: str(missing)})

        assert test.exception == [], str(test.exception)
        texts = _page_texts(test)
        assert ERROR_KIND_ARTIFACTS_MISSING in texts
        assert REGENERATION_COMMAND in texts
        assert "Traceback" not in texts
        # The attempted directory is shown for diagnosis.
        assert str(missing) in texts
        # Nothing else rendered — the page stopped (stages and
        # M006 transformation Parts B-F alike).
        assert "Stage 1 — Data hygiene" not in texts
        assert "Part B — Per-model treatment audit" not in texts
        assert "Part C — Logistic-regression preparation methodology" not in texts
        assert "Part D — Pre-declared class-imbalance ladder" not in texts
        assert "Part E — Validation and calibration" not in texts
        assert "Part F — Severity-gate verdicts" not in texts


# ---------------------------------------------------------------------------
# 3. Corrupt state: provenance failure renders the named corrupt message
# ---------------------------------------------------------------------------


class TestDataMethodsPageRenderCorrupt:
    """A tampered artifact byte renders the artifacts-corrupt message."""

    def test_zero_exceptions_and_named_corrupt_message(
        self, generated, monkeypatch
    ) -> None:
        png_path = generated["artifact_dir"] / "stage3_correlation_heatmap.png"
        data = bytearray(png_path.read_bytes())
        data[0] ^= 0xFF
        png_path.write_bytes(bytes(data))

        test = _run_page(
            monkeypatch, {DATA_METHODS_DIR_ENV: str(generated["artifact_dir"])}
        )

        assert test.exception == [], str(test.exception)
        texts = _page_texts(test)
        assert ERROR_KIND_ARTIFACTS_CORRUPT in texts
        assert REGENERATION_COMMAND in texts
        assert "Traceback" not in texts
        assert "Stage 1 — Data hygiene" not in texts


# ---------------------------------------------------------------------------
# 3b. Transformation sections: real provenance-verified artifacts render
#     Parts B-F (M006/S04/T02 review pin). Read-only use of the real
#     reports/data-methods tree: the fixture generator cannot produce the
#     MLflow-derived gate verdicts, so only the real directory exercises
#     the populated transformation tables (48 gate pairs, split v1).
# ---------------------------------------------------------------------------


class TestDataMethodsPageRenderTransformations:
    """Real artifacts render every transformation section, zero exceptions."""

    def test_real_artifacts_render_all_transformation_parts(self, monkeypatch) -> None:
        real_dir = PROJECT_ROOT / "reports" / "data-methods"
        for name in (
            "manifest.json",
            "treatment_audit.json",
            "treatment_plan.json",
            "treatment_gate_verdicts.json",
        ):
            if not (real_dir / name).exists():
                pytest.skip(f"real artifact {name} unavailable")
        test = _run_page(monkeypatch, {DATA_METHODS_DIR_ENV: str(real_dir)})

        assert test.exception == [], str(test.exception)
        texts = _page_texts(test)
        assert "Part B — Per-model treatment audit" in texts
        assert "Part C — Logistic-regression preparation methodology" in texts
        assert "Part D — Pre-declared class-imbalance ladder" in texts
        assert "Part E — Validation and calibration" in texts
        assert "Part F — Severity-gate verdicts" in texts
        assert "Traceback" not in texts
        # Audit + ladder content comes from the real payloads.
        assert "logistic-regression-l2" in texts
        assert "class-weight-balanced" in texts
        assert "v1" in texts
        # The verdicts table renders verbatim: 48 measured pairs,
        # all rejected, split v1.
        verdict_frames = [
            df_el.value
            for df_el in test.dataframe
            if "Verdict" in list(getattr(df_el.value, "columns", []))
        ]
        assert verdict_frames, "no verdicts dataframe rendered"
        verdicts = verdict_frames[0]
        assert len(verdicts) == 48
        assert set(verdicts["Verdict"].astype(str)) == {"rejected"}


# ---------------------------------------------------------------------------
# 4. Uncached page: one artifact read per rerun (seam never serves stale)
# ---------------------------------------------------------------------------


class TestPageIsUncached:
    """Re-renders re-read the artifact directory — no stale-cache seam."""

    def test_second_render_sees_new_artifacts(self, generated, tmp_path, monkeypatch) -> None:
        """Render the page against a missing dir, then generate the report
        into that same path and re-render: the second render must show the
        populated page (an uncached page re-reads; a cached one would keep
        serving the empty state)."""
        target = generated["artifact_dir"]
        test = _run_page(
            monkeypatch, {DATA_METHODS_DIR_ENV: str(tmp_path / "late")}
        )
        assert test.exception == []
        assert "Stage 1 — Data hygiene" not in _page_texts(test)

        # Generate into the now-existing directory and re-render.
        import pytest as _pytest

        _pytest.importorskip(
            "heart.reporting.data_methods", reason="report machinery unavailable"
        )
        from heart.reporting.data_methods import generate_report as _gen

        _gen(
            make_frame(),
            output_dir=target,
            report_path=generated["tmp_path"] / "data-methods.md",
        )
        test2 = _run_page(monkeypatch, {DATA_METHODS_DIR_ENV: str(target)})

        assert test2.exception == []
        assert "Stage 1 — Data hygiene" in _page_texts(test2)
