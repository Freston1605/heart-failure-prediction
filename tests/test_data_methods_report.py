"""Tests for the provenance-locked report assembly and CLI seam (M005/S01/T02).

The contract under test is the report layer: :func:`generate_report` must run
the four-stage pre-split EDA over a raw fixture frame, write the stage tables
and figures as artifacts under an output directory, lock them all with a
SHA-256 manifest, and render ``data-methods.md`` part A where

* the four stage headings and the takeaways checklist are present,
* every cited table/figure artifact exists on disk,
* the manifest digests match the written artifact bytes,
* regeneration over the same fixture is byte-identical (determinism), and
* a missing or altered artifact raises the named provenance exception.

The CLI contract is covered by redirecting the module-default paths into
``tmp_path`` and calling :func:`generate_report` with no path arguments —
the real ``python -m`` CLI against the committed ``reports/`` tree is never
run inside tests (that is the S01/T03 real-data end-to-end surface).

Matplotlib is forced to the headless Agg backend before the engine imports it
(same guard as ``tests/test_data_methods_stages.py``), and every write in this
module lands under ``tmp_path`` — never the committed ``reports/`` tree.
"""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from heart.data.load import PINNED_SHA256
from heart.reporting.data_methods import (
    ProvenanceManifestError,
    generate_report,
    verify_manifest,
)


# ---------------------------------------------------------------------------
# Fixture frame (compact model of the raw dataset, mirrors T01's fixture)
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
    """Generate the report + artifacts from the fixture into ``tmp_path``."""
    report_path = generate_report(
        make_frame(),
        output_dir=tmp_path / "data-methods",
        report_path=tmp_path / "data-methods.md",
    )
    return {
        "report": report_path,
        "report_text": report_path.read_text(encoding="utf-8"),
        "artifact_dir": tmp_path / "data-methods",
    }


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------


def test_report_contains_four_stage_headings(generated):
    text = generated["report_text"]
    assert "## Stage 1 — Hygiene" in text
    assert "## Stage 2 — Univariate distributions" in text
    assert "## Stage 3 — Bivariate and multivariate structure" in text
    assert "## Stage 4 — Assumption checks" in text


def test_report_contains_takeaways_checklist(generated):
    text = generated["report_text"]
    assert "## Takeaways checklist" in text
    checklist_items = [
        line
        for line in text.splitlines()
        if line.startswith("- [ ]") and "artifact: `" in line
    ]
    assert len(checklist_items) >= 5  # hygiene, policy, univariate, bivariate, assumptions


def test_fmt_scales_fraction_shares_by_100():
    """Pinned fix: ``pct=True`` inputs are fractions and render as real percentages."""
    from heart.reporting.data_methods.report import _fmt

    assert _fmt(0.5, pct=True) == "50.0000%"


def test_report_percentages_are_scaled_fractions(generated):
    """Regression: shares must never render 100x understated.

    The fixture has 4 Cholesterol sentinel zeros and 1 RestingBP sentinel
    zero out of 120 rows, i.e. true shares of 3.3333% and 0.8333%. Before
    the fix the renderer emitted the unscaled fractions (0.0333% / 0.0083%).
    """
    text = generated["report_text"]
    assert "3.3333%" in text  # Cholesterol: 4/120 impossible zeros
    assert "0.8333%" in text  # RestingBP: 1/120 impossible zero
    assert "0.0333%" not in text
    assert "0.0083%" not in text


def test_every_cited_artifact_exists(generated):
    text = generated["report_text"]
    artifact_dir = generated["artifact_dir"]

    table_citations = [
        line.split("`")[1]
        for line in text.splitlines()
        if line.startswith("*Artifact:* `")
    ]
    figure_citations = [
        line.split("`")[1]
        for line in text.splitlines()
        if line.startswith("*Figure:* `")
    ]
    assert len(table_citations) >= 4  # one per stage minimum
    assert len(figure_citations) >= 1

    cited = set(table_citations + figure_citations)
    for name in cited:
        assert (artifact_dir / name).is_file(), f"cited artifact missing: {name}"


def test_stage_json_artifacts_written_and_json_valid(generated):
    artifact_dir = generated["artifact_dir"]
    expected_files = {
        "meta.json",
        "stage1_hygiene.json",
        "stage2_univariate.json",
        "stage3_bivariate.json",
        "stage4_assumptions.json",
    }
    for name in expected_files:
        payload = json.loads((artifact_dir / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)


def test_fixture_digest_rendered_as_fixture_run(generated):
    # A bare DataFrame carries no committed dataset digest: the report must
    # say so rather than invent one.
    assert "n/a (fixture run" in generated["report_text"]
    assert PINNED_SHA256  # pinned digest is importable for the fixture check


# ---------------------------------------------------------------------------
# Manifest / provenance
# ---------------------------------------------------------------------------


def test_manifest_digests_match_written_artifacts(generated):
    artifact_dir = generated["artifact_dir"]
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "sha256"
    artifacts = manifest["artifacts"]
    assert manifest["artifact_count"] == len(artifacts) > 0
    for name, recorded in artifacts.items():
        actual = hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
        assert actual == recorded, f"digest drift on {name}"


def test_verify_manifest_passes_on_intact_artifacts(generated):
    verified = verify_manifest(generated["artifact_dir"])
    assert len(verified) >= 5  # the five stage JSONs are all manifested


def test_missing_artifact_raises_named_exception(generated):
    (generated["artifact_dir"] / "stage3_bivariate.json").unlink()
    with pytest.raises(ProvenanceManifestError) as excinfo:
        verify_manifest(generated["artifact_dir"])
    assert "stage3_bivariate.json" in str(excinfo.value)


def test_missing_manifest_raises_named_exception(generated):
    (generated["artifact_dir"] / "manifest.json").unlink()
    with pytest.raises(ProvenanceManifestError, match="not found"):
        verify_manifest(generated["artifact_dir"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_regeneration_is_byte_identical(generated, tmp_path):
    """Same fixture + same output dir -> byte-identical report and manifest.

    This property is the provenance lock: any hand-edit to a number in the
    report diverges from a fresh regeneration and fails the comparison
    (the T03 tamper tests exercise that negative path explicitly).
    """
    report_path = generated["report"]
    artifact_dir = generated["artifact_dir"]
    baseline_report = report_path.read_bytes()
    baseline_manifest = (artifact_dir / "manifest.json").read_bytes()

    generate_report(
        make_frame(),
        output_dir=artifact_dir,
        report_path=report_path,
    )
    assert report_path.read_bytes() == baseline_report
    assert (artifact_dir / "manifest.json").read_bytes() == baseline_manifest


def test_stage_json_artifacts_deterministic_across_directories(tmp_path):
    """Stage table bytes depend only on the fixture, not the output path."""
    frame = make_frame()
    dirs = [tmp_path / "run-a", tmp_path / "run-b"]
    for directory in dirs:
        generate_report(
            frame,
            output_dir=directory,
            report_path=tmp_path / (directory.name + ".md"),
        )
    for name in ("meta.json", "stage1_hygiene.json", "stage2_univariate.json",
                 "stage3_bivariate.json", "stage4_assumptions.json"):
        assert (dirs[0] / name).read_bytes() == (dirs[1] / name).read_bytes()


# ---------------------------------------------------------------------------
# CLI contract (default-style paths redirected into tmp_path)
# ---------------------------------------------------------------------------


def test_default_paths_redirect_via_monkeypatch(tmp_path, monkeypatch):
    """Default-path contract: generate_report() with no path arguments writes
    to the module defaults — redirected into tmp_path so the committed
    reports/ tree is never touched by tests.
    """
    from heart.reporting.data_methods import report as report_module

    fake_artifact_dir = tmp_path / "data-methods"
    fake_report = tmp_path / "data-methods.md"
    monkeypatch.setattr(report_module, "DATAMETHODS_DIR", fake_artifact_dir)
    monkeypatch.setattr(report_module, "REPORT_PATH", fake_report)

    written = generate_report(make_frame())

    assert written == fake_report
    assert fake_report.is_file()
    assert (fake_artifact_dir / "manifest.json").is_file()
    assert verify_manifest(fake_artifact_dir)


def test_non_frame_dataset_raises_named_exception():
    from heart.reporting.data_methods import DataMethodsError

    with pytest.raises(DataMethodsError, match="LoadedDataset"):
        generate_report([1, 2, 3], output_dir="/tmp/never-written-by-this-test")
