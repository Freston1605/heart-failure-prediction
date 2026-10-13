"""Tests for report parts B-F (M006/S01/T03).

The contract under test is the parts B-F rendering layer added to the
provenance-locked report: :func:`generate_report` must render the five new
sections, in fixed order, after part A's stage sections and before the
Technical Appendix, where

* part B cites ``treatment_audit.json`` (written only when the committed
  split is available; otherwise a benign n/a notice names the populate
  command and never a stack trace),
* part C carries the six fixed checklist headings verbatim, with numbers
  cited from the part A stage artifacts,
* part D's ladder rows equal the declared module's rungs (D032),
* part E contains the ``calibration_did_not_improve_brier`` honesty flag,
* part F contains exactly ``gate_text()`` and the ``+0.0100`` threshold
  derived from the declared constant (never a pasted literal), and
* every ``*Artifact:*`` citation under parts B/D/F names a file actually
  written to the output directory, and an audit-engine failure surfaces as
  a named error with no partial report.

The audit is driven through the ``train_frame`` seam with a deterministic
fixture frame so the tests do not depend on the environment's committed
split state; the missing-split path is exercised by pointing the split
version at a nonexistent version via monkeypatch.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from heart.data.split import SPLIT_VERSION
from heart.eval.treatment_declared import (
    DECLARED_CANDIDATES,
    GATE_MIN_DELTA_ROC_AUC,
    IMBALANCE_LADDER,
    PAIRED_SIGNIFICANCE_ALPHA,
    gate_text,
)
from heart.reporting.data_methods import generate_report
from heart.reporting.data_methods.audit import (
    ROBUSTNESS_JUSTIFICATIONS,
    TreatmentAuditError,
)
from heart.reporting.data_methods.narrative import (
    CALIBRATION_FLAG_NAME,
    PART_C_CHECKLIST_HEADINGS,
    SELECTION_REPORT_POINTER,
    SPLIT_POPULATE_COMMAND,
)


# ---------------------------------------------------------------------------
# Fixture frame (compact model of the raw dataset, mirrors the other modules)
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
    """Generate the full report into tmp_path with a fixture train frame."""
    report_path = generate_report(
        make_frame(),
        output_dir=tmp_path / "data-methods",
        report_path=tmp_path / "data-methods.md",
        train_frame=make_frame(seed=7),
    )
    return {
        "report": report_path,
        "report_text": report_path.read_text(encoding="utf-8"),
        "artifact_dir": tmp_path / "data-methods",
    }


def _part_region(text: str, part_letter: str) -> str:
    """Extract one part's region from the rendered markdown."""
    start = text.index(f"## Part {part_letter} — ")
    ends = [
        index
        for index in (
            text.find(marker, start + 1)
            for marker in ("## Part ", "## Technical appendix")
        )
        if index != -1
    ]
    end = min(ends) if ends else len(text)
    return text[start:end]


def _cited_artifacts(region: str) -> set[str]:
    return {
        line.split("`")[1]
        for line in region.splitlines()
        if line.startswith("*Artifact:* `")
    }


# ---------------------------------------------------------------------------
# Part ordering and artifact citations
# ---------------------------------------------------------------------------


def test_five_part_headings_exist_in_order_after_part_a_stages(generated):
    text = generated["report_text"]
    positions = {
        letter: text.index(f"## Part {letter} — ")
        for letter in "BCDEF"
    }
    assert positions["B"] < positions["C"] < positions["D"] < positions["E"] < positions["F"]
    assert text.index("## Stage 4 — Assumption checks") < positions["B"]
    assert positions["F"] < text.index("## Technical appendix")


def test_every_part_bdf_citation_names_a_written_file(generated):
    text = generated["report_text"]
    artifact_dir = generated["artifact_dir"]
    for letter in ("B", "D", "F"):
        region = _part_region(text, letter)
        cited = _cited_artifacts(region)
        assert cited, f"part {letter} cites no artifact"
        for name in cited:
            assert (artifact_dir / name).is_file(), f"cited artifact missing: {name}"
    assert "treatment_audit.json" in _cited_artifacts(_part_region(text, "B"))
    assert "treatment_plan.json" in _cited_artifacts(_part_region(text, "D"))
    assert "treatment_plan.json" in _cited_artifacts(_part_region(text, "F"))
    # The plan artifact is written even without the split; the audit artifact
    # is written here because the fixture environment provides the split.
    assert (artifact_dir / "treatment_plan.json").is_file()
    assert (artifact_dir / "treatment_audit.json").is_file()
    # The manifest covers both new artifacts.
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "treatment_audit.json" in manifest["artifacts"]
    assert "treatment_plan.json" in manifest["artifacts"]


def test_treatment_artifacts_are_valid_json(generated):
    artifact_dir = generated["artifact_dir"]
    for name in ("treatment_audit.json", "treatment_plan.json"):
        payload = json.loads((artifact_dir / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)


# ---------------------------------------------------------------------------
# Part B content
# ---------------------------------------------------------------------------


def test_part_b_renders_summary_deep_and_light_blocks(generated):
    region = _part_region(generated["report_text"], "B")
    assert "### Summary table (whole battery)" in region
    assert "### Deep audit — the linear family (decision D028)" in region
    assert "### Light audit — robust families (decision D028)" in region
    # Actual sklearn chain material rendered from the audit artifact.
    assert "ZeroMedianImputer" in region
    assert "StandardScaler" in region
    assert "OneHotEncoder" in region
    assert "handle_unknown='ignore'" in region
    # Fitted train-side statistics render (fixture medians are real numbers).
    assert "fitted median" in region
    assert "fitted mean" in region
    assert "observed categories" in region


def test_part_b_light_members_carry_declared_justifications(generated):
    region = _part_region(generated["report_text"], "B")
    for justification in ROBUSTNESS_JUSTIFICATIONS.values():
        assert justification in region


def test_part_b_all_ten_battery_members_present(generated):
    from heart.models.registry import BATTERY_MODEL_TYPES

    region = _part_region(generated["report_text"], "B")
    for model_type in sorted(BATTERY_MODEL_TYPES):
        assert f"`{model_type}`" in region


# ---------------------------------------------------------------------------
# Part C content
# ---------------------------------------------------------------------------


def test_part_c_contains_the_six_checklist_headings_verbatim(generated):
    region = _part_region(generated["report_text"], "C")
    positions = [region.index(f"### {heading}") for heading in PART_C_CHECKLIST_HEADINGS]
    assert positions == sorted(positions)


def test_part_c_numbers_come_from_the_stage_artifacts(generated):
    region = _part_region(generated["report_text"], "C")
    assert "*Artifact:* `stage2_univariate.json`" in region
    assert "*Artifact:* `stage4_assumptions.json`" in region


# ---------------------------------------------------------------------------
# Part D content
# ---------------------------------------------------------------------------


def test_part_d_rung_rows_equal_the_declared_module_rungs(generated):
    region = _part_region(generated["report_text"], "D")
    table_rows = [
        line
        for line in region.splitlines()
        if line.startswith("| ") and not line.startswith("| rung")
        and not set(line) <= {"|", "-", " "}
    ]
    assert len(table_rows) == len(IMBALANCE_LADDER)
    by_key = {candidate.variant_key: candidate for candidate in DECLARED_CANDIDATES}
    for row, rung in zip(table_rows, IMBALANCE_LADDER, strict=True):
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells[0] == str(rung.rung)
        assert rung.label in cells[1]
        if rung.variant_key is None:
            assert cells[2] == "—"
        else:
            assert cells[2] == f"`{rung.variant_key}`"
            candidate = by_key[rung.variant_key]
            assert cells[3] == candidate.mechanism
            assert candidate.applies_to


def test_part_d_marks_rung_zero_as_the_incumbent(generated):
    region = _part_region(generated["report_text"], "D")
    assert "none (documented, not re-measured)" in region


# ---------------------------------------------------------------------------
# Part E content
# ---------------------------------------------------------------------------


def test_part_e_contains_the_calibration_honesty_flag(generated):
    region = _part_region(generated["report_text"], "E")
    assert CALIBRATION_FLAG_NAME in region
    assert SELECTION_REPORT_POINTER in region
    # Pointer citation, not a number copy: no table lives in part E.
    assert not [line for line in region.splitlines() if line.startswith("| ")]


# ---------------------------------------------------------------------------
# Part F content (expected strings derived from the declared constants)
# ---------------------------------------------------------------------------


def test_part_f_contains_gate_text_and_threshold_from_constants(generated):
    region = _part_region(generated["report_text"], "F")
    # Derived from the constants, never pasted literals:
    expected_gate_text = gate_text()
    expected_delta = f"+{GATE_MIN_DELTA_ROC_AUC:.4f}"
    expected_alpha = f"{PAIRED_SIGNIFICANCE_ALPHA:g}"
    assert expected_gate_text in region
    assert expected_delta in region
    assert expected_alpha in region


def test_part_f_declares_antecedence(generated):
    region = _part_region(generated["report_text"], "F")
    assert "declared_before_any_candidate_number: true" in region
    assert "before any candidate was measured" in region


# ---------------------------------------------------------------------------
# Missing-split degradation (benign n/a, never a stack trace)
# ---------------------------------------------------------------------------


def test_missing_split_renders_benign_audit_block(tmp_path, monkeypatch):
    from heart.reporting.data_methods import report as report_module

    monkeypatch.setattr(report_module, "SPLIT_VERSION", "v-does-not-exist")
    report_path = generate_report(
        make_frame(),
        output_dir=tmp_path / "data-methods",
        report_path=tmp_path / "data-methods.md",
    )
    text = report_path.read_text(encoding="utf-8")
    region_b = _part_region(text, "B")
    assert "n/a (audit requires the committed split" in region_b
    assert SPLIT_POPULATE_COMMAND in region_b
    # The plan artifact still renders parts D and F.
    assert "## Part D — Pre-Declared Class-Imbalance Ladder" in text
    assert "## Part F — Pre-Declared Severity Gate (fixed before any candidate " in text
    artifact_dir = tmp_path / "data-methods"
    assert not (artifact_dir / "treatment_audit.json").is_file()
    assert (artifact_dir / "treatment_plan.json").is_file()
    # No dangling citation: part B cites no artifact in the missing-split run.
    assert _cited_artifacts(region_b) == set()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "treatment_plan.json" in manifest["artifacts"]
    assert "treatment_audit.json" not in manifest["artifacts"]


def test_missing_split_versions_constant_is_the_committed_default():
    """The seam defaults to the real committed split version, never a fake."""
    assert SPLIT_VERSION == "v1"


# ---------------------------------------------------------------------------
# Negative: renderer-level drift guards (defense-in-depth, D032)
# ---------------------------------------------------------------------------


def test_part_f_renderer_fails_loudly_on_gate_text_drift():
    """A gate text that disagrees with the declared module fails, not renders."""
    from heart.reporting.data_methods.eda import DataMethodsError
    from heart.reporting.data_methods.narrative import render_part_f

    plan = {
        "declared_before_any_candidate_number": True,
        "gate": {"text": "hand-edited verdict sentence +0.0100"},
    }
    with pytest.raises(DataMethodsError, match="drifted"):
        render_part_f(plan)


def test_part_f_renderer_fails_without_the_antecedence_marker():
    from heart.reporting.data_methods.eda import DataMethodsError
    from heart.reporting.data_methods.narrative import render_part_f

    plan = {"declared_before_any_candidate_number": False, "gate": {"text": gate_text()}}
    with pytest.raises(DataMethodsError, match="declared_before_any_candidate_number"):
        render_part_f(plan)


def test_part_d_renderer_fails_loudly_on_an_unknown_rung_candidate():
    from heart.reporting.data_methods.eda import DataMethodsError
    from heart.reporting.data_methods.narrative import render_part_d

    plan = {"candidates": [], "imbalance_ladder": [{"rung": 1, "label": "x", "variant_key": "ghost"}]}
    with pytest.raises(DataMethodsError, match="never invent a rung"):
        render_part_d(plan)


# ---------------------------------------------------------------------------
# Negative: audit-engine failures surface as named errors, no partial report
# ---------------------------------------------------------------------------


def test_non_frame_train_override_raises_named_error_with_no_report(tmp_path):
    with pytest.raises(TreatmentAuditError) as excinfo:
        generate_report(
            make_frame(),
            output_dir=tmp_path / "data-methods",
            report_path=tmp_path / "data-methods.md",
            train_frame=["not", "a", "frame"],  # type: ignore[arg-type]
        )
    assert "pandas.DataFrame" in str(excinfo.value)
    # Nothing at all was written: the audit is built before any artifact.
    assert not (tmp_path / "data-methods.md").is_file()
    assert not any((tmp_path / "data-methods").rglob("*"))


def test_train_frame_missing_feature_column_raises_named_error(tmp_path):
    frame = make_frame().drop(columns=["Oldpeak"])
    with pytest.raises(TreatmentAuditError) as excinfo:
        generate_report(
            make_frame(),
            output_dir=tmp_path / "data-methods",
            report_path=tmp_path / "data-methods.md",
            train_frame=frame,
        )
    assert "Oldpeak" in str(excinfo.value)
    assert not (tmp_path / "data-methods.md").is_file()
    assert not any((tmp_path / "data-methods").rglob("*"))
