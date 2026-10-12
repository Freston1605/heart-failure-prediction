"""Dispositions renderer tests (M006/S05/T01).

(a) Real-artifact: 48 gate-measured lines + 1 rung-0 line, every
    gate-measured line carrying delta + significance + verdict.
(b) N/A-by-evidence: synthetic plan variant absent from verdicts.
(c) Verdicts-None: pre-declared wording preserved, no exception.
(d) Malformed outcome entry raises DataMethodsError loudly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from heart.reporting.data_methods.eda import DataMethodsError
from heart.reporting.data_methods.narrative import (
    _render_gate_dispositions_block,
    format_gate_disposition,
    render_part_d,
)

REAL_DIR = Path(__file__).resolve().parents[1] / "reports" / "data-methods"
PLAN_PATH = REAL_DIR / "treatment_plan.json"
VERDICTS_PATH = REAL_DIR / "treatment_gate_verdicts.json"


def _load_real():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    verdicts = json.loads(VERDICTS_PATH.read_text(encoding="utf-8"))
    return plan, verdicts


def test_real_artifacts_render_48_gate_measured_plus_rung0() -> None:
    plan, verdicts = _load_real()
    assert len(verdicts.get("outcomes", [])) == 48
    lines = _render_gate_dispositions_block(plan, verdicts)
    gate_lines = [ln for ln in lines if ln.startswith("- gate-measured:")]
    assert len(gate_lines) == 48, f"expected 48 gate lines, got {len(gate_lines)}"
    rung0 = [ln for ln in lines if ln.startswith("- rung 0 (")]
    assert len(rung0) == 1
    for ln in gate_lines:
        assert "delta " in ln
        assert "significance " in ln
        assert "verdict " in ln
    # Byte-stability spot check: helper format unchanged.
    assert gate_lines[0].startswith("- " + format_gate_disposition(verdicts["outcomes"][0])[:20])


def test_synthetic_variant_absent_renders_na_by_evidence() -> None:
    plan, verdicts = _load_real()
    plan2 = copy.deepcopy(plan)
    plan2["candidates"].append(
        {
            "variant_key": "synthetic-unmeasured-variant",
            "mechanism": "pipeline-variant",
            "applies_to": ["lda"],
            "is_incumbent": False,
            "order": 99,
        }
    )
    lines = _render_gate_dispositions_block(plan2, verdicts)
    na_lines = [ln for ln in lines if "N/A-by-evidence" in ln]
    assert len(na_lines) == 1
    assert "synthetic-unmeasured-variant" in na_lines[0]
    # The 48 measured lines still render.
    assert len([ln for ln in lines if ln.startswith("- gate-measured:")]) == 48


def test_verdicts_none_returns_predeclared_wording_without_exception() -> None:
    plan, _ = _load_real()
    assert _render_gate_dispositions_block(plan, None) == []
    # render_part_d with no verdicts keeps the ladder table, no gate lines.
    lines = render_part_d(plan, None)
    assert not any(ln.startswith("- gate-measured:") for ln in lines)
    assert any("treatment_plan.json" in ln for ln in lines)


def test_malformed_outcome_entry_raises_loudly() -> None:
    plan, verdicts = _load_real()
    bad = copy.deepcopy(verdicts)
    bad["outcomes"] = [{"candidate_key": "knn-imputer"}]  # missing full field vector
    with pytest.raises(DataMethodsError):
        _render_gate_dispositions_block(plan, bad)


def test_unknown_variant_key_raises_loudly() -> None:
    plan, verdicts = _load_real()
    bad = copy.deepcopy(verdicts)
    alien = copy.deepcopy(bad["outcomes"][0])
    alien["candidate_key"] = "not-a-declared-variant"
    bad["outcomes"] = [alien]
    with pytest.raises(DataMethodsError):
        _render_gate_dispositions_block(plan, bad)
