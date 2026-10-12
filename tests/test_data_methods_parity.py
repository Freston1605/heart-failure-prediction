"""Parity probe tests (T04 adapted: real path src/heart/reporting/data_methods/parity.py)."""
from pathlib import Path
from heart.eval.treatment_declared import GATE_METRIC, GATE_MIN_DELTA_ROC_AUC, PAIRED_SIGNIFICANCE_ALPHA, gate_text
from heart.reporting.data_methods import parity as parity_mod

def test_parity_module_has_no_gsd_paths():
    text = Path(parity_mod.__file__).read_text()
    assert ".gsd/" not in text

def test_parity_checks_cover_declared_constants():
    names = {n for n in dir(parity_mod) if n.startswith("check_")}
    assert "check_alpha" in names
    assert "check_verdicts_match_store" in names
    assert hasattr(parity_mod, "run_parity")
    assert hasattr(parity_mod, "main")
    assert GATE_METRIC == "roc_auc"
    assert GATE_MIN_DELTA_ROC_AUC == 0.01
    assert 0.0 < PAIRED_SIGNIFICANCE_ALPHA < 1.0
    assert gate_text().strip() != ""

def test_parity_probe_passes_on_real_reports():
    report = parity_mod.run_parity()
    assert report.measured_pair_count >= 0
    assert len(report.checks) > 0
    for check in report.checks:
        assert check.passed, "parity check failed: " + check.name + ": " + check.actual

def test_negative_mismatched_alpha_fails_loudly():
    try:
        parity_mod.check_alpha({"alpha": 999.0}, "alpha=999", "alpha=999")
    except parity_mod.ParityProbeError:
        return
    raise AssertionError("mismatched alpha should raise ParityProbeError")

def test_negative_bad_geometry_fails_loudly():
    try:
        parity_mod.check_gate_geometry({"delta": -1.0, "metric": "nope"}, [])
    except Exception:
        return
    raise AssertionError("bad geometry should fail loudly")
