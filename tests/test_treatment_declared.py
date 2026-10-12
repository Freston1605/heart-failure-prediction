"""Tests for the pre-declared treatment plan module (M006/S01, D032).

Contract pinned here:

1. **Import-time validity** — every declared candidate passes the module
   guard (slug key, allowed mechanism/group, battery-only applies_to,
   one-sentence description, matched MLflow tag, ordered list), and the
   imbalance ladder validates against the candidate list.
2. **Rung-set completeness against the roadmap** — the declared candidate
   list is exactly the rung set the M006 S02 roadmap wording names, checked
   against a literal transcription (comment cites the wording).
3. **Gate constants are exact** — delta threshold 0.01, primary metric
   roc_auc, alpha mirroring heart.eval.significance.DEFAULT_ALPHA, and a
   gate_text() that contains each threshold exactly once.
4. **Determinism** — the declared list is byte-stable across a
   module reload (importlib.reload).
5. **Battery containment** — no candidate names a model outside the
   battery, cross-checked against the real registry.
6. **Cheap import** — the module imports with no sklearn/pandas/numpy in
   sys.modules (verified in a subprocess).
7. **Negative surface** — malformed candidates and ladders raise
   TreatmentDeclarationError through the same validators the import guard
   uses.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from heart.eval import treatment_declared as td

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# 1. Import-time validity of every candidate
# --------------------------------------------------------------------------


def test_module_imports_and_full_guard_passes() -> None:
    """validate_all() covers every candidate and the ladder; imports cleanly."""

    td.validate_all()


def test_every_candidate_field_is_from_allowed_literal_sets() -> None:
    allowed_mechanisms = {"estimator-param", "pipeline-variant", "sampling"}
    allowed_groups = {"transform", "imputation", "encoding", "imbalance"}
    for candidate in td.DECLARED_CANDIDATES:
        assert candidate.mechanism in allowed_mechanisms, candidate.variant_key
        assert candidate.group in allowed_groups, candidate.variant_key
        assert isinstance(candidate.applies_to, tuple) and candidate.applies_to
        assert candidate.description.strip().endswith(".", )
        assert candidate.mlflow_treatment_variant == candidate.variant_key
        candidate.validate()  # per-candidate guard raises loudly if malformed


def test_order_values_are_unique_contiguous_one_to_n() -> None:
    orders = [candidate.order for candidate in td.DECLARED_CANDIDATES]
    assert orders == list(range(1, len(orders) + 1))


def test_exactly_one_incumbent_and_it_is_documented_not_measured() -> None:
    incumbents = [c for c in td.DECLARED_CANDIDATES if c.is_incumbent]
    assert len(incumbents) == 1
    assert incumbents[0] is td.INCUMBENT_CANDIDATE
    assert incumbents[0].variant_key == "median-imputer-incumbent"
    assert incumbents[0].group == "imputation"
    # The incumbent's description must say it is documented, not re-measured.
    assert "not re-measured" in incumbents[0].description


# --------------------------------------------------------------------------
# 2. Rung-set completeness, transcribed from the roadmap wording
# --------------------------------------------------------------------------

# Literal transcription of every rung named in the M006 S02 roadmap text
# (".gsd/phases/06-data-treatment-audit-and-severity-gate/06-ROADMAP.md").
# The S02 slice says: "A gate runner measures the declared candidates
# (skew/Box-Cox transforms, median/KNN/missingness-flag imputation variants,
# encoding checks, class_weight balanced, under/over-sampling, SMOTE/ADASYN
# rungs) through evaluate() + repeated stratified CV + paired significance".
# Mapping of roadmap phrases to declared keys:
#   "skew/Box-Cox transforms"          -> yeo-johnson-power-transform,
#                                         box-cox-power-transform
#   "median/KNN/missingness-flag"      -> median-imputer-incumbent
#     "imputation variants"               (documented incumbent, not
#                                          re-measured), knn-imputer,
#                                         missingness-flag-columns
#   "encoding checks"                  -> one-hot-drop-first
#   "class_weight balanced"            -> class-weight-balanced
#   "under/over-sampling"              -> random-undersampling,
#                                         random-oversampling
#   "SMOTE/ADASYN rungs"               -> smote-oversampling,
#                                         adasyn-oversampling
_ROADMAP_TRANSCRIBED_KEYS = (
    "yeo-johnson-power-transform",
    "box-cox-power-transform",
    "median-imputer-incumbent",
    "knn-imputer",
    "missingness-flag-columns",
    "one-hot-drop-first",
    "class-weight-balanced",
    "random-undersampling",
    "random-oversampling",
    "smote-oversampling",
    "adasyn-oversampling",
)


def test_declared_candidates_exactly_match_roadmap_rung_set() -> None:
    assert tuple(td.DECLARED_CANDIDATE_KEYS) == _ROADMAP_TRANSCRIBED_KEYS
    assert len(td.DECLARED_CANDIDATES) == len(set(td.DECLARED_CANDIDATE_KEYS))


def test_imbalance_ladder_order_matches_plan() -> None:
    """Rung 0 none/incumbent -> class weights -> under -> over -> SMOTE -> ADASYN."""

    assert [rung.rung for rung in td.IMBALANCE_LADDER] == [0, 1, 2, 3, 4, 5]
    rung_keys = [rung.variant_key for rung in td.IMBALANCE_LADDER[1:]]
    assert rung_keys == [
        "class-weight-balanced",
        "random-undersampling",
        "random-oversampling",
        "smote-oversampling",
        "adasyn-oversampling",
    ]
    # Rung 0 is the honest none/incumbent rung: no candidate key attached.
    assert td.IMBALANCE_LADDER[0].variant_key is None
    assert "incumbent" in td.IMBALANCE_LADDER[0].label


# --------------------------------------------------------------------------
# 3. Gate constants and gate_text()
# --------------------------------------------------------------------------


def test_gate_min_delta_is_exactly_0_01() -> None:
    assert td.GATE_MIN_DELTA_ROC_AUC == 0.01
    assert isinstance(td.GATE_MIN_DELTA_ROC_AUC, float)


def test_gate_metric_matches_d012_primary_metric() -> None:
    # Cross-check against the real metric contract constant (D012).
    from heart.eval.metrics import PRIMARY_METRIC

    assert td.GATE_METRIC == PRIMARY_METRIC == "roc_auc"


def test_significance_alpha_mirrors_significance_module() -> None:
    from heart.eval.significance import DEFAULT_ALPHA

    assert td.PAIRED_SIGNIFICANCE_ALPHA == DEFAULT_ALPHA == 0.05


def test_gate_text_contains_each_threshold_exactly_once() -> None:
    text = td.gate_text()
    delta = f"+{td.GATE_MIN_DELTA_ROC_AUC:.4f}"  # "+0.0100"
    alpha = f"alpha={td.PAIRED_SIGNIFICANCE_ALPHA:g}"  # "alpha=0.05"
    assert text.count(delta) == 1
    assert text.count(alpha) == 1
    assert td.GATE_METRIC in text
    # The winner-standing rule is rendered verbatim inside the sentence.
    assert td.GATE_WINNER_STANDING_RULE in text


def test_gate_text_is_deterministic() -> None:
    assert td.gate_text() == td.gate_text() == td.gate_text()


def test_gate_text_is_sole_parameterization() -> None:
    """gate_text() reads only its module constants -- perturbation proof."""

    original = td.GATE_MIN_DELTA_ROC_AUC
    try:
        td.GATE_MIN_DELTA_ROC_AUC = 0.02  # type: ignore[misc]
        text = td.gate_text()
        assert "+0.0200" in text and "+0.0100" not in text
    finally:
        td.GATE_MIN_DELTA_ROC_AUC = original  # type: ignore[misc]


# --------------------------------------------------------------------------
# 4. Deterministic declaration list
# --------------------------------------------------------------------------


def test_declared_candidates_deterministic_across_reload() -> None:
    before = [dataclasses.asdict(c) for c in td.DECLARED_CANDIDATES]
    before_ladder = [dataclasses.asdict(r) for r in td.IMBALANCE_LADDER]
    reloaded = importlib.reload(td)
    after = [dataclasses.asdict(c) for c in reloaded.DECLARED_CANDIDATES]
    after_ladder = [dataclasses.asdict(r) for r in reloaded.IMBALANCE_LADDER]
    assert before == after
    assert before_ladder == after_ladder


# --------------------------------------------------------------------------
# 5. Battery containment (cross-checked against the real registry)
# --------------------------------------------------------------------------


def test_battery_members_copy_matches_real_registry() -> None:
    from heart.models.registry import BATTERY_MODEL_TYPES

    assert td.BATTERY_MEMBERS == BATTERY_MODEL_TYPES


def test_no_candidate_names_a_model_outside_the_battery() -> None:
    from heart.models.registry import BATTERY_MODEL_TYPES

    battery = set(BATTERY_MODEL_TYPES)
    for candidate in td.DECLARED_CANDIDATES:
        unknown = set(candidate.applies_to) - battery
        assert not unknown, (candidate.variant_key, unknown)


def test_linear_family_scope_is_within_real_registry_battery() -> None:
    from heart.models.registry import BATTERY_MODEL_TYPES

    assert set(td.LINEAR_FAMILY_SCOPE) <= set(BATTERY_MODEL_TYPES)
    # D028 scope: the three LR variants plus LDA and QDA.
    assert td.LINEAR_FAMILY_SCOPE == (
        "logistic-regression-l2",
        "logistic-regression-l1",
        "logistic-regression-elasticnet",
        "lda",
        "qda",
    )


# --------------------------------------------------------------------------
# 6. Cheap import: no sklearn/pandas/numpy needed
# --------------------------------------------------------------------------


def test_module_imports_without_numeric_or_ml_libraries() -> None:
    code = (
        "import sys\n"
        "import heart.eval.treatment_declared as td\n"
        "banned = ('sklearn', 'pandas', 'numpy', 'scipy', 'mlflow')\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in banned]\n"
        "assert not loaded, f'heavy modules imported: {loaded}'\n"
        "assert len(td.DECLARED_CANDIDATES) == 11\n"
        "assert td.gate_text()\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# 7. Negative surface: malformed declarations fail loudly
# --------------------------------------------------------------------------


def _candidate(**overrides: object) -> td.TreatmentCandidate:
    base = {
        "variant_key": "test-candidate",
        "mechanism": "pipeline-variant",
        "group": "transform",
        "applies_to": ("lda",),
        "description": "Replace one pipeline step with a declared alternative.",
        "mlflow_treatment_variant": "test-candidate",
        "order": 1,
        "is_incumbent": False,
    }
    base.update(overrides)
    # Keep the MLflow tag synced with the variant key unless the test
    # deliberately overrides it (the tag-drift negative case relies on that).
    if "mlflow_treatment_variant" not in overrides:
        base["mlflow_treatment_variant"] = base["variant_key"]
    return td.TreatmentCandidate(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message_fragment"),
    [
        ({"variant_key": ""}, "variant_key"),
        ({"variant_key": "class weight balanced"}, "no spaces"),
        ({"variant_key": "Upper-Kase"}, "slug"),
        ({"mechanism": "hyperparameter"}, "mechanism"),
        ({"group": "scaling"}, "group"),
        ({"applies_to": ()}, "non-empty tuple"),
        ({"applies_to": ("svm-not-in-battery",)}, "outside the battery"),
        ({"applies_to": ("lda", "lda")}, "repeats members"),
        ({"description": "no terminal period"}, "sentence"),
        (
            {"description": "Two sentences. Also a second one."},
            "exactly one sentence",
        ),
        (
            {"mlflow_treatment_variant": "other-key"},
            "must equal the variant key",
        ),
        ({"order": "one"}, "order"),
        ({"is_incumbent": True}, "imputation"),
    ],
)
def test_invalid_candidate_raises(
    overrides: dict[str, object], message_fragment: str
) -> None:
    with pytest.raises(td.TreatmentDeclarationError, match=message_fragment):
        _candidate(**overrides).validate()


def test_non_candidate_object_raises() -> None:
    with pytest.raises(td.TreatmentDeclarationError, match="TreatmentCandidate"):
        td.validate_candidates(
            ("not-a-candidate",), td.BATTERY_MEMBERS  # type: ignore[list-item]
        )


def test_empty_candidates_raises() -> None:
    with pytest.raises(
        td.TreatmentDeclarationError, match="non-empty tuple"
    ):
        td.validate_candidates((), td.BATTERY_MEMBERS)


def test_duplicate_order_raises() -> None:
    first = _candidate()
    second = _candidate(variant_key="another-candidate", order=1)
    with pytest.raises(
        td.TreatmentDeclarationError, match="unique integers 1..N"
    ):
        td.validate_candidates(
            (first, second), td.BATTERY_MEMBERS
        )


def test_ladder_with_candidate_on_rung_zero_raises() -> None:
    ladder = (
        td.LadderRung(rung=0, label="x", variant_key="class-weight-balanced"),
    )
    with pytest.raises(
        td.TreatmentDeclarationError, match="rung 0"
    ):
        td.validate_ladder(ladder, td.DECLARED_CANDIDATES)


def test_ladder_non_contiguous_rungs_raise() -> None:
    ladder = (
        td.LadderRung(rung=0, label="x"),
        td.LadderRung(rung=2, label="y", variant_key=" random-undersampling".strip()),
    )
    with pytest.raises(
        td.TreatmentDeclarationError, match="contiguous"
    ):
        td.validate_ladder(ladder, td.DECLARED_CANDIDATES)


def test_ladder_unknown_reference_raises() -> None:
    ladder = (
        td.LadderRung(rung=0, label="x"),
        td.LadderRung(rung=1, label="y", variant_key="no-such-rung"),
    )
    with pytest.raises(
        td.TreatmentDeclarationError, match="unknown candidate"
    ):
        td.validate_ladder(ladder, td.DECLARED_CANDIDATES)


def test_ladder_non_imbalance_reference_raises() -> None:
    ladder = (
        td.LadderRung(rung=0, label="x"),
        td.LadderRung(rung=1, label="y", variant_key="knn-imputer"),
    )
    with pytest.raises(
        td.TreatmentDeclarationError, match="not an imbalance candidate"
    ):
        td.validate_ladder(ladder, td.DECLARED_CANDIDATES)


def test_ladder_repeated_reference_raises() -> None:
    ladder = (
        td.LadderRung(rung=0, label="x"),
        td.LadderRung(rung=1, label="y", variant_key="smote-oversampling"),
        td.LadderRung(rung=2, label="z", variant_key="smote-oversampling"),
    )
    with pytest.raises(td.TreatmentDeclarationError, match="repeats candidate"):
        td.validate_ladder(ladder, td.DECLARED_CANDIDATES)


def test_empty_ladder_raises() -> None:
    with pytest.raises(td.TreatmentDeclarationError, match="non-empty tuple"):
        td.validate_ladder((), td.DECLARED_CANDIDATES)
