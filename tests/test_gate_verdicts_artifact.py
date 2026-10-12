"""T02: store->projection assembly for the treatment-gate verdicts artifact.

Seeds a tmp tracking store with fake candidate runs (the exact param/tag/
artifact contract the S02 runner writes) and proves:
- the artifact assembly follows the store (fields, order, derived_from_mlflow),
- the loud mismatch failures (missing pair, duplicate pair, threshold-constant
  contradiction, clause disagreement,几何 fingerprint disagreement, missing or
  artifact-inconsistent treated key) all raise with the offending run named,
- :func:`declared_pairs` enumerates the declared measurable pairs and refuses
  unknown keys,
- :func:`standings_for_pair` accepts :class:`heart.eval.treatment_gate_real.
  PairKeys` (the pre-measurement supply the real command uses).

Real-run mechanics stay manual: this slice's runtime step executes the real
command against the real store (the artifact then verifies the real store in
the slice's runtime evidence, not in pytest).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from heart.config import RANDOM_SEED
from heart.eval.metrics import compute_metric_dict
from heart.eval.treatment_declared import (
    DECLARED_CANDIDATES,
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
    MLFLOW_TAG_TREATMENT_VARIANT,
    PAIRED_SIGNIFICANCE_ALPHA,
    gate_text,
)
from heart.eval.treatment_gate import (
    GATE_REPORT_ARTIFACT,
    VERDICT_ACCEPTED,
    GateConfig,
    GateOutcome,
    incumbent_key_for,
)
from heart.eval.treatment_gate_real import (
    REAL_SPLIT_VERSION,
    PairKeys,
    declared_pairs,
    standings_for_pair,
)
from heart.reporting.data_methods.gate_verdicts import (
    GateVerdictsError,
    verify_gate_generation,
    write_gate_verdicts_artifact,
)
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import RUN_KIND_TAG

GATE_FINGERPRINT: str = GateConfig().fingerprint()


def _fixture_metric_dict(roc_mode: str = "mid") -> dict:
    """A complete, schema-valid canonical metric dict (same pattern as T01)."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 8)
    probabilities = 0.55 * labels + 0.25
    predictions = (probabilities >= 0.5).astype(int)
    return compute_metric_dict(labels, probabilities, predictions)


def _outcome_dict(
    *,
    variant_key: str,
    model_type: str,
    verdict: str,
    candidate_key: str | None = None,
) -> dict[str, object]:
    """A per-pair GateOutcome projection exactly like the stored artifact."""
    outcome = GateOutcome(
        candidate_key=candidate_key or variant_key,
        model_type=model_type,
        mechanism="pipeline-variant",
        group="imputation",
        description="Fixture outcome declaration sentence.",
        is_incumbent=False,
        re_measured=True,
        verdict=verdict,
        verdict_text="Fixture verdict text. " + gate_text(),
        delta=0.015,
        significant=True,
        delta_ok=True,
        standing_ok=True,
        n_folds=5,
        n_repeats=5,
        pooled_oof_metrics=_fixture_metric_dict(),
        treated_key=f"treatment:{variant_key}/{model_type}",
    )
    return outcome.to_dict()


def _log_candidate_run(
    tracking_dir: Path,
    *,
    variant_key: str,
    model_type: str,
    verdict: str,
    report_outcome: dict[str, object],
    params_override: dict[str, object] | None = None,
    tags_override: dict[str, str] | None = None,
    artifact_outcomes: list[dict[str, object]] | None = None,
    gate_fingerprint: str = GATE_FINGERPRINT,
) -> str:
    """Log one fake candidate run under the S02 candidate-run contract."""
    treated_key = f"treatment:{variant_key}/{model_type}"
    params: dict[str, object] = {
        "variant_key": variant_key,
        "mechanism": "pipeline-variant",
        "group": "imputation",
        "candidate_group": "imputation",
        "model_type": model_type,
        "applies_to": ("logistic-regression-l2",),
        "n_folds": 5,
        "n_repeats": 5,
        "seed": RANDOM_SEED,
        "gate_fingerprint": gate_fingerprint,
        "gate_metric": GATE_METRIC,
        "alpha": PAIRED_SIGNIFICANCE_ALPHA,
        "min_delta": GATE_MIN_DELTA_ROC_AUC,
        "delta": 0.015,
        "significance_reading": True,
        "delta_ok": True,
        "standing_ok": True,
        "verdict": verdict,
    }
    if params_override:
        params.update(params_override)
    tags = {
        RUN_KIND_TAG: MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
        MLFLOW_TAG_TREATMENT_VARIANT: variant_key,
        "model_type": model_type,
        "family": "linear",
        "gate_verdict": verdict,
        "candidate_group": "imputation",
    }
    if tags_override:
        tags.update(tags_override)
    run_art_dir = tracking_dir / "artifacts-src" / treated_key.replace('/', '_').replace(':', '_')
    run_art_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_art_dir / GATE_REPORT_ARTIFACT
    artifact_path.write_text(
        json.dumps(
            {
                "config": GateConfig().to_dict(),
                "gate_sentence": gate_text(),
                "standings_supplied": True,
                "outcomes": list(
                    artifact_outcomes
                    if artifact_outcomes is not None
                    else [report_outcome]
                ),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return log_evaluation_run(
        _fixture_metric_dict(),
        model_name=treated_key,
        split_version=REAL_SPLIT_VERSION,
        params=params,
        tags=tags,
        artifacts=[artifact_path],
        tracking_dir=tracking_dir,
    ).run_id


P1 = ("knn-imputer", "logistic-regression-l2")
P2 = ("knn-imputer", "logistic-regression-l1")


@pytest.fixture()
def seeded_store(tmp_path: Path):
    """Two stored declared pairs (accepted), tmp store, matched artifacts."""
    for variant_key, model_type in (P1, P2):
        _log_candidate_run(
            tmp_path,
            variant_key=variant_key,
            model_type=model_type,
            verdict=VERDICT_ACCEPTED,
            report_outcome=_outcome_dict(
                variant_key=variant_key, model_type=model_type, verdict=VERDICT_ACCEPTED
            ),
        )
    return tmp_path


NARROW = dict(
    candidate_keys=["knn-imputer"],
    model_types=["logistic-regression-l2", "logistic-regression-l1"],
)

NARROW1 = dict(
    candidate_keys=["knn-imputer"],
    model_types=["logistic-regression-l2"],
)


def test_verify_gate_generation_passes_on_consistent_store(seeded_store):
    check = verify_gate_generation(tracking_dir=seeded_store, **NARROW)
    assert check["run_count"] == 2
    assert check["pairs_covered"] == 2
    assert check["verdicts"] == {
        f"treatment:{key}/{model}": VERDICT_ACCEPTED for key, model in (P1, P2)
    }
    assert check["gate_fingerprint"] == GATE_FINGERPRINT


def test_gate_verdicts_artifact_follows_the_store(seeded_store, tmp_path):
    destination = tmp_path / "art" / "treatment_gate_verdicts.json"
    first = write_gate_verdicts_artifact(
        path=destination, tracking_dir=seeded_store, **NARROW
    )
    second = write_gate_verdicts_artifact(
        path=destination, tracking_dir=seeded_store, **NARROW
    )
    assert first == second
    text = destination.read_text(encoding="utf-8")
    assert text.endswith("}\n") and not text.endswith("!")  # trailing newline
    payload = json.loads(text)
    assert payload["derived_from_mlflow"] is True
    assert payload["gate_sentence"] == gate_text()
    assert payload["standings_supplied"] is True
    assert payload["split_version"] == REAL_SPLIT_VERSION
    assert payload["measured_pair_count"] == 2
    assert set(payload["accepted_treated_keys"]) == {
        f"treatment:{key}/{model}" for key, model in (P1, P2)
    }
    assert [out["run_id"] and out["treated_key"] for out in payload["outcomes"]] == [
        f"treatment:{key}/{model}" for key, model in (P1, P2)
    ]
    first_keys = payload["outcomes"][0].keys()
    assert {
        "candidate_key", "model_type", "mechanism", "group", "re_measured",
        "verdict", "verdict_text", "delta", "significant", "delta_ok",
        "standing_ok", "n_folds", "n_repeats", "pooled_oof_metrics",
        "treated_key", "run_id",
    }.issubset(set(first_keys))
    outcome = payload["outcomes"][0]
    assert outcome["standing_ok"] is True
    rewritten = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert text == rewritten


def test_missing_pair_fails_loudly(seeded_store):
    # Only P1/P2 stored for a full declared enumeration: verify runs over the
    # FULL declared pair set unless a candidate_keys filter is supplied.
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(tracking_dir=seeded_store)
    assert "missing a split_version=" in str(raised.value)
    # With the matching filter the same store verifies clean.
    filtered = verify_gate_generation(
        tracking_dir=seeded_store,
        candidate_keys=["knn-imputer"],
        model_types=["logistic-regression-l2", "logistic-regression-l1"],
    )
    assert filtered["pairs_covered"] == 2


def test_empty_store_fails_loudly(tmp_path):
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(tracking_dir=tmp_path)
    message = str(raised.value)
    assert (
        "holds NO run_kind=" in message
        or "No MLflow experiment named" in message
    )


def test_duplicate_pair_fails_loudly(seeded_store):
    _log_candidate_run(
        seeded_store,
        variant_key=P1[0],
        model_type=P1[1],
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key=P1[0], model_type=P1[1], verdict=VERDICT_ACCEPTED
        ),
    )
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(tracking_dir=seeded_store, **NARROW)
    assert "Duplicated candidate pair(s)" in str(raised.value)


@pytest.mark.parametrize(
    "params",
    [
        {"alpha": 0.5},
        {"min_delta": 0.2},
        {"gate_metric": "f1"},
        {"standing_ok": None},
    ],
)
def test_param_contradictions_fail_loudly(tmp_path, params):
    _log_candidate_run(
        tmp_path,
        variant_key=P1[0],
        model_type=P1[1],
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key=P1[0], model_type=P1[1], verdict=VERDICT_ACCEPTED
        ),
        params_override=params,
    )
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(tracking_dir=tmp_path, **NARROW1)
    message = str(raised.value)
    assert "contradicts the declared" in message or "standing_ok" in message


def test_gate_fingerprint_disagreement_fails_loudly(tmp_path):
    _log_candidate_run(
        tmp_path,
        variant_key=P1[0],
        model_type=P1[1],
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key=P1[0], model_type=P1[1], verdict=VERDICT_ACCEPTED
        ),
    )
    _log_candidate_run(
        tmp_path,
        variant_key=P2[0],
        model_type=P2[1],
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key=P2[0], model_type=P2[1], verdict=VERDICT_ACCEPTED
        ),
        gate_fingerprint=f"{GATE_FINGERPRINT}-hw",
    )
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(tracking_dir=tmp_path, **NARROW)
    assert "disagree on the measurement geometry" in str(raised.value)


def test_verdict_disagreement_between_params_tags_and_artifact(tmp_path):
    _log_candidate_run(
        tmp_path,
        variant_key=P1[0],
        model_type=P1[1],
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key=P1[0],
            model_type=P1[1],
            verdict="rejected",
        ),
    )
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(tracking_dir=tmp_path, **NARROW1)
    assert "verdict disagreement" in str(raised.value)


def test_artifact_lacking_the_runs_treated_key_fails_loudly(tmp_path):
    _log_candidate_run(
        tmp_path,
        variant_key=P1[0],
        model_type=P1[1],
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key=P2[0], model_type=P2[1], verdict=VERDICT_ACCEPTED
        ),
    )
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(tracking_dir=tmp_path, **NARROW1)
    assert "outcome entry/entries" in str(raised.value)


def test_non_treatment_model_name_is_rejected(tmp_path):
    _log_candidate_run(
        tmp_path,
        variant_key=P1[0],
        model_type=P1[1],
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key=P1[0], model_type=P1[1], verdict=VERDICT_ACCEPTED
        ),
        tags_override={"model_name": "plain-model"},
    )
    with pytest.raises(GateVerdictsError) as raised:
        verify_gate_generation(
            tracking_dir=tmp_path,
            candidate_keys=["knn-imputer"],
            model_types=["logistic-regression-l2", "logistic-regression-l1"],
        )
    # log_evaluation_run stamps model_name from the run's "+model_name" tag...
    assert (
        "not a 'treatment:" in str(raised.value) or "holds NO run_kind" in str(raised.value)
    )


def test_declared_pairs_enumeration():
    pairs = declared_pairs()
    expected = {
        f"treatment:{c.variant_key}/{model}"
        for c in DECLARED_CANDIDATES
        if not c.is_incumbent
        for model in c.applies_to
    }
    keys = [pair.pair_keys for pair in pairs]
    assert {k.treated_key for k in keys} == expected
    assert all(k.incumbent_key == incumbent_key_for(k.treated_key.split("/")[-1]) for k in keys)
    with pytest.raises(Exception, match="unknown variant"):
        declared_pairs(["not-a-key"])


def test_standings_for_pair_accepts_pair_keys():
    model_types = ["logistic-regression-l2", "logistic-regression-l1", "knn"]
    board = {
        incumbent_key_for(mt): (float(rank) * 0.01 + 0.8, rank)
        for rank, mt in enumerate(model_types, start=1)
    }
    pair_keys = PairKeys(
        treated_key=f"treatment:knn-imputer/{model_types[0]}",
        incumbent_key=incumbent_key_for(model_types[0]),
    )
    supplied = standings_for_pair(pair_keys, board)
    assert set(supplied) == {pair_keys.treated_key, pair_keys.incumbent_key}
    assert supplied[pair_keys.treated_key] == supplied[pair_keys.incumbent_key]
