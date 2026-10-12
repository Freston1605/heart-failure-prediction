"""T03: the acceptance orchestrator — treated final generation, regeneration,
significance re-verification, and the competitive guard + ship-path branch.

Pinned contracts (MEM077/D039):

1. **Refit-generation logging** — ``run_treated_final_generation`` logs the
   battery as one coherent ``run_kind=final`` generation tagged
   ``treatment_generation`` (superseded runs soft-deleted after a successful
   generation; MLflow retains them), the treated build through the declared
   ``build_candidate`` carrying ``treatment_variant``, every member evaluated
   through :func:`heart.eval.contract.evaluate` on the same held-out split.
2. **No-acceptance branch** — the disposition says "no-acceptance" with the
   measured/rejected counts and ZERO final runs written.
3. **Non-competitive acceptance branch** — the treated generation runs, the
   report regenerates with regeneration linkage, and the ship path triggers
   iff the regenerated winner carries accepted coverage (recorded in writing
   with the coverage claim as evidence).
4. **Competitive guard** — a winner model type covered by an accepted pair is
   a REQUIRED ship path: with no executor wired the orchestrator refuses
   loudly (simulated entirely on a tmp/monkeypatched store — never the real
   store).

All tests run on tmp stores / fixture frames / tmp report paths; the real
store is only read by this slice's manual runtime step, never by pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.eval.metrics import compute_metric_dict
from heart.eval.treatment_declared import (
    LINEAR_FAMILY_SCOPE,
    PAIRED_SIGNIFICANCE_ALPHA,
)
from heart.eval.treatment_gate import VERDICT_ACCEPTED
from heart.eval.treatment_gate_real import (
    REAL_SPLIT_VERSION,
    AcceptanceDisposition,
    TreatmentGateRealError,
    apply_acceptances,
)
from heart.models.spaces import default_params
from heart.models.treatment_final import (
    GENERATION_TAG,
    TreatedFinalError,
    accepted_pairs_from_keys,
    clear_stale_selection_annotations,
    coverage_of,
    declared_coverage,
    run_treated_final_generation,
)
from heart.tracking.mlflow_store import configure_tracking
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import MODEL_TYPE_TAG, RUN_KIND_TAG
from mlflow.tracking import MlflowClient
from tests.helpers_treatment import make_treatment_fixture, treatment_train_test_split
from tests.test_gate_verdicts_artifact import (
    _fixture_metric_dict,
    _log_candidate_run,
    _outcome_dict,
)

TREATED_MODEL_TYPE = "logistic-regression-l2"
ACCEPTED = ("class-weight-balanced", TREATED_MODEL_TYPE)


# ---------------------------------------------------------------------------
# Seams: tmp stores, frames, ledger
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frames() -> tuple:
    """One small deterministic fixture pair of train/test frames (+target).

    240 rows (the helper default): QDA's post-preprocessing feature space is
    20 columns (one-hot dummy trap included), and a class must contribute
    more rows than features for a full-rank covariance — 120 rows leaves only
    14 class-1 train rows, which makes the treated-generation battery fit
    structurally impossible regardless of reg_param.
    """
    X, y = make_treatment_fixture(n_rows=240)
    X_train, X_test, y_train, y_test = treatment_train_test_split(
        X, y, test_size=0.25
    )
    return (
        X_train.assign(**{TARGET_COLUMN: y_train}),
        X_test.assign(**{TARGET_COLUMN: y_test}),
    )


@pytest.fixture()
def seed_store(tmp_path: Path) -> Path:
    return tmp_path / "mlruns"


def _write_ledger(tmp_path: Path) -> Path:
    """A committed battery ledger synthesized from the declared defaults.

    ``load_committed_fit_params`` requires ONE non-empty scalar best_params
    entry per battery member; the declared space defaults satisfy exactly
    that contract (they ARE registry-buildable tuned values) — except QDA:
    the fixture's one-hot dummy-trap collinearity leaves its covariance
    singular at the space default ``reg_param=0.1``, so the ledger records
    the REAL committed battery value (reports/battery-ledger.json) instead,
    which fits the fixture and keeps the test honest about what a committed
    battery actually supplies.
    """
    from heart.models.registry import BATTERY_MODELS

    committed_reg_params = {"qda": {"reg_param": 0.20848923483175424}}
    results = [
        {
            "model_type": spec.model_type,
            "best_params": (
                dict(committed_reg_params[spec.model_type])
                if spec.model_type in committed_reg_params
                else dict(default_params(spec.space))
            ),
        }
        for spec in BATTERY_MODELS
    ]
    path = tmp_path / "battery-ledger.json"
    path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    return path


def _mid_metric_dict() -> dict:
    """A 0.5-ROC metric dict (an 'untouched refit' losing to the treated run)."""
    labels = np.zeros(64, dtype="int64")
    labels[::2] = 1
    probabilities = np.where(labels == 1, 0.5, 0.5)
    return compute_metric_dict(labels, probabilities, np.zeros(64, dtype="int64"))


def _seed_winner_and_candidate(tracking_dir: Path, *, accepted: bool) -> None:
    """A tmp store mirroring the real shape: one untreated winner final run
    plus one stored candidate run for the accepted pair."""
    log_evaluation_run(
        _fixture_metric_dict(),
        model_name="random-forest",
        split_version=REAL_SPLIT_VERSION,
        params={"committed": "params"},
        tags={
            RUN_KIND_TAG: "final",
            MODEL_TYPE_TAG: "random-forest",
            "family": "ensemble",
        },
        tracking_dir=tracking_dir,
        experiment_name="heart-failure-prediction",
    )
    _log_candidate_run(
        tracking_dir,
        variant_key="knn-imputer",
        model_type="qda",
        verdict=VERDICT_ACCEPTED if accepted else "rejected",
        report_outcome=_outcome_dict(
            variant_key="knn-imputer",
            model_type="qda",
            verdict=VERDICT_ACCEPTED if accepted else "rejected",
        ),
    )


def _all_final_runs(tracking_dir: Path) -> list[tuple[object, str]]:
    """Every final run in the tmp store as ``(run, lifecycle_stage)``."""
    config = configure_tracking(tracking_dir=tracking_dir)
    import mlflow
    from mlflow.entities import ViewType

    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        return []
    return [
        (run, str(run.info.lifecycle_stage))
        for run in client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.{RUN_KIND_TAG} = 'final'",
            run_view_type=ViewType.ALL,
            max_results=10000,
        )
    ]


def _run_generation(
    store_path: Path,
    frames: tuple,
    ledger: Path,
    *,
    members: tuple[str, ...] = ("logistic-regression-l2", "qda"),
    accepted: tuple[tuple[str, str], ...] = (ACCEPTED,),
):
    train_frame, test_frame = frames
    return run_treated_final_generation(
        train_frame,
        test_frame,
        accepted_pairs_from_keys(list(accepted)),
        split_version=REAL_SPLIT_VERSION,
        ledger_path=ledger,
        tracking_dir=store_path,
        experiment_name="heart-failure-prediction",
        members=list(members),
    )


# ---------------------------------------------------------------------------
# 1. Coherent treated final generation (logging contract)
# ---------------------------------------------------------------------------


def test_generation_logs_coherent_treated_final_generation(seed_store, frames, tmp_path):
    """run_kind=final + treatment_generation; treatment marks only on treated."""
    result = _run_generation(seed_store, frames, _write_ledger(tmp_path))
    runs_by_id = {run.info.run_id: run for run, _stage in _all_final_runs(seed_store)}
    for run, stage in _all_final_runs(seed_store):
        tags = dict(run.data.tags)
        assert stage == "active"
        assert tags[RUN_KIND_TAG] == "final"
        assert tags["split_version"] == REAL_SPLIT_VERSION == "v1"
        assert tags[GENERATION_TAG] == result.generation_id
        assert 0.0 <= run.data.metrics["roc_auc"] <= 1.0
    treated_run = runs_by_id[result.run_ids[TREATED_MODEL_TYPE]]
    refit_run = runs_by_id[result.run_ids["qda"]]
    assert treated_run.data.tags["treatment_variant"] == "class-weight-balanced"
    assert refit_run.data.tags["treatment_variant"] == "none"
    assert "treatment:" in treated_run.data.tags["model_name"]
    assert "treatment:" not in refit_run.data.tags["model_name"]
    assert set(result.run_ids) == {"logistic-regression-l2", "qda"}


def test_treated_build_evaluates_on_the_supplied_test_split(seed_store, frames, tmp_path):
    """The logged roc_auc IS contract.evaluate() on that held-out split."""
    from heart.eval.contract import evaluate, evaluation_split_from_frames
    from heart.eval.treatment_variants import build_candidate, candidate_by_key

    ledger = _write_ledger(tmp_path)
    result = _run_generation(seed_store, frames, ledger)
    train_frame, test_frame = frames
    committed = {
        entry["model_type"]: entry["best_params"]
        for entry in json.loads(ledger.read_text())["results"]
    }
    pipeline = build_candidate(
        candidate_by_key(ACCEPTED[0]), TREATED_MODEL_TYPE, committed[TREATED_MODEL_TYPE]
    )
    pipeline.fit(train_frame[list(FEATURE_COLUMNS)], train_frame[TARGET_COLUMN])
    expected = float(
        evaluate(pipeline, evaluation_split_from_frames(test_frame))["roc_auc"]
    )
    config = configure_tracking(tracking_dir=Path(seed_store))
    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient()
    run = client.get_run(result.run_ids[TREATED_MODEL_TYPE])
    assert float(run.data.metrics["roc_auc"]) == pytest.approx(expected)


def test_generation_supersedes_prior_final_runs(seed_store, frames, tmp_path):
    """After a successful generation exactly one final run per member is active."""
    log_evaluation_run(
        _fixture_metric_dict(),
        model_name="stale-final",
        split_version=REAL_SPLIT_VERSION,
        params={"committed": "params"},
        tags={
            RUN_KIND_TAG: "final",
            MODEL_TYPE_TAG: TREATED_MODEL_TYPE,
            "family": "linear",
        },
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
    )
    result = _run_generation(seed_store, frames, _write_ledger(tmp_path))
    active_by_member: dict[str, int] = {}
    deleted: list[str] = []
    for run, stage in _all_final_runs(seed_store):
        tags = dict(run.data.tags)
        if stage == "active":
            active_by_member[tags[MODEL_TYPE_TAG]] = (
                active_by_member.get(tags[MODEL_TYPE_TAG], 0) + 1
            )
            assert tags[GENERATION_TAG] == result.generation_id
        else:
            deleted.append(str(run.info.run_id))
    assert active_by_member == {TREATED_MODEL_TYPE: 1, "qda": 1}
    assert len(deleted) == 1  # the stale run, retained for provenance


def test_generation_refuses_unknown_or_out_of_scope_pairs(seed_store, frames, tmp_path):
    """Out-of-scope accepted pair, duplicate model type, unknown variant fail."""
    ledger = _write_ledger(tmp_path)
    with pytest.raises(TreatedFinalError, match="outside the variant's"):
        _run_generation(
            seed_store,
            frames,
            ledger,
            accepted=(("class-weight-balanced", "qda"),),
        )
    with pytest.raises(TreatedFinalError, match="reuse model type"):
        _run_generation(
            seed_store,
            frames,
            ledger,
            accepted=(
                ("class-weight-balanced", TREATED_MODEL_TYPE),
                ("knn-imputer", TREATED_MODEL_TYPE),
            ),
        )
    with pytest.raises(TreatedFinalError, match="unknown variant"):
        accepted_pairs_from_keys(["treatment:nope/x"])


def test_declared_coverage_excludes_the_incumbent():
    """The guard's coverage union is the measurable linear family; the
    committed selection winner is outside it."""
    assert declared_coverage() == frozenset(LINEAR_FAMILY_SCOPE)
    covered_rf, rf_variants = coverage_of("random-forest")
    assert covered_rf is False and rf_variants == ()
    covered_qda, qda_variants = coverage_of("qda")
    assert covered_qda is True and "knn-imputer" in qda_variants


class _FakeGateReport:
    """A duck-typed GateReport subset the orchestrator's extractor consumes."""

    def __init__(self, outcomes, config) -> None:
        self.outcomes = outcomes
        self.measured_outcomes = tuple(outcomes)


# ---------------------------------------------------------------------------
# 2. No-acceptance branch
# ---------------------------------------------------------------------------


_NO_ACCEPTANCE_PAYLOAD = {
    "accepted_treated_keys": [],
    "measured_pair_count": 3,
    "rejected_pair_count": 3,
}


def test_no_acceptance_branch_emits_disposition_and_zero_final_runs(
    seed_store, frames, tmp_path
):
    """Rejected-only payload -> no-acceptance disposition, zero final runs."""
    _seed_winner_and_candidate(seed_store, accepted=False)
    before = len(_all_final_runs(seed_store))
    report_path = tmp_path / "leaderboard.md"
    disposition = apply_acceptances(
        dict(_NO_ACCEPTANCE_PAYLOAD),
        train_frame=frames[0],
        test_frame=frames[1],
        ledger_path=_write_ledger(tmp_path),
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
        leaderboard_report_path=report_path,
    )
    record = disposition.to_dict()
    assert record["branch"] == "no-acceptance"
    assert record["measured_pair_count"] == 3
    assert record["rejected_pair_count"] == 3
    assert record["ship_path_triggered"] is False
    assert record["generation"] is None
    assert record["post_generation_winner_model_type"] == "random-forest"
    assert record["winner_in_accepted_coverage"] is False
    assert "no-acceptance" in record["ship_path_reasoning"]
    assert "ship path not triggered" in record["ship_path_reasoning"]
    assert "measured pair(s)" in record["ship_path_reasoning"]
    # ZERO final runs written: exactly one final run remains in the store and
    # no treated generation exists.
    after = _all_final_runs(seed_store)
    assert len(after) == before
    assert all(
        dict(run.data.tags).get(GENERATION_TAG) is None for run, _stage in after
    )
    assert report_path.exists()


def test_no_acceptance_counts_come_from_the_gate_report(seed_store, frames, tmp_path):
    """A duck-typed report (no stored counts) is counted from its outcomes."""
    from heart.eval.treatment_gate import GateConfig, GateOutcome

    outcomes = tuple(
        GateOutcome(
            candidate_key=variant,
            model_type="qda",
            mechanism="pipeline-variant",
            group="imputation",
            description="Fixture outcome sentence.",
            is_incumbent=False,
            re_measured=True,
            verdict="rejected",
            verdict_text="Fixture rejected verdict.",
            delta=0.0,
            significant=False,
            delta_ok=False,
            standing_ok=True,
            n_folds=5,
            n_repeats=5,
            pooled_oof_metrics=None,
            treated_key=f"treatment:{variant}/qda",
        )
        for variant in ("knn-imputer", "class-weight-balanced")
    )
    report = _FakeGateReport(outcomes, GateConfig())
    _seed_winner_and_candidate(seed_store, accepted=False)
    disposition = apply_acceptances(
        report,
        train_frame=None,
        test_frame=None,
        ledger_path=_write_ledger(tmp_path),
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
        leaderboard_report_path=tmp_path / "leaderboard.md",
    )
    record = disposition.to_dict()
    assert record["branch"] == "no-acceptance"
    assert record["measured_pair_count"] == 2
    assert record["rejected_pair_count"] == 2


def test_accepted_keys_disagreement_is_loud(seed_store):
    """accepted_treated_keys vs outcomes verdicts conflict fails loudly."""
    payload = {
        "accepted_treated_keys": ["treatment:knn-imputer/qda"],
        "outcomes": [
            {
                "treated_key": "treatment:knn-imputer/qda",
                "re_measured": True,
                "verdict": "rejected",
            }
        ],
    }
    with pytest.raises(TreatmentGateRealError, match="disagree"):
        apply_acceptances(
            payload, train_frame=None, test_frame=None
        )


# ---------------------------------------------------------------------------
# 3+4. Treated-generation branch: regeneration + competitive guard
# ---------------------------------------------------------------------------


def _accepted_payload() -> dict[str, object]:
    return {
        "accepted_treated_keys": ["treatment:knn-imputer/qda"],
        "measured_pair_count": 1,
        "rejected_pair_count": 0,
    }


def test_noncompetitive_acceptance_writes_treated_generation_and_regenerates(
    seed_store, frames, tmp_path
):
    """Accepted pair, winner outside coverage -> treated final runs, report
    regeneration linkage, guard's written no-trigger reasoning."""
    _seed_winner_and_candidate(seed_store, accepted=True)
    report_path = tmp_path / "leaderboard.md"
    recorded_ships: list[dict] = []

    def recording_ship_executor(*, split_version, tracking_dir):
        recorded_ships.append({"split_version": split_version})
        return {"machinery": "recording-fake"}

    # members=("random-forest", "qda"): RF re-logged untreated (an untouched
    # model type) with the treated qda build; the winner coverage decision is
    # read back for real from the tmp store.
    disposition = apply_acceptances(
        dict(_accepted_payload()),
        train_frame=frames[0],
        test_frame=frames[1],
        ledger_path=_write_ledger(tmp_path),
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
        members=["random-forest", "qda"],
        leaderboard_report_path=report_path,
        ship_executor=recording_ship_executor,
    )
    record = disposition.to_dict()
    assert record["branch"] == "treated-generation"
    assert record["accepted_pairs"] == [
        {
            "variant_key": "knn-imputer",
            "model_type": "qda",
            "treated_key": "treatment:knn-imputer/qda",
        }
    ]
    generation = record["generation"]
    assert set(generation["run_ids"]) == {"random-forest", "qda"}
    assert generation["treated_run_ids"] == {
        "treatment:knn-imputer/qda": generation["run_ids"]["qda"]
    }
    assert record["leaderboard_report_path"] == str(report_path)
    assert report_path.exists()
    # Self-consistent winner-coverage decision (read back from the tmp store):
    assert record["winner_in_accepted_coverage"] == (
        record["post_generation_winner_model_type"] == "qda"
    )
    if record["ship_path_triggered"]:
        assert recorded_ships, "ship evidence must come from the executor"
        assert record["ship_evidence"] == {"machinery": "recording-fake"}
    else:
        assert not recorded_ships
        assert record["ship_evidence"] is None
        assert "not in any declared candidate's applies_to set" in (
            record["ship_path_reasoning"]
        )
    # the treated final run carries the tags the contract pins
    treated_run_id = generation["treated_run_ids"]["treatment:knn-imputer/qda"]
    runs = {run.info.run_id: run for run, _s in _all_final_runs(seed_store)}
    tags = dict(runs[treated_run_id].data.tags)
    assert tags[GENERATION_TAG] == generation["generation_id"]
    assert tags["treatment_variant"] == "knn-imputer"


def test_competitive_guard_refuses_loudly_when_winner_has_accepted_coverage(
    seed_store, frames, tmp_path, monkeypatch
):
    """Winner model type covered by the accepted pair + no executor wired ->
    loud CompetitiveGuardError (simulated winner coverage; the real store is
    never touched)."""
    import heart.eval.treatment_gate_real as tgr
    from heart.models.treatment_final import TreatedGenerationResult

    _seed_winner_and_candidate(seed_store, accepted=True)

    def fake_generation(*args, **kwargs):
        return TreatedGenerationResult(
            generation_id="treatment-test-final",
            run_ids={"qda": "fake-run-qda"},
            treated_run_ids={"treatment:knn-imputer/qda": "fake-run-qda"},
            superseded_run_ids=(),
        )

    reads = iter(["random-forest", "qda"])
    monkeypatch.setattr(
        "heart.models.treatment_final.run_treated_final_generation", fake_generation
    )
    monkeypatch.setattr(tgr, "_read_leaderboard_winner", lambda **kwargs: next(reads))
    with pytest.raises(TreatedFinalError, match="Competitive ship path REQUIRED"):
        tgr.apply_acceptances(
            dict(_accepted_payload()),
            train_frame=frames[0],
            test_frame=frames[1],
            ledger_path=_write_ledger(tmp_path),
            tracking_dir=seed_store,
            experiment_name="heart-failure-prediction",
            members=["qda"],
            leaderboard_report_path=tmp_path / "leaderboard.md",
        )
    assert next(reads, "empty") == "empty"  # exactly the two reads happened


def test_competitive_trigger_with_executor_records_ship_evidence(
    seed_store, frames, tmp_path, monkeypatch
):
    """Trigger + wired executor -> ship evidence recorded with the trigger."""
    import heart.eval.treatment_gate_real as tgr
    from heart.models.treatment_final import TreatedGenerationResult

    _seed_winner_and_candidate(seed_store, accepted=True)

    def fake_generation(*args, **kwargs):
        return TreatedGenerationResult(
            generation_id="treatment-test-final",
            run_ids={"qda": "fake-run-qda"},
            treated_run_ids={"treatment:knn-imputer/qda": "fake-run-qda"},
            superseded_run_ids=(),
        )

    def fake_ship_executor(*, split_version, tracking_dir):
        return {"machinery": "fake", "selection_winner": "qda"}

    reads = iter(["random-forest", "qda"])
    monkeypatch.setattr(
        "heart.models.treatment_final.run_treated_final_generation", fake_generation
    )
    monkeypatch.setattr(tgr, "_read_leaderboard_winner", lambda **kwargs: next(reads))
    monkeypatch.setattr(
        tgr,
        "build_real_standings",
        lambda members, tracking_dir, split_version: {"incumbent:qda": (0.5, 1)},
    )
    monkeypatch.setattr(tgr, "_stored_significance_readings", lambda *a, **k: ())
    disposition = tgr.apply_acceptances(
        dict(_accepted_payload()),
        train_frame=frames[0],
        test_frame=frames[1],
        ledger_path=_write_ledger(tmp_path),
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
        members=["qda"],
        leaderboard_report_path=tmp_path / "leaderboard.md",
        ship_executor=fake_ship_executor,
    )
    record = disposition.to_dict()
    assert record["ship_path_triggered"] is True
    assert record["winner_in_accepted_coverage"] is True
    assert record["ship_evidence"] == {"machinery": "fake", "selection_winner": "qda"}
    assert "IS covered by the accepted treatment(s)" in record["ship_path_reasoning"]


def test_stored_significance_readings_are_reverified(
    seed_store, frames, tmp_path, monkeypatch
):
    """The treated generation re-reads each accepted pair's stored reading."""

    import heart.eval.treatment_gate_real as tgr
    from heart.models.treatment_final import TreatedGenerationResult

    _seed_winner_and_candidate(seed_store, accepted=True)

    monkeypatch.setattr(
        "heart.models.treatment_final.run_treated_final_generation",
        lambda *a, **k: TreatedGenerationResult(
            generation_id="g",
            run_ids={"qda": "r"},
            treated_run_ids={"treatment:knn-imputer/qda": "r"},
            superseded_run_ids=(),
        ),
    )
    monkeypatch.setattr(
        tgr, "_read_leaderboard_winner", lambda **kwargs: "random-forest"
    )
    monkeypatch.setattr(
        tgr,
        "build_real_standings",
        lambda members, tracking_dir, split_version: {"incumbent:qda": (0.5, 1)},
    )
    disposition = tgr.apply_acceptances(
        dict(_accepted_payload()),
        train_frame=frames[0],
        test_frame=frames[1],
        ledger_path=_write_ledger(tmp_path),
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
        members=["qda"],
        leaderboard_report_path=tmp_path / "leaderboard.md",
    )
    record = disposition.to_dict()
    assert len(record["significance"]) == 1
    entry = record["significance"][0]
    assert entry["treated_key"] == "treatment:knn-imputer/qda"
    assert entry["significant"] is True
    assert float(entry["delta"]) == pytest.approx(0.015)
    assert "stored repeated-CV comparison" in entry["reading_source"]


# ---------------------------------------------------------------------------
# 5. s6 annotation hygiene + regeneration cost (the S06 trap)
# ---------------------------------------------------------------------------


def test_clear_stale_selection_annotations_deletes_prior_s6_tags(seed_store):
    """Stale s6.* tags on prior final runs are deleted before re-annotation."""
    from heart.reporting.leaderboard import SELECTION_TAG_PREFIX

    log_evaluation_run(
        _fixture_metric_dict(),
        model_name="annotation-holder",
        split_version=REAL_SPLIT_VERSION,
        params={"committed": "params"},
        tags={
            RUN_KIND_TAG: "final",
            MODEL_TYPE_TAG: "qda",
            "family": "linear",
            f"{SELECTION_TAG_PREFIX}winner": "qda",
            f"{SELECTION_TAG_PREFIX}metric": "roc_auc",
            f"{SELECTION_TAG_PREFIX}cv_mean": "0.5",
            f"{SELECTION_TAG_PREFIX}cv_std": "0.0",
            f"{SELECTION_TAG_PREFIX}alpha": str(PAIRED_SIGNIFICANCE_ALPHA),
            f"{SELECTION_TAG_PREFIX}ship": "False",
        },
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
    )
    cleared = clear_stale_selection_annotations(
        split_version=REAL_SPLIT_VERSION,
        tracking_dir=seed_store,
        experiment_name="heart-failure-prediction",
    )
    assert len(cleared) == 6
    for run, _stage in _all_final_runs(seed_store):
        remaining = [
            key
            for key in dict(run.data.tags)
            if key.startswith(SELECTION_TAG_PREFIX)
        ]
        assert remaining == []


def test_disposition_serializes_round_trip(tmp_path):
    """AcceptanceDisposition.to_dict() is JSON-serializable verbatim."""
    disposition = AcceptanceDisposition(
        branch="no-acceptance",
        split_version="v1",
        measured_pair_count=5,
        rejected_pair_count=5,
        accepted_pairs=(),
        declared_coverage=tuple(sorted(LINEAR_FAMILY_SCOPE)),
        pre_generation_winner_model_type=None,
        post_generation_winner_model_type="random-forest",
        winner_model_type_changed=None,
        winner_in_accepted_coverage=False,
        ship_path_triggered=False,
        ship_path_reasoning="fixture reasoning",
        ship_evidence=None,
        generation=None,
        selection_annotations_cleared=(),
        annotations_restored={"status": "restored"},
        provenance_inherited=(),
        leaderboard_report_path=str(tmp_path / "leaderboard.md"),
        regenerated_standings=None,
        significance=(),
    )
    assert disposition.to_dict() == json.loads(json.dumps(disposition.to_dict()))


def test_write_gate_verdicts_artifact_embeds_acceptance_record(tmp_path):
    """write_gate_verdicts_artifact embeds the top-level 'acceptance' record
    verbatim when the acceptance branch ran, and omits the key entirely when
    no branch was supplied (the T02 artifact shape stays unchanged)."""
    from heart.reporting.data_methods.gate_verdicts import (
        write_gate_verdicts_artifact,
    )

    _log_candidate_run(
        tmp_path,
        variant_key="knn-imputer",
        model_type="logistic-regression-l2",
        verdict=VERDICT_ACCEPTED,
        report_outcome=_outcome_dict(
            variant_key="knn-imputer",
            model_type="logistic-regression-l2",
            verdict=VERDICT_ACCEPTED,
        ),
    )
    narrow = dict(
        candidate_keys=["knn-imputer"], model_types=["logistic-regression-l2"]
    )
    record = {
        "branch": "no-acceptance",
        "ship_path_triggered": False,
        "accepted_pairs": [],
        "ship_path_reasoning": "no-acceptance: ship path not triggered",
    }
    destination = tmp_path / "artifacts" / "treatment_gate_verdicts.json"
    with_acceptance = write_gate_verdicts_artifact(
        path=destination, tracking_dir=tmp_path, acceptance=record, **narrow
    )
    assert with_acceptance["acceptance"] == record
    reread = json.loads(destination.read_text(encoding="utf-8"))
    assert reread["acceptance"] == record

    without = write_gate_verdicts_artifact(
        path=destination, tracking_dir=tmp_path, **narrow
    )
    assert "acceptance" not in without
    reread = json.loads(destination.read_text(encoding="utf-8"))
    assert "acceptance" not in reread
