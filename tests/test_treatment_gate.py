"""Tests for the treatment gate runner (S02/T04): clauses, pairing, pooled OOF.

The contract pinned here:

1. **Verdict logic mirrors gate_text() clause by clause** — a clause-table
   over synthetic measurements forces each clause alone to fail (delta short
   / not significant / standing worsened), the all-clauses-pass path
   accepts, verdict text embeds the verbatim pre-declared sentence, and a
   missing standings mapping records clause (c) as NOT evaluated — never
   silently true (verdict is then ``rejected-by-incomplete-evidence``, never
   ``accepted``).
2. **No threshold/alpha escape hatches** — ``GateConfig.alpha`` is a
   read-only mirror of the declared constant (any override raises
   ``TreatmentGateError``), and no delta-threshold parameter exists at all.
3. **One evaluation path** — measured pairs run incumbent + variant on the
   same RepeatedCVConfig (identical (repeat, fold) keys -> genuinely paired)
   with ``capture_predictions=True``, and the D036 pooled-OOF metric dict
   computed from the captured predictions passes
   ``validate_metric_dict`` (log-loss included).
4. **Golden-rule wiring** — a sampling measurement produces one
   ``ResampleLineage`` + ``FoldPartition`` per fold and
   ``assert_golden_rule`` passes for the honest in-fold path.

Outcome direction is deliberately NOT asserted on the two fixture
measurements (class-weight-balanced and random-undersampling): sampling luck
must not flip a test, so the clause STRUCTURE is pinned instead.

The T05 section pins the MLflow path (D036): exactly one run per measured
(candidate, model_type) pair under ``run_kind=treatment-candidate`` with the
declared ``treatment_variant`` tag and a contract-valid pooled-OOF metric
dict, the ``gate_report.json`` artifact embedding the verbatim pre-declared
sentence, zero runs for the documented incumbent, the report-only dry mode,
and the leaderboard exclusion-by-construction proof — plus the runner-level
negative test: a pre-split-resampled sampling lineage makes the runner abort
loudly with a :class:`GoldenRuleError` before any verdict exists.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from heart.eval.contract import (
    PRIMARY_METRIC,
    compute_metric_dict,
    validate_metric_dict,
)
from heart.eval.golden_rule import (
    FoldPartition,
    GoldenRuleError,
    ResampleLineage,
    assert_golden_rule,
    random_resample_train_rows,
)
from heart.eval.repeated_cv import RepeatedCVConfig, RepeatedCVResult, FoldScore
from heart.eval.significance import DEFAULT_ALPHA, CorrectedTResult, PairwiseComparison
from heart.eval.treatment_declared import (
    BATTERY_MEMBERS,
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    INCUMBENT_CANDIDATE,
    MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
    MLFLOW_TAG_TREATMENT_VARIANT,
    PAIRED_SIGNIFICANCE_ALPHA,
    gate_text,
)
from heart.eval.treatment_gate import (
    FAMILY_TAG,
    GATE_REPORT_ARTIFACT,
    GATE_SPLIT_VERSION,
    VERDICT_ACCEPTED,
    VERDICT_DOCUMENTED_NOT_RE_MEASURED,
    VERDICT_REJECTED,
    VERDICT_REJECTED_INCOMPLETE_EVIDENCE,
    CandidateMeasurement,
    GateConfig,
    GateReport,
    GateVerdict,
    incumbent_key_for,
    measure_pair,
    run_gate,
    run_gate_core,
    treatment_key_for,
    verdict_treatment,
)
from heart.eval.treatment_variants import (
    TreatmentGateError,
    candidate_by_key,
    iter_buildable_candidates,
    resolve_spec,
)
from heart.reporting.leaderboard import (
    FAMILY_TAG as LEADERBOARD_FAMILY_TAG,
    LeaderboardConfig,
    select_leaderboard_rows,
)
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT, configure_tracking
from heart.tracking.run import (
    load_metrics_artifact,
    load_run,
    load_run_artifact,
    log_evaluation_run,
)
from heart.tuning.runner import FINAL_RUN_KIND, MODEL_TYPE_TAG, RUN_KIND_TAG
from tests.helpers_treatment import make_treatment_fixture

#: Reduced measurement geometry for tests (alpha/thresholds stay locked).
REDUCED = GateConfig(n_folds=3, n_repeats=1)


@pytest.fixture(scope="module")
def fixture() -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd  # noqa: F401 - keep the fixture tuple's types obvious

    X, y = make_treatment_fixture()
    return X, y


@pytest.fixture(scope="module")
def weighted(fixture) -> CandidateMeasurement:
    """class-weight-balanced on logistic-regression-l2 (estimator-param)."""
    X, y = fixture
    return measure_pair(
        candidate_by_key("class-weight-balanced"),
        "logistic-regression-l2",
        X,
        y,
        config=REDUCED,
    )


@pytest.fixture(scope="module")
def undersampled(fixture) -> CandidateMeasurement:
    """random-undersampling on logistic-regression-l2 (sampling mechanism)."""
    X, y = fixture
    return measure_pair(
        candidate_by_key("random-undersampling"),
        "logistic-regression-l2",
        X,
        y,
        config=REDUCED,
    )


# ---------------------------------------------------------------------------
# 1. Constants: mirrors are equal, and there is no override surface
# ---------------------------------------------------------------------------


def test_gate_config_mirrors_the_declared_constants():
    config = GateConfig()
    assert config.alpha == PAIRED_SIGNIFICANCE_ALPHA
    assert config.alpha == DEFAULT_ALPHA  # significance layer agrees too
    assert config.min_delta == GATE_MIN_DELTA_ROC_AUC == 0.01
    assert config.gate_metric == GATE_METRIC == PRIMARY_METRIC == "roc_auc"
    repeated = config.to_repeated_cv_config()
    assert repeated.n_folds == config.n_folds and repeated.seed == config.seed
    assert repeated.n_repeats == config.n_repeats


def test_gate_thresholds_cannot_be_overridden():
    with pytest.raises(TreatmentGateError, match="READ-ONLY mirror"):
        GateConfig(alpha=0.10)
    with pytest.raises(TreatmentGateError, match="READ-ONLY mirror"):
        GateConfig(alpha=0.05 + 1e-12)
    # No delta-threshold parameter exists at all — the attribute surface is
    # fixed, and a hypothetical override kwarg cannot even be passed.
    field_names = {f.name for f in dataclasses.fields(GateConfig)}
    assert not field_names & {"min_delta", "threshold", "min_auc_delta"}


def test_gate_only_alpha_is_locked_not_the_measurement_geometry():
    # n_folds/n_repeats/seed are measurement geometry, freely configurable.
    config = GateConfig(n_folds=3, n_repeats=2, seed=7)
    assert (config.n_folds, config.n_repeats, config.seed) == (3, 2, 7)
    with pytest.raises(TreatmentGateError, match="n_folds"):
        GateConfig(n_folds=1)
    with pytest.raises(TreatmentGateError, match="n_repeats"):
        GateConfig(n_repeats=0)


# ---------------------------------------------------------------------------
# 2. Synthetic measurements for the clause table
# ---------------------------------------------------------------------------


def _synth_result(name: str, value: float) -> RepeatedCVResult:
    config = RepeatedCVConfig(n_folds=2, n_repeats=2, seed=3)
    folds = tuple(
        FoldScore(
            model_name=name,
            repeat=repeat,
            fold=fold,
            n_train=80,
            n_test=20,
            metrics={PRIMARY_METRIC: float(value), "log_loss": 0.41},
        )
        for repeat in range(config.n_repeats)
        for fold in range(config.n_folds)
    )
    means = {
        repeat: {PRIMARY_METRIC: float(value), "log_loss": 0.41}
        for repeat in range(config.n_repeats)
    }
    return RepeatedCVResult(
        model_name=name, config=config, per_fold=folds, per_repeat_means=means
    )


def _synth_comparison(
    delta: float, *, significant: bool
) -> PairwiseComparison:
    return PairwiseComparison(
        model_a="treated",
        model_b="incumbent",
        metric=GATE_METRIC,
        agreement=None,  # McNemar omitted: the declared layer is the t-test
        corrected_t=CorrectedTResult(
            mean_diff=delta,
            std_diff=0.02,
            t_statistic=2.6,
            df=3,
            n_samples_paired=4,
            overfit_correction=0.9,
            p_value=0.001 if significant else 0.5,
            alpha=PAIRED_SIGNIFICANCE_ALPHA,
            significant=significant,
        ),
        alpha=PAIRED_SIGNIFICANCE_ALPHA,
    )


def _synth_measurement(
    treated_mean: float,
    incumbent_mean: float,
    *,
    significant: bool,
    variant_key: str = "class-weight-balanced",
    model_type: str = "logistic-regression-l2",
) -> CandidateMeasurement:
    candidate = candidate_by_key(variant_key)
    return CandidateMeasurement(
        candidate=candidate,
        model_type=model_type,
        treated=_synth_result(treatment_key_for(candidate, model_type), treated_mean),
        incumbent=_synth_result(incumbent_key_for(model_type), incumbent_mean),
        comparison=_synth_comparison(treated_mean - incumbent_mean, significant=significant),
    )


_TREATED_KEY = treatment_key_for(
    candidate_by_key("class-weight-balanced"), "logistic-regression-l2"
)
_INCUMBENT_KEY = incumbent_key_for("logistic-regression-l2")


def _standings(
    treated_rank: int = 1,
    incumbent_rank: int = 2,
    treated_mean: float = 0.96,
    incumbent_mean: float = 0.75,
) -> dict[str, tuple[float, int]]:
    """Battery-style standings including baseline/runner-up positions in the ranks.

    Clause (c) is a comparator over the SUPPLIED standings: it must carry the
    treated variant's entry (key ``treatment:<variant>/<model_type>``) and the
    incumbent's entry (key ``incumbent:<model_type>``), and the baseline and
    runner-up live in the mapping/ranks by construction.
    """
    return {
        # other battery entries (baseline and runner-up embedded in the ranks)
        "xgboss-stub": (0.95, 4),
        "knn-stub": (0.60, 7),
        _INCUMBENT_KEY: (incumbent_mean, incumbent_rank),
        _TREATED_KEY: (treated_mean, treated_rank),
    }


# ---------------------------------------------------------------------------
# 3. Clause table: each clause fails alone; the accept path accepts
# ---------------------------------------------------------------------------


def test_clause_table_delta_short_fails_only_clause_a():
    measurement = _synth_measurement(0.755, 0.75, significant=True)  # delta 0.005
    verdict = verdict_treatment(measurement, _standings())
    assert verdict.delta_ok is False
    assert verdict.significant is True
    assert verdict.standing_ok is True
    assert verdict.verdict == VERDICT_REJECTED


def test_clause_table_boundary_delta_passes_clause_a():
    # Exactly at the declared threshold: >= is inclusive by gate text.
    measurement = _synth_measurement(
        0.75 + GATE_MIN_DELTA_ROC_AUC, 0.75, significant=True
    )
    verdict = verdict_treatment(measurement, _standings())
    assert verdict.delta_ok is True
    assert verdict.verdict == VERDICT_ACCEPTED


def test_clause_table_not_significant_fails_only_clause_b():
    measurement = _synth_measurement(0.90, 0.75, significant=False)
    verdict = verdict_treatment(measurement, _standings())
    assert (verdict.delta_ok, verdict.significant, verdict.standing_ok) == (
        True,
        False,
        True,
    )
    assert verdict.verdict == VERDICT_REJECTED


def test_clause_table_standing_worsened_fails_only_clause_c():
    # The supplied standings already place the treated variant's standing
    # BEHIND its incumbent's standing (rank 3 vs rank 2) — clause (c) fails
    # alone, exactly the leaderboard-authoritative comparison the rule texts.
    measurement = _synth_measurement(0.96, 0.75, significant=True)
    verdict = verdict_treatment(measurement, _standings(treated_rank=3))
    assert (verdict.delta_ok, verdict.significant, verdict.standing_ok) == (
        True,
        True,
        False,
    )
    assert verdict.verdict == VERDICT_REJECTED


def test_clause_table_equal_standing_is_not_a_worsening():
    # An equal rank is not a worsened standing; the clause holds.
    measurement = _synth_measurement(0.96, 0.75, significant=True)
    verdict = verdict_treatment(measurement, _standings(treated_rank=2, incumbent_rank=2))
    assert verdict.standing_ok is True
    assert verdict.verdict == VERDICT_ACCEPTED


def test_clause_table_all_clauses_pass_accepts():
    # Treated mean beats every standings entry -> hypothetical rank 1.
    measurement = _synth_measurement(0.96, 0.75, significant=True)
    verdict = verdict_treatment(measurement, _standings())
    assert (verdict.delta_ok, verdict.significant, verdict.standing_ok) == (
        True,
        True,
        True,
    )
    assert verdict.verdict == VERDICT_ACCEPTED


def test_clause_table_standings_none_records_clause_c_not_evaluated():
    # (a) and (b) pass with no standings: rejected-by-incomplete-evidence,
    # never accepted, and standing_ok is recorded as None (not silently True).
    passing = _synth_measurement(0.96, 0.75, significant=True)
    verdict = verdict_treatment(passing, None)
    assert verdict.standing_ok is None
    assert verdict.verdict == VERDICT_REJECTED_INCOMPLETE_EVIDENCE
    # (a) fails as well -> plain rejection regardless of the missing clause.
    failing = _synth_measurement(0.755, 0.75, significant=False)
    assert verdict_treatment(failing, None).verdict == VERDICT_REJECTED


def test_verdict_text_embeds_the_gate_text_sentence_verbatim():
    sentence = gate_text()
    for measurement, standings in (
        (_synth_measurement(0.96, 0.75, significant=True), _standings()),
        (_synth_measurement(0.755, 0.75, significant=True), _standings()),
        (_synth_measurement(0.96, 0.75, significant=False), None),
        (_synth_measurement(0.90, 0.75, significant=True), _standings(treated_rank=3)),
    ):
        verdict = verdict_treatment(measurement, standings)
        assert isinstance(verdict, GateVerdict)
        assert sentence in verdict.text


def test_standings_comparator_rejects_unusable_standings():
    measurement = _synth_measurement(0.96, 0.75, significant=True)
    with pytest.raises(TreatmentGateError, match="non-empty standings"):
        verdict_treatment(measurement, {})
    # The treated variant's entry is mandatory.
    with pytest.raises(TreatmentGateError, match="mandatory gate key"):
        verdict_treatment(measurement, {_INCUMBENT_KEY: (0.75, 2)})
    with pytest.raises(TreatmentGateError, match="mandatory gate key"):
        verdict_treatment(measurement, {_TREATED_KEY: (0.96, 3)})
    # Ranks must be positive integers (1-based, lower is better).
    with pytest.raises(TreatmentGateError, match="ranks are 1-based"):
        verdict_treatment(
            measurement,
            _standings(treated_rank=0, incumbent_rank=2),
        )
    with pytest.raises(TreatmentGateError, match="rank"):
        verdict_treatment(
            measurement,
            _standings(treated_rank=1, incumbent_rank="2"),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 4. Fixture measurements: paired folds, clause structure, pooled OOF
# ---------------------------------------------------------------------------


def test_measured_pair_results_are_actually_paired(weighted):
    treated_keys = {(s.repeat, s.fold) for s in weighted.treated.per_fold}
    incumbent_keys = {(s.repeat, s.fold) for s in weighted.incumbent.per_fold}
    assert treated_keys == incumbent_keys
    assert len(treated_keys) == REDUCED.n_folds * REDUCED.n_repeats
    assert weighted.comparison.metric == GATE_METRIC
    assert weighted.comparison.alpha == PAIRED_SIGNIFICANCE_ALPHA


def test_measure_pair_fixture_clause_structure(weighted):
    """Clause STRUCTURE pinned; outcome direction left to sampling luck."""
    verdict = verdict_treatment(weighted, None)
    assert verdict.delta_ok == bool(weighted.delta >= GATE_MIN_DELTA_ROC_AUC)
    assert verdict.significant == (
        weighted.comparison.corrected_t.p_value < PAIRED_SIGNIFICANCE_ALPHA
    )
    assert verdict.standing_ok is None
    assert verdict.verdict in {VERDICT_REJECTED, VERDICT_REJECTED_INCOMPLETE_EVIDENCE}
    assert gate_text() in verdict.text
    # The pooled-OOF delta is consistent with the mean delta of the two sides.
    assert weighted.delta == pytest.approx(
        weighted.treated_primary_mean - weighted.incumbent_primary_mean
    )


def test_sampling_measurement_records_lineages_that_pass_the_golden_rule(
    undersampled,
):
    """One lineage + partition per fold, and the honest in-fold path is clean."""
    n_expected = REDUCED.n_folds * REDUCED.n_repeats
    assert len(undersampled.fold_lineages) == n_expected
    assert len(undersampled.fold_partitions) == n_expected
    for lineage, partition in zip(
        undersampled.fold_lineages, undersampled.fold_partitions
    ):
        assert isinstance(lineage, ResampleLineage)
        assert isinstance(partition, FoldPartition)
        assert lineage.variant_key == "random-undersampling"
        # Random undersampling duplicates/drops REAL rows only: no synthetic ids.
        assert lineage.synthetic_ids == () and lineage.synthetic_count == 0
        assert set(lineage.resampled_train_ids).issubset(
            set(lineage.original_train_ids)
        )
        # The exact runner-level assertion passes again for the honest path.
        assert_golden_rule(lineage, partition)  # no raise


def test_pooled_oof_metric_dict_flows_through_validate_metric_dict(weighted):
    metrics = weighted.pooled_oof_metric_dict()
    validate_metric_dict(metrics)  # must pass — no MetricSchemaError raised
    assert isinstance(metrics, dict)
    assert float(metrics["log_loss"]) >= 0.0
    assert 0.0 <= float(metrics[PRIMARY_METRIC]) <= 1.0
    n_pooled = weighted.treated.pooled_y_true.shape[0]
    assert n_pooled == len(make_treatment_fixture()[1]) * REDUCED.n_repeats


# ---------------------------------------------------------------------------
# 5. measure_pair guard rails
# ---------------------------------------------------------------------------


def test_measure_pair_rejects_out_of_scope_model_type():
    X, y = make_treatment_fixture()
    with pytest.raises(TreatmentGateError, match="applies_to"):
        measure_pair(
            candidate_by_key("class-weight-balanced"),
            "knn",  # not in the declared applies_to
            X,
            y,
            config=REDUCED,
        )


def test_measure_pair_rejects_the_incumbent_candidate():
    X, y = make_treatment_fixture()
    with pytest.raises(TreatmentGateError, match="rung 0|incumbent"):
        measure_pair(
            candidate_by_key("median-imputer-incumbent"),
            "logistic-regression-l2",
            X,
            y,
            config=REDUCED,
        )


def test_measure_pair_rejects_non_gate_config():
    X, y = make_treatment_fixture()
    with pytest.raises(TreatmentGateError, match="GateConfig"):
        measure_pair(
            candidate_by_key("class-weight-balanced"),
            "logistic-regression-l2",
            X,
            y,
            config=RepeatedCVConfig(n_folds=3, n_repeats=1),
        )


# ---------------------------------------------------------------------------
# 6. run_gate_core: report assembly, incumbent rung-0 honesty
# ---------------------------------------------------------------------------


def test_run_gate_core_records_incumbent_as_documented_not_re_measured(fixture):
    X, y = fixture
    report = run_gate_core(
        X,
        y,
        candidate_keys=("median-imputer-incumbent", "class-weight-balanced"),
        model_types=("logistic-regression-l2",),
        config=REDUCED,
        standings=None,
    )
    assert len(report.outcomes) == 2
    incumbent = report.outcomes[0]
    assert incumbent.is_incumbent and not incumbent.re_measured
    assert incumbent.verdict == VERDICT_DOCUMENTED_NOT_RE_MEASURED
    assert incumbent.delta is None and incumbent.significant is None
    assert incumbent.model_type is None
    assert incumbent.treated_key is None
    assert incumbent.pooled_oof_metrics is None
    measured = report.outcomes[1]
    assert measured.is_incumbent is False and measured.re_measured is True
    assert measured.candidate_key == "class-weight-balanced"
    assert measured.model_type == "logistic-regression-l2"
    assert measured.treated_key == treatment_key_for(
        candidate_by_key("class-weight-balanced"), "logistic-regression-l2"
    )
    validate_metric_dict(measured.pooled_oof_metrics)  # D036: contract-valid
    assert report.gate_sentence == gate_text()
    assert report.standings_supplied is False


def test_run_gate_core_report_projection_is_json_serializable(fixture):
    X, y = fixture
    report = run_gate_core(
        X,
        y,
        candidate_keys=("median-imputer-incumbent", "class-weight-balanced"),
        model_types=("logistic-regression-l2",),
        config=REDUCED,
    )
    assert isinstance(report, GateReport)
    projection = report.to_dict()
    encoded = json.dumps(projection)
    decoded = json.loads(encoded)
    assert decoded["gate_sentence"] == gate_text()
    assert decoded["config"]["alpha"] == PAIRED_SIGNIFICANCE_ALPHA
    assert decoded["outcomes"][0]["verdict"] == VERDICT_DOCUMENTED_NOT_RE_MEASURED


def test_run_gate_core_smoke_reports_verdicts_for_multiple_candidates(fixture):
    X, y = fixture
    report = run_gate_core(
        X,
        y,
        candidate_keys=("class-weight-balanced", "missingness-flag-columns"),
        model_types=("logistic-regression-l2",),
        config=REDUCED,
    )
    measured = report.measured_outcomes
    assert len(measured) == 2
    for outcome in measured:
        assert outcome.pooled_oof_metrics is not None
        validate_metric_dict(outcome.pooled_oof_metrics)
        assert outcome.verdict in {
            VERDICT_ACCEPTED,
            VERDICT_REJECTED,
            VERDICT_REJECTED_INCOMPLETE_EVIDENCE,
        }
        assert outcome.n_folds == REDUCED.n_folds
        assert outcome.n_repeats == REDUCED.n_repeats
        assert outcome.is_incumbent is False


def test_run_gate_core_and_run_gate_share_the_report_surface(fixture):
    """T05: run_gate is the D036 public entry; the dry mode writes no runs."""
    X, y = fixture
    report = run_gate(
        X,
        y,
        candidate_keys=("median-imputer-incumbent", "class-weight-balanced"),
        model_types=("logistic-regression-l2",),
        config=REDUCED,
        log_to_mlflow=False,
    )
    assert isinstance(report, GateReport)
    assert len(report.measured_outcomes) == 1
    # Dry mode: the same GateReport with run_id=None everywhere.
    assert all(outcome.run_id is None for outcome in report.outcomes)
    assert report.outcomes[0].verdict == VERDICT_DOCUMENTED_NOT_RE_MEASURED


def test_run_gate_core_rejects_unknown_candidate_keys_and_model_types(fixture):
    X, y = fixture
    with pytest.raises(TreatmentGateError, match="unknown variant"):
        run_gate_core(
            X,
            y,
            candidate_keys=("not-a-declared-candidate",),
            config=REDUCED,
        )
    with pytest.raises(TreatmentGateError, match="battery members"):
        run_gate_core(
            X,
            y,
            candidate_keys=("class-weight-balanced",),
            model_types=("not-a-battery-model",),
            config=REDUCED,
        )
    # At least one declared candidate always exists.
    assert BATTERY_MEMBERS and len(BATTERY_MEMBERS) == 10


# ---------------------------------------------------------------------------
# 7. T05: MLflow candidate-run logging (D036), tmp-path store
# ---------------------------------------------------------------------------


def test_family_tag_matches_the_leaderboard_tag():
    """The eval-layer family tag must never drift from the leaderboard's."""
    assert FAMILY_TAG == LEADERBOARD_FAMILY_TAG == "family"


def _shadow_final_metrics() -> dict:
    """A complete, schema-valid metric dict for the shadow run_kind=final run."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 8)
    probabilities = np.clip(0.2 + 0.6 * labels, 0.0, 1.0)
    predictions = (probabilities >= 0.5).astype(int)
    return compute_metric_dict(labels, probabilities, predictions)


class _LoggedGate:
    """One heavy gate run over ALL declared candidates into one tmp store.

    Built once per module (the measurement itself takes ~20s): run_gate with
    ``log_to_mlflow=True`` at the reduced gate geometry plus the QDA variance
    floor (fixture fits cannot do without it), then ONE shadow
    ``run_kind=final`` run logged into the SAME store for the
    exclusion-by-construction proof.
    """

    def __init__(self, report, tracking_dir, config, shadow_run_id):
        self.report = report
        self.tracking_dir = tracking_dir
        self.config = config
        self.shadow_run_id = shadow_run_id

    @property
    def measured(self):
        return self.report.measured_outcomes


def _expected_measured_pairs() -> int:
    """Declared measured pairs: every non-incumbent candidate x applies_to."""
    return sum(len(candidate.applies_to) for candidate in iter_buildable_candidates())


@pytest.fixture(scope="module")
def logged_gate(tmp_path_factory) -> _LoggedGate:
    tracking_dir = tmp_path_factory.mktemp("gate-mlflow") / "mlruns"
    config = configure_tracking(tracking_dir=tracking_dir)
    X, y = make_treatment_fixture()
    report = run_gate(
        X,
        y,
        config=GateConfig(n_folds=3, n_repeats=2),
        log_to_mlflow=True,
        tracking_dir=tracking_dir,
        fit_params={"qda": {"reg_param": 0.1}},
    )
    shadow_run_id = log_evaluation_run(
        _shadow_final_metrics(),
        model_name="shadow-final-logistic",
        split_version="v1",
        params={"C": 1.0},
        tags={
            RUN_KIND_TAG: FINAL_RUN_KIND,
            MODEL_TYPE_TAG: "logistic-regression-l2",
            FAMILY_TAG: "linear",
        },
        tracking_dir=tracking_dir,
        experiment_name=config.experiment_name,
    ).run_id
    return _LoggedGate(report, tracking_dir, config, shadow_run_id)


def _all_store_runs(config) -> list:
    client = MlflowClient(tracking_uri=config.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name)
    assert experiment is not None
    return client.search_runs(
        [experiment.experiment_id], max_results=500
    )


def test_run_gate_logs_exactly_one_run_per_measured_pair(logged_gate):
    """No gaps, no duplicates: one run per (candidate, model_type) pair."""
    report = logged_gate.report
    expected = _expected_measured_pairs()
    assert expected == 48  # 9 linear-family candidates x 5 + class-weights x 3
    assert len(report.outcomes) == expected + 1  # + the incumbent entry
    measured = report.measured_outcomes
    assert len(measured) == expected
    run_ids = [outcome.run_id for outcome in measured]
    assert all(run_id for run_id in run_ids)  # every measured pair got a run
    assert len(set(run_ids)) == len(run_ids) == expected  # no duplicates/gaps
    # Re-read every id back from the store: they all exist and are finished.
    for outcome, run_id in zip(measured, run_ids):
        record = load_run(run_id, config=logged_gate.config)
        assert record.status == "FINISHED"
        assert record.run_name.startswith("treatment-")
    # The incumbent rung-0 entry carries NO run id by construction.
    assert report.incumbent_outcome.run_id is None


def test_candidate_runs_carry_declared_tags_and_contract_metrics(logged_gate):
    """Tags: run_kind, treatment_variant, model_type, family, verdict, group."""
    for outcome in logged_gate.measured:
        record = load_run(outcome.run_id, config=logged_gate.config)
        tags = record.tags
        assert tags[RUN_KIND_TAG] == MLFLOW_RUN_KIND_TREATMENT_CANDIDATE
        assert tags[MLFLOW_TAG_TREATMENT_VARIANT] == (
            candidate_by_key(outcome.candidate_key).mlflow_treatment_variant
        )
        assert tags[MODEL_TYPE_TAG] == outcome.model_type
        assert tags[FAMILY_TAG] == resolve_spec(outcome.model_type).family
        assert tags["gate_verdict"] == outcome.verdict
        assert tags["candidate_group"] == outcome.group
        assert tags["split_version"] == GATE_SPLIT_VERSION == "fixture"
        # The canonical metrics.json artifact loads and matches the report.
        metrics = load_metrics_artifact(outcome.run_id, config=logged_gate.config)
        validate_metric_dict(metrics)  # contract-valid, never validate=False
        assert metrics == dict(outcome.pooled_oof_metrics)
        # Every run carries the gate_report.json artifact alongside.
        assert GATE_REPORT_ARTIFACT in record.artifact_paths
        assert "metrics.json" in record.artifact_paths
        assert "run_config.json" in record.artifact_paths


def test_leaderboard_collection_excludes_candidate_runs_by_construction(
    logged_gate,
):
    """The default run_kind=final collection never picks candidate runs."""
    rows, considered = select_leaderboard_rows(
        config=LeaderboardConfig(),  # default run_kind=final, LATEST-ONLY dedup
        tracking_dir=logged_gate.tracking_dir,
    )
    assert considered == 1  # exactly the one shadow final run matched
    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == logged_gate.shadow_run_id
    assert row.tags[RUN_KIND_TAG] == FINAL_RUN_KIND
    assert MLFLOW_TAG_TREATMENT_VARIANT not in row.tags
    # And the collected set carries no treatment-candidate run at all.
    for candidate_run in logged_gate.measured:
        assert candidate_run.run_id != row.run_id


def test_incumbent_never_logged_and_run_kind_set_is_exact(logged_gate):
    """Zero MLflow runs carry the incumbent variant or an unknown run_kind."""
    runs = _all_store_runs(logged_gate.config)
    expected_total = len(logged_gate.measured) + 1  # + the shadow final run
    assert len(runs) == expected_total
    incumbent_variant = INCUMBENT_CANDIDATE.variant_key
    incumbent_model_names = [
        run.data.tags.get("mlflow.runName", "")
        for run in runs
    ]
    assert not any(
        incumbent_variant in name or name.startswith("incumbent:")
        for name in incumbent_model_names
    )
    assert not any(
        run.data.tags.get(MLFLOW_TAG_TREATMENT_VARIANT) == incumbent_variant
        for run in runs
    )
    observed_kinds = {run.data.tags.get(RUN_KIND_TAG) for run in runs}
    assert observed_kinds == {MLFLOW_RUN_KIND_TREATMENT_CANDIDATE, FINAL_RUN_KIND}


def test_gate_report_artifact_parses_and_embeds_the_verbatim_sentence(
    logged_gate,
):
    """gate_report.json parses and embeds gate_text() exactly, on every run."""
    sentence = gate_text()
    for outcome in logged_gate.measured:
        artifact = load_run_artifact(
            outcome.run_id, GATE_REPORT_ARTIFACT, config=logged_gate.config
        )
        assert isinstance(artifact, dict)
        assert artifact["gate_sentence"] == sentence  # verbatim, never edited
        assert artifact["config"]["alpha"] == PAIRED_SIGNIFICANCE_ALPHA
        incumbent_rows = [
            row for row in artifact["outcomes"] if row["is_incumbent"]
        ]
        assert len(incumbent_rows) == 1
        assert incumbent_rows[0]["verdict"] == VERDICT_DOCUMENTED_NOT_RE_MEASURED
        assert incumbent_rows[0]["run_id"] is None


def test_run_gate_dry_mode_writes_zero_runs(tmp_path):
    """log_to_mlflow=False + a tmp store: report-only, NOTHING written."""
    tracking_dir = tmp_path / "mlruns"
    config = configure_tracking(tracking_dir=tracking_dir)
    X, y = make_treatment_fixture()
    report = run_gate(
        X,
        y,
        candidate_keys=("class-weight-balanced",),
        model_types=("logistic-regression-l2",),
        config=REDUCED,
        log_to_mlflow=False,
        tracking_dir=tracking_dir,
    )
    assert all(outcome.run_id is None for outcome in report.outcomes)
    assert len(_all_store_runs(config)) == 0


def test_fit_params_reject_unknown_or_malformed_supplies(tmp_path):
    """fit_params configures estimators only; bad shapes raise loudly."""
    X, y = make_treatment_fixture()
    with pytest.raises(TreatmentGateError, match="non-battery model"):
        run_gate_core(
            X, y,
            candidate_keys=("class-weight-balanced",),
            model_types=("logistic-regression-l2",),
            config=REDUCED,
            fit_params={"not-a-battery-model": {}},
        )
    with pytest.raises(TreatmentGateError, match="mapping"):
        run_gate_core(
            X, y,
            candidate_keys=("class-weight-balanced",),
            model_types=("logistic-regression-l2",),
            config=REDUCED,
            fit_params={"logistic-regression-l2": "C=1"},  # type: ignore[arg-type]
        )
    with pytest.raises(TreatmentGateError, match="mapping"):
        run_gate_core(
            X, y,
            candidate_keys=("class-weight-balanced",),
            model_types=("logistic-regression-l2",),
            config=REDUCED,
            fit_params=["logistic-regression-l2"],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 8. T05: runner-level golden-rule negative test
# ---------------------------------------------------------------------------


def test_gate_runner_refuses_a_pre_split_resampled_sampling_fixture(
    fixture, monkeypatch
):
    """Resampling the FULL dataset before folding aborts the runner loudly.

    A leaky lineage must never produce a gate verdict: the harness sabotages
    the runner's per-fold lineage builder so the recorded lineage looks as if
    random undersampling ran over the whole dataset BEFORE the split (the
    resampled training rows then contain validation ids). The runner must
    raise ``GoldenRuleError`` naming the contaminating ids, before any number
    exists.
    """
    import heart.eval.treatment_gate as gate_module

    X, y = fixture
    n_original = int(len(X))
    full_ids = tuple(range(n_original))

    def sabotaged_lineage(model, variant_key, train_idx, *, n_original, counter):
        # Sabotage: the lineage as if resampling happened on the FULL dataset
        # before the split (original ids = every row; kept ids include the
        # fold's validation rows -> contamination).
        return random_resample_train_rows(
            full_ids,
            y,
            strategy="under",
            variant_key=variant_key,
            mechanism="RandomUnderSampler",
        )

    monkeypatch.setattr(gate_module, "_build_fold_lineage", sabotaged_lineage)
    with pytest.raises(GoldenRuleError, match="sampling after split") as excinfo:
        measure_pair(
            candidate_by_key("random-undersampling"),
            "logistic-regression-l2",
            X,
            y,
            config=REDUCED,
        )
    message = str(excinfo.value)
    # Named evidence: the detector quotes the offending row ids and the rule.
    assert "resampled training ids outside the fold train set" in message
    assert "never before the split" in message
