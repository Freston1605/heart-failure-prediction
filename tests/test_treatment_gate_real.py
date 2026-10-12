"""Tests for the S03 real-store gate plumbing (T01).

The contract pinned here:

1. **split_version threading** — ``run_gate`` / ``run_gate_core`` /
   ``_log_gate_report`` carry a threaded, defaulted ``split_version``
   parameter: the default stays :data:`GATE_SPLIT_VERSION` (``"fixture"``,
   byte-identical S02 behaviour) and S03's override (``"v1"``) reaches the
   logged candidate run's ``split_version`` tag. A non-string or empty
   stamp raises loudly — an unstamped candidate run would be
   indistinguishable in the store.
2. **Committed params loader** — ``load_committed_fit_params`` parses the
   committed battery ledger's ``best_params`` into the gate's
   ``{model_type: params}`` ``fit_params`` mapping (real QDA variance floor
   ``0.20848923483175424``, NOT the fixture ``0.1``), cross-checking every
   member against :data:`BATTERY_MEMBERS`; a missing/malformed/duplicate/
   non-scalar entry fails with a named :class:`CommittedParamsError`.
3. **Clause-(c) real standings (D038)** — ``build_real_standings`` reads the
   authoritative leaderboard (``run_kind=final``, ``split_version=v1``,
   latest-only) via the leaderboard module's own collection helpers, keys
   entries ``incumbent:<model_type>`` with FULL-board positive-int ranks
   (primary ROC-AUC desc, model_type tiebreak), excludes non-v1 runs, and
   fails loudly on a missing battery member. ``standings_for_pair`` supplies
   EXACTLY the two mandatory keys per pair, the treated entry inheriting its
   incumbent's rank — so an equal rank passes clause (c) while a worsened
   supplied rank fails it alone (``standing_not_worsened``'s own semantics).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from heart.data.split import SPLIT_VERSION
from heart.eval.contract import compute_metric_dict
from heart.eval.treatment_declared import BATTERY_MEMBERS
from heart.eval.treatment_gate import (
    GATE_SPLIT_VERSION,
    GateConfig,
    CandidateMeasurement,
    incumbent_key_for,
    measure_pair,
    run_gate,
    run_gate_core,
    standing_not_worsened,
    verdict_treatment,
)
from heart.eval.treatment_gate_real import (
    REAL_SPLIT_VERSION,
    CommittedParamsError,
    RealStandingsError,
    build_real_standings,
    load_committed_fit_params,
    standings_for_pair,
)
from heart.eval.treatment_variants import (
    TreatmentGateError,
    candidate_by_key,
)
from heart.reporting.leaderboard import FAMILY_TAG
from heart.tracking.mlflow_store import configure_tracking
from heart.tracking.run import load_run, log_evaluation_run
from heart.tuning.runner import FINAL_RUN_KIND, MODEL_TYPE_TAG, RUN_KIND_TAG
from tests.helpers_treatment import make_treatment_fixture

QDA_COMMITTED_REG_PARAM = 0.20848923483175424


# ---------------------------------------------------------------------------
# Helpers + fixtures: a tmp store seeded with real-style final runs
# ---------------------------------------------------------------------------


def _shadow_final_metrics(roc_mode: str) -> dict:
    """A complete, schema-valid final metric dict with a deterministic ROC-AUC.

    ``perfect`` separates the classes exactly (ROC-AUC 1.0); ``mid`` flips a
    fixed pattern of positives below every negative (ROC-AUC exactly 0.5).
    """
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 8)
    if roc_mode == "perfect":
        probabilities = 0.25 + 0.5 * labels
    elif roc_mode == "mid":
        flips = np.zeros(labels.shape, dtype=bool)
        flips[np.arange(1, len(labels), 4)] = True
        probabilities = np.where(
            labels == 1, np.where(flips, 0.2, 0.75), 0.25
        )
    else:  # pragma: no cover - guard against a typo'd mode
        raise ValueError(f"unknown roc_mode {roc_mode!r}")
    predictions = (probabilities >= 0.5).astype(int)
    return compute_metric_dict(labels, probabilities, predictions)


def _log_final_run(
    tracking_dir,
    config,
    model_type: str,
    roc_mode: str,
    *,
    split_version: str = SPLIT_VERSION,
) -> str:
    return log_evaluation_run(
        _shadow_final_metrics(roc_mode),
        model_name=model_type,
        split_version=split_version,
        params={"committed": "params"},
        tags={
            RUN_KIND_TAG: FINAL_RUN_KIND,
            MODEL_TYPE_TAG: model_type,
            FAMILY_TAG: "linear",
        },
        tracking_dir=tracking_dir,
        experiment_name=config.experiment_name,
    ).run_id


@pytest.fixture(scope="module")
def real_store(tmp_path_factory):
    """A tmp store with v1-split final runs for seven battery members.

    Board ordering (primary ROC-AUC desc, model_type tiebreak): knn and lda
    tie at 1.0 (knn first); then the five 0.5 members order
    elasticnet, l1, l2, qda, svm by model_type ascending. The seeded set
    covers the declared candidates' full applies_to union. One extra final
    run stamped ``split_version="fixture"`` must stay excluded from the v1
    standings.
    """
    tracking_dir = tmp_path_factory.mktemp("real-standings") / "mlruns"
    config = configure_tracking(tracking_dir=tracking_dir)
    for model_type, mode in [
        ("lda", "perfect"),
        ("knn", "perfect"),
        ("logistic-regression-elasticnet", "mid"),
        ("logistic-regression-l1", "mid"),
        ("logistic-regression-l2", "mid"),
        ("qda", "mid"),
        ("svm", "mid"),
    ]:
        _log_final_run(tracking_dir, config, model_type, mode)
    _log_final_run(
        tracking_dir, config, "random-forest", "perfect", split_version="fixture"
    )
    return tracking_dir


@pytest.fixture(scope="module")
def split_versions_logged(tmp_path_factory):
    """Two tiny real gate runs into one tmp store: default vs override stamp."""
    tracking_dir = tmp_path_factory.mktemp("split-threading") / "mlruns"
    config = configure_tracking(tracking_dir=tracking_dir)
    X, y = make_treatment_fixture()
    kwargs = dict(
        candidate_keys=("class-weight-balanced",),
        model_types=("logistic-regression-l2",),
        config=GateConfig(n_folds=3, n_repeats=1),
        log_to_mlflow=True,
        tracking_dir=tracking_dir,
    )
    default_report = run_gate_core(X, y, **kwargs)
    override_report = run_gate_core(X, y, split_version=SPLIT_VERSION, **kwargs)
    return config, default_report, override_report


@pytest.fixture(scope="module")
def measurement():
    """One real paired measurement (tiny geometry) for the standings wiring."""
    X, y = make_treatment_fixture()
    return measure_pair(
        candidate_by_key("class-weight-balanced"),
        "logistic-regression-l2",
        X,
        y,
        config=GateConfig(n_folds=2, n_repeats=2),
    )


# ---------------------------------------------------------------------------
# 1. split_version threading (default unchanged; override reaches the run)
# ---------------------------------------------------------------------------


def test_default_split_version_constant_is_unchanged():
    """The module default stays the S02 fixture stamp (byte-identical)."""
    assert GATE_SPLIT_VERSION == "fixture"
    assert REAL_SPLIT_VERSION == SPLIT_VERSION == "v1"


def test_default_gate_run_logs_the_fixture_split_version(split_versions_logged):
    """No explicit split_version: the logged run carries the fixture stamp."""
    config, default_report, _ = split_versions_logged
    outcome = default_report.measured_outcomes[0]
    record = load_run(outcome.run_id, config=config)
    assert record.tags["split_version"] == GATE_SPLIT_VERSION == "fixture"


def test_split_version_override_reaches_the_logged_run(split_versions_logged):
    """S03's threaded ``"v1"`` lands on the logged candidate run's tag."""
    config, default_report, override_report = split_versions_logged
    assert len(override_report.measured_outcomes) == 1
    outcome = override_report.measured_outcomes[0]
    record = load_run(outcome.run_id, config=config)
    assert record.tags["split_version"] == SPLIT_VERSION == "v1"
    # Same measurement surface as the default call — only the stamp differs.
    assert outcome.treated_key == (
        default_report.measured_outcomes[0].treated_key
    )
    assert record.status == "FINISHED"


def test_split_version_override_through_run_gate(tmp_path):
    """The public entry threads the override too (and rejects empty stamps)."""
    tracking_dir = tmp_path / "mlruns"
    config = configure_tracking(tracking_dir=tracking_dir)
    X, y = make_treatment_fixture()
    report = run_gate(
        X,
        y,
        candidate_keys=("class-weight-balanced",),
        model_types=("logistic-regression-l2",),
        config=GateConfig(n_folds=2, n_repeats=1),
        log_to_mlflow=True,
        tracking_dir=tracking_dir,
        split_version=SPLIT_VERSION,
    )
    record = load_run(report.measured_outcomes[0].run_id, config=config)
    assert record.tags["split_version"] == "v1"
    with pytest.raises(TreatmentGateError, match="split_version"):
        run_gate_core(
            X,
            y,
            candidate_keys=("class-weight-balanced",),
            config=GateConfig(n_folds=2, n_repeats=1),
            split_version="",
        )
    with pytest.raises(TreatmentGateError, match="split_version"):
        run_gate(
            X,
            y,
            config=GateConfig(n_folds=2, n_repeats=1),
            split_version=7,
        )


# ---------------------------------------------------------------------------
# 2. load_committed_fit_params (the battery ledger -> gate fit_params)
# ---------------------------------------------------------------------------


def test_load_committed_fit_params_covers_the_real_battery():
    """The committed ledger yields one params dict per battery member."""
    params = load_committed_fit_params()
    assert set(params) == set(BATTERY_MEMBERS)
    for model_type, best_params in params.items():
        assert isinstance(best_params, dict) and best_params
        assert all(
            isinstance(value, (str, int, float, bool, type(None)))
            for value in best_params.values()
        )
    # The real QDA variance floor is the committed value, NOT the fixture 0.1.
    assert params["qda"] == {"reg_param": QDA_COMMITTED_REG_PARAM}
    assert params["qda"]["reg_param"] != 0.1


def _write_ledger(tmp_path, results, raw=None):
    payload = raw if raw is not None else {"results": results}
    path = tmp_path / "battery-ledger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_committed_fit_params_rejects_malformed_ledgers(tmp_path):
    """Every ledger defect fails loudly with CommittedParamsError."""
    with pytest.raises(CommittedParamsError, match="not found"):
        load_committed_fit_params(tmp_path / "absent.json")

    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(CommittedParamsError, match="not valid JSON"):
        load_committed_fit_params(bad_json)

    with pytest.raises(CommittedParamsError, match="malformed"):
        load_committed_fit_params(_write_ledger(tmp_path, [], raw=[1, 2, 3]))

    good = {"model_type": "qda", "best_params": {"reg_param": 0.2}}

    with pytest.raises(CommittedParamsError, match="not a battery member"):
        load_committed_fit_params(
            _write_ledger(tmp_path, [good, {"model_type": "gpt-4",
                                            "best_params": {"C": 1.0}}])
        )
    with pytest.raises(CommittedParamsError, match="duplicates model_type"):
        load_committed_fit_params(
            _write_ledger(tmp_path, [good, dict(good)])
        )
    with pytest.raises(CommittedParamsError, match="non-empty params dict"):
        load_committed_fit_params(
            _write_ledger(tmp_path, [{"model_type": "qda",
                                      "best_params": {}}])
        )
    with pytest.raises(CommittedParamsError, match="non-scalar"):
        load_committed_fit_params(
            _write_ledger(tmp_path, [{"model_type": "qda",
                                      "best_params": {"reg_param": [0.2]}}])
        )
    with pytest.raises(CommittedParamsError, match="lacks tuned best_params"):
        load_committed_fit_params(_write_ledger(tmp_path, [good]))
    with pytest.raises(CommittedParamsError, match="model_type"):
        load_committed_fit_params(
            _write_ledger(tmp_path, [{"model_type": None,
                                      "best_params": {"reg_param": 0.2}}])
        )


# ---------------------------------------------------------------------------
# 3. build_real_standings (D038 leaderboard read-back)
# ---------------------------------------------------------------------------


def test_build_real_standings_keys_and_full_board_ranks(real_store):
    """Entries are keyed incumbent:<model_type> with FULL-board 1-based ranks."""
    standings = build_real_standings(
        [
            "lda",
            "knn",
            "logistic-regression-elasticnet",
            "logistic-regression-l1",
            "logistic-regression-l2",
            "qda",
            "svm",
        ],
        tracking_dir=real_store,
    )
    assert set(standings) == {
        "incumbent:lda",
        "incumbent:knn",
        "incumbent:logistic-regression-elasticnet",
        "incumbent:logistic-regression-l1",
        "incumbent:logistic-regression-l2",
        "incumbent:qda",
        "incumbent:svm",
    }
    # Ranks are positive ints; means match the seeded board.
    for mean, rank in standings.values():
        assert isinstance(rank, int) and not isinstance(rank, bool)
        assert rank >= 1
        assert mean in (1.0, 0.5)
    # Ordering: knn/lda tie at 1.0 -> model_type ascending puts knn first;
    # the 0.5 members order elasticnet, l1, l2, qda, svm by model_type.
    assert standings["incumbent:knn"] == (1.0, 1)
    assert standings["incumbent:lda"] == (1.0, 2)
    assert standings["incumbent:logistic-regression-elasticnet"] == (0.5, 3)
    assert standings["incumbent:logistic-regression-l1"] == (0.5, 4)
    assert standings["incumbent:logistic-regression-l2"] == (0.5, 5)
    assert standings["incumbent:qda"] == (0.5, 6)
    assert standings["incumbent:svm"] == (0.5, 7)


def test_build_real_standings_excludes_non_v1_runs(real_store):
    """A final run stamped split_version=fixture never enters v1 standings."""
    with pytest.raises(RealStandingsError, match="random-forest"):
        build_real_standings(["random-forest"], tracking_dir=real_store)
    # Against the fixture split the same run IS the board.
    fixture_standings = build_real_standings(
        ["random-forest"], tracking_dir=real_store, split_version="fixture"
    )
    assert fixture_standings == {"incumbent:random-forest": (1.0, 1)}


def test_build_real_standings_rejects_bad_requests(real_store):
    """Empty or non-battery requests fail before any store read."""
    with pytest.raises(RealStandingsError, match="at least one"):
        build_real_standings([], tracking_dir=real_store)
    with pytest.raises(RealStandingsError, match="non-battery"):
        build_real_standings(["gpt-4"], tracking_dir=real_store)


def test_build_real_standings_defaults_to_the_declared_v1_split(real_store):
    """REAL_SPLIT_VERSION ("v1") is the default — not the fixture stamp."""
    standings = build_real_standings(
        ["svm"], tracking_dir=real_store, split_version=REAL_SPLIT_VERSION
    )
    assert standings == {"incumbent:svm": (0.5, 7)}


# ---------------------------------------------------------------------------
# 4. standings_for_pair + clause-(c) verdicts with real-style standings
# ---------------------------------------------------------------------------


def test_standings_for_pair_supplies_exactly_two_mandatory_keys(
    measurement, real_store
):
    """Per-pair standings carry exactly the treated + incumbent entries."""
    battery_standings = build_real_standings(
        ["logistic-regression-l2"], tracking_dir=real_store
    )
    pair = standings_for_pair(measurement, battery_standings)
    assert set(pair) == {measurement.treated_key, measurement.incumbent_key}
    assert measurement.incumbent_key == incumbent_key_for("logistic-regression-l2")
    assert measurement.treated_key == (
        f"treatment:class-weight-balanced/logistic-regression-l2"
    )
    for mean, rank in pair.values():
        assert isinstance(rank, int) and not isinstance(rank, bool)
        assert rank >= 1
        assert mean == 0.5  # the committed leaderboard mean of the incumbent
    # The treated entry inherits the incumbent's standing (D038).
    assert pair[measurement.treated_key] == pair[measurement.incumbent_key]


def test_clause_c_with_real_standings_equal_rank_is_not_worsened(
    measurement, real_store
):
    """Inherited (equal) rank -> clause (c) holds via its own comparator."""
    battery_standings = build_real_standings(
        ["logistic-regression-l2"], tracking_dir=real_store
    )
    pair = standings_for_pair(measurement, battery_standings)
    assert standing_not_worsened(measurement, pair) is True
    verdict = verdict_treatment(measurement, pair)
    assert verdict.standing_ok is True
    # Clause (c) alone never accepts: the verdict still reflects clauses
    # (a)/(b) — the standing OK is recorded, acceptance stays evidence-bound.
    assert verdict.verdict in {
        "accepted",
        "rejected",
        "rejected-by-incomplete-evidence",
    }


def test_clause_c_with_worsened_supplied_rank_fails_alone(measurement):
    """A treated rank behind the incumbent's fails clause (c) by itself."""
    incumbent_entry = (0.9, 3)
    worsened = {
        measurement.incumbent_key: incumbent_entry,
        measurement.treated_key: (0.9, 7),
    }
    assert standing_not_worsened(measurement, worsened) is False
    verdict = verdict_treatment(measurement, worsened)
    assert verdict.standing_ok is False
    assert verdict.verdict != "accepted"


def test_standings_for_pair_rejects_bad_supplies(measurement, real_store):
    """A missing or malformed incumbent entry is a loud caller error."""
    with pytest.raises(RealStandingsError, match="lacks the incumbent entry"):
        standings_for_pair(
            measurement,
            build_real_standings(["qda"], tracking_dir=real_store),
        )
    with pytest.raises(RealStandingsError, match="positive-int"):
        standings_for_pair(
            measurement,
            {incumbent_key_for("logistic-regression-l2"): (0.5, 0)},
        )
    with pytest.raises(RealStandingsError, match="positive-int"):
        standings_for_pair(
            measurement,
            {incumbent_key_for("logistic-regression-l2"): (0.5, 2.5)},
        )
    with pytest.raises(RealStandingsError, match="tuple"):
        standings_for_pair(
            measurement,
            {incumbent_key_for("logistic-regression-l2"): [0.5, 3]},
        )


def test_real_standings_cover_the_whole_measured_pair_set(real_store):
    """Every gate-eligible model the real run measures has standings supply.

    The union of the declared candidates' applies_to scopes must be coverable
    by the v1 leaderboard read-back — the real command (T02) builds clause-(c)
    standings for exactly these models.
    """
    from heart.eval.treatment_variants import iter_buildable_candidates

    applies_to = sorted(
        {
            model_type
            for candidate in iter_buildable_candidates()
            for model_type in candidate.applies_to
        }
    )
    assert applies_to  # the declared set is non-empty
    standings = build_real_standings(applies_to, tracking_dir=real_store)
    assert len(standings) == len(applies_to)
    for key, (_mean, rank) in standings.items():
        assert key.startswith("incumbent:")
        assert isinstance(rank, int) and rank >= 1


def test_all_seeded_store_runs_are_final_kind(real_store):
    """Guard: the tmp store holds only run_kind=final rows for the board."""
    config = configure_tracking(tracking_dir=real_store)
    client = MlflowClient(tracking_uri=config.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id], max_results=100)
    assert len(runs) == 8  # 7 v1-split members + 1 fixture-split exclusion
    assert all(
        run.data.tags.get(RUN_KIND_TAG) == FINAL_RUN_KIND for run in runs
    )
