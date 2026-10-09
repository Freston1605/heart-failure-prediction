"""Tests for the full classical-battery runner and its MLflow logging (S03/T03).

Contracts under test:

1. **Selection and configuration** — :func:`heart.models.run_battery.select_specs`
   resolves the fixed battery (or an explicit subset/exclusion), and
   :class:`BatteryConfig` validates every knob.
2. **The shared protocol** — :func:`run_battery` tunes each selected member
   with the T02 runner, refits the best parameters on the full training frame,
   scores the held-out split through the shared ``evaluate`` contract, and
   reports a canonical metric dict per model.
3. **One final run per model** — running the full battery against a fresh
   MLflow store produces exactly ``BATTERY_SIZE`` final runs in the portfolio
   experiment, each tagged ``run_kind=final`` with its ``model_type`` and each
   carrying the complete metric dict as ``metrics.json``; trial runs land in
   the dedicated tuning experiment.
4. **Fail-soft recording** — a model that fails (missing dependency, no
   completed trial, unexpected error) is recorded with a categorised error and
   never dropped; strict mode raises :class:`BatteryRunError` with the result
   attached.

The full-battery MLflow proof uses a deterministic, cheap Optuna sampler
(declared below) so the smoke configuration still drives the real protocol
without paying for XGBoost's default tree budget. Every store-backed test runs
against a fresh ``tmp_path`` SQLite database, so the suite never touches the
repository's ``experiments/mlruns`` directory.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import optuna
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from heart.data.schema import TARGET_COLUMN
from heart.eval.contract import METRIC_KEYS
from heart.models.registry import (
    BATTERY_MODEL_TYPES,
    BATTERY_SIZE,
    ModelDependencyError,
)
from heart.models.run_battery import (
    BATTERY_EXPERIMENT,
    ERROR_CATEGORY_DEPENDENCY,
    ERROR_CATEGORY_NO_COMPLETED_TRIAL,
    ERROR_CATEGORY_TUNING,
    ERROR_CATEGORY_UNEXPECTED,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    BatteryConfig,
    BatteryConfigError,
    BatteryDataError,
    BatteryLedgerError,
    BatteryResult,
    BatteryRunError,
    NoModelsSelectedError,
    UnknownModelSelectionError,
    battery_ledger,
    build_parser,
    main,
    render_battery_summary,
    run_battery,
    select_specs,
    smoke_config,
    write_battery_ledger,
)
from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    TrackingConfig,
    configure_tracking,
)
from heart.tracking.run import METRICS_ARTIFACT, load_metrics_artifact, load_run
from heart.tuning.runner import (
    DEFAULT_TUNING_EXPERIMENT,
    FINAL_RUN_KIND,
    RUN_KIND_TAG,
    RUN_KIND_TRIAL,
    NoCompletedTrialError,
    TuningRunnerError,
)

run_battery_module = importlib.import_module("heart.models.run_battery")


# ---------------------------------------------------------------------------
# Cheap deterministic sampler
# ---------------------------------------------------------------------------


class _CheapSampler(optuna.samplers.BaseSampler):
    """Deterministic sampler: low ints, mid floats, first category.

    Low integer values keep tree counts (and thus the smoke run) cheap; middle
    float values keep regularisation (for example QDA's ``reg_param``) safely
    away from the singular-covariance boundary. This exercises the real Optuna
    pipeline without XGBoost's default tree budget.
    """

    def infer_relative_search_space(self, study, trial):
        # All battery search spaces are independent (no conditional params).
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, distribution):
        if isinstance(distribution, optuna.distributions.CategoricalDistribution):
            return distribution.choices[0]
        if isinstance(distribution, optuna.distributions.IntDistribution):
            return int(distribution.low)
        if isinstance(distribution, optuna.distributions.FloatDistribution):
            return float((distribution.low + distribution.high) / 2.0)
        raise ValueError(f"unsupported distribution {distribution!r}")


def _cheap_sampler() -> _CheapSampler:
    return _CheapSampler()


def _cheap_pruner() -> optuna.pruners.NopPruner:
    return optuna.pruners.NopPruner()


# ---------------------------------------------------------------------------
# Frames / config helpers
# ---------------------------------------------------------------------------


def _schema_valid_frame(rows: int, seed: int) -> pd.DataFrame:
    """A frame matching the declared schema, labels included."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(rows) < 0.55).astype(int)
    return pd.DataFrame(
        {
            "Age": rng.integers(29, 78, rows),
            "Sex": rng.choice(["F", "M"], rows),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], rows),
            "RestingBP": rng.integers(95, 190, rows),
            "Cholesterol": np.where(
                rng.random(rows) < 0.2, 0, rng.integers(120, 340, rows)
            ),
            "FastingBS": rng.integers(0, 2, rows),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], rows),
            "MaxHR": rng.integers(70, 200, rows),
            "ExerciseAngina": rng.choice(["N", "Y"], rows),
            "Oldpeak": rng.uniform(0.0, 4.0, rows).round(2),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], rows),
            TARGET_COLUMN: labels,
        }
    )


@pytest.fixture(scope="module")
def train_frame() -> pd.DataFrame:
    return _schema_valid_frame(rows=160, seed=7)


@pytest.fixture(scope="module")
def test_frame() -> pd.DataFrame:
    return _schema_valid_frame(rows=80, seed=11)


def _small_config(**overrides) -> BatteryConfig:
    base = smoke_config(n_trials=1, cv_folds=2)
    if overrides:
        return BatteryConfig(**{**base.to_dict(), **overrides})
    return base


def _run(train: pd.DataFrame, test: pd.DataFrame, **kwargs):
    """run_battery with the cheap deterministic sampler/pruner by default."""
    kwargs.setdefault("sampler", _cheap_sampler())
    kwargs.setdefault("pruner", _cheap_pruner())
    return run_battery(train, test, **kwargs)


def _client_for(config: TrackingConfig) -> MlflowClient:
    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri)
    return MlflowClient()


# ---------------------------------------------------------------------------
# Full-battery MLflow fixture (runs all 10 models once)
# ---------------------------------------------------------------------------


class _BatteryFixture:
    def __init__(self, result, tracking_dir, main_config, tuning_config, train, test):
        self.result = result
        self.tracking_dir = tracking_dir
        self.main_config = main_config
        self.tuning_config = tuning_config
        self.train = train
        self.test = test


@pytest.fixture(scope="module")
def battery_fixture(tmp_path_factory, train_frame, test_frame) -> _BatteryFixture:
    tracking_dir = tmp_path_factory.mktemp("battery-mlruns")
    result = run_battery(
        train_frame,
        test_frame,
        config=smoke_config(n_trials=1, cv_folds=2),
        tracking_dir=tracking_dir,
        split_version="smoke",
        battery_id="test-battery",
        sampler=_cheap_sampler(),
        pruner=_cheap_pruner(),
    )
    main_config = configure_tracking(
        tracking_dir=tracking_dir, experiment_name=BATTERY_EXPERIMENT
    )
    tuning_config = configure_tracking(
        tracking_dir=tracking_dir, experiment_name=DEFAULT_TUNING_EXPERIMENT
    )
    return _BatteryFixture(
        result=result,
        tracking_dir=tracking_dir,
        main_config=main_config,
        tuning_config=tuning_config,
        train=train_frame,
        test=test_frame,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_config_matches_declared_defaults():
    config = BatteryConfig()
    assert config.cv_folds >= 2
    assert config.n_trials >= 1
    assert config.sampler == "tpe"
    assert config.pruner == "median"
    assert config.to_tuning_config().n_trials == config.n_trials


def test_smoke_config_is_reduced_and_deterministic():
    config = smoke_config()
    assert config.n_trials < BatteryConfig().n_trials
    assert config.cv_folds >= 2
    assert config.sampler == "random"
    assert config.pruner == "none"
    assert config.to_dict()["n_trials"] == config.n_trials


@pytest.mark.parametrize("value", [0, 1, 1.5, True])
def test_config_rejects_invalid_cv_folds(value):
    with pytest.raises(BatteryConfigError):
        BatteryConfig(cv_folds=value)


@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_invalid_trials(value):
    with pytest.raises(BatteryConfigError):
        BatteryConfig(n_trials=value)


def test_config_rejects_unknown_sampler():
    with pytest.raises(BatteryConfigError):
        BatteryConfig(sampler="magic")


def test_config_rejects_unknown_pruner():
    with pytest.raises(BatteryConfigError):
        BatteryConfig(pruner="magic")


def test_config_serialises_round_trip():
    payload = BatteryConfig(n_trials=3, cv_folds=4).to_dict()
    assert payload["n_trials"] == 3
    assert payload["cv_folds"] == 4


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_select_specs_defaults_to_the_full_battery():
    specs = select_specs()
    assert len(specs) == BATTERY_SIZE
    assert tuple(spec.model_type for spec in specs) == BATTERY_MODEL_TYPES


def test_select_specs_subset_preserves_registry_order():
    specs = select_specs(["xgboost", "naive-bayes"])
    assert [spec.model_type for spec in specs] == ["naive-bayes", "xgboost"]


def test_select_specs_applies_exclusion():
    specs = select_specs(exclude=["xgboost", "svm"])
    names = {spec.model_type for spec in specs}
    assert "xgboost" not in names
    assert "svm" not in names
    assert len(specs) == BATTERY_SIZE - 2


def test_select_specs_rejects_unknown_model():
    with pytest.raises(UnknownModelSelectionError):
        select_specs(["not-a-model"])


def test_select_specs_rejects_a_bare_string():
    with pytest.raises(BatteryConfigError):
        select_specs("lda")  # type: ignore[arg-type]


def test_select_specs_rejects_a_blank_name():
    with pytest.raises(BatteryConfigError):
        select_specs([""])


def test_select_specs_empty_after_exclusion_raises():
    with pytest.raises(NoModelsSelectedError):
        select_specs(exclude=list(BATTERY_MODEL_TYPES))


# ---------------------------------------------------------------------------
# Input validation (negative surface)
# ---------------------------------------------------------------------------


def test_run_battery_rejects_non_frame_train(test_frame):
    with pytest.raises(BatteryDataError):
        _run("not-a-frame", test_frame, config=_small_config(), log_to_mlflow=False)


def test_run_battery_rejects_non_frame_test(train_frame):
    with pytest.raises(BatteryDataError):
        _run(train_frame, "not-a-frame", config=_small_config(), log_to_mlflow=False)


def test_run_battery_rejects_empty_train(test_frame):
    empty = _schema_valid_frame(rows=5, seed=1).iloc[0:0]
    with pytest.raises(BatteryDataError):
        _run(empty, test_frame, config=_small_config(), log_to_mlflow=False)


def test_run_battery_rejects_missing_target(train_frame, test_frame):
    with pytest.raises(BatteryDataError):
        _run(
            train_frame.drop(columns=[TARGET_COLUMN]),
            test_frame,
            config=_small_config(),
            log_to_mlflow=False,
        )


def test_run_battery_rejects_missing_test_target(train_frame, test_frame):
    with pytest.raises(BatteryDataError):
        _run(
            train_frame,
            test_frame.drop(columns=[TARGET_COLUMN]),
            config=_small_config(),
            log_to_mlflow=False,
        )


def test_run_battery_rejects_more_folds_than_rows(train_frame, test_frame):
    config = BatteryConfig(n_trials=1, cv_folds=len(train_frame) + 5)
    with pytest.raises(BatteryDataError):
        _run(train_frame, test_frame, config=config, log_to_mlflow=False)


# ---------------------------------------------------------------------------
# Workflow (no MLflow)
# ---------------------------------------------------------------------------


def test_run_battery_runs_a_subset_without_mlflow(train_frame, test_frame):
    result = _run(
        train_frame,
        test_frame,
        model_types=["naive-bayes", "logistic-regression-l2", "qda"],
        config=_small_config(),
        log_to_mlflow=False,
    )
    assert isinstance(result, BatteryResult)
    assert result.n_models == 3
    assert result.n_succeeded == 3
    assert result.n_failed == 0
    assert result.run_ids == ()
    for record in result.results:
        assert record.status == STATUS_SUCCEEDED
        assert record.metric_dict is not None
        assert set(record.metric_dict.keys()) == set(METRIC_KEYS)
        assert record.best_params is not None
        assert record.n_complete == 1
        assert record.error is None


def test_run_battery_records_primary_metric_for_each_model(train_frame, test_frame):
    result = _run(
        train_frame,
        test_frame,
        model_types=["lda", "knn"],
        config=_small_config(),
        log_to_mlflow=False,
    )
    for record in result.results:
        assert 0.0 <= record.primary_metric <= 1.0


def test_battery_result_to_dict_is_json_serialisable(train_frame, test_frame):
    result = _run(
        train_frame,
        test_frame,
        model_types=["qda"],
        config=_small_config(),
        log_to_mlflow=False,
    )
    payload = battery_ledger(result)
    json.dumps(payload)
    assert payload["n_models"] == 1
    assert payload["results"][0]["model_type"] == "qda"


# ---------------------------------------------------------------------------
# One final run per model — the MLflow proof
# ---------------------------------------------------------------------------


def test_full_battery_smoke_runs_every_registry_member(battery_fixture):
    result = battery_fixture.result
    assert result.n_models == BATTERY_SIZE
    assert result.n_succeeded == BATTERY_SIZE
    assert result.n_failed == 0
    assert set(result.failed_models) == set()
    assert result.battery_run_id == "test-battery"


def test_run_count_matches_registry_size(battery_fixture):
    client = _client_for(battery_fixture.main_config)
    runs = client.search_runs(
        experiment_ids=[battery_fixture.main_config.experiment_id]
    )
    assert len(runs) == BATTERY_SIZE


def test_every_final_run_is_tagged_and_complete(battery_fixture):
    client = _client_for(battery_fixture.main_config)
    runs = client.search_runs(
        experiment_ids=[battery_fixture.main_config.experiment_id]
    )
    tagged = set()
    for run in runs:
        assert run.data.tags[RUN_KIND_TAG] == FINAL_RUN_KIND
        tagged.add(run.data.tags["model_type"])
        record = load_run(run.info.run_id, config=battery_fixture.main_config)
        assert METRICS_ARTIFACT in record.artifact_paths
        metrics = load_metrics_artifact(
            run.info.run_id, config=battery_fixture.main_config
        )
        assert set(metrics.keys()) == set(METRIC_KEYS)
        assert 0.0 <= float(metrics["roc_auc"]) <= 1.0
    assert tagged == set(BATTERY_MODEL_TYPES)


def test_every_model_result_carries_the_complete_metric_dict(battery_fixture):
    for record in battery_fixture.result.results:
        assert record.run_id is not None
        assert record.metric_dict is not None
        assert set(record.metric_dict.keys()) == set(METRIC_KEYS)


def test_tuning_trials_are_logged_to_the_tuning_experiment(battery_fixture):
    client = _client_for(battery_fixture.tuning_config)
    runs = client.search_runs(
        experiment_ids=[battery_fixture.tuning_config.experiment_id]
    )
    assert len(runs) == BATTERY_SIZE  # one trial per model in the smoke config
    for run in runs:
        assert run.data.tags[RUN_KIND_TAG] == RUN_KIND_TRIAL


def test_final_runs_live_only_in_the_portfolio_experiment(battery_fixture):
    client = _client_for(battery_fixture.tuning_config)
    tuning_runs = client.search_runs(
        experiment_ids=[battery_fixture.tuning_config.experiment_id]
    )
    assert all(
        run.data.tags[RUN_KIND_TAG] != FINAL_RUN_KIND for run in tuning_runs
    )


def test_render_battery_summary_lists_every_model(battery_fixture):
    text = render_battery_summary(battery_fixture.result)
    for model_type in BATTERY_MODEL_TYPES:
        assert f"`{model_type}`" in text
    assert f"{BATTERY_SIZE}/{BATTERY_SIZE} models succeeded" in text


# ---------------------------------------------------------------------------
# Fail-soft recording
# ---------------------------------------------------------------------------


def test_run_battery_records_dependency_failure_and_continues(
    monkeypatch, train_frame, test_frame
):
    real_run_study = run_battery_module.run_study

    def fake_run_study(spec, folds, **kwargs):
        if spec.model_type == "xgboost":
            raise ModelDependencyError("xgboost missing (injected)")
        return real_run_study(spec, folds, **kwargs)

    monkeypatch.setattr(run_battery_module, "run_study", fake_run_study)

    result = _run(
        train_frame,
        test_frame,
        model_types=["naive-bayes", "xgboost"],
        config=_small_config(),
        log_to_mlflow=False,
    )
    assert result.n_models == 2
    assert result.n_succeeded == 1
    assert result.n_failed == 1
    failed = result.record_for("xgboost")
    assert failed.status == STATUS_FAILED
    assert failed.error_category == ERROR_CATEGORY_DEPENDENCY
    assert "xgboost missing" in failed.error
    assert failed.run_id is None
    assert result.record_for("naive-bayes").succeeded


def test_run_battery_records_no_completed_trial_failure(
    monkeypatch, train_frame, test_frame
):
    def fake_build(result, spec, frame):
        raise NoCompletedTrialError(f"no completed trial for {spec.model_type}")

    monkeypatch.setattr(run_battery_module, "build_best_pipeline", fake_build)

    result = _run(
        train_frame,
        test_frame,
        model_types=["lda"],
        config=_small_config(),
        log_to_mlflow=False,
    )
    record = result.record_for("lda")
    assert record.status == STATUS_FAILED
    assert record.error_category == ERROR_CATEGORY_NO_COMPLETED_TRIAL
    assert record.n_complete == 1  # tuning did complete; the refit hand-off failed


def test_run_battery_classifies_tuning_failure(monkeypatch, train_frame, test_frame):
    def fake_run_study(spec, folds, **kwargs):
        raise TuningRunnerError("study exploded (injected)")

    monkeypatch.setattr(run_battery_module, "run_study", fake_run_study)

    result = _run(
        train_frame,
        test_frame,
        model_types=["qda"],
        config=_small_config(),
        log_to_mlflow=False,
    )
    record = result.record_for("qda")
    assert record.status == STATUS_FAILED
    assert record.error_category == ERROR_CATEGORY_TUNING


def test_run_battery_records_every_model_when_all_fail(
    monkeypatch, train_frame, test_frame
):
    def fake_run_study(spec, folds, **kwargs):
        raise RuntimeError(f"boom-{spec.model_type}")

    monkeypatch.setattr(run_battery_module, "run_study", fake_run_study)

    result = _run(
        train_frame,
        test_frame,
        model_types=["naive-bayes", "lda", "qda"],
        config=_small_config(),
        log_to_mlflow=False,
    )
    assert result.n_models == 3
    assert result.n_succeeded == 0
    assert result.n_failed == 3
    assert all(
        record.error_category == ERROR_CATEGORY_UNEXPECTED
        for record in result.results
    )
    assert set(result.failed_models) == {"naive-bayes", "lda", "qda"}


def test_run_battery_strict_raises_with_the_result_attached(
    monkeypatch, train_frame, test_frame
):
    def fake_run_study(spec, folds, **kwargs):
        raise ModelDependencyError("injected failure")

    monkeypatch.setattr(run_battery_module, "run_study", fake_run_study)

    with pytest.raises(BatteryRunError) as excinfo:
        _run(
            train_frame,
            test_frame,
            model_types=["naive-bayes", "xgboost"],
            config=_small_config(),
            log_to_mlflow=False,
            strict=True,
        )
    error = excinfo.value
    assert error.result.n_models == 2
    assert error.result.n_failed == 2
    assert set(error.failed_models) == {"naive-bayes", "xgboost"}


def test_run_battery_strict_does_not_raise_when_all_succeed(
    train_frame, test_frame
):
    result = _run(
        train_frame,
        test_frame,
        model_types=["naive-bayes"],
        config=_small_config(),
        log_to_mlflow=False,
        strict=True,
    )
    assert result.n_succeeded == 1


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def test_run_battery_writes_a_ledger(tmp_path, train_frame, test_frame):
    destination = tmp_path / "nested" / "battery_ledger.json"
    result = _run(
        train_frame,
        test_frame,
        model_types=["qda"],
        config=_small_config(),
        log_to_mlflow=False,
        ledger_path=destination,
        battery_id="ledger-test",
    )
    assert result.ledger_path == destination
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["battery_run_id"] == "ledger-test"
    assert payload["n_models"] == 1
    assert payload["results"][0]["model_type"] == "qda"
    assert set(payload["results"][0]["metric_dict"].keys()) == set(METRIC_KEYS)


def test_battery_ledger_rejects_a_non_result():
    with pytest.raises(BatteryLedgerError):
        battery_ledger({"not": "a result"})  # type: ignore[arg-type]


def test_write_battery_ledger_rejects_a_non_result(tmp_path):
    with pytest.raises(BatteryLedgerError):
        write_battery_ledger("nope", tmp_path / "x.json")  # type: ignore[arg-type]


def test_render_battery_summary_rejects_a_non_result():
    with pytest.raises(BatteryLedgerError):
        render_battery_summary({"not": "a result"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.smoke is False
    assert args.trials is None
    assert args.folds is None
    assert args.experiment == DEFAULT_EXPERIMENT


def test_main_rejects_a_missing_split(capsys):
    rc = main(["--split-version", "does-not-exist", "--no-mlflow"])
    assert rc == 1
    assert "could not load split" in capsys.readouterr().out


def test_main_rejects_an_invalid_config(capsys):
    rc = main(["--no-mlflow", "--folds", "1"])
    assert rc == 2
    assert "configuration error" in capsys.readouterr().out


def test_main_runs_a_smoke_subset(capsys):
    rc = main(
        [
            "--smoke",
            "--models",
            "naive-bayes,lda",
            "--folds",
            "2",
            "--trials",
            "1",
            "--no-mlflow",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "2/2 models succeeded" in out
    assert "`naive-bayes`" in out
    assert "`lda`" in out


def test_main_returns_nonzero_when_every_model_fails(monkeypatch, capsys):
    def fake_run_study(spec, folds, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_battery_module, "run_study", fake_run_study)
    rc = main(["--models", "naive-bayes", "--folds", "2", "--trials", "1", "--no-mlflow"])
    assert rc == 1
    assert "0/1 models succeeded" in capsys.readouterr().out
