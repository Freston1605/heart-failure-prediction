"""Tests for the Optuna tuning runner and its MLflow trial logging (S03/T02).

Contracts under test:

1. **Study configuration** — :class:`heart.tuning.study.TuningConfig` validates
   every knob and :func:`create_study` builds a study with the configured
   direction, sampler, and pruner (in memory by default).
2. **Cross-validation folds** — :func:`heart.tuning.runner.build_cv_folds`
   partitions a training frame into group-aware, leakage-safe folds using the
   S01 splitter; every fold has both classes and the folds cover the frame.
3. **The objective** — :class:`ClassicalTuningObjective` scores each fold
   through the shared ``evaluate`` contract, reports the running mean primary
   metric for pruning, raises :class:`optuna.TrialPruned` when the pruner says
   so, and builds one canonical out-of-fold metric dict per completed trial.
4. **Recording every trial** — :func:`run_study` returns a :class:`TuningResult`
   that never drops a trial: completed, pruned, and failed trials all appear,
   with failed ones carrying their exception message. Completed trials are
   logged to the tuning experiment through the frozen MLflow convention;
   pruned/failed trials are logged as convention-tagged runs too.

Negative tests pin the named failure paths: invalid config values, unknown
samplers/pruners, malformed frames, empty folds, unknown model types, and a
study with no completed trial.

Every store-backed test runs against a fresh ``tmp_path`` SQLite database, so
the suite never touches the repository's ``experiments/mlruns`` directory.
Studies are in-memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.tree import DecisionTreeClassifier

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.eval.contract import METRIC_KEYS
from heart.models.registry import ModelSpec
from heart.models.spaces import (
    INT_KIND,
    ParamSpec,
    SearchSpace,
)
from heart.tracking.mlflow_store import configure_tracking
from heart.tracking.run import (
    METRICS_ARTIFACT,
    RUN_CONFIG_ARTIFACT,
    load_metrics_artifact,
    load_run,
)
from heart.tuning import (
    DEFAULT_TUNING_EXPERIMENT,
    ERROR_TAG,
    FINAL_RUN_KIND,
    RUN_KIND_FAILED,
    RUN_KIND_PRUNED,
    RUN_KIND_TAG,
    RUN_KIND_TRIAL,
    STUDY_NAME_TAG,
    TRIAL_STATE_TAG,
    ClassicalTuningObjective,
    NoCompletedTrialError,
    TuningConfig,
    TuningConfigError,
    TuningDataError,
    TuningFold,
    TuningLedgerError,
    TuningResult,
    UnknownPrunerError,
    UnknownSamplerError,
    build_best_pipeline,
    build_cv_folds,
    build_pruner,
    build_sampler,
    create_study,
    default_study_name,
    describe_tuning_config,
    run_study,
    suggest_params,
    tuning_ledger,
    write_tuning_ledger,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _schema_valid_frame(rows: int = 120, seed: int = 7) -> pd.DataFrame:
    """A small frame matching the declared schema, labels included."""
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


def _tree_space() -> SearchSpace:
    return SearchSpace(
        params=(
            ParamSpec(name="max_depth", kind=INT_KIND, low=1, high=4, default=2),
            ParamSpec(
                name="min_samples_leaf", kind=INT_KIND, low=1, high=5, default=1
            ),
        )
    )


def _tree_spec(model_type: str = "fixture-tree") -> ModelSpec:
    return ModelSpec(
        model_type=model_type,
        model_name="Fixture Tree",
        family="ensemble",
        space=_tree_space(),
        estimator_factory=lambda params: DecisionTreeClassifier(
            random_state=0, **dict(params)
        ),
        description="fixture decision tree",
    )


class _AlwaysFailFactory:
    """Callable estimator factory that always raises (a deterministic failure)."""

    def __init__(self, message: str = "always-boom") -> None:
        self.message = message
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        raise ValueError(self.message)


def _failing_spec(message: str = "always-boom") -> ModelSpec:
    return ModelSpec(
        model_type="fixture-failing",
        model_name="Fixture Failing",
        family="ensemble",
        space=_tree_space(),
        estimator_factory=_AlwaysFailFactory(message),
        description="fixture whose factory always raises",
    )


class _FlakyFactory:
    """Factory that raises on its first call and then succeeds (one bad trial)."""

    def __init__(self, message: str = "injected-first-trial-failure") -> None:
        self.message = message
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(self.message)
        return DecisionTreeClassifier(random_state=0, **dict(params))


def _flaky_spec() -> ModelSpec:
    return ModelSpec(
        model_type="fixture-flaky",
        model_name="Fixture Flaky",
        family="ensemble",
        space=_tree_space(),
        estimator_factory=_FlakyFactory(),
        description="fixture whose factory fails exactly once",
    )


class _AlwaysPrune(optuna.pruners.BasePruner):
    """Deterministic pruner: prune as soon as one intermediate value exists."""

    def prune(self, study, trial):
        return len(trial.intermediate_values) >= 1


@dataclass
class _FakeTrial:
    """Stand-in Optuna trial driving the objective deterministically."""

    number: int = 0
    params: dict = field(default_factory=dict)
    reports: list = field(default_factory=list)
    prune_when: bool = False
    attrs: dict = field(default_factory=dict)
    requested: list = field(default_factory=list)

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self.requested.append(name)
        return float(self.params[name])

    def suggest_int(self, name, low, high, *, step=None, log=False):
        self.requested.append(name)
        return int(self.params[name])

    def suggest_categorical(self, name, choices):
        self.requested.append(name)
        return self.params[name]

    def report(self, value, step):
        self.reports.append((float(value), int(step)))

    def should_prune(self):
        return self.prune_when and len(self.reports) >= 1

    def set_user_attr(self, key, value):
        self.attrs[key] = value


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _schema_valid_frame(rows=120, seed=7)


@pytest.fixture(scope="module")
def folds(frame) -> tuple[TuningFold, ...]:
    return build_cv_folds(frame, n_folds=3)


def _fast_config(**overrides) -> TuningConfig:
    base = dict(
        n_trials=3,
        sampler="random",
        sampler_seed=0,
        pruner="none",
        n_startup_trials=1,
    )
    base.update(overrides)
    return TuningConfig(**base)


def _row_keys(df: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, df[list(FEATURE_COLUMNS)].to_numpy().tolist()))


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_maximizes():
    config = TuningConfig()
    assert config.direction == "maximize"
    assert config.n_trials >= 1
    assert config.sampler == "tpe"
    assert config.pruner == "median"


def test_config_rejects_unknown_direction():
    with pytest.raises(TuningConfigError):
        TuningConfig(direction="sideways")


def test_config_rejects_unknown_sampler():
    with pytest.raises(UnknownSamplerError):
        TuningConfig(sampler="magic")


def test_config_rejects_unknown_pruner():
    with pytest.raises(UnknownPrunerError):
        TuningConfig(pruner="magic")


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_config_rejects_invalid_n_trials(value):
    with pytest.raises(TuningConfigError):
        TuningConfig(n_trials=value)


@pytest.mark.parametrize("value", [0, -3.0, True])
def test_config_rejects_invalid_timeout(value):
    with pytest.raises(TuningConfigError):
        TuningConfig(timeout=value)


@pytest.mark.parametrize("field_name", ["n_startup_trials", "n_warmup_steps"])
def test_config_rejects_negative_counts(field_name):
    with pytest.raises(TuningConfigError):
        TuningConfig(**{field_name: -1})


@pytest.mark.parametrize("value", [0, -2, True])
def test_config_rejects_nonpositive_interval(value):
    with pytest.raises(TuningConfigError):
        TuningConfig(interval_steps=value)


def test_config_rejects_blank_study_name():
    with pytest.raises(TuningConfigError):
        TuningConfig(study_name="   ")


def test_config_rejects_blank_storage():
    with pytest.raises(TuningConfigError):
        TuningConfig(storage="   ")


def test_config_rejects_non_bool_load_if_exists():
    with pytest.raises(TuningConfigError):
        TuningConfig(load_if_exists="yes")  # type: ignore[arg-type]


def test_config_serialises():
    payload = TuningConfig(n_trials=7).to_dict()
    assert payload["n_trials"] == 7
    assert payload["direction"] == "maximize"
    assert payload["sampler"] == "tpe"


def test_build_sampler_returns_configured_types():
    assert isinstance(build_sampler(TuningConfig(sampler="tpe")), optuna.samplers.TPESampler)
    assert isinstance(
        build_sampler(TuningConfig(sampler="random")), optuna.samplers.RandomSampler
    )


def test_build_pruner_returns_configured_types():
    assert isinstance(
        build_pruner(TuningConfig(pruner="median")), optuna.pruners.MedianPruner
    )
    assert isinstance(build_pruner(TuningConfig(pruner="none")), optuna.pruners.NopPruner)


def test_build_sampler_rejects_non_config():
    with pytest.raises(TuningConfigError):
        build_sampler("tpe")  # type: ignore[arg-type]


def test_default_study_name_uses_model_type():
    assert default_study_name(_tree_spec()) == "tune-fixture-tree"


def test_create_study_uses_config_direction_and_name():
    study = create_study(_tree_spec(), _fast_config())
    assert study.study_name == "tune-fixture-tree"
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert isinstance(study.pruner, optuna.pruners.NopPruner)


def test_create_study_accepts_sampler_and_pruner_overrides():
    sampler = optuna.samplers.RandomSampler(seed=1)
    pruner = _AlwaysPrune()
    study = create_study(_tree_spec(), _fast_config(), sampler=sampler, pruner=pruner)
    assert study.sampler is sampler
    assert study.pruner is pruner


def test_create_study_resolves_a_registry_model_type():
    study = create_study("naive-bayes", _fast_config(n_trials=1))
    assert study.study_name == "tune-naive-bayes"


def test_suggest_params_matches_the_declared_keys():
    spec = _tree_spec()
    trial = _FakeTrial(params={"max_depth": 3, "min_samples_leaf": 2})
    sampled = suggest_params(spec.space, trial)
    assert set(sampled) == set(spec.space.keys)
    assert trial.requested == list(spec.space.keys)


def test_suggest_params_rejects_non_space():
    with pytest.raises(Exception):
        suggest_params("not-a-space", _FakeTrial())  # type: ignore[arg-type]


def test_describe_tuning_config_names_the_settings():
    report = describe_tuning_config(_fast_config())
    assert "maximize" in report
    assert "random" in report
    assert "in-memory" in report


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------


def test_build_cv_folds_partitions_the_frame(frame):
    cv_folds = build_cv_folds(frame, n_folds=3)
    assert len(cv_folds) == 3
    assert sum(fold.n_eval for fold in cv_folds) == len(frame)

    all_keys = _row_keys(frame)
    seen: list[set[tuple]] = []
    for fold in cv_folds:
        assert fold.n_train + fold.n_eval == len(frame)
        assert set(fold.split.y_test) == {0, 1}
        assert set(fold.train_frame[TARGET_COLUMN]) == {0, 1}
        seen.append(all_keys - _row_keys(fold.train_frame))
    assert set().union(*seen) == all_keys
    for left in range(len(seen)):
        for right in range(left + 1, len(seen)):
            assert seen[left].isdisjoint(seen[right])


def test_build_cv_folds_rejects_missing_target(frame):
    without_target = frame.drop(columns=[TARGET_COLUMN])
    with pytest.raises(TuningDataError):
        build_cv_folds(without_target, n_folds=3)


def test_build_cv_folds_rejects_non_frame():
    with pytest.raises(TuningDataError):
        build_cv_folds("not-a-frame", n_folds=3)  # type: ignore[arg-type]


def test_build_cv_folds_rejects_too_few_folds(frame):
    with pytest.raises(TuningDataError):
        build_cv_folds(frame, n_folds=1)


def test_build_cv_folds_rejects_more_folds_than_rows():
    tiny = _schema_valid_frame(rows=5, seed=1)
    with pytest.raises(TuningDataError):
        build_cv_folds(tiny, n_folds=10)


def test_tuning_fold_rejects_an_empty_train_frame(folds):
    with pytest.raises(TuningDataError):
        TuningFold(fold=0, train_frame=folds[0].train_frame.iloc[0:0], split=folds[0].split)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def test_objective_scores_the_mean_and_records_oof_metrics(folds):
    spec = _tree_spec()
    objective = ClassicalTuningObjective(spec, folds)
    trial = _FakeTrial(number=0, params={"max_depth": 2, "min_samples_leaf": 2})
    value = objective(trial)

    assert 0.0 <= value <= 1.0
    assert set(trial.attrs) == {"fold_scores", "mean_score"}
    assert len(objective.fold_scores[0]) == len(folds)
    oof = objective.metric_dicts[0]
    assert set(oof.keys()) == set(METRIC_KEYS)
    assert oof["confusion_matrix"]["n_samples"] == len(folds[0].split.y_test) * len(folds)


def test_objective_reports_every_fold_for_pruning(folds):
    objective = ClassicalTuningObjective(_tree_spec(), folds)
    trial = _FakeTrial(params={"max_depth": 2, "min_samples_leaf": 1})
    objective(trial)
    assert [step for _, step in trial.reports] == list(range(len(folds)))


def test_objective_honours_pruning_and_records_partial_scores(folds):
    objective = ClassicalTuningObjective(_tree_spec(), folds)
    trial = _FakeTrial(params={"max_depth": 2, "min_samples_leaf": 1}, prune_when=True)
    with pytest.raises(optuna.TrialPruned):
        objective(trial)
    assert len(trial.reports) == 1
    assert len(objective.fold_scores[0]) == 1
    assert objective.metric_dicts == {}
    assert trial.attrs["folds_completed"] == 1


def test_objective_records_error_user_attr_and_reraises(folds):
    objective = ClassicalTuningObjective(_failing_spec("kaboom"), folds)
    trial = _FakeTrial(params={"max_depth": 2, "min_samples_leaf": 1})
    with pytest.raises(Exception):
        objective(trial)
    assert "kaboom" in trial.attrs["error"]


def test_objective_rejects_empty_folds():
    with pytest.raises(TuningDataError):
        ClassicalTuningObjective(_tree_spec(), [])


def test_objective_rejects_unknown_primary_metric(folds):
    with pytest.raises(TuningDataError):
        ClassicalTuningObjective(_tree_spec(), folds, primary_metric="not_a_metric")


# ---------------------------------------------------------------------------
# run_study — recording (no MLflow)
# ---------------------------------------------------------------------------


def test_run_study_records_every_trial(folds):
    result = run_study(_tree_spec(), folds, config=_fast_config(), log_to_mlflow=False)
    assert isinstance(result, TuningResult)
    assert result.n_trials == 3
    assert result.n_complete == 3
    assert result.n_failed == 0
    assert result.best_params is not None
    assert set(result.best_params) == set(_tree_space().keys)
    assert result.best_value == pytest.approx(
        max(record.value for record in result.trials)
    )
    assert result.metric_dict is not None
    assert set(result.metric_dict.keys()) == set(METRIC_KEYS)
    assert result.run_ids == ()


def test_run_study_records_a_flaky_failure_and_continues(folds):
    result = run_study(_flaky_spec(), folds, config=_fast_config(), log_to_mlflow=False)
    assert len(result.trials) == 3
    assert result.n_complete == 2
    assert result.n_failed == 1
    failed = [record for record in result.trials if record.state == "failed"]
    assert len(failed) == 1
    assert "injected-first-trial-failure" in failed[0].error
    assert failed[0].value is None


def test_run_study_records_all_failures_without_dropping_them(folds):
    result = run_study(_failing_spec(), folds, config=_fast_config(), log_to_mlflow=False)
    assert len(result.trials) == 3
    assert result.n_failed == 3
    assert result.n_complete == 0
    assert result.best_params is None
    assert result.best_value is None
    assert result.metric_dict is None
    assert all(record.state == "failed" for record in result.trials)
    assert all(record.error for record in result.trials)


def test_run_study_prunes_and_records_partial_fold_scores(folds):
    config = _fast_config(n_trials=2)
    result = run_study(
        _tree_spec(), folds, config=config, pruner=_AlwaysPrune(), log_to_mlflow=False
    )
    assert result.n_pruned == 2
    assert result.best_params is None
    for record in result.trials:
        assert record.state == "pruned"
        assert len(record.fold_scores) == 1


def test_run_study_accepts_a_registry_model_type_string(frame):
    small_folds = build_cv_folds(frame, n_folds=3)
    result = run_study(
        "naive-bayes", small_folds, config=_fast_config(n_trials=1), log_to_mlflow=False
    )
    assert result.model_type == "naive-bayes"
    assert result.n_complete == 1


def test_run_study_rejects_unknown_model_type(folds):
    with pytest.raises(Exception):
        run_study("not-a-model", folds, config=_fast_config(n_trials=1), log_to_mlflow=False)


def test_run_study_rejects_empty_folds():
    with pytest.raises(TuningDataError):
        run_study(_tree_spec(), [], config=_fast_config(n_trials=1), log_to_mlflow=False)


# ---------------------------------------------------------------------------
# Ledger and best-model hand-off
# ---------------------------------------------------------------------------


def test_run_study_writes_a_ledger(frame, tmp_path):
    result = run_study(
        _flaky_spec(),
        build_cv_folds(frame, n_folds=3),
        config=_fast_config(),
        log_to_mlflow=False,
        ledger_path=tmp_path / "ledger.json",
    )
    destination = tmp_path / "ledger.json"
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["n_trials"] == 3
    assert payload["n_failed"] == 1
    assert {trial["state"] for trial in payload["trials"]} == {"complete", "failed"}
    failed = next(t for t in payload["trials"] if t["state"] == "failed")
    assert "injected-first-trial-failure" in failed["error"]


def test_tuning_ledger_round_trips_through_json(folds):
    result = run_study(_tree_spec(), folds, config=_fast_config(), log_to_mlflow=False)
    payload = tuning_ledger(result)
    assert payload["model_type"] == "fixture-tree"
    assert len(payload["trials"]) == result.n_trials
    json.dumps(payload)  # must be JSON-serialisable


def test_write_tuning_ledger_rejects_a_non_result(tmp_path):
    with pytest.raises(TuningLedgerError):
        write_tuning_ledger({"not": "a result"}, tmp_path / "x.json")  # type: ignore[arg-type]


def test_build_best_pipeline_refits_the_best_params(frame, folds):
    spec = _tree_spec()
    result = run_study(spec, folds, config=_fast_config(), log_to_mlflow=False)
    pipeline = build_best_pipeline(result, spec, frame)
    probabilities = pipeline.predict_proba(frame[list(FEATURE_COLUMNS)])
    assert probabilities.shape == (len(frame), 2)
    assert set(pipeline.named_steps["clf"].get_params()) >= set(result.best_params)


def test_build_best_pipeline_raises_without_a_completed_trial(frame, folds):
    spec = _failing_spec()
    result = run_study(spec, folds, config=_fast_config(), log_to_mlflow=False)
    with pytest.raises(NoCompletedTrialError):
        build_best_pipeline(result, spec, frame)


# ---------------------------------------------------------------------------
# MLflow trial logging
# ---------------------------------------------------------------------------


def _tuning_runs(tracking_dir, name: str):
    config = configure_tracking(
        tracking_dir=tracking_dir, experiment_name=name
    )
    return config, MlflowClient().search_runs(experiment_ids=[config.experiment_id])


def test_run_study_logs_completed_trials_to_the_tuning_experiment(folds, tmp_path):
    result = run_study(
        _tree_spec(), folds, config=_fast_config(), tracking_dir=tmp_path / "mlruns"
    )
    assert result.experiment_name == DEFAULT_TUNING_EXPERIMENT
    assert len(result.run_ids) == result.n_complete

    _, runs = _tuning_runs(tmp_path / "mlruns", DEFAULT_TUNING_EXPERIMENT)
    assert len(runs) == result.n_complete
    for run in runs:
        assert run.data.tags[RUN_KIND_TAG] == RUN_KIND_TRIAL
        assert run.data.tags[TRIAL_STATE_TAG] == "complete"
        assert set(run.data.params) == set(_tree_space().keys)
        assert "roc_auc" in run.data.metrics
        assert run.data.tags[STUDY_NAME_TAG] == "tune-fixture-tree"

    record = load_run(result.run_ids[0])
    assert METRICS_ARTIFACT in record.artifact_paths
    assert RUN_CONFIG_ARTIFACT in record.artifact_paths
    assert set(load_metrics_artifact(result.run_ids[0]).keys()) == set(METRIC_KEYS)


def test_run_study_logs_a_failed_trial_run(folds, tmp_path):
    result = run_study(
        _failing_spec("logged-failure"),
        folds,
        config=_fast_config(n_trials=2),
        tracking_dir=tmp_path / "mlruns",
    )
    assert result.n_failed == 2

    _, runs = _tuning_runs(tmp_path / "mlruns", DEFAULT_TUNING_EXPERIMENT)
    assert len(runs) == 2
    for run in runs:
        assert run.data.tags[RUN_KIND_TAG] == RUN_KIND_FAILED
        assert run.data.tags[TRIAL_STATE_TAG] == "failed"
        assert "logged-failure" in run.data.tags[ERROR_TAG]
        assert set(run.data.params) == set(_tree_space().keys)


def test_run_study_logs_a_pruned_trial_run(folds, tmp_path):
    result = run_study(
        _tree_spec(),
        folds,
        config=_fast_config(n_trials=2),
        pruner=_AlwaysPrune(),
        tracking_dir=tmp_path / "mlruns",
    )
    assert result.n_pruned == 2

    _, runs = _tuning_runs(tmp_path / "mlruns", DEFAULT_TUNING_EXPERIMENT)
    assert len(runs) == 2
    for run in runs:
        assert run.data.tags[RUN_KIND_TAG] == RUN_KIND_PRUNED
        assert "fold_0_roc_auc" in run.data.metrics


def test_trial_run_names_follow_the_convention(folds, tmp_path):
    result = run_study(
        _tree_spec(), folds, config=_fast_config(n_trials=2), tracking_dir=tmp_path / "mlruns"
    )
    _, runs = _tuning_runs(tmp_path / "mlruns", DEFAULT_TUNING_EXPERIMENT)
    names = sorted(run.data.tags["mlflow.runName"] for run in runs)
    assert names == [
        "fixture-tree-v1-trial-000",
        "fixture-tree-v1-trial-001",
    ]
    assert len(result.run_ids) == 2


def test_run_study_can_skip_mlflow(folds):
    result = run_study(_tree_spec(), folds, config=_fast_config(), log_to_mlflow=False)
    assert result.run_ids == ()
    assert result.experiment_name is None


def test_final_run_kind_is_reserved_for_portfolio_runs(folds, tmp_path):
    # Tuning never writes a final run; T03/T04 filter on this tag.
    run_study(
        _tree_spec(), folds, config=_fast_config(n_trials=2), tracking_dir=tmp_path / "mlruns"
    )
    _, runs = _tuning_runs(tmp_path / "mlruns", DEFAULT_TUNING_EXPERIMENT)
    assert all(run.data.tags[RUN_KIND_TAG] != FINAL_RUN_KIND for run in runs)
