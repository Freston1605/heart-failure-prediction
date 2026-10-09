"""Tests for the tuned MLP architecture sweep and its MLflow protocol (S05/T03).

Contracts under test:

1. **The deliberate architecture set** — :mod:`heart.models.mlp_spaces`
   declares a small, fixed set of MLP architectures, each valid, with distinct
   capacities and a shared tuned training space whose keys are exactly
   :data:`~heart.models.mlp_spaces.TUNED_PARAM_NAMES`.
2. **Reuse of the classical runner** — every architecture converts to a
   :class:`~heart.models.registry.ModelSpec` (``family=neural``,
   ``preprocessing=False``) whose estimator is
   :class:`~heart.models.mlp_spaces.TunableMLPClassifier`;
   :class:`~heart.models.run_mlp_sweep.MLPTuningObjective` swaps only the
   per-fold scoring leaf, and :func:`heart.tuning.runner.run_study` drives the
   neural trial through the identical fold loop, pruning, out-of-fold metric
   dict, and MLflow trial recording used by the classical battery (D013).
3. **The MLflow proof** — a reduced-trial smoke sweep produces exactly one
   final run per configured architecture in the portfolio experiment (the run
   count matches the architecture set), each carrying its params, the complete
   metric dict, and the compute-device tags (``device`` / ``gpu_available`` /
   ``cpu_fallback``); trial runs land in the dedicated tuning experiment.
4. **The logged CPU fallback and no-GPU host path** — the production trainer
   falls back to CPU with an explicit warning when the GPU is unavailable; on
   a host without torch the sweep records every architecture as failed with a
   categorised error (torch reason surfaced) rather than crashing.
5. **Fail-soft and negative surfaces** — invalid selections, configs, frames,
   and non-result ledger writes all raise named errors; strict mode attaches
   the result.

Host tests require no torch: the full-sweep MLflow proof injects a
deterministic trainer/refit stand-in, exactly as T02 injects a fake torch for
device resolution. The one real-torch end-to-end test uses
``pytest.importorskip`` and runs inside the S05/T01 ROCm container.

Every store-backed test runs against a fresh ``tmp_path`` SQLite database, so
the suite never touches the repository's ``experiments/mlruns`` directory.
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
from heart.eval.metrics import compute_metric_dict
from heart.models.mlp import MLPConfig
from heart.models.mlp_spaces import (
    FAMILY_NEURAL,
    MLP_ARCHITECTURES,
    MLP_MODEL_TYPES,
    MLP_SWEEP_SIZE,
    MLPSpaceError,
    UnknownMLPArchitectureError,
    describe_mlp_sweep_plan,
    fixed_training_params,
    mlp_trainer,
    mlp_tuning_space,
    select_mlp_architectures,
    to_model_spec,
)
from heart.models.registry import build_estimator
from heart.models.run_mlp_sweep import (
    DEFAULT_TUNING_EXPERIMENT,
    ERROR_CATEGORY_NO_COMPLETED_TRIAL,
    MLPTuningObjective,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    SWEEP_EXPERIMENT,
    SweepConfig,
    SweepConfigError,
    SweepDataError,
    SweepLedgerError,
    SweepModelResult,
    SweepResult,
    SweepRunError,
    build_parser,
    main,
    render_sweep_summary,
    run_mlp_sweep,
    smoke_sweep_config,
    sweep_ledger,
    write_sweep_ledger,
)
from heart.tracking.mlflow_store import configure_tracking
from heart.tracking.run import METRICS_ARTIFACT, load_metrics_artifact, load_run
from heart.models.train_torch import DEVICE_CPU, DeviceResolution
from heart.tuning.runner import (
    FINAL_RUN_KIND,
    RUN_KIND_FAILED,
    RUN_KIND_TAG,
    RUN_KIND_TRIAL,
    NoCompletedTrialError,
    TuningFold,
    build_cv_folds,
    run_study,
)
from heart.tuning.study import TuningConfig

run_mlp_sweep_module = importlib.import_module("heart.models.run_mlp_sweep")

try:
    import torch  # noqa: F401

    _TORCH_IMPORTABLE = True
except Exception:  # pragma: no cover - exercised on torch-less hosts
    _TORCH_IMPORTABLE = False


# ---------------------------------------------------------------------------
# Deterministic sampler / pruner
# ---------------------------------------------------------------------------


class _CheapSampler(optuna.samplers.BaseSampler):
    """Deterministic sampler: low ints, mid floats, first category."""

    def infer_relative_search_space(self, study, trial):
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
# Frames / synthetic scorer stand-ins
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


def _synthetic_trainer(seed: int = 0):
    """Deterministic scorer: no torch, schema-valid metrics per fold."""

    def trainer(params, fold):
        n = fold.n_eval
        rng = np.random.default_rng(seed + int(fold.fold))
        labels = np.asarray(fold.split.y_test)
        proba = rng.random(n)
        predictions = (proba >= 0.5).astype(int)
        metrics = compute_metric_dict(labels, proba, predictions)
        return metrics, predictions, proba

    return trainer


def _stub_refit_factory():
    """Refit stand-in mirroring ``build_best_pipeline``'s contract.

    The production hand-off refuses to refit a study without a completed
    trial (:class:`NoCompletedTrialError`); the stand-in reproduces that check
    before returning a fitted stub carrying a recorded device, so the sweep's
    fail-soft path is exercised identically in tests.
    """

    def factory(tuning_result, spec, train_frame):
        if tuning_result.best_params is None:
            raise NoCompletedTrialError(
                f"Study {tuning_result.study_name!r} for {spec.model_type!r} "
                "has no completed trial (stand-in)."
            )
        resolution = DeviceResolution(
            requested="gpu",
            device=DEVICE_CPU,
            is_gpu=False,
            gpu_available=False,
            torch_available=True,
            reason="synthetic stand-in (CPU)",
            warnings=(),
        )
        return _StubFitted(resolution)

    return factory


class _StubFitted:
    """A fitted model exposing predict/predict_proba and the device."""

    def __init__(self, resolution: DeviceResolution) -> None:
        self.device_resolution = resolution
        self.device = resolution.device

    def predict_proba(self, X: object) -> np.ndarray:
        n = len(X)
        proba = np.random.default_rng(0).random(n)  # varying: valid roc_auc
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: object) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _client_for(config) -> MlflowClient:
    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri)
    return MlflowClient()


# ---------------------------------------------------------------------------
# The deliberate architecture set (torch-free)
# ---------------------------------------------------------------------------


def test_architecture_set_is_small_and_deliberate():
    assert MLP_SWEEP_SIZE == 3
    assert len(MLP_ARCHITECTURES) == MLP_SWEEP_SIZE
    assert tuple(spec.model_type for spec in MLP_ARCHITECTURES) == MLP_MODEL_TYPES
    # three distinct capacity regimes, not an exhaustive search
    capacities = {spec.hidden_sizes for spec in MLP_ARCHITECTURES}
    assert len(capacities) == MLP_SWEEP_SIZE
    # every architecture is a valid MLPConfig shape
    for spec in MLP_ARCHITECTURES:
        MLPConfig(
            input_dim=1,
            hidden_sizes=spec.hidden_sizes,
            dropout=spec.dropout_default,
            activation=spec.activation,
            batch_norm=spec.batch_norm,
        )
    assert describe_mlp_sweep_plan().startswith("neural sweep: 3 configured")


def test_tuning_space_keys_match_the_declared_tuned_params():
    space = mlp_tuning_space()
    assert space.keys == ("learning_rate", "weight_decay", "dropout", "batch_size")
    assert tuple(space.keys) == (
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
    )
    # defaults fall inside the declared regions
    defaults = space.default_params()
    assert 1e-4 <= defaults["learning_rate"] <= 1e-2
    assert 0.0 <= defaults["dropout"] <= 0.5
    assert defaults["batch_size"] in (16, 32, 64)


def test_architecture_spec_converts_to_runner_spec():
    for arch in MLP_ARCHITECTURES:
        spec = to_model_spec(arch)
        assert spec.model_type == arch.model_type
        assert spec.family == FAMILY_NEURAL
        assert spec.preprocessing is False
        assert set(spec.declared_hyperparameters) == {
            "learning_rate",
            "weight_decay",
            "dropout",
            "batch_size",
        }
        # architecture + fixed training knobs are pinned, never tuned
        fixed = dict(spec.fixed_params)
        assert fixed["hidden_sizes"] == list(arch.hidden_sizes)
        assert fixed["activation"] == arch.activation
        assert fixed["batch_norm"] == arch.batch_norm
        assert "epochs" in fixed and "seed" in fixed and "pos_weight" in fixed
        # the plain defaults construct a real (unfitted) adapter
        estimator = build_estimator(spec)
        assert hasattr(estimator, "fit") and hasattr(estimator, "predict_proba")
        assert estimator.hidden_sizes == tuple(arch.hidden_sizes)


def test_fixed_training_params_are_stable():
    params = fixed_training_params()
    assert params["seed"] == 42
    assert params["epochs"] >= 1
    assert 0.0 <= params["validation_fraction"] < 0.5


def test_mlp_trainer_rejects_non_bool_prefer_gpu():
    with pytest.raises(MLPSpaceError, match="prefer_gpu"):
        mlp_trainer(prefer_gpu="yes")  # type: ignore[arg-type]


def test_select_architectures_subset_preserves_declared_order():
    selected = select_mlp_architectures(["mlp-deep", "mlp-shallow"])
    assert [spec.model_type for spec in selected] == ["mlp-shallow", "mlp-deep"]


def test_select_architectures_rejects_unknown():
    with pytest.raises(UnknownMLPArchitectureError, match="mlp-giant"):
        select_mlp_architectures(["mlp-giant"])


def test_select_architectures_rejects_a_bare_string():
    with pytest.raises(MLPSpaceError, match="sequence"):
        select_mlp_architectures("mlp-shallow")  # type: ignore[arg-type]


def test_select_architectures_empty_raises():
    with pytest.raises(MLPSpaceError, match="empty"):
        select_mlp_architectures([])


# ---------------------------------------------------------------------------
# The neural tuning objective (torch-free)
# ---------------------------------------------------------------------------


def test_mlp_objective_merges_fixed_and_tuned_params(folds):
    spec = to_model_spec(MLP_ARCHITECTURES[0])
    captured: dict[str, object] = {}

    def recorder(params, fold):
        captured["merged"] = dict(params)
        n = fold.n_eval
        rng = np.random.default_rng(0)
        labels = np.asarray(fold.split.y_test)
        proba = rng.random(n)
        predictions = (proba >= 0.5).astype(int)
        return compute_metric_dict(labels, proba, predictions), predictions, proba

    objective = MLPTuningObjective(spec, folds, trainer=recorder)
    defaults = spec.space.default_params()
    trial = _FakeTrial(defaults)
    value = objective(trial)
    assert 0.0 <= value <= 1.0
    merged = captured["merged"]
    # fixed architecture knobs arrive alongside the tuned training knobs
    assert merged["hidden_sizes"] == list(MLP_ARCHITECTURES[0].hidden_sizes)
    assert "learning_rate" in merged and "batch_size" in merged
    assert len(objective.fold_scores[0]) == len(folds)
    assert set(objective.metric_dicts[0].keys()) == set(METRIC_KEYS)


def test_run_study_with_mlp_objective_logs_complete_trials(folds, tmp_path):
    spec = to_model_spec(MLP_ARCHITECTURES[0])
    config = TuningConfig(
        n_trials=1, sampler="random", sampler_seed=0, pruner="none"
    )
    result = run_study(
        spec,
        folds,
        config=config,
        objective=MLPTuningObjective(spec, folds, trainer=_synthetic_trainer()),
        tracking_dir=tmp_path,
        split_version="smoke",
    )
    assert result.n_trials == 1
    assert result.n_complete == 1
    assert result.n_failed == 0
    assert result.best_params is not None
    assert set(result.metric_dict.keys()) == set(METRIC_KEYS)
    assert len(result.run_ids) == 1
    # the trial run follows the classical convention: params and the complete
    # metric dict logged as metrics.json under the tuning experiment.
    tuning_config = configure_tracking(
        tracking_dir=tmp_path, experiment_name=DEFAULT_TUNING_EXPERIMENT
    )
    record = load_run(result.run_ids[0], config=tuning_config)
    assert record.tags[RUN_KIND_TAG] == RUN_KIND_TRIAL
    assert "learning_rate" in record.params
    assert "hidden_sizes" in record.params
    payload = load_metrics_artifact(result.run_ids[0], config=tuning_config)
    assert set(payload.keys()) == set(METRIC_KEYS)


# ---------------------------------------------------------------------------
# Full sweep — the MLflow proof (uses deterministic stand-ins, no torch)
# ---------------------------------------------------------------------------


class _SweepFixture:
    def __init__(self, result, tracking_dir, main_config, tuning_config):
        self.result = result
        self.tracking_dir = tracking_dir
        self.main_config = main_config
        self.tuning_config = tuning_config


@pytest.fixture(scope="module")
def sweep_fixture(
    tmp_path_factory, train_frame, test_frame
) -> _SweepFixture:
    tracking_dir = tmp_path_factory.mktemp("mlp-sweep-mlruns")
    result = run_mlp_sweep(
        train_frame,
        test_frame,
        config=smoke_sweep_config(n_trials=1, cv_folds=2),
        trainer=_synthetic_trainer(),
        refit_factory=_stub_refit_factory(),
        tracking_dir=tracking_dir,
        split_version="smoke",
        sweep_id="test-sweep",
        sampler=_cheap_sampler(),
        pruner=_cheap_pruner(),
    )
    main_config = configure_tracking(
        tracking_dir=tracking_dir, experiment_name=SWEEP_EXPERIMENT
    )
    tuning_config = configure_tracking(
        tracking_dir=tracking_dir, experiment_name=DEFAULT_TUNING_EXPERIMENT
    )
    return _SweepFixture(
        result=result,
        tracking_dir=tracking_dir,
        main_config=main_config,
        tuning_config=tuning_config,
    )


def test_smoke_sweep_runs_every_configured_architecture(sweep_fixture):
    result = sweep_fixture.result
    assert result.n_models == MLP_SWEEP_SIZE
    assert result.n_succeeded == MLP_SWEEP_SIZE
    assert result.n_failed == 0
    assert result.sweep_run_id == "test-sweep"
    assert set(record.model_type for record in result.results) == set(MLP_MODEL_TYPES)


def test_run_count_matches_the_configured_architecture_set(sweep_fixture):
    client = _client_for(sweep_fixture.main_config)
    runs = client.search_runs(
        experiment_ids=[sweep_fixture.main_config.experiment_id]
    )
    assert len(runs) == MLP_SWEEP_SIZE


def test_every_final_run_is_tagged_and_carries_params_and_metrics(sweep_fixture):
    client = _client_for(sweep_fixture.main_config)
    runs = client.search_runs(
        experiment_ids=[sweep_fixture.main_config.experiment_id]
    )
    tagged = set()
    for run in runs:
        assert run.data.tags[RUN_KIND_TAG] == FINAL_RUN_KIND
        tagged.add(run.data.tags["model_type"])
        record = load_run(run.info.run_id, config=sweep_fixture.main_config)
        assert METRICS_ARTIFACT in record.artifact_paths
        assert "learning_rate" in record.params
        assert "hidden_sizes" in record.params
        metrics = load_metrics_artifact(
            run.info.run_id, config=sweep_fixture.main_config
        )
        assert set(metrics.keys()) == set(METRIC_KEYS)
        assert 0.0 <= float(metrics["roc_auc"]) <= 1.0
    assert tagged == set(MLP_MODEL_TYPES)


def test_final_runs_record_the_compute_device(sweep_fixture):
    client = _client_for(sweep_fixture.main_config)
    runs = client.search_runs(
        experiment_ids=[sweep_fixture.main_config.experiment_id]
    )
    for run in runs:
        assert run.data.tags["device"] in ("cuda:0", DEVICE_CPU)
        assert run.data.tags["gpu_available"] in ("true", "false")
        assert run.data.tags["cpu_fallback"] in ("true", "false")
    for record in sweep_fixture.result.results:
        assert record.device == DEVICE_CPU
        assert record.device_reason is not None


def test_every_result_carries_the_complete_metric_dict(sweep_fixture):
    for record in sweep_fixture.result.results:
        assert record.run_id is not None
        assert record.metric_dict is not None
        assert set(record.metric_dict.keys()) == set(METRIC_KEYS)


def test_trial_runs_logged_under_the_tuning_experiment(sweep_fixture):
    client = _client_for(sweep_fixture.tuning_config)
    runs = client.search_runs(
        experiment_ids=[sweep_fixture.tuning_config.experiment_id]
    )
    assert len(runs) == MLP_SWEEP_SIZE  # one trial per architecture, smoke
    assert all(run.data.tags[RUN_KIND_TAG] == RUN_KIND_TRIAL for run in runs)


def test_final_runs_live_only_in_the_portfolio_experiment(sweep_fixture):
    client = _client_for(sweep_fixture.tuning_config)
    runs = client.search_runs(
        experiment_ids=[sweep_fixture.tuning_config.experiment_id]
    )
    assert all(run.data.tags[RUN_KIND_TAG] != FINAL_RUN_KIND for run in runs)


def test_run_mlp_sweep_without_mlflow_still_records_results(train_frame, test_frame):
    result = run_mlp_sweep(
        train_frame,
        test_frame,
        architectures=["mlp-shallow"],
        config=smoke_sweep_config(n_trials=1, cv_folds=2),
        trainer=_synthetic_trainer(),
        refit_factory=_stub_refit_factory(),
        log_to_mlflow=False,
    )
    assert result.n_models == 1
    assert result.n_succeeded == 1
    assert result.run_ids == ()


# ---------------------------------------------------------------------------
# Fail-soft recording — the production no-torch host path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _TORCH_IMPORTABLE,
    reason="asserts genuine no-torch fail-soft semantics; on torch-bearing hosts "
    "the sweep succeeds instead — real-torch behavior is covered by "
    "test_smoke_sweep_end_to_end_with_real_torch",
)
def test_sweep_records_torch_unavailable_as_failure(tmp_path, train_frame, test_frame):
    """On a host without torch every architecture fails softly, never crashes."""
    result = run_mlp_sweep(
        train_frame,
        test_frame,
        config=smoke_sweep_config(n_trials=1, cv_folds=2),
        tracking_dir=tmp_path,
        split_version="smoke",
        sweep_id="no-torch-sweep",
        sampler=_cheap_sampler(),
        pruner=_cheap_pruner(),
    )
    assert result.n_models == MLP_SWEEP_SIZE
    assert result.n_succeeded == 0
    assert result.n_failed == MLP_SWEEP_SIZE
    for record in result.results:
        assert record.status == STATUS_FAILED
        assert record.error_category == ERROR_CATEGORY_NO_COMPLETED_TRIAL
        assert "torch" in record.error.lower()
    # the failed trials are still recorded under the classical convention
    tuning_config = configure_tracking(
        tracking_dir=tmp_path, experiment_name=DEFAULT_TUNING_EXPERIMENT
    )
    client = _client_for(tuning_config)
    runs = client.search_runs(experiment_ids=[tuning_config.experiment_id])
    assert len(runs) == MLP_SWEEP_SIZE  # one failed trial per architecture
    assert all(run.data.tags[RUN_KIND_TAG] == RUN_KIND_FAILED for run in runs)


def test_strict_mode_raises_with_the_result_attached(train_frame, test_frame):
    def _failing_trainer(params, fold):
        raise RuntimeError("injected trainer failure")

    with pytest.raises(SweepRunError) as excinfo:
        run_mlp_sweep(
            train_frame,
            test_frame,
            architectures=["mlp-shallow"],
            config=smoke_sweep_config(n_trials=1, cv_folds=2),
            trainer=_failing_trainer,
            refit_factory=_stub_refit_factory(),
            log_to_mlflow=False,
            strict=True,
        )
    error = excinfo.value
    assert error.result.n_models == 1
    assert error.result.n_failed == 1
    assert error.failed_architectures == ("mlp-shallow",)


# ---------------------------------------------------------------------------
# Input validation / config / ledger
# ---------------------------------------------------------------------------


def test_run_mlp_sweep_rejects_malformed_frames(test_frame, train_frame):
    with pytest.raises(SweepDataError):
        run_mlp_sweep("not-a-frame", test_frame, log_to_mlflow=False)
    with pytest.raises(SweepDataError):
        run_mlp_sweep(train_frame, "not-a-frame", log_to_mlflow=False)
    empty = _schema_valid_frame(rows=5, seed=1).iloc[0:0]
    with pytest.raises(SweepDataError):
        run_mlp_sweep(empty, test_frame, log_to_mlflow=False)
    with pytest.raises(SweepDataError):
        run_mlp_sweep(
            train_frame.drop(columns=[TARGET_COLUMN]),
            test_frame,
            log_to_mlflow=False,
        )


def test_run_mlp_sweep_rejects_more_folds_than_rows(train_frame, test_frame):
    config = SweepConfig(n_trials=1, cv_folds=len(train_frame) + 5)
    with pytest.raises(SweepDataError):
        run_mlp_sweep(train_frame, test_frame, config=config, log_to_mlflow=False)


@pytest.mark.parametrize("value", [0, 1, 1.5, True])
def test_sweep_config_rejects_invalid_cv_folds(value):
    with pytest.raises(SweepConfigError):
        SweepConfig(cv_folds=value)


@pytest.mark.parametrize("value", [0, -1, True])
def test_sweep_config_rejects_invalid_smoke_epochs(value):
    with pytest.raises(SweepConfigError):
        SweepConfig(smoke_epochs=value)


def test_sweep_config_rejects_invalid_sampler_and_pruner():
    with pytest.raises(SweepConfigError):
        SweepConfig(sampler="magic")
    with pytest.raises(SweepConfigError):
        SweepConfig(pruner="magic")


def test_smoke_sweep_config_is_reduced_and_records_epoch_override():
    config = smoke_sweep_config()
    assert config.n_trials == 2
    assert config.cv_folds == 3
    assert config.smoke_epochs == 3
    assert config.sampler == "random"
    assert config.pruner == "none"


def test_sweep_ledger_round_trip_persists_device(tmp_path, train_frame, test_frame):
    destination = tmp_path / "ledger.json"
    result = run_mlp_sweep(
        train_frame,
        test_frame,
        architectures=["mlp-shallow"],
        config=smoke_sweep_config(n_trials=1, cv_folds=2),
        trainer=_synthetic_trainer(),
        refit_factory=_stub_refit_factory(),
        log_to_mlflow=False,
        ledger_path=destination,
        sweep_id="ledger-test",
    )
    assert result.ledger_path == destination
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["sweep_run_id"] == "ledger-test"
    assert payload["n_models"] == 1
    record = payload["results"][0]
    assert record["model_type"] == "mlp-shallow"
    assert record["device"] == DEVICE_CPU
    assert set(record["metric_dict"].keys()) == set(METRIC_KEYS)


def test_sweep_ledger_rejects_a_non_result():
    with pytest.raises(SweepLedgerError):
        sweep_ledger({"not": "a result"})  # type: ignore[arg-type]


def test_write_sweep_ledger_rejects_a_non_result(tmp_path):
    with pytest.raises(SweepLedgerError):
        write_sweep_ledger("nope", tmp_path / "x.json")  # type: ignore[arg-type]


def test_render_sweep_summary_lists_every_architecture(sweep_fixture):
    text = render_sweep_summary(sweep_fixture.result)
    for model_type in MLP_MODEL_TYPES:
        assert f"`{model_type}`" in text
    assert f"{MLP_SWEEP_SIZE}/{MLP_SWEEP_SIZE} architectures succeeded" in text


def test_render_sweep_summary_rejects_a_non_result():
    with pytest.raises(SweepLedgerError):
        render_sweep_summary({"not": "a result"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.smoke is False
    assert args.trials is None
    assert args.folds is None
    assert args.cpu is False
    assert args.experiment == SWEEP_EXPERIMENT


def test_main_rejects_a_missing_split(capsys):
    rc = main(["--split-version", "does-not-exist", "--no-mlflow"])
    assert rc == 1
    assert "could not load split" in capsys.readouterr().out


def test_main_rejects_an_invalid_config(capsys):
    rc = main(["--no-mlflow", "--folds", "1"])
    assert rc == 2
    assert "configuration error" in capsys.readouterr().out


def test_main_runs_a_smoke_subset(monkeypatch, capsys):
    def fake_sweep(train, test, **kwargs):
        records = (
            SweepModelResult(
                model_type="mlp-shallow",
                model_name="MLP (shallow)",
                family=FAMILY_NEURAL,
                status=STATUS_SUCCEEDED,
                duration_seconds=0.1,
                metric_dict=_synthetic_metrics(),
                best_params={},
                n_trials=1,
                n_complete=1,
                device=DEVICE_CPU,
            ),
            SweepModelResult(
                model_type="mlp-wide",
                model_name="MLP (wide)",
                family=FAMILY_NEURAL,
                status=STATUS_SUCCEEDED,
                duration_seconds=0.1,
                metric_dict=_synthetic_metrics(),
                best_params={},
                n_trials=1,
                n_complete=1,
                device=DEVICE_CPU,
            ),
        )
        return SweepResult(
            sweep_run_id="cli-test",
            split_version="smoke",
            experiment_name=SWEEP_EXPERIMENT,
            tuning_experiment_name=DEFAULT_TUNING_EXPERIMENT,
            train_rows=len(train),
            test_rows=len(test),
            cv_folds=2,
            n_trials=1,
            config=smoke_sweep_config(n_trials=1, cv_folds=2),
            results=records,
            generated_at="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(run_mlp_sweep_module, "run_mlp_sweep", fake_sweep)
    rc = main(
        ["--smoke", "--architectures", "mlp-shallow,mlp-wide", "--no-mlflow"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "2/2 architectures succeeded" in out
    assert "`mlp-shallow`" in out
    assert "`mlp-wide`" in out


def _synthetic_metrics() -> dict[str, object]:
    rng = np.random.default_rng(0)
    labels = (rng.random(60) < 0.5).astype(int)
    proba = rng.random(60)
    predictions = (proba >= 0.5).astype(int)
    return compute_metric_dict(labels, proba, predictions)


# ---------------------------------------------------------------------------
# Folds fixture for the objective / run_study tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def folds(train_frame) -> tuple[TuningFold, ...]:
    return build_cv_folds(train_frame, n_folds=3)


class _FakeTrial:
    """Stand-in Optuna trial driving the objective deterministically."""

    number = 0

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params
        self.reports: list[tuple[float, int]] = []
        self.attrs: dict[str, object] = {}

    def suggest_float(self, name, low, high, *, step=None, log=False):
        return float(self.params[name])

    def suggest_int(self, name, low, high, *, step=None, log=False):
        return int(self.params[name])

    def suggest_categorical(self, name, choices):
        return self.params[name]

    def report(self, value, step):
        self.reports.append((float(value), int(step)))

    def should_prune(self):
        return False

    def set_user_attr(self, key, value):
        self.attrs[key] = value


# ---------------------------------------------------------------------------
# Real-torch end-to-end smoke sweep (torch-required; runs in the container)
# ---------------------------------------------------------------------------


def test_smoke_sweep_end_to_end_with_real_torch(train_frame, test_frame, tmp_path):
    pytest.importorskip("torch")
    result = run_mlp_sweep(
        train_frame,
        test_frame,
        config=smoke_sweep_config(n_trials=1, cv_folds=2),
        tracking_dir=tmp_path,
        split_version="smoke",
        prefer_gpu=False,  # deterministic CPU path, logged fallback
        sweep_id="torch-cpu-sweep",
    )
    assert result.n_models == MLP_SWEEP_SIZE
    assert result.n_succeeded == MLP_SWEEP_SIZE
    for record in result.results:
        assert record.device == DEVICE_CPU
        assert record.device_reason and "CPU requested" in record.device_reason
        assert set(record.metric_dict.keys()) == set(METRIC_KEYS)