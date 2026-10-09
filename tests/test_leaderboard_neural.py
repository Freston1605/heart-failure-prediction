"""Neural runs on the leaderboard (S05/T04).

Contracts under test:

1. **No special-casing** — the leaderboard generator selects neural runs by the
   exact same generic rules as classical runs (``run_kind=final``, a
   ``model_type`` tag, ``FINISHED`` status, the complete metric suite). A neural
   MLflow run is a candidate because it went through the shared tracking
   convention, not because the generator knows about ``family=neural``.
2. **Device is answerable from the board** — neural runs record an optional
   ``device`` tag; the leaderboard reads it generically into a Device column
   (``cuda:0`` / ``cpu``), and a run without the tag renders ``-`` rather than
   being dropped.
3. **Directly comparable** — the real ``run_battery`` and ``run_mlp_sweep``
   runners, logging into one store, produce one ranked leaderboard containing
   every model from both protocols, ordered by the shared primary metric.

Every store-backed test runs against a fresh ``tmp_path`` SQLite database, so
the suite never touches the repository's ``experiments/mlruns`` directory.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import optuna
import pandas as pd
import pytest

from heart.data.schema import TARGET_COLUMN
from heart.eval.contract import PRIMARY_METRIC, compute_metric_dict
from heart.models.mlp_spaces import MLP_MODEL_TYPES
from heart.models.run_battery import run_battery, smoke_config
from heart.models.run_mlp_sweep import run_mlp_sweep, smoke_sweep_config
from heart.models.train_torch import DEVICE_CPU, DeviceResolution
from heart.reporting.leaderboard import (
    DEVICE_TAG,
    NO_DEVICE_LABEL,
    build_leaderboard,
    generate_leaderboard,
    render_leaderboard,
    select_leaderboard_rows,
)
from heart.tracking.mlflow_store import configure_tracking
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import (
    FINAL_RUN_KIND,
    MODEL_TYPE_TAG,
    RUN_KIND_TAG,
    NoCompletedTrialError,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self, tracking_dir, config):
        self.tracking_dir = tracking_dir
        self.config = config


@pytest.fixture
def store(tmp_path) -> _Store:
    tracking_dir = tmp_path / "mlruns"
    config = configure_tracking(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(config.tracking_uri)
    return _Store(tracking_dir=tracking_dir, config=config)


def _sample_metrics(*, roc_auc: float | None = None) -> dict:
    """A complete, schema-valid metric dict built without fitting a model."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 4)
    probabilities = np.clip(0.18 + 0.64 * labels, 0.0, 1.0)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = compute_metric_dict(labels, probabilities, predictions, n_bins=5)
    if roc_auc is not None:
        metrics[PRIMARY_METRIC] = float(roc_auc)
    return metrics


def _log_run(
    store: _Store,
    *,
    model_type: str,
    model_name: str,
    family: str,
    device: str | None = None,
    roc_auc: float | None = None,
) -> str:
    """Log one final run through the shared tracking convention."""
    mlflow.set_tracking_uri(store.config.tracking_uri)
    tags = {
        RUN_KIND_TAG: FINAL_RUN_KIND,
        MODEL_TYPE_TAG: model_type,
        "family": family,
        "split_version": "v1",
    }
    if device is not None:
        tags[DEVICE_TAG] = device
    run = log_evaluation_run(
        _sample_metrics(roc_auc=roc_auc),
        model_name=model_name,
        split_version="v1",
        params={"learning_rate": 1e-3} if family == "neural" else {"C": 1.0},
        tags=tags,
        config=store.config,
    )
    return run.run_id


def _row_line(text: str, model_name: str) -> str:
    for line in text.splitlines():
        if line.startswith("|") and model_name in line:
            return line
    raise AssertionError(f"no rendered row for {model_name!r}")


# ---------------------------------------------------------------------------
# No special-casing: the generator is family-agnostic
# ---------------------------------------------------------------------------


def test_neural_run_is_selected_without_special_casing(store):
    _log_run(
        store,
        model_type="logistic-regression-l2",
        model_name="Logistic Regression (L2)",
        family="linear",
        roc_auc=0.90,
    )
    _log_run(
        store,
        model_type="mlp-wide",
        model_name="MLP Wide",
        family="neural",
        device="cuda:0",
        roc_auc=0.92,
    )

    rows, considered = select_leaderboard_rows(tracking_dir=store.tracking_dir)
    assert considered == 2
    assert {row.model_type for row in rows} == {"mlp-wide", "logistic-regression-l2"}
    assert {row.family for row in rows} == {"neural", "linear"}

    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert board.n_models == 2
    assert board.row_for("mlp-wide") is not None


def test_neural_rows_rank_by_roc_auc_alongside_classical(store):
    _log_run(store, model_type="lda", model_name="LDA", family="discriminant", roc_auc=0.90)
    _log_run(
        store,
        model_type="mlp-deep",
        model_name="MLP Deep",
        family="neural",
        device="cuda:0",
        roc_auc=0.95,
    )
    _log_run(
        store,
        model_type="mlp-shallow",
        model_name="MLP Shallow",
        family="neural",
        device="cpu",
        roc_auc=0.80,
    )

    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert [row.model_type for row in board.rows] == ["mlp-deep", "lda", "mlp-shallow"]
    scores = [row.primary_metric for row in board.rows]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Device column
# ---------------------------------------------------------------------------


def test_device_tag_is_read_generically(store):
    _log_run(store, model_type="lda", model_name="LDA", family="discriminant", roc_auc=0.90)
    _log_run(
        store,
        model_type="mlp-wide",
        model_name="MLP Wide",
        family="neural",
        device="cuda:0",
        roc_auc=0.92,
    )
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert board.row_for("mlp-wide").device == "cuda:0"
    assert board.row_for("lda").device is None


def test_device_column_renders_device_and_dash(store):
    _log_run(store, model_type="lda", model_name="LDA", family="discriminant", roc_auc=0.90)
    _log_run(
        store,
        model_type="mlp-wide",
        model_name="MLP Wide",
        family="neural",
        device="cuda:0",
        roc_auc=0.92,
    )
    text = render_leaderboard(build_leaderboard(tracking_dir=store.tracking_dir))

    assert "| Device |" in text
    assert "cuda:0" in _row_line(text, "MLP Wide")
    # Classical runs carry no device tag: rendered as the dash, not omitted.
    assert f"| {NO_DEVICE_LABEL} |" in _row_line(text, "LDA")


def test_cpu_fallback_device_is_rendered(store):
    _log_run(
        store,
        model_type="mlp-deep",
        model_name="MLP Deep",
        family="neural",
        device="cpu",
        roc_auc=0.88,
    )
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert board.row_for("mlp-deep").device == "cpu"
    assert "cpu" in render_leaderboard(board)


def test_missing_device_tag_is_not_an_error(store):
    # A neural-family run that never recorded a device must still be a
    # candidate — no special-casing may require the tag.
    _log_run(store, model_type="mlp-wide", model_name="MLP Wide", family="neural", roc_auc=0.91)
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert board.n_models == 1
    assert board.row_for("mlp-wide").device is None
    assert NO_DEVICE_LABEL in _row_line(render_leaderboard(board), "MLP Wide")


def test_device_is_serialised_in_the_board_dict(store):
    _log_run(
        store,
        model_type="mlp-wide",
        model_name="MLP Wide",
        family="neural",
        device="cuda:0",
        roc_auc=0.92,
    )
    payload = build_leaderboard(tracking_dir=store.tracking_dir).to_dict()
    assert payload["rows"][0]["device"] == "cuda:0"
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Deterministic stand-ins for the real sweep (torch-free, mirrors T03)
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


def _synthetic_trainer(seed: int = 0):
    def trainer(params, fold):
        n = fold.n_eval
        rng = np.random.default_rng(seed + int(fold.fold))
        labels = np.asarray(fold.split.y_test)
        proba = rng.random(n)
        predictions = (proba >= 0.5).astype(int)
        metrics = compute_metric_dict(labels, proba, predictions)
        return metrics, predictions, proba

    return trainer


class _StubFitted:
    def __init__(self, resolution: DeviceResolution) -> None:
        self.device_resolution = resolution
        self.device = resolution.device

    def predict_proba(self, X: object) -> np.ndarray:
        proba = np.random.default_rng(0).random(len(X))
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: object) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _stub_refit_factory():
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


def _schema_valid_frame(rows: int, seed: int) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# End-to-end: both protocols, one leaderboard
# ---------------------------------------------------------------------------


def test_real_runners_share_one_leaderboard(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    train = _schema_valid_frame(rows=160, seed=7)
    test = _schema_valid_frame(rows=80, seed=11)

    classical = run_battery(
        train,
        test,
        model_types=["naive-bayes", "lda"],
        config=smoke_config(n_trials=1, cv_folds=2),
        tracking_dir=tracking_dir,
        split_version="v1",
        battery_id="t04-classical",
    )
    assert classical.n_succeeded == 2

    sweep = run_mlp_sweep(
        train,
        test,
        config=smoke_sweep_config(n_trials=1, cv_folds=2),
        trainer=_synthetic_trainer(),
        refit_factory=_stub_refit_factory(),
        tracking_dir=tracking_dir,
        split_version="v1",
        sweep_id="t04-neural",
        sampler=_CheapSampler(),
        pruner=optuna.pruners.NopPruner(),
    )
    assert sweep.n_succeeded == len(MLP_MODEL_TYPES)

    report_path = tmp_path / "leaderboard.md"
    board = generate_leaderboard(tracking_dir=tracking_dir, report_path=report_path)

    # Every model from both protocols is on one board — no family filtering.
    assert board.n_models == 2 + len(MLP_MODEL_TYPES)
    assert {row.model_type for row in board.rows} >= set(MLP_MODEL_TYPES)
    assert {"naive-bayes", "lda"} <= {row.model_type for row in board.rows}
    neural_rows = [row for row in board.rows if row.family == "neural"]
    assert len(neural_rows) == len(MLP_MODEL_TYPES)
    # The device recorded by the sweep is readable from the leaderboard row.
    assert {row.device for row in neural_rows} == {DEVICE_CPU}

    text = report_path.read_text(encoding="utf-8")
    assert "| Device |" in text
    assert "MLP (wide)" in text and DEVICE_CPU in text
    scores = [row.primary_metric for row in board.rows]
    assert scores == sorted(scores, reverse=True)
