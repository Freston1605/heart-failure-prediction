"""Tests for the MLP definition and its device-aware training loop (S05/T02).

Contracts under test:

1. **Architecture spec (torch-free)** — :class:`heart.models.mlp.MLPConfig`
   validates every value, and :func:`heart.models.mlp.layer_plan` describes the
   exact ordered layer list. Both run with no torch installed, which is the
   default host environment.
2. **Lazy torch import** — :func:`heart.models.mlp.import_torch` raises the
   named :class:`TorchUnavailableError` when torch is absent, rather than an
   opaque ``ImportError``; :func:`build_mlp` materialises a real
   ``torch.nn.Sequential`` when torch *is* present (these tests are skipped on
   the host and executed inside the ROCm container).
3. **Device resolution and the logged CPU fallback** —
   :func:`heart.models.train_torch.resolve_device` resolves ``cuda:0`` when a
   GPU is usable and ``cpu`` otherwise, always logging an explicit ``WARNING``
   when it falls back. The no-GPU branch is asserted by simulation (a fake
   torch module), so the fallback path is covered without a GPU.
4. **The selected feature representation** — :func:`resolve_feature_mode` reads
   the committed S04 selection ledger and returns its ``selected`` mode; a
   missing ledger falls back to baseline with a warning (or raises when strict).
   :func:`fit_feature_representation` fits on training rows only and produces a
   finite design matrix whose width matches the reported feature names.
5. **Evaluation parity** — the trained :class:`MLPClassifier` is scored through
   the shared :func:`heart.eval.contract.evaluate`, producing the canonical
   metric schema used by every classical model (torch-required tests).
6. **Negative paths** — invalid configs, malformed frames, missing feature
   columns, non-result report writes, and single-class validation splits all
   raise/return the named outcome rather than leaking an unhandled error.

No host test requires torch: torch-dependent tests use ``pytest.importorskip``
and run inside ``containers/run-rocm.sh shell``. Every store-free test uses
inline fixtures; nothing reads a gitignored path.
"""

from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from heart.data.schema import TARGET_COLUMN
from heart.eval.contract import METRIC_KEYS
from heart.eval.metrics import compute_metric_dict
from heart.features.ablation import FeatureMode
from heart.features.selection import (
    DEFAULT_LEDGER_PATH as SELECTION_LEDGER_PATH,
    read_selection,
)
from heart.models.mlp import (
    ACTIVATIONS,
    DEFAULT_HIDDEN_SIZES,
    LAYER_ACTIVATION,
    LAYER_BATCH_NORM,
    LAYER_DROPOUT,
    LAYER_LINEAR,
    MLPBuildError,
    MLPConfig,
    MLPConfigError,
    TorchUnavailableError,
    build_mlp,
    describe_mlp,
    import_torch,
    layer_plan,
)
from heart.models.train_torch import (
    CPU_FALLBACK_WARNING,
    CPU_FORCED_WARNING,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEFAULT_REPORT_PATH,
    FeatureRepresentation,
    TorchConfigError,
    TorchDataError,
    TorchReportError,
    TrainingConfig,
    TrainingHistory,
    TrainingResult,
    build_feature_pipeline,
    build_parser,
    describe_training_result,
    fit_feature_representation,
    main,
    render_training_report,
    resolve_device,
    resolve_feature_mode,
    run_training,
    smoke_training_config,
    train_mlp_classifier,
    write_training_json,
    write_training_report,
)
from heart.models.train_torch import _sigmoid, _split_validation

train_torch_module = importlib.import_module("heart.models.train_torch")


# ---------------------------------------------------------------------------
# Fakes / fixtures
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
    return _schema_valid_frame(rows=140, seed=3)


@pytest.fixture(scope="module")
def test_frame() -> pd.DataFrame:
    return _schema_valid_frame(rows=70, seed=5)


class _FakeCuda:
    def __init__(
        self,
        available: bool,
        count: int = 1,
        name: str = "AMD Radeon RX 9070 XT",
        gcn: str = "gfx1201",
    ) -> None:
        self._available = available
        self._count = count if available else 0
        self._name = name
        self._gcn = gcn

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_properties(self, index: int) -> object:
        assert index == 0
        return SimpleNamespace(name=self._name, gcnArchName=self._gcn)


class _FakeTorch:
    """Duck-typed torch exposing just what the device probe reads."""

    __version__ = "2.10.0+rocm7.2.4"
    version = SimpleNamespace(hip="7.2.4")

    def __init__(self, available: bool, count: int = 1) -> None:
        self.cuda = _FakeCuda(available, count)


def _raising_importer(name: str) -> object:
    raise ImportError(f"simulated missing module {name!r}")


def _synthetic_metrics() -> dict[str, object]:
    rng = np.random.default_rng(0)
    y = (rng.random(60) < 0.5).astype(int)
    proba = rng.random(60)
    pred = (proba >= 0.5).astype(int)
    return compute_metric_dict(y, proba, pred)


def _cpu_history() -> TrainingHistory:
    return TrainingHistory(
        train_losses=(0.7, 0.6),
        val_losses=(0.72, 0.68),
        best_epoch=1,
        epochs_run=2,
        stopped_early=False,
        best_val_loss=0.68,
    )


def _synthetic_result(device: str) -> TrainingResult:
    gpu = device != DEVICE_CPU
    resolution = train_torch_module.DeviceResolution(
        requested="gpu",
        device=device,
        is_gpu=gpu,
        gpu_available=gpu,
        torch_available=True,
        reason="synthetic",
        warnings=() if gpu else (CPU_FALLBACK_WARNING,),
        torch_version="2.10.0+rocm7.2.4",
        hip_version="7.2.4",
        device_name="AMD Radeon RX 9070 XT" if gpu else None,
        gcn_arch_name="gfx1201" if gpu else None,
    )
    return TrainingResult(
        device=device,
        used_gpu=gpu,
        cpu_fallback=not gpu,
        mode_name="selected",
        is_baseline_features=False,
        n_features=17,
        split_version="v1",
        train_rows=734,
        test_rows=184,
        mlp_config=MLPConfig(input_dim=17),
        training_config=smoke_training_config(),
        history=_cpu_history(),
        metrics=_synthetic_metrics(),
        device_resolution=resolution,
        duration_seconds=1.25,
        generated_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Architecture spec (torch-free)
# ---------------------------------------------------------------------------


def test_mlp_config_defaults_are_valid():
    config = MLPConfig(input_dim=17)
    assert config.hidden_sizes == DEFAULT_HIDDEN_SIZES
    assert config.activation == "relu"
    assert config.output_dim == 1
    assert config.depth == len(DEFAULT_HIDDEN_SIZES) + 1
    assert config.width == DEFAULT_HIDDEN_SIZES[0]


def test_mlp_config_binds_input_dim_without_mutating():
    base = MLPConfig(input_dim=1, hidden_sizes=(8, 4), dropout=0.1)
    rebound = base.with_input_dim(17)
    assert rebound.input_dim == 17
    assert rebound.hidden_sizes == (8, 4)
    assert base.input_dim == 1


def test_mlp_config_to_params_flattens_layers():
    params = MLPConfig(input_dim=12, hidden_sizes=(32, 16)).to_params()
    assert params["mlp_input_dim"] == 12
    assert params["mlp_hidden_0_units"] == 32
    assert params["mlp_hidden_1_units"] == 16
    assert params["mlp_hidden_sizes"] == [32, 16]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"input_dim": 0}, MLPConfigError),
        ({"input_dim": -3}, MLPConfigError),
        ({"input_dim": 4, "hidden_sizes": ()}, MLPConfigError),
        ({"input_dim": 4, "hidden_sizes": (8, 0)}, MLPConfigError),
        ({"input_dim": 4, "dropout": 1.0}, MLPConfigError),
        ({"input_dim": 4, "dropout": -0.1}, MLPConfigError),
        ({"input_dim": 4, "activation": "sigmoid"}, MLPConfigError),
        ({"input_dim": 4, "output_dim": 0}, MLPConfigError),
        ({"input_dim": 4, "batch_norm": "yes"}, MLPConfigError),
    ],
)
def test_mlp_config_rejects_invalid_values(kwargs, expected):
    with pytest.raises(expected):
        MLPConfig(**kwargs)


def test_layer_plan_shapes_and_order():
    config = MLPConfig(input_dim=10, hidden_sizes=(16, 8), dropout=0.25)
    plan = layer_plan(config)
    kinds = [entry.kind for entry in plan]
    assert kinds == [
        LAYER_LINEAR,
        LAYER_ACTIVATION,
        LAYER_DROPOUT,
        LAYER_LINEAR,
        LAYER_ACTIVATION,
        LAYER_DROPOUT,
        LAYER_LINEAR,
    ]
    assert (plan[0].in_features, plan[0].out_features) == (10, 16)
    assert (plan[3].in_features, plan[3].out_features) == (16, 8)
    assert (plan[-1].in_features, plan[-1].out_features) == (8, 1)
    assert plan[1].activation == "relu"
    assert plan[2].rate == pytest.approx(0.25)


def test_layer_plan_dropout_disabled_omits_dropout_layers():
    plan = layer_plan(MLPConfig(input_dim=5, hidden_sizes=(4,), dropout=0.0))
    assert LAYER_DROPOUT not in {entry.kind for entry in plan}


def test_layer_plan_batch_norm_added_after_each_hidden_linear():
    plan = layer_plan(
        MLPConfig(input_dim=5, hidden_sizes=(7, 3), batch_norm=True, dropout=0.0)
    )
    first = plan[0]
    second = plan[1]
    assert first.kind == LAYER_LINEAR
    assert second.kind == LAYER_BATCH_NORM
    assert second.out_features == 7
    # Two hidden layers -> two batch-norm entries.
    assert sum(1 for entry in plan if entry.kind == LAYER_BATCH_NORM) == 2


def test_layer_plan_rejects_non_config():
    with pytest.raises(MLPConfigError):
        layer_plan("not-a-config")  # type: ignore[arg-type]


def test_import_torch_missing_raises_named_error():
    with pytest.raises(TorchUnavailableError):
        import_torch(_raising_importer)


def test_build_mlp_rejects_non_config():
    with pytest.raises(MLPConfigError):
        build_mlp("nope")  # type: ignore[arg-type]


def test_describe_mlp_mentions_architecture():
    text = describe_mlp(MLPConfig(input_dim=10, hidden_sizes=(8,), dropout=0.0))
    assert "10 -> 8 -> 1" in text
    assert "relu" in text
    assert "dropout: 0.00 (disabled)" in text


# ---------------------------------------------------------------------------
# Architecture materialisation (torch-required; skipped on the host)
# ---------------------------------------------------------------------------


def test_build_mlp_materialises_sequential_and_forwards():
    torch = pytest.importorskip("torch")
    config = MLPConfig(input_dim=6, hidden_sizes=(12, 4), dropout=0.0)
    model = build_mlp(config, torch_module=torch)
    assert isinstance(model, torch.nn.Sequential)
    out = model(torch.zeros(5, 6))
    assert tuple(out.shape) == (5, 1)


def test_build_mlp_batch_norm_forwards():
    torch = pytest.importorskip("torch")
    config = MLPConfig(
        input_dim=6, hidden_sizes=(8,), dropout=0.1, batch_norm=True
    )
    model = build_mlp(config, torch_module=torch)
    out = model(torch.randn(6, 6))
    assert tuple(out.shape) == (6, 1)


def test_build_mlp_wraps_layer_failures():
    class _BrokenNN:
        def Linear(self, *args, **kwargs):  # noqa: N802 - mirrors torch.nn
            raise RuntimeError("no linear for you")

    broken_torch = SimpleNamespace(nn=_BrokenNN())
    with pytest.raises(MLPBuildError):
        build_mlp(MLPConfig(input_dim=3, hidden_sizes=(4,)), torch_module=broken_torch)


# ---------------------------------------------------------------------------
# Device resolution and the logged CPU fallback
# ---------------------------------------------------------------------------


def test_resolve_device_gpu_path():
    resolution = resolve_device(torch_module=_FakeTorch(available=True))
    assert resolution.device == DEVICE_CUDA
    assert resolution.is_gpu is True
    assert resolution.cpu_fallback is False
    assert resolution.warnings == ()
    assert resolution.device_name == "AMD Radeon RX 9070 XT"
    assert resolution.gcn_arch_name == "gfx1201"


def test_resolve_device_cpu_fallback_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="heart.models.train_torch"):
        resolution = resolve_device(torch_module=_FakeTorch(available=False))
    assert resolution.device == DEVICE_CPU
    assert resolution.cpu_fallback is True
    assert resolution.is_gpu is False
    assert CPU_FALLBACK_WARNING in resolution.warnings
    assert CPU_FALLBACK_WARNING in caplog.text


def test_resolve_device_torch_missing_falls_back(caplog):
    with caplog.at_level(logging.WARNING, logger="heart.models.train_torch"):
        resolution = resolve_device(torch_importer=_raising_importer)
    assert resolution.device == DEVICE_CPU
    assert resolution.torch_available is False
    assert resolution.cpu_fallback is True
    assert any("torch is not importable" in warning for warning in resolution.warnings)
    assert "torch is not importable" in caplog.text


def test_resolve_device_forced_cpu_logs_forced_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="heart.models.train_torch"):
        resolution = resolve_device(
            torch_module=_FakeTorch(available=True), prefer_gpu=False
        )
    assert resolution.device == DEVICE_CPU
    assert resolution.requested == "cpu"
    assert CPU_FORCED_WARNING in resolution.warnings
    assert CPU_FORCED_WARNING in caplog.text


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


def test_training_config_defaults_and_smoke():
    config = TrainingConfig()
    assert config.epochs == 40
    assert config.seed == 42
    smoke = smoke_training_config()
    assert smoke.epochs == 3
    assert smoke.early_stopping_patience == 2
    assert isinstance(smoke.to_dict(), dict)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
        {"validation_fraction": 0.6},
        {"validation_fraction": -0.1},
        {"early_stopping_patience": -1},
        {"seed": "42"},
        {"pos_weight": 0.0},
        {"shuffle": "yes"},
    ],
)
def test_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(TorchConfigError):
        TrainingConfig(**kwargs)


def test_split_validation_single_class_returns_no_validation():
    features = np.zeros((6, 2))
    labels = np.zeros(6, dtype=int)  # one class only
    x_train, y_train, x_val, y_val = _split_validation(features, labels, TrainingConfig())
    assert x_val is None and y_val is None
    assert len(y_train) == 6


def test_split_validation_stratifies_when_both_classes_present():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 3))
    labels = np.array([0, 1] * 20)
    _, _, x_val, y_val = _split_validation(
        features, labels, TrainingConfig(validation_fraction=0.25, seed=1)
    )
    assert x_val is not None and y_val is not None
    assert set(np.unique(y_val)) == {0, 1}


def test_sigmoid_is_numerically_stable():
    values = np.array([-1000.0, 0.0, 1000.0])
    result = _sigmoid(values)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Selected feature representation
# ---------------------------------------------------------------------------


def test_resolve_feature_mode_from_tracked_ledger():
    selection = read_selection(SELECTION_LEDGER_PATH)
    mode = resolve_feature_mode(selection_path=SELECTION_LEDGER_PATH)
    assert isinstance(mode, FeatureMode)
    assert mode.name == selection.selected_mode.name
    assert set(mode.include) == set(selection.selected_mode.include)


def test_resolve_feature_mode_explicit_mode_wins():
    explicit = FeatureMode.baseline(name="explicit-baseline")
    mode = resolve_feature_mode(
        mode=explicit, selection_path="does-not-exist.json"
    )
    assert mode is explicit


def test_resolve_feature_mode_missing_ledger_falls_back_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="heart.models.train_torch"):
        mode = resolve_feature_mode(
            selection_path="does-not-exist-anywhere.json", allow_baseline_fallback=True
        )
    assert mode.is_baseline is True
    assert "baseline feature set" in caplog.text


def test_resolve_feature_mode_missing_ledger_strict_raises():
    with pytest.raises(train_torch_module.TorchRepresentationError):
        resolve_feature_mode(
            selection_path="does-not-exist-anywhere.json",
            allow_baseline_fallback=False,
        )


def test_build_feature_pipeline_drops_classifier():
    pipeline = build_feature_pipeline(FeatureMode.baseline())
    assert [name for name, _ in pipeline.steps] == [
        "zero_policy",
        "engineering",
        "features",
    ]


def test_fit_feature_representation_selected_is_wider_than_baseline(train_frame):
    baseline = fit_feature_representation(train_frame, mode=FeatureMode.baseline())
    selected = fit_feature_representation(train_frame)  # from the committed ledger
    assert baseline.is_baseline is True
    assert selected.is_baseline is False
    assert selected.n_features > baseline.n_features
    matrix = selected.transform(train_frame)
    assert matrix.shape == (len(train_frame), selected.n_features)
    assert np.isfinite(matrix).all()
    assert len(selected.feature_names) == selected.n_features


def test_feature_representation_transform_rejects_missing_columns(train_frame):
    representation = fit_feature_representation(
        train_frame, mode=FeatureMode.baseline()
    )
    with pytest.raises(TorchDataError):
        representation.transform(train_frame[["Age", "Sex"]])


def test_fit_feature_representation_rejects_bad_frame():
    with pytest.raises(TorchDataError):
        fit_feature_representation(pd.DataFrame({"Age": [1, 2]}))


def test_feature_representation_rejects_width_mismatch(train_frame):
    representation = fit_feature_representation(
        train_frame, mode=FeatureMode.baseline()
    )
    with pytest.raises(train_torch_module.TorchRepresentationError):
        FeatureRepresentation(
            mode_name="broken",
            is_baseline=True,
            n_engineered_columns=0,
            feature_names=("a",),
            n_features=2,
            preprocessor=representation.preprocessor,
        )


def test_train_mlp_classifier_raises_when_torch_missing(train_frame):
    with pytest.raises(TorchUnavailableError):
        train_mlp_classifier(train_frame, torch_importer=_raising_importer)


# ---------------------------------------------------------------------------
# Report rendering / writing (torch-free)
# ---------------------------------------------------------------------------


def test_render_training_report_states_cpu_device():
    text = render_training_report(_synthetic_result(DEVICE_CPU))
    assert "**device used: cpu** (CPU)" in text
    assert "**cpu fallback: YES**" in text
    assert "roc_auc" in text


def test_render_training_report_states_gpu_device():
    text = render_training_report(_synthetic_result(DEVICE_CUDA))
    assert "**device used: cuda:0** (GPU)" in text
    assert "**cpu fallback: NO**" in text
    assert "AMD Radeon RX 9070 XT" in text


def test_write_training_report_and_json_round_trip(tmp_path):
    result = _synthetic_result(DEVICE_CPU)
    report = write_training_report(result, tmp_path / "mlp_training.md")
    sidecar = write_training_json(result, tmp_path / "mlp_training.json")
    assert report.read_text(encoding="utf-8").startswith("# MLP training run")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["device"] == DEVICE_CPU
    assert payload["cpu_fallback"] is True
    assert set(payload["metrics"]) == set(METRIC_KEYS)


def test_write_training_report_rejects_non_result(tmp_path):
    with pytest.raises(TorchReportError):
        write_training_report("nope", tmp_path / "x.md")  # type: ignore[arg-type]


def test_describe_training_result_mentions_device():
    text = describe_training_result(_synthetic_result(DEVICE_CPU))
    assert "device used: cpu" in text
    assert "cpu fallback: yes" in text


def test_default_report_path_is_under_reports():
    assert DEFAULT_REPORT_PATH.name == "mlp_training.md"
    assert DEFAULT_REPORT_PATH.parent.name == "reports"


# ---------------------------------------------------------------------------
# End-to-end training + shared-contract evaluation (torch-required)
# ---------------------------------------------------------------------------


def _tiny_mlp_config(width: int) -> MLPConfig:
    return MLPConfig(input_dim=width, hidden_sizes=(8,), dropout=0.0)


def test_train_mlp_classifier_cpu_records_device(train_frame):
    pytest.importorskip("torch")
    classifier = train_mlp_classifier(
        train_frame,
        mlp_config=_tiny_mlp_config(1),
        training_config=smoke_training_config(),
        prefer_gpu=False,
    )
    assert classifier.device == DEVICE_CPU
    assert classifier.device_resolution.cpu_fallback is True
    assert classifier.history is not None
    assert classifier.history.epochs_run >= 1
    # input_dim is rebound from the fitted representation, not the template.
    assert classifier.mlp_config.input_dim == classifier.n_features


def test_run_training_scores_through_shared_contract(train_frame, test_frame):
    pytest.importorskip("torch")
    _, result = run_training(
        train_frame,
        test_frame,
        mlp_config=_tiny_mlp_config(1),
        training_config=smoke_training_config(),
        prefer_gpu=False,
    )
    assert result.device == DEVICE_CPU
    assert result.cpu_fallback is True
    assert set(result.metrics) == set(METRIC_KEYS)
    assert 0.0 <= result.primary_metric <= 1.0
    assert result.train_rows == len(train_frame)
    assert result.test_rows == len(test_frame)


def test_run_training_report_records_device(train_frame, test_frame, tmp_path):
    pytest.importorskip("torch")
    _, result = run_training(
        train_frame,
        test_frame,
        mlp_config=_tiny_mlp_config(1),
        training_config=smoke_training_config(),
        prefer_gpu=False,
    )
    path = write_training_report(result, tmp_path / "report.md")
    assert "device used: cpu" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_parser_defaults():
    args = build_parser().parse_args([])
    assert args.split_version == "v1"
    assert args.cpu is False
    assert args.baseline_features is False
    assert args.report_path == str(DEFAULT_REPORT_PATH)


def test_cli_build_configs_parses_hidden_sizes():
    args = build_parser().parse_args(
        ["--hidden-sizes", "16,8", "--dropout", "0.3", "--smoke"]
    )
    mlp_config, training = train_torch_module._build_configs(args)
    assert mlp_config is not None
    assert mlp_config.hidden_sizes == (16, 8)
    assert mlp_config.dropout == pytest.approx(0.3)
    assert training.epochs == 3


def test_cli_build_configs_defaults_to_template_none():
    args = build_parser().parse_args([])
    mlp_config, training = train_torch_module._build_configs(args)
    assert mlp_config is None
    assert isinstance(training, TrainingConfig)


def test_cli_rejects_bad_hidden_sizes():
    args = build_parser().parse_args(["--hidden-sizes", "16,oops"])
    with pytest.raises(TorchConfigError):
        train_torch_module._build_configs(args)


def test_cli_main_returns_two_on_bad_configuration():
    assert main(["--hidden-sizes", "3,x"]) == 2


def test_cli_main_reports_torch_unavailable_without_traceback(
    monkeypatch, tmp_path, train_frame, test_frame, capsys
):
    def _raising_import_torch(importer):  # noqa: ANN001 - mirrors import_torch seam
        raise TorchUnavailableError("simulated missing torch")

    monkeypatch.setattr(
        train_torch_module,
        "load_split_frames",
        lambda version: (train_frame, test_frame),
    )
    monkeypatch.setattr(train_torch_module, "import_torch", _raising_import_torch)

    code = main(
        [
            "--smoke",
            "--report-path",
            str(tmp_path / "r.md"),
            "--json-path",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 1
    assert "training error" in capsys.readouterr().out
