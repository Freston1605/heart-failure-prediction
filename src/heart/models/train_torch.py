"""PyTorch MLP training loop with an explicit, logged CPU fallback (S05/T02).

This module owns **how** the neural model is trained and, critically, **where**
it ran. It turns the project's highest environment risk — "does the GPU path
actually work?" — into a recorded fact on every run:

* :func:`resolve_device` probes torch (through the S05/T01
  :func:`heart.gpu.device_check.probe_torch`) and resolves ``cuda:0`` or
  ``cpu``. When the GPU is unavailable it logs an explicit ``WARNING`` and
  records the fallback reason; a forced-CPU run records that too.
* :class:`MLPClassifier` wraps the fitted feature representation (the S04
  selected representation), the ``torch.nn`` model, and the resolved device, and
  exposes ``predict`` / ``predict_proba`` so the model is scored through the
  **same** :func:`heart.eval.contract.evaluate` on the **same** S01 split as
  every classical model. That is what keeps neural results comparable.

Feature representation and splits
---------------------------------
:func:`resolve_feature_mode` reads the committed S04 selection ledger
(``reports/feature_selection.json``) and returns its ``selected``
:class:`~heart.features.ablation.FeatureMode`; a missing/unreadable ledger falls
back to the baseline mode with a warning (never silently). The feature chain is
reused verbatim from :func:`heart.features.ablation.build_ablation_pipeline`
(``zero_policy -> engineering -> ColumnTransformer``) with the classifier step
dropped, so the MLP sees exactly the selected representation. Training frames
come from :func:`heart.data.split.load_split_frames`, i.e. the identical split
the classical battery uses.

Torch is imported lazily
------------------------
Nothing here imports torch at module load, so the module (and its tests) load on
a CPU-only host. The real training loop runs inside the ROCm container
(``containers/run-rocm.sh shell``); on the host, the device-resolution and
fallback paths remain fully testable through injected fakes.

Observability
-------------
:func:`resolve_device` logs the resolved device and every warning;
:func:`train_mlp_classifier` logs the representation, feature width, epochs,
best epoch, and — when applicable — a final ``WARNING`` that the run completed
on CPU. :func:`run_training` logs the primary metric, and
:func:`render_training_report` / :func:`write_training_report` persist a
device-stamped run report (markdown plus a JSON sidecar) so the device used is
answerable from the artifact alone.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from heart.config import RANDOM_SEED, REPORTS_DIR
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, SplitError, load_split_frames
from heart.eval.contract import (
    PRIMARY_METRIC,
    EvaluationSplit,
    evaluate,
    evaluation_split_from_frames,
)
from heart.features.ablation import (
    AblationError,
    FeatureMode,
    build_ablation_pipeline,
)
from heart.features.selection import (
    DEFAULT_LEDGER_PATH as DEFAULT_SELECTION_LEDGER_PATH,
    FeatureSelectionError,
    read_selection,
)
from heart.gpu.device_check import (
    DEVICE_CPU,
    DEVICE_CUDA,
    TARGET_GPU_ARCH,
    probe_torch,
)
from heart.runtime import atomic_write_text, json_default, write_json_document
from heart.models.mlp import (
    MLPConfig,
    MLPError,
    TorchUnavailableError,
    build_mlp,
    describe_mlp,
    import_torch,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEVICE_CUDA",
    "DEVICE_CPU",
    "CPU_FALLBACK_WARNING",
    "CPU_FORCED_WARNING",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_VALIDATION_FRACTION",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "SMOKE_EPOCHS",
    "DEFAULT_REPORT_FILENAME",
    "DEFAULT_JSON_FILENAME",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_JSON_PATH",
    "TorchTrainingError",
    "TorchConfigError",
    "TorchDataError",
    "TorchRepresentationError",
    "TorchReportError",
    "TrainingConfig",
    "DeviceResolution",
    "FeatureRepresentation",
    "TrainingHistory",
    "MLPClassifier",
    "TrainingResult",
    "smoke_training_config",
    "resolve_device",
    "resolve_feature_mode",
    "build_feature_pipeline",
    "fit_feature_representation",
    "train_mlp_classifier",
    "run_training",
    "render_training_report",
    "write_training_report",
    "write_training_json",
    "describe_training_result",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: The torch device string used when a GPU is usable.
#: (Re-exported from :mod:`heart.gpu.device_check`; repeated here for clarity.)
# DEVICE_CUDA / DEVICE_CPU come from device_check above.

#: Warning emitted when the GPU is unavailable and the loop falls back to CPU.
CPU_FALLBACK_WARNING: str = (
    "GPU unavailable; training the MLP on CPU (explicit, logged fallback). "
    "The run completes; it is not a failure."
)

#: Warning emitted when the CPU path was requested explicitly.
CPU_FORCED_WARNING: str = (
    "GPU acceleration disabled by request; training the MLP on CPU."
)

#: Default number of training epochs.
DEFAULT_EPOCHS: int = 40

#: Default mini-batch size.
DEFAULT_BATCH_SIZE: int = 32

#: Default AdamW learning rate.
DEFAULT_LEARNING_RATE: float = 1e-3

#: Default AdamW weight decay.
DEFAULT_WEIGHT_DECAY: float = 1e-4

#: Default fraction of training rows held out for early stopping.
DEFAULT_VALIDATION_FRACTION: float = 0.15

#: Default early-stopping patience (epochs without improvement).
DEFAULT_EARLY_STOPPING_PATIENCE: int = 10

#: Epoch budget for the reduced smoke configuration.
SMOKE_EPOCHS: int = 3

#: Default device-stamped run report filename (under ``reports/``).
DEFAULT_REPORT_FILENAME: str = "mlp_training.md"

#: Default machine-readable sidecar filename (under ``reports/``).
DEFAULT_JSON_FILENAME: str = "mlp_training.json"

#: Default markdown report location.
DEFAULT_REPORT_PATH: Path = Path(REPORTS_DIR) / DEFAULT_REPORT_FILENAME

#: Default JSON sidecar location.
DEFAULT_JSON_PATH: Path = Path(REPORTS_DIR) / DEFAULT_JSON_FILENAME


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class TorchTrainingError(Exception):
    """Base class for every neural training failure."""


class TorchConfigError(TorchTrainingError):
    """A training or architecture configuration value is invalid."""


class TorchDataError(TorchTrainingError):
    """A frame, split, or label vector handed to training is malformed."""


class TorchRepresentationError(TorchTrainingError):
    """The selected feature representation could not be resolved or fitted."""


class TorchReportError(TorchTrainingError):
    """The device-stamped training report could not be written."""


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """The resolved, validated knobs for one MLP training run."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = 0.0
    seed: int = RANDOM_SEED
    pos_weight: float | None = None
    shuffle: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "early_stopping_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TorchConfigError(f"{name} must be an integer, got {value!r}.")
        if self.epochs < 1:
            raise TorchConfigError(f"epochs must be >= 1, got {self.epochs!r}.")
        if self.batch_size < 1:
            raise TorchConfigError(
                f"batch_size must be >= 1, got {self.batch_size!r}."
            )
        if self.early_stopping_patience < 0:
            raise TorchConfigError(
                "early_stopping_patience must be >= 0, got "
                f"{self.early_stopping_patience!r}."
            )
        for name in ("learning_rate", "weight_decay", "validation_fraction", "min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TorchConfigError(f"{name} must be a number, got {value!r}.")
        if float(self.learning_rate) <= 0.0:
            raise TorchConfigError(
                f"learning_rate must be > 0, got {self.learning_rate!r}."
            )
        if float(self.weight_decay) < 0.0:
            raise TorchConfigError(
                f"weight_decay must be >= 0, got {self.weight_decay!r}."
            )
        if not 0.0 <= float(self.validation_fraction) < 0.5:
            raise TorchConfigError(
                "validation_fraction must be in [0, 0.5), got "
                f"{self.validation_fraction!r}."
            )
        if float(self.min_delta) < 0.0:
            raise TorchConfigError(f"min_delta must be >= 0, got {self.min_delta!r}.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TorchConfigError(f"seed must be an integer, got {self.seed!r}.")
        if self.pos_weight is not None:
            if isinstance(self.pos_weight, bool) or not isinstance(
                self.pos_weight, (int, float)
            ):
                raise TorchConfigError(
                    f"pos_weight must be a number or None, got {self.pos_weight!r}."
                )
            if float(self.pos_weight) <= 0.0:
                raise TorchConfigError(
                    f"pos_weight must be > 0, got {self.pos_weight!r}."
                )
        if not isinstance(self.shuffle, bool):
            raise TorchConfigError(f"shuffle must be a bool, got {self.shuffle!r}.")

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "validation_fraction": float(self.validation_fraction),
            "early_stopping_patience": int(self.early_stopping_patience),
            "min_delta": float(self.min_delta),
            "seed": int(self.seed),
            "pos_weight": None if self.pos_weight is None else float(self.pos_weight),
            "shuffle": bool(self.shuffle),
        }


def smoke_training_config() -> TrainingConfig:
    """Return the reduced-epoch smoke configuration used by tests and the CLI."""
    return TrainingConfig(
        epochs=SMOKE_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        early_stopping_patience=2,
        seed=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceResolution:
    """The resolved compute device for one neural run, with its provenance."""

    requested: str
    device: str
    is_gpu: bool
    gpu_available: bool
    torch_available: bool
    reason: str
    warnings: tuple[str, ...] = ()
    torch_version: str | None = None
    hip_version: str | None = None
    device_name: str | None = None
    gcn_arch_name: str | None = None

    @property
    def cpu_fallback(self) -> bool:
        """``True`` when the run executes on CPU (usable GPU or not)."""
        return self.device == DEVICE_CPU

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "device": self.device,
            "is_gpu": bool(self.is_gpu),
            "gpu_available": bool(self.gpu_available),
            "torch_available": bool(self.torch_available),
            "cpu_fallback": self.cpu_fallback,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "torch_version": self.torch_version,
            "hip_version": self.hip_version,
            "device_name": self.device_name,
            "gcn_arch_name": self.gcn_arch_name,
        }


def resolve_device(
    *,
    torch_module: object | None = None,
    torch_importer: Callable[[str], object] = importlib.import_module,
    prefer_gpu: bool = True,
    gfx_target: str = TARGET_GPU_ARCH,
) -> DeviceResolution:
    """Resolve ``cuda:0`` or ``cpu`` for a neural run, and record why.

    The probe is :func:`heart.gpu.device_check.probe_torch`, so the resolution
    agrees with the S05/T01 device check by construction. The function never
    raises on a missing or unusable GPU: that is a documented fallback and is
    logged at ``WARNING``. ``prefer_gpu=False`` forces the CPU path and logs
    :data:`CPU_FORCED_WARNING`.
    """
    probe = probe_torch(torch_module, importer=torch_importer)
    warnings: list[str] = []
    requested = "gpu" if prefer_gpu else "cpu"

    if not probe.torch_available:
        warnings.append(
            "torch is not importable in this environment; neural training "
            f"cannot run here ({probe.error})."
        )
        if prefer_gpu:
            warnings.append(CPU_FALLBACK_WARNING)
        device = DEVICE_CPU
        reason = probe.error or "torch unavailable"
    elif not prefer_gpu:
        warnings.append(CPU_FORCED_WARNING)
        device = DEVICE_CPU
        reason = "CPU requested explicitly (GPU path disabled)."
    elif probe.gpu_available:
        device = DEVICE_CUDA
        reason = (
            f"{probe.device_name or 'GPU'} usable by torch "
            f"(gcnArchName={probe.gcn_arch_name or 'unknown'})."
        )
    else:
        warnings.append(CPU_FALLBACK_WARNING)
        device = DEVICE_CPU
        reason = "GPU not usable by torch (device_count=0 or unavailable)."
        if (
            probe.gcn_arch_name
            and probe.gcn_arch_name.lower() != gfx_target.lower()
        ):
            warnings.append(
                f"torch resolved CPU but reports architecture "
                f"{probe.gcn_arch_name!r}, not the target {gfx_target!r}."
            )

    resolution = DeviceResolution(
        requested=requested,
        device=device,
        is_gpu=device != DEVICE_CPU,
        gpu_available=bool(probe.gpu_available),
        torch_available=bool(probe.torch_available),
        reason=reason,
        warnings=tuple(warnings),
        torch_version=probe.torch_version,
        hip_version=probe.hip_version,
        device_name=probe.device_name,
        gcn_arch_name=probe.gcn_arch_name,
    )
    logger.info(
        "Resolved MLP compute device: %s (requested=%s, gpu_available=%s, "
        "torch=%s) — %s",
        resolution.device,
        resolution.requested,
        resolution.gpu_available,
        resolution.torch_version or "unavailable",
        resolution.reason,
    )
    for warning in resolution.warnings:
        logger.warning("%s", warning)
    return resolution


# ---------------------------------------------------------------------------
# Feature representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureRepresentation:
    """The fitted selected-feature pipeline and the design matrix it produces."""

    mode_name: str
    is_baseline: bool
    n_engineered_columns: int
    feature_names: tuple[str, ...]
    n_features: int
    preprocessor: Pipeline = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_features != len(self.feature_names):
            raise TorchRepresentationError(
                f"n_features ({self.n_features}) does not match the number of "
                f"feature names ({len(self.feature_names)})."
            )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform a raw feature frame into the design matrix the MLP sees."""
        if not isinstance(frame, pd.DataFrame):
            raise TorchDataError(
                f"frame must be a pandas.DataFrame, got {type(frame).__name__}."
            )
        missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing:
            raise TorchDataError(
                f"frame is missing feature column(s) {missing}; pass the raw "
                "schema feature columns."
            )
        matrix = self.preprocessor.transform(frame[list(FEATURE_COLUMNS)])
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features:
            raise TorchRepresentationError(
                f"Transform produced shape {array.shape}, expected "
                f"(n, {self.n_features})."
            )
        if not np.isfinite(array).all():
            raise TorchRepresentationError(
                "Transform produced non-finite values in the design matrix."
            )
        return array

    def to_dict(self) -> dict[str, object]:
        return {
            "mode_name": self.mode_name,
            "is_baseline": bool(self.is_baseline),
            "n_engineered_columns": int(self.n_engineered_columns),
            "n_features": int(self.n_features),
            "feature_names": list(self.feature_names),
        }


def _require_frame(frame: object, *, role: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TorchDataError(
            f"{role} must be a pandas.DataFrame, got {type(frame).__name__}."
        )
    if frame.empty:
        raise TorchDataError(f"{role} is empty; cannot train on zero rows.")
    missing = [
        column for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise TorchDataError(
            f"{role} is missing declared column(s) {missing}; load the S01 "
            "split with heart.data.split.load_split_frames."
        )
    return frame


def resolve_feature_mode(
    *,
    mode: FeatureMode | None = None,
    selection_path: str | Path = DEFAULT_SELECTION_LEDGER_PATH,
    allow_baseline_fallback: bool = True,
) -> FeatureMode:
    """Resolve the feature mode the MLP trains on.

    An explicit ``mode`` wins. Otherwise the committed S04 selection ledger is
    read and its ``selected`` mode returned. A missing or unreadable ledger is
    a **loud** event: it is logged at ``WARNING`` and the baseline feature set
    is used, unless ``allow_baseline_fallback`` is ``False`` (then it raises).
    """
    if mode is not None:
        if not isinstance(mode, FeatureMode):
            raise TorchConfigError(
                f"mode must be a FeatureMode, got {type(mode).__name__}."
            )
        return mode

    path = Path(selection_path)
    if path.exists():
        try:
            selection = read_selection(path)
        except FeatureSelectionError as exc:
            if not allow_baseline_fallback:
                raise TorchRepresentationError(
                    f"Could not read the feature-selection ledger at {path}: {exc}"
                ) from exc
            logger.warning(
                "Could not read the feature-selection ledger at %s (%s); "
                "falling back to the baseline feature set.",
                path,
                exc,
            )
            return FeatureMode.baseline()
        logger.info(
            "Using the selected feature representation %r from %s "
            "(%d kept engineered transform(s)).",
            selection.selected_mode.name,
            path,
            selection.n_kept,
        )
        return selection.selected_mode

    if not allow_baseline_fallback:
        raise TorchRepresentationError(
            f"No feature-selection ledger at {path} and baseline fallback is "
            "disabled."
        )
    logger.warning(
        "No feature-selection ledger at %s; using the baseline feature set.",
        path,
    )
    return FeatureMode.baseline()


def build_feature_pipeline(mode: FeatureMode) -> Pipeline:
    """Build the transformer-only feature chain for ``mode`` (no classifier).

    The chain is the S04 ablation chain with its classifier step removed, so the
    MLP's inputs are exactly the selected representation's design matrix.
    """
    if not isinstance(mode, FeatureMode):
        raise TorchConfigError(
            f"mode must be a FeatureMode, got {type(mode).__name__}."
        )
    try:
        base = build_ablation_pipeline(mode)
    except AblationError as exc:
        raise TorchRepresentationError(
            f"Could not build the feature pipeline for mode {mode.name!r}: {exc}"
        ) from exc
    return Pipeline(steps=[*base.steps[:-1]])


def fit_feature_representation(
    train_frame: pd.DataFrame,
    *,
    mode: FeatureMode | None = None,
    selection_path: str | Path = DEFAULT_SELECTION_LEDGER_PATH,
    allow_baseline_fallback: bool = True,
) -> FeatureRepresentation:
    """Fit the selected feature chain on the training rows only.

    Fitting on the training frame alone is the leakage-safety contract shared
    with every classical model: the held-out split never influences the
    transformers.
    """
    frame = _require_frame(train_frame, role="train_frame")
    resolved_mode = resolve_feature_mode(
        mode=mode,
        selection_path=selection_path,
        allow_baseline_fallback=allow_baseline_fallback,
    )
    pipeline = build_feature_pipeline(resolved_mode)
    try:
        pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_COLUMN])
        transformer = pipeline.named_steps["features"]
        names = tuple(str(name) for name in transformer.get_feature_names_out())
    except (AblationError, ValueError, TypeError) as exc:
        raise TorchRepresentationError(
            f"Could not fit the feature representation {resolved_mode.name!r} "
            f"on {len(frame)} training row(s): {type(exc).__name__}: {exc}"
        ) from exc
    representation = FeatureRepresentation(
        mode_name=resolved_mode.name,
        is_baseline=resolved_mode.is_baseline,
        n_engineered_columns=resolved_mode.n_engineered_columns,
        feature_names=names,
        n_features=len(names),
        preprocessor=pipeline,
    )
    logger.info(
        "Fitted feature representation %r on %d row(s): %d input column(s) -> "
        "%d design feature(s)%s",
        representation.mode_name,
        len(frame),
        len(FEATURE_COLUMNS) + representation.n_engineered_columns,
        representation.n_features,
        " (baseline)" if representation.is_baseline else "",
    )
    return representation


# ---------------------------------------------------------------------------
# Training history / classifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingHistory:
    """Per-epoch losses and the early-stopping outcome of one training run."""

    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    best_val_loss: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "train_losses": [float(x) for x in self.train_losses],
            "val_losses": [float(x) for x in self.val_losses],
            "best_epoch": int(self.best_epoch),
            "epochs_run": int(self.epochs_run),
            "stopped_early": bool(self.stopped_early),
            "best_val_loss": None if self.best_val_loss is None else float(self.best_val_loss),
        }


class MLPClassifier:
    """A fitted MLP plus its feature representation, evaluated like any model.

    The object exposes the ``predict`` / ``predict_proba`` surface
    :func:`heart.eval.contract.evaluate` requires, so the neural model goes
    through the identical evaluation path (and therefore produces the identical
    canonical metric dict) as every classical model.
    """

    def __init__(
        self,
        *,
        representation: FeatureRepresentation,
        mlp_config: MLPConfig,
        model: object,
        device_resolution: DeviceResolution,
        torch_module: object,
        history: TrainingHistory | None = None,
    ) -> None:
        if not isinstance(representation, FeatureRepresentation):
            raise TorchConfigError(
                f"representation must be a FeatureRepresentation, got "
                f"{type(representation).__name__}."
            )
        if not isinstance(mlp_config, MLPConfig):
            raise TorchConfigError(
                f"mlp_config must be an MLPConfig, got {type(mlp_config).__name__}."
            )
        self.representation = representation
        self.mlp_config = mlp_config
        self.model = model
        self.device_resolution = device_resolution
        self.torch_module = torch_module
        self.history = history
        self.classes_ = np.array([0, 1])

    @property
    def n_features(self) -> int:
        return self.representation.n_features

    @property
    def device(self) -> str:
        return self.device_resolution.device

    def _positive_probabilities(self, X: object) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TorchDataError(
                "predict/predict_proba expect a pandas DataFrame of raw schema "
                f"feature columns, got {type(X).__name__}."
            )
        matrix = self.representation.transform(X)
        torch = self.torch_module
        tensor = torch.as_tensor(np.asarray(matrix, dtype=np.float32))
        tensor = tensor.to(self.device_resolution.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
        values = np.asarray(
            logits.detach().cpu().numpy(), dtype=float
        ).reshape(-1)
        return _sigmoid(values)

    def predict_proba(self, X: object) -> np.ndarray:
        positive = self._positive_probabilities(X)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: object) -> np.ndarray:
        return (self._positive_probabilities(X) >= 0.5).astype(int)

    def parameter_dict(self) -> dict[str, object]:
        """All architecture and training parameters (for MLflow logging)."""
        params = dict(self.mlp_config.to_params())
        if self.history is not None:
            params["epochs_run"] = int(self.history.epochs_run)
            params["best_epoch"] = int(self.history.best_epoch)
        return params

    def to_dict(self) -> dict[str, object]:
        return {
            "mlp_config": self.mlp_config.to_dict(),
            "representation": self.representation.to_dict(),
            "device": self.device_resolution.to_dict(),
            "history": None if self.history is None else self.history.to_dict(),
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _split_validation(
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split off a stratified validation set for early stopping (or none)."""
    n = len(labels)
    labels_int = labels.astype(int)
    if (
        float(config.validation_fraction) <= 0.0
        or n < 4
        or len(np.unique(labels_int)) < 2
        or int(np.bincount(labels_int).min()) < 2
    ):
        return features, labels, None, None
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=float(config.validation_fraction),
            random_state=config.seed,
            stratify=labels_int,
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=float(config.validation_fraction),
            random_state=config.seed,
        )
    if len(y_val) < 2 or len(y_train) < 2:
        return features, labels, None, None
    return x_train, y_train, x_val, y_val


def _fit_torch(
    torch: object,
    model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    config: TrainingConfig,
    device: str,
) -> TrainingHistory:
    """The real PyTorch training loop (batching, AdamW, early stopping)."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    criterion_kwargs: dict[str, object] = {}
    if config.pos_weight is not None:
        criterion_kwargs["pos_weight"] = torch.tensor(
            [float(config.pos_weight)], dtype=torch.float32, device=device_obj
        )
    criterion = torch.nn.BCEWithLogitsLoss(**criterion_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    y_train = y_train.astype(np.float32)
    has_val = x_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        y_val = y_val.astype(np.float32)

    x_val_tensor = (
        torch.as_tensor(np.asarray(x_val, dtype=np.float32)).to(device_obj)
        if has_val
        else None
    )
    y_val_tensor = (
        torch.as_tensor(np.asarray(y_val, dtype=np.float32)).to(device_obj)
        if has_val
        else None
    )

    rng = np.random.default_rng(config.seed)
    n = len(y_train)
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    best_epoch = -1
    patience = 0
    stopped_early = False
    train_losses: list[float] = []
    val_losses: list[float] = []
    epochs_run = 0

    for epoch in range(int(config.epochs)):
        model.train()
        order = rng.permutation(n) if config.shuffle else np.arange(n)
        running = 0.0
        n_batches = 0
        for start in range(0, n, int(config.batch_size)):
            index = order[start : start + int(config.batch_size)]
            x_batch = torch.as_tensor(
                np.asarray(x_train[index], dtype=np.float32)
            ).to(device_obj)
            y_batch = torch.as_tensor(y_train[index]).to(device_obj)
            optimizer.zero_grad()
            logits = model(x_batch).reshape(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu().item())
            n_batches += 1
        epochs_run = epoch + 1
        train_losses.append(running / max(n_batches, 1))

        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val_tensor).reshape(-1)
                val_loss = float(criterion(val_logits, y_val_tensor).detach().cpu().item())
            val_losses.append(val_loss)
            if val_loss < best_val - float(config.min_delta):
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if (
                    int(config.early_stopping_patience) > 0
                    and patience >= int(config.early_stopping_patience)
                ):
                    stopped_early = True
                    break
        else:
            best_epoch = epoch
            best_val = train_losses[-1]

    if has_val and best_epoch >= 0:
        model.load_state_dict(best_state)

    return TrainingHistory(
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
        best_epoch=int(best_epoch),
        epochs_run=int(epochs_run),
        stopped_early=bool(stopped_early),
        best_val_loss=None if best_val == math.inf else float(best_val),
    )


def train_mlp_classifier(
    train_frame: pd.DataFrame,
    *,
    mlp_config: MLPConfig | None = None,
    training_config: TrainingConfig | None = None,
    mode: FeatureMode | None = None,
    selection_path: str | Path = DEFAULT_SELECTION_LEDGER_PATH,
    device_resolution: DeviceResolution | None = None,
    torch_module: object | None = None,
    torch_importer: Callable[[str], object] = importlib.import_module,
    prefer_gpu: bool = True,
) -> MLPClassifier:
    """Fit the MLP on ``train_frame`` and return the evaluated-surface model.

    The device is resolved **before** training and recorded on the returned
    classifier; a CPU fallback logs an explicit ``WARNING`` and still completes.
    """
    frame = _require_frame(train_frame, role="train_frame")
    resolved_training = training_config or TrainingConfig()
    if not isinstance(resolved_training, TrainingConfig):
        raise TorchConfigError(
            f"training_config must be a TrainingConfig, got "
            f"{type(resolved_training).__name__}."
        )

    torch = torch_module if torch_module is not None else import_torch(torch_importer)

    resolution = device_resolution or resolve_device(
        torch_module=torch,
        torch_importer=torch_importer,
        prefer_gpu=prefer_gpu,
    )
    if not isinstance(resolution, DeviceResolution):
        raise TorchConfigError(
            f"device_resolution must be a DeviceResolution, got "
            f"{type(resolution).__name__}."
        )
    if not resolution.torch_available:
        raise TorchUnavailableError(
            "torch is not importable; cannot train the MLP. "
            f"Reason: {resolution.reason}"
        )

    representation = fit_feature_representation(
        frame, mode=mode, selection_path=selection_path
    )

    if mlp_config is None:
        mlp_config = MLPConfig(input_dim=representation.n_features)
    elif not isinstance(mlp_config, MLPConfig):
        raise TorchConfigError(
            f"mlp_config must be an MLPConfig, got {type(mlp_config).__name__}."
        )
    elif int(mlp_config.input_dim) != representation.n_features:
        # input width is a property of the fitted representation, not a tunable;
        # bind the real width rather than failing a CLI that passes a template.
        logger.debug(
            "Rebinding MLP input_dim %d -> %d to match the feature "
            "representation.",
            mlp_config.input_dim,
            representation.n_features,
        )
        mlp_config = mlp_config.with_input_dim(representation.n_features)

    features = representation.transform(frame)
    labels = frame[TARGET_COLUMN].to_numpy()
    x_train, y_train, x_val, y_val = _split_validation(
        features, labels, resolved_training
    )

    # Deterministic weight initialisation: torch's global RNG is otherwise
    # unseeded, which would make two identical runs diverge.
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(int(resolved_training.seed))

    model = build_mlp(mlp_config, torch_module=torch)
    logger.info(
        "Training MLP on %s: mode=%s features=%d epochs=%d batch=%d lr=%g "
        "train_rows=%d val_rows=%s",
        resolution.device,
        representation.mode_name,
        representation.n_features,
        resolved_training.epochs,
        resolved_training.batch_size,
        resolved_training.learning_rate,
        len(y_train),
        "none" if y_val is None else str(len(y_val)),
    )
    history = _fit_torch(
        torch,
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        resolved_training,
        resolution.device,
    )
    logger.info(
        "Trained MLP on %s: epochs_run=%d best_epoch=%d best_val_loss=%s "
        "stopped_early=%s",
        resolution.device,
        history.epochs_run,
        history.best_epoch,
        "n/a" if history.best_val_loss is None else f"{history.best_val_loss:.4f}",
        history.stopped_early,
    )
    if resolution.device == DEVICE_CPU:
        logger.warning(
            "MLP training completed on CPU (device=%s) — CPU fallback in "
            "effect for this run.",
            resolution.device,
        )

    return MLPClassifier(
        representation=representation,
        mlp_config=mlp_config,
        model=model,
        device_resolution=resolution,
        torch_module=torch,
        history=history,
    )


# ---------------------------------------------------------------------------
# End-to-end run + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingResult:
    """Everything one end-to-end MLP training+eval run produced."""

    device: str
    used_gpu: bool
    cpu_fallback: bool
    mode_name: str
    is_baseline_features: bool
    n_features: int
    split_version: str
    train_rows: int
    test_rows: int
    mlp_config: MLPConfig
    training_config: TrainingConfig
    history: TrainingHistory
    metrics: dict[str, object]
    device_resolution: DeviceResolution
    duration_seconds: float
    generated_at: str

    @property
    def primary_metric(self) -> float:
        return float(self.metrics[PRIMARY_METRIC])

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "used_gpu": bool(self.used_gpu),
            "cpu_fallback": bool(self.cpu_fallback),
            "mode_name": self.mode_name,
            "is_baseline_features": bool(self.is_baseline_features),
            "n_features": int(self.n_features),
            "split_version": self.split_version,
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "mlp_config": self.mlp_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "history": self.history.to_dict(),
            "metrics": self.metrics,
            "device_resolution": self.device_resolution.to_dict(),
            "duration_seconds": float(self.duration_seconds),
            "generated_at": self.generated_at,
        }


def run_training(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    mlp_config: MLPConfig | None = None,
    training_config: TrainingConfig | None = None,
    mode: FeatureMode | None = None,
    selection_path: str | Path = DEFAULT_SELECTION_LEDGER_PATH,
    prefer_gpu: bool = True,
    torch_module: object | None = None,
    torch_importer: Callable[[str], object] = importlib.import_module,
    split_version: str = SPLIT_VERSION,
    device_resolution: DeviceResolution | None = None,
) -> tuple[MLPClassifier, TrainingResult]:
    """Train the MLP on ``train_frame`` and score it once on ``test_frame``.

    Returns the fitted :class:`MLPClassifier` and a :class:`TrainingResult`
    recording the device used, the feature mode, the training history, and the
    canonical metric dict produced by the shared evaluation contract.
    """
    start = time.perf_counter()
    train = _require_frame(train_frame, role="train_frame")
    test = _require_frame(test_frame, role="test_frame")
    classifier = train_mlp_classifier(
        train,
        mlp_config=mlp_config,
        training_config=training_config,
        mode=mode,
        selection_path=selection_path,
        device_resolution=device_resolution,
        torch_module=torch_module,
        torch_importer=torch_importer,
        prefer_gpu=prefer_gpu,
    )
    split = evaluation_split_from_frames(test, name=f"{split_version}/test")
    metrics = evaluate(classifier, split)
    duration = round(time.perf_counter() - start, 6)
    resolution = classifier.device_resolution
    result = TrainingResult(
        device=resolution.device,
        used_gpu=resolution.is_gpu,
        cpu_fallback=resolution.cpu_fallback,
        mode_name=classifier.representation.mode_name,
        is_baseline_features=classifier.representation.is_baseline,
        n_features=classifier.n_features,
        split_version=split_version,
        train_rows=int(len(train)),
        test_rows=int(len(test)),
        mlp_config=classifier.mlp_config,
        training_config=training_config or TrainingConfig(),
        history=classifier.history or TrainingHistory((), (), -1, 0, False, None),
        metrics=metrics,
        device_resolution=resolution,
        duration_seconds=duration,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    logger.info(
        "MLP run finished: device=%s mode=%s features=%d %s=%.4f "
        "(train=%d test=%d, %.2fs)",
        result.device,
        result.mode_name,
        result.n_features,
        PRIMARY_METRIC,
        result.primary_metric,
        result.train_rows,
        result.test_rows,
        result.duration_seconds,
    )
    return classifier, result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_training_report(result: TrainingResult) -> str:
    """Render the device-stamped markdown training report.

    The first bold line is the contract: it states which compute device the run
    used, so the GPU question is answerable from the artifact alone.
    """
    if not isinstance(result, TrainingResult):
        raise TorchReportError(
            f"result must be a TrainingResult, got {type(result).__name__}."
        )
    resolution = result.device_resolution
    lines: list[str] = [
        "# MLP training run",
        "",
        "_Generated by `heart.models.train_torch`. Do not edit by hand._",
        "",
        f"**device used: {result.device}**"
        + (" (GPU)" if result.used_gpu else " (CPU)"),
        f"**cpu fallback: {'YES' if result.cpu_fallback else 'NO'}**",
        "",
        "## Device",
        "",
        f"- requested: {resolution.requested}",
        f"- resolved device: {result.device}",
        f"- reason: {resolution.reason}",
        f"- torch available: {'yes' if resolution.torch_available else 'no'}",
        f"- torch version: {resolution.torch_version or 'unavailable'}",
        f"- HIP version: {resolution.hip_version or 'unknown'}",
        f"- device name: {resolution.device_name or 'none'}",
        f"- gcnArchName: {resolution.gcn_arch_name or 'none'}",
        "",
        "## Run",
        "",
        f"- split: `{result.split_version}` (train {result.train_rows} / "
        f"test {result.test_rows})",
        f"- feature mode: `{result.mode_name}`"
        + (" (baseline)" if result.is_baseline_features else ""),
        f"- design features: {result.n_features}",
        f"- epochs run: {result.history.epochs_run} "
        f"(best epoch {result.history.best_epoch})",
        f"- early stopped: {'yes' if result.history.stopped_early else 'no'}",
        f"- best val loss: "
        + (
            "n/a"
            if result.history.best_val_loss is None
            else f"{result.history.best_val_loss:.4f}"
        ),
        f"- train loss (last): "
        + (f"{result.history.train_losses[-1]:.4f}" if result.history.train_losses else "n/a"),
        f"- duration: {result.duration_seconds:.2f}s",
        f"- generated at: {result.generated_at}",
        "",
        "## Architecture",
        "",
        "```",
        describe_mlp(result.mlp_config),
        "```",
        "",
        "## Metrics",
        "",
        f"| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        PRIMARY_METRIC,
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "specificity",
        "npv",
        "brier_score",
    ):
        if key in result.metrics:
            lines.append(f"| `{key}` | {float(result.metrics[key]):.4f} |")
    n_samples = result.metrics.get("confusion_matrix")
    if isinstance(n_samples, Mapping) and "n_samples" in n_samples:
        lines.append(f"| `n_samples` | {int(n_samples['n_samples'])} |")
    lines.append("")

    if resolution.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in resolution.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "containers/run-rocm.sh shell",
            "python -m heart.models.train_torch --smoke",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_report(path: str | Path, content: str) -> Path:
    """Atomically write ``content``, raising TorchReportError on fs failure."""
    destination = Path(path)
    try:
        atomic_write_text(destination, content)
    except OSError as exc:
        raise TorchReportError(
            f"Could not write the MLP training report to {destination}: {exc}"
        ) from exc
    return destination


def write_training_report(result: TrainingResult, path: str | Path) -> Path:
    """Atomically write the markdown training report for ``result``."""
    if not isinstance(result, TrainingResult):
        raise TorchReportError(
            f"result must be a TrainingResult, got {type(result).__name__}."
        )
    destination = _atomic_write_report(path, render_training_report(result))
    logger.info("Wrote MLP training report to %s", destination)
    return destination


def write_training_json(result: TrainingResult, path: str | Path) -> Path:
    """Atomically write the machine-readable training result sidecar."""
    if not isinstance(result, TrainingResult):
        raise TorchReportError(
            f"result must be a TrainingResult, got {type(result).__name__}."
        )
    destination = write_json_document(
        path,
        result.to_dict(),
        error_factory=TorchReportError,
        label="training result",
        # The original shared writer raised the report-flavoured message on
        # the file-system failure path; keep it byte-exact.
        write_label="MLP training report",
    )
    logger.info("Wrote MLP training JSON to %s", destination)
    return destination


def describe_training_result(result: TrainingResult) -> str:
    """Render the one-block CLI summary of a training run."""
    if not isinstance(result, TrainingResult):
        raise TorchReportError(
            f"result must be a TrainingResult, got {type(result).__name__}."
        )
    return "\n".join(
        [
            f"device used: {result.device}"
            + (" (GPU)" if result.used_gpu else " (CPU)"),
            f"cpu fallback: {'yes' if result.cpu_fallback else 'no'}",
            f"torch: {result.device_resolution.torch_version or 'unavailable'}"
            + (
                f" (HIP {result.device_resolution.hip_version})"
                if result.device_resolution.hip_version
                else ""
            ),
            f"feature mode: {result.mode_name}"
            + (" (baseline)" if result.is_baseline_features else ""),
            f"features: {result.n_features}",
            f"epochs run: {result.history.epochs_run} "
            f"(best {result.history.best_epoch})",
            f"{PRIMARY_METRIC}: {result.primary_metric:.4f}",
            f"split: {result.split_version} "
            f"(train {result.train_rows} / test {result.test_rows})",
        ]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_hidden_sizes(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise TorchConfigError(
            f"--hidden-sizes must be comma-separated integers, got {raw!r}."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.models.train_torch",
        description=(
            "Train the MLP on the selected feature representation and the S01 "
            "split, recording the compute device used."
        ),
    )
    parser.add_argument(
        "--split-version", default=SPLIT_VERSION, help="S01 split version."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Reduced-epoch smoke configuration ({SMOKE_EPOCHS} epochs).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=None, help="AdamW learning rate.")
    parser.add_argument(
        "--hidden-sizes",
        default=None,
        help="Comma-separated hidden layer widths (default: 64,32).",
    )
    parser.add_argument(
        "--dropout", type=float, default=None, help="Dropout probability."
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "tanh", "gelu", "leaky_relu"),
        default=None,
        help="Hidden activation (default: relu).",
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Add BatchNorm1d after each hidden linear layer.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU path (GPU acceleration disabled).",
    )
    parser.add_argument(
        "--baseline-features",
        action="store_true",
        help="Ignore the feature-selection ledger and use the baseline feature set.",
    )
    parser.add_argument(
        "--selection-path",
        default=str(DEFAULT_SELECTION_LEDGER_PATH),
        help=(
            "Feature-selection ledger to read the selected mode from "
            f"(default: {DEFAULT_SELECTION_LEDGER_PATH})."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Markdown report destination (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--json-path",
        default=str(DEFAULT_JSON_PATH),
        help=f"JSON sidecar destination (default: {DEFAULT_JSON_PATH}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of the summary block.",
    )
    return parser


def _build_configs(args: argparse.Namespace) -> tuple[MLPConfig | None, TrainingConfig]:
    base = smoke_training_config() if args.smoke else TrainingConfig()
    training = TrainingConfig(
        epochs=base.epochs if args.epochs is None else args.epochs,
        batch_size=base.batch_size if args.batch_size is None else args.batch_size,
        learning_rate=base.learning_rate if args.lr is None else args.lr,
        weight_decay=base.weight_decay,
        validation_fraction=base.validation_fraction,
        early_stopping_patience=base.early_stopping_patience,
        min_delta=base.min_delta,
        seed=base.seed,
        pos_weight=base.pos_weight,
        shuffle=base.shuffle,
    )
    hidden = _parse_hidden_sizes(args.hidden_sizes)
    if hidden is None and args.dropout is None and args.activation is None and not args.batch_norm:
        return None, training
    mlp_kwargs: dict[str, object] = {}
    if hidden is not None:
        mlp_kwargs["hidden_sizes"] = hidden
    if args.dropout is not None:
        mlp_kwargs["dropout"] = args.dropout
    if args.activation is not None:
        mlp_kwargs["activation"] = args.activation
    mlp_kwargs["batch_norm"] = bool(args.batch_norm)
    # input_dim is patched after the representation is fitted.
    base_config = MLPConfig(input_dim=1, **mlp_kwargs)  # type: ignore[arg-type]
    return base_config, training


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        mlp_config, training_config = _build_configs(args)
    except (TorchConfigError, TorchTrainingError) as exc:
        print(f"configuration error: {exc}")
        return 2

    try:
        train_frame, test_frame = load_split_frames(args.split_version)
    except SplitError as exc:
        print(f"could not load split {args.split_version!r}: {exc}")
        return 1

    mode = None
    if args.baseline_features:
        mode = FeatureMode.baseline()

    try:
        classifier, result = run_training(
            train_frame,
            test_frame,
            mlp_config=mlp_config,
            training_config=training_config,
            mode=mode,
            selection_path=args.selection_path,
            prefer_gpu=not args.cpu,
            split_version=args.split_version,
        )
    except (TorchTrainingError, TorchUnavailableError, MLPError) as exc:
        print(f"training error: {exc}")
        return 1

    try:
        write_training_report(result, args.report_path)
        write_training_json(result, args.json_path)
    except TorchReportError as exc:
        print(f"report error: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=json_default))
    else:
        print(describe_training_result(result))
        print(f"report: {args.report_path}")
        print(f"json:   {args.json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
