"""The deliberate MLP architecture set and its tuned Optuna spaces (S05/T03).

This module owns **what** the neural sweep tunes: a small, deliberate set of
MLP architectures — a few properly tuned configurations rather than an
exhaustive search — plus, for each, the Optuna search space over the training
hyperparameters that make that architecture perform.

Design
------
* :class:`MLPSpec` is the architecture registry entry: a fixed network shape
  (hidden widths, activation, batch normalisation, default dropout) and the
  tuned training knobs it searches over (learning rate, weight decay, dropout,
  batch size). :data:`MLP_ARCHITECTURES` declares three entries —
  ``mlp-shallow``, ``mlp-wide``, and ``mlp-deep`` — so the sweep cost is
  bounded and each entry probes a different capacity/regularisation regime.
* :func:`to_model_spec` converts an :class:`MLPSpec` into a
  :class:`~heart.models.registry.ModelSpec` whose ``estimator_factory`` builds
  :class:`TunableMLPClassifier` — a scikit-learn-style adapter around the
  :mod:`heart.models.train_torch` training loop. Because the adapter exposes
  ``fit`` / ``predict`` / ``predict_proba`` and the spec declares
  ``preprocessing=False``, the **existing** tuning runner
  (:func:`heart.tuning.runner.run_study`) drives the neural sweep unchanged:
  trials are sampled from the space, scored through the shared evaluation
  contract, and logged to MLflow under exactly the same convention as the
  classical battery.
* :func:`mlp_trainer` is the production scoring leaf: it merges fixed and tuned
  parameters into a :class:`TunableMLPClassifier`, fits it on one fold's
  training rows, and scores the fold's held-out split through
  :func:`heart.eval.contract.evaluate`. The sweep and the runner accept the
  trainer as an injection seam so the full sweep (folds -> Optuna -> MLflow)
  is testable on a CPU-only host with a deterministic scorer.

torch is imported lazily: nothing here imports torch at module load, so the
specs, the adapter, and the trainer are constructible on a host without torch
(default environment); a missing torch raises the named
:class:`~heart.models.mlp.TorchUnavailableError` from :func:`fit` with install
guidance (the neural runs execute inside the S05/T01 ROCm container).

Observability
-------------
:func:`describe_mlp_sweep_plan` renders the architecture set and its search
spaces for diagnostics. The sweep itself records the compute device per
architecture in ``run_mlp_sweep``'s final MLflow runs (``device``,
``gpu_available``, ``cpu_fallback`` tags) and in its ledger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from heart.config import RANDOM_SEED
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.eval.contract import EvaluationSplit, evaluate
from heart.features.selection import (
    DEFAULT_LEDGER_PATH as DEFAULT_SELECTION_LEDGER_PATH,
)
from heart.models.mlp import (
    DEFAULT_ACTIVATION,
    DEFAULT_BATCH_NORM,
    DEFAULT_DROPOUT,
    MLPConfig,
)
from heart.models.registry import ModelSpec
from heart.models.spaces import (
    CATEGORICAL_KIND,
    FLOAT_KIND,
    ParamSpec,
    SearchSpace,
)
from heart.models.train_torch import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_WEIGHT_DECAY,
    TrainingConfig,
    train_mlp_classifier,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MLP_SWEEP_SIZE",
    "MLP_ARCHITECTURES",
    "MLP_MODEL_TYPES",
    "TUNED_PARAM_NAMES",
    "FAMILY_NEURAL",
    "MLPSpaceError",
    "MLPArchitectureError",
    "UnknownMLPArchitectureError",
    "MLPSpec",
    "build_mlp_architectures",
    "mlp_tuning_space",
    "fixed_training_params",
    "to_model_spec",
    "TunableMLPClassifier",
    "mlp_trainer",
    "describe_mlp_sweep_plan",
    "select_mlp_architectures",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: The neural family tag the MLP specs carry (registry family set, S05/T03).
FAMILY_NEURAL: str = "neural"

#: The tuned hyperparameter names (one per search-space order, asserted in
#: ``tests/test_mlp_sweep.py`` against the spec's declared keys).
TUNED_PARAM_NAMES: tuple[str, ...] = (
    "learning_rate",
    "weight_decay",
    "dropout",
    "batch_size",
)


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class MLPSpaceError(Exception):
    """Base class for every MLP architecture-set failure."""


class MLPArchitectureError(MLPSpaceError):
    """An :class:`MLPSpec` entry is malformed."""


class UnknownMLPArchitectureError(MLPSpaceError):
    """A requested architecture key is not part of the declared set."""


# ---------------------------------------------------------------------------
# The tuned training space
# ---------------------------------------------------------------------------


def mlp_tuning_space(*, dropout_default: float = DEFAULT_DROPOUT) -> SearchSpace:
    """The Optuna space over the training knobs tuned for one architecture.

    The architecture itself is fixed per :class:`MLPSpec` (that is what makes
    the set "a few properly tuned configurations"); what Optuna tunes here is
    how the network is trained: the AdamW learning rate and weight decay, the
    hidden dropout, and the mini-batch size. The regions are deliberately
    small so a handful of trials explore them thoroughly.
    """
    return SearchSpace(
        params=(
            ParamSpec(
                name="learning_rate",
                kind=FLOAT_KIND,
                low=1e-4,
                high=1e-2,
                log=True,
                default=DEFAULT_LEARNING_RATE,
            ),
            ParamSpec(
                name="weight_decay",
                kind=FLOAT_KIND,
                low=1e-6,
                high=1e-2,
                log=True,
                default=DEFAULT_WEIGHT_DECAY,
            ),
            ParamSpec(
                name="dropout",
                kind=FLOAT_KIND,
                low=0.0,
                high=0.5,
                default=float(dropout_default),
            ),
            ParamSpec(
                name="batch_size",
                kind=CATEGORICAL_KIND,
                choices=(16, 32, 64),
                default=DEFAULT_BATCH_SIZE,
            ),
        )
    )


def fixed_training_params(*, seed: int = RANDOM_SEED) -> dict[str, object]:
    """The training knobs pinned for every architecture (never tuned).

    Bounded epochs with early stopping keep each neural trial finite on both
    the GPU and the CPU-fallback path; the seed makes identical configurations
    reproducible.
    """
    return {
        "epochs": DEFAULT_EPOCHS,
        "validation_fraction": DEFAULT_VALIDATION_FRACTION,
        "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
        "min_delta": 0.0,
        "seed": int(seed),
        "pos_weight": None,
        "shuffle": True,
    }


# ---------------------------------------------------------------------------
# The architecture set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLPSpec:
    """One declared MLP architecture and its tuned training region.

    Parameters
    ----------
    model_type:
        Registry key used by the sweep and by CLI ``--architectures``.
    model_name:
        Human-readable name (slugified into the MLflow run name).
    description:
        One-sentence summary of the capacity/regularisation regime probed.
    hidden_sizes:
        Fixed hidden-layer widths, in order.
    activation:
        Fixed hidden activation (one of ``heart.models.mlp.ACTIVATIONS``).
    batch_norm:
        Fixed batch-normalisation policy (between linear and activation).
    dropout_default:
        The dropout value the tuned region centres on (dropout itself is
        tuned in :data:`TUNED_PARAM_NAMES`).
    """

    model_type: str
    model_name: str
    description: str
    hidden_sizes: tuple[int, ...]
    activation: str = DEFAULT_ACTIVATION
    batch_norm: bool = DEFAULT_BATCH_NORM
    dropout_default: float = DEFAULT_DROPOUT

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise MLPArchitectureError(
                f"model_type must be a non-empty string, got {self.model_type!r}."
            )
        if self.model_type.strip() != self.model_type or any(
            character.isspace() for character in self.model_type
        ):
            raise MLPArchitectureError(
                f"model_type {self.model_type!r} may not contain whitespace."
            )
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise MLPArchitectureError(
                f"model_name must be a non-empty string, got {self.model_name!r}."
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise MLPArchitectureError(
                f"description must be a non-empty string, got {self.description!r}."
            )
        # Delegating to MLPConfig validates the hidden sizes, activation,
        # dropout, and batch-norm values with the shared named errors.
        MLPConfig(
            input_dim=1,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout_default,
            activation=self.activation,
            batch_norm=self.batch_norm,
        )

    @property
    def depth(self) -> int:
        """Number of weight layers (hidden layers plus the output layer)."""
        return len(self.hidden_sizes) + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "description": self.description,
            "hidden_sizes": [int(size) for size in self.hidden_sizes],
            "activation": self.activation,
            "batch_norm": bool(self.batch_norm),
            "dropout_default": float(self.dropout_default),
            "tuned_parameters": list(TUNED_PARAM_NAMES),
        }


def build_mlp_architectures() -> tuple[MLPSpec, ...]:
    """The deliberate neural architecture set (three entries).

    Each entry pins a different capacity/regularisation regime and is tuned
    over the shared training space, so the sweep stays small and comparable:

    * ``mlp-shallow`` — one 32-unit hidden layer, light dropout (0.1): the
      cheap baseline neural capacity;
    * ``mlp-wide`` — one 128-unit layer, stronger dropout (0.3): capacity via
      width, regularised by dropout;
    * ``mlp-deep`` — three layers (64-32-16) with batch normalisation (0.2):
      depth with normalised activations.
    """
    return (
        MLPSpec(
            model_type="mlp-shallow",
            model_name="MLP (shallow)",
            description=(
                "Single 32-unit hidden layer with light dropout (0.1); the "
                "cheap neural baseline."
            ),
            hidden_sizes=(32,),
            activation="relu",
            batch_norm=False,
            dropout_default=0.1,
        ),
        MLPSpec(
            model_type="mlp-wide",
            model_name="MLP (wide)",
            description=(
                "Single 128-unit hidden layer with strong dropout (0.3); "
                "capacity by width, regularised by dropout."
            ),
            hidden_sizes=(128,),
            activation="relu",
            batch_norm=False,
            dropout_default=0.3,
        ),
        MLPSpec(
            model_type="mlp-deep",
            model_name="MLP (deep)",
            description=(
                "Three hidden layers (64-32-16) with batch normalisation and "
                "moderate dropout (0.2); depth with normalised activations."
            ),
            hidden_sizes=(64, 32, 16),
            activation="relu",
            batch_norm=True,
            dropout_default=0.2,
        ),
    )


#: The deliberate, configured architecture set (sweep order).
MLP_ARCHITECTURES: tuple[MLPSpec, ...] = build_mlp_architectures()

#: Architecture keys, in sweep order.
MLP_MODEL_TYPES: tuple[str, ...] = tuple(
    spec.model_type for spec in MLP_ARCHITECTURES
)

#: Number of configured architectures (asserted by ``tests/test_mlp_sweep.py``
#: against the sweep's final-run count).
MLP_SWEEP_SIZE: int = len(MLP_ARCHITECTURES)


def select_mlp_architectures(
    architectures: Sequence[str] | None = None,
) -> tuple[MLPSpec, ...]:
    """Resolve the architectures to sweep, in declared (sweep) order.

    ``None`` or an empty sequence selects every configured architecture; an
    unknown name raises :class:`UnknownMLPArchitectureError`. Like the
    classical battery's ``select_specs``, the result keeps the declared order
    (never the caller's), which is what makes the sweep's run order stable.
    """
    if architectures is None:
        return MLP_ARCHITECTURES
    if isinstance(architectures, (str, bytes)):
        raise MLPSpaceError(
            f"architectures must be a sequence of model_type strings, not a "
            f"bare string {architectures!r}; pass e.g. ['mlp-shallow']."
        )
    requested: list[str] = []
    for value in architectures:
        if not isinstance(value, str) or not value.strip():
            raise MLPSpaceError(
                f"architectures contains a non-string or blank key: {value!r}."
            )
        requested.append(value.strip())
    by_type = {spec.model_type: spec for spec in MLP_ARCHITECTURES}
    unknown = sorted(set(requested) - set(by_type))
    if unknown:
        raise UnknownMLPArchitectureError(
            f"Unknown MLP architecture(s) {unknown}; declared architectures "
            f"are {list(MLP_MODEL_TYPES)}."
        )
    wanted = set(requested)
    resolved = tuple(by_type[name] for name in MLP_MODEL_TYPES if name in wanted)
    if not resolved:
        raise MLPSpaceError(
            "The architecture selection is empty; nothing to sweep. Configured "
            f"architectures: {list(MLP_MODEL_TYPES)}."
        )
    return resolved


# ---------------------------------------------------------------------------
# Registry spec conversion
# ---------------------------------------------------------------------------


def to_model_spec(
    arch: MLPSpec,
    *,
    prefer_gpu: bool = True,
    selection_path: str | Path = DEFAULT_SELECTION_LEDGER_PATH,
) -> ModelSpec:
    """Convert an :class:`MLPSpec` into a runner-ready :class:`ModelSpec`.

    The returned spec pins the architecture (hidden sizes, activation, batch
    normalisation) and the fixed training knobs, tunes the shared training
    space, and declares ``preprocessing=False`` so the runner's ``build_pipeline``
    returns the bare :class:`TunableMLPClassifier` (the MLP fits its own S04
    selected feature representation inside ``fit``). ``prefer_gpu`` and
    ``selection_path`` are bound into the estimator factory so they never
    appear as logged hyperparameters.
    """
    if not isinstance(arch, MLPSpec):
        raise MLPSpaceError(
            f"arch must be an MLPSpec, got {type(arch).__name__}."
        )

    def _factory(params: Mapping[str, object]) -> "TunableMLPClassifier":
        return TunableMLPClassifier(
            **dict(params),
            prefer_gpu=prefer_gpu,
            selection_path=selection_path,
        )

    fixed_params: dict[str, object] = {
        "hidden_sizes": list(int(size) for size in arch.hidden_sizes),
        "activation": arch.activation,
        "batch_norm": bool(arch.batch_norm),
        **fixed_training_params(),
    }
    return ModelSpec(
        model_type=arch.model_type,
        model_name=arch.model_name,
        family=FAMILY_NEURAL,
        space=mlp_tuning_space(dropout_default=arch.dropout_default),
        estimator_factory=_factory,
        fixed_params=fixed_params,
        preprocessing=False,
        description=arch.description,
    )


# ---------------------------------------------------------------------------
# The scikit-learn-style torch adapter
# ---------------------------------------------------------------------------


class TunableMLPClassifier:
    """A torch MLP exposed as a scikit-learn-style estimator for the sweep.

    The constructor accepts exactly the *merged* spec parameters: the fixed
    architecture and training knobs (hidden sizes, activation, batch norm,
    epochs, validation fraction, ...) plus the tuned training knobs
    (learning rate, weight decay, dropout, batch size). :meth:`fit` trains
    :func:`heart.models.train_torch.train_mlp_classifier` on the raw schema
    feature frame, which internally fits the S04 selected feature
    representation on the training rows only (leakage-safe) and resolves and
    records the compute device; :meth:`predict` / :meth:`predict_proba` then
    score raw feature frames through the same shared evaluation surface the
    classical battery uses.

    torch is imported lazily, so the class is constructible on a CPU-only
    host; a missing torch raises
    :class:`~heart.models.mlp.TorchUnavailableError` from :meth:`fit`.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (64, 32),
        dropout: float = DEFAULT_DROPOUT,
        activation: str = DEFAULT_ACTIVATION,
        batch_norm: bool = DEFAULT_BATCH_NORM,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
        min_delta: float = 0.0,
        seed: int = RANDOM_SEED,
        pos_weight: float | None = None,
        shuffle: bool = True,
        # Run-mode knobs, not tuned hyperparameters:
        prefer_gpu: bool = True,
        selection_path: str | Path = DEFAULT_SELECTION_LEDGER_PATH,
    ) -> None:
        if isinstance(hidden_sizes, (str, bytes)) or not isinstance(
            hidden_sizes, Sequence
        ):
            raise MLPSpaceError(
                "hidden_sizes must be a sequence of positive integers, got "
                f"{hidden_sizes!r}."
            )
        sizes = tuple(int(size) for size in hidden_sizes)
        if not sizes or any(size < 1 for size in sizes):
            raise MLPSpaceError(
                f"hidden_sizes must contain only positive integers, got "
                f"{hidden_sizes!r}."
            )
        self.hidden_sizes = sizes
        self.dropout = float(dropout)
        self.activation = activation
        self.batch_norm = bool(batch_norm)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.validation_fraction = float(validation_fraction)
        self.early_stopping_patience = int(early_stopping_patience)
        self.min_delta = float(min_delta)
        self.seed = int(seed)
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        self.shuffle = bool(shuffle)
        self.prefer_gpu = bool(prefer_gpu)
        self.selection_path = Path(selection_path)
        #: The fitted :class:`~heart.models.train_torch.MLPClassifier`.
        self.classifier_ = None

    # -- construction ------------------------------------------------------

    def _mlp_config(self) -> MLPConfig:
        return MLPConfig(
            input_dim=1,  # rebound to the fitted representation width
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
        )

    def _training_config(self) -> TrainingConfig:
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            validation_fraction=self.validation_fraction,
            early_stopping_patience=self.early_stopping_patience,
            min_delta=self.min_delta,
            seed=self.seed,
            pos_weight=self.pos_weight,
            shuffle=self.shuffle,
        )

    # -- sklearn-style surface ---------------------------------------------

    def fit(self, X: object, y: object) -> "TunableMLPClassifier":
        """Train the MLP on a raw feature frame (inputs + labels).

        ``X`` must be a pandas DataFrame carrying the schema feature columns;
        the target is bound into the frame and :func:`train_mlp_classifier`
        handles the rest (representation fit on these rows only, device
        resolution with a logged CPU fallback, early stopping).
        """
        if not isinstance(X, pd.DataFrame):
            raise MLPSpaceError(
                "fit expects a pandas DataFrame of raw schema feature columns, "
                f"got {type(X).__name__}."
            )
        frame = X.copy()
        frame[TARGET_COLUMN] = np.asarray(y)
        self.classifier_ = train_mlp_classifier(
            frame,
            mlp_config=self._mlp_config(),
            training_config=self._training_config(),
            selection_path=self.selection_path,
            prefer_gpu=self.prefer_gpu,
        )
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        self._require_fitted()
        return self.classifier_.predict_proba(X)  # type: ignore[union-attr]

    def predict(self, X: object) -> np.ndarray:
        self._require_fitted()
        return self.classifier_.predict(X)  # type: ignore[union-attr]

    # -- observability -----------------------------------------------------

    @property
    def device(self) -> str:
        """The compute device the fitted run used (``cuda:0`` or ``cpu``)."""
        self._require_fitted()
        return self.classifier_.device  # type: ignore[union-attr]

    @property
    def device_resolution(self) -> object:
        """The full device :class:`~heart.models.train_torch.DeviceResolution`."""
        self._require_fitted()
        return self.classifier_.device_resolution  # type: ignore[union-attr]

    def _require_fitted(self) -> None:
        if self.classifier_ is None:
            raise MLPSpaceError(
                f"{type(self).__name__} has not been fit; call fit(X, y) first."
            )

    def parameter_dict(self) -> dict[str, object]:
        """Every knobs this estimator was configured with (for MLflow)."""
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "validation_fraction": self.validation_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
            "seed": self.seed,
            "pos_weight": self.pos_weight,
            "shuffle": self.shuffle,
        }


# ---------------------------------------------------------------------------
# The production scoring leaf
# ---------------------------------------------------------------------------


def mlp_trainer(
    *,
    prefer_gpu: bool = True,
    selection_path: str | Path = DEFAULT_SELECTION_LEDGER_PATH,
) -> Callable[[Mapping[str, object], object], tuple[dict[str, object], np.ndarray, np.ndarray]]:
    """Build the production ``trainer(params, fold)`` used by the neural sweep.

    The trainer merges the full parameter dict (fixed architecture + tuned
    training knobs, as the objective hands it over) into a
    :class:`TunableMLPClassifier`, fits it on the fold's training rows, scores
    the fold's held-out split through the shared
    :func:`heart.eval.contract.evaluate`, and returns the canonical metric
    dict plus the fold's predictions and positive-class probabilities —
    exactly what :class:`MLPTuningObjective` needs to assemble the
    out-of-fold metric dict of a completed trial.

    Off the GPU, :func:`train_mlp_classifier` falls back to CPU with an
    explicit ``WARNING`` and records the device, so a sweep never silently
    changes compute.
    """
    if not isinstance(prefer_gpu, bool):
        raise MLPSpaceError(f"prefer_gpu must be a bool, got {prefer_gpu!r}.")

    def _train_and_score(
        params: Mapping[str, object], fold: object
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
        if not isinstance(fold, object) or not hasattr(fold, "train_frame"):
            raise MLPSpaceError(
                f"trainer expects a TuningFold, got {type(fold).__name__}."
            )
        classifier = TunableMLPClassifier(
            **dict(params),
            prefer_gpu=prefer_gpu,
            selection_path=selection_path,
        )
        classifier.fit(
            fold.train_frame[list(FEATURE_COLUMNS)],  # type: ignore[union-attr]
            fold.train_frame[TARGET_COLUMN],  # type: ignore[union-attr]
        )
        split: EvaluationSplit = fold.split  # type: ignore[union-attr]
        metrics = evaluate(classifier, split)
        predictions = np.asarray(classifier.predict(split.X_test)).ravel()
        probabilities = np.asarray(
            classifier.predict_proba(split.X_test), dtype=float
        )[:, 1]
        return metrics, predictions, probabilities

    return _train_and_score


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def describe_mlp_sweep_plan() -> str:
    """Render the configured architecture set and its spaces as a report."""
    lines = [
        f"neural sweep: {MLP_SWEEP_SIZE} configured architecture(s)",
        "",
        "| # | model_type | model_name | hidden sizes | activation | batch_norm | dropout default |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, arch in enumerate(MLP_ARCHITECTURES, start=1):
        lines.append(
            f"| {index} | `{arch.model_type}` | {arch.model_name} | "
            f"{list(arch.hidden_sizes)} | {arch.activation} | "
            f"{'yes' if arch.batch_norm else 'no'} | {arch.dropout_default} |"
        )
    lines.append("")
    lines.append("Tuned training space (shared by every architecture):")
    lines.append("")
    lines.append("| parameter | kind | region | default |")
    lines.append("| --- | --- | --- | --- |")
    space = mlp_tuning_space()
    for param in space:
        if param.kind == CATEGORICAL_KIND:
            region = str(list(param.choices))
        else:
            region = (
                f"[{param.low}, {param.high}]"
                + (" log" if param.log else "")
            )
        lines.append(
            f"| `{param.name}` | {param.kind} | {region} | {param.default!r} |"
        )
    lines.append("")
    return "\n".join(lines)