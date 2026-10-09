"""PyTorch MLP definition for the neural slice (S05/T02).

This module owns the **architecture** of the portfolio's only neural model: a
feed-forward multi-layer perceptron for binary heart-disease classification. It
is deliberately split into two halves so the architecture can be reasoned about
and tested without a GPU (or torch at all):

* a **pure-Python spec** — :class:`MLPConfig` (the declared hyperparameters)
  and :func:`layer_plan` (the exact ordered layer list). Both import and run on
  a CPU-only host with no torch installed.
* a **materialiser** — :func:`build_mlp` turns a plan into a
  ``torch.nn.Sequential``. torch is imported lazily through :func:`import_torch`,
  so importing this module never requires torch. A missing torch raises the
  named :class:`TorchUnavailableError` with install guidance instead of an
  opaque ``ImportError``.

Why a lazy import
-----------------
The repository's default test environment has no torch (torch lives in the
ROCm Podman image, see S05/T01). Keeping the import lazy means the architecture
metadata and every negative path are testable on the host, while the real
network is exercised inside the documented container. This mirrors the
lazy-torch design of :mod:`heart.gpu.device_check`.

Observability
-------------
:func:`describe_mlp` renders the architecture (layer count, hidden widths,
activation, dropout, parameter count) for logs and reports. Build failures are
wrapped in :class:`MLPBuildError` naming the offending layer.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MLP_SCHEMA_VERSION",
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_DROPOUT",
    "DEFAULT_ACTIVATION",
    "DEFAULT_BATCH_NORM",
    "DEFAULT_OUTPUT_DIM",
    "ACTIVATIONS",
    "LAYER_LINEAR",
    "LAYER_BATCH_NORM",
    "LAYER_ACTIVATION",
    "LAYER_DROPOUT",
    "LAYER_KINDS",
    "MLPError",
    "MLPConfigError",
    "TorchUnavailableError",
    "MLPBuildError",
    "MLPConfig",
    "LayerPlan",
    "import_torch",
    "layer_plan",
    "build_mlp",
    "describe_mlp",
]

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Machine-readable schema version recorded alongside an architecture spec.
MLP_SCHEMA_VERSION: str = "1"

#: Default hidden-layer widths (a small, deliberately tuned-capacity default).
DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (64, 32)

#: Default dropout probability applied after each hidden activation.
DEFAULT_DROPOUT: float = 0.2

#: Default hidden activation function.
DEFAULT_ACTIVATION: str = "relu"

#: Default: no batch normalisation between the linear layer and the activation.
DEFAULT_BATCH_NORM: bool = False

#: Binary classification uses a single logit (paired with ``BCEWithLogitsLoss``).
DEFAULT_OUTPUT_DIM: int = 1

#: Supported activation function names.
ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "gelu", "leaky_relu")

#: Layer-plan kinds.
LAYER_LINEAR: str = "linear"
LAYER_BATCH_NORM: str = "batch_norm"
LAYER_ACTIVATION: str = "activation"
LAYER_DROPOUT: str = "dropout"

#: Every layer-plan kind :class:`LayerPlan` accepts.
LAYER_KINDS: tuple[str, ...] = (
    LAYER_LINEAR,
    LAYER_BATCH_NORM,
    LAYER_ACTIVATION,
    LAYER_DROPOUT,
)


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class MLPError(Exception):
    """Base class for every MLP-definition failure."""


class MLPConfigError(MLPError):
    """An :class:`MLPConfig` value is invalid."""


class TorchUnavailableError(MLPError):
    """torch is not importable in this environment."""


class MLPBuildError(MLPError):
    """A layer could not be materialised from the plan."""


# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------


def import_torch(
    importer: Callable[[str], object] = importlib.import_module,
) -> object:
    """Import and return the torch module, or raise :class:`TorchUnavailableError`.

    ``importer`` is an injection seam: tests pass one that raises
    :class:`ImportError` to exercise the missing-torch path without uninstalling
    anything.
    """
    try:
        return importer("torch")
    except ImportError as exc:  # pragma: no cover - exercised via injection
        raise TorchUnavailableError(
            "The MLP requires PyTorch, which is not importable in this "
            "environment. Neural runs are executed inside the ROCm container: "
            "run `containers/run-rocm.sh shell`, or install a CPU torch for a "
            f"local fallback. Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Architecture spec
# ---------------------------------------------------------------------------


def _as_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MLPConfigError(f"{field_name} must be an integer, got {value!r}.")
    if value < 1:
        raise MLPConfigError(f"{field_name} must be >= 1, got {value!r}.")
    return value


@dataclass(frozen=True)
class MLPConfig:
    """The declared hyperparameters of one MLP architecture.

    Parameters
    ----------
    input_dim:
        Width of the design matrix the first linear layer consumes.
    hidden_sizes:
        Width of each hidden layer, in order. At least one entry; the number of
        entries is the network's hidden depth.
    dropout:
        Dropout probability applied after each hidden activation (``0.0``
        disables dropout entirely).
    activation:
        Hidden activation name, one of :data:`ACTIVATIONS`.
    batch_norm:
        When ``True`` a ``BatchNorm1d`` follows each hidden linear layer.
    output_dim:
        Output width; ``1`` for binary classification (a single logit).
    """

    input_dim: int
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    dropout: float = DEFAULT_DROPOUT
    activation: str = DEFAULT_ACTIVATION
    batch_norm: bool = DEFAULT_BATCH_NORM
    output_dim: int = DEFAULT_OUTPUT_DIM
    schema_version: str = MLP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _as_positive_int(self.input_dim, field_name="input_dim")
        _as_positive_int(self.output_dim, field_name="output_dim")
        if isinstance(self.hidden_sizes, (str, bytes)) or not isinstance(
            self.hidden_sizes, Sequence
        ):
            raise MLPConfigError(
                "hidden_sizes must be a sequence of positive integers, got "
                f"{self.hidden_sizes!r}."
            )
        sizes = tuple(self.hidden_sizes)
        if not sizes:
            raise MLPConfigError("hidden_sizes must contain at least one layer.")
        for size in sizes:
            _as_positive_int(size, field_name="hidden_sizes entry")
        object.__setattr__(self, "hidden_sizes", sizes)

        if isinstance(self.dropout, bool) or not isinstance(
            self.dropout, (int, float)
        ):
            raise MLPConfigError(f"dropout must be a number, got {self.dropout!r}.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise MLPConfigError(
                f"dropout must be in [0, 1), got {self.dropout!r}."
            )
        if self.activation not in ACTIVATIONS:
            raise MLPConfigError(
                f"activation must be one of {list(ACTIVATIONS)}, got "
                f"{self.activation!r}."
            )
        if not isinstance(self.batch_norm, bool):
            raise MLPConfigError(
                f"batch_norm must be a bool, got {self.batch_norm!r}."
            )

    def with_input_dim(self, input_dim: int) -> "MLPConfig":
        """Return a copy of this architecture with ``input_dim`` set.

        The input width is a property of the fitted feature representation, not
        a tunable hyperparameter, so callers build a template and let training
        bind the real width.
        """
        return MLPConfig(
            input_dim=int(input_dim),
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            output_dim=self.output_dim,
            schema_version=self.schema_version,
        )

    @property
    def depth(self) -> int:
        """Number of weight layers (hidden layers plus the output layer)."""
        return len(self.hidden_sizes) + 1

    @property
    def width(self) -> int:
        """Width of the first hidden layer."""
        return int(self.hidden_sizes[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_dim": int(self.input_dim),
            "hidden_sizes": [int(size) for size in self.hidden_sizes],
            "dropout": float(self.dropout),
            "activation": self.activation,
            "batch_norm": bool(self.batch_norm),
            "output_dim": int(self.output_dim),
            "depth": self.depth,
        }

    def to_params(self, *, prefix: str = "mlp") -> dict[str, object]:
        """Flatten the architecture into MLflow-friendly scalar parameters."""
        params: dict[str, object] = {
            f"{prefix}_input_dim": int(self.input_dim),
            f"{prefix}_hidden_sizes": list(int(size) for size in self.hidden_sizes),
            f"{prefix}_dropout": float(self.dropout),
            f"{prefix}_activation": self.activation,
            f"{prefix}_batch_norm": bool(self.batch_norm),
            f"{prefix}_output_dim": int(self.output_dim),
        }
        for index, size in enumerate(self.hidden_sizes):
            params[f"{prefix}_hidden_{index}_units"] = int(size)
        return params


# ---------------------------------------------------------------------------
# Layer plan (pure, torch-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerPlan:
    """One planned layer, expressed without any torch dependency."""

    kind: str
    name: str
    in_features: int | None = None
    out_features: int | None = None
    activation: str | None = None
    rate: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in LAYER_KINDS:
            raise MLPConfigError(
                f"Unknown layer kind {self.kind!r}; expected one of "
                f"{list(LAYER_KINDS)}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation,
            "rate": self.rate,
        }


def layer_plan(config: MLPConfig) -> tuple[LayerPlan, ...]:
    """Return the ordered layer plan for ``config``.

    The plan is the architecture contract: ``Linear`` (optionally
    ``BatchNorm1d``), then the activation, then dropout for each hidden layer,
    followed by a single output ``Linear``. It is pure data, so the plan can be
    asserted exactly without torch.
    """
    if not isinstance(config, MLPConfig):
        raise MLPConfigError(
            f"config must be an MLPConfig, got {type(config).__name__}."
        )
    plan: list[LayerPlan] = []
    previous = int(config.input_dim)
    for index, width in enumerate(config.hidden_sizes):
        width = int(width)
        plan.append(
            LayerPlan(
                kind=LAYER_LINEAR,
                name=f"hidden_{index}_linear",
                in_features=previous,
                out_features=width,
            )
        )
        if config.batch_norm:
            plan.append(
                LayerPlan(
                    kind=LAYER_BATCH_NORM,
                    name=f"hidden_{index}_batch_norm",
                    out_features=width,
                )
            )
        plan.append(
            LayerPlan(
                kind=LAYER_ACTIVATION,
                name=f"hidden_{index}_activation",
                activation=config.activation,
            )
        )
        if float(config.dropout) > 0.0:
            plan.append(
                LayerPlan(
                    kind=LAYER_DROPOUT,
                    name=f"hidden_{index}_dropout",
                    rate=float(config.dropout),
                )
            )
        previous = width
    plan.append(
        LayerPlan(
            kind=LAYER_LINEAR,
            name="output_linear",
            in_features=previous,
            out_features=int(config.output_dim),
        )
    )
    return tuple(plan)


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


def _activation_layer(nn_module: object, name: str) -> object:
    constructors: dict[str, str] = {
        "relu": "ReLU",
        "tanh": "Tanh",
        "gelu": "GELU",
        "leaky_relu": "LeakyReLU",
    }
    constructor_name = constructors[name]
    constructor = getattr(nn_module, constructor_name, None)
    if constructor is None:  # pragma: no cover - guards against a torch rename
        raise MLPBuildError(
            f"torch.nn has no {constructor_name!r} activation for {name!r}."
        )
    return constructor()


def build_mlp(config: MLPConfig, *, torch_module: object | None = None) -> object:
    """Materialise ``config`` into a ``torch.nn.Sequential``.

    ``torch_module`` is an injection seam for tests; when ``None`` torch is
    imported lazily. A missing torch raises :class:`TorchUnavailableError` and a
    layer that cannot be constructed raises :class:`MLPBuildError` naming the
    plan entry.
    """
    if not isinstance(config, MLPConfig):
        raise MLPConfigError(
            f"config must be an MLPConfig, got {type(config).__name__}."
        )
    torch = torch_module if torch_module is not None else import_torch()
    nn_module = getattr(torch, "nn", None)
    if nn_module is None:
        raise MLPBuildError(
            "The injected torch module exposes no `nn` attribute; cannot build "
            "the network."
        )

    layers: list[object] = []
    try:
        for entry in layer_plan(config):
            if entry.kind == LAYER_LINEAR:
                layers.append(
                    nn_module.Linear(int(entry.in_features), int(entry.out_features))
                )
            elif entry.kind == LAYER_BATCH_NORM:
                layers.append(nn_module.BatchNorm1d(int(entry.out_features)))
            elif entry.kind == LAYER_ACTIVATION:
                layers.append(_activation_layer(nn_module, str(entry.activation)))
            elif entry.kind == LAYER_DROPOUT:
                layers.append(nn_module.Dropout(float(entry.rate)))
    except MLPError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrapped into the named failure path
        raise MLPBuildError(
            f"Could not materialise the MLP layer plan: {type(exc).__name__}: {exc}"
        ) from exc

    sequential = getattr(nn_module, "Sequential", None)
    if sequential is None:
        raise MLPBuildError("The injected torch module exposes no nn.Sequential.")
    model = sequential(*layers)
    logger.debug(
        "Built MLP: input=%d hidden=%s output=%d layers=%d dropout=%.2f "
        "activation=%s batch_norm=%s",
        config.input_dim,
        list(config.hidden_sizes),
        config.output_dim,
        len(layers),
        float(config.dropout),
        config.activation,
        config.batch_norm,
    )
    return model


def describe_mlp(config: MLPConfig) -> str:
    """Render a one-block, human-readable description of the architecture."""
    if not isinstance(config, MLPConfig):
        raise MLPConfigError(
            f"config must be an MLPConfig, got {type(config).__name__}."
        )
    widths = [int(config.input_dim), *[int(s) for s in config.hidden_sizes], int(config.output_dim)]
    return "\n".join(
        [
            "MLP architecture:",
            f"  layers: {' -> '.join(str(width) for width in widths)}",
            f"  hidden depth: {len(config.hidden_sizes)}",
            f"  activation: {config.activation}",
            f"  dropout: {float(config.dropout):.2f}"
            + (" (disabled)" if float(config.dropout) == 0.0 else ""),
            f"  batch norm: {'yes' if config.batch_norm else 'no'}",
            f"  output: {config.output_dim} logit"
            + ("" if config.output_dim == 1 else "s"),
        ]
    )
