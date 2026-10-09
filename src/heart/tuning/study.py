"""Optuna study construction and tuning configuration (S03/T02).

This module owns **how a tuning study is configured and built** — the sampler,
the pruner, the trial budget, and the study name. It is deliberately separate
from :mod:`heart.tuning.runner`, which owns *what is optimised* (the model
spec's search space scored through the shared evaluation contract) and how
trials are recorded in MLflow.

The split matters: the configuration layer is pure Optuna plumbing, so the
runner can stay generic. Adding a new sampler or pruner is a change here; the
objective never switches on model identity.

Guarantees
----------
* :class:`TuningConfig` is a validated frozen dataclass. An unknown direction,
  sampler, or pruner, a non-positive trial budget, a negative warmup, or a
  blank study name raises a named :class:`TuningConfigError` subclass rather
  than being passed to Optuna and failing later.
* :func:`create_study` builds an in-memory study by default (``storage=None``),
  which is exactly what tests and smoke runs want; a SQLite storage URI can be
  supplied for a resumable study.
* :func:`suggest_params` is the only place a search space talks to a trial, so
  the space's declared keys are sampled in one deterministic place.

Observability
-------------
:func:`create_study` logs the study name, direction, sampler, and pruner at
``INFO``. :func:`describe_tuning_config` renders the resolved configuration for
diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:  # pragma: no cover - exercised by the environment, not by logic
    import optuna
    from optuna.pruners import BasePruner, MedianPruner, NopPruner
    from optuna.samplers import BaseSampler, RandomSampler, TPESampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "heart.tuning requires Optuna, which is not installed. Install the "
        'tuning extra with: pip install -e ".[ml]"'
    ) from exc

from heart.config import RANDOM_SEED
from heart.models.registry import ModelSpec, resolve_spec
from heart.models.spaces import SearchSpace

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DIRECTION",
    "DEFAULT_N_TRIALS",
    "DEFAULT_SAMPLER",
    "DEFAULT_PRUNER",
    "DEFAULT_N_STARTUP_TRIALS",
    "DEFAULT_N_WARMUP_STEPS",
    "DEFAULT_INTERVAL_STEPS",
    "DIRECTIONS",
    "SAMPLER_NAMES",
    "PRUNER_NAMES",
    "TuningError",
    "TuningConfigError",
    "UnknownSamplerError",
    "UnknownPrunerError",
    "StudyCreationError",
    "TuningConfig",
    "build_sampler",
    "build_pruner",
    "default_study_name",
    "create_study",
    "suggest_params",
    "describe_tuning_config",
]


# ---------------------------------------------------------------------------
# Declared defaults
# ---------------------------------------------------------------------------

#: Optimisation direction. ROC-AUC is maximised, so ``maximize`` is the default.
DEFAULT_DIRECTION: str = "maximize"

#: Default number of trials per study. Large enough for TPE to matter.
DEFAULT_N_TRIALS: int = 50

#: Candidate samplers, by the short name stored in a :class:`TuningConfig`.
SAMPLER_TPE: str = "tpe"
SAMPLER_RANDOM: str = "random"

#: Default sampler: Tree-structured Parzen Estimator.
DEFAULT_SAMPLER: str = SAMPLER_TPE

#: Candidate pruners, by short name.
PRUNER_MEDIAN: str = "median"
PRUNER_NONE: str = "none"

#: Default pruner: median stopping rule over the per-fold intermediate values.
DEFAULT_PRUNER: str = PRUNER_MEDIAN

#: TPE trials before it starts modelling (a pure random-initialisation phase).
DEFAULT_N_STARTUP_TRIALS: int = 10

#: Trials the median pruner observes before it may prune anything.
DEFAULT_N_WARMUP_STEPS: int = 5

#: Steps between pruning evaluations (our step is one CV fold).
DEFAULT_INTERVAL_STEPS: int = 1

#: Every accepted direction.
DIRECTIONS: tuple[str, ...] = ("maximize", "minimize")

#: Every accepted sampler name.
SAMPLER_NAMES: tuple[str, ...] = (SAMPLER_TPE, SAMPLER_RANDOM)

#: Every accepted pruner name.
PRUNER_NAMES: tuple[str, ...] = (PRUNER_MEDIAN, PRUNER_NONE)


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class TuningError(Exception):
    """Base class for every tuning configuration or study failure."""


class TuningConfigError(TuningError):
    """The requested tuning configuration is invalid."""


class UnknownSamplerError(TuningConfigError):
    """The requested sampler name is not declared."""


class UnknownPrunerError(TuningConfigError):
    """The requested pruner name is not declared."""


class StudyCreationError(TuningError):
    """Optuna refused to create the study."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningConfig:
    """The resolved, validated tuning configuration for one study.

    Parameters
    ----------
    n_trials:
        Trial budget (``>= 1``).
    timeout:
        Optional wall-clock budget in seconds; ``None`` means no limit.
    direction:
        ``"maximize"`` (default, for ROC-AUC) or ``"minimize"``.
    sampler:
        ``"tpe"`` (default) or ``"random"``.
    pruner:
        ``"median"`` (default) or ``"none"``.
    sampler_seed:
        Seed for the sampler so a study is reproducible.
    n_startup_trials:
        TPE random-initialisation trials (also the median pruner's startup).
    n_warmup_steps:
        Median-pruner warmup steps (CV folds) before pruning may trigger.
    interval_steps:
        Steps between median-pruner evaluations.
    study_name:
        Explicit study name; ``None`` derives ``tune-<model_type>``.
    storage:
        Optional Optuna storage URI; ``None`` keeps the study in memory.
    load_if_exists:
        Resume an existing stored study instead of failing on a name clash.
    """

    n_trials: int = DEFAULT_N_TRIALS
    timeout: float | None = None
    direction: str = DEFAULT_DIRECTION
    sampler: str = DEFAULT_SAMPLER
    pruner: str = DEFAULT_PRUNER
    sampler_seed: int = RANDOM_SEED
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS
    n_warmup_steps: int = DEFAULT_N_WARMUP_STEPS
    interval_steps: int = DEFAULT_INTERVAL_STEPS
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.n_trials, bool) or not isinstance(self.n_trials, int):
            raise TuningConfigError(
                f"n_trials must be an integer, got {self.n_trials!r}."
            )
        if self.n_trials < 1:
            raise TuningConfigError(
                f"n_trials must be >= 1, got {self.n_trials!r}."
            )
        if self.timeout is not None:
            if (
                isinstance(self.timeout, bool)
                or not isinstance(self.timeout, (int, float))
                or float(self.timeout) <= 0
            ):
                raise TuningConfigError(
                    f"timeout must be a positive number of seconds or None, "
                    f"got {self.timeout!r}."
                )
        if self.direction not in DIRECTIONS:
            raise TuningConfigError(
                f"Unknown direction {self.direction!r}; expected one of "
                f"{list(DIRECTIONS)}."
            )
        if self.sampler not in SAMPLER_NAMES:
            raise UnknownSamplerError(
                f"Unknown sampler {self.sampler!r}; expected one of "
                f"{list(SAMPLER_NAMES)}."
            )
        if self.pruner not in PRUNER_NAMES:
            raise UnknownPrunerError(
                f"Unknown pruner {self.pruner!r}; expected one of "
                f"{list(PRUNER_NAMES)}."
            )
        if isinstance(self.sampler_seed, bool) or not isinstance(
            self.sampler_seed, int
        ):
            raise TuningConfigError(
                f"sampler_seed must be an integer, got {self.sampler_seed!r}."
            )
        for field_name in ("n_startup_trials", "n_warmup_steps"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TuningConfigError(
                    f"{field_name} must be a non-negative integer, got {value!r}."
                )
        if (
            isinstance(self.interval_steps, bool)
            or not isinstance(self.interval_steps, int)
            or self.interval_steps < 1
        ):
            raise TuningConfigError(
                f"interval_steps must be a positive integer, got "
                f"{self.interval_steps!r}."
            )
        if self.study_name is not None and (
            not isinstance(self.study_name, str) or not self.study_name.strip()
        ):
            raise TuningConfigError(
                f"study_name must be a non-empty string or None, got "
                f"{self.study_name!r}."
            )
        if self.storage is not None and (
            not isinstance(self.storage, str) or not self.storage.strip()
        ):
            raise TuningConfigError(
                f"storage must be a non-empty URI or None, got {self.storage!r}."
            )
        if not isinstance(self.load_if_exists, bool):
            raise TuningConfigError(
                f"load_if_exists must be a bool, got {self.load_if_exists!r}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_trials": int(self.n_trials),
            "timeout": None if self.timeout is None else float(self.timeout),
            "direction": self.direction,
            "sampler": self.sampler,
            "pruner": self.pruner,
            "sampler_seed": int(self.sampler_seed),
            "n_startup_trials": int(self.n_startup_trials),
            "n_warmup_steps": int(self.n_warmup_steps),
            "interval_steps": int(self.interval_steps),
            "study_name": self.study_name,
            "storage": self.storage,
            "load_if_exists": bool(self.load_if_exists),
        }


# ---------------------------------------------------------------------------
# Sampler / pruner construction
# ---------------------------------------------------------------------------


def build_sampler(config: TuningConfig) -> BaseSampler:
    """Build the configured Optuna sampler (seeded for reproducibility)."""
    if not isinstance(config, TuningConfig):
        raise TuningConfigError(
            f"config must be a TuningConfig, got {type(config).__name__}."
        )
    if config.sampler == SAMPLER_RANDOM:
        return RandomSampler(seed=config.sampler_seed)
    return TPESampler(
        seed=config.sampler_seed, n_startup_trials=config.n_startup_trials
    )


def build_pruner(config: TuningConfig) -> BasePruner:
    """Build the configured Optuna pruner (``none`` disables pruning)."""
    if not isinstance(config, TuningConfig):
        raise TuningConfigError(
            f"config must be a TuningConfig, got {type(config).__name__}."
        )
    if config.pruner == PRUNER_NONE:
        return NopPruner()
    return MedianPruner(
        n_startup_trials=config.n_startup_trials,
        n_warmup_steps=config.n_warmup_steps,
        interval_steps=config.interval_steps,
    )


def default_study_name(spec: ModelSpec) -> str:
    """Return the convention study name, ``tune-<model_type>``."""
    if not isinstance(spec, ModelSpec):
        raise TuningConfigError(
            f"spec must be a ModelSpec, got {type(spec).__name__}."
        )
    return f"tune-{spec.model_type}"


# ---------------------------------------------------------------------------
# Study construction
# ---------------------------------------------------------------------------


def _as_spec(model: str | ModelSpec) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    return resolve_spec(model)  # type: ignore[arg-type]


def create_study(
    model: str | ModelSpec,
    config: TuningConfig | None = None,
    *,
    sampler: BaseSampler | None = None,
    pruner: BasePruner | None = None,
) -> "optuna.Study":
    """Create (or load) the Optuna study for ``model``.

    ``model`` is a registry ``model_type`` or a :class:`ModelSpec` (the latter
    keeps the runner usable with fixture models in tests). A prebuilt
    ``sampler``/``pruner`` overrides the configuration, which lets a caller
    supply a deterministic sampler for a smoke run.

    The study is in-memory unless ``config.storage`` is set, so repeated tests
    never accumulate state on disk.
    """
    resolved_config = config or TuningConfig()
    spec = _as_spec(model)
    study_name = resolved_config.study_name or default_study_name(spec)
    resolved_sampler = sampler or build_sampler(resolved_config)
    resolved_pruner = pruner or build_pruner(resolved_config)
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=resolved_config.direction,
            sampler=resolved_sampler,
            pruner=resolved_pruner,
            storage=resolved_config.storage,
            load_if_exists=resolved_config.load_if_exists,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a named error
        raise StudyCreationError(
            f"Optuna could not create study {study_name!r} "
            f"(direction={resolved_config.direction!r}, "
            f"sampler={resolved_config.sampler!r}, "
            f"pruner={resolved_config.pruner!r}): {exc}"
        ) from exc
    logger.info(
        "Created study '%s' for model=%s: direction=%s sampler=%s pruner=%s "
        "n_trials=%d storage=%s",
        study_name,
        spec.model_type,
        resolved_config.direction,
        type(resolved_sampler).__name__,
        type(resolved_pruner).__name__,
        resolved_config.n_trials,
        resolved_config.storage or "in-memory",
    )
    return study


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def suggest_params(space: SearchSpace, trial: object) -> dict[str, object]:
    """Sample one full parameter dict from ``space`` through ``trial``.

    Thin, named-error wrapper around :meth:`SearchSpace.sample` so the runner
    has a single sampling call site.
    """
    if not isinstance(space, SearchSpace):
        raise TuningError(
            f"space must be a SearchSpace, got {type(space).__name__}."
        )
    try:
        return space.sample(trial)
    except TuningError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as a named error
        raise TuningError(
            f"Could not sample the search space {list(space.keys)} from trial "
            f"{getattr(trial, 'number', '?')}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def describe_tuning_config(config: TuningConfig | None = None) -> str:
    """Render the resolved tuning configuration as a human-readable report."""
    resolved = config or TuningConfig()
    return "\n".join(
        [
            f"direction: {resolved.direction}",
            f"trials: {resolved.n_trials}"
            + (
                f" (timeout {resolved.timeout:g}s)"
                if resolved.timeout is not None
                else ""
            ),
            f"sampler: {resolved.sampler} (seed {resolved.sampler_seed}, "
            f"startup {resolved.n_startup_trials})",
            f"pruner: {resolved.pruner} (warmup {resolved.n_warmup_steps}, "
            f"interval {resolved.interval_steps})",
            f"study name: {resolved.study_name or 'tune-<model_type>'}",
            f"storage: {resolved.storage or 'in-memory'}",
            "intermediate step: one cross-validation fold (primary metric)",
        ]
    )
