"""Shared interpretability infrastructure (S04/T04).

Why this module exists
---------------------
:mod:`heart.interpret.shap_export` and :mod:`heart.interpret.coefficients` both
need the *same* thing before they can explain anything: a **fitted pipeline for
the leading model on the selected feature representation**. If each module
built that pipeline independently the two explanations could silently drift
onto different feature spaces, which would make the whole report incoherent.

This module is the single place that answers "what did the leading model see?":

1. :func:`resolve_leading_model` reads the S03 leaderboard and returns the
   top-ranked ``run_kind=final`` model together with its tuned
   hyperparameters. The model is never hard-coded — whatever tops the
   leaderboard is the one explained.
2. :func:`build_selected_pipeline` composes the S04/T02 ablation chain
   (``zero_policy -> engineering -> ColumnTransformer``) with that model's
   estimator, so the explained representation is exactly the
   :class:`~heart.features.selection.FeatureSelection` ledger's ``selected``
   mode.
3. :func:`transformed_feature_names` / :func:`model_input_feature_names` expose
   the two feature spaces (raw inputs vs. the scaled/one-hot design matrix the
   classifier actually sees) so artifact dimensions can be asserted instead of
   assumed.

Reproducibility
---------------
A resolved :class:`LeadingModel` round-trips through a committed JSON ledger
(``reports/leading_model.json``) so
:mod:`heart.interpret.report` can be regenerated from the repository without a
populated MLflow store — the same ledger pattern used by S04/T02 and S04/T03.

Observability
-------------
:func:`resolve_leading_model` logs the winning model, its run id, and its
primary metric at ``INFO``; :func:`build_interpretability_context` logs the fit
row counts and the transformed feature width. Every failure is a named
exception carrying the missing piece (experiment, model type, ledger, split).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from heart.config import RANDOM_SEED, REPORTS_DIR
from heart.data.schema import TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.features.ablation import FeatureMode, build_ablation_pipeline
from heart.features.selection import (
    DEFAULT_LEDGER_PATH as DEFAULT_SELECTION_LEDGER_PATH,
)
from heart.features.selection import (
    FeatureSelection,
    FeatureSelectionError,
    read_selection,
)
from heart.models.registry import (
    ModelSpec,
    RegistryError,
    build_estimator,
    resolve_spec,
)
from heart.models.spaces import FLOAT_KIND, INT_KIND

logger = logging.getLogger(__name__)

__all__ = [
    "INTERPRETABILITY_SLICE_NAME",
    "LEADING_MODEL_LEDGER_FILENAME",
    "DEFAULT_LEADING_MODEL_PATH",
    "InterpretError",
    "InterpretConfigError",
    "InterpretDataError",
    "LeadingModelError",
    "LeadingModelLedgerError",
    "SelectedPipelineError",
    "LeadingModel",
    "coerce_tuned_params",
    "leading_model_from_row",
    "resolve_leading_model",
    "write_leading_model_ledger",
    "read_leading_model_ledger",
    "build_selected_pipeline",
    "transformed_feature_names",
    "model_input_feature_names",
    "transformer_input_feature_names",
    "display_feature_name",
    "matches_engineered",
    "engineered_feature_names",
    "InterpretabilityContext",
    "build_interpretability_context",
    "atomic_write_text",
    "atomic_write_bytes",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Slice that produces the interpretability artifacts.
INTERPRETABILITY_SLICE_NAME: str = "S04"

#: Committed ledger naming the leading model and its tuned parameters.
LEADING_MODEL_LEDGER_FILENAME: str = "leading_model.json"

#: Default location of the leading-model ledger.
DEFAULT_LEADING_MODEL_PATH: Path = Path(REPORTS_DIR) / LEADING_MODEL_LEDGER_FILENAME

#: Version stamped on the ledger so a future schema change is detectable.
LEADING_MODEL_LEDGER_VERSION: int = 1


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class InterpretError(Exception):
    """Base class for every interpretability failure."""


class InterpretConfigError(InterpretError):
    """An interpretability configuration value is invalid."""


class InterpretDataError(InterpretError):
    """An input frame, split, or ledger is missing or malformed."""


class LeadingModelError(InterpretError):
    """The leading model could not be resolved or declared."""


class LeadingModelLedgerError(InterpretError):
    """The leading-model ledger could not be serialised, written, or read."""


class SelectedPipelineError(InterpretError):
    """The selected-representation pipeline could not be built or inspected."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadingModel:
    """The top-ranked battery member and the parameters it was tuned with.

    ``raw_params`` preserves the strings exactly as MLflow recorded them (so
    provenance is auditable); ``tuned_params`` are the coerced values the
    registry's estimator factory receives. ``primary_metric`` is the held-out
    ROC-AUC the leaderboard ranked the model by.
    """

    model_type: str
    model_name: str
    family: str
    run_id: str
    run_name: str
    split_version: str
    primary_metric: float
    raw_params: dict[str, str]
    tuned_params: dict[str, object]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("model_type", self.model_type),
            ("model_name", self.model_name),
            ("run_id", self.run_id),
            ("split_version", self.split_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise LeadingModelError(
                    f"LeadingModel.{field_name} must be a non-empty string, got "
                    f"{value!r}."
                )
        if not isinstance(self.primary_metric, (int, float)) or isinstance(
            self.primary_metric, bool
        ):
            raise LeadingModelError(
                f"LeadingModel.primary_metric must be a number, got "
                f"{type(self.primary_metric).__name__}."
            )
        object.__setattr__(self, "raw_params", dict(self.raw_params))
        object.__setattr__(self, "tuned_params", dict(self.tuned_params))

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_version": LEADING_MODEL_LEDGER_VERSION,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "family": self.family,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "split_version": self.split_version,
            "primary_metric": float(self.primary_metric),
            "raw_params": dict(self.raw_params),
            "tuned_params": dict(self.tuned_params),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LeadingModel":
        return cls(
            model_type=str(payload["model_type"]),
            model_name=str(payload["model_name"]),
            family=str(payload.get("family", "")),
            run_id=str(payload["run_id"]),
            run_name=str(payload.get("run_name", "")),
            split_version=str(payload["split_version"]),
            primary_metric=float(payload["primary_metric"]),  # type: ignore[arg-type]
            raw_params={str(k): str(v) for k, v in dict(payload.get("raw_params", {})).items()},  # type: ignore[arg-type]
            tuned_params=dict(payload.get("tuned_params", {})),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Parameter coercion
# ---------------------------------------------------------------------------


def _render_choice(value: object) -> str:
    """Render a categorical choice the way MLflow would have stored it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _coerce_value(spec, value: object) -> object:
    """Coerce one MLflow string parameter into the type its spec declares."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if spec.kind == INT_KIND:
        try:
            return int(float(text))
        except ValueError as exc:
            raise LeadingModelError(
                f"Could not read integer parameter {spec.name!r} from "
                f"{value!r}."
            ) from exc
    if spec.kind == FLOAT_KIND:
        try:
            return float(text)
        except ValueError as exc:
            raise LeadingModelError(
                f"Could not read float parameter {spec.name!r} from {value!r}."
            ) from exc
    # Categorical: match a declared choice first (this is what disambiguates
    # the string "None" from a real "None" level and the sentinel "null").
    for choice in spec.choices:
        if _render_choice(choice) == text:
            return choice
    if text in ("null", "None"):
        return None
    return text


def coerce_tuned_params(
    spec: ModelSpec, raw_params: Mapping[str, object]
) -> dict[str, object]:
    """Coerce an MLflow parameter mapping into the spec's tuned parameter dict.

    Only the spec's declared search-space keys are returned; fixed parameters
    are pinned by the spec and re-applied by
    :func:`heart.models.registry.build_estimator`. A missing tuned key raises
    :class:`LeadingModelError` rather than silently falling back to a default,
    because a silently-defaulted hyperparameter would mean the explanation is
    of a different model than the leaderboard row.
    """
    if not isinstance(spec, ModelSpec):
        raise LeadingModelError(
            f"spec must be a ModelSpec, got {type(spec).__name__}."
        )
    if not isinstance(raw_params, Mapping):
        raise LeadingModelError(
            f"raw_params must be a mapping, got {type(raw_params).__name__}."
        )
    tuned: dict[str, object] = {}
    missing: list[str] = []
    for name in spec.space.keys:
        if name not in raw_params:
            missing.append(name)
            continue
        tuned[name] = _coerce_value(spec.space.get(name), raw_params[name])
    if missing:
        raise LeadingModelError(
            f"Model {spec.model_type!r} is missing tuned parameter(s) "
            f"{sorted(missing)} in the MLflow run; cannot reconstruct the "
            "leading model."
        )
    return tuned


def leading_model_from_row(row: object) -> LeadingModel:
    """Build a :class:`LeadingModel` from a leaderboard row (duck-typed).

    The row must expose ``model_type``, ``model_name``, ``family``, ``run_id``,
    ``run_name``, ``split_version``, ``primary_metric`` and ``params`` — the
    shape of :class:`heart.reporting.leaderboard.LeaderboardRow`. Duck typing
    keeps this module importable without MLflow.
    """
    for attribute in (
        "model_type",
        "model_name",
        "run_id",
        "split_version",
        "primary_metric",
        "params",
    ):
        if not hasattr(row, attribute):
            raise LeadingModelError(
                f"Leaderboard row is missing attribute {attribute!r}; got "
                f"{type(row).__name__}."
            )
    try:
        spec = resolve_spec(str(getattr(row, "model_type")))
    except RegistryError as exc:
        raise LeadingModelError(
            f"Leaderboard row names unknown model type "
            f"{getattr(row, 'model_type')!r}: {exc}"
        ) from exc
    raw_params = {str(k): str(v) for k, v in dict(getattr(row, "params")).items()}
    tuned = coerce_tuned_params(spec, raw_params)
    model = LeadingModel(
        model_type=spec.model_type,
        model_name=str(getattr(row, "model_name", spec.model_name)),
        family=str(getattr(row, "family", spec.family)),
        run_id=str(getattr(row, "run_id")),
        run_name=str(getattr(row, "run_name", "")),
        split_version=str(getattr(row, "split_version", SPLIT_VERSION)),
        primary_metric=float(getattr(row, "primary_metric")),
        raw_params=raw_params,
        tuned_params=tuned,
    )
    logger.info(
        "Leading model resolved: %s (run %s, %s=%.4f, tuned=%s)",
        model.model_name,
        model.run_id,
        "roc_auc",
        model.primary_metric,
        model.tuned_params,
    )
    return model


def resolve_leading_model(
    *,
    experiment_name: str | None = None,
    split_version: str | None = None,
    tracking_dir: str | Path | None = None,
) -> LeadingModel:
    """Resolve the leading model from the S03 ``run_kind=final`` leaderboard.

    MLflow is imported lazily so this module (and the artifact exporters that
    depend on it) stay importable without the ``ml`` extra. A missing
    experiment, an empty leaderboard, or an unreadable store raises
    :class:`LeadingModelError`.
    """
    try:  # pragma: no cover - import path exercised by integration runs
        from heart.reporting.leaderboard import (
            LeaderboardConfig,
            select_leaderboard_rows,
        )
        from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT
    except ImportError as exc:  # pragma: no cover - exercised when ml absent
        raise LeadingModelError(
            "Resolving the leading model requires MLflow, which is not "
            'installed. Install the tracking extra with: pip install -e ".[ml]"'
        ) from exc

    resolved_experiment = experiment_name or DEFAULT_EXPERIMENT
    config = LeaderboardConfig(
        experiment_name=resolved_experiment,
        split_version=split_version,
        strict=True,
    )
    try:
        rows, considered = select_leaderboard_rows(
            config=config, tracking_dir=tracking_dir
        )
    except LeadingModelError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raise as a named error
        raise LeadingModelError(
            f"Could not read the leaderboard from experiment "
            f"{resolved_experiment!r}: {exc}"
        ) from exc
    if not rows:
        raise LeadingModelError(
            f"No finished {config.run_kind!r} runs matched experiment "
            f"{resolved_experiment!r}"
            + (
                f" and split {split_version!r}"
                if split_version is not None
                else ""
            )
            + f" ({considered} candidate run(s) considered). Run the battery "
            "first, or pass the leading-model ledger explicitly."
        )
    leading = leading_model_from_row(rows[0])
    if split_version is not None and leading.split_version != split_version:
        raise LeadingModelError(
            f"Leading model run {leading.run_id} reports split "
            f"{leading.split_version!r}, expected {split_version!r}."
        )
    return leading


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Write ``content`` to ``path`` atomically (temp file + ``os.replace``)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write ``payload`` to ``path`` atomically (temp file + ``os.replace``)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    tmp.write_bytes(payload)
    os.replace(tmp, destination)
    return destination


def write_leading_model_ledger(
    leading_model: LeadingModel, path: str | Path
) -> Path:
    """Atomically write the leading-model ledger to ``path``."""
    if not isinstance(leading_model, LeadingModel):
        raise LeadingModelLedgerError(
            f"leading_model must be a LeadingModel, got "
            f"{type(leading_model).__name__}."
        )
    payload = leading_model.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["slice"] = INTERPRETABILITY_SLICE_NAME
    try:
        destination = atomic_write_text(
            path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    except OSError as exc:
        raise LeadingModelLedgerError(
            f"Could not write the leading-model ledger to {path}: {exc}"
        ) from exc
    logger.info(
        "Wrote leading-model ledger to %s (%s, run %s)",
        destination,
        leading_model.model_type,
        leading_model.run_id,
    )
    return destination


def read_leading_model_ledger(path: str | Path) -> LeadingModel:
    """Read a leading-model ledger written by :func:`write_leading_model_ledger`."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LeadingModelLedgerError(
            f"Could not read the leading-model ledger at {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise LeadingModelLedgerError(
            f"Leading-model ledger at {source} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise LeadingModelLedgerError(
            f"Leading-model ledger at {source} must contain a JSON object, got "
            f"{type(payload).__name__}."
        )
    try:
        return LeadingModel.from_dict(payload)
    except (KeyError, TypeError, ValueError, LeadingModelError) as exc:
        raise LeadingModelLedgerError(
            f"Leading-model ledger at {source} is malformed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Selected-representation pipeline
# ---------------------------------------------------------------------------


def build_selected_pipeline(
    mode: FeatureMode, leading_model: LeadingModel
) -> Pipeline:
    """Build the unfitted pipeline: the ablation chain + the leading estimator.

    The feature chain is built by :func:`heart.features.ablation.build_ablation_pipeline`
    (zero-as-missing imputation, then the mode's engineered transforms, then
    scaling/one-hot encoding) and only the classifier step is replaced with the
    leading model's estimator. Reusing the ablation builder is deliberate: it
    guarantees the explained feature space is byte-for-byte the one the
    selected representation was scored on, so there is no second, divergent
    pipeline definition to keep in sync.
    """
    if not isinstance(mode, FeatureMode):
        raise SelectedPipelineError(
            f"mode must be a FeatureMode, got {type(mode).__name__}."
        )
    if not isinstance(leading_model, LeadingModel):
        raise SelectedPipelineError(
            f"leading_model must be a LeadingModel, got "
            f"{type(leading_model).__name__}."
        )
    try:
        spec = resolve_spec(leading_model.model_type)
        estimator = build_estimator(spec, leading_model.tuned_params)
    except RegistryError as exc:
        raise SelectedPipelineError(
            f"Could not build an estimator for leading model type "
            f"{leading_model.model_type!r}: {exc}"
        ) from exc
    base = build_ablation_pipeline(mode)
    pipeline = Pipeline(steps=[*base.steps[:-1], ("clf", estimator)])
    logger.debug(
        "Built selected pipeline for %s on mode %r (%d engineered column(s))",
        leading_model.model_type,
        mode.name,
        mode.n_engineered_columns,
    )
    return pipeline


def transformed_feature_names(pipeline: Pipeline) -> tuple[str, ...]:
    """Return the fitted design-matrix column names the classifier saw."""
    if not isinstance(pipeline, Pipeline):
        raise SelectedPipelineError(
            f"pipeline must be a sklearn Pipeline, got {type(pipeline).__name__}."
        )
    features = pipeline.named_steps.get("features")
    if features is None or not hasattr(features, "get_feature_names_out"):
        raise SelectedPipelineError(
            "Pipeline is not fitted (or has no 'features' step); fit it before "
            "reading the transformed feature names."
        )
    return tuple(str(name) for name in features.get_feature_names_out())


def model_input_feature_names(pipeline: Pipeline) -> tuple[str, ...]:
    """Return the raw schema columns the pipeline consumes as ``X``.

    This is the frame shape an explanation must be handed — the raw eleven
    columns, *not* the raw-plus-engineered columns the ``ColumnTransformer``
    sees (engineering happens inside the pipeline). Those transformer inputs
    are recoverable as ``pipeline.named_steps['features'].feature_names_in_``
    and are useful for provenance, but they are not what a caller passes to
    :func:`heart.interpret.shap_export.compute_shap_values`.
    """
    if not isinstance(pipeline, Pipeline):
        raise SelectedPipelineError(
            f"pipeline must be a sklearn Pipeline, got {type(pipeline).__name__}."
        )
    names = getattr(pipeline, "feature_names_in_", None)
    if names is None:
        zero_step = pipeline.named_steps.get("zero_policy")
        names = getattr(zero_step, "feature_names_in_", None)
    if names is None:
        raise SelectedPipelineError(
            "Pipeline is not fitted (no feature_names_in_ is available); fit it "
            "before reading the raw input columns."
        )
    return tuple(str(name) for name in names)


def transformer_input_feature_names(pipeline: Pipeline) -> tuple[str, ...]:
    """Return the raw-plus-engineered columns entering the ``ColumnTransformer``."""
    if not isinstance(pipeline, Pipeline):
        raise SelectedPipelineError(
            f"pipeline must be a sklearn Pipeline, got {type(pipeline).__name__}."
        )
    features = pipeline.named_steps.get("features")
    names = getattr(features, "feature_names_in_", None)
    if names is None:
        raise SelectedPipelineError(
            "Pipeline is not fitted (the 'features' step has no "
            "feature_names_in_); fit it before reading the transformer inputs."
        )
    return tuple(str(name) for name in names)


def display_feature_name(raw_name: str) -> str:
    """Strip the ``<transformer>__`` prefix sklearn adds to a transformed name."""
    text = str(raw_name)
    if "__" in text:
        return text.split("__", 1)[1]
    return text


def matches_engineered(
    raw_name: str, engineered_columns: Sequence[str]
) -> bool:
    """Whether a transformed feature column comes from an engineered column.

    A numeric engineered column keeps its exact name in the design matrix; a
    categorical engineered column is one-hot encoded into ``<column>_<level>``
    variants. Matching therefore accepts an exact name or a ``<column>_``
    prefix, which is why a plain substring test is not used.
    """
    display = display_feature_name(raw_name)
    return any(
        display == str(column) or display.startswith(f"{column}_")
        for column in engineered_columns
    )


def engineered_feature_names(selection: FeatureSelection) -> tuple[str, ...]:
    """The raw engineered column names of the selected representation."""
    if not isinstance(selection, FeatureSelection):
        raise SelectedPipelineError(
            f"selection must be a FeatureSelection, got "
            f"{type(selection).__name__}."
        )
    return tuple(selection.selected_mode.engineered_columns)


# ---------------------------------------------------------------------------
# Interpretability context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterpretabilityContext:
    """Everything the exporters need: a fitted pipeline and its feature spaces.

    The context records the *resolved* selected representation rather than a
    whole :class:`~heart.features.selection.FeatureSelection`, so it can be
    constructed directly in tests (or from a different selection source)
    without re-deriving the S04/T03 decision policy.
    """

    selected_mode: FeatureMode
    engineered_columns: tuple[str, ...]
    n_dropped: int
    leading_model: LeadingModel
    split_version: str
    pipeline: Pipeline
    test_frame: pd.DataFrame
    train_rows: int
    test_rows: int
    feature_names: tuple[str, ...]
    input_feature_names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_engineered_features(self) -> int:
        return len(self.engineered_columns)

    def engineered_mask(self) -> tuple[bool, ...]:
        """Per-transformed-feature flag: is this column an engineered feature?"""
        return tuple(
            matches_engineered(name, self.engineered_columns)
            for name in self.feature_names
        )


def build_interpretability_context(
    *,
    selection: FeatureSelection | None = None,
    leading_model: LeadingModel | None = None,
    selection_ledger_path: str | Path | None = DEFAULT_SELECTION_LEDGER_PATH,
    leading_model_ledger_path: str | Path | None = None,
    split_version: str | None = None,
    tracking_dir: str | Path | None = None,
    max_train_rows: int | None = None,
    random_state: int = RANDOM_SEED,
) -> InterpretabilityContext:
    """Load the selection + leading model, fit the pipeline, return the context.

    Evidence precedence mirrors the rest of the slice: an explicit object wins;
    otherwise a ledger is read (selection ledger always, leading-model ledger
    when a path is supplied); otherwise the leading model is resolved from the
    MLflow leaderboard and the selection ledger is read from ``reports/``.

    ``max_train_rows`` deterministically subsamples the training rows (with
    ``random_state``) for fast smoke runs; leave it ``None`` for the published
    artifacts, which fit on the full training split.
    """
    if selection is None:
        if selection_ledger_path is None:
            raise InterpretConfigError(
                "Provide either a FeatureSelection or a selection ledger path."
            )
        try:
            selection = read_selection(selection_ledger_path)
        except FeatureSelectionError as exc:
            raise InterpretDataError(
                f"Could not read the selection ledger at "
                f"{selection_ledger_path}: {exc}"
            ) from exc

    if leading_model is None:
        if leading_model_ledger_path is not None and Path(
            leading_model_ledger_path
        ).exists():
            leading_model = read_leading_model_ledger(leading_model_ledger_path)
        else:
            leading_model = resolve_leading_model(
                split_version=split_version, tracking_dir=tracking_dir
            )

    resolved_split = split_version or leading_model.split_version or SPLIT_VERSION
    try:
        train_frame, test_frame = load_split_frames(resolved_split)
    except Exception as exc:  # noqa: BLE001 - re-raise as a named error
        raise InterpretDataError(
            f"Could not load split {resolved_split!r}: {exc}"
        ) from exc

    if max_train_rows is not None:
        if not isinstance(max_train_rows, int) or isinstance(max_train_rows, bool):
            raise InterpretConfigError(
                f"max_train_rows must be None or an int, got "
                f"{type(max_train_rows).__name__}."
            )
        if max_train_rows < 2:
            raise InterpretConfigError(
                f"max_train_rows must be at least 2, got {max_train_rows}."
            )
        if len(train_frame) > max_train_rows:
            train_frame = train_frame.sample(
                n=max_train_rows, random_state=random_state
            ).sort_index()

    pipeline = build_selected_pipeline(selection.selected_mode, leading_model)
    try:
        pipeline.fit(
            train_frame.drop(columns=[TARGET_COLUMN]),
            train_frame[TARGET_COLUMN],
        )
    except Exception as exc:  # noqa: BLE001 - re-raise as a named error
        raise SelectedPipelineError(
            f"Could not fit the selected pipeline for "
            f"{leading_model.model_type!r}: {exc}"
        ) from exc

    context = InterpretabilityContext(
        selected_mode=selection.selected_mode,
        engineered_columns=tuple(selection.selected_mode.engineered_columns),
        n_dropped=int(selection.n_dropped),
        leading_model=leading_model,
        split_version=resolved_split,
        pipeline=pipeline,
        test_frame=test_frame,
        train_rows=int(len(train_frame)),
        test_rows=int(len(test_frame)),
        feature_names=transformed_feature_names(pipeline),
        input_feature_names=model_input_feature_names(pipeline),
    )
    logger.info(
        "Interpretability context ready: %s on selected mode %r "
        "(train %d / test %d rows, %d transformed features, %d engineered)",
        leading_model.model_type,
        selection.selected_mode.name,
        context.train_rows,
        context.test_rows,
        context.n_features,
        context.n_engineered_features,
    )
    return context
