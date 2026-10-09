"""Versioned model artifact for the serving app (S06/T04).

The winner selected in :mod:`heart.eval.selection` (calibrated wrapper plus
its tuned decision threshold) must reach the app as a *self-describing*
artifact, not a bare pickle: a model loaded without its feature order and
threshold invites silent column misalignment — the classic way a correct
pipeline serves wrong numbers. This module serializes the calibrated winner
as a single versioned file carrying:

* ``format_version`` — the artifact format version; loaders refuse
  unsupported versions with :class:`ArtifactVersionError` instead of
  unpickling blindly;
* an :class:`ArtifactMetadata` block with the embedded **feature schema**
  (the exact input column order the model was fitted on), the tuned
  ``threshold``, the calibration state (method, wrapper applied, Brier
  before/after), the ship verdict and its flags, and a creation timestamp;
* the fitted model payload (the calibration wrapper, which exposes
  ``predict``/``predict_proba`` on the raw feature frame because the
  registry pipelines embed preprocessing).

Validation is enforced **at load time**: a malformed payload raises
:class:`ArtifactSchemaError`, a feature order that does not match the
declared dataset schema (:data:`heart.data.schema.FEATURE_COLUMNS`) raises
:class:`FeatureSchemaError`, and a payload without the predict/predict_proba
interface raises :class:`ArtifactPayloadError`. Nothing about the payload's
fitness is inferred — every check has a named failure.

Prediction-time schema checking is available through
:func:`check_features` and :func:`predict_positive_proba`, so the app cannot
feed a model a frame with reordered, missing, or extra columns without a
named :class:`FeatureSchemaError` first.

Serialization format
--------------------
A single pickle of a flat dict::

    {
        "format_version": 1,
        "metadata": {...},   # ArtifactMetadata.to_dict() shape
        "model": <fitted wrapper>,
    }

Pickle loads execute code; the loader trusts only artifacts written by this
project's own flow (see the security note in :func:`load_artifact`).

Failure contract
----------------
Named subclasses of :class:`ArtifactError`: :class:`ArtifactDataError`
(missing/unreadable file, malformed payload), :class:`ArtifactSchemaError`
(metadata contract violation), :class:`ArtifactVersionError` (unsupported
format version), :class:`ArtifactPayloadError` (model without the serving
interface), and :class:`FeatureSchemaError` (embedded or runtime
feature-order mismatch).
"""

from __future__ import annotations

import datetime as _dt
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from heart.config import RANDOM_SEED
from heart.data.schema import FEATURE_COLUMNS
from heart.eval.calibration import DEFAULT_CALIBRATION_METHOD
from heart.eval.selection import DEFAULT_ALPHA, DEFAULT_OBJECTIVE

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "SUPPORTED_FORMAT_VERSIONS",
    "DEFAULT_ARTIFACT_PATH",
    "METADATA_KEYS",
    "ArtifactError",
    "ArtifactDataError",
    "ArtifactSchemaError",
    "ArtifactVersionError",
    "ArtifactPayloadError",
    "FeatureSchemaError",
    "ArtifactMetadata",
    "ServingArtifact",
    "save_artifact",
    "load_artifact",
    "check_features",
    "predict_positive_proba",
    "build_winner_artifact",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Current artifact format version. Bump when the payload shape changes in a
#: way older loaders must refuse.
ARTIFACT_FORMAT_VERSION: int = 1

#: Format versions this loader accepts.
SUPPORTED_FORMAT_VERSIONS: tuple[int, ...] = (1,)

#: Default serialized-artifact location (models/ ships a .gitkeep only).
DEFAULT_ARTIFACT_PATH: str = "models/heart-winner-v1.pkl"

#: Exact metadata key set; a missing or extra key is a schema violation.
METADATA_KEYS: tuple[str, ...] = (
    "format_version",
    "created_at",
    "model_name",
    "feature_columns",
    "threshold",
    "threshold_objective",
    "calibration_method",
    "calibration_applied",
    "brier_before",
    "brier_after",
    "ship",
    "flags",
)


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class ArtifactError(Exception):
    """Base class for every model-artifact failure."""


class ArtifactDataError(ArtifactError):
    """The artifact is missing, unreadable, or structurally malformed."""


class ArtifactSchemaError(ArtifactError):
    """The artifact metadata does not match the declared contract."""


class ArtifactVersionError(ArtifactError):
    """The artifact was written by an unsupported format version."""


class ArtifactPayloadError(ArtifactError):
    """The serialized model payload lacks the serving interface."""


class FeatureSchemaError(ArtifactError):
    """The embedded (or runtime) feature columns mismatch the declared schema."""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ArtifactSchemaError(
            f"{label} must be a mapping with the declared keys "
            f"{list(METADATA_KEYS)}, got {type(payload).__name__}."
        )
    missing = [key for key in METADATA_KEYS if key not in payload]
    if missing:
        raise ArtifactSchemaError(
            f"{label} is missing key(s) {missing}; the artifact is incomplete "
            "and will not be served partially."
        )
    unexpected = [key for key in payload if key not in METADATA_KEYS]
    if unexpected:
        raise ArtifactSchemaError(
            f"{label} carries unexpected key(s) {unexpected}; a misnamed "
            "metadata field is a schema violation, not a new feature."
        )
    return payload


def _require_str(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactSchemaError(
            f"Metadata {label} must be a non-empty string, got {value!r} "
            f"({type(value).__name__})."
        )
    return value


def _require_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ArtifactSchemaError(
            f"Metadata {label} must be a number, got {value!r} "
            f"({type(value).__name__})."
        )
    number = float(value)
    if not np.isfinite(number):
        raise ArtifactSchemaError(f"Metadata {label} is non-finite ({value!r}).")
    return number


@dataclass(frozen=True)
class ArtifactMetadata:
    """Embedded, validated description of a serialized serving artifact."""

    created_at: str
    model_name: str
    feature_columns: tuple[str, ...]
    threshold: float
    threshold_objective: str
    calibration_method: str
    calibration_applied: bool
    brier_before: float | None
    brier_after: float | None
    ship: bool
    flags: tuple[str, ...]
    format_version: int = ARTIFACT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ArtifactVersionError(
                f"Artifact format version {self.format_version} is not in the "
                f"supported set {SUPPORTED_FORMAT_VERSIONS}; bump the loader "
                "before reading this artifact."
            )
        _require_str(self.created_at, label="created_at")
        _require_str(self.model_name, label="model_name")
        _require_str(self.threshold_objective, label="threshold_objective")
        _require_str(self.calibration_method, label="calibration_method")
        if not isinstance(self.calibration_applied, bool):
            raise ArtifactSchemaError(
                "Metadata calibration_applied must be a bool, got "
                f"{self.calibration_applied!r}."
            )
        columns = tuple(self.feature_columns)
        if not columns or not all(isinstance(c, str) and c for c in columns):
            raise ArtifactSchemaError(
                "Metadata feature_columns must be a non-empty tuple of "
                f"non-empty strings, got {columns!r}."
            )
        if len(set(columns)) != len(columns):
            raise ArtifactSchemaError(
                "Metadata feature_columns contains duplicate columns; a "
                "decision feature order must name every column once."
            )
        threshold = _require_float(self.threshold, label="threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ArtifactSchemaError(
                f"Metadata threshold = {threshold!r} is outside [0, 1]; "
                "thresholds are probability operating points."
            )
        if not isinstance(self.ship, bool):
            raise ArtifactSchemaError(
                f"Metadata ship must be a bool, got {self.ship!r}."
            )
        flags = tuple(self.flags)
        if not all(isinstance(flag, str) and flag for flag in flags):
            raise ArtifactSchemaError(
                "Metadata flags must be a tuple of non-empty strings, got "
                f"{flags!r}."
            )
        for label, value in (
            ("brier_before", self.brier_before),
            ("brier_after", self.brier_after),
        ):
            if value is not None:
                number = _require_float(value, label=label)
                if not 0.0 <= number <= 1.0:
                    raise ArtifactSchemaError(
                        f"Metadata {label} = {number!r} is outside [0, 1]; "
                        "Brier scores are probabilities."
                    )
        object.__setattr__(self, "feature_columns", columns)
        object.__setattr__(self, "flags", flags)
        object.__setattr__(self, "threshold", threshold)

    def matches_declared_schema(
        self, declared: Sequence[str] = FEATURE_COLUMNS
    ) -> bool:
        """True exactly when the embedded columns equal the declared order."""
        return tuple(self.feature_columns) == tuple(declared)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "feature_columns": list(self.feature_columns),
            "threshold": self.threshold,
            "threshold_objective": self.threshold_objective,
            "calibration_method": self.calibration_method,
            "calibration_applied": self.calibration_applied,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "ship": self.ship,
            "flags": list(self.flags),
        }


def _metadata_from_mapping(payload: object) -> ArtifactMetadata:
    mapping = dict(_require_mapping(payload, label="artifact metadata"))
    columns = mapping["feature_columns"]
    if not isinstance(columns, (list, tuple)):
        raise ArtifactSchemaError(
            "Metadata feature_columns must be a list, got "
            f"{type(columns).__name__}."
        )
    mapping["feature_columns"] = tuple(columns)
    flags = mapping["flags"]
    if not isinstance(flags, (list, tuple)):
        raise ArtifactSchemaError(
            f"Metadata flags must be a list, got {type(flags).__name__}."
        )
    mapping["flags"] = tuple(flags)
    return ArtifactMetadata(**mapping)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_artifact(
    model: object,
    path: str | Path,
    *,
    created_at: str | None = None,
    **metadata_kwargs: Any,
) -> ArtifactMetadata:
    """Serialize a fitted model plus its metadata to one versioned file.

    ``created_at`` defaults to the current UTC ISO timestamp. The remaining
    keyword arguments are the :class:`ArtifactMetadata` fields; they are
    validated *on save* so a malformed artifact can never be written. The
    model must already be fitted and expose ``predict``/``predict_proba``
    (:class:`ArtifactPayloadError` otherwise).

    Returns the validated :class:`ArtifactMetadata` that was embedded.
    """
    if created_at is None:
        created_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    metadata = ArtifactMetadata(created_at=created_at, **metadata_kwargs)  # type: ignore[arg-type]
    if not hasattr(model, "predict") or not callable(model.predict):
        raise ArtifactPayloadError(
            f"Model payload of type {type(model).__name__} exposes no callable "
            "predict; only serving-ready wrappers are serialized."
        )
    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        raise ArtifactPayloadError(
            f"Model payload of type {type(model).__name__} exposes no callable "
            "predict_proba; the app serves probabilities and thresholds."
        )
    target = Path(path)
    if str(target.parent) and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "metadata": metadata.to_dict(),
        "model": model,
    }
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Saved artifact for %s to %s (%d bytes), format v%d, %d feature columns.",
        metadata.model_name,
        target,
        target.stat().st_size,
        ARTIFACT_FORMAT_VERSION,
        len(metadata.feature_columns),
    )
    return metadata


def load_artifact(path: str | Path) -> "ServingArtifact":
    """Load, validate, and return a :class:`ServingArtifact`.

    Every check happens before anything is served to the caller:

    1. The file must exist and unpickle into the flat declared dict
       (:class:`ArtifactDataError` otherwise).
    2. ``format_version`` must be supported (:class:`ArtifactVersionError`).
    3. The metadata block must satisfy the declared key set and field types
       (:class:`ArtifactSchemaError`).
    4. The embedded feature column order must equal the declared dataset
       schema :data:`heart.data.schema.FEATURE_COLUMNS`
       (:class:`FeatureSchemaError`) — an artifact fitted on a different
       frame shape is refused rather than trusted.
    5. The model payload must expose ``predict`` and ``predict_proba``
       (:class:`ArtifactPayloadError`).

    Security note: pickle loads execute code. Only load artifacts written by
    this project's own training flow; never deserialize files from unknown
    sources.
    """
    target = Path(path)
    if not target.is_file():
        raise ArtifactDataError(
            f"Artifact file {str(target)!r} does not exist or is a directory."
        )
    try:
        with target.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ArtifactDataError(
            f"Artifact file {str(target)!r} could not be read: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactDataError(
            f"Artifact payload in {str(target)!r} is {type(payload).__name__}, "
            "not the declared flat dict."
        )
    for key in ("format_version", "metadata", "model"):
        if key not in payload:
            raise ArtifactDataError(
                f"Artifact payload is missing top-level key {key!r}; it is "
                "incomplete, not partially servable."
            )
    version = payload["format_version"]
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ArtifactVersionError(
            f"Artifact declares format version {version!r} but this loader "
            f"supports {SUPPORTED_FORMAT_VERSIONS}; retrain or upgrade."
        )
    metadata = _metadata_from_mapping(payload["metadata"])
    if not metadata.matches_declared_schema():
        raise FeatureSchemaError(
            f"Embedded feature schema {list(metadata.feature_columns)} does "
            f"not match the declared schema {list(FEATURE_COLUMNS)}; a "
            "mismatched feature order silently permutes the model's inputs."
        )
    model = payload["model"]
    if not hasattr(model, "predict") or not callable(model.predict):
        raise ArtifactPayloadError(
            f"Serialized model payload of type {type(model).__name__} exposes "
            "no callable predict; refusing to return a non-servable payload."
        )
    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        raise ArtifactPayloadError(
            f"Serialized model payload of type {type(model).__name__} exposes "
            "no callable predict_proba."
        )
    logger.info(
        "Loaded artifact %s (model=%s, format v%d, threshold=%.2f, %d columns).",
        target,
        metadata.model_name,
        metadata.format_version,
        metadata.threshold,
        len(metadata.feature_columns),
    )
    return ServingArtifact(model=model, metadata=metadata, path=target)


# ---------------------------------------------------------------------------
# Runtime feature-schema guard + serving path
# ---------------------------------------------------------------------------


def check_features(
    frame: object, metadata: ArtifactMetadata | "ServingArtifact"
) -> pd.DataFrame:
    """Validate an input frame against the embedded feature schema.

    Returns a copy of the frame in the embedded column order — the model's
    declared input — or raises :class:`FeatureSchemaError` naming the exact
    mismatch. Missing, unexpected, and reordered columns are separate,
    explicit failures; the shared-column subset is checked after the set
    check, so a reordered frame can never mask as valid.
    """
    if isinstance(metadata, ServingArtifact):
        metadata = metadata.metadata
    if not isinstance(frame, pd.DataFrame):
        raise FeatureSchemaError(
            f"Serving input must be a pandas.DataFrame, got "
            f"{type(frame).__name__}."
        )
    expected = list(metadata.feature_columns)
    missing = [c for c in expected if c not in frame.columns]
    unexpected = [c for c in frame.columns if c not in expected]
    if missing or unexpected:
        raise FeatureSchemaError(
            "Serving input mismatches the artifact's feature schema: "
            f"missing={missing}, unexpected={unexpected}. The model was "
            f"fitted on columns {expected}."
        )
    observed = list(frame.columns)
    if observed != expected:
        # Same column set, wrong order: reordering is contract, not cosmetic.
        raise FeatureSchemaError(
            f"Serving input columns are reordered: got {observed}, expected "
            f"{expected}. A reordered frame silently permutes the model's "
            "inputs, which is exactly what this guard refuses."
        )
    subset = frame[expected]
    return subset.copy()


@dataclass(frozen=True)
class ServingArtifact:
    """A loaded artifact: validated model + metadata + its file path."""

    model: object
    metadata: ArtifactMetadata
    path: Path

    def predict_positive_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Validated P(y=1|x) from the artifact's model.

        Raises :class:`FeatureSchemaError` on any column mismatch, so a
        reordered or partial frame fails loudly instead of predicting
        silently-wrong numbers, and :class:`ArtifactPayloadError` if the
        payload's probabilities are malformed.
        """
        features = check_features(frame, self)
        probabilities = np.asarray(self.model.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ArtifactPayloadError(
                "predict_proba must return an (n_samples, 2) matrix; got "
                f"shape {probabilities.shape}."
            )
        rows = probabilities[:, 1]
        if rows.size != len(frame):
            raise ArtifactPayloadError(
                f"predict_proba produced {rows.size} rows for {len(frame)} "
                "input rows."
            )
        return rows


def predict_positive_proba(
    artifact: ServingArtifact, frame: pd.DataFrame
) -> np.ndarray:
    """Module-level alias of :meth:`ServingArtifact.predict_positive_proba`."""
    return artifact.predict_positive_proba(frame)


# ---------------------------------------------------------------------------
# Real artifact generation entrypoint (the app's actual artifact)
# ---------------------------------------------------------------------------

#: Registry members the generation flow compares to name the winner (same
#: defaults as the T03 selection flow, so the artifact matches selection.md).
_GENERATION_CANDIDATES: tuple[str, ...] = (
    "logistic-regression-l2",
    "random-forest",
)


def serving_pickle_size_mb(model: object) -> float:
    """Delegate the serving-weight probe to the selection module's measure."""
    from heart.eval.selection import _pickle_size_mb

    return _pickle_size_mb(model)


def build_winner_artifact(
    *,
    split_version: str = "v1",
    candidate_types: Sequence[str] = _GENERATION_CANDIDATES,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = RANDOM_SEED,
    objective: str = DEFAULT_OBJECTIVE,
    alpha: float = DEFAULT_ALPHA,
    path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> tuple[ServingArtifact, Path]:
    """Regenerate the real selected winner and serialize it into ``models/``.

    This reuses the S06 selection flow — it does not re-decide anything:
    :mod:`heart.eval.selection` owns the statistical decision and the
    committed ``reports/selection.md``; this function rebuilds the winner
    *exactly as that flow did* (same split, same carving, same seeds, the
    real :func:`heart.eval.selection.select_model` gates) and serializes
    the serving object: the calibration wrapper around the fitted winner.
    A NO-SHIP decision is never silently converted into a shippable
    artifact verdict — the ``ship`` flag and the selection's named flags
    ride along in the metadata, so the app can gate on them visibly.

    Selection-side failures bubble as the named :class:`heart.eval.Selection`
    error family; serialization and load-time validation failures as this
    module's :class:`ArtifactError` family.
    """
    from sklearn.model_selection import train_test_split as _split

    from heart.data.schema import TARGET_COLUMN
    from heart.data.split import load_split_frames
    from heart.eval.calibration import fit_calibration_wrapper
    from heart.eval.repeated_cv import (
        RepeatedCVConfig,
        run_repeated_cv_comparison,
        rank_candidates,
    )
    from heart.eval.selection import select_model
    from heart.models.registry import build_pipeline, resolve_spec

    train, _test = load_split_frames(version=split_version)
    features = train[list(FEATURE_COLUMNS)].copy()
    labels = train[TARGET_COLUMN].astype(int)

    # Identical carving to the T03 selection flow: validation rows carved
    # first, then a calibration split from the remainder. Winner fitting,
    # repeated CV, and calibration never touch the validation rows used for
    # threshold tuning.
    x_rest, x_val, y_rest, y_val = _split(
        features, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    x_train, x_calib, y_train, y_calib = _split(
        x_rest, y_rest, test_size=0.20, random_state=seed, stratify=y_rest
    )

    candidates = {
        model_type: (lambda mt=model_type: build_pipeline(resolve_spec(mt)))
        for model_type in candidate_types
    }
    config = RepeatedCVConfig(n_folds=n_folds, n_repeats=n_repeats, seed=seed)
    results = run_repeated_cv_comparison(candidates, x_train, y_train, config=config)

    winner = rank_candidates(results)[0][0]
    fitted_winner = candidates[winner]()
    fitted_winner.fit(x_train, y_train)
    serving_weights: dict[str, float] = {}
    for model_type in candidate_types:
        probe = candidates[model_type]()
        probe.fit(x_train, y_train)
        serving_weights[model_type] = serving_pickle_size_mb(probe)

    decision = select_model(
        results,
        fitted_winner=fitted_winner,
        X_calib=x_calib,
        y_calib=y_calib,
        X_val=x_val,
        y_val=y_val,
        serving_weights=serving_weights,
        objective=objective,
        alpha=alpha,
    )

    # The serialized payload must reproduce the decision's measured
    # probabilities, so the wrapper is fitted on the identical (frozen
    # winner + calibration split) pairing the decision used — not a
    # sibling wrapper with a different fit.
    fitted_for_calibration = candidates[decision.winner]()
    fitted_for_calibration.fit(x_train, y_train)
    served_model = fit_calibration_wrapper(
        fitted_for_calibration, x_calib, y_calib, method=DEFAULT_CALIBRATION_METHOD
    )

    metadata = ArtifactMetadata(
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        model_name=decision.winner,
        feature_columns=tuple(FEATURE_COLUMNS),
        threshold=decision.threshold.best_threshold,
        threshold_objective=decision.threshold.objective,
        calibration_method=DEFAULT_CALIBRATION_METHOD,
        calibration_applied=True,
        brier_before=decision.calibration.brier_score_raw,
        brier_after=decision.calibration.brier_score_calibrated,
        ship=decision.ship,
        flags=decision.flags,
    )
    saved_metadata = save_artifact(
        served_model,
        path,
        created_at=metadata.created_at,
        model_name=metadata.model_name,
        feature_columns=metadata.feature_columns,
        threshold=metadata.threshold,
        threshold_objective=metadata.threshold_objective,
        calibration_method=metadata.calibration_method,
        calibration_applied=metadata.calibration_applied,
        brier_before=metadata.brier_before,
        brier_after=metadata.brier_after,
        ship=metadata.ship,
        flags=metadata.flags,
    )
    target = path
    artifact = load_artifact(target)
    return artifact, target


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m heart.serving.artifact",
        description="Rebuild the selected winner and serialize the serving artifact.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(_GENERATION_CANDIDATES),
        help="Registry model types to compare (winner is ranked top).",
    )
    parser.add_argument("--path", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact, written = build_winner_artifact(
        candidate_types=tuple(args.candidates),
        split_version=args.version,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        objective=args.objective,
        alpha=args.alpha,
        path=args.path,
    )
    print(
        f"wrote {written} (model={artifact.metadata.model_name}, "
        f"threshold={artifact.metadata.threshold:.2f}, "
        f"ship={artifact.metadata.ship})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
