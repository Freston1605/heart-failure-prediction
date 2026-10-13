"""Never-raise per-patient SHAP attribution for the serving app (M007/S02).

What this module explains
-------------------------
The attribution explains the **raw, pre-calibration RF score** — the winning
estimator's own P(y=1|x) before the calibration wrapper reshapes it (per
D011/MEM118: the served probability is calibrated, but the explanation is
honest about what the forest itself voted). The calibrated value is what the
app serves; the attribution names the features that pushed *this* patient's
raw score up or down.

Where the computation lives
---------------------------
This module never reimplements SHAP. It rides
:class:`heart.serving.artifact.ServingArtifact.raw_estimator` — the unfrozen
base estimator behind the calibration wrapper — and delegates the entire
TreeExplainer computation to
:func:`heart.interpret.shap_export.compute_shap_values`, the repo's single
sanctioned SHAP contract (transform via the pipeline prefix, explain the
``clf`` step, reduce to positive-class values with transformed + display
feature names and a base value). One adapter, no second explainer.

Failure contract
----------------
:func:`attribute_patient` is the raising path (schema guard
:class:`~heart.serving.artifact.FeatureSchemaError` propagates unchanged so
callers see the exact serving-boundary failure); every other failure is
mapped to a named :class:`ExplanationError` carrying a stable ``kind`` in
:data:`EXPLANATION_ERROR_KINDS` and a friendly, plain-language
``user_message`` mirroring :mod:`app.lib.errors`.
:func:`explain_patient` is the **only entry the page consumes**: it never
raises — every outcome is an :class:`ExplanationOutcome` holding either an
attribution or the named error, so a SHAP failure can never white-screen a
page that already served a probability.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import heart.interpret.shap_export as shap_export
from heart.interpret.shap_export import (
    ShapComputationError,
    ShapDataError,
    ShapDependencyError,
    UnsupportedShapModelError,
)
from heart.serving.artifact import (
    ArtifactPayloadError,
    FeatureSchemaError,
    check_features,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXPLANATION_ERROR_KINDS",
    "ExplanationError",
    "PatientAttribution",
    "ExplanationOutcome",
    "attribute_patient",
    "explain_patient",
]


#: Stable named kinds surfaced by the explanation contract.
EXPLANATION_ERROR_KINDS: tuple[str, ...] = (
    "shap-unavailable",
    "explanation-unsupported",
    "explanation-failed",
)

_SHAP_UNAVAILABLE = (
    "The explanation engine (SHAP) is not available in this environment, "
    "so this patient's score cannot be explained right now. The prediction "
    "above was served normally; only the explanation is affected. Install "
    "the ml extra (pip install -e '.[ml]') to restore explanations."
)
_UNSUPPORTED = (
    "This trained model file cannot produce per-patient explanations: its "
    "payload does not expose the raw pre-calibration estimator the "
    "explanation contract requires. The prediction above was served "
    "normally; only the explanation is affected. Re-run the "
    "winner-generation flow to recreate the artifact."
)
_FAILED = (
    "Something went wrong while computing this patient's explanation, so it "
    "was skipped rather than shown incorrectly. The prediction above was "
    "served normally; only the explanation is affected. Re-running the "
    "prediction may succeed; if it keeps failing, re-export the artifact."
)


class ExplanationError(Exception):
    """An explanation failure translated into user-facing language.

    Mirrors :class:`app.lib.errors.AppError`: a stable ``kind`` (for tests
    and diagnostics) plus a plain-language ``user_message`` with a next
    action, never a stack trace.
    """

    kind: str
    user_message: str

    def __init__(self, kind: str, user_message: str) -> None:
        if kind not in EXPLANATION_ERROR_KINDS:
            raise ValueError(
                f"kind must be one of {EXPLANATION_ERROR_KINDS}, got {kind!r}"
            )
        self.kind = kind
        self.user_message = user_message
        super().__init__(user_message)

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"[{self.kind}] {self.user_message}"


@dataclass(frozen=True, eq=False)
class PatientAttribution:
    """Positive-class SHAP attribution for a single explained patient."""

    feature_names: tuple[str, ...]
    display_names: tuple[str, ...]
    values: np.ndarray  # 1-D signed SHAP values for the single explained row
    base_value: float
    explainer_class: str

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        display = tuple(self.display_names)
        array = np.asarray(self.values, dtype=float)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(
                f"values must be a non-empty 1-D array, got shape {array.shape}."
            )
        if not (len(names) == len(display) == array.size):
            raise ValueError(
                "feature_names, display_names, and values must have equal "
                f"length ({len(names)}, {len(display)}, {array.size})."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("values contains non-finite entry(ies).")
        base = float(self.base_value)
        if not np.isfinite(base):
            raise ValueError(f"base_value must be finite, got {self.base_value!r}.")
        if not self.explainer_class:
            raise ValueError("explainer_class must be a non-empty string.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "display_names", display)
        object.__setattr__(self, "values", array)
        object.__setattr__(self, "base_value", base)

    @property
    def n_features(self) -> int:
        return int(self.values.size)

    @property
    def top_feature(self) -> str:
        """Display name of the feature that moved this score the most."""
        return self.display_names[int(np.argmax(np.abs(self.values)))]


@dataclass(frozen=True, eq=False)
class ExplanationOutcome:
    """Exactly one of ``attribution`` / ``error`` is set; never neither."""

    attribution: PatientAttribution | None
    error: ExplanationError | None

    def __post_init__(self) -> None:
        if (self.attribution is None) == (self.error is None):
            raise ValueError(
                "ExplanationOutcome requires exactly one of attribution/error."
            )

    @property
    def ok(self) -> bool:
        return self.attribution is not None

    @property
    def degraded(self) -> bool:
        return self.error is not None


def _as_explainable_pipeline(raw: object) -> object:
    """View the serving pipeline under the step names shap_export expects.

    The serving registry pipeline names its transformer step ``preprocess``
    while :func:`heart.interpret.shap_export.compute_shap_values` reads the
    transformed feature names from a ``features`` step (the interpretability
    pipeline's convention). This builds a zero-copy *view* renaming that one
    step, reusing the exact same fitted transformer and classifier objects —
    a naming adapter, never a refit and never a second explainer. Pipelines
    that already have a ``features`` step (or that are not pipelines) are
    returned unchanged so the named shap_export contract applies as-is.
    """
    from sklearn.pipeline import Pipeline

    if not isinstance(raw, Pipeline) or "clf" not in raw.named_steps:
        return raw
    if "features" in raw.named_steps:
        return raw
    steps = list(raw.steps)
    if len(steps) < 2 or steps[0][0] == "clf":
        return raw
    return Pipeline(steps=[("features", steps[0][1]), *steps[1:]])


def _to_explanation_error(exc: Exception) -> ExplanationError:
    """Map any exception escaping the explanation path to a named kind."""
    if isinstance(exc, ExplanationError):
        return exc
    if isinstance(exc, ShapDependencyError) or isinstance(exc, ImportError):
        return ExplanationError("shap-unavailable", _SHAP_UNAVAILABLE)
    if isinstance(
        exc, (ArtifactPayloadError, ShapDataError, UnsupportedShapModelError)
    ):
        return ExplanationError("explanation-unsupported", _UNSUPPORTED)
    # FeatureSchemaError and every remaining failure (ShapComputationError
    # included) land here: the computation failed or the input was invalid.
    if isinstance(exc, FeatureSchemaError):
        return ExplanationError("explanation-unsupported", _UNSUPPORTED)
    return ExplanationError("explanation-failed", _FAILED)


def attribute_patient(artifact, frame) -> PatientAttribution:
    """Attribute one patient's RAW pre-calibration score (raising path).

    Runs the byte-identical serving schema guard first
    (:func:`heart.serving.artifact.check_features`, whose
    :class:`FeatureSchemaError` propagates unchanged), then delegates the
    SHAP computation for ``artifact.raw_estimator`` to
    :func:`heart.interpret.shap_export.compute_shap_values` and extracts row
    0. Every other failure is mapped to a named :class:`ExplanationError`.
    """
    try:
        features = check_features(frame, artifact)
        pipeline = _as_explainable_pipeline(artifact.raw_estimator)
        computed = shap_export.compute_shap_values(
            pipeline, features, max_samples=None
        )
        values = np.asarray(computed.values, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ShapComputationError(
                "SHAP values for a single explained row must be a 2-D "
                f"(1, n_features) matrix, got shape {values.shape}."
            )
        base = computed.base_value
        if base is None or not np.isfinite(float(base)):
            raise ShapComputationError(
                "The SHAP explainer produced no finite base value; refusing "
                "to serve an attribution that cannot be tied back to the "
                "raw score."
            )
        attribution = PatientAttribution(
            feature_names=tuple(computed.feature_names),
            display_names=tuple(computed.display_names),
            values=values[0].copy(),
            base_value=float(base),
            explainer_class=str(computed.explainer_class),
        )
    except (FeatureSchemaError, ExplanationError):
        raise
    except Exception as exc:  # noqa: BLE001 - the boundary maps everything
        raise _to_explanation_error(exc) from exc
    logger.debug(
        "Attributed one patient: %d feature(s) via %s, top=%s",
        attribution.n_features,
        attribution.explainer_class,
        attribution.top_feature,
    )
    return attribution


def explain_patient(artifact, frame) -> ExplanationOutcome:
    """Never-raise attribution entry: outcome carries attribution or error.

    This is the ONLY entry the page consumes. No exception of any kind
    escapes it: schema failures, missing shap, unsupported payloads, and
    computation crashes all come back as a named ``error`` so an already
    served probability can never be white-screened by an explanation.
    """
    try:
        attribution = attribute_patient(artifact, frame)
    except Exception as exc:  # noqa: BLE001 - the boundary swallows everything
        return ExplanationOutcome(attribution=None, error=_to_explanation_error(exc))
    return ExplanationOutcome(attribution=attribution, error=None)


# Public-surface guard: the contract is non-empty by construction.
_PUBLIC: tuple[object, ...] = (
    EXPLANATION_ERROR_KINDS,
    ExplanationError,
    PatientAttribution,
    ExplanationOutcome,
    attribute_patient,
    explain_patient,
)
assert _PUBLIC
