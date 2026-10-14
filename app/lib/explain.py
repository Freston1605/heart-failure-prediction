"""Never-raise SHAP attribution (per-patient + global cohort) (M007/S02, S03).

What this module explains
-------------------------
The attribution explains the **raw, pre-calibration RF score** — the winning
estimator's own P(y=1|x) before the calibration wrapper reshapes it (per
D011/MEM118: the served probability is calibrated, but the explanation is
honest about what the forest itself voted). The calibrated value is what the
app serves; the attribution names the features that pushed *this* patient's
raw score up or down.

The **global cohort drivers** (M007/S03) explain that *same* raw
pre-calibration RF score over a deterministically sampled cohort of the
pinned 918-row dataset (D011/MEM118 seam stays honest: one estimator, one
seam — the cohort section explains the forest's vote, never the isotonic
mapping). Both surfaces ride :class:`heart.serving.artifact.ServingArtifact`
``raw_estimator`` — no parallel model path.

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
:func:`explain_patient` is the **only entry the page consumes** for the
per-patient chart: it never raises — every outcome is an
:class:`ExplanationOutcome` holding either an attribution or the named
error, so a SHAP failure can never white-screen a page that already served
a probability.

Global cohort contract (M007/S03)
---------------------------------
:func:`explain_global` is the global twin of ``explain_patient``: same
byte-identical ``check_features`` schema guard, same
``_as_explainable_pipeline`` naming adapter, and the entire computation
delegated to :func:`heart.interpret.shap_export.compute_shap_values`
(with an explicit ``max_rows`` cap — the repo default cap is ``None``, so
free-tier load shedding must be requested) +
:func:`heart.interpret.shap_export.summarize_shap`. Never a second
explainer. Every failure maps through the same named kinds; no exception
escapes.

:func:`cohort_drivers` is the session glue the page calls: it loads the
serving artifact, loads the pinned dataset, and delegates to
``explain_global``. It is wrapped in ``st.cache_resource`` **by design**:
the cohort is computed once per session, so both the success and any
*failed* outcome are sticky for the session (the cached-error precedent
from :mod:`app.lib.plots`) and only :func:`clear_cohort_cache` drops them.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import heart.interpret.shap_export as shap_export
from app.lib.artifact_loader import load_winning_artifact
from app.lib.errors import APP_ERROR_KINDS
from app.lib.plots import load_explore_data
from heart.config import RANDOM_SEED
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
    "COHORT_MAX_ROWS",
    "ExplanationError",
    "PatientAttribution",
    "ExplanationOutcome",
    "CohortDrivers",
    "CohortOutcome",
    "attribute_patient",
    "explain_patient",
    "explain_global",
    "cohort_drivers",
    "clear_cohort_cache",
]


#: Stable named kinds surfaced by the explanation contract.
#:
#: The first three are the compute kinds produced by
#: :func:`_to_explanation_error` (used by the per-patient and the global
#: compute paths alike). The rest are glue-level kinds for the global
#: cohort path (M007/S03): ``dataset-unavailable`` when the pinned dataset
#: cannot back the cohort, ``cohort-failed`` for a call-site failure around
#: the glue, and the ``artifact-*`` family re-surfaced verbatim from
#: :data:`app.lib.errors.APP_ERROR_KINDS` when the cohort glue cannot load
#: the serving artifact (so an :class:`ExplanationError` can carry the
#: loader's named kind without dropping it).
EXPLANATION_ERROR_KINDS: tuple[str, ...] = (
    "shap-unavailable",
    "explanation-unsupported",
    "explanation-failed",
    "dataset-unavailable",
    "cohort-failed",
    *APP_ERROR_KINDS,
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


# ---------------------------------------------------------------------------
# Global cohort drivers (M007/S03)
# ---------------------------------------------------------------------------

#: Row cap for the global cohort explanation — the sampling cap and
#: free-tier load-shed control (a sample of the pinned 918-row dataset).
#: ``shap_export.DEFAULT_MAX_SAMPLES`` is ``None`` (explain every row), so
#: the cap must be requested explicitly on every call.
COHORT_MAX_ROWS: int = 200


@dataclass(frozen=True, eq=False)
class CohortDrivers:
    """Mean |SHAP| and mean signed SHAP per feature over the sampled cohort.

    ``mean_abs`` is sorted descending (ties keep model column order — the
    same stable ranking ``ShapSummary._order`` uses), so
    ``display_names[0]`` is the top driver and ``top_feature`` is pinned to
    the max-mean-|SHAP| display name. ``n_rows`` is the number of rows that
    were actually explained (the deterministic sample size).
    """

    feature_names: tuple[str, ...]
    display_names: tuple[str, ...]
    mean_abs: tuple[float, ...]  # descending
    mean_signed: tuple[float, ...]
    n_rows: int  # rows actually explained
    top_feature: str  # display name of the max mean_abs feature

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        display = tuple(self.display_names)
        mean_abs = tuple(float(value) for value in self.mean_abs)
        mean_signed = tuple(float(value) for value in self.mean_signed)
        if not (len(names) == len(display) == len(mean_abs) == len(mean_signed)):
            raise ValueError(
                "CohortDrivers fields must have equal length "
                f"({len(names)}, {len(display)}, {len(mean_abs)}, "
                f"{len(mean_signed)})."
            )
        if not mean_abs:
            raise ValueError("CohortDrivers requires at least one feature.")
        if not all(np.isfinite(value) and value >= 0 for value in mean_abs):
            raise ValueError("mean_abs entries must be finite and non-negative.")
        if not all(np.isfinite(value) for value in mean_signed):
            raise ValueError("mean_signed entries must be finite.")
        if any(left < right for left, right in zip(mean_abs, mean_abs[1:])):
            raise ValueError("mean_abs must be sorted descending.")
        if (
            isinstance(self.n_rows, bool)
            or not isinstance(self.n_rows, (int, np.integer))
            or self.n_rows < 1
        ):
            raise ValueError(f"n_rows must be a positive int, got {self.n_rows!r}.")
        expected = display[int(np.argmax(mean_abs))]
        if self.top_feature != expected:
            raise ValueError(
                f"top_feature {self.top_feature!r} is not the max mean_abs "
                f"feature {expected!r}."
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "display_names", display)
        object.__setattr__(self, "mean_abs", mean_abs)
        object.__setattr__(self, "mean_signed", mean_signed)
        object.__setattr__(self, "n_rows", int(self.n_rows))


@dataclass(frozen=True, eq=False)
class CohortOutcome:
    """Exactly one of ``drivers`` / ``error`` is set; never neither.

    Mirrors :class:`ExplanationOutcome` so the page branches the same way
    for the per-patient and the global section.
    """

    drivers: CohortDrivers | None
    error: ExplanationError | None

    def __post_init__(self) -> None:
        if (self.drivers is None) == (self.error is None):
            raise ValueError(
                "CohortOutcome requires exactly one of drivers/error."
            )

    @property
    def ok(self) -> bool:
        return self.drivers is not None

    @property
    def degraded(self) -> bool:
        return self.error is not None


def _as_cohort_drivers(summary: shap_export.ShapSummary) -> CohortDrivers:
    """Map the sanctioned ShapSummary into the frozen descending contract.

    Reuses ``ShapSummary._order``'s ranking key exactly (-mean_abs, then
    column index) so the descending order is deterministic under ties.
    """
    order = sorted(
        range(len(summary.feature_names)),
        key=lambda index: (-summary.mean_abs_shap[index], index),
    )
    if not order:
        raise ShapComputationError(
            "The SHAP summary carries no features; refusing to serve an "
            "empty cohort-drivers section."
        )
    return CohortDrivers(
        feature_names=tuple(summary.feature_names[index] for index in order),
        display_names=tuple(summary.display_names[index] for index in order),
        mean_abs=tuple(float(summary.mean_abs_shap[index]) for index in order),
        mean_signed=tuple(float(summary.mean_shap[index]) for index in order),
        n_rows=int(summary.n_samples),
        top_feature=str(summary.display_names[order[0]]),
    )


def explain_global(
    artifact,
    frame,
    *,
    max_rows: int = COHORT_MAX_ROWS,
    seed: int | None = None,
) -> CohortOutcome:
    """Never-raise global cohort attribution: drivers or named error.

    Runs the byte-identical serving schema guard first
    (:func:`heart.serving.artifact.check_features`), then explains
    ``artifact.raw_estimator`` over at most ``max_rows`` deterministically
    sampled rows (``random_state`` defaults to
    :data:`heart.config.RANDOM_SEED` so repeated runs explain the same
    cohort) and reduces the matrix through the sanctioned
    :func:`heart.interpret.shap_export.summarize_shap`. No exception of
    any kind escapes: every failure comes back as a named ``error`` via
    :func:`_to_explanation_error`.
    """
    try:
        features = check_features(frame, artifact)
        pipeline = _as_explainable_pipeline(artifact.raw_estimator)
        computed = shap_export.compute_shap_values(
            pipeline,
            features,
            max_samples=max_rows,
            random_state=RANDOM_SEED if seed is None else seed,
        )
        drivers = _as_cohort_drivers(shap_export.summarize_shap(computed))
    except Exception as exc:  # noqa: BLE001 - the boundary maps everything
        error = _to_explanation_error(exc)
        logger.warning(
            "Global cohort explanation degraded [%s]: %s",
            error.kind,
            error.user_message,
        )
        return CohortOutcome(drivers=None, error=error)
    logger.debug(
        "Cohort drivers: %d row(s) x %d feature(s), top=%s",
        drivers.n_rows,
        len(drivers.feature_names),
        drivers.top_feature,
    )
    return CohortOutcome(drivers=drivers, error=None)


@st.cache_resource(
    show_spinner="Computing what drives scores across patients (once per session)..."
)
def cohort_drivers() -> CohortOutcome:
    """Session-cached global cohort drivers: never raises, computed once.

    Zero-argument by design: the cache key is the function itself, which
    sidesteps Streamlit's argument hashing of the (non-hashable)
    ``ServingArtifact``. Load order mirrors the page: serving artifact,
    then the pinned dataset, then :func:`explain_global`.

    Global cohort drivers are session-cached **by design**: the successful
    outcome AND any failed outcome are sticky for the whole session (the
    cached-error precedent from :mod:`app.lib.plots`) until
    :func:`clear_cohort_cache` runs — a deliberate trade so the free tier
    pays the cohort SHAP cost (and any failure) exactly once per session.
    """
    try:
        loaded = load_winning_artifact()
        if not loaded.ok:
            artifact_error = loaded.error
            kind = (
                artifact_error.kind
                if artifact_error is not None
                else "artifact-unknown"
            )
            message = (
                artifact_error.user_message
                if artifact_error is not None
                else (
                    "The trained model could not be loaded, so the cohort "
                    "drivers cannot be computed."
                )
            )
            return CohortOutcome(
                drivers=None, error=ExplanationError(kind, message)
            )
        data = load_explore_data()
        if not data.ok:
            return CohortOutcome(
                drivers=None,
                error=ExplanationError(
                    "dataset-unavailable",
                    data.error_message
                    or (
                        "The pinned dataset could not be loaded, so the "
                        "cohort drivers cannot be computed."
                    ),
                ),
            )
        try:
            features = data.frame[list(loaded.artifact.metadata.feature_columns)]
        except Exception as exc:  # noqa: BLE001 - dataset-side stays dataset-side
            logger.warning("Cohort frame selection failed: %s", exc)
            return CohortOutcome(
                drivers=None,
                error=ExplanationError(
                    "dataset-unavailable",
                    "The pinned dataset does not carry the features the model "
                    f"needs ({exc}), so the cohort drivers cannot be computed.",
                ),
            )
        return explain_global(loaded.artifact, features)
    except Exception as exc:  # noqa: BLE001 - the glue swallows everything
        error = _to_explanation_error(exc)
        logger.warning(
            "Cohort drivers glue degraded [%s]: %s", error.kind, error.user_message
        )
        return CohortOutcome(drivers=None, error=error)


def clear_cohort_cache() -> None:
    """Drop the session-cached cohort outcome (mirrors clear_explore_cache).

    The only way a sticky failed outcome leaves the session: tests clear it
    around each run, and a host app can clear it after restoring the dataset
    or artifact that caused the degradation.
    """
    cohort_drivers.clear()


# Public-surface guard: the contract is non-empty by construction.
_PUBLIC: tuple[object, ...] = (
    EXPLANATION_ERROR_KINDS,
    COHORT_MAX_ROWS,
    ExplanationError,
    PatientAttribution,
    ExplanationOutcome,
    attribute_patient,
    explain_patient,
    CohortDrivers,
    CohortOutcome,
    explain_global,
    cohort_drivers,
    clear_cohort_cache,
)
assert _PUBLIC
