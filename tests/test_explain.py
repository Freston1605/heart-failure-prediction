"""Per-patient explanation contract tests (M007/S02/T01).

Pinned contract of the T01 deliverable:

1. **Determinism** — two ``attribute_patient`` calls on the same held-out
   patient return byte-identical values (``np.array_equal``) and identical
   top-3 pushed-up / pushed-down display names, on the real artifact.
2. **Independent pin** — the values equal a *direct*
   ``shap.TreeExplainer(clf).shap_values(preprocess.transform(row))``
   positive-class row (allclose 1e-8), and the display names equal
   ``display_feature_name`` applied to the transformer's real output
   columns — so explain.py is pinned to the math, not to itself.
3. **Additivity** — ``base_value + values.sum()`` reproduces the artifact's
   raw pre-calibration score ``predict_raw_positive_proba(row)[0]`` within
   1e-6, tying the attribution to the RAW score (D011/MEM118).
4. **Never-raise contract** — a forced computation failure maps to
   ``explanation-failed``, a missing shap import to ``shap-unavailable``,
   and a plain-pipeline payload (no calibration wrapper) to
   ``explanation-unsupported``; ``explain_patient`` never raises.
5. **Schema guard** — a reordered-columns frame raises
   ``FeatureSchemaError`` on the raising path and degrades on the
   never-raise path; an empty frame is rejected.
6. **Import layer** — the app sources never ``import shap`` directly; the
   only sanctioned path is app.lib.explain over heart.interpret.shap_export.
7. **Global cohort drivers (M007/S03)** — ``explain_global`` returns a
   frozen, descending-sorted ``CohortDrivers`` pinned to a direct
   ``shap_export.compute_shap_values`` + ``summarize_shap`` run
   (deterministic across calls, ``n_rows == COHORT_MAX_ROWS``), degrades
   with named kinds for schema / compute / shap failures without ever
   raising, and the session-cached ``cohort_drivers`` glue never raises
   either (``dataset-unavailable`` on a failed dataset load; the
   process-global cache is cleared around every test).

The real-artifact pins skip cleanly only when the artifact is absent; with
the artifact present (the done-when condition) they all execute for real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib.artifact_loader import DEFAULT_ARTIFACT_PATH  # noqa: E402
from app.lib.explain import (  # noqa: E402
    EXPLANATION_ERROR_KINDS,
    ExplanationError,
    ExplanationOutcome,
    PatientAttribution,
    attribute_patient,
    explain_patient,
)
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from heart.data.split import load_split_frames  # noqa: E402
from heart.interpret import shap_export  # noqa: E402
from heart.interpret.model import display_feature_name  # noqa: E402
from heart.serving.artifact import (  # noqa: E402
    ArtifactMetadata,
    FeatureSchemaError,
    ServingArtifact,
    load_artifact,
)

# M007/S03: global cohort contract imports (separate block so the S02
# import block above stays byte-identical).
from app.lib import explain as explain_lib  # noqa: E402
from app.lib.explain import (  # noqa: E402
    COHORT_MAX_ROWS,
    CohortDrivers,
    CohortOutcome,
    clear_cohort_cache,
    cohort_drivers,
    explain_global,
    _as_explainable_pipeline,
)
from app.lib.plots import ExploreData  # noqa: E402
from heart.config import RANDOM_SEED  # noqa: E402
from heart.data.load import load_dataset  # noqa: E402

REAL_ARTIFACT = PROJECT_ROOT / DEFAULT_ARTIFACT_PATH

N_HELD_OUT_PATIENTS = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_artifact():
    """The real serialized winning model (skip cleanly if not serialized)."""
    if not REAL_ARTIFACT.is_file():
        pytest.skip("Real artifact not present on disk; run the S06 flow first.")
    return load_artifact(REAL_ARTIFACT)


@pytest.fixture(scope="module")
def held_out(real_artifact):
    """First rows of the v1 held-out split, in the artifact's feature order."""
    _, test = load_split_frames(version="v1")
    return test.reindex(columns=list(real_artifact.metadata.feature_columns)).head(
        N_HELD_OUT_PATIENTS
    )


def _top_names(attribution: PatientAttribution, direction: str, k: int = 3):
    """Top-k display names pushed up (value > 0) or down (value < 0)."""
    values = attribution.values
    order = np.argsort(-values) if direction == "up" else np.argsort(values)
    picked = [
        index
        for index in order
        if (values[index] > 0 if direction == "up" else values[index] < 0)
    ]
    return tuple(attribution.display_names[index] for index in picked[:k])


def _plain_pipeline_artifact() -> tuple[ServingArtifact, pd.DataFrame]:
    """A synthetic artifact whose payload is a plain pipeline (no wrapper).

    Same fixture style as tests/test_artifact.py's plain-pipeline payload
    test: the load contract is unchanged, but raw access fails at access
    time with a named ArtifactPayloadError -> explanation-unsupported.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from heart.data.pipeline import build_preprocessing_pipeline

    train, _ = load_split_frames(version="v1")
    frame = train.head(60)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", RandomForestClassifier(n_estimators=8, random_state=7)),
        ]
    )
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_COLUMN].astype(int))
    metadata = ArtifactMetadata(
        created_at="2026-10-13T00:00:00+00:00",
        model_name="plain-test",
        feature_columns=tuple(FEATURE_COLUMNS),
        threshold=0.5,
        threshold_objective="f1",
        calibration_method="isotonic",
        calibration_applied=True,
        brier_before=0.20,
        brier_after=0.15,
        ship=True,
        flags=(),
    )
    artifact = ServingArtifact(model=pipeline, metadata=metadata, path=Path("unused.pkl"))
    return artifact, frame


# ---------------------------------------------------------------------------
# 1. Determinism on known held-out patients (real artifact)
# ---------------------------------------------------------------------------


def test_attribution_deterministic_across_calls(real_artifact, held_out) -> None:
    """Same patient -> byte-identical values and stable top-3 names."""
    for position in range(N_HELD_OUT_PATIENTS):
        row = held_out.iloc[[position]]
        first = attribute_patient(real_artifact, row)
        second = attribute_patient(real_artifact, row)

        assert isinstance(first, PatientAttribution)
        assert np.array_equal(first.values, second.values)
        assert first.feature_names == second.feature_names
        assert first.display_names == second.display_names
        assert first.base_value == second.base_value
        assert first.explainer_class == second.explainer_class

        assert len(first.feature_names) == len(first.display_names) == first.n_features
        assert first.explainer_class
        assert np.isfinite(first.base_value)
        assert _top_names(first, "up") == _top_names(second, "up")
        assert _top_names(first, "down") == _top_names(second, "down")
        # The real winner genuinely splits features into both directions.
        assert len(_top_names(first, "up")) == 3
        assert len(_top_names(first, "down")) == 3


# ---------------------------------------------------------------------------
# 2. Independent pin: direct TreeExplainer + display-name derivation
# ---------------------------------------------------------------------------


def test_attribution_matches_direct_tree_explainer(real_artifact, held_out) -> None:
    """Values equal a direct shap.TreeExplainer row; names pin to the transform."""
    import shap

    row = held_out.iloc[[1]]
    attribution = attribute_patient(real_artifact, row)

    raw = real_artifact.raw_estimator
    transformed = np.asarray(raw[:-1].transform(row))
    explanation = shap.TreeExplainer(raw.named_steps["clf"])(transformed)
    expected = np.asarray(explanation.values, dtype=float)
    if expected.ndim == 3:  # (n, f, classes) -> positive class
        expected = expected[:, :, 1]
    assert expected.shape == (1, attribution.n_features)
    assert np.allclose(attribution.values, expected[0], atol=1e-8)

    expected_names = tuple(
        display_feature_name(name)
        for name in raw.named_steps["preprocess"].get_feature_names_out()
    )
    assert attribution.display_names == expected_names


# ---------------------------------------------------------------------------
# 3. Additivity: attribution pins to the RAW pre-calibration score
# ---------------------------------------------------------------------------


def test_additivity_reproduces_raw_score(real_artifact, held_out) -> None:
    """base_value + sum(values) ~= raw P(y=1|x) within 1e-6."""
    for position in range(N_HELD_OUT_PATIENTS):
        row = held_out.iloc[[position]]
        attribution = attribute_patient(real_artifact, row)
        raw_score = real_artifact.predict_raw_positive_proba(row)[0]
        assert abs(attribution.base_value + float(attribution.values.sum()) - raw_score) < 1e-6


# ---------------------------------------------------------------------------
# 4. Never-raise contract: named kinds for every failure family
# ---------------------------------------------------------------------------


def test_forced_compute_failure_is_never_raise(real_artifact, held_out, monkeypatch) -> None:
    """A crashing compute_shap_values maps to explanation-failed, both paths."""

    def _boom(*args, **kwargs):
        raise shap_export.ShapComputationError("synthetic compute failure")

    monkeypatch.setattr(shap_export, "compute_shap_values", _boom)
    row = held_out.iloc[[0]]

    with pytest.raises(ExplanationError) as excinfo:
        attribute_patient(real_artifact, row)
    assert excinfo.value.kind == "explanation-failed"
    assert excinfo.value.user_message

    outcome = explain_patient(real_artifact, row)
    assert outcome.attribution is None
    assert outcome.error is not None
    assert outcome.error.kind == "explanation-failed"
    assert outcome.degraded and not outcome.ok
    assert outcome.error.user_message


def test_missing_shap_import_is_shap_unavailable(real_artifact, held_out, monkeypatch) -> None:
    """A missing shap package maps to shap-unavailable, never an exception."""

    def _no_shap(*args, **kwargs):
        raise ImportError("shap is not installed (synthetic)")

    monkeypatch.setattr(shap_export, "_import_shap", _no_shap)
    row = held_out.iloc[[0]]

    outcome = explain_patient(real_artifact, row)
    assert outcome.attribution is None
    assert outcome.error is not None
    assert outcome.error.kind == "shap-unavailable"

    with pytest.raises(ExplanationError) as excinfo:
        attribute_patient(real_artifact, row)
    assert excinfo.value.kind == "shap-unavailable"


def test_plain_pipeline_payload_is_explanation_unsupported() -> None:
    """No calibration wrapper -> raw access refuses -> unsupported, named."""
    artifact, frame = _plain_pipeline_artifact()
    row = frame[list(FEATURE_COLUMNS)].iloc[[0]]

    with pytest.raises(ExplanationError) as excinfo:
        attribute_patient(artifact, row)
    assert excinfo.value.kind == "explanation-unsupported"

    outcome = explain_patient(artifact, row)
    assert outcome.attribution is None
    assert outcome.error is not None
    assert outcome.error.kind == "explanation-unsupported"


# ---------------------------------------------------------------------------
# 5. Schema guard: byte-identical serving contract on the explanation path
# ---------------------------------------------------------------------------


def test_reordered_and_empty_frames_are_rejected(real_artifact, held_out) -> None:
    """Reordered columns raise FeatureSchemaError; empty frames are refused."""
    row = held_out.iloc[[0]]

    columns = list(row.columns)
    columns[0], columns[2] = columns[2], columns[0]
    reordered = row[columns]

    with pytest.raises(FeatureSchemaError):
        attribute_patient(real_artifact, reordered)
    outcome = explain_patient(real_artifact, reordered)
    assert outcome.attribution is None
    assert outcome.error is not None

    empty = held_out.iloc[0:0]
    with pytest.raises(ExplanationError) as excinfo:
        attribute_patient(real_artifact, empty)
    assert excinfo.value.kind == "explanation-unsupported"
    empty_outcome = explain_patient(real_artifact, empty)
    assert empty_outcome.attribution is None
    assert empty_outcome.error is not None


def test_outcome_requires_exactly_one_branch() -> None:
    """The outcome type cannot encode 'both' or 'neither'."""
    with pytest.raises(ValueError):
        ExplanationOutcome(attribution=None, error=None)

    error = ExplanationError("explanation-failed", "synthetic")
    both = ExplanationOutcome(attribution=None, error=error)
    assert both.error is error and not both.ok and both.degraded
    assert error.kind in EXPLANATION_ERROR_KINDS
    with pytest.raises(ValueError):
        ExplanationError("not-a-real-kind", "synthetic")


# ---------------------------------------------------------------------------
# 6. Import layer: shap stays behind heart.interpret.shap_export
# ---------------------------------------------------------------------------


def test_app_sources_do_not_import_shap_directly() -> None:
    """The app never imports shap; only the sanctioned adapter may touch it."""
    for relative in ("app/lib/explain.py", "app/pages/2_Predict.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "import shap" not in source, f"{relative} imports shap directly"


# ---------------------------------------------------------------------------
# 7. Global cohort drivers (M007/S03): explain_global + session-cached glue
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cohort_cache_isolated():
    """Bracket every test with a clean session-cached cohort outcome.

    The ``st.cache_resource`` cache behind ``cohort_drivers`` is
    process-global and spans AppTest runs, so a cached success (or a
    cached sticky error) would leak into later tests and later suites.
    Clear on entry AND on exit — teardown runs even when an assertion
    fails mid-test.
    """
    clear_cohort_cache()
    yield
    clear_cohort_cache()


@pytest.fixture(scope="module")
def cohort_frame(real_artifact):
    """All 918 pinned rows in the artifact's exact feature order.

    The cohort sample source: larger than COHORT_MAX_ROWS, so the
    deterministic sampling inside compute_shap_values actually caps at
    COHORT_MAX_ROWS rows (the v1 held-out split is too small to pin that).
    """
    dataset = load_dataset(allow_download=False)
    frame = dataset.frame[list(real_artifact.metadata.feature_columns)]
    assert len(frame) > COHORT_MAX_ROWS, "cohort frame must exceed the cap"
    return frame


def test_cohort_drivers_ok_contract_and_determinism(real_artifact, cohort_frame) -> None:
    """Two explain_global runs -> identical frozen descending drivers."""
    first = explain_global(real_artifact, cohort_frame)
    second = explain_global(real_artifact, cohort_frame)

    assert first.ok and first.error is None
    assert second.ok and second.error is None
    drivers, again = first.drivers, second.drivers
    assert isinstance(drivers, CohortDrivers)

    assert np.array_equal(drivers.mean_abs, again.mean_abs)
    assert drivers.feature_names == again.feature_names
    assert drivers.display_names == again.display_names
    assert drivers.mean_signed == again.mean_signed
    assert drivers.top_feature == again.top_feature
    assert drivers.n_rows == again.n_rows == COHORT_MAX_ROWS

    mean_abs = np.asarray(drivers.mean_abs, dtype=float)
    assert len(drivers.display_names) == len(drivers.feature_names) == mean_abs.size
    assert mean_abs.size >= 1
    assert np.all(np.isfinite(mean_abs)) and np.all(mean_abs >= 0)
    assert np.all(np.diff(mean_abs) <= 0)  # descending: the render contract
    assert drivers.top_feature == drivers.display_names[int(np.argmax(mean_abs))]
    assert drivers.top_feature


def test_cohort_drivers_pin_to_the_sanctioned_shap_export_path(
    real_artifact, cohort_frame
) -> None:
    """explain_global's tuple equals a direct compute+summarize run (1e-10)."""
    outcome = explain_global(real_artifact, cohort_frame)
    assert outcome.drivers is not None

    computed = shap_export.compute_shap_values(
        _as_explainable_pipeline(real_artifact.raw_estimator),
        cohort_frame,
        max_samples=COHORT_MAX_ROWS,
        random_state=RANDOM_SEED,
    )
    summary = shap_export.summarize_shap(computed)
    order = sorted(
        range(len(summary.feature_names)),
        key=lambda index: (-summary.mean_abs_shap[index], index),
    )
    expected_abs = tuple(float(summary.mean_abs_shap[index]) for index in order)
    expected_names = tuple(summary.display_names[index] for index in order)

    assert np.allclose(outcome.drivers.mean_abs, expected_abs, rtol=0.0, atol=1e-10)
    assert outcome.drivers.display_names == expected_names
    assert outcome.drivers.n_rows == summary.n_samples == COHORT_MAX_ROWS


def test_explain_global_schema_guard_degrades(real_artifact, cohort_frame) -> None:
    """Reordered columns -> named error outcome, never an exception."""
    columns = list(cohort_frame.columns)
    columns[0], columns[2] = columns[2], columns[0]
    outcome = explain_global(real_artifact, cohort_frame[columns])

    assert outcome.degraded and not outcome.ok
    assert outcome.drivers is None
    assert outcome.error is not None
    # The schema guard (FeatureSchemaError) maps to this named kind.
    assert outcome.error.kind == "explanation-unsupported"
    assert outcome.error.user_message


def test_explain_global_forced_compute_failure_degrades(
    real_artifact, cohort_frame, monkeypatch
) -> None:
    """ShapComputationError and any escaping exception -> explanation-failed."""

    def _boom(*args, **kwargs):
        raise shap_export.ShapComputationError("synthetic cohort compute failure")

    monkeypatch.setattr(shap_export, "compute_shap_values", _boom)
    outcome = explain_global(real_artifact, cohort_frame)
    assert outcome.drivers is None
    assert outcome.error is not None
    assert outcome.error.kind == "explanation-failed"
    assert outcome.error.user_message

    def _explode(*args, **kwargs):
        raise RuntimeError("synthetic unexpected cohort explosion")

    monkeypatch.setattr(shap_export, "compute_shap_values", _explode)
    outcome = explain_global(real_artifact, cohort_frame)
    assert outcome.drivers is None
    assert outcome.error is not None
    assert outcome.error.kind == "explanation-failed"


def test_explain_global_missing_shap_degrades(
    real_artifact, cohort_frame, monkeypatch
) -> None:
    """A failing shap import -> shap-unavailable, never an exception."""

    def _no_shap(*args, **kwargs):
        raise ImportError("shap is not installed (synthetic)")

    monkeypatch.setattr(shap_export, "_import_shap", _no_shap)
    outcome = explain_global(real_artifact, cohort_frame)
    assert outcome.drivers is None
    assert outcome.error is not None
    assert outcome.error.kind == "shap-unavailable"
    assert outcome.error.user_message


def test_cohort_drivers_session_glue_ok() -> None:
    """The session glue loads artifact + dataset and computes once."""
    outcome = cohort_drivers()
    assert outcome.error is None, outcome.error
    assert isinstance(outcome.drivers, CohortDrivers)
    assert outcome.drivers.n_rows == COHORT_MAX_ROWS
    assert outcome.drivers.top_feature
    # Second call is a cache hit: the SAME outcome object, not a recompute.
    assert cohort_drivers() is outcome


def test_cohort_drivers_dataset_failure_is_named(monkeypatch) -> None:
    """A not-ok pinned-dataset load degrades to dataset-unavailable."""
    broken = ExploreData(None, "data-missing", "synthetic dataset failure (injected).")
    monkeypatch.setattr(explain_lib, "load_explore_data", lambda: broken)

    outcome = cohort_drivers()
    assert outcome.drivers is None
    assert outcome.error is not None
    assert outcome.error.kind == "dataset-unavailable"
    assert "synthetic dataset failure" in outcome.error.user_message


def test_cohort_outcome_requires_exactly_one_branch() -> None:
    """The cohort outcome type cannot encode 'both' or 'neither'."""
    with pytest.raises(ValueError):
        CohortOutcome(drivers=None, error=None)

    error = ExplanationError("dataset-unavailable", "synthetic")
    degraded = CohortOutcome(drivers=None, error=error)
    assert degraded.error is error and not degraded.ok and degraded.degraded
    assert error.kind in EXPLANATION_ERROR_KINDS


def test_cohort_drivers_validates_shape_order_and_top_feature() -> None:
    """The frozen driver bundle rejects malformed input at construction."""
    valid = dict(
        feature_names=("age", "oldpeak"),
        display_names=("Age", "Oldpeak"),
        mean_abs=(0.3, 0.1),
        mean_signed=(0.2, -0.05),
        n_rows=200,
        top_feature="Age",
    )
    drivers = CohortDrivers(**valid)
    assert drivers.top_feature == "Age" and drivers.n_rows == 200

    with pytest.raises(ValueError):  # length mismatch
        CohortDrivers(**{**valid, "mean_abs": (0.3,)})
    with pytest.raises(ValueError):  # negative magnitude
        CohortDrivers(**{**valid, "mean_abs": (-0.3, 0.1)})
    with pytest.raises(ValueError):  # not sorted descending
        CohortDrivers(**{**valid, "mean_abs": (0.1, 0.3)})
    with pytest.raises(ValueError):  # top_feature != argmax mean_abs
        CohortDrivers(**{**valid, "top_feature": "Oldpeak"})
    with pytest.raises(ValueError):  # no rows explained
        CohortDrivers(**{**valid, "n_rows": 0})

    with pytest.raises(ValueError):  # both branches set
        CohortOutcome(
            drivers=drivers,
            error=ExplanationError("explanation-failed", "synthetic"),
        )
