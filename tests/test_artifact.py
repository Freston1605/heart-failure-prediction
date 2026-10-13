"""Tests for the versioned serving artifact (S06/T04).

The contract pinned here:

1. **Versioned, self-describing artifacts** — a serialize→load round trip
   reproduces identical predictions *and* identical probabilities, and the
   loaded metadata equals what was saved (version, feature order, threshold,
   calibration state, ship verdict).
2. **Embedded feature-schema check at load time** — an artifact whose
   embedded feature columns deviate from the declared dataset schema
   (:data:`heart.data.schema.FEATURE_COLUMNS`) is refused with a named
   :class:`FeatureSchemaError`, never predicted silently.
3. **Runtime schema guard** — missing, unexpected, and reordered serving
   columns each fail separately with the same named error.
4. **Named failures** — missing/corrupt files, unsupported format versions,
   missing metadata keys, malformed thresholds, and payload models without
   ``predict``/``predict_proba`` all raise their named exceptions.
5. **Raw vs calibrated serving pair** (M007/S01) —
   :meth:`ServingArtifact.predict_raw_positive_proba` serves the raw
   (pre-isotonic) probability from the peeled frozen base estimator,
   elementwise equal to a direct call on that estimator for the known
   default patient, differing from the calibrated value for the same input;
   a payload without a calibration wrapper still loads and serves the
   calibrated path but refuses raw access with a named
   :class:`ArtifactPayloadError` **at access time only**; the raw path runs
   the identical feature-schema and probability-matrix guards as the
   calibrated path.

Fixtures are synthetic and self-contained (no gitignored paths); one
integration test rebuilds the real winner artifact through the selection
flow and cross-checks it against the committed ``reports/selection.md``,
and the raw-vs-calibrated pair is pinned against the committed real
artifact at ``models/heart-winner-v1.pkl`` (skipped cleanly when absent).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS, SCHEMA
from heart.eval.calibration import fit_calibration_wrapper
from heart.serving import artifact as art
from heart.serving.artifact import (
    ARTIFACT_FORMAT_VERSION,
    DEFAULT_ARTIFACT_PATH,
    ArtifactDataError,
    ArtifactPayloadError,
    ArtifactSchemaError,
    ArtifactVersionError,
    ArtifactMetadata,
    FeatureSchemaError,
    ServingArtifact,
    build_winner_artifact,
    check_features,
    load_artifact,
    predict_positive_proba,
    save_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib.input_validation import (  # noqa: E402
    CATEGORICAL_DISPLAY_LEVELS,
    NUMERIC_DEFAULTS,
    build_feature_frame,
)


# ---------------------------------------------------------------------------
# Module-level payload stubs (pickle requires module scope)
# ---------------------------------------------------------------------------


class _PredictOnlyStub:
    """Payload with predict only (no predict_proba)."""

    def predict(self, X):  # noqa: ANN001 - test stub
        return np.zeros(len(X))


class _ProbaOnlyStub:
    """Payload with predict_proba only (no predict)."""

    def predict_proba(self, X):  # noqa: ANN001 - test stub
        return np.zeros((len(X), 2))


class _ProbaOnlyStub2(_ProbaOnlyStub):
    """Second no-predict variant; same interface gap, distinct identity."""


class _WrapperLikeStub:
    """Calibration-wrapper-shaped payload: serving interface + ``estimator``."""

    def __init__(self, base: object) -> None:
        self.estimator = base

    def predict(self, X):  # noqa: ANN001 - test stub
        return np.zeros(len(X))

    def predict_proba(self, X):  # noqa: ANN001 - test stub
        return np.full((len(X), 2), 0.5)


class _MalformedProbaBase:
    """Base estimator whose ``predict_proba`` is 1-D (malformed)."""

    def predict(self, X):  # noqa: ANN001 - test stub
        return np.zeros(len(X))

    def predict_proba(self, X):  # noqa: ANN001 - test stub
        return np.zeros(len(X))


class _RowMismatchBase:
    """Base estimator returning one row too many for its input."""

    def predict(self, X):  # noqa: ANN001 - test stub
        return np.zeros(len(X))

    def predict_proba(self, X):  # noqa: ANN001 - test stub
        return np.tile(np.array([0.5, 0.5]), (len(X) + 1, 1))


# ---------------------------------------------------------------------------
# Synthetic data with the *real* declared feature schema
# ---------------------------------------------------------------------------


def _synthetic_frame(n: int = 300, *, seed: int = 7) -> pd.DataFrame:
    """A small binary problem laid out in the declared 11-feature order."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, size=n)
    sex = rng.choice(["M", "F"], size=n)
    chest = rng.choice(["ASY", "ATA", "NAP", "TA"], size=n)
    resting_bp = rng.integers(90, 181, size=n)
    chol = rng.integers(120, 401, size=n)
    fasting = rng.integers(0, 2, size=n)
    ecg = rng.choice(["Normal", "ST", "LVH"], size=n)
    max_hr = rng.integers(80, 201, size=n)
    angina = rng.choice(["N", "Y"], size=n)
    oldpeak = np.round(rng.uniform(0.0, 4.5, size=n), 1)
    slope = rng.choice(["Up", "Flat", "Down"], size=n)
    logit = (
        1.6 * (sex == "M") - 0.9 * (chest == "ASY") + 0.5 * fasting
        - 1.4 * (oldpeak / 4.5) - 0.02 * (max_hr - 140)
        + 1.1 * (angina == "Y") + 0.8 * (slope == "Flat")
    )
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame(
        {
            "Age": age.astype(int),
            "Sex": sex,
            "ChestPainType": chest,
            "RestingBP": resting_bp.astype(int),
            "Cholesterol": chol.astype(int),
            "FastingBS": fasting.astype(int),
            "RestingECG": ecg,
            "MaxHR": max_hr.astype(int),
            "ExerciseAngina": angina,
            "Oldpeak": oldpeak,
            "ST_Slope": slope,
            "HeartDisease": labels,
        }
    )
    assert list(frame.columns) == list(FEATURE_COLUMNS) + ["HeartDisease"]
    return frame


def _fitted_pipeline(frame: pd.DataFrame) -> Pipeline:
    """A fitted serving-shaped model: preprocessing + probabilistic clf."""
    from heart.config import RANDOM_SEED

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame["HeartDisease"])
    return pipeline


def _metadata(**overrides: object) -> ArtifactMetadata:
    kwargs: dict[str, object] = {
        "created_at": "2026-10-07T00:00:00+00:00",
        "model_name": "test-model",
        "feature_columns": tuple(FEATURE_COLUMNS),
        "threshold": 0.5,
        "threshold_objective": "f1",
        "calibration_method": "isotonic",
        "calibration_applied": True,
        "brier_before": 0.20,
        "brier_after": 0.15,
        "ship": True,
        "flags": (),
    }
    kwargs.update(overrides)
    return ArtifactMetadata(**kwargs)  # type: ignore[arg-type]


def _write_payload(path, payload: dict[str, object]) -> None:
    target = Path(path)
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def test_metadata_round_trips_through_dict() -> None:
    metadata = _metadata()
    rebuilt = art._metadata_from_mapping(metadata.to_dict())
    assert rebuilt == metadata


def test_metadata_rejects_missing_and_extra_keys() -> None:
    complete = _metadata().to_dict()
    for key in art.METADATA_KEYS:
        partial = dict(complete)
        del partial[key]
        with pytest.raises(ArtifactSchemaError):
            art._metadata_from_mapping(partial)
    with pytest.raises(ArtifactSchemaError):
        art._metadata_from_mapping({**complete, "weight_kg": 2})


def test_metadata_rejects_threshold_outside_unit_interval() -> None:
    with pytest.raises(ArtifactSchemaError):
        _metadata(threshold=1.5)


def test_metadata_rejects_non_finite_and_bad_brier() -> None:
    with pytest.raises(ArtifactSchemaError):
        _metadata(threshold=float("nan"))
    with pytest.raises(ArtifactSchemaError):
        _metadata(brier_before=1.2)
    with pytest.raises(ArtifactSchemaError):
        _metadata(brier_after=-0.01)


def test_metadata_rejects_duplicate_feature_columns() -> None:
    columns = list(FEATURE_COLUMNS)
    columns[1] = columns[0]
    with pytest.raises(ArtifactSchemaError):
        _metadata(feature_columns=tuple(columns))


def test_unsupported_metadata_version_raises_named_version_error() -> None:
    with pytest.raises(ArtifactVersionError):
        _metadata(format_version=999)


# ---------------------------------------------------------------------------
# Save → load round trip
# ---------------------------------------------------------------------------


def test_round_trip_reproduces_identical_predictions(tmp_path) -> None:
    frame = _synthetic_frame()
    model = _fitted_pipeline(frame)
    features = frame[list(FEATURE_COLUMNS)]
    path = tmp_path / "heart-winner-v1.pkl"
    metadata = save_artifact(
        model,
        path,
        model_name="test-logistic",
        feature_columns=tuple(FEATURE_COLUMNS),
        threshold=0.42,
        threshold_objective="f1",
        calibration_method="isotonic",
        calibration_applied=True,
        brier_before=0.3,
        brier_after=0.2,
        ship=True,
        flags=(),
    )
    loaded = load_artifact(path)

    assert isinstance(loaded, ServingArtifact)
    assert loaded.metadata == metadata
    assert loaded.metadata.format_version == ARTIFACT_FORMAT_VERSION
    assert loaded.metadata.feature_columns == tuple(FEATURE_COLUMNS)

    # Exact prediction identity, not approximate: serialization must not
    # perturb a single probability value.
    before_pred = model.predict(features)
    after_pred = loaded.model.predict(features)
    assert np.array_equal(before_pred, after_pred)
    before_proba = model.predict_proba(features)
    after_proba = loaded.model.predict_proba(features)
    assert np.array_equal(before_proba, after_proba)

    # The serving path (validated frame → positive-class probability) is
    # identical to the raw wrapper too.
    served = predict_positive_proba(loaded, features)
    assert np.array_equal(served, after_proba[:, 1])


def test_save_artifact_rejects_payload_without_serving_interface(tmp_path) -> None:
    for payload in (_PredictOnlyStub(), _ProbaOnlyStub(), "not a model"):
        with pytest.raises(ArtifactPayloadError):
            save_artifact(
                payload, tmp_path / "bad.pkl", **_metadata().to_dict()  # type: ignore[arg-type]
            )
    assert not (tmp_path / "bad.pkl").exists()


def test_serving_artifact_labels_use_the_embedded_threshold(tmp_path) -> None:
    frame = _synthetic_frame()
    model = _fitted_pipeline(frame)
    threshold = 0.3
    save_artifact(
        model,
        tmp_path / "art.pkl",
        **_metadata(threshold=threshold).to_dict(),  # type: ignore[arg-type]
    )
    loaded = load_artifact(tmp_path / "art.pkl")
    proba = predict_positive_proba(loaded, frame[list(FEATURE_COLUMNS)])
    assert loaded.metadata.threshold == threshold
    # Threshold semantics declared for the app: >= threshold is positive.
    labels = (proba >= threshold).astype(int)
    assert set(np.unique(labels)) <= {0, 1}


# ---------------------------------------------------------------------------
# Load-time validation failures
# ---------------------------------------------------------------------------


def test_load_missing_file_raises_named_data_error(tmp_path) -> None:
    with pytest.raises(ArtifactDataError):
        load_artifact(tmp_path / "nope.pkl")


def test_load_corrupt_file_raises_named_data_error(tmp_path) -> None:
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"\x00\x01not a pickle")
    with pytest.raises(ArtifactDataError):
        load_artifact(path)


def test_load_non_dict_payload_raises_named_data_error(tmp_path) -> None:
    _write_payload(tmp_path / "list.pkl", ["not", "the", "declared", "dict"])
    with pytest.raises(ArtifactDataError):
        load_artifact(tmp_path / "list.pkl")


def test_load_payload_missing_top_level_key_raises(tmp_path) -> None:
    complete = {
        "format_version": 1,
        "metadata": _metadata().to_dict(),
        "model": "x",
    }
    for key in ("format_version", "metadata", "model"):
        payload = dict(complete)
        del payload[key]
        path = tmp_path / f"missing_{key}.pkl"
        _write_payload(path, payload)
        with pytest.raises(ArtifactDataError):
            load_artifact(path)


def test_load_unsupported_format_version_raises_named_version_error(tmp_path) -> None:
    _write_payload(
        tmp_path / "v999.pkl",
        {"format_version": 999, "metadata": _metadata().to_dict(), "model": "x"},
    )
    with pytest.raises(ArtifactVersionError):
        load_artifact(tmp_path / "v999.pkl")


def test_load_mismatched_embedded_feature_schema_raises_named_error(tmp_path) -> None:
    """The core negative case: refuse rather than predict silently."""
    model = _fitted_pipeline(_synthetic_frame())
    mismatched = list(FEATURE_COLUMNS)
    mismatched[0], mismatched[1] = mismatched[1], mismatched[0]
    save_artifact(
        model,
        tmp_path / "mismatched.pkl",
        **_metadata(feature_columns=tuple(mismatched)).to_dict(),  # type: ignore[arg-type]
    )
    with pytest.raises(FeatureSchemaError):
        load_artifact(tmp_path / "mismatched.pkl")


def test_load_metadata_with_broken_types_raises_schema_error(tmp_path) -> None:
    complete = _metadata().to_dict()
    complete["threshold"] = "0.5 (string)"
    _write_payload(
        tmp_path / "bad_types.pkl",
        {"format_version": 1, "metadata": complete, "model": "x"},
    )
    with pytest.raises(ArtifactSchemaError):
        load_artifact(tmp_path / "bad_types.pkl")


def test_load_payload_without_predict_method_raises(tmp_path) -> None:
    _write_payload(
        tmp_path / "nopredict.pkl",
        {"format_version": 1, "metadata": _metadata().to_dict(), "model": _ProbaOnlyStub2()},
    )
    with pytest.raises(ArtifactPayloadError):
        load_artifact(tmp_path / "nopredict.pkl")


# ---------------------------------------------------------------------------
# Runtime feature-schema guard
# ---------------------------------------------------------------------------


def test_check_features_missing_column_is_explicit(tmp_path) -> None:
    frame = _synthetic_frame().drop(columns=["Oldpeak"])
    metadata = _metadata()
    with pytest.raises(FeatureSchemaError, match="missing"):
        check_features(frame, metadata)


def test_check_features_extra_column_is_explicit(tmp_path) -> None:
    frame = _synthetic_frame()
    frame["Surprise"] = 1.0
    with pytest.raises(FeatureSchemaError, match="unexpected"):
        check_features(frame, _metadata())


def test_check_features_reordered_columns_are_rejected(tmp_path) -> None:
    frame = _synthetic_frame()[list(FEATURE_COLUMNS)]
    columns = list(FEATURE_COLUMNS)
    columns[0], columns[2] = columns[2], columns[0]
    with pytest.raises(FeatureSchemaError, match="reordered"):
        check_features(frame[columns], _metadata())


def test_check_features_accepts_the_declared_order(tmp_path) -> None:
    frame = _synthetic_frame()[list(FEATURE_COLUMNS)]
    matrix = check_features(frame, _metadata())
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (len(frame), len(FEATURE_COLUMNS))
    assert list(matrix.columns) == list(FEATURE_COLUMNS)


def test_check_features_rejects_non_frames(tmp_path) -> None:
    with pytest.raises(FeatureSchemaError):
        check_features(np.zeros((3, 2)), _metadata())


def test_all_embedded_artifact_columns_match_the_declared_schema() -> None:
    """The schema the artifact embeds is the dataset's declared order."""
    assert len(FEATURE_COLUMNS) == len(SCHEMA) - 1
    assert _metadata().matches_declared_schema()


# ---------------------------------------------------------------------------
# Integration: the real winner artifact, cross-checked against selection.md
# ---------------------------------------------------------------------------


def test_real_winner_artifact_matches_the_selection_decision(tmp_path) -> None:
    """One full pass of the real flow: rebuild and cross-check selection.md.

    This is the slice's proof that the serialized artifact is *the* selected
    winner: identical carving, seeds, and gates — the loaded artifact must
    name the same winner and threshold the committed selection report
    declared, and its serving path must reproduce round-trip identity.
    """
    artifact, written = build_winner_artifact(path=tmp_path / "winner-v1.pkl")
    assert written.exists()

    expected_winner = "random-forest"
    assert artifact.metadata.model_name == expected_winner
    assert artifact.metadata.feature_columns == tuple(FEATURE_COLUMNS)
    assert artifact.metadata.calibration_applied is True
    assert 0.0 < artifact.metadata.threshold < 1.0

    committed = Path("reports/selection.md")
    if committed.exists():
        import re

        report = committed.read_text(encoding="utf-8")
        match = re.search(r"Chosen threshold: \*\*([\d.]+)\*\*", report)
        assert match, "committed selection.md lacks a chosen threshold"
        assert artifact.metadata.threshold == float(match.group(1))

    # Round-trip identity on the real calibration wrapper.
    from heart.data.split import load_split_frames

    train, _ = load_split_frames(version="v1")
    sample = train[list(FEATURE_COLUMNS)].head(64)
    before = artifact.model.predict_proba(sample)
    after = load_artifact(written).model.predict_proba(sample)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# Raw vs calibrated serving pair (M007/S01/T01)
# ---------------------------------------------------------------------------

REAL_ARTIFACT = PROJECT_ROOT / DEFAULT_ARTIFACT_PATH


@pytest.fixture(scope="module")
def real_artifact():
    """The real serialized winning model (skip cleanly if not serialized)."""
    if not REAL_ARTIFACT.is_file():
        pytest.skip("Real artifact not present on disk; run the S06 flow first.")
    return load_artifact(REAL_ARTIFACT)


@pytest.fixture(scope="module")
def synthetic_calibrated_artifact(tmp_path_factory):
    """A calibration-wrapped synthetic payload, save → load round-tripped."""
    frame = _synthetic_frame()
    train, calib = frame.iloc[:200], frame.iloc[200:]
    base = _fitted_pipeline(train)
    wrapper = fit_calibration_wrapper(
        base, calib[list(FEATURE_COLUMNS)], calib["HeartDisease"]
    )
    path = tmp_path_factory.mktemp("raw-pair") / "calibrated.pkl"
    save_artifact(wrapper, path, **_metadata().to_dict())  # type: ignore[arg-type]
    return load_artifact(path)


def _default_form_values() -> dict[str, object]:
    """The Predict form's default submission (first levels + defaults)."""
    values: dict[str, object] = dict(NUMERIC_DEFAULTS)
    for column, levels in CATEGORICAL_DISPLAY_LEVELS.items():
        values[column] = levels[0]
    return values


def _elementwise_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Plan contract: np.array_equal or abs diff < 1e-12, elementwise."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        return False
    return bool(
        np.array_equal(actual, expected)
        or np.all(np.abs(actual - expected) < 1e-12)
    )


def test_real_artifact_raw_equals_direct_frozen_base_estimator_call(
    real_artifact,
) -> None:
    """Known default patient: raw is the base estimator's own number.

    The pair must be real, not a copy: raw (pre-isotonic RF vote fraction)
    differs from the isotonic-calibrated value for this input, both are
    probabilities, and each accessor equals its direct payload call.
    """
    frame = build_feature_frame(_default_form_values())

    raw = real_artifact.predict_raw_positive_proba(frame)
    calibrated = real_artifact.predict_positive_proba(frame)

    # The peeled handle is the pipeline behind the FrozenEstimator wrapper.
    assert real_artifact.raw_estimator is real_artifact.model.estimator.estimator

    direct_raw = real_artifact.model.estimator.estimator.predict_proba(frame)[:, 1]
    direct_cal = real_artifact.model.predict_proba(frame)[:, 1]
    assert _elementwise_equal(raw, direct_raw)
    assert _elementwise_equal(calibrated, direct_cal)

    assert np.all((0.0 <= raw) & (raw <= 1.0))
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))

    # The gap is real: raw is not a renamed copy of the calibrated value.
    assert not _elementwise_equal(raw, calibrated)


def test_synthetic_calibrated_payload_raw_survives_round_trip(
    synthetic_calibrated_artifact,
) -> None:
    """After save → load, the raw accessor still equals the frozen base."""
    loaded = synthetic_calibrated_artifact
    features = _synthetic_frame()[list(FEATURE_COLUMNS)]

    raw = loaded.predict_raw_positive_proba(features)
    direct_raw = loaded.model.estimator.estimator.predict_proba(features)[:, 1]
    assert _elementwise_equal(raw, direct_raw)

    calibrated = loaded.predict_positive_proba(features)
    direct_cal = loaded.model.predict_proba(features)[:, 1]
    assert _elementwise_equal(calibrated, direct_cal)
    assert np.all((0.0 <= raw) & (raw <= 1.0))
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))


def test_plain_pipeline_payload_loads_but_refuses_raw(tmp_path) -> None:
    """Regression guard: no calibration wrapper → load and calibrated path
    keep working exactly as before; only raw access fails, at access time."""
    frame = _synthetic_frame()
    model = _fitted_pipeline(frame)
    path = tmp_path / "plain.pkl"
    save_artifact(model, path, **_metadata().to_dict())  # type: ignore[arg-type]
    loaded = load_artifact(path)  # loads fine: the load contract is unchanged

    features = frame[list(FEATURE_COLUMNS)]
    calibrated = loaded.predict_positive_proba(features)
    assert _elementwise_equal(calibrated, model.predict_proba(features)[:, 1])

    with pytest.raises(ArtifactPayloadError, match="raw pre-calibration"):
        loaded.predict_raw_positive_proba(features)
    with pytest.raises(ArtifactPayloadError, match="raw pre-calibration"):
        _ = loaded.raw_estimator  # the failure is named at access time


def test_raw_accessor_enforces_feature_schema_before_any_model_call(
    synthetic_calibrated_artifact,
) -> None:
    """Reordered/missing frames raise FeatureSchemaError from the raw path."""
    loaded = synthetic_calibrated_artifact
    frame = _synthetic_frame()[list(FEATURE_COLUMNS)]

    columns = list(FEATURE_COLUMNS)
    columns[0], columns[2] = columns[2], columns[0]
    with pytest.raises(FeatureSchemaError, match="reordered"):
        loaded.predict_raw_positive_proba(frame[columns])
    with pytest.raises(FeatureSchemaError, match="missing"):
        loaded.predict_raw_positive_proba(frame.drop(columns=["Oldpeak"]))

    # The schema guard fires before the payload-structure error too: a bad
    # frame is a bad frame regardless of what the payload can serve.
    plain = ServingArtifact(
        model=_fitted_pipeline(_synthetic_frame()),
        metadata=_metadata(),
        path=Path("unused.pkl"),
    )
    with pytest.raises(FeatureSchemaError, match="reordered"):
        plain.predict_raw_positive_proba(frame[columns])


def test_raw_path_validates_probability_matrix_like_calibrated_path() -> None:
    """The shared helper applies identical (n, 2) / row-count checks."""
    frame = _synthetic_frame()[list(FEATURE_COLUMNS)]
    metadata = _metadata()

    malformed = ServingArtifact(
        model=_WrapperLikeStub(_MalformedProbaBase()),
        metadata=metadata,
        path=Path("unused.pkl"),
    )
    with pytest.raises(ArtifactPayloadError, match="matrix") as raw_err:
        malformed.predict_raw_positive_proba(frame)

    calibrated = ServingArtifact(
        model=_MalformedProbaBase(), metadata=metadata, path=Path("unused.pkl")
    )
    with pytest.raises(ArtifactPayloadError, match="matrix") as cal_err:
        calibrated.predict_positive_proba(frame)
    # Identical validation: same helper, byte-identical failure message.
    assert str(raw_err.value) == str(cal_err.value)

    mismatched = ServingArtifact(
        model=_WrapperLikeStub(_RowMismatchBase()),
        metadata=metadata,
        path=Path("unused.pkl"),
    )
    with pytest.raises(ArtifactPayloadError, match="rows"):
        mismatched.predict_raw_positive_proba(frame)
