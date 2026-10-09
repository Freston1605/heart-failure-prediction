"""Tests for the app's Layer 3 serving-boundary contract (S07/T01).

The contract pinned here:

1. **Successful load** — the cached loader returns a validated
   :class:`ServingArtifact` for a well-formed artifact (built inline with
   ``save_artifact`` and a module-scope stub wrapper), and the same
   result is cached on a second call.
2. **Missing artifact** — a nonexistent path yields
   ``AppError(kind="artifact-missing")`` with a user-facing message, and
   the loader *does not raise*.
3. **Corrupt artifact** — unpicklable bytes and a pickle whose payload is
   not the declared flat dict both yield ``artifact-corrupt``.
4. **Schema-incompatible artifact** — a payload with a broken metadata
   keyset, and one whose embedded feature order mismatches the declared
   dataset schema, each yield ``artifact-schema``; an unsupported
   ``format_version`` yields ``artifact-incompatible``.
5. **Total mapping** — an unexpected load-time exception is bucketed to
   ``artifact-unknown`` rather than escaping as a stack trace.

Fixtures are synthetic and inline (module-scope classes so pickle round
trips); no gitignored paths are touched. The real artifact is exercised in
the S07/T04 end-to-end test.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.lib.artifact_loader import (
    DEFAULT_ARTIFACT_PATH,
    LoadedArtifact,
    load_winning_artifact,
    project_root,
)
from app.lib.errors import APP_ERROR_KINDS, AppError, to_app_error
from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS
from heart.serving.artifact import ArtifactError, save_artifact


# ---------------------------------------------------------------------------
# Module-scope stub payload: pickle requires importable classes
# ---------------------------------------------------------------------------


class _ScaffoldStubModel:
    """Predict/predict_proba interface over a trivial logistic pipeline."""

    def __init__(self) -> None:
        pipeline = Pipeline(
            [
                ("prep", build_preprocessing_pipeline()),
                ("clf", LogisticRegression()),
            ]
        )
        frame = _synthetic_frame()
        target = np.array([0, 1] * (len(frame) // 2))
        pipeline.fit(frame, target)
        self._model = pipeline

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._model.predict(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(frame)


def _synthetic_frame(n: int = 20) -> pd.DataFrame:
    """A minimal, schema-valid feature frame (inline fixture)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Age": rng.integers(29, 78, n),
            "Sex": rng.choice(["M", "F"], n),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], n),
            "RestingBP": rng.integers(90, 180, n),
            "Cholesterol": rng.integers(120, 400, n),
            "FastingBS": rng.choice([0, 1], n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], n),
            "MaxHR": rng.integers(70, 202, n),
            "ExerciseAngina": rng.choice(["N", "Y"], n),
            "Oldpeak": np.round(rng.uniform(-0.5, 4.0, n), 1),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], n),
        }
    )


@pytest.fixture()
def stub_model() -> _ScaffoldStubModel:
    return _ScaffoldStubModel()


@pytest.fixture()
def valid_artifact_path(tmp_path: Path, stub_model: _ScaffoldStubModel) -> Path:
    """A minimal valid artifact serialized next to the loaded real one."""
    path = tmp_path / "winner-scaffold.pkl"
    save_artifact(
        stub_model,
        path,
        model_name="scaffold-stub",
        feature_columns=tuple(FEATURE_COLUMNS),
        threshold=0.5,
        threshold_objective="accuracy",
        calibration_method="none",
        calibration_applied=False,
        brier_before=0.2,
        brier_after=0.2,
        ship=True,
        flags=(),
    )
    return path


# ---------------------------------------------------------------------------
# 1. Successful load
# ---------------------------------------------------------------------------


def test_loader_returns_validated_artifact(valid_artifact_path: Path) -> None:
    result = load_winning_artifact(valid_artifact_path)
    assert isinstance(result, LoadedArtifact)
    assert result.ok
    assert result.error is None
    result.artifact.predict_positive_proba(_synthetic_frame(5))


def test_loader_result_caches_per_path(valid_artifact_path: Path) -> None:
    first = load_winning_artifact(valid_artifact_path)
    second = load_winning_artifact(valid_artifact_path)
    assert first.artifact is second.artifact  # same cached object


def test_default_path_points_at_project_root() -> None:
    assert (project_root() / DEFAULT_ARTIFACT_PATH).name == "heart-winner-v1.pkl"


# ---------------------------------------------------------------------------
# 2. Missing artifact
# ---------------------------------------------------------------------------


def test_missing_artifact_is_friendly_error(tmp_path: Path) -> None:
    result = load_winning_artifact(tmp_path / "absent.pkl")
    assert not result.ok
    assert result.artifact is None
    error = result.error
    assert error is not None and error.kind == "artifact-missing"
    assert "could not be found" in error.user_message  # friendly, not a traceback


def test_loader_never_raises_on_any_path(tmp_path: Path) -> None:
    # even a directory-as-path must stay within the contract
    result = load_winning_artifact(tmp_path)
    assert isinstance(result.error, AppError)


# ---------------------------------------------------------------------------
# 3. Corrupt artifact
# ---------------------------------------------------------------------------


def _write_pickle(path: Path, payload: object) -> Path:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_unpicklable_bytes_give_corrupt_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"this is definitely not a pickle stream")
    result = load_winning_artifact(path)
    assert result.error is not None and result.error.kind == "artifact-corrupt"


def test_non_dict_payload_gives_corrupt_error(tmp_path: Path) -> None:
    result = load_winning_artifact(_write_pickle(tmp_path / "list.pkl", [1, 2, 3]))
    assert result.error is not None and result.error.kind == "artifact-corrupt"


def test_incomplete_payload_gives_corrupt_error(tmp_path: Path) -> None:
    result = load_winning_artifact(
        _write_pickle(tmp_path / "partial.pkl", {"format_version": 1})
    )
    assert result.error is not None and result.error.kind == "artifact-corrupt"


# ---------------------------------------------------------------------------
# 4. Schema-incompatible artifact
# ---------------------------------------------------------------------------


def _corrupt_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "format_version": 1,
        "metadata": {"format_version": 1, "created_at": "2026-10-07T00:00:00+00:00"},
        "model": object(),
    }
    base.update(overrides)
    return base


def test_broken_metadata_gives_schema_error(tmp_path: Path) -> None:
    result = load_winning_artifact(
        _write_pickle(tmp_path / "schema.pkl", _corrupt_payload())
    )
    assert result.error is not None and result.error.kind == "artifact-schema"


def test_mismatched_feature_order_gives_schema_error(
    tmp_path: Path, valid_artifact_path: Path
) -> None:
    payload = pickle.loads(valid_artifact_path.read_bytes())
    metadata = dict(payload["metadata"])
    metadata["feature_columns"] = list(reversed(list(FEATURE_COLUMNS)))
    payload["metadata"] = metadata
    result = load_winning_artifact(
        _write_pickle(tmp_path / "reordered.pkl", payload)
    )
    assert result.error is not None and result.error.kind == "artifact-schema"


def test_unsupported_version_gives_incompatible_error(tmp_path: Path) -> None:
    path = tmp_path / "future.pkl"
    save_artifact(
        _ScaffoldStubModel(),
        path,
        model_name="future",
        feature_columns=tuple(FEATURE_COLUMNS),
        threshold=0.5,
        threshold_objective="accuracy",
        calibration_method="none",
        calibration_applied=False,
        brier_before=0.2,
        brier_after=0.2,
        ship=True,
        flags=(),
    )
    raw = path.read_bytes()
    # rewrite the declared format version without re-saving normally
    payload = pickle.loads(raw)
    payload["format_version"] = 999
    future_path = _write_pickle(tmp_path / "future-v999.pkl", payload)
    result = load_winning_artifact(future_path)
    assert result.error is not None and result.error.kind == "artifact-incompatible"


def test_payload_without_serving_interface_gives_corrupt_error(
    tmp_path: Path,
) -> None:
    payload = pickle.loads(_write_pickle(tmp_path / "noop.pkl", None).read_bytes())  # type: ignore[arg-type]
    payload = {"format_version": 1, "metadata": payload, "model": "not-a-model"}
    payload["metadata"] = {
        "format_version": 1,
        "created_at": "2026-10-07T00:00:00+00:00",
        "model_name": "stub",
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": 0.5,
        "threshold_objective": "accuracy",
        "calibration_method": "none",
        "calibration_applied": False,
        "brier_before": 0.2,
        "brier_after": 0.2,
        "ship": True,
        "flags": [],
    }
    result = load_winning_artifact(_write_pickle(tmp_path / "nointerface.pkl", payload))
    assert result.error is not None and result.error.kind == "artifact-corrupt"


# ---------------------------------------------------------------------------
# 5. Total failure mapping
# ---------------------------------------------------------------------------


def test_every_artifact_error_has_a_named_kind() -> None:
    assert set(APP_ERROR_KINDS) == {
        "artifact-missing",
        "artifact-corrupt",
        "artifact-schema",
        "artifact-incompatible",
        "artifact-unknown",
    }


def test_unexpected_exception_buckets_to_unknown() -> None:
    class _Boom(Exception):
        pass

    mapped = to_app_error(_Boom("kaboom"))
    assert mapped.kind == "artifact-unknown"
    assert "unexpected" in mapped.user_message.lower()


def test_mapping_from_real_artifact_errors() -> None:
    from heart.serving.artifact import (
        ArtifactDataError,
        ArtifactPayloadError,
        ArtifactSchemaError,
        ArtifactVersionError,
        FeatureSchemaError,
    )

    assert to_app_error(FileNotFoundError("gone")).kind == "artifact-missing"
    assert to_app_error(ArtifactDataError("could not be read")).kind == "artifact-corrupt"
    assert to_app_error(ArtifactSchemaError("bad metadata")).kind == "artifact-schema"
    assert to_app_error(FeatureSchemaError("reordered")).kind == "artifact-schema"
    assert to_app_error(ArtifactVersionError("999")).kind == "artifact-incompatible"
    assert to_app_error(ArtifactPayloadError("no interface")).kind == "artifact-corrupt"
    assert isinstance(to_app_error(ArtifactError("x")), AppError)
