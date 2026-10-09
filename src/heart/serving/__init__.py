"""Serving boundary for the heart-failure prediction app.

Serialization of the selected, calibrated winner lives in
:mod:`heart.serving.artifact`; it is the only way a model leaves the
training side of the codebase.
"""

from heart.serving.artifact import (
    ARTIFACT_FORMAT_VERSION,
    ArtifactDataError,
    ArtifactError,
    ArtifactPayloadError,
    ArtifactSchemaError,
    ArtifactVersionError,
    ArtifactMetadata,
    FeatureSchemaError,
    ServingArtifact,
    check_features,
    load_artifact,
    predict_positive_proba,
    save_artifact,
)

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ArtifactDataError",
    "ArtifactError",
    "ArtifactPayloadError",
    "ArtifactSchemaError",
    "ArtifactVersionError",
    "ArtifactMetadata",
    "FeatureSchemaError",
    "ServingArtifact",
    "check_features",
    "load_artifact",
    "predict_positive_proba",
    "save_artifact",
]
