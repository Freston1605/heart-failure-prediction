"""Cached artifact loader bridging the serving contracts into the app.

This module is the single place the Streamlit app touches the serialized
winning model. It:

1. resolves the artifact path (default: ``models/heart-winner-v1.pkl`` at
   the project root, overridable for tests);
2. delegates validation to the Layer 2 loader
   :func:`heart.serving.artifact.load_artifact`, which raises the named
   :class:`ArtifactError` family;
3. translates *every* load failure — including unexpected exceptions such
   as import errors at unpickle time — into the Layer 3 user-facing
   :class:`app.lib.errors.AppError` contract;
4. caches the result through ``st.cache_resource`` so the pickle is
   deserialized once per app process, keyed on the resolved path.

The public entrypoint never raises: :func:`load_winning_artifact` returns a
:class:`LoadedArtifact` whose ``ok`` flag tells the page whether it may
serve predictions. Pages are therefore free of try/except around model
loading, and a corrupt artifact can only ever produce the friendly error
surface, not a stack-trace white screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from app.lib.errors import AppError, to_app_error
from heart.serving.artifact import ServingArtifact, load_artifact

__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "LoadedArtifact",
    "load_winning_artifact",
    "clear_artifact_cache",
    "project_root",
]


#: Where the winner-generation flow (S06) serializes the serving artifact.
DEFAULT_ARTIFACT_PATH: str = "models/heart-winner-v1.pkl"


def project_root() -> Path:
    """Project root, computed from this file's location (``app/lib/``)."""
    return Path(__file__).resolve().parents[2]


class LoadedArtifact:
    """Result of one artifact-load attempt; never raises.

    Attributes:
        ok: True when a validated serving artifact was loaded.
        artifact: The :class:`ServingArtifact` (only when ``ok`` is True).
        error: The :class:`AppError` describing the failure (only when
            ``ok`` is False).
        path: The artifact path that was attempted.
    """

    def __init__(
        self,
        path: Path,
        artifact: ServingArtifact | None,
        error: AppError | None,
    ) -> None:
        self.path = path
        self.artifact = artifact
        self.error = error
        self.ok = artifact is not None and error is None

    def unwrap(self) -> ServingArtifact:
        """Return the artifact, raising :class:`AppError` if the load failed."""
        if self.artifact is None:
            raise self.error or AppError(
                "artifact-unknown", "Artifact load failed without a named error."
            )
        return self.artifact


def _load_attempt(path: Path) -> LoadedArtifact:
    """One uncached load attempt; maps every exception to the app contract."""
    try:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        artifact = load_artifact(path)
    except Exception as exc:  # noqa: BLE001 - the boundary swallows everything
        return LoadedArtifact(path, None, to_app_error(exc))
    return LoadedArtifact(path, artifact, None)


@st.cache_resource(show_spinner="Loading the trained model artifact...")
def _cached_load(path: Path) -> LoadedArtifact:
    """Streamlit-cached load attempt: one deserialization per process."""
    return _load_attempt(path)


def load_winning_artifact(
    path: str | Path | None = None,
) -> LoadedArtifact:
    """Load the serialized winner, or the friendly error explaining why not.

    Never raises for load failures — pages call this and branch on
    ``result.ok``. Only a caller bug (e.g. a non-Path of a bad type) escapes.
    """
    resolved = Path(path) if path is not None else project_root() / DEFAULT_ARTIFACT_PATH
    return _cached_load(resolved)


def clear_artifact_cache() -> None:
    """Drop the cached load result (e.g. after re-exporting the artifact)."""
    _cached_load.clear()


def _reset_cached_entry(path: Path) -> None:  # pragma: no cover - test helper
    """Force the cached function to forget its result for ``path``."""
    _cached_load.clear()


# Public-surface guard: the loader's contract is non-empty by construction.
_PUBLIC: tuple[object, ...] = (LoadedArtifact, load_winning_artifact)
assert _PUBLIC
