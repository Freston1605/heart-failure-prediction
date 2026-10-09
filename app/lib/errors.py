"""User-facing error contract (Layer 3) at the serving boundary.

S06's artifact loader already raises *named* errors
(:class:`heart.serving.artifact.ArtifactError` subclasses) for every
artifact failure. Layer 3 is where those internal exceptions stop being
internal: nothing here ever re-raises a raw artifact exception, and the UI
layer (:mod:`app.Home`) never imports artifact internals directly — it
consumes :class:`AppError` only.

Each serving-boundary failure is mapped to a stable ``kind`` (useful for
tests and future diagnostics) plus a ``user_message`` written for a
non-author: what went wrong, in plain language, with a next action (retrain
via the documented flow, re-run the winner generation, or check the file).
The mapping is total over :class:`ArtifactError` and also catches unknown
load-time exceptions (import errors, incompatible numpy/sklearn versions at
unpickle time), because "some unexpected exception propagated" is exactly
the stack-trace white screen this contract forbids.
"""

from __future__ import annotations

from heart.serving.artifact import (
    ArtifactDataError,
    ArtifactError,
    ArtifactPayloadError,
    ArtifactSchemaError,
    ArtifactVersionError,
    FeatureSchemaError,
)

__all__ = [
    "AppError",
    "APP_ERROR_KINDS",
    "to_app_error",
]


#: Stable named kinds surfaced by the app's error contract.
APP_ERROR_KINDS: tuple[str, ...] = (
    "artifact-missing",
    "artifact-corrupt",
    "artifact-schema",
    "artifact-incompatible",
    "artifact-unknown",
)


class AppError(Exception):
    """A serving-boundary failure translated into user-facing language.

    Attributes mirror the Layer 2 contract so tests and diagnostics can
    assert on both the kind and the message without re-processing the
    original exception.
    """

    kind: str
    user_message: str

    def __init__(self, kind: str, user_message: str) -> None:
        self.kind = kind
        self.user_message = user_message
        super().__init__(user_message)

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"[{self.kind}] {self.user_message}"


_MISSING = "The trained model file could not be found."
_CORRUPT = (
    "The trained model file could not be read. It looks damaged or "
    "incomplete, so the app refuses to serve predictions from it."
)
_SCHEMA = (
    "The trained model file is structurally inconsistent with this "
    "version of the app (its metadata does not match the declared "
    "contract). Re-train and re-export the winner before serving it."
)
_INCOMPATIBLE = (
    "The trained model file was produced by a different (or incompatible) "
    "version of this project, or by an incompatible library version. "
    "Re-run the winner-generation flow to create a matching artifact."
)
_UNKNOWN = (
    "An unexpected problem occurred while loading the trained model. "
    "The app refuses to continue rather than serving unverified results."
)


def to_app_error(exc: Exception) -> AppError:
    """Translate a serving-boundary exception into an :class:`AppError`.

    The mapping is total: any ``ArtifactError`` subclass has a named kind,
    and any other exception is deliberately bucketed as
    ``artifact-unknown`` so an unpredicted failure can still never leak a
    stack trace into the UI.
    """
    if isinstance(exc, FileNotFoundError):
        return AppError(
            "artifact-missing",
            f"{_MISSING} {_next_action('retrain')}",
        )
    if isinstance(exc, ArtifactVersionError):
        return AppError("artifact-incompatible", _INCOMPATIBLE)
    if isinstance(exc, ArtifactSchemaError) or isinstance(exc, FeatureSchemaError):
        return AppError("artifact-schema", f"{_SCHEMA} {_next_action('retrain')}")
    if (
        isinstance(exc, ArtifactDataError)
        or isinstance(exc, ArtifactPayloadError)
        or isinstance(exc, FeatureSchemaError)
    ):
        return AppError("artifact-corrupt", f"{_CORRUPT} {_next_action('regen')}")
    return AppError("artifact-unknown", f"{_UNKNOWN} {_next_action('regen')}")


def _next_action(verb: str) -> str:
    if verb == "retrain":
        return (
            "Run the documented winner-generation flow (S06) to produce a "
            "fresh artifact at models/heart-winner-v1.pkl, then reload."
        )
    return (
        "Re-run the winner-generation flow to recreate "
        "models/heart-winner-v1.pkl, then reload the app."
    )
