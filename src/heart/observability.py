"""Structured events + metrics-as-log-lines for the heart portfolio.

Stdlib-only (no new runtime deps) so it works in the default
test environment (no mlflow/torch) and under the fully pinned manifest.

Conventions live in the DESIGN/CONVENTIONS doc (`.gsd/workflows/...`); the
enforced ones here:

* one JSON record per log line: ts/level/service/logger/event/status/message
  + event-specific fields,
* event names are dotted, area-prefixed (`app.artifact_load`),
* spans are `timed_event("verb.object")` durations in milliseconds,
* metrics are `record_metric` JSON lines (`heart.<area>.<what>_<unit>`),
* **clinical-data denylist:** field keys matching `heart.data.schema.FEATURE_COLUMNS`
  or `TARGET_COLUMN` (case-insensitive) raise `ValueError` at the call site —
  submitted patient-style values are health data and must never reach a log.

Env knobs: ``HEART_LOG_LEVEL`` (default INFO), ``HEART_LOG_FORMAT``
(``json`` default ``text``; set to ``json`` for deployment drains).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN

_SERVICE = "heart"
_RECORD_EXTRA = None
CONFIGURED = False

__all__ = [
    "configure_logging",
    "log_event",
    "record_metric",
    "timed_event",
]


def _denylisted_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in {column.lower() for column in (*FEATURE_COLUMNS, TARGET_COLUMN)}


def _validate_fields(fields: dict[str, object], message: str) -> None:
    offending = [key for key in fields if _denylisted_key(key)]
    if offending:
        raise ValueError(
            "observability: refusing to log clinical-data field key(s) "
            f"{offending}; patient-style values may never reach a log record "
            "(message must not contain them either)."
        )


def configure_logging(level: str | int | None = None, fmt: str | None = None) -> None:
    """Configure the root logger once per process (idempotent).

    ``level``/``fmt`` default to ``HEART_LOG_LEVEL`` / ``HEART_LOG_FORMAT``.
    ``fmt``: ``json`` (machine Record schema) or ``text`` (human one-liner).
    """
    global CONFIGURED, _RECORD_EXTRA
    env_level = os.environ.get("HEART_LOG_LEVEL", "INFO")
    env_fmt = os.environ.get("HEART_LOG_FORMAT", "text")
    chosen_level = level if level is not None else env_level
    chosen_fmt = fmt if fmt is not None else env_fmt
    root = logging.getLogger()
    if CONFIGURED:
        for handler in root.handlers:
            if getattr(handler, "heart_record", False):
                handler.setLevel(chosen_level if isinstance(chosen_level, int) else chosen_level.upper())
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = _JsonRecordFormatter() if chosen_fmt == "json" else _TextFormatter()
    handler.setFormatter(formatter)
    handler.set_name("heart.observability")
    handler.heart_record = True  # type: ignore[attr-defined]
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    resolved = (
        chosen_level
        if isinstance(chosen_level, int)
        else getattr(logging, chosen_level.upper(), logging.INFO)
    )
    root.setLevel(resolved)
    _RECORD_EXTRA = {"service": _SERVICE}
    CONFIGURED = True


class _JsonRecordFormatter(logging.Formatter):
    """Emit the single canonical event record for every heart.* record.

    Records routed through :func:`log_event`/ :func:`record_metric` carry keys
    `event`, `status`, `message` (plus fields). Unrelated records (stdlib noise,
    warnings) get a bare schema record with the message only — same shape, one
    record per line.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "service": _SERVICE,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "status": getattr(record, "status", "ok"),
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            _validate_fields(fields, str(payload["message"]))
            payload.update(fields)
        return json.dumps(payload, default=str, sort_keys=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", "log")
        fields = getattr(record, "fields", None) or {}
        extras = ", ".join(f"{key}={value}" for key, value in fields.items())
        message = record.getMessage()
        base = f"{record.name} {record.levelname} {event} {message}"
        return f"{base} ({extras})" if extras else base


def _emit(
    logger: logging.Logger,
    level: int,
    event: str,
    status: str,
    message: str,
    **fields: object,
) -> None:
    _validate_fields(fields, message)
    configure_logging()
    logger.log(
        level,
        message,
        extra={
            "event": event,
            "status": status,
            "fields": dict(fields),
        },
    )


def log_event(
    logger: logging.Logger,
    event: str,
    status: str = "ok",
    *,
    level: int = logging.INFO,
    message: str = "",
    **fields: object,
) -> None:
    """Emit one structured event record (idempotently configuring logging)."""
    _emit(logger, level, event, status, message, **fields)


def record_metric(
    logger: logging.Logger,
    name: str,
    value: float | int,
    *,
    kind: str = "gauge",
    **labels: str,
) -> None:
    """Emit one metric point as a JSON metric record (aggregated downstream)."""
    _emit(
        logger,
        logging.INFO,
        "metric",
        "ok",
        f"heartbeat metric {name}",
        metric=name,
        metric_kind=kind,
        value=value,
        **labels,
    )


@contextmanager
def timed_event(
    logger: logging.Logger,
    event: str,
    metric_name: str | None = None,
    *,
    level: int = logging.INFO,
    message: str = "completed",
    **fields: object,
) -> Iterator[dict[str, object]]:
    """Time a block, emitting `{event}.complete` + optional metric on exit.

    Yields a dict into which extra fields may be written by the caller before
    exit (they are merged into the completion record).
    """
    carry: dict[str, object] = {}
    start = time.perf_counter()
    try:
        yield carry
    except Exception as exc:  # noqa: BLE001 - observability boundary
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        _emit(
            logger,
            logging.ERROR,
            f"{event}.failed",
            "error",
            message if message != "completed" else f"{message} with error",
            duration_ms=duration_ms,
            error_kind=type(exc).__name__,
            **fields,
            **carry,
        )
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 3)
    _emit(
        logger,
        level,
        f"{event}.complete",
        "ok",
        message,
        duration_ms=duration_ms,
        **fields,
        **carry,
    )
    if metric_name is not None:
        record_metric(logger, metric_name, duration_ms, kind="histogram")
