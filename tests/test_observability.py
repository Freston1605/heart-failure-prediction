"""Unit tests for heart.observability structured logging + metrics.

Asserts on LogRecord attributes (source of truth) and on the JSON formatter's
output by calling it directly — stream capture is not involved.
"""

from __future__ import annotations

import json
import logging

import pytest


@pytest.fixture()
def fresh_logging(tmp_path):
    """Reset CONFIGURED state and restore handlers/levels afterwards."""
    from heart import observability as obs_module

    obs_module.CONFIGURED = False
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield obs_module
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    obs_module.CONFIGURED = False


@pytest.fixture()
def obs(fresh_logging):
    """Fresh module state + a record-consuming caplog."""
    return fresh_logging


def _by_event(caplog, event: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == event]


def _last_fields(caplog, event: str) -> dict:
    found = _by_event(caplog, event)
    assert found, f"no record with event={event!r}; have {[getattr(r, 'event', None) for r in caplog.records]}"
    return found[-1]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_json_formatter_produces_canonical_record(obs, caplog):
    obs.configure_logging(fmt="json")
    logger = logging.getLogger("test.obs")
    with caplog.at_level(logging.INFO, logger="test.obs"):
        obs.log_event(
            logger, "app.artifact_load", "error", level=logging.ERROR,
            message="artifact load failed",
            error_kind="artifact-corrupt", path="/models/heart-winner-v1.pkl",
        )
    record = _by_event(caplog, "app.artifact_load")[-1]
    assert record.status == "error"
    assert record.fields == {
        "error_kind": "artifact-corrupt", "path": "/models/heart-winner-v1.pkl"
    }
    formatter = logging.getLogger().handlers[-1].formatter
    assert isinstance(formatter, obs._JsonRecordFormatter)
    rendered = json.loads(formatter.format(record))
    assert rendered["service"] == "heart"
    assert rendered["event"] == "app.artifact_load"
    assert rendered["status"] == "error"
    assert rendered["error_kind"] == "artifact-corrupt"
    assert rendered["level"] == "error"
    assert rendered["logger"] == "app.artifact_load" or rendered["logger"] == "test.obs"
    assert len(rendered["ts"]) >= 19


def test_non_event_records_get_bare_schema_when_formatted(obs, caplog):
    obs.configure_logging(fmt="json")
    logger = logging.getLogger("test.obs.bare")
    with caplog.at_level(logging.INFO, logger="test.obs.bare"):
        logger.info("plain library record")
    record = caplog.records[-1]
    assert getattr(record, "event", None) in (None, "log")
    formatter = logging.getLogger().handlers[-1].formatter
    rendered = json.loads(formatter.format(record))
    assert rendered["event"] in (None, "log") or rendered["event"] == "log"
    assert rendered["message"] == "plain library record"


def test_text_format_drops_fields_inline(obs, caplog):
    obs.configure_logging(fmt="text")
    logger = logging.getLogger("test.obs.text")
    with caplog.at_level(logging.INFO, logger="test.obs.text"):
        obs.log_event(
            logger, "app.predict_submit", "ok", level=logging.INFO,
            message="prediction served", outcome="served",
        )
    formatter = logging.getLogger().handlers[-1].formatter
    assert isinstance(formatter, obs._TextFormatter)
    rendered = formatter.format(caplog.records[-1])
    assert "app.predict_submit" in rendered and "outcome=served" in rendered


def test_configure_logging_is_idempotent(obs):
    obs.configure_logging(fmt="json")
    obs.configure_logging(fmt="json")
    handlers = [
        h for h in logging.getLogger().handlers
        if getattr(h, "heart_record", False)
    ]
    assert len(handlers) == 1


def test_env_vars_set_defaults(obs, monkeypatch):
    monkeypatch.setenv("HEART_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("HEART_LOG_FORMAT", "json")
    root = logging.getLogger()
    before = root.level
    obs.configure_logging()
    assert root.level == logging.DEBUG
    handlers = [h for h in root.handlers if getattr(h, "heart_record", False)]
    assert isinstance(handlers[-1].formatter, obs._JsonRecordFormatter)
    # restore readable state for caplog
    root.setLevel(before)


# ---------------------------------------------------------------------------
# PII denylist: clinical columns may never be field keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["Age", "age", " HeartDisease ", "cholesterol", "restingECG"])
def test_denylisted_field_keys_are_refused(key: str, obs, caplog):
    logger = logging.getLogger("test.obs.pii")
    with pytest.raises(ValueError) as info:
        obs.log_event(
            logger, "app.predict_submit", "ok", level=logging.INFO,
            message="prediction", **{key: 42},
        )
    assert "clinical-data field key" in str(info.value)
    assert not _by_event(caplog, "app.predict_submit")


# ---------------------------------------------------------------------------
# timed_event
# ---------------------------------------------------------------------------


def test_timed_failure_record_carries_error_kind_and_reraises(obs, caplog):
    logger = logging.getLogger("test.obs.timed")
    with caplog.at_level(logging.INFO, logger="test.obs.timed"), pytest.raises(RuntimeError):
        with obs.timed_event(logger, "app.dataset_load"):
            raise RuntimeError("pandas choked")
    failed = _last_fields(caplog, "app.dataset_load.failed")
    assert failed.fields["error_kind"] == "RuntimeError"
    assert failed.fields["duration_ms"] >= 0
    assert failed.status == "error"


def test_timed_event_emits_completion_plus_metric(obs, caplog):
    logger = logging.getLogger("test.obs.metric")
    with caplog.at_level(logging.INFO, logger="test.obs.metric"):
        with obs.timed_event(
            logger, "app.artifact_load", metric_name="heart.serving.artifact_load_ms",
        ) as carry:
            carry["path"] = "models/heart-winner-v1.pkl"
    complete = _last_fields(caplog, "app.artifact_load.complete")
    assert complete.fields["path"] == "models/heart-winner-v1.pkl"
    assert complete.fields["duration_ms"] >= 0
    assert complete.event == "app.artifact_load.complete"
    metric_record = _by_event(caplog, "metric")[-1]
    assert metric_record.fields["metric"] == "heart.serving.artifact_load_ms"
    assert metric_record.fields["metric_kind"] == "histogram"


def test_record_metric_carries_labels(obs, caplog):
    logger = logging.getLogger("test.obs.count")
    with caplog.at_level(logging.INFO, logger="test.obs.count"):
        obs.record_metric(
            logger, "heart.serving.predict_count", 1,
            kind="counter", outcome="served",
        )
    record = _by_event(caplog, "metric")[-1]
    assert record.fields["metric"] == "heart.serving.predict_count"
    assert record.fields["metric_kind"] == "counter"
    assert record.fields["outcome"] == "served"
    assert record.fields["value"] == 1
