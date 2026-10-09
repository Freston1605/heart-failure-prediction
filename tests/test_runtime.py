"""Unit tests for heart.runtime shared orchestration helpers."""

from __future__ import annotations

import json
import re
import threading

import numpy as np
import pytest


class _SentinelError(Exception):
    pass


# ---------------------------------------------------------------------------
# json_default
# ---------------------------------------------------------------------------


def test_json_default_coerces_numpy_scalars() -> None:
    from heart.runtime import json_default

    assert json_default(np.float64(1.5)) == 1.5
    assert json_default(np.int64(3)) == 3
    assert json_default(np.bool_(True)) is True
    assert isinstance(json_default(np.float32(2.0)), float)


def test_json_default_rejects_unknown_objects() -> None:
    from heart.runtime import json_default

    with pytest.raises(TypeError, match="not JSON serializable"):
        json_default(object())


def test_json_default_used_default_by_dumps() -> None:
    from heart.runtime import json_default

    payload = {"auc": np.float64(0.5), "n": np.int64(2)}
    encoded = json.dumps(payload, default=json_default)
    assert json.loads(encoded) == {"auc": 0.5, "n": 2}
    # numpy bool_ often serialises natively in recent numpy; force the hook
    with pytest.raises(TypeError):
        json.dumps({"weird": object()}, default=json_default)


# ---------------------------------------------------------------------------
# atomic_write_text / write_json_document
# ---------------------------------------------------------------------------


def test_atomic_write_text_creates_parents_and_no_part_residue(tmp_path) -> None:
    from heart.runtime import atomic_write_text

    destination = tmp_path / "nested" / "dir" / "out.txt"
    returned = atomic_write_text(destination, "hello\n")
    assert returned == destination
    assert destination.read_text(encoding="utf-8") == "hello\n"
    assert not list(destination.parent.glob(destination.name + ".part*"))


def test_atomic_write_is_not_interleaved(tmp_path) -> None:
    from heart.runtime import atomic_write_text

    destination = tmp_path / "shared.txt"
    errors: list[Exception] = []

    def writer(tag: str) -> None:
        for _ in range(50):
            try:
                atomic_write_text(destination, f"{tag}\n" * 100)
            except RuntimeError as exc:  # pragma: no cover - defensive
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100  # never a torn mix of writers
    assert len(set(lines)) == 1  # one writer's full payload per read


def test_write_json_document_pretty_sorted(tmp_path) -> None:
    from heart.runtime import write_json_document

    destination = tmp_path / "ledger.json"
    returned = write_json_document(
        destination,
        {"b": 1, "a": np.float64(0.5)},
        error_factory=_SentinelError,
        label="test ledger",
    )
    assert returned == destination
    assert destination.read_text(encoding="utf-8") == json.dumps(
        {"a": 0.5, "b": 1}, indent=2, sort_keys=True, default=None
    ) + "\n"


def test_write_json_document_serialise_error_message(tmp_path) -> None:
    from heart.runtime import write_json_document

    with pytest.raises(_SentinelError) as info:
        write_json_document(
            tmp_path / "out.json",
            {"bad": object()},
            error_factory=_SentinelError,
            label="battery ledger",
        )
    assert str(info.value).startswith("Could not serialise the battery ledger: ")


def test_write_json_document_write_error_message(tmp_path) -> None:
    from heart.runtime import write_json_document

    destination = tmp_path / "dir"
    destination.mkdir()

    with pytest.raises(_SentinelError) as info:
        write_json_document(
            destination,  # destination is an existing, occupied directory
            {"ok": True},
            error_factory=_SentinelError,
            label="tuning ledger",
        )
    assert str(info.value).startswith("Could not write the tuning ledger to ")


# ---------------------------------------------------------------------------
# error_message
# ---------------------------------------------------------------------------


def test_error_message_format() -> None:
    from heart.runtime import error_message

    try:
        raise ValueError("boom")
    except ValueError as exc:
        assert error_message(exc) == "ValueError: boom"


# ---------------------------------------------------------------------------
# timestamped_run_id
# ---------------------------------------------------------------------------


def test_timestamped_run_id_shape_and_prefixes() -> None:
    from heart.runtime import timestamped_run_id

    battery = timestamped_run_id("battery")
    sweep = timestamped_run_id("sweep")
    pattern = re.compile(
        r"^battery-\d{8}T\d{6}-[0-9a-f]{8}$"
    )
    assert pattern.match(battery)
    assert sweep.startswith("sweep-")
    assert battery != sweep


# ---------------------------------------------------------------------------
# merge_effective_params
# ---------------------------------------------------------------------------


def test_merge_effective_params() -> None:
    from heart.runtime import merge_effective_params

    assert merge_effective_params({"epochs": 20}, None) == {"epochs": 20}
    assert merge_effective_params({"epochs": 20}, {}) == {"epochs": 20}
    merged = merge_effective_params(
        {"epochs": 20, "hidden": (64,)}, {"epochs": 40, "lr": 0.001}
    )
    assert merged == {"epochs": 40, "hidden": (64,), "lr": 0.001}
    # fixed params dict is not mutated
    fixed = {"epochs": 20}
    merge_effective_params(fixed, {"epochs": 50})
    assert fixed == {"epochs": 20}


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------


class _Base(Exception):
    pass


class _Derived(_Base):
    pass


class _Sibling(Exception):
    pass


def test_classify_error_first_match_wins() -> None:
    from heart.runtime import classify_error

    error_map = ((_Derived, "derived"), (_Base, "base"))
    with pytest.raises(_Derived) as derived:
        raise _Derived()
    assert classify_error(derived.value, error_map, default="unexpected") == "derived"
    with pytest.raises(_Base) as base:
        raise _Base()
    assert classify_error(base.value, error_map, default="unexpected") == "base"
    with pytest.raises(_Sibling) as sibling:
        raise _Sibling()
    assert classify_error(sibling.value, error_map, default="unexpected") == "unexpected"


def test_classify_error_empty_map_falls_back() -> None:
    from heart.runtime import classify_error

    assert classify_error(ValueError("x"), (), default="unexpected") == "unexpected"
