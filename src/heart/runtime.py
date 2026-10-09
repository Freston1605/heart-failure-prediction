"""Shared runtime plumbing for the benchmark / tuning / sweep runners.

This module exists to remove duplication that previously lived, copy-pasted,
in ``heart.models.run_battery``, ``heart.models.run_mlp_sweep``,
``heart.tuning.runner``, and ``heart.models.train_torch``:

* the numpy-scalar-aware ``json.dumps`` default hook,
* the atomic (write-to-``.part`` then ``os.replace``) file-write dance,
* the "``TypeName: message``" exception formatting, used in fail-soft records,
* the UTC-stamped ``prefix-<stamp>-<hex>`` run ids,
* merging tuned params into an estimator spec's fixed params,
* ordered ``isinstance``-chain error classification.

Design constraints (load-bearing - do not "improve" them away):

* **Optional-dependency free.** The default test environment has no torch /
  mlflow / optuna installed; this module imports only the standard library
  (with numpy deferred to first use inside ``json_default``).
* **No upward imports.** It must not import from ``heart.tuning.runner`` or
  any runner module (those import this module - a cycle would result).
* **Zero behavior change.** Callers keep their own concrete exception types
  and log lines; this module never invents an error type of its own.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

__all__ = [
    "atomic_write_text",
    "classify_error",
    "error_message",
    "json_default",
    "merge_effective_params",
    "timestamped_run_id",
    "write_json_document",
]

_E = TypeVar("_E", bound=Exception)


def json_default(value: object) -> object:
    """Return a JSON-serialisable form of numpy scalars, else raise ``TypeError``.

    Behaviour is byte-compatible with the seven private ``_json_default`` hooks
    this replaces (plus two variant copies retained deliberately): only ``np.generic`` values are coerced (via ``.item()``);
    anything else raises, so silent type loss is impossible.
    """
    # Imported lazily so heart.runtime itself stays importable in
    # numpy-free contexts; only non-serialisable payloads pay the cost.
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Write ``content`` to ``path`` atomically and return the destination.

    Creates parent directories, writes to ``<name>.part``, then ``os.replace``
    onto the destination so readers never observe a partial file. Raises
    ``OSError`` (callers wrap into their own error types).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def write_json_document(
    path: str | Path,
    payload: object,
    *,
    error_factory: Callable[[str], _E],
    label: str,
    write_label: str | None = None,
) -> Path:
    """Atomically write ``payload`` as a pretty (indent 2, key-sorted) JSON doc.

    On serialisation failure raises ``error_factory(f"Could not serialise the
    {label}: {exc}")``; on file-system failure ``error_factory(f"Could not
    write the {write_label or label} to {destination}: {exc}")`` - the exact
    message shapes the original ledger writers produced. Logging stays at the
    call site so each writer keeps its own wording.
    """
    destination = Path(path)
    try:
        content = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    except TypeError as exc:
        raise error_factory(f"Could not serialise the {label}: {exc}") from exc
    try:
        atomic_write_text(destination, content + "\n")
    except OSError as exc:
        raise error_factory(
            f"Could not write the {write_label or label} to {destination}: {exc}"
        ) from exc
    return destination


def error_message(exc: BaseException) -> str:
    """Return the canonical fail-soft record form: ``TypeName: message``."""
    return f"{type(exc).__name__}: {exc}"


def timestamped_run_id(prefix: str) -> str:
    """Return ``prefix-<UTC YYYYmmddTHHMMSS>-<8 hex>`` for run bookkeeping."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def merge_effective_params(
    fixed_params: Mapping[str, object],
    best_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Merge tuned ``best_params`` over an estimator spec's fixed params."""
    merged = dict(fixed_params)
    if best_params:
        merged.update(best_params)
    return merged


def classify_error(
    exc: BaseException,
    error_map: tuple[tuple[type[BaseException], str], ...],
    *,
    default: str,
) -> str:
    """Classify ``exc`` by ordered ``isinstance`` match against ``error_map``.

    First match wins, so subclass-before-parent ordering is the caller's
    contract; unmatched exceptions fall back to ``default``. Previously each
    runner hand-rolled this chain with a module-specific extra case list.
    """
    for exc_type, category in error_map:
        if isinstance(exc, exc_type):
            return category
    return default
