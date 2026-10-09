"""Feature selection with documented rationale (S04/T03).

Why this module exists
----------------------
:mod:`heart.features.ablation` measures each engineered candidate's *marginal*
contribution but deliberately decides nothing. This module turns that evidence
into a single decision — **which engineered transforms earn a permanent
column** — and writes down the number behind every keep/drop so a reviewer can
audit the choice instead of trusting it.

The policy
----------
Selection is a pure function of an :class:`~heart.features.ablation.AblationResult`
and a :class:`SelectionConfig`:

1. Every transform in the T01 registry is looked up by its single-transform
   ablation mode (``add-<transform>``).
2. Its **marginal delta** is ``mode_metric - baseline_metric`` on the declared
   metric (ROC-AUC by default).
3. It is **kept** when the delta is at least ``min_delta`` (default ``0.0005``)
   and **dropped** otherwise. A transform whose mode is missing, failed, or
   lacks the metric is dropped with an explicit reason rather than silently
   vanished.
4. Optionally, ``max_features`` caps the kept set to the highest-delta
   transforms, tie-broken by registry order.

The threshold is a parsimony budget, not a significance test: the held-out
split is small enough that a delta of a few ten-thousandths of ROC-AUC is not
distinguishable from sampling noise, so the policy keeps only the transforms
that clear a disclosed bar and can be re-derived deterministically. The
``all-engineered`` reference run is reported alongside the kept set because a
union of individually-positive transforms can *underperform* a smaller set —
evidence the report makes explicit.

Reproducibility
---------------
Given the same ablation metrics, :func:`select_features` returns a selection
whose :meth:`FeatureSelection.fingerprint` is byte-for-byte stable, and the
chosen representation is available as an ordinary
:class:`~heart.features.ablation.FeatureMode` (``selected``) that downstream
slices can rebuild without re-deriving the policy. The selection is persisted
as a JSON ledger so S04/T04 can load the exact representation the ablated model
was scored on.

Observability
-------------
:func:`select_features` logs the baseline, the threshold, and the kept/dropped
counts at ``INFO``. :func:`render_features_report` renders ``reports/features.md``
with one row of ablation evidence per decision, and
:func:`write_selection_ledger` persists the machine-readable selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


from heart.config import PROJECT_ROOT, REPORTS_DIR
from heart.eval.contract import PRIMARY_METRIC
from heart.runtime import atomic_write_text, write_json_document
from heart.features.ablation import (
    AblationError,
    AblationResult,
    AblationRun,
    FeatureMode,
    default_variants,
    run_ablation,
)
from heart.features.engineering import (
    ENGINEERED_TRANSFORMS,
    TRANSFORM_NAMES,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SELECTION_METRIC",
    "DEFAULT_MIN_DELTA",
    "ADD_MODE_PREFIX",
    "SELECTED_MODE_NAME",
    "ALL_ENGINEERED_MODE_NAME",
    "DECISION_KEPT",
    "DECISION_DROPPED",
    "FEATURES_REPORT_FILENAME",
    "SELECTION_LEDGER_FILENAME",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_SELECTION_CONFIG",
    "FeatureSelectionError",
    "SelectionConfigError",
    "SelectionDataError",
    "SelectionLedgerError",
    "SelectionReportError",
    "SelectionConfig",
    "TransformEvidence",
    "FeatureSelection",
    "select_features",
    "render_features_report",
    "write_features_report",
    "write_selection_ledger",
    "read_selection",
    "read_ablation_ledger",
    "generate_feature_selection",
    "describe_selection",
    "build_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Metric the selection ranks engineered transforms by.
SELECTION_METRIC: str = PRIMARY_METRIC

#: Default marginal-delta bar a transform must clear to be kept.
DEFAULT_MIN_DELTA: float = 0.0005

#: Prefix of a single-transform ablation mode (``add-<transform>``).
ADD_MODE_PREFIX: str = "add-"

#: Name of the :class:`FeatureMode` describing the selected representation.
SELECTED_MODE_NAME: str = "selected"

#: Name of the all-engineered ablation mode used as the combination reference.
ALL_ENGINEERED_MODE_NAME: str = "all-engineered"

#: Evidence decision: the transform earns a permanent column.
DECISION_KEPT: str = "kept"

#: Evidence decision: the transform was dropped.
DECISION_DROPPED: str = "dropped"

#: File name of the human-readable features report.
FEATURES_REPORT_FILENAME: str = "features.md"

#: File name of the machine-readable selection ledger.
SELECTION_LEDGER_FILENAME: str = "feature_selection.json"

#: Default location of the features report.
DEFAULT_REPORT_PATH: Path = Path(REPORTS_DIR) / FEATURES_REPORT_FILENAME

#: Default location of the selection ledger.
DEFAULT_LEDGER_PATH: Path = Path(REPORTS_DIR) / SELECTION_LEDGER_FILENAME


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class FeatureSelectionError(Exception):
    """Base class for every feature-selection failure."""


class SelectionConfigError(FeatureSelectionError):
    """A :class:`SelectionConfig` value is invalid."""


class SelectionDataError(FeatureSelectionError):
    """The ablation evidence handed to the selector is missing or malformed."""


class SelectionLedgerError(FeatureSelectionError):
    """The selection ledger could not be serialised, written, or read."""


class SelectionReportError(FeatureSelectionError):
    """The features report could not be rendered or written."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionConfig:
    """The fixed, documented knobs that make a selection reproducible.

    Parameters
    ----------
    metric:
        Ablation metric the deltas are computed on. Defaults to the project
        primary metric (ROC-AUC).
    min_delta:
        Minimum marginal improvement a transform must show to be kept. This is
        a disclosed parsimony bar, not a significance threshold.
    max_features:
        Optional cap on how many engineered transforms may be kept; the
        highest-delta transforms win, ties broken by registry order.
    mode_prefix:
        Prefix used to find the single-transform ablation mode for each
        registry transform (``add-`` by default).
    """

    metric: str = SELECTION_METRIC
    min_delta: float = DEFAULT_MIN_DELTA
    max_features: int | None = None
    mode_prefix: str = ADD_MODE_PREFIX

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise SelectionConfigError(
                f"metric must be a non-empty string, got {self.metric!r}."
            )
        if isinstance(self.min_delta, bool) or not isinstance(
            self.min_delta, (int, float)
        ):
            raise SelectionConfigError(
                "min_delta must be a number, got "
                f"{type(self.min_delta).__name__}."
            )
        if not math.isfinite(float(self.min_delta)):
            raise SelectionConfigError(
                f"min_delta must be finite, got {self.min_delta!r}."
            )
        if self.max_features is not None:
            if isinstance(self.max_features, bool) or not isinstance(
                self.max_features, int
            ):
                raise SelectionConfigError(
                    "max_features must be None or an int, got "
                    f"{type(self.max_features).__name__}."
                )
            if self.max_features < 1:
                raise SelectionConfigError(
                    f"max_features must be at least 1, got {self.max_features}."
                )
        if not isinstance(self.mode_prefix, str) or not self.mode_prefix.strip():
            raise SelectionConfigError(
                f"mode_prefix must be a non-empty string, got {self.mode_prefix!r}."
            )
        object.__setattr__(self, "metric", self.metric.strip())
        object.__setattr__(self, "min_delta", float(self.min_delta))

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "min_delta": float(self.min_delta),
            "max_features": self.max_features,
            "mode_prefix": self.mode_prefix,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelectionConfig":
        return cls(
            metric=str(payload["metric"]),
            min_delta=float(payload["min_delta"]),  # type: ignore[arg-type]
            max_features=payload.get("max_features"),  # type: ignore[arg-type]
            mode_prefix=str(payload.get("mode_prefix", ADD_MODE_PREFIX)),
        )


#: The project's single documented selection configuration.
DEFAULT_SELECTION_CONFIG: SelectionConfig = SelectionConfig()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformEvidence:
    """One engineered transform's ablation evidence and the resulting decision.

    ``delta`` and ``mode_metric`` are ``None`` when no usable ablation run
    existed (missing mode, failed run, or metric absent); such a transform is
    always dropped, and ``reason`` says why rather than hiding the gap.
    """

    transform: str
    columns: tuple[str, ...]
    mode_name: str
    decision: str
    delta: float | None
    mode_metric: float | None
    threshold: float
    reason: str

    def __post_init__(self) -> None:
        if self.decision not in (DECISION_KEPT, DECISION_DROPPED):
            raise FeatureSelectionError(
                f"unknown decision {self.decision!r}; expected one of "
                f"{[DECISION_KEPT, DECISION_DROPPED]}."
            )

    @property
    def kept(self) -> bool:
        return self.decision == DECISION_KEPT

    @property
    def dropped(self) -> bool:
        return self.decision == DECISION_DROPPED

    def to_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform,
            "columns": list(self.columns),
            "mode_name": self.mode_name,
            "decision": self.decision,
            "delta": None if self.delta is None else float(self.delta),
            "mode_metric": None if self.mode_metric is None else float(self.mode_metric),
            "threshold": float(self.threshold),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TransformEvidence":
        delta = payload.get("delta")
        mode_metric = payload.get("mode_metric")
        return cls(
            transform=str(payload["transform"]),
            columns=tuple(str(item) for item in payload.get("columns", [])),  # type: ignore[arg-type]
            mode_name=str(payload["mode_name"]),
            decision=str(payload["decision"]),
            delta=None if delta is None else float(delta),  # type: ignore[arg-type]
            mode_metric=None if mode_metric is None else float(mode_metric),  # type: ignore[arg-type]
            threshold=float(payload["threshold"]),  # type: ignore[arg-type]
            reason=str(payload["reason"]),
        )


def _portable_path(path: Path | None) -> str | None:
    """Render a path relative to the project root when it lives under it.

    The selection ledger is a committed artifact, so it must not embed an
    absolute worktree path that differs between checkouts.
    """
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return str(path)


@dataclass(frozen=True)
class FeatureSelection:
    """The selected engineered-feature representation and its justification.

    ``selected_mode`` is the ready-to-use
    :class:`~heart.features.ablation.FeatureMode` (named ``selected``) holding
    exactly ``kept``; it can be handed to
    :func:`heart.features.ablation.build_ablation_pipeline` or
    :func:`heart.features.engineering.engineer_features` unchanged.
    """

    config: SelectionConfig
    baseline_mode_name: str
    baseline_metric: float
    all_engineered_mode_name: str | None
    all_engineered_metric: float | None
    all_engineered_delta: float | None
    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    evidence: tuple[TransformEvidence, ...]
    selected_mode: FeatureMode
    generated_at: str
    report_path: Path | None = None
    ledger_path: Path | None = None

    @property
    def metric(self) -> str:
        return self.config.metric

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    @property
    def kept_columns(self) -> tuple[str, ...]:
        return tuple(
            column
            for transform in self.kept
            for column in ENGINEERED_TRANSFORMS[transform].produces
        )

    @property
    def evidence_for(self) -> dict[str, TransformEvidence]:
        return {item.transform: item for item in self.evidence}

    def fingerprint(self) -> dict[str, object]:
        """A stable, regeneration-time-independent digest of the decision.

        Excludes ``generated_at`` and file paths, so two selections derived
        from identical ablation metrics share a fingerprint.
        """
        return {
            "metric": self.config.metric,
            "min_delta": float(self.config.min_delta),
            "max_features": self.config.max_features,
            "mode_prefix": self.config.mode_prefix,
            "baseline_mode_name": self.baseline_mode_name,
            "baseline_metric": float(self.baseline_metric),
            "all_engineered_metric": (
                None
                if self.all_engineered_metric is None
                else float(self.all_engineered_metric)
            ),
            "all_engineered_delta": (
                None
                if self.all_engineered_delta is None
                else float(self.all_engineered_delta)
            ),
            "kept": list(self.kept),
            "dropped": list(self.dropped),
            "evidence": [
                {
                    "transform": item.transform,
                    "decision": item.decision,
                    "delta": None if item.delta is None else float(item.delta),
                }
                for item in self.evidence
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "baseline_mode_name": self.baseline_mode_name,
            "baseline_metric": float(self.baseline_metric),
            "all_engineered_mode_name": self.all_engineered_mode_name,
            "all_engineered_metric": (
                None
                if self.all_engineered_metric is None
                else float(self.all_engineered_metric)
            ),
            "all_engineered_delta": (
                None
                if self.all_engineered_delta is None
                else float(self.all_engineered_delta)
            ),
            "kept": list(self.kept),
            "dropped": list(self.dropped),
            "kept_columns": list(self.kept_columns),
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "evidence": [item.to_dict() for item in self.evidence],
            "selected_mode": self.selected_mode.to_dict(),
            "fingerprint": self.fingerprint(),
            "generated_at": self.generated_at,
            "report_path": _portable_path(self.report_path),
            "ledger_path": _portable_path(self.ledger_path),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureSelection":
        config = SelectionConfig.from_dict(payload["config"])  # type: ignore[arg-type]
        mode_payload = payload["selected_mode"]  # type: ignore[assignment]
        selected_mode = FeatureMode(
            name=str(mode_payload["name"]),  # type: ignore[index]
            include=tuple(mode_payload.get("include", [])),  # type: ignore[union-attr,arg-type]
            exclude=tuple(mode_payload.get("exclude", [])),  # type: ignore[union-attr,arg-type]
            description=str(mode_payload.get("description", "")),  # type: ignore[union-attr]
        )
        report_path = payload.get("report_path")
        ledger_path = payload.get("ledger_path")
        return cls(
            config=config,
            baseline_mode_name=str(payload["baseline_mode_name"]),
            baseline_metric=float(payload["baseline_metric"]),  # type: ignore[arg-type]
            all_engineered_mode_name=(
                None
                if payload.get("all_engineered_mode_name") is None
                else str(payload["all_engineered_mode_name"])
            ),
            all_engineered_metric=(
                None
                if payload.get("all_engineered_metric") is None
                else float(payload["all_engineered_metric"])  # type: ignore[arg-type]
            ),
            all_engineered_delta=(
                None
                if payload.get("all_engineered_delta") is None
                else float(payload["all_engineered_delta"])  # type: ignore[arg-type]
            ),
            kept=tuple(str(item) for item in payload.get("kept", [])),  # type: ignore[arg-type]
            dropped=tuple(str(item) for item in payload.get("dropped", [])),  # type: ignore[arg-type]
            evidence=tuple(
                TransformEvidence.from_dict(item)  # type: ignore[arg-type]
                for item in payload.get("evidence", [])  # type: ignore[union-attr]
            ),
            selected_mode=selected_mode,
            generated_at=str(payload["generated_at"]),
            report_path=None if report_path is None else Path(str(report_path)),
            ledger_path=None if ledger_path is None else Path(str(ledger_path)),
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _require_result(ablation_result: object) -> AblationResult:
    if not isinstance(ablation_result, AblationResult):
        raise SelectionDataError(
            "ablation_result must be an AblationResult, got "
            f"{type(ablation_result).__name__}."
        )
    return ablation_result


def _mode_metric(run: AblationRun, metric: str) -> float | None:
    if not run.succeeded or run.metrics is None:
        return None
    if metric not in run.metrics:
        return None
    return float(run.metrics[metric])


def _evaluate_transform(
    transform_name: str,
    *,
    result: AblationResult,
    baseline_metric: float,
    config: SelectionConfig,
) -> TransformEvidence:
    transform = ENGINEERED_TRANSFORMS[transform_name]
    mode_name = f"{config.mode_prefix}{transform_name}"
    run = result.run_for(mode_name)

    if run is None:
        return TransformEvidence(
            transform=transform_name,
            columns=transform.produces,
            mode_name=mode_name,
            decision=DECISION_DROPPED,
            delta=None,
            mode_metric=None,
            threshold=config.min_delta,
            reason=(
                f"no ablation evidence: mode {mode_name!r} is not present in "
                "the ablation result; re-run the ablation harness before "
                "selecting."
            ),
        )
    if not run.succeeded:
        return TransformEvidence(
            transform=transform_name,
            columns=transform.produces,
            mode_name=mode_name,
            decision=DECISION_DROPPED,
            delta=None,
            mode_metric=None,
            threshold=config.min_delta,
            reason=(
                f"ablation mode {mode_name!r} failed "
                f"({run.error_type or run.status}: {run.error or 'no error recorded'})"
            ),
        )
    mode_metric = _mode_metric(run, config.metric)
    if mode_metric is None:
        return TransformEvidence(
            transform=transform_name,
            columns=transform.produces,
            mode_name=mode_name,
            decision=DECISION_DROPPED,
            delta=None,
            mode_metric=None,
            threshold=config.min_delta,
            reason=(
                f"ablation mode {mode_name!r} did not record metric "
                f"{config.metric!r}"
            ),
        )

    delta = mode_metric - baseline_metric
    if delta >= config.min_delta:
        decision = DECISION_KEPT
        reason = (
            f"marginal {config.metric} {delta:+.4f} clears the "
            f"{config.min_delta:.4f} bar"
        )
    else:
        decision = DECISION_DROPPED
        reason = (
            f"marginal {config.metric} {delta:+.4f} is below the "
            f"{config.min_delta:.4f} bar"
        )
    return TransformEvidence(
        transform=transform_name,
        columns=transform.produces,
        mode_name=mode_name,
        decision=decision,
        delta=delta,
        mode_metric=mode_metric,
        threshold=config.min_delta,
        reason=reason,
    )


def _apply_max_features(
    evidence: Sequence[TransformEvidence], config: SelectionConfig
) -> tuple[TransformEvidence, ...]:
    """Cap the kept set to ``max_features`` highest-delta transforms."""
    if config.max_features is None:
        return tuple(evidence)
    kept = [item for item in evidence if item.kept]
    if len(kept) <= config.max_features:
        return tuple(evidence)
    ranked = sorted(
        kept,
        key=lambda item: (
            -(item.delta if item.delta is not None else -math.inf),
            TRANSFORM_NAMES.index(item.transform),
        ),
    )
    winners = {item.transform for item in ranked[: config.max_features]}
    capped: list[TransformEvidence] = []
    for item in evidence:
        if item.kept and item.transform not in winners:
            capped.append(
                replace(
                    item,
                    decision=DECISION_DROPPED,
                    reason=(
                        f"capped by max_features={config.max_features}: ranked "
                        f"below the top {config.max_features} by {config.metric}"
                    ),
                )
            )
        else:
            capped.append(item)
    return tuple(capped)


def select_features(
    ablation_result: AblationResult,
    config: SelectionConfig | None = None,
    *,
    generated_at: str | None = None,
) -> FeatureSelection:
    """Select the engineered-feature representation from ablation evidence.

    Every transform in the T01 registry is looked up by its single-transform
    ablation mode and kept only if its marginal metric delta clears
    ``config.min_delta``. The baseline and all-engineered modes are reported
    for context; a missing baseline is a hard error, because without it no
    delta can be computed.

    Parameters
    ----------
    ablation_result:
        The result of :func:`heart.features.ablation.run_ablation`. Every mode
        must already be scored; this function never retrains.
    config:
        The selection policy; defaults to :data:`DEFAULT_SELECTION_CONFIG`.
    generated_at:
        Override for the recorded timestamp (tests).
    """
    result = _require_result(ablation_result)
    resolved = config or DEFAULT_SELECTION_CONFIG
    if not isinstance(resolved, SelectionConfig):
        raise SelectionConfigError(
            f"config must be a SelectionConfig, got {type(resolved).__name__}."
        )

    baseline = result.baseline_run
    if baseline is None or baseline.metrics is None or resolved.metric not in baseline.metrics:
        raise SelectionDataError(
            "The ablation result has no succeeded baseline mode carrying "
            f"{resolved.metric!r}; feature deltas cannot be computed. Run "
            "heart.features.ablation.run_ablation first."
        )
    baseline_metric = float(baseline.metrics[resolved.metric])

    evidence = _apply_max_features(
        tuple(
            _evaluate_transform(
                name,
                result=result,
                baseline_metric=baseline_metric,
                config=resolved,
            )
            for name in TRANSFORM_NAMES
        ),
        resolved,
    )

    kept = tuple(item.transform for item in evidence if item.kept)
    dropped = tuple(item.transform for item in evidence if item.dropped)

    all_run = result.run_for(ALL_ENGINEERED_MODE_NAME)
    all_metric = None if all_run is None else _mode_metric(all_run, resolved.metric)
    all_delta = None if all_metric is None else all_metric - baseline_metric

    selected_mode = FeatureMode.from_include(
        SELECTED_MODE_NAME,
        kept,
        description=(
            f"Selected engineered representation (S04/T03): {len(kept)} of "
            f"{len(TRANSFORM_NAMES)} transforms kept when marginal "
            f"{resolved.metric} >= {resolved.min_delta:.4f}."
        ),
    )

    selection = FeatureSelection(
        config=resolved,
        baseline_mode_name=baseline.mode.name,
        baseline_metric=baseline_metric,
        all_engineered_mode_name=None if all_run is None else all_run.mode.name,
        all_engineered_metric=all_metric,
        all_engineered_delta=all_delta,
        kept=kept,
        dropped=dropped,
        evidence=evidence,
        selected_mode=selected_mode,
        generated_at=generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    logger.info(
        "Feature selection: baseline %s=%.4f, bar=%.4f, kept %d/%d "
        "(%s), dropped %d",
        resolved.metric,
        baseline_metric,
        resolved.min_delta,
        selection.n_kept,
        len(TRANSFORM_NAMES),
        ", ".join(kept) if kept else "none",
        selection.n_dropped,
    )
    return selection


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def _combination_note(selection: FeatureSelection) -> list[str]:
    """Render the all-engineered combination caveat when the evidence warrants it."""
    lines: list[str] = []
    deltas = [
        item.delta
        for item in selection.evidence
        if item.delta is not None
    ]
    best_single = max(deltas) if deltas else None
    if best_single is not None and selection.all_engineered_delta is not None:
        if selection.all_engineered_delta < best_single:
            lines.append(
                "Combining every engineered transform (`all-engineered`, "
                f"Δ {_fmt_delta(selection.all_engineered_delta)}) did **not** beat "
                "the best single transform "
                f"(Δ {_fmt_delta(best_single)}). A union of individually positive "
                "transforms can therefore underperform a smaller set, which is "
                "why the selection uses a disclosed threshold instead of keeping "
                "every positive delta."
            )
        else:
            lines.append(
                "The all-engineered set "
                f"(Δ {_fmt_delta(selection.all_engineered_delta)}) matched or beat "
                "the best single transform "
                f"(Δ {_fmt_delta(best_single)}); the thresholded selection remains "
                "the parsimonious choice."
            )
    return lines


def render_features_report(selection: FeatureSelection) -> str:
    """Render the feature-selection decision as the markdown report."""
    if not isinstance(selection, FeatureSelection):
        raise SelectionReportError(
            f"selection must be a FeatureSelection, got "
            f"{type(selection).__name__}."
        )
    config = selection.config
    kept_evidence = [item for item in selection.evidence if item.kept]
    dropped_evidence = [item for item in selection.evidence if item.dropped]

    lines: list[str] = []
    lines.append("# Feature Selection Report")
    lines.append("")
    lines.append(
        "_Generated by `heart.features.selection` (S04/T03) from the S04/T02 "
        "feature-ablation results. Every kept and dropped transform is justified "
        "by its marginal ablation delta on the same held-out split through the "
        "shared `heart.eval.contract.evaluate` path. Do not edit by hand — "
        "regenerate with `python -m heart.features.selection`._"
    )
    lines.append("")
    lines.append("## Decision policy")
    lines.append("")
    lines.append(f"- primary metric: `{config.metric}`")
    lines.append(
        f"- keep rule: marginal Δ ≥ `{config.min_delta:.4f}` over the baseline"
    )
    lines.append(
        "- max engineered transforms: "
        + ("unlimited" if config.max_features is None else f"`{config.max_features}`")
    )
    lines.append(
        f"- single-transform mode prefix: `{config.mode_prefix}`"
    )
    lines.append(
        "- this bar is a disclosed parsimony budget, not a significance test: "
        "on a ~184-row held-out split a delta of a few ten-thousandths of "
        "ROC-AUC is within sampling noise, so the policy keeps only transforms "
        "that clear a stated bar and re-derives identically."
    )
    lines.append("")
    lines.append("## Selected representation")
    lines.append("")
    lines.append(
        f"- engineered transforms kept: **{selection.n_kept} / "
        f"{len(TRANSFORM_NAMES)}**"
    )
    lines.append(
        f"- selection mode: `{selection.selected_mode.name}`"
    )
    lines.append(
        "- include: "
        + (", ".join(f"`{name}`" for name in selection.kept) if selection.kept else "_none_")
    )
    lines.append(
        "- engineered columns: "
        + (
            ", ".join(f"`{column}`" for column in selection.kept_columns)
            if selection.kept_columns
            else "_none_"
        )
    )
    lines.append(
        f"- total engineered columns: {len(selection.kept_columns)} "
        f"(raw schema columns are unchanged)"
    )
    lines.append("")
    lines.append("## Kept features")
    lines.append("")
    if kept_evidence:
        lines.append(
            "| transform | output columns | ablation mode | mode ROC-AUC | Δ ROC-AUC |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for item in kept_evidence:
            lines.append(
                f"| `{item.transform}` | {', '.join(item.columns)} | "
                f"`{item.mode_name}` | {_fmt(item.mode_metric)} | "
                f"{_fmt_delta(item.delta)} |"
            )
    else:
        lines.append(
            "_No engineered transform cleared the bar; the selected "
            "representation is the baseline feature set._"
        )
    lines.append("")
    lines.append("## Dropped features")
    lines.append("")
    if dropped_evidence:
        lines.append("| transform | ablation mode | Δ ROC-AUC | reason |")
        lines.append("| --- | --- | --- | --- |")
        for item in dropped_evidence:
            lines.append(
                f"| `{item.transform}` | `{item.mode_name}` | "
                f"{_fmt_delta(item.delta)} | {item.reason} |"
            )
    else:
        lines.append(
            "_Every engineered transform cleared the bar; nothing was dropped._"
        )
    lines.append("")
    lines.append("## Combination reference")
    lines.append("")
    lines.append(
        f"- baseline (`{selection.baseline_mode_name}`): ROC-AUC "
        f"{_fmt(selection.baseline_metric)}"
    )
    if selection.all_engineered_metric is not None:
        lines.append(
            f"- all-engineered (`{selection.all_engineered_mode_name}`): ROC-AUC "
            f"{_fmt(selection.all_engineered_metric)}, Δ "
            f"{_fmt_delta(selection.all_engineered_delta)}"
        )
    note = _combination_note(selection)
    if note:
        lines.append("")
        lines.extend(note)
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "- evidence source: `reports/ablation.md` (S04/T02) — run ids for each "
        "ablation mode are recorded there under `run_kind=ablation`."
    )
    lines.append(f"- selection config: `{config.to_dict()}`")
    lines.append(f"- generated at: {selection.generated_at}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m heart.features.ablation")
    lines.append("python -m heart.features.selection")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writing / reading
# ---------------------------------------------------------------------------


def _write_text_atomic(path: Path, content: str) -> None:
    """Thin wrapper delegating to heart.runtime.atomic_write_text."""
    atomic_write_text(path, content)


def write_features_report(selection: FeatureSelection, path: str | Path) -> Path:
    """Atomically write the features report to ``path``."""
    if not isinstance(selection, FeatureSelection):
        raise SelectionReportError(
            f"selection must be a FeatureSelection, got "
            f"{type(selection).__name__}."
        )
    destination = Path(path)
    try:
        _write_text_atomic(destination, render_features_report(selection))
    except OSError as exc:
        raise SelectionReportError(
            f"Could not write the features report to {destination}: {exc}"
        ) from exc
    logger.info(
        "Wrote features report to %s (%d kept, %d dropped)",
        destination,
        selection.n_kept,
        selection.n_dropped,
    )
    return destination


def write_selection_ledger(selection: FeatureSelection, path: str | Path) -> Path:
    """Atomically write the machine-readable selection ledger to ``path``."""
    if not isinstance(selection, FeatureSelection):
        raise SelectionLedgerError(
            f"selection must be a FeatureSelection, got "
            f"{type(selection).__name__}."
        )
    destination = write_json_document(
        path,
        selection.to_dict(),
        error_factory=SelectionLedgerError,
        label="selection ledger",
    )
    logger.info(
        "Wrote selection ledger to %s (%d kept transform(s))",
        destination,
        selection.n_kept,
    )
    return destination


def read_selection(path: str | Path) -> FeatureSelection:
    """Read a selection ledger written by :func:`write_selection_ledger`."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SelectionLedgerError(
            f"Could not read the selection ledger at {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SelectionLedgerError(
            f"Selection ledger at {source} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SelectionLedgerError(
            f"Selection ledger at {source} must contain a JSON object, got "
            f"{type(payload).__name__}."
        )
    try:
        return FeatureSelection.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionLedgerError(
            f"Selection ledger at {source} is malformed: {exc}"
        ) from exc


def read_ablation_ledger(path: str | Path) -> AblationResult:
    """Read an ablation ledger written by :func:`heart.features.ablation.write_ablation_ledger`."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SelectionDataError(
            f"Could not read the ablation ledger at {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SelectionDataError(
            f"Ablation ledger at {source} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict) or "runs" not in payload:
        raise SelectionDataError(
            f"Ablation ledger at {source} must be an object with a 'runs' key."
        )
    runs: list[AblationRun] = []
    try:
        for run in payload["runs"]:
            mode_payload = run["mode"]
            include = tuple(str(item) for item in mode_payload.get("include", []))
            exclude = tuple(str(item) for item in mode_payload.get("exclude", []))
            mode = FeatureMode(
                name=str(mode_payload["name"]),
                include=include,
                exclude=exclude,
                description=str(mode_payload.get("description", "")),
            )
            runs.append(
                AblationRun(
                    mode=mode,
                    status=str(run["status"]),
                    duration_seconds=float(run.get("duration_seconds", 0.0)),
                    n_engineered_columns=int(run.get("n_engineered_columns", 0)),
                    n_input_features=run.get("n_input_features"),
                    n_transformed_features=run.get("n_transformed_features"),
                    metrics=run.get("metric_dict"),
                    run_id=run.get("run_id"),
                    experiment_name=run.get("experiment_name"),
                    error=run.get("error"),
                    error_type=run.get("error_type"),
                    error_category=run.get("error_category"),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionDataError(
            f"Ablation ledger at {source} is malformed: {exc}"
        ) from exc
    return AblationResult(
        model_name=str(payload.get("model_name", "")),
        split_version=str(payload.get("split_version", "")),
        experiment_name=str(payload.get("experiment_name", "")),
        train_rows=int(payload.get("train_rows", 0)),
        test_rows=int(payload.get("test_rows", 0)),
        params=dict(payload.get("params", {})),
        runs=tuple(runs),
        generated_at=str(payload.get("generated_at", "")),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_fresh_ablation(
    *, split_version: str | None = None, model_name: str | None = None
) -> AblationResult:
    from heart.data.split import SPLIT_VERSION, load_split_frames

    resolved_split = split_version or SPLIT_VERSION
    train_frame, test_frame = load_split_frames(resolved_split)
    kwargs: dict[str, object] = {
        "split_version": resolved_split,
        "log_to_mlflow": False,
        "modes": default_variants(),
    }
    if model_name:
        kwargs["model_name"] = model_name
    return run_ablation(train_frame, test_frame, **kwargs)  # type: ignore[arg-type]


def generate_feature_selection(
    ablation_result: AblationResult | None = None,
    *,
    config: SelectionConfig | None = None,
    ablation_ledger_path: str | Path | None = None,
    split_version: str | None = None,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    ledger_path: str | Path | None = DEFAULT_LEDGER_PATH,
    write: bool = True,
) -> FeatureSelection:
    """Select features and (optionally) publish the report and ledger.

    Evidence precedence:

    1. an explicit ``ablation_result``;
    2. ``ablation_ledger_path`` (a T02 ledger JSON);
    3. a fresh, deterministic ablation run over the S01 split (no MLflow
       logging), so regeneration works from the repository alone.
    """
    if ablation_result is not None:
        result = _require_result(ablation_result)
    elif ablation_ledger_path is not None:
        result = read_ablation_ledger(ablation_ledger_path)
    else:
        result = _run_fresh_ablation(split_version=split_version)

    selection = select_features(result, config=config)

    if not write:
        return selection
    if report_path is not None:
        selection = replace(
            selection,
            report_path=write_features_report(selection, report_path),
        )
    if ledger_path is not None:
        selection = replace(
            selection,
            ledger_path=write_selection_ledger(selection, ledger_path),
        )
    return selection


def describe_selection(selection: FeatureSelection) -> str:
    """Render a short human-readable selection summary for the CLI."""
    if not isinstance(selection, FeatureSelection):
        raise FeatureSelectionError(
            f"selection must be a FeatureSelection, got "
            f"{type(selection).__name__}."
        )
    lines = [
        f"metric: {selection.metric} "
        f"(bar {selection.config.min_delta:.4f})",
        f"baseline {selection.baseline_mode_name}: "
        f"{_fmt(selection.baseline_metric)}",
        f"kept {selection.n_kept}/{len(TRANSFORM_NAMES)}: "
        + (", ".join(selection.kept) if selection.kept else "none"),
        f"dropped {selection.n_dropped}: "
        + (", ".join(selection.dropped) if selection.dropped else "none"),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.features.selection",
        description=(
            "Select the final engineered-feature representation from the "
            "S04/T02 ablation evidence and write reports/features.md."
        ),
    )
    parser.add_argument(
        "--ablation-ledger",
        default=None,
        help="Path to a T02 ablation ledger JSON to select from instead of "
        "re-running the ablation.",
    )
    parser.add_argument(
        "--split-version",
        default=None,
        help="S01 split version used when running a fresh ablation.",
    )
    parser.add_argument(
        "--metric",
        default=SELECTION_METRIC,
        help=f"Ablation metric to select on (default: {SELECTION_METRIC}).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help=f"Minimum marginal delta to keep a transform "
        f"(default: {DEFAULT_MIN_DELTA}).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional cap on kept engineered transforms.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Where to write the features report (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--ledger-path",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"Where to write the selection ledger (default: {DEFAULT_LEDGER_PATH}).",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip writing the report file."
    )
    parser.add_argument(
        "--no-ledger", action="store_true", help="Skip writing the ledger file."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from heart.data.split import SplitError

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        config = SelectionConfig(
            metric=args.metric,
            min_delta=args.min_delta,
            max_features=args.max_features,
        )
    except SelectionConfigError as exc:
        print(f"configuration error: {exc}")
        return 2

    try:
        selection = generate_feature_selection(
            config=config,
            ablation_ledger_path=args.ablation_ledger,
            split_version=args.split_version,
            report_path=None if args.no_report else args.report_path,
            ledger_path=None if args.no_ledger else args.ledger_path,
            write=True,
        )
    except (FeatureSelectionError, AblationError, SplitError) as exc:
        print(f"selection error: {exc}")
        return 1

    print(describe_selection(selection))
    if selection.report_path is not None:
        print(f"report: {selection.report_path}")
    if selection.ledger_path is not None:
        print(f"ledger: {selection.ledger_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
