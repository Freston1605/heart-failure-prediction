"""Generate the model leaderboard from recorded MLflow runs (S03/T04, S05/T04).

This module closes the evaluation loop for every benchmarked model, classical
and neural alike. :mod:`heart.models.run_battery` and
:mod:`heart.models.run_mlp_sweep` each log exactly one **final** run per model
(tagged ``run_kind=final`` and ``model_type=<registry key>``) through the
frozen tracking convention. This module reads those runs back out of MLflow and
renders ``reports/leaderboard.md`` — the ranked, full-metric comparison table.

Neural runs are not special-cased
---------------------------------
The generator has no notion of "classical" versus "neural": any finished run
that tagged ``run_kind=final`` and ``model_type`` with the complete metric
suite is a candidate. Neural runs additionally carry an optional ``device``
tag (``cuda:0`` or a logged ``cpu`` fallback); the leaderboard reads it
generically into a Device column, so "which compute device did this neural run
use?" is answerable from the leaderboard itself. A run without the tag simply
renders ``-``.

Selection annotations (S06/T05)
------------------------------
The selection flow (``heart.eval.selection``) tags the **winner's** MLflow
run with an ``s6.*`` annotation set — winner identity, the repeated-CV
distribution that named it, the paired significance verdict against the
baseline, the calibrated threshold and scores, the Brier movement, the
serving-weight check and any NO-SHIP flags. The leaderboard reads those tags
back generically, so the published table carries the statistical evidence
behind the winner from the tracking store alone: regenerating the report
re-derives the section from MLflow, never from a hand-maintained list. An
incomplete, malformed or ambiguous annotation set raises a named
:class:`S06LeaderboardDataError` / :class:`SelectionAnnotationsError`
rather than rendering a half-story.

It is never hand-written
------------------------
Every number in the report comes from an MLflow run. Regenerating is the only
supported edit path::

    python -m heart.reporting.leaderboard

Selection
---------
A candidate run is included when **all** of these hold:

* it tagged ``run_kind=<run_kind>`` (default ``final`` — trial runs are
  excluded by design);
* it carries a ``model_type`` tag (benchmarked models only);
* its MLflow status is ``FINISHED``;
* if a ``split_version`` filter is given, its ``split_version`` tag matches.

Each candidate must also carry the complete flattened metric suite produced by
:func:`heart.tracking.run.log_evaluation_run` (the scalar leaves of the
canonical metric dict). A run missing any required leaf is a *data error*, not
a silently omitted row: a leaderboard that quietly drops a model is worse than
no leaderboard.

When ``latest_only`` is true (default) a model that has been benchmarked more
than once contributes exactly one row — its most recent run, tie-broken by the
higher primary metric. ``latest_only=False`` lists every matching run, which is
useful when auditing repeated batteries.

Ordering and the reported suite
-------------------------------
Rows are ordered by ROC-AUC (the declared primary metric, R007) descending.
Each row reports the full scalar suite — accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, specificity, NPV, prevalence, Brier score, expected
calibration error, and sample count — and the structured parts of the suite
are expanded into their own tables: the confusion matrix (TN/FP/FN/TP and the
positive/negative counts) and the calibration summary (Brier, ECE, bins).

Observability
-------------
:func:`build_leaderboard` logs the experiment, how many runs were considered
versus selected, and the top-ranked model at ``INFO``. :func:`describe_leaderboard`
renders the provenance block, and every :class:`Leaderboard` is available as a
JSON-serialisable dict via :meth:`Leaderboard.to_dict`.

Failure modes are named: a missing experiment raises
:class:`LeaderboardExperimentNotFoundError`; a backend/search failure raises
:class:`LeaderboardDataError`; a run with an incomplete metric suite raises
:class:`LeaderboardDataError`; a report that cannot be written raises
:class:`LeaderboardReportError`; and ``strict=True`` turns an empty
leaderboard into :class:`NoBenchmarkedModelsError` rather than publishing a
blank report.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

try:  # pragma: no cover - exercised by the environment, not by logic
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "heart.reporting.leaderboard requires MLflow, which is not installed. "
        'Install the tracking extra with: pip install -e ".[ml]"'
    ) from exc

from heart.config import REPORTS_DIR
from heart.eval.contract import (
    CALIBRATION_KEY,
    CONFUSION_MATRIX_KEY,
    PRIMARY_METRIC,
    SCALAR_METRIC_KEYS,
)
from heart.eval.metrics import CONFUSION_MATRIX_KEYS
from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    resolve_tracking_uri,
)
from heart.tuning.runner import FINAL_RUN_KIND, MODEL_TYPE_TAG, RUN_KIND_TAG

logger = logging.getLogger(__name__)

__all__ = [
    "LEADERBOARD_FILENAME",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_RUN_KIND",
    "LEADERBOARD_TITLE",
    "REQUIRED_FLAT_KEYS",
    "RANKED_SCALAR_COLUMNS",
    "DEVICE_TAG",
    "NO_DEVICE_LABEL",
    "RUN_STATUS_FINISHED",
    "SEARCH_PAGE_SIZE",
    "MAX_RUNS_CONSIDERED",
    "SELECTION_TAG_PREFIX",
    "MAX_TAG_VALUE_LENGTH",
    "SelectionAnnotationsError",
    "S06LeaderboardDataError",
    "LeaderboardError",
    "LeaderboardConfigError",
    "LeaderboardDataError",
    "LeaderboardExperimentNotFoundError",
    "LeaderboardReportError",
    "NoBenchmarkedModelsError",
    "selection_tags",
    "SelectionAnnotations",
    "read_selection_annotations",
    "selection_annotation_tags",
    "annotate_winner_run",
    "LeaderboardConfig",
    "LeaderboardRow",
    "Leaderboard",
    "select_leaderboard_rows",
    "build_leaderboard",
    "render_leaderboard",
    "write_leaderboard",
    "generate_leaderboard",
    "describe_leaderboard",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: File name of the generated leaderboard report.
LEADERBOARD_FILENAME: str = "leaderboard.md"

#: Default location of the generated leaderboard report.
DEFAULT_REPORT_PATH: Path = Path(REPORTS_DIR) / LEADERBOARD_FILENAME

#: Only final (per-model) runs belong on the leaderboard; trials are excluded.
DEFAULT_RUN_KIND: str = FINAL_RUN_KIND

#: Document title used by the renderer.
LEADERBOARD_TITLE: str = "Heart-Failure Model Leaderboard"

#: An MLflow run is only comparable once it has finished.
RUN_STATUS_FINISHED: str = "FINISHED"

#: Convenience tag names written by the battery runner / tracking convention.
SPLIT_VERSION_TAG: str = "split_version"
MODEL_NAME_TAG: str = "model_name"
FAMILY_TAG: str = "family"

#: Optional tag recording the compute device a run used. Neural runs set it
#: (see ``heart.models.run_mlp_sweep``); classical runs leave it unset. The
#: leaderboard reads it generically — a missing tag renders as no-device, it is
#: never a reason to drop or special-case a row.
DEVICE_TAG: str = "device"

#: Page size when paging the MLflow run search.
SEARCH_PAGE_SIZE: int = 1000

#: Hard cap on runs read from the store, to bound memory on a bloated store.
MAX_RUNS_CONSIDERED: int = 10_000

#: Rendered when a run did not record a compute device (e.g. classical models).
NO_DEVICE_LABEL: str = "-"

#: Structured-suite leaves that live under ``calibration.*`` and are reported.
CALIBRATION_SCALAR_SUFFIXES: tuple[str, ...] = (
    "brier_score",
    "expected_calibration_error",
    "n_bins",
)

# --- S06 selection annotation tag vocabulary (see SelectionAnnotations) ---

#: Prefix every S06 selection annotation tag carries on the winner's run.
SELECTION_TAG_PREFIX: str = "s6."

#: Tags that MUST be present once any ``s6.*`` annotation exists: without the
#: winner identity, the selection metric, its repeated-CV distribution and the
#: ship verdict, the annotation set is not a coherent selection story.
REQUIRED_SELECTION_FIELDS: tuple[str, ...] = (
    "winner",
    "metric",
    "cv_mean",
    "cv_std",
    "alpha",
    "ship",
)

#: Tags rendered when present, tolerated when absent (e.g. a single-candidate
#: selection has no baseline comparison to annotate).
OPTIONAL_SELECTION_FIELDS: tuple[str, ...] = (
    "baseline",
    "mean_diff",
    "p_value",
    "significance",
    "mcnemar_p_value",
    "threshold",
    "threshold_objective",
    "threshold_score",
    "brier_raw",
    "brier_calibrated",
    "serving_weight_mb",
    "serving_weight_budget_mb",
    "flags",
)

#: All accepted ``s6.*`` field names (the closed vocabulary).
ALL_SELECTION_FIELDS: frozenset[str] = frozenset(
    (*REQUIRED_SELECTION_FIELDS, *OPTIONAL_SELECTION_FIELDS)
)

#: Separator used inside the ``s6.flags`` tag value (flag detail parentheses
#: may contain commas, spaces, and equals signs — never the separator).
SELECTION_FLAGS_SEPARATOR: str = " || "

#: Upper bound on one annotation tag value, checked when tagging a run
#: (MLflow refuses values beyond roughly 6000 bytes; we stay clearly inside).
MAX_TAG_VALUE_LENGTH: int = 5000


def selection_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """Return the ``s6.*`` selection-annotation sub-dict of a run's tag dict.

    Keys are returned *unprefixed* (``s6.baseline_p_value`` becomes
    ``baseline_p_value``) — which is the vocabulary
    :class:`SelectionAnnotations` validates names against. Runs without any
    ``s6.*`` tag yield an empty dict, the leaderboard's normal state before
    the selection flow has annotated a winner.
    """
    prefix = SELECTION_TAG_PREFIX
    return {
        str(key)[len(prefix):]: str(value)
        for key, value in dict(tags).items()
        if str(key).startswith(prefix)
    }


def _confusion_flat_key(key: str) -> str:
    return f"{CONFUSION_MATRIX_KEY}.{key}"


def _calibration_flat_key(key: str) -> str:
    return f"{CALIBRATION_KEY}.{key}"


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return tuple(seen)


#: Every scalar leaf of the canonical metric dict a candidate run must carry.
REQUIRED_FLAT_KEYS: tuple[str, ...] = _dedupe(
    (
        *SCALAR_METRIC_KEYS,
        *(_confusion_flat_key(key) for key in CONFUSION_MATRIX_KEYS),
        *(_calibration_flat_key(key) for key in CALIBRATION_SCALAR_SUFFIXES),
    )
)

#: Scalar columns of the ranked table: primary metric first, then the rest.
RANKED_SCALAR_COLUMNS: tuple[str, ...] = (PRIMARY_METRIC,) + tuple(
    key for key in SCALAR_METRIC_KEYS if key != PRIMARY_METRIC
)

#: Human-readable headers for the ranked scalar columns.
COLUMN_LABELS: dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1": "F1",
    "npv": "NPV",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "prevalence": "Prevalence",
    "brier_score": "Brier",
    "expected_calibration_error": "ECE",
}


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class LeaderboardError(Exception):
    """Base class for every leaderboard generation failure."""


class LeaderboardConfigError(LeaderboardError):
    """The requested leaderboard configuration is invalid."""


class LeaderboardDataError(LeaderboardError):
    """The MLflow store could not be read, or a run's data is malformed."""


class LeaderboardExperimentNotFoundError(LeaderboardDataError):
    """The named MLflow experiment does not exist in the store."""


class LeaderboardReportError(LeaderboardError):
    """The leaderboard report could not be rendered or written."""


class NoBenchmarkedModelsError(LeaderboardDataError):
    """Strict mode: no benchmarked models matched the selection."""


class SelectionAnnotationsError(LeaderboardError):
    """The selection annotation set is ambiguous or cannot be produced.

    Raised when more than one board row carries ``s6.*`` annotations (which
    winner?), or when the source ``selection.md`` cannot be parsed into the
    tag vocabulary.
    """


class S06LeaderboardDataError(LeaderboardDataError):
    """A run's ``s6.*`` annotation set is incomplete, malformed, or oversized."""


@dataclass(frozen=True)
class SelectionAnnotations:
    """The validated S06 selection narrative read from the winner run's tags.

    Required fields (``winner_type``, ``metric``, ``cv_mean``, ``cv_std``,
    ``alpha``, ``ship``) are the non-negotiable story; optional fields render
    only when present, so a single-candidate selection with no baseline
    comparison is still a coherent annotation set.
    """

    winner_type: str
    model_name: str
    run_id: str
    metric: str
    cv_mean: float
    cv_std: float
    alpha: float
    ship: bool
    fields: dict[str, str]

    @classmethod
    def from_row(cls, row: LeaderboardRow) -> SelectionAnnotations:
        fields = selection_tags(row.tags)
        missing = [
            key for key in REQUIRED_SELECTION_FIELDS if key not in fields
        ]
        if missing:
            raise S06LeaderboardDataError(
                f"Run {row.run_id!r} ({row.model_type!r}) carries a partial "
                f"{SELECTION_TAG_PREFIX!r}* annotation set; missing required "
                f"field(s) {missing}. The winner run must be annotated by the "
                "selection flow, not hand-edited."
            )
        unknown = sorted(set(fields) - ALL_SELECTION_FIELDS)
        if unknown:
            raise S06LeaderboardDataError(
                f"Run {row.run_id!r} carries unknown selection annotation "
                f"field(s) {unknown}; the annotation vocabulary is "
                f"{sorted(ALL_SELECTION_FIELDS)}."
            )
        oversized = sorted(
            key for key, value in fields.items() if len(value) > MAX_TAG_VALUE_LENGTH
        )
        if oversized:
            raise S06LeaderboardDataError(
                f"Run {row.run_id!r} annotation field(s) {oversized} exceed the "
                f"{MAX_TAG_VALUE_LENGTH}-character tag-value cap."
            )
        winner_type = fields["winner"]
        if winner_type != row.model_type:
            raise S06LeaderboardDataError(
                f"Run {row.run_id!r} is annotated as the winner of model type "
                f"{winner_type!r} but its own model_type tag is "
                f"{row.model_type!r}; the annotation points at the wrong run."
            )
        ship = fields["ship"]
        if ship not in ("SHIP", "NO-SHIP"):
            raise S06LeaderboardDataError(
                f"Run {row.run_id!r} annotation 'ship' must be SHIP or NO-SHIP, "
                f"got {ship!r}."
            )
        alpha = _annotation_float(fields["alpha"], field="alpha", run_id=row.run_id)
        if not 0.0 < alpha < 1.0:
            raise S06LeaderboardDataError(
                f"Run {row.run_id!r} annotation 'alpha' must be strictly "
                f"between 0 and 1, got {fields['alpha']!r}."
            )
        cv_std = _annotation_float(
            fields["cv_std"], field="cv_std", run_id=row.run_id
        )
        if cv_std < 0.0:
            raise S06LeaderboardDataError(
                f"Run {row.run_id!r} annotation 'cv_std' must be "
                f"non-negative, got {fields['cv_std']!r}."
            )
        return cls(
            winner_type=winner_type,
            model_name=row.model_name,
            run_id=row.run_id,
            metric=fields["metric"],
            cv_mean=_annotation_float(
                fields["cv_mean"], field="cv_mean", run_id=row.run_id
            ),
            cv_std=cv_std,
            alpha=alpha,
            ship=ship == "SHIP",
            fields=fields,
        )

    # -- optional fields --------------------------------------------------

    def optional_float(self, field: str) -> float | None:
        raw = self.fields.get(field)
        if raw is None:
            return None
        return _annotation_float(raw, field=field, run_id=self.run_id)

    @property
    def baseline(self) -> str | None:
        return self.fields.get("baseline") or None

    @property
    def p_value(self) -> float | None:
        return self.optional_float("p_value")

    @property
    def mcnemar_p_value(self) -> float | None:
        return self.optional_float("mcnemar_p_value")

    @property
    def significance(self) -> str | None:
        return self.fields.get("significance") or None

    @property
    def threshold(self) -> float | None:
        return self.optional_float("threshold")

    @property
    def threshold_objective(self) -> str | None:
        return self.fields.get("threshold_objective") or None

    @property
    def threshold_score(self) -> float | None:
        return self.optional_float("threshold_score")

    @property
    def brier_raw(self) -> float | None:
        return self.optional_float("brier_raw")

    @property
    def brier_calibrated(self) -> float | None:
        return self.optional_float("brier_calibrated")

    @property
    def serving_weight_mb(self) -> float | None:
        return self.optional_float("serving_weight_mb")

    @property
    def serving_weight_budget_mb(self) -> float | None:
        return self.optional_float("serving_weight_budget_mb")

    @property
    def flags(self) -> tuple[str, ...]:
        raw = self.fields.get("flags", "")
        return tuple(
            part.strip()
            for part in raw.split(SELECTION_FLAGS_SEPARATOR)
            if part.strip()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "winner_type": self.winner_type,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "metric": self.metric,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "alpha": self.alpha,
            "ship": self.ship,
            "baseline": self.baseline,
            "p_value": self.p_value,
            "significance": self.significance,
            "mcnemar_p_value": self.mcnemar_p_value,
            "threshold": self.threshold,
            "threshold_objective": self.threshold_objective,
            "threshold_score": self.threshold_score,
            "brier_raw": self.brier_raw,
            "brier_calibrated": self.brier_calibrated,
            "serving_weight_mb": self.serving_weight_mb,
            "serving_weight_budget_mb": self.serving_weight_budget_mb,
            "flags": list(self.flags),
        }


def _annotation_float(value: str, *, field: str, run_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise S06LeaderboardDataError(
            f"Run {run_id!r} annotation {field!r} is not a number: {value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise S06LeaderboardDataError(
            f"Run {run_id!r} annotation {field!r} must be finite, got {value!r}."
        )
    return parsed


def read_selection_annotations(
    board: Leaderboard,
) -> SelectionAnnotations | None:
    """Promote the one annotated winner row's tags into annotations.

    Returns ``None`` before the selection flow has annotated anyone; raises
    :class:`SelectionAnnotationsError` on two (ambiguous winners); any
    malformed field raises :class:`S06LeaderboardDataError`.
    """
    if not board.rows:
        return None
    annotated = [row for row in board.rows if selection_tags(row.tags)]
    if not annotated:
        return None
    if len(annotated) > 1:
        raise SelectionAnnotationsError(
            f"{len(annotated)} leaderboard row(s) carry "
            f"{SELECTION_TAG_PREFIX!r}* annotations: "
            f"{sorted(row.model_type for row in annotated)}. Exactly one run "
            "(the winner's) may carry the selection set; re-run the selection "
            "flow so only its winner is annotated."
        )
    return SelectionAnnotations.from_row(annotated[0])


def selection_annotation_tags(decision: object) -> dict[str, str]:
    """Flatten a selection decision into the ``s6.*`` winner-run tag dict.

    Consumes a :class:`heart.eval.selection.SelectionDecision`-shaped object
    (duck-typed; the selection dataclass is imported only for typing) and
    produces one string value per annotation field. Floats are rendered at
    fixed precision so :meth:`SelectionAnnotations.from_row` can round-trip
    them exactly. The joined ``s6.flags`` value is checked against
    :data:`MAX_TAG_VALUE_LENGTH` *before* MLflow ever sees it, so an overflow
    surfaces as a named error here, not as an opaque MLflow refusal.

    Required fields are always emitted; optional fields are emitted when the
    decision carries them (single-candidate selections have no baseline
    comparison to annotate).

    Raises
    ------
    SelectionAnnotationsError
        The joined flag list exceeds the declared tag-value cap.
    """
    ranked = tuple(getattr(decision, "ranked"))
    if not ranked:
        raise SelectionAnnotationsError(
            "Selection decision has an empty ranking; no winner run can be "
            "annotated from it."
        )
    flags_joined = SELECTION_FLAGS_SEPARATOR.join(
        str(flag) for flag in tuple(getattr(decision, "flags"))
    )
    if len(flags_joined) > MAX_TAG_VALUE_LENGTH:
        raise SelectionAnnotationsError(
            f"Joined flags value is {len(flags_joined)} characters, beyond the "
            f"{MAX_TAG_VALUE_LENGTH}-character tag-value cap; trim flag detail "
            "before journaling the selection."
        )

    prefix = SELECTION_TAG_PREFIX
    tags: dict[str, str] = {
        f"{prefix}winner": str(getattr(decision, "winner")),
        f"{prefix}metric": str(getattr(decision, "metric")),
        f"{prefix}cv_mean": f"{ranked[0][1]:.6f}",
        f"{prefix}cv_std": f"{ranked[0][2]:.6f}",
        f"{prefix}alpha": f"{float(getattr(decision, 'alpha')):g}",
        f"{prefix}ship": "SHIP" if getattr(decision, "ship") else "NO-SHIP",
    }

    comparison = getattr(decision, "winner_vs_baseline", None)
    if comparison is not None:
        corrected = comparison.corrected_t
        tags[f"{prefix}baseline"] = str(getattr(decision, "baseline"))
        tags[f"{prefix}p_value"] = f"{corrected.p_value:.4g}"
        tags[f"{prefix}mean_diff"] = f"{corrected.mean_diff:+.6f}"
        tags[f"{prefix}significance"] = corrected.verdict()
        if comparison.agreement is not None:
            tags[f"{prefix}mcnemar_p_value"] = (
                f"{comparison.agreement.p_value:.4g}"
            )

    tuning = getattr(decision, "threshold", None)
    if tuning is not None:
        tags[f"{prefix}threshold"] = f"{tuning.best_threshold:.2f}"
        tags[f"{prefix}threshold_objective"] = str(tuning.objective)
        tags[f"{prefix}threshold_score"] = f"{tuning.best_score:.6g}"

    calibration = getattr(decision, "calibration", None)
    if calibration is not None:
        tags[f"{prefix}brier_raw"] = f"{calibration.brier_score_raw:.6f}"
        tags[f"{prefix}brier_calibrated"] = (
            f"{calibration.brier_score_calibrated:.6f}"
        )

    weight = getattr(decision, "serving_weight", None)
    if weight is not None:
        tags[f"{prefix}serving_weight_mb"] = f"{weight.weight_mb:.2f}"
        tags[f"{prefix}serving_weight_budget_mb"] = f"{weight.budget_mb:.2f}"

    if flags_joined:
        tags[f"{prefix}flags"] = flags_joined
    return tags


def annotate_winner_run(
    decision: object,
    *,
    split_version: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_dir: str | Path | None = None,
) -> str | None:
    """Tag the winner's latest final MLflow run with the ``s6.*`` annotations.

    Write half of the S06/T05 annotation loop: the selection flow journals its
    decision as MLflow tags so the leaderboard regeneration re-derives the
    selection narrative from the store alone. The run is the winner model
    type's most recent finished ``final`` run — the same selection rule the
    leaderboard applies, so the annotated run and the rendered row can never
    drift apart.

    Returns the annotated run id, or ``None`` when the winner has no final run
    in the store yet (warning-logged: annotation is a soft dependency of
    selection — never a reason to refuse writing selection.md).

    Raises
    ------
    SelectionAnnotationsError
        The tag payload is malformed (e.g. flag-value overflow).
    LeaderboardDataError / LeaderboardExperimentNotFoundError
        The experiment could not be read from the store.
    """
    tags = selection_annotation_tags(decision)
    winner = tags[f"{SELECTION_TAG_PREFIX}winner"]
    tracking_uri = resolve_tracking_uri(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
    except (MlflowException, OSError) as exc:
        raise LeaderboardDataError(
            f"Could not read experiment {experiment_name!r} from the MLflow "
            f"store at {tracking_uri} while annotating the winner run: {exc}"
        ) from exc
    if experiment is None:
        raise LeaderboardExperimentNotFoundError(
            f"No MLflow experiment named {experiment_name!r} exists in the "
            f"store at {tracking_uri}; the winner run cannot be annotated. Run "
            "the battery first, or point the selection flow at a configured "
            "tracking store."
        )
    filters = [
        f"tags.{MODEL_TYPE_TAG} = '{winner}'",
        f"tags.{RUN_KIND_TAG} = '{DEFAULT_RUN_KIND}'",
    ]
    if split_version is not None:
        filters.append(f"tags.{SPLIT_VERSION_TAG} = '{split_version}'")
    try:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=" and ".join(filters),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    except MlflowException as exc:
        raise LeaderboardDataError(
            f"Could not search for the winner run ({winner!r}) in experiment "
            f"{experiment_name!r}: {exc}"
        ) from exc
    if not runs:
        logger.warning(
            "No finished %s run for winner %r in experiment %r; S06 leaderboard "
            "annotations were skipped (the leaderboard renders without the "
            "selection section until the battery logs the winner).",
            DEFAULT_RUN_KIND,
            winner,
            experiment_name,
        )
        return None
    run_id = str(runs[0].info.run_id)
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    logger.info(
        "Tagged winner run %s (%s) with %d S06 annotation tag(s) "
        "(ship=%s, threshold=%s, p=%s)",
        run_id,
        winner,
        len(tags),
        tags[f"{SELECTION_TAG_PREFIX}ship"],
        tags.get(f"{SELECTION_TAG_PREFIX}threshold", "n/a"),
        tags.get(f"{SELECTION_TAG_PREFIX}p_value", "n/a"),
    )
    return run_id


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaderboardConfig:
    """The validated knobs for one leaderboard generation.

    ``strict=True`` refuses to publish an empty leaderboard; the default
    ``False`` emits a report that explains that nothing matched, which keeps
    regeneration usable before the battery has ever run.
    """

    experiment_name: str = DEFAULT_EXPERIMENT
    split_version: str | None = None
    run_kind: str = DEFAULT_RUN_KIND
    latest_only: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise LeaderboardConfigError(
                f"experiment_name must be a non-empty string, got "
                f"{self.experiment_name!r}."
            )
        if not isinstance(self.run_kind, str) or not self.run_kind.strip():
            raise LeaderboardConfigError(
                f"run_kind must be a non-empty string, got {self.run_kind!r}."
            )
        if self.split_version is not None and (
            not isinstance(self.split_version, str) or not self.split_version.strip()
        ):
            raise LeaderboardConfigError(
                "split_version must be None or a non-empty string, got "
                f"{self.split_version!r}."
            )
        if not isinstance(self.latest_only, bool):
            raise LeaderboardConfigError(
                f"latest_only must be a bool, got {type(self.latest_only).__name__}."
            )
        if not isinstance(self.strict, bool):
            raise LeaderboardConfigError(
                f"strict must be a bool, got {type(self.strict).__name__}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "split_version": self.split_version,
            "run_kind": self.run_kind,
            "latest_only": self.latest_only,
            "strict": self.strict,
        }


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaderboardRow:
    """One benchmarked model on the leaderboard, read from an MLflow run."""

    model_type: str
    model_name: str
    family: str
    run_id: str
    run_name: str
    split_version: str
    flat_metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]
    start_time_ms: int | None = None

    def metric(self, key: str, default: float | None = None) -> float | None:
        """Return a flattened metric by key (or ``default`` when absent)."""
        return self.flat_metrics.get(key, default)

    @property
    def primary_metric(self) -> float:
        return float(self.flat_metrics[PRIMARY_METRIC])

    @property
    def device(self) -> str | None:
        """The compute device this run used, or ``None`` when unrecorded.

        Read generically from the optional ``device`` tag: neural runs set it
        (``cuda:0`` on the ROCm path, ``cpu`` on a logged fallback) while
        classical runs simply leave it unset.
        """
        return self.tags.get(DEVICE_TAG) or None

    @property
    def n_samples(self) -> int | None:
        value = self.flat_metrics.get(_confusion_flat_key("n_samples"))
        return None if value is None else int(value)

    def structured(self, key: str, *, block: str) -> float | None:
        prefix = {
            CONFUSION_MATRIX_KEY: CONFUSION_MATRIX_KEY,
            CALIBRATION_KEY: CALIBRATION_KEY,
        }[block]
        return self.flat_metrics.get(f"{prefix}.{key}")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "family": self.family,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "split_version": self.split_version,
            "device": self.device,
            "start_time_ms": self.start_time_ms,
            "params": dict(self.params),
            "tags": dict(self.tags),
            "flat_metrics": {key: float(value) for key, value in self.flat_metrics.items()},
        }


@dataclass(frozen=True)
class Leaderboard:
    """The ranked leaderboard and the provenance needed to regenerate it."""

    experiment_name: str
    tracking_uri: str
    run_kind: str
    split_version: str | None
    latest_only: bool
    rows: tuple[LeaderboardRow, ...]
    n_runs_considered: int
    generated_at: str
    report_path: Path | None = None
    annotations: "SelectionAnnotations | None" = None

    @property
    def n_models(self) -> int:
        return len(self.rows)

    @property
    def top(self) -> LeaderboardRow | None:
        return self.rows[0] if self.rows else None

    def row_for(self, model_type: str) -> LeaderboardRow | None:
        for row in self.rows:
            if row.model_type == model_type:
                return row
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "tracking_uri": self.tracking_uri,
            "run_kind": self.run_kind,
            "split_version": self.split_version,
            "latest_only": self.latest_only,
            "n_runs_considered": int(self.n_runs_considered),
            "n_models": self.n_models,
            "generated_at": self.generated_at,
            "report_path": str(self.report_path) if self.report_path else None,
            "annotations": (
                self.annotations.to_dict() if self.annotations is not None else None
            ),
            "rows": [row.to_dict() for row in self.rows],
        }


# ---------------------------------------------------------------------------
# Store reading
# ---------------------------------------------------------------------------


def _iter_runs(client: MlflowClient, experiment_id: str):
    """Yield every run in the experiment, paging and capping for safety."""
    page_token: str | None = None
    seen = 0
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=SEARCH_PAGE_SIZE,
            page_token=page_token,
        )
        for run in page:
            yield run
            seen += 1
            if seen >= MAX_RUNS_CONSIDERED:
                logger.warning(
                    "Reached the %d-run read cap for experiment %s; remaining "
                    "runs were not considered.",
                    MAX_RUNS_CONSIDERED,
                    experiment_id,
                )
                return
        page_token = getattr(page, "token", None)
        if not page_token:
            return


def _run_candidate(run: object) -> LeaderboardRow:
    """Convert one finished, tagged MLflow run into a leaderboard row."""
    info = run.info  # type: ignore[attr-defined]
    data = run.data  # type: ignore[attr-defined]
    tags = {str(key): str(value) for key, value in dict(data.tags).items()}
    model_type = tags.get(MODEL_TYPE_TAG)
    if not model_type:
        raise LeaderboardDataError(
            f"Run {info.run_id!r} is missing the {MODEL_TYPE_TAG!r} tag; only "
            "benchmarked runs belong on the leaderboard."
        )
    flat_metrics = {
        str(key): float(value) for key, value in dict(data.metrics).items()
    }
    missing = [key for key in REQUIRED_FLAT_KEYS if key not in flat_metrics]
    if missing:
        raise LeaderboardDataError(
            f"Run {info.run_id!r} ({model_type!r}) is missing {len(missing)} "
            f"required metric(s): {missing}. The leaderboard needs the complete "
            "flattened suite; re-log the run through heart.tracking.run instead "
            "of hand-assembling metrics."
        )
    start_time = getattr(info, "start_time", None)
    return LeaderboardRow(
        model_type=model_type,
        model_name=tags.get(MODEL_NAME_TAG, model_type),
        family=tags.get(FAMILY_TAG, ""),
        run_id=str(info.run_id),
        run_name=str(tags.get("mlflow.runName", "")),
        split_version=tags.get(SPLIT_VERSION_TAG, ""),
        flat_metrics=flat_metrics,
        params={str(key): str(value) for key, value in dict(data.params).items()},
        tags=tags,
        start_time_ms=None if start_time is None else int(start_time),
    )


def _prefer_candidate(candidate: LeaderboardRow, current: LeaderboardRow) -> bool:
    """True when ``candidate`` should win the per-model dedup."""
    candidate_start = candidate.start_time_ms or 0
    current_start = current.start_time_ms or 0
    if candidate_start != current_start:
        return candidate_start > current_start
    return candidate.primary_metric > current.primary_metric


def _dedupe_by_model(
    candidates: Sequence[LeaderboardRow], *, latest_only: bool
) -> list[LeaderboardRow]:
    if not latest_only:
        return list(candidates)
    best: dict[str, LeaderboardRow] = {}
    for candidate in candidates:
        current = best.get(candidate.model_type)
        if current is None or _prefer_candidate(candidate, current):
            best[candidate.model_type] = candidate
    return list(best.values())


def _rank(rows: Sequence[LeaderboardRow]) -> tuple[LeaderboardRow, ...]:
    return tuple(
        sorted(rows, key=lambda row: (-row.primary_metric, row.model_type))
    )


# ---------------------------------------------------------------------------
# Selection / building
# ---------------------------------------------------------------------------


def select_leaderboard_rows(
    *,
    config: LeaderboardConfig | None = None,
    tracking_dir: str | Path | None = None,
) -> tuple[tuple[LeaderboardRow, ...], int]:
    """Select and rank the leaderboard rows from the MLflow store.

    Returns ``(rows, n_runs_considered)`` where ``n_runs_considered`` counts
    every finished, tagged run that matched the tag/split filters *before*
    per-model deduplication.
    """
    resolved = config or LeaderboardConfig()
    tracking_uri = resolve_tracking_uri(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(resolved.experiment_name)
    except (MlflowException, OSError) as exc:
        raise LeaderboardDataError(
            f"Could not read experiment {resolved.experiment_name!r} from the "
            f"MLflow store at {tracking_uri}: {exc}"
        ) from exc
    if experiment is None:
        raise LeaderboardExperimentNotFoundError(
            f"No MLflow experiment named {resolved.experiment_name!r} exists in "
            f"the store at {tracking_uri}. Run the battery (or a tracked model) "
            "first, or point --experiment at the right experiment."
        )

    candidates: list[LeaderboardRow] = []
    try:
        for run in _iter_runs(client, experiment.experiment_id):
            tags = dict(run.data.tags)
            if tags.get(RUN_KIND_TAG) != resolved.run_kind:
                continue
            if str(run.info.status) != RUN_STATUS_FINISHED:
                continue
            if not tags.get(MODEL_TYPE_TAG):
                continue
            if (
                resolved.split_version is not None
                and tags.get(SPLIT_VERSION_TAG) != resolved.split_version
            ):
                continue
            candidates.append(_run_candidate(run))
    except MlflowException as exc:
        raise LeaderboardDataError(
            f"Could not search runs in experiment {resolved.experiment_name!r}: "
            f"{exc}"
        ) from exc

    rows = _rank(_dedupe_by_model(candidates, latest_only=resolved.latest_only))
    return rows, len(candidates)


def build_leaderboard(
    *,
    config: LeaderboardConfig | None = None,
    tracking_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> Leaderboard:
    """Build the ranked leaderboard from the MLflow store."""
    resolved = config or LeaderboardConfig()
    rows, considered = select_leaderboard_rows(
        config=resolved, tracking_dir=tracking_dir
    )
    if not rows and resolved.strict:
        raise NoBenchmarkedModelsError(
            f"No benchmarked models matched experiment "
            f"{resolved.experiment_name!r} with run_kind={resolved.run_kind!r}"
            + (
                f" and split_version={resolved.split_version!r}"
                if resolved.split_version
                else ""
            )
            + ". Run the battery first, or drop strict mode to publish an "
            "explicit empty report."
        )

    board = Leaderboard(
        experiment_name=resolved.experiment_name,
        tracking_uri=resolve_tracking_uri(tracking_dir=tracking_dir),
        run_kind=resolved.run_kind,
        split_version=resolved.split_version,
        latest_only=resolved.latest_only,
        rows=rows,
        n_runs_considered=considered,
        generated_at=generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report_path=None,
    )
    # S06 selection annotations are validated at build time so a malformed
    # winner-run tag set fails the generation with a named error, never a
    # silently half-annotated report.
    try:
        annotations = read_selection_annotations(board)
    except (SelectionAnnotationsError, S06LeaderboardDataError):
        logger.exception("S06 selection annotations could not be read")
        raise
    board = replace(board, annotations=annotations)
    logger.info(
        "Built leaderboard from experiment %s: %d run(s) considered, %d model(s) "
        "ranked%s%s",
        board.experiment_name,
        board.n_runs_considered,
        board.n_models,
        f", top={board.top.model_type} ({PRIMARY_METRIC}={board.top.primary_metric:.4f})"
        if board.top
        else "",
        f", S06 winner annotated from run {board.annotations.run_id}"
        if board.annotations is not None
        else "",
    )
    return board


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _num(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _count(value: object) -> str:
    return str(int(float(value)))


def _column_label(key: str) -> str:
    return COLUMN_LABELS.get(key, key)


def _main_table(board: Leaderboard) -> list[str]:
    headers = [
        "Rank",
        "Model",
        "Type",
        "Family",
        "Device",
        *( _column_label(key) for key in RANKED_SCALAR_COLUMNS),
        "ECE",
        "N",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for rank, row in enumerate(board.rows, start=1):
        cells = [
            str(rank),
            row.model_name,
            f"`{row.model_type}`",
            row.family,
            row.device or NO_DEVICE_LABEL,
            *(_num(row.metric(key)) for key in RANKED_SCALAR_COLUMNS),
            _num(
                row.metric(_calibration_flat_key("expected_calibration_error"), 0.0)
            ),
            _count(row.n_samples if row.n_samples is not None else 0),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _confusion_table(board: Leaderboard) -> list[str]:
    columns = ("tn", "fp", "fn", "tp", "n_samples", "n_positive", "n_negative")
    lines = [
        "| Model | TN | FP | FN | TP | Samples | Positives | Negatives |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in board.rows:
        cells = [_count(row.metric(_confusion_flat_key(key), 0.0)) for key in columns]
        lines.append("| " + " | ".join([row.model_name, *cells]) + " |")
    return lines


def _calibration_table(board: Leaderboard) -> list[str]:
    lines = [
        "| Model | Brier | Expected calibration error | Bins |",
        "| --- | --- | --- | --- |",
    ]
    for row in board.rows:
        brier = _num(row.metric(_calibration_flat_key("brier_score"), 0.0))
        ece = _num(
            row.metric(_calibration_flat_key("expected_calibration_error"), 0.0)
        )
        n_bins = _count(row.metric(_calibration_flat_key("n_bins"), 0.0))
        lines.append(f"| {row.model_name} | {brier} | {ece} | {n_bins} |")
    return lines


def _provenance_lines(board: Leaderboard) -> list[str]:
    lines = [
        f"- experiment: `{board.experiment_name}`",
        f"- run kind: `{board.run_kind}` (one final run per benchmarked model)",
    ]
    if board.split_version is not None:
        lines.append(f"- split version: `{board.split_version}`")
    lines.append(
        f"- models ranked: **{board.n_models}** "
        f"(from {board.n_runs_considered} candidate run(s)"
        + (", latest run per model" if board.latest_only else ", all runs")
        + ")"
    )
    lines.append(f"- generated at: {board.generated_at}")
    return lines


def _run_budget(row: LeaderboardRow) -> str:
    """Render the recorded tuning budget for a row, when the tags carry it."""
    folds = row.tags.get("cv_folds")
    trials = row.tags.get("n_trials")
    if folds and trials:
        return f" — {folds}-fold CV, {trials} trial(s)"
    if folds:
        return f" — {folds}-fold CV"
    return ""


def render_leaderboard(board: Leaderboard) -> str:
    """Render a :class:`Leaderboard` as the markdown report."""
    if not isinstance(board, Leaderboard):
        raise LeaderboardReportError(
            f"board must be a Leaderboard, got {type(board).__name__}."
        )
    lines: list[str] = []
    lines.append(f"# {LEADERBOARD_TITLE}")
    lines.append("")
    lines.append(
        "_Generated by `heart.reporting.leaderboard` from recorded MLflow runs. "
        "Do not edit by hand — regenerate with "
        "`python -m heart.reporting.leaderboard`._"
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.extend(_provenance_lines(board))
    lines.append("")
    if not board.rows:
        lines.append("## Leaderboard")
        lines.append("")
        lines.append(
            "_No benchmarked models matched this selection. Run the battery "
            "first: `python -m heart.models.run_battery --smoke`._"
        )
        lines.append("")
        lines.append("## Reproduce")
        lines.append("")
        lines.append("```bash")
        lines.append("python -m heart.models.run_battery --smoke")
        lines.append("python -m heart.models.run_mlp_sweep --smoke")
        lines.append("python -m heart.reporting.leaderboard")
        lines.append("```")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"## Leaderboard (ordered by {_column_label(PRIMARY_METRIC)})")
    lines.append("")
    lines.extend(_main_table(board))
    lines.append("")
    lines.append("## Confusion matrices")
    lines.append("")
    lines.extend(_confusion_table(board))
    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    lines.extend(_calibration_table(board))
    lines.append("")
    selection = board.annotations
    if selection is None and board.rows:
        selection = read_selection_annotations(board)
    if selection is not None:
        lines.append("## Winner selection (S06)")
        lines.append("")
        lines.extend(_selection_lines(selection))
        lines.append("")
    lines.append("## Run provenance")
    lines.append("")
    for row in board.rows:
        lines.append(
            f"- **{row.model_name}** (`{row.model_type}`) — run `{row.run_id}`"
            + (f", split `{row.split_version}`" if row.split_version else "")
            + (f", device `{row.device}`" if row.device else "")
            + _run_budget(row)
        )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m heart.models.run_battery --smoke")
    lines.append("python -m heart.models.run_mlp_sweep --smoke")
    lines.append("python -m heart.reporting.leaderboard")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _selection_lines(annotations: SelectionAnnotations) -> list[str]:
    """Render one winner's selection narrative as a bullet block."""
    lines = [
        f"- winner: **{annotations.model_name}** (`{annotations.winner_type}`) "
        f"— run `{annotations.run_id}`",
        f"- repeated-CV {annotations.metric}: "
        f"{annotations.cv_mean:.6f} ± {annotations.cv_std:.6f} (alpha {annotations.alpha:g})",
    ]
    if annotations.baseline is not None and annotations.p_value is not None:
        verdict = annotations.significance or (
            "significant" if annotations.ship else "not significant"
        )
        lines.append(
            f"- significance vs baseline `{annotations.baseline}`: "
            f"p = {annotations.p_value:.4g} — {verdict}"
        )
    if annotations.mcnemar_p_value is not None:
        lines.append(
            f"- McNemar disagreement test: p = {annotations.mcnemar_p_value:.4g}"
        )
    if annotations.threshold is not None:
        objective = annotations.threshold_objective
        score = annotations.threshold_score
        score_part = (
            f" (objective `{objective}` at {score:.6g})"
            if objective and score is not None
            else ""
        )
        lines.append(
            f"- calibrated threshold: **{annotations.threshold:.2f}**{score_part} "
            "— swept on validation rows only"
        )
    if annotations.brier_raw is not None and annotations.brier_calibrated is not None:
        direction = (
            "improved" if annotations.brier_calibrated < annotations.brier_raw
            else "did not improve"
        )
        lines.append(
            f"- calibration Brier: {annotations.brier_raw:.4f} → "
            f"{annotations.brier_calibrated:.4f} ({direction})"
        )
    if annotations.serving_weight_mb is not None:
        budget = annotations.serving_weight_budget_mb
        budget_part = (
            f" (budget {budget:.2f} MB)" if budget is not None else ""
        )
        lines.append(
            f"- serving weight: **{annotations.serving_weight_mb:.2f} MB**"
            f"{budget_part} — "
            + (
                "within budget"
                if budget is None or annotations.serving_weight_mb <= budget
                else "OVER budget"
            )
        )
    lines.append(
        f"- flags: {', '.join(annotations.flags) if annotations.flags else 'none'}"
    )
    lines.append(
        f"- ship verdict: **{'SHIP' if annotations.ship else 'NO-SHIP'}**"
    )
    return lines


def describe_leaderboard(board: Leaderboard) -> str:
    """Render the provenance block as a human-readable diagnostic string."""
    return "\n".join(_provenance_lines(board))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_leaderboard(board: Leaderboard, path: str | Path) -> Path:
    """Atomically write ``board`` as markdown to ``path``."""
    if not isinstance(board, Leaderboard):
        raise LeaderboardReportError(
            f"board must be a Leaderboard, got {type(board).__name__}."
        )
    destination = Path(path)
    rendered = render_leaderboard(board)
    try:
        _write_text_atomic(destination, rendered)
    except OSError as exc:
        raise LeaderboardReportError(
            f"Could not write the leaderboard to {destination}: {exc}"
        ) from exc
    logger.info(
        "Wrote leaderboard to %s (%d model(s))", destination, board.n_models
    )
    return destination


def generate_leaderboard(
    *,
    config: LeaderboardConfig | None = None,
    tracking_dir: str | Path | None = None,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    write: bool = True,
) -> Leaderboard:
    """Build the leaderboard and (optionally) publish it as a report artifact."""
    board = build_leaderboard(config=config, tracking_dir=tracking_dir)
    if not write or report_path is None:
        return board
    destination = write_leaderboard(board, report_path)
    return replace(board, report_path=destination)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.reporting.leaderboard",
        description=(
            "Generate the model leaderboard from recorded MLflow runs "
            "(one final run per benchmarked model, classical or neural)."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--tracking-dir",
        default=None,
        help="MLflow store directory (default: experiments/mlruns).",
    )
    parser.add_argument(
        "--split-version",
        default=None,
        help="Only include runs for this S01 split version (default: all).",
    )
    parser.add_argument(
        "--run-kind",
        default=DEFAULT_RUN_KIND,
        help=f"MLflow run_kind tag to include (default: {DEFAULT_RUN_KIND}).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include every matching run instead of the latest per model.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Where to write the report (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Print summary only; do not write the report file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when no benchmarked models match instead of "
        "writing an empty report.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> LeaderboardConfig:
    return LeaderboardConfig(
        experiment_name=args.experiment,
        split_version=args.split_version,
        run_kind=args.run_kind,
        latest_only=not args.all_runs,
        strict=args.strict,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        config = _config_from_args(args)
    except LeaderboardConfigError as exc:
        print(f"configuration error: {exc}")
        return 2

    try:
        board = generate_leaderboard(
            config=config,
            tracking_dir=args.tracking_dir,
            report_path=None if args.no_report else args.report_path,
        )
    except LeaderboardError as exc:
        print(f"leaderboard error: {exc}")
        return 1

    print(describe_leaderboard(board))
    if board.top is not None:
        print(
            f"top model: {board.top.model_name} "
            f"({board.top.model_type}) {PRIMARY_METRIC}="
            f"{board.top.primary_metric:.4f}"
        )
    else:
        print("no benchmarked models matched")
    if board.report_path is not None:
        print(f"report: {board.report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
