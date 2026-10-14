"""Treated final-battery generation + competitive ship-path guard (M006 S03/T03).

Why this module exists
----------------------
MEM077/D039 fix the acceptance semantics of the treatment gate: when the gate
ACCEPTS a (candidate, model_type) pair, the claimed treated performance must
be turned into a real, coherent, store-backed battery generation before any
report, ranking, or serving decision consumes it — not hand-assembled into a
report row. This module owns that full-battery re-run:

- :func:`run_treated_final_generation` re-runs the FULL committed battery as
  ONE coherent ``run_kind=final`` generation tagged ``treatment_generation``
  (plus ``treatment_variant`` on the treated builds): every accepted pair is
  built through the declared treated builder
  (:func:`heart.eval.treatment_variants.build_candidate` — the SAME builder
  the gate measured), fitted on the full v1 train frame, and evaluated on the
  held-out test split through :func:`heart.eval.contract.evaluate`; every
  remaining battery member is refit from the committed tuned params of
  ``reports/battery-ledger.json`` (same evaluate() path, same test split,
  identical tags minus ``treatment_variant``). Superseded final runs are
  soft-deleted after a successful coherent generation (MLflow retains the
  deleted rows for provenance) so the leaderboard read-back carries exactly
  one coherent generation.
- :func:`clear_stale_selection_annotations` implements the S06 annotation
  hygiene trap: before any winner re-annotation, every ``s6.*`` tag still
  present on prior final runs is deleted, so
  ``heart.reporting.leaderboard``'s one-annotated-row invariant can never see
  two annotated rows after a generation change.
- :func:`regenerate_leaderboard_report` regenerates ``reports/leaderboard.md``
  through the leaderboard module's own CLI semantics (no re-implemented
  rendering).
- :func:`run_competitive_ship_path` wires the competitive branch (only when a
  regenerated leaderboard puts a treated model type on top) to the EXISTING
  selection/serving machinery — it runs no new statistics of its own.
- :func:`declared_coverage` / :func:`coverage_of` expose the guard's evidence
  surface: the union of the measurable (non-incumbent) candidates'
  ``applies_to`` sets. The committed selection winner ``random-forest`` is
  outside that union, so a non-competitive acceptance can only ever update
  reports and standings — the guard records exactly that with the written
  claim.

Honesty rules pinned here (D039): a treated final run is a fresh ``contract.
evaluate()`` number on the same held-out split, never a projection of gate
payloads; nothing is soft-deleted before the replacing generation is fully
logged; and no branch decision is silent — every disposition produced by
:func:`heart.eval.treatment_gate_real.apply_acceptances` names its branch,
its artifacts, and its written reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import pandas as pd

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.eval.contract import evaluate, evaluation_split_from_frames
from heart.eval.treatment_declared import (
    DECLARED_CANDIDATES,
    MLFLOW_TAG_TREATMENT_VARIANT,
    TreatmentCandidate,
)
from heart.eval.treatment_variants import (
    TreatmentGateError,
    build_candidate,
    candidate_by_key,
)
from heart.eval.treatment_gate_real import (
    CommittedParamsError,
    load_committed_fit_params,
)
from heart.models.registry import BATTERY_MODELS, build_pipeline
from heart.runtime import merge_effective_params
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import FINAL_RUN_KIND, MODEL_TYPE_TAG, RUN_KIND_TAG

logger = logging.getLogger(__name__)

__all__ = [
    "COMPETITIVE_SHIP_REASONING",
    "GENERATION_NOTE_TAG",
    "GENERATION_TAG",
    "NON_COMPETITIVE_ACCEPTANCE_REASONING",
    "NO_ACCEPTANCE_REASONING",
    "AcceptedPair",
    "CompetitiveGuardError",
    "TreatedFinalError",
    "TreatedGenerationResult",
    "accepted_model_types",
    "accepted_pairs_from_keys",
    "clear_stale_selection_annotations",
    "copy_prior_winner_annotations",
    "coverage_of",
    "declared_coverage",
    "inherit_prior_generation_provenance",
    "regenerate_leaderboard_report",
    "run_competitive_ship_path",
    "run_treated_final_generation",
]

#: The MLflow tag that makes a set of final runs one treated generation.
GENERATION_TAG: Final[str] = "treatment_generation"

#: The MLflow tag recording the generation's provenance (what produced it).
GENERATION_NOTE_TAG: Final[str] = "battery_note"

#: Human-readable page anchor for the non-competitive guard disposition.
NON_COMPETITIVE_ACCEPTANCE_REASONING: Final[str] = (
    "competitive guard: the regenerated leaderboard's winner model type "
    "'{model_type}' is not in any declared candidate's applies_to set "
    "(declared coverage {coverage}); accepted treatment(s) {treated} update "
    "reports and standings only and the serving artifact stays the "
    "committed winner's — ship path not triggered."
)

#: Human-readable page anchor for the no-acceptance branch disposition.
NO_ACCEPTANCE_REASONING: Final[str] = (
    "no-acceptance: all {measured} measured pair(s) were rejected at "
    "alpha={alpha} with min_delta={min_delta} (accepted count {accepted}); "
    "no treated final generation was run and the ship path not triggered."
)

#: Human-readable page anchor for the competitive trigger.
COMPETITIVE_SHIP_REASONING: Final[str] = (
    "competitive trigger: the regenerated leaderboard's winner model type "
    "'{model_type}' IS covered by the accepted treatment(s) {treated}, so "
    "the ship path executed through the existing selection/serving "
    "machinery (recalibrate, re-serialize the serving artifact, re-verify)."
)

#: How a treated final run's ``model_name`` marks its treatment build
#: (the model_type/family tags stay the member's own; the model name is the
#: human-readable marker the leaderboard renders).
TREATED_MODEL_NAME_TEMPLATE: Final[str] = (
    "{model_name} [treatment:{variant_key}]"
)


#: The provenance tags an Optuna-tuned generation records on its final runs
#: (cv geometry + tuning budget). A refit-from-committed-params generation
#: (refresh or treated) produces the SAME committed numbers but cannot
#: self-declare the tuning budget it inherited — so the values are copied
#: over from the superseded Optuna runs (provenance inheritance, never a
#: re-measurement).
PROVENANCE_TAG_KEYS: Final[tuple[str, ...]] = ("cv_folds", "n_trials")


class TreatedFinalError(Exception):
    """The treated final generation was requested in an unusable shape."""


class CompetitiveGuardError(TreatedFinalError):
    """The competitive ship path is required but cannot be executed honestly.

    Raised when a regenerated leaderboard puts a treated (accepted-coverage)
    model type on top and no ship executor is wired: the branch MUST run the
    selection/serving machinery and MUST NOT be recorded as 'not triggered'.
    """


@dataclass(frozen=True)
class AcceptedPair:
    """One gate-accepted (candidate, model_type) pair entering the branch."""

    variant_key: str
    model_type: str

    @property
    def treated_key(self) -> str:
        return f"treatment:{self.variant_key}/{self.model_type}"

    def to_dict(self) -> dict[str, str]:
        return {
            "variant_key": self.variant_key,
            "model_type": self.model_type,
            "treated_key": self.treated_key,
        }


def declared_coverage() -> frozenset[str]:
    """Union of the MEASURABLE declared candidates' ``applies_to`` sets.

    The documented incumbent (rung 0) is excluded on purpose: it is never
    measured and can never be accepted, so counting its battery-wide scope
    would claim coverage the gate cannot grant. The result is the model-type
    set the competitive guard actually judges the winner against.
    """
    return frozenset(
        model_type
        for candidate in DECLARED_CANDIDATES
        if not candidate.is_incumbent
        for model_type in candidate.applies_to
    )


def coverage_of(model_type: str) -> tuple[bool, tuple[str, ...]]:
    """Which measurable candidates' ``applies_to`` sets cover ``model_type``.

    Returns ``(covered, variant_keys)`` — the variant keys of every declared
    non-incumbent candidate whose scope includes ``model_type``. This is the
    written claim the non-competitive guard disposition carries as evidence.
    """
    keys = tuple(
        candidate.variant_key
        for candidate in DECLARED_CANDIDATES
        if not candidate.is_incumbent and model_type in candidate.applies_to
    )
    return bool(keys), keys


def accepted_pairs_from_keys(
    accepted_treated_keys: Iterable[str]
    | Iterable[AcceptedPair]
    | Iterable[TreatmentCandidate]
    | Iterable[tuple[str, str]],
) -> tuple[AcceptedPair, ...]:
    """Normalize accepted-pair inputs into validated :class:`AcceptedPair`.

    Accepts treated keys (``treatment:<variant>/<model_type>``), tight
    ``(variant_key, model_type)`` tuples, or already-validated
    :class:`AcceptedPair` objects (re-validated against the declared
    candidates). Every pair is validated against the
    declared candidates: the variant must exist, must NOT be the incumbent
    (rung 0 is never measurable/acceptable), and the model type must be in
    the candidate's declared ``applies_to``. Any violation is loud — a
    silently narrowed acceptance list would misstate the branch.
    """
    pairs: list[AcceptedPair] = []
    seen: set[str] = set()
    for raw in accepted_treated_keys:
        if isinstance(raw, TreatmentCandidate):
            raise TreatedFinalError(
                f"accepted pair input of type {type(raw).__name__} is a bare "
                "candidate; pass (variant_key, model_type) tuples or treated "
                "'treatment:<variant>/<model_type>' keys."
            )
        if isinstance(raw, AcceptedPair):
            variant_key, model_type = raw.variant_key, raw.model_type
        elif isinstance(raw, str):
            if not raw.startswith("treatment:") or "/" not in raw:
                raise TreatedFinalError(
                    f"accepted treated key {raw!r} is not a "
                    "'treatment:<variant>/<model_type>' key."
                )
            variant_key, model_type = raw[len("treatment:"):].split("/", 1)
        elif isinstance(raw, tuple) and len(raw) == 2:
            variant_key, model_type = str(raw[0]), str(raw[1])
        else:
            raise TreatedFinalError(
                f"accepted pair entry {raw!r} must be a treated key string or "
                f"a (variant_key, model_type) tuple, got "
                f"{type(raw).__name__}."
            )
        pair = AcceptedPair(variant_key=variant_key, model_type=model_type)
        try:
            candidate = candidate_by_key(pair.variant_key)
        except TreatmentGateError as exc:
            raise TreatedFinalError(
                f"Accepted pair {pair.treated_key!r} names an unknown variant: "
                f"{exc}"
            ) from exc
        if candidate.is_incumbent:
            raise TreatedFinalError(
                f"Accepted pair {pair.treated_key!r} names the documented "
                "incumbent (rung 0); the incumbent is never measured and can "
                "never be accepted."
            )
        if pair.model_type not in candidate.applies_to:
            raise TreatedFinalError(
                f"Accepted pair {pair.treated_key!r} is outside the variant's "
                f"declared applies_to {list(candidate.applies_to)}."
            )
        if pair.treated_key in seen:
            raise TreatedFinalError(
                f"Duplicate accepted pair {pair.treated_key!r}; one entry per "
                "accepted pair."
            )
        seen.add(pair.treated_key)
        pairs.append(pair)
    return tuple(pairs)


def accepted_model_types(pairs: Sequence[AcceptedPair]) -> frozenset[str]:
    """The model types the accepted pairs re-run treated (one build each)."""
    return frozenset(pair.model_type for pair in pairs)


# ---------------------------------------------------------------------------
# The coherent treated final generation (D039)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreatedGenerationResult:
    """The run ids of one coherent treated full-battery generation."""

    generation_id: str
    run_ids: dict[str, str]
    treated_run_ids: dict[str, str]
    superseded_run_ids: tuple[str, ...]

    @property
    def model_types(self) -> tuple[str, ...]:
        return tuple(self.run_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "run_ids": dict(self.run_ids),
            "treated_run_ids": dict(self.treated_run_ids),
            "superseded_run_ids": list(self.superseded_run_ids),
        }


def run_treated_final_generation(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    accepted_pairs: Sequence[AcceptedPair],
    *,
    split_version: str,
    ledger_path: str | Path,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    members: Sequence[str] | None = None,
) -> TreatedGenerationResult:
    """Re-run the FULL battery as one coherent treated ``run_kind=final`` gen.

    Per D039 (MEM077): every battery member is re-logged once under ONE
    fresh ``treatment_generation`` id —

    - accepted pairs' members build through the DECLARED treated builder
      (:func:`heart.eval.treatment_variants.build_candidate`, the same one
      the gate measured) with accepted pair's committed tuned params,
    - every remaining member refits from the ledger's committed best_params
      (narrowed to the spec's tuned space keys exactly like
      :func:`heart.eval.treatment_gate_real.refresh_battery_generation`),

    all fitted on the full train frame and evaluated ONCE on the held-out
    test split via :func:`heart.eval.contract.evaluate` (canonical metric
    dict, schema-checked at log time). Runs are tagged
    ``run_kind=final`` / ``model_type`` / ``split_version`` /
    ``treatment_generation`` with the ``treatment_variant`` tag only on the
    treated builds. After every member logged successfully, superseded final
    runs are soft-deleted (MLflow keeps deleted rows for provenance),
    mirroring the refresh semantics.

    ``members`` exists ONLY for tmp-store tests: it restricts the members the
    generation re-logs (coherence is then judged over that subset). The real
    orchestrator passes ``None`` (the full battery).

    Raises
    ------
    TreatedFinalError
        Empty/out-of-scope members, two accepted pairs for one model type,
        or a fit/evaluate failure that aborted the generation (with the
        member named).
    CommittedParamsError
        The committed ledger cannot supply every member's tuned params.
    """
    if not isinstance(train_frame, pd.DataFrame) or not isinstance(
        test_frame, pd.DataFrame
    ):
        raise TreatedFinalError(
            f"train_frame/test_frame must be pandas DataFrames, got "
            f"{type(train_frame).__name__} and {type(test_frame).__name__}."
        )
    pairs = accepted_pairs_from_keys(accepted_pairs)
    if not pairs:
        raise TreatedFinalError(
            "run_treated_final_generation requires at least one accepted "
            "(candidate, model_type) pair; with zero acceptances the "
            "no-acceptance branch logs NO generation (see apply_acceptances)."
        )
    re_run: dict[str, AcceptedPair] = {}
    for pair in pairs:
        if pair.model_type in re_run:
            raise TreatedFinalError(
                f"Two accepted pairs reuse model type {pair.model_type!r} "
                f"({re_run[pair.model_type].variant_key!r} and "
                f"{pair.variant_key!r}); one coherent final run per member — "
                "the pair candidates must be narrowed first."
            )
        re_run[pair.model_type] = pair
    member_types = [spec.model_type for spec in BATTERY_MODELS]
    if members is None:
        selected = member_types
    else:
        selected = list(members)
        unknown = [m for m in selected if m not in member_types]
        if unknown:
            raise TreatedFinalError(
                f"members {unknown} are not battery members {member_types}."
            )
        if not selected:
            raise TreatedFinalError("members must be a non-empty battery subset.")
    accepted_types = accepted_model_types(pairs)
    outside = sorted(accepted_types - set(selected))
    if outside:
        raise TreatedFinalError(
            f"accepted pair(s) target model type(s) {outside} that the "
            "generation does not re-run (members="
            f"{selected}); the treated build would be silently dropped."
        )
    committed = load_committed_fit_params(ledger_path)
    test_split = evaluation_split_from_frames(test_frame)
    generation_id = (
        "treatment-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-final"
    )
    run_ids: dict[str, str] = {}
    treated_run_ids: dict[str, str] = {}
    for model_type in selected:
        spec = next(s for s in BATTERY_MODELS if s.model_type == model_type)
        tuned = {
            key: value
            for key, value in committed[model_type].items()
            if key in set(spec.space.keys)
        }
        is_treated = model_type in accepted_types
        if is_treated:
            pair = re_run[model_type]
            pipeline = build_candidate(
                candidate_by_key(pair.variant_key), model_type, tuned
            )
            model_name = TREATED_MODEL_NAME_TEMPLATE.format(
                model_name=spec.model_name, variant_key=pair.variant_key
            )
        else:
            pipeline = build_pipeline(spec, tuned)
            model_name = spec.model_name
        try:
            pipeline.fit(
                train_frame[list(FEATURE_COLUMNS)], train_frame[TARGET_COLUMN]
            )
            metrics = evaluate(pipeline, test_split)
            result = log_evaluation_run(
                metrics,
                model_name=model_name,
                split_version=split_version,
                params={
                    "battery": "treated-generation",
                    "builder": (
                        "treatment_variants.build_candidate"
                        if is_treated
                        else "registry.build_pipeline"
                    ),
                    "variant_key": (
                        re_run[model_type].variant_key if is_treated else "none"
                    ),
                    **{
                        f"param_{key}": value
                        for key, value in merge_effective_params(
                            spec.fixed_params, tuned
                        ).items()
                    },
                },
                tags={
                    RUN_KIND_TAG: FINAL_RUN_KIND,
                    MODEL_TYPE_TAG: model_type,
                    "family": spec.family,
                    GENERATION_TAG: generation_id,
                    GENERATION_NOTE_TAG: (
                        "treated full-battery generation under an accepted "
                        "treatment (D039): refit from committed battery "
                        "params, evaluate() on the held-out split"
                        + (
                            f"; treated build {pair.variant_key!r}"
                            if is_treated
                            else "; untreated committed refit"
                        )
                    ),
                    "slice": "S03",
                    MLFLOW_TAG_TREATMENT_VARIANT: (
                        re_run[model_type].variant_key if is_treated else "none"
                    ),
                },
                tracking_dir=tracking_dir,
                experiment_name=experiment_name,
            )
        except TreatedFinalError:
            raise
        except Exception as exc:
            raise TreatedFinalError(
                f"Treated final generation aborted at member {model_type!r} "
                f"({generation_id!r}): {type(exc).__name__}: {exc}"
            ) from exc
        run_ids[model_type] = result.run_id
        if is_treated:
            treated_run_ids[re_run[model_type].treated_key] = result.run_id
        logger.info(
            "Treated generation %s: %s member %s (%s) logged as %s "
            "(roc_auc=%.6f)",
            generation_id,
            "treated" if is_treated else "refit",
            model_type,
            model_name,
            result.run_id,
            float(metrics["roc_auc"]),
        )
    superseded = _soft_delete_superseded_final_runs(
        keep_run_ids=set(run_ids.values()),
        split_version=split_version,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        generation_id=generation_id,
    )
    return TreatedGenerationResult(
        generation_id=generation_id,
        run_ids=run_ids,
        treated_run_ids=treated_run_ids,
        superseded_run_ids=tuple(x[0] for x in superseded),
    )


def _soft_delete_superseded_final_runs(
    *,
    keep_run_ids: set[str],
    split_version: str,
    tracking_dir: str | Path | None,
    experiment_name: str,
    generation_id: str,
) -> list[tuple[str, str]]:
    """Soft-delete active final runs outside the freshly logged generation.

    Same semantics as the battery refresh: the leaderboard read-back
    validates EVERY matched run (not only latest-dedup winners), so a
    superseded generation must leave the ACTIVE view for the read-back to
    carry exactly one coherent generation. MLflow retains deleted rows for
    provenance. Returns ``[(run_id, model_type)]`` of every deletion.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from heart.reporting.leaderboard import resolve_tracking_uri

    tracking_uri = resolve_tracking_uri(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    superseded: list[tuple[str, str]] = []
    if experiment is not None:
        for run in client.search_runs([experiment.experiment_id], max_results=10000):
            tags = {str(k): str(v) for k, v in dict(run.data.tags).items()}
            if tags.get(RUN_KIND_TAG) != FINAL_RUN_KIND:
                continue
            if tags.get("split_version") != split_version:
                continue
            if str(run.info.run_id) in keep_run_ids:
                continue
            superseded.append((str(run.info.run_id), tags.get(MODEL_TYPE_TAG, "")))
    for run_id, model_type in superseded:
        client.delete_run(run_id)
        logger.info(
            "Treated generation %s: soft-deleted superseded %s run %s "
            "(MLflow retains the deleted row for provenance)",
            generation_id,
            model_type,
            run_id,
        )
    logger.info(
        "Treated generation %s complete: %d member run(s) under "
        "split_version=%r (%d superseded run(s) soft-deleted)",
        generation_id,
        len(keep_run_ids),
        split_version,
        len(superseded),
    )
    return superseded


# ---------------------------------------------------------------------------
# S06 annotation hygiene + leaderboard regeneration
# ---------------------------------------------------------------------------


def clear_stale_selection_annotations(
    *,
    split_version: str | None = None,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> list[dict[str, str]]:
    """Delete every ``s6.*`` selection tag still present on final runs.

    The S06 hygiene trap (verbatim from the slice plan): before any winner
    re-annotation, stale ``s6.*`` tags on prior final runs MUST be cleared
    with ``MlflowClient.delete_tag``, so the leaderboard's
    one-annotated-row invariant (``SelectionAnnotationsError`` never sees two
    annotated rows) holds after a generation change. Active final runs only
    (soft-deleted rows are invisible to the leaderboard read-back). Returns
    the ``{"run_id", "tag"}`` record of every deletion (logged, not silent).
    """
    from heart.reporting.leaderboard import (
        SELECTION_TAG_PREFIX,
        resolve_tracking_uri,
    )
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = resolve_tracking_uri(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    filters = [f"tags.{RUN_KIND_TAG} = '{FINAL_RUN_KIND}'"]
    if split_version is not None:
        filters.append(f"tags.split_version = '{split_version}'")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=" and ".join(filters),
        max_results=10000,
    )
    cleared: list[dict[str, str]] = []
    for run in runs:
        tags = dict(run.data.tags)
        annotation_keys = sorted(
            str(k) for k in tags if str(k).startswith(SELECTION_TAG_PREFIX)
        )
        for key in annotation_keys:
            client.delete_tag(str(run.info.run_id), key)
            cleared.append({"run_id": str(run.info.run_id), "tag": key})
    if cleared:
        logger.info(
            "Cleared %d stale s6.* selection annotation tag(s) from final "
            "runs before re-annotation: %s",
            len(cleared),
            cleared,
        )
    else:
        logger.info(
            "No stale s6.* selection annotation tags on final runs; nothing "
            "to clear."
        )
    return cleared


def restore_winner_annotations(
    *,
    expected_winner: str | None = None,
    split_version: str | None = None,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> dict[str, object]:
    """Re-tag the current winner run with its prior ``s6.*`` annotation set.

    A refresh (or treated) re-log of the committed battery soft-deletes the
    superseded final runs — including the run the S06 selection flow
    annotated. The regenerated board would then render WITHOUT the selection
    narrative even though the committed selection story is unchanged. This
    function finds the most recent annotated final run (ANY lifecycle stage,
    including soft-deleted provenance rows), reads its ``s6.winner``
    identity, and copies its complete ``s6.*`` field set onto that model
    type's CURRENT ACTIVE final run via ``MlflowClient.set_tag``. At most
    one run is ever annotated (exactly one copy target, after the caller ran
    :func:`clear_stale_selection_annotations`), so the leaderboard's
    one-annotated-row invariant holds.

    Note: the selection narrative's winner is the SERVING selection's winner
    (the S06 flow's own candidate set), which need not be the leaderboard's
    #1 row; the copy follows the annotation's own identity, never the
    board's top row. ``expected_winner`` optionally asserts that identity —
    a mismatch is REPORTED (status ``annotation-winner-differs``), never
    silently copied: a changed selection story may only be re-annotated by
    the selection flow itself (the competitive ship path).

    Statuses returned in the evidence record:

    - ``"restored"`` — the prior annotation set was copied onto the current
      winner run (``copied_fields`` records every field),
    - ``"already-annotated"`` — the current winner run already carries the
      full required field set,
    - ``"annotation-winner-differs"`` — the prior annotation's winner does
      not match ``expected_winner``; nothing copied,
    - ``"no-prior-annotation"`` — no annotated final run exists anywhere in
      the store for this split,
    - ``"no-winner-run"`` — the annotation winner's model type has no
      active final run anymore (nothing to annotate).
    """
    import mlflow
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    from heart.reporting.leaderboard import (
        REQUIRED_SELECTION_FIELDS,
        SELECTION_TAG_PREFIX,
        resolve_tracking_uri,
    )

    tracking_uri = resolve_tracking_uri(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {"status": "no-prior-annotation", "annotation_winner": None}
    final_all = [
        run
        for run in client.search_runs(
            [experiment.experiment_id], run_view_type=ViewType.ALL, max_results=10000
        )
        if dict(run.data.tags).get(RUN_KIND_TAG) == FINAL_RUN_KIND
        and (
            split_version is None
            or dict(run.data.tags).get("split_version") == split_version
        )
    ]
    annotated_prior = [
        run
        for run in sorted(
            final_all,
            key=lambda r: (r.info.start_time or 0, str(r.info.run_id)),
            reverse=True,
        )
        if any(str(k).startswith(SELECTION_TAG_PREFIX) for k in dict(run.data.tags))
    ]
    if not annotated_prior:
        return {
            "status": "no-prior-annotation",
            "annotation_winner": None,
            "split_version": split_version,
        }
    # The annotated rows must agree on exactly one selection winner — the
    # same ambiguity the leaderboard's own read-back refuses.
    winners = {
        str(dict(run.data.tags)[f"{SELECTION_TAG_PREFIX}winner"])
        for run in annotated_prior
        if f"{SELECTION_TAG_PREFIX}winner" in dict(run.data.tags)
    }
    if not winners:
        return {
            "status": "no-prior-annotation",
            "annotation_winner": None,
            "split_version": split_version,
        }
    if len(winners) > 1:
        raise TreatedFinalError(
            f"{len(annotated_prior)} annotated final run(s) in the store "
            f"disagree on the selection winner {sorted(winners)}; the "
            "selection narrative cannot be restored mechanically — re-run "
            "the selection flow or repair the annotations explicitly."
        )
    annotation_winner = next(iter(winners))
    if expected_winner is not None and expected_winner != annotation_winner:
        return {
            "status": "annotation-winner-differs",
            "annotation_winner": annotation_winner,
            "expected_winner": expected_winner,
        }
    targets = sorted(
        (
            run
            for run in final_all
            if run.info.lifecycle_stage == "active"
            and dict(run.data.tags).get(MODEL_TYPE_TAG) == annotation_winner
        ),
        key=lambda r: (r.info.start_time or 0, str(r.info.run_id)),
        reverse=True,
    )
    if not targets:
        return {
            "status": "no-winner-run",
            "annotation_winner": annotation_winner,
            "split_version": split_version,
        }
    target = targets[0]
    present = {
        str(k)
        for k in dict(target.data.tags)
        if str(k).startswith(SELECTION_TAG_PREFIX)
    }
    if all(
        f"{SELECTION_TAG_PREFIX}{key}" in present
        for key in REQUIRED_SELECTION_FIELDS
    ):
        return {
            "status": "already-annotated",
            "annotation_winner": annotation_winner,
            "run_id": str(target.info.run_id),
        }
    source_fields = {
        key: str(value)
        for key, value in dict(annotated_prior[0].data.tags).items()
        if key.startswith(SELECTION_TAG_PREFIX)
    }
    for key, value in sorted(source_fields.items()):
        client.set_tag(str(target.info.run_id), key, value)
    logger.info(
        "Restored %d s6.* annotation field(s) onto winner run %s (%s) from "
        "prior annotated run %s — the regenerated leaderboard re-renders "
        "the unchanged selection narrative from the store alone",
        len(source_fields),
        target.info.run_id,
        annotation_winner,
        annotated_prior[0].info.run_id,
    )
    return {
        "status": "restored",
        "annotation_winner": annotation_winner,
        "target_run_id": str(target.info.run_id),
        "source_run_id": str(annotated_prior[0].info.run_id),
        "copied_fields": source_fields,
    }


def inherit_prior_generation_provenance(
    *,
    split_version: str | None = None,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tag_keys: Sequence[str] = PROVENANCE_TAG_KEYS,
) -> list[dict[str, str]]:
    """Copy tuning-budget provenance tags from the superseded generation.

    A refit-from-committed-params generation (the battery refresh or the
    treated re-run) reproduces the Optuna generation's numbers but logs no
    ``cv_folds``/``n_trials`` provenance of its own (it re-fit, it did not
    re-tune). The leaderboard's run-provenance suffix (``5-fold CV, 50
    trial(s)``) reads exactly those tags, so they are inherited: every ACTIVE
    final run missing a tag key gets it copied from the newest prior final
    run of the SAME model type (any lifecycle stage) that records it. Pure
    provenance inheritance; no metric or param is touched.
    """
    import mlflow
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    from heart.reporting.leaderboard import resolve_tracking_uri

    tracking_uri = resolve_tracking_uri(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    filters = [f"tags.{RUN_KIND_TAG} = '{FINAL_RUN_KIND}'"]
    if split_version is not None:
        filters.append(f"tags.split_version = '{split_version}'")
    active = client.search_runs(
        [experiment.experiment_id],
        filter_string=" and ".join(filters),
        max_results=10000,
    )
    all_final = client.search_runs(
        [experiment.experiment_id],
        run_view_type=ViewType.ALL,
        max_results=10000,
    )
    history: dict[str, list[object]] = {}
    for run in sorted(
        all_final, key=lambda r: (r.info.start_time or 0, str(r.info.run_id)),
        reverse=True,
    ):
        tags = dict(run.data.tags)
        if tags.get(RUN_KIND_TAG) != FINAL_RUN_KIND:
            continue
        model_type = tags.get(MODEL_TYPE_TAG, "")
        history.setdefault(model_type, []).append(run)
    inherited: list[dict[str, str]] = []
    for run in active:
        tags = dict(run.data.tags)
        model_type = str(tags.get(MODEL_TYPE_TAG, ""))
        for key in tag_keys:
            if key in tags and str(tags[key]).strip():
                continue
            prior_value = None
            for prior in history.get(model_type, []):
                prior_tags = dict(prior.data.tags)
                value = prior_tags.get(key)
                if value is not None and str(value).strip():
                    prior_value = value
                    break
            if prior_value is None:
                continue
            client.set_tag(str(run.info.run_id), key, str(prior_value))
            inherited.append(
                {
                    "run_id": str(run.info.run_id),
                    "model_type": model_type,
                    "tag": key,
                    "value": str(prior_value),
                }
            )
    if inherited:
        logger.info(
            "Inherited %d provenance tag(s) (keys=%s) from the superseded "
            "generation onto the re-logged final runs: %s",
            len(inherited),
            list(tag_keys),
            inherited,
        )
    else:
        logger.info(
            "Provenance inheritance: nothing to inherit (keys=%s already "
            "present or unrecorded in the store).", list(tag_keys)
        )
    return inherited


def regenerate_leaderboard_report(
    *,
    tracking_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    split_version: str | None = None,
) -> object:
    """Regenerate ``reports/leaderboard.md`` via the leaderboard CLI semantics.

    One direct call to the leaderboard module's own generation entry
    (:func:`heart.reporting.leaderboard.generate_leaderboard` — the same
    builder ``python -m heart.reporting.leaderboard`` runs): no re-rendered
    numbers, no hand-assembled rows, and the returned board is what became
    the report. Note: the renderer stamps a fresh "generated at" line, so a
    byte-identical report vs an earlier rendering is not expected; the ROW
    content derives from the store alone.
    """
    from heart.reporting.leaderboard import (
        DEFAULT_REPORT_PATH,
        LeaderboardConfig,
        generate_leaderboard,
    )

    destination = DEFAULT_REPORT_PATH if report_path is None else Path(report_path)
    board = generate_leaderboard(
        config=LeaderboardConfig(split_version=split_version),
        tracking_dir=tracking_dir,
        report_path=destination,
    )
    logger.info(
        "Regenerated leaderboard report %s (%d model(s), top=%s; the "
        "'generated at' line is re-stamped, rows derive from the store)",
        destination,
        board.n_models,
        board.top.model_type if board.top is not None else None,
    )
    return board


# ---------------------------------------------------------------------------
# The competitive ship path (only reachable when a treated model tops the board)
# ---------------------------------------------------------------------------


def run_competitive_ship_path(
    *,
    split_version: str,
    tracking_dir: str | Path | None = None,
) -> dict[str, object]:
    """Execute the competitive ship path through the EXISTING machinery.

    Triggered only when the regenerated leaderboard's winner is a treated
    (accepted-coverage) model type. What runs — nothing below is new
    statistics:

    1. the S06 selection flow
       (:func:`heart.eval.selection.generate_real_selection_report`,
       ``selection.md`` + winner re-annotation on the new winning run — the
       fresh selection's paired significance vs baseline/runner-up), and
    2. the serving serialization machinery
       (:func:`heart.serving.artifact.build_winner_artifact`, recalibration +
       re-serialized artifact the prediction page serves).

    Documented caveat (the executor records it in its evidence): the S06
    machinery rebuilds its selection candidates from the registry at
    committed default parameters; the treated-build evidence itself lives in
    the regenerated ``final`` generation's runs. A served build that must be
    the treated object (not merely the treated model type) needs the
    artifact machinery to accept treated candidate factories — that wiring is
    the follow-up this path's evidence records; the guard NEVER silently
    converts that gap into a shipped artifact.
    """
    from heart.eval.selection import generate_real_selection_report
    from heart.serving.artifact import build_winner_artifact

    logger.info(
        "Competitive ship path: running the S06 selection flow (split %r).",
        split_version,
    )
    decision, selection_path = generate_real_selection_report(
        split_version=split_version
    )
    artifact, artifact_path = build_winner_artifact(split_version=split_version)
    evidence: dict[str, object] = {
        "machinery": (
            "heart.eval.selection.generate_real_selection_report + "
            "heart.serving.artifact.build_winner_artifact"
        ),
        "selection_report_path": str(selection_path),
        "selection_winner": decision.winner,
        "selection_ship": bool(decision.ship),
        "artifact_path": str(artifact_path),
        "artifact_model_name": artifact.metadata.model_name,
        "caveat": (
            "the S06 machinery rebuilds selection candidates from the "
            "registry at committed defaults; the treated-build evidence "
            "lives in the regenerated final-generation runs. A served build "
            "that must BE the treated object is the recorded follow-up — "
            "never silently converted into a shipped artifact."
        ),
    }
    logger.info(
        "Competitive ship path executed: selection.md=%s winner=%s "
        "ship=%s artifact=%s",
        selection_path,
        decision.winner,
        decision.ship,
        artifact_path,
    )
    return evidence
