"""Real-store treatment-gate plumbing (M006 S03; D037-D039).

S02's gate runner (:mod:`heart.eval.treatment_gate`) measures the declared
candidates and logs them under ``split_version=GATE_SPLIT_VERSION``
(``"fixture"``) with fixture-scale geometry. This module carries the S03
real-store wiring around that runner:

- :func:`load_committed_fit_params` parses the committed battery ledger's
  tuned ``best_params`` into the ``{model_type: params}`` mapping the gate
  runner accepts as ``fit_params`` (for the real battery the QDA variance
  floor is the committed ``reg_param=0.20848923483175424``, NOT the fixture
  ``0.1`` — MEM100's ``FoldFitError`` awaits any caller who substitutes).
- :func:`build_real_standings` reads the authoritative leaderboard
  (``run_kind=final``, ``split_version=v1``, latest-only per model) from the
  real MLflow store — via the leaderboard module's own collection helpers so
  the ordering/tiebreak policy has exactly one implementation — and returns
  the clause-(c) battery standings keyed ``incumbent:<model_type>``.
- :func:`standings_for_pair` supplies the per-pair standings mapping clause
  (c) compares, entering the treated variant at its incumbent model's
  standing (D038; see that function's docstring for the reasoning verbatim).

The gate runner itself stays the single measurement path: S03's real command
calls :func:`heart.eval.treatment_gate.run_gate` with ``split_version="v1"``,
``fit_params=load_committed_fit_params()``, and clause-(c) standings wired per
measured pair (each pair's two mandatory entries supplied through
:func:`standings_for_pair`). Nothing here re-implements a clause, a threshold,
or a rank derivation.

T02 adds the real-data gate command itself: :func:`run_real_gate` loads the
pinned dataset and the v1 split frames, wires committed params and per-pair
clause-(c) standings, and runs the gate runner once with
``log_to_mlflow=True`` (48 measured pairs, one MLflow run each). Because a
stale generation (a battery logged before the current metric contract, or
missing v1 final runs) makes the authoritative leaderboard read-back — and
therefore clause (c) — unreadable, the command's setup step re-logges the
committed battery coherently through the same evaluate() path
(:func:`refresh_battery_generation`) before the gate measures anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

from heart.data.load import load_dataset
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION as _DECLARED_SPLIT_VERSION
from heart.data.split import load_split_frames
from heart.eval.contract import evaluate, evaluation_split_from_frames
from heart.eval.treatment_declared import (BATTERY_MEMBERS, DECLARED_CANDIDATES,
    TreatmentCandidate)
from heart.eval.treatment_declared import (
    GATE_MIN_DELTA_ROC_AUC,
    PAIRED_SIGNIFICANCE_ALPHA,
)
from heart.eval.treatment_gate import (
    VERDICT_ACCEPTED,
    CandidateMeasurement,
    GateConfig,
    GateReport,
    TreatmentGateError,
    incumbent_key_for,
    run_gate,
    treatment_key_for,
)
from heart.models.registry import BATTERY_MODELS, build_pipeline
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT
from heart.reporting.leaderboard import (
    DEFAULT_RUN_KIND,
    LeaderboardConfig,
    resolve_tracking_uri,
    select_leaderboard_rows,
)
from heart.reporting.leaderboard import LeaderboardError as _LeaderboardError
from heart.runtime import merge_effective_params
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import MODEL_TYPE_TAG, RUN_KIND_TAG

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BATTERY_LEDGER_PATH",
    "REAL_SPLIT_VERSION",
    "AcceptanceDisposition",
    "CommittedParamsError",
    "PairKeys",
    "RealStandingsError",
    "TreatmentGateRealError",
    "apply_acceptances",
    "apply_acceptances_from_store",
    "build_real_standings",
    "declared_pairs",
    "load_committed_fit_params",
    "refresh_battery_generation",
    "run_real_gate",
    "standings_for_pair",
]

#: The real declared split version the S03 gate measures and logs under.
#: Mirrors :data:`heart.data.split.SPLIT_VERSION` by import (not by copy), so
#: the stamp can never silently drift from the committed split machinery.
REAL_SPLIT_VERSION: Final[str] = _DECLARED_SPLIT_VERSION

#: The committed battery ledger produced by the tuning battery; the real
#: gate's ``fit_params`` come from here, never from declared defaults.
DEFAULT_BATTERY_LEDGER_PATH: Final[Path] = (
    Path("reports") / "battery-ledger.json"
)

#: Module path the verdicts artifact names as the command that produced the
#: stored gate generation (set once here; ``gate_verdicts`` imports it).
REAL_GATE_COMMAND: Final[str] = "heart.eval.treatment_gate_real.run_real_gate"


class TreatmentGateRealError(Exception):
    """Base class for the S03 real-store gate plumbing errors."""


class CommittedParamsError(TreatmentGateRealError):
    """The committed battery ledger is missing, malformed, or incomplete.

    The real gate must fit every battery member with the ledger's tuned
    ``best_params``; anything less would measure candidates against an
    incumbent the battery never committed, so the failure is loud and named.
    """


@dataclass(frozen=True)
class PairKeys:
    """The two mandatory clause-(c) gate keys for one (candidate, model_type).

    The real command wires per-pair standings BEFORE each pair is measured
    (the treated entry inherits its incumbent model's standing, so it needs
    no measurement to exist yet). :func:`standings_for_pair` therefore
    accepts either a real
    :class:`heart.eval.treatment_gate.CandidateMeasurement` or this record --
    both expose exactly the ``treated_key``/``incumbent_key`` identities the
    comparator consumes. Keys are derived deterministically from the
    declaration via :func:`treatment_key_for` / :func:`incumbent_key_for`.
    """

    treated_key: str
    incumbent_key: str


class RealStandingsError(TreatmentGateRealError):
    """The authoritative leaderboard read-back cannot supply the standings.

    Raised when a requested battery member is not on the
    ``run_kind=final`` / ``split_version=v1`` leaderboard, when a requested
    model type is not a battery member, or when a standings entry is
    malformed. Clause (c) is never evaluated against invented standings.
    """


# ---------------------------------------------------------------------------
# Committed tuned params (the battery ledger -> gate fit_params mapping)
# ---------------------------------------------------------------------------


def load_committed_fit_params(
    ledger_path: str | Path = DEFAULT_BATTERY_LEDGER_PATH,
) -> dict[str, dict[str, object]]:
    """Parse the committed battery ledger into the gate's ``fit_params``.

    The ledger's ``results[]`` entries carry ``model_type`` and the tuning
    battery's winning ``best_params`` per member. This loader merges each
    member's effective params into ``{model_type: params dict}`` — exactly
    the mapping :func:`heart.eval.treatment_gate.run_gate_core` accepts as
    ``fit_params`` (applied identically to the treated and the incumbent
    build of that member).

    Cross-checks (all loud, named :class:`CommittedParamsError`):

    - every ``results[].model_type`` must be a :data:`BATTERY_MEMBERS` member
      (the ledger covers the real battery only),
    - every battery member must be covered exactly once (a member with no
      committed params would silently measure against declared defaults),
    - every ``best_params`` must be a non-empty dict of JSON-scalar values
      (registry-buildable hyperparameters).

    Returns
    -------
    dict[str, dict[str, object]]
        ``{model_type: best_params}`` for all of :data:`BATTERY_MEMBERS`.
    """
    path = Path(ledger_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CommittedParamsError(
            f"Committed battery ledger not found at {path!s}; the real gate "
            "must fit every battery member with the ledger's tuned "
            "best_params, never declared defaults."
        ) from exc
    except json.JSONDecodeError as exc:
        raise CommittedParamsError(
            f"Committed battery ledger at {path!s} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        raise CommittedParamsError(
            f"Committed battery ledger at {path!s} is malformed: expected a "
            f"JSON object with a 'results' list, got "
            f"{type(raw).__name__} with results="
            f"{type(raw.get('results')).__name__ if isinstance(raw, dict) else 'n/a'}."
        )
    params: dict[str, dict[str, object]] = {}
    for index, entry in enumerate(raw["results"]):
        if not isinstance(entry, dict):
            raise CommittedParamsError(
                f"results[{index}] in {path!s} is {type(entry).__name__}; "
                "expected an object with 'model_type' and 'best_params'."
            )
        model_type = entry.get("model_type")
        best_params = entry.get("best_params")
        if not isinstance(model_type, str) or not model_type.strip():
            raise CommittedParamsError(
                f"results[{index}].model_type in {path!s} is "
                f"{model_type!r}; expected a non-empty battery member name."
            )
        if model_type not in BATTERY_MEMBERS:
            raise CommittedParamsError(
                f"results[{index}].model_type {model_type!r} in {path!s} is "
                f"not a battery member {list(BATTERY_MEMBERS)}; the gate's "
                "fit_params cover the committed battery only."
            )
        if model_type in params:
            raise CommittedParamsError(
                f"results[{index}] duplicates model_type {model_type!r} in "
                f"{path!s}; exactly one committed params entry per member."
            )
        if not isinstance(best_params, dict) or not best_params:
            raise CommittedParamsError(
                f"results[{index}].best_params for {model_type!r} in "
                f"{path!s} is {best_params!r}; expected a non-empty params "
                "dict (the member has no committed tuned params to fit)."
            )
        non_scalar = [
            key
            for key, value in best_params.items()
            if isinstance(value, (dict, list))
        ]
        if non_scalar:
            raise CommittedParamsError(
                f"results[{index}].best_params for {model_type!r} in "
                f"{path!s} carries non-scalar value(s) {sorted(non_scalar)}; "
                "registry-buildable hyperparameters are JSON scalars."
            )
        params[model_type] = dict(best_params)
    missing = [member for member in BATTERY_MEMBERS if member not in params]
    if missing:
        raise CommittedParamsError(
            f"Committed battery ledger at {path!s} lacks tuned best_params "
            f"for battery member(s) {missing}; the real gate measures every "
            "declared applies_to scope against committed params only."
        )
    return params


# ---------------------------------------------------------------------------
# Clause-(c) standings from the authoritative leaderboard read-back (D038)
# ---------------------------------------------------------------------------


def build_real_standings(
    model_types: Sequence[str],
    tracking_dir: str | Path | None = None,
    *,
    split_version: str = REAL_SPLIT_VERSION,
) -> dict[str, tuple[float, int]]:
    """Build the clause-(c) battery standings from the real MLflow store.

    Reads the leaderboard's authoritative collection —
    ``run_kind=final``, the supplied ``split_version`` (default ``"v1"``,
    D038), latest-only per model — through
    :func:`heart.reporting.leaderboard.select_leaderboard_rows` so the
    ordering policy (primary ROC-AUC descending, ``model_type`` ascending
    tiebreak) has exactly one implementation. Each requested model's entry is
    keyed ``incumbent:<model_type>`` (:func:`incumbent_key_for`) and carries
    ``(primary_metric, leaderboard_rank)`` with the FULL-board 1-based rank
    (lower is better): ranks are computed over every ranked row, so an entry
    requested here keeps its real position among all battery members.

    Raises
    ------
    RealStandingsError
        When ``model_types`` is empty, names a non-battery member, or names
        a battery member absent from the leaderboard read-back (loudly
        listing the missing members and the collection filters).

    Returns
    -------
    dict[str, tuple[float, int]]
        ``{"incumbent:<model_type>": (primary_roc_auc, rank)}``.
    """
    requested = [str(model_type) for model_type in model_types]
    if not requested:
        raise RealStandingsError(
            "build_real_standings requires at least one battery model type; "
            "clause (c) standings are built for the pairs being measured."
        )
    outside = [
        model_type
        for model_type in requested
        if model_type not in BATTERY_MEMBERS
    ]
    if outside:
        raise RealStandingsError(
            f"build_real_standings got non-battery model type(s) {outside}; "
            f"standings cover the committed battery {list(BATTERY_MEMBERS)} "
            "only."
        )
    config = LeaderboardConfig(
        run_kind=DEFAULT_RUN_KIND,
        split_version=split_version,
        latest_only=True,
        strict=False,
    )
    # select_leaderboard_rows returns the rows already in leaderboard order
    # (primary ROC-AUC desc, model_type tiebreak) — the rank IS the position.
    rows, _considered = select_leaderboard_rows(
        config=config, tracking_dir=tracking_dir
    )
    position = {
        row.model_type: (index, row) for index, row in enumerate(rows, start=1)
    }
    standings: dict[str, tuple[float, int]] = {}
    missing: list[str] = []
    for model_type in requested:
        hit = position.get(model_type)
        if hit is None:
            missing.append(model_type)
            continue
        rank, row = hit
        standings[incumbent_key_for(model_type)] = (
            row.primary_metric,
            int(rank),
        )
    if missing:
        raise RealStandingsError(
            f"split_version={split_version!r} run_kind={DEFAULT_RUN_KIND!r} "
            f"leaderboard read-back lacks battery member(s) {missing}; "
            "clause (c) standings must cover every requested member with its "
            "real rank — run the battery (run_kind=final, split_version=v1) "
            "first or point the caller at the right store."
        )
    return standings


def standings_for_pair(
    measurement: CandidateMeasurement | PairKeys,
    battery_standings: Mapping[str, tuple[float, int]],
) -> dict[str, tuple[float, int]]:
    """Supply the two mandatory clause-(c) standings entries for one pair.

    D038 (verbatim rationale): "Clause (c) per MEM099 judges supplied ranks;
    inventing a projected holdout rank from repeated-CV means would be
    dishonest and un-failable, so the treated variant enters the standings at
    its incumbent model's standing -- the standing it cannot have worsened at
    verdict time -- and the post-acceptance generation re-bench (paired
    significance vs baseline and runner-up + leaderboard regeneration)
    validates the standing claim with real holdout numbers."

    Concretely, the treated entry
    ``treatment:<variant>/<model_type>`` (:attr:`CandidateMeasurement.
    treated_key`) inherits its incumbent model's leaderboard standing — the
    ``(primary_mean, rank)`` carried by ``incumbent:<model_type>`` — because
    the treated variant has not earned a leaderboard position of its own yet:
    it is a re-fit of the SAME committed model under a treatment, and the
    post-acceptance generation re-bench (D039) is what establishes real
    holdout standings afterwards. With an inherited standing the treated rank
    equals the incumbent's rank, so
    :func:`heart.eval.treatment_gate.standing_not_worsened`'s own check
    (``treated_rank <= incumbent_rank``) holds at verdict time (an equal rank
    is not a worsening) while the clause stays independently failable: any
    caller supplying a treated rank worse than the incumbent's fails clause
    (c) alone. The leaderboard (with ties and its own ordering policy) stays
    authoritative; this function invents nothing and derives no rank.

    Raises
    ------
    RealStandingsError
        When ``battery_standings`` lacks the ``incumbent:<model_type>``
        entry, or that entry is not a ``(primary_mean, positive-int rank)``
        tuple — a malformed supply is a caller error, never a passable
        clause (:func:`standing_not_worsened` re-validates authoritatively).

    Returns
    -------
    dict[str, tuple[float, int]]
        Exactly the two mandatory keys clause (c) compares:
        ``{incumbent_key: (mean, rank), treated_key: (mean, rank)}`` with the
        treated entry's rank inherited from the incumbent's.
    """
    incumbent_key = measurement.incumbent_key
    if incumbent_key not in battery_standings:
        raise RealStandingsError(
            f"battery_standings lacks the incumbent entry {incumbent_key!r}; "
            "build it with build_real_standings(model_types) first — the "
            "treated entry inherits the incumbent model's standing (D038) "
            "and cannot be supplied without it."
        )
    incumbent_entry = battery_standings[incumbent_key]
    if (
        not isinstance(incumbent_entry, tuple)
        or len(incumbent_entry) != 2
        or isinstance(incumbent_entry[1], bool)
        or not isinstance(incumbent_entry[1], int)
        or incumbent_entry[1] < 1
    ):
        raise RealStandingsError(
            f"battery_standings entry for {incumbent_key!r} is "
            f"{incumbent_entry!r}; expected a (primary_mean, positive-int "
            "leaderboard rank) tuple."
        )
    incumbent_mean, incumbent_rank = incumbent_entry
    return {
        incumbent_key: (float(incumbent_mean), int(incumbent_rank)),
        measurement.treated_key: (float(incumbent_mean), int(incumbent_rank)),
    }


# ---------------------------------------------------------------------------
# Declared pair enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredPair:
    """One declared measurable (candidate, model_type) pair.

    Derived programmatically from :data:`DECLARED_CANDIDATES` (never
    hard-coded): every DECLARED candidate that is NOT the documented
    incumbent contributes one pair per member of its ``applies_to`` scope.
    The incumbent rung-0 candidate is documented-not-re-measured and never
    appears here.
    """

    candidate: TreatmentCandidate
    model_type: str

    @property
    def pair_keys(self) -> PairKeys:
        """The pair's mandatory clause-(c) gate keys."""
        return PairKeys(
            treated_key=treatment_key_for(self.candidate, self.model_type),
            incumbent_key=incumbent_key_for(self.model_type),
        )


def declared_pairs(
    candidate_keys: Sequence[str] | None = None,
) -> tuple[DeclaredPair, ...]:
    """Enumerate the declared measurable pairs in declaration order.

    ``candidate_keys`` optionally restricts the enumeration to a subset of
    declared variant keys; an unknown key is a loud caller error (the same
    contract the gate runner enforces) so a partially-specified gate never
    quietly shrinks itself.
    """
    known_keys = {candidate.variant_key for candidate in DECLARED_CANDIDATES}
    if candidate_keys is None:
        keys_filter: set[str] | None = None
    else:
        keys_filter = set(candidate_keys)
        unknown = keys_filter - known_keys
        if unknown:
            raise TreatmentGateError(
                f"candidate_keys naming unknown variant(s) {sorted(unknown)}; "
                f"declared keys are {sorted(known_keys)}."
            )
    pairs: list[DeclaredPair] = []
    for candidate in DECLARED_CANDIDATES:
        if candidate.is_incumbent:
            continue
        if keys_filter is not None and candidate.variant_key not in keys_filter:
            continue
        for model_type in candidate.applies_to:
            pairs.append(DeclaredPair(candidate=candidate, model_type=model_type))
    return tuple(pairs)


# ---------------------------------------------------------------------------
# Coherent battery-generation refresh (setup step for the standings read-back)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryRefreshResult:
    """The run ids of one coherent refit-from-committed-params generation."""

    battery_run_id: str
    run_ids: dict[str, str]


def refresh_battery_generation(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    split_version: str = REAL_SPLIT_VERSION,
    ledger_path: str | Path = DEFAULT_BATTERY_LEDGER_PATH,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> BatteryRefreshResult:
    """Re-log the committed battery coherently under the CURRENT contract.

    Setup step for the gate command (never a verify command): the real
    store's committed battery may predate the current metric contract (the
    M006 S02 schema bump added ``log_loss``; the stored generation lacks it),
    which makes the authoritative leaderboard read-back — and therefore
    clause (c) — raise regardless of what the gate measures. The honest
    remedy per the leaderboard's own failure guidance is to RE-LOG: refit
    every battery member from its committed tuned params (the SAME ledger
    the gate's ``fit_params`` come from) on the SAME v1 train frame, evaluate
    once on the SAME held-out test split through
    :func:`heart.eval.contract.evaluate`, and log one
    ``run_kind=final`` / ``split_version=v1`` run per member through the
    frozen :func:`heart.tracking.run.log_evaluation_run` convention. Nothing
    is hand-assembled and no threshold, split, or tuning choice changes; a
    refit from the same committed params on the same split reproduces the
    committed generation's numbers under the current code.

    The generation is COHERENT: one fresh ``battery_run_id`` is generated and
    tagged onto every member (mirroring the tuning battery's own
    ``battery_run_id`` convention), with a ``battery_note`` tag recording the
    refresh provenance so the store can tell a refreshed generation apart
    from an Optuna-tuned one. Any member failing its refit/evaluate/log
    sequence aborts the refresh LOUDLY before any of it is accepted: a
    partially-refreshed generation would flip ``latest_only`` dedup in only
    some members and poison the standings read.

    Returns
    -------
    BatteryRefreshResult
        The generation id and ``{model_type: run_id}`` for all members.

    Raises
    ------
    CommittedParamsError
        The ledger path is unusable for any member.
    TrackingBackendError / TrackingValueError
        The store rejected a metric dict, param, tag, or write.
    SplitContractError
        The frames do not expose the declared feature/target columns.
    Exception
        Any member's fit/evaluate failure (re-raised with the member named).
    """
    committed = load_committed_fit_params(ledger_path)
    test_split = evaluation_split_from_frames(test_frame)
    battery_run_id = (
        "battery-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-refresh"
    )
    run_ids: dict[str, str] = {}
    for spec in BATTERY_MODELS:
        # The ledger's best_params are MERGED effective params; the fitter
        # accepts only the tuned search-space keys (fixed params are pinned by
        # the spec itself), so the fit supply is narrowed and the logged
        # provenance keeps the full merged param set (exactly what the tuning
        # battery recorded).
        effective = committed[spec.model_type]
        tuned = {key: value for key, value in effective.items() if key in set(spec.space.keys)}
        params = merge_effective_params(spec.fixed_params, tuned)
        pipeline = build_pipeline(spec, tuned)
        pipeline.fit(train_frame[list(FEATURE_COLUMNS)], train_frame[TARGET_COLUMN])
        metrics = evaluate(pipeline, test_split)
        try:
            result = log_evaluation_run(
                metrics,
                model_name=spec.model_name,
                split_version=split_version,
                params=params,
                tags={
                    RUN_KIND_TAG: DEFAULT_RUN_KIND,
                    MODEL_TYPE_TAG: spec.model_type,
                    "family": spec.family,
                    "battery_run_id": battery_run_id,
                    "slice": "S03",
                    "battery_note": (
                        "schema-v2 refresh of the committed battery generation "
                        "(refit from reports/battery-ledger.json best_params; "
                        "untreated incumbent, pre-gate)"
                    ),
                },
                tracking_dir=tracking_dir,
                experiment_name=experiment_name,
            )
        except Exception as exc:
            raise TreatmentGateRealError(
                f"Battery refresh failed to LOG {spec.model_type!r} under the "
                f"current contract: {type(exc).__name__}: {exc}"
            ) from exc
        run_ids[spec.model_type] = result.run_id
        logger.info(
            "Battery refresh: re-logged %s (roc_auc=%.6f, log_loss present=%s) "
            "as run %s",
            spec.model_type,
            float(metrics["roc_auc"]),
            "log_loss" in metrics,
            result.run_id,
        )
    # The refreshed generation is now the coherent current one; the
    # superseded runs are soft-deleted (MLflow keeps them in the store DB for
    # provenance; the ACTIVE view carries exactly one generation) because the
    # leaderboard read-back validates EVERY matched run, not only the
    # latest-dedup winners — a stale schema would permanently poison the
    # clause-(c) standings read otherwise.
    from mlflow.tracking import MlflowClient as _MlflowClient

    client = _MlflowClient()
    tracking_uri = resolve_tracking_uri(tracking_dir=tracking_dir)
    client = _MlflowClient(tracking_uri=tracking_uri)
    refreshed_ids = set(run_ids.values())
    experiment = client.get_experiment_by_name(experiment_name)
    superseded: list[tuple[str, str, str]] = []
    if experiment is not None:
        for run in client.search_runs([experiment.experiment_id], max_results=10000):
            tags = {str(k): str(v) for k, v in dict(run.data.tags).items()}
            if tags.get(RUN_KIND_TAG) != DEFAULT_RUN_KIND:
                continue
            if tags.get("split_version") != split_version:
                continue
            if str(run.info.run_id) in refreshed_ids:
                continue
            superseded.append(
                (str(run.info.run_id), tags.get(MODEL_TYPE_TAG, ""), battery_run_id)
            )
    for run_id, model_type, generation in superseded:
        client.delete_run(run_id)
        logger.info(
            "Battery refresh: soft-deleted superseded %s run %s (replaced by "
            "coherent generation %s; MLflow retains the deleted row for "
            "provenance)",
            model_type,
            run_id,
            generation,
        )
    logger.info(
        "Battery refresh complete: coherent generation %s with %d member "
        "run(s) under split_version=%r (%d superseded run(s) soft-deleted)",
        battery_run_id,
        len(run_ids),
        split_version,
        len(superseded),
    )
    return BatteryRefreshResult(battery_run_id=battery_run_id, run_ids=run_ids)


# ---------------------------------------------------------------------------
# Standings readiness + the real-data gate command (T02)
# ---------------------------------------------------------------------------


def ensure_real_standings(
    model_types: Sequence[str],
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    split_version: str = REAL_SPLIT_VERSION,
    ledger_path: str | Path = DEFAULT_BATTERY_LEDGER_PATH,
) -> dict[str, tuple[float, int]]:
    """Read the clause-(c) standings, refreshing the battery if unreadable.

    The happy path is :func:`build_real_standings` untouched. When the
    authoritative leaderboard read-back cannot supply the requested members
    (a committed generation predating the current metric contract, or no
    ``split_version=v1`` generation at all), the command's setup step
    :func:`refresh_battery_generation` re-logs the committed battery coherently
    under the current contract and the read-back is attempted exactly once
    more; a second failure aborts loudly — the gate never judges clause (c)
    against stand-in rankings.
    """
    try:
        return build_real_standings(model_types, tracking_dir, split_version=split_version)
    except (RealStandingsError, _LeaderboardError) as primary:
        logger.info(
            "Leaderboard read-back unusable for clause-(c) standings (%s: %s); "
            "refreshing the committed battery generation under the current "
            "contract before the gate measures any pair.",
            type(primary).__name__,
            primary,
        )
    refresh_battery_generation(
        train_frame,
        test_frame,
        split_version=split_version,
        ledger_path=ledger_path,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
    )
    try:
        return build_real_standings(model_types, tracking_dir, split_version=split_version)
    except Exception as secondary:  # noqa: BLE001 - loud rewrap, never silent
        raise RealStandingsError(
            f"Leaderboard read-back still unusable after the coherent battery "
            f"refresh: {type(secondary).__name__}: {secondary}"
        ) from secondary


def run_real_gate(
    candidate_keys: Sequence[str] | None = None,
    *,
    config: GateConfig | None = None,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    ledger_path: str | Path = DEFAULT_BATTERY_LEDGER_PATH,
) -> GateReport:
    """Measure every declared candidate against the real data via one command.

    The D038 real-store gate run, end to end:

    1. load the pinned dataset (``allow_download=False``; digest enforced
       inside :func:`heart.data.load.load_dataset`),
    2. load the v1 train/test frames via
       :func:`heart.data.split.load_split_frames` and log the 734/184
       partition in the run preamble,
    3. build ``X`` from the train frame's declared feature columns and
       ``y`` from the target column,
    4. ``fit_params = load_committed_fit_params()`` (the members are fit with
       their committed tuned hyperparameters; MEM100), and
    5. run :func:`heart.eval.treatment_gate.run_gate` ONCE with
       ``log_to_mlflow=True``, ``split_version=REAL_SPLIT_VERSION``, and the
       per-pair clause-(c) standings union built through
       :func:`standings_for_pair` — one mandatory treated/incumbent entry pair
       per measured pair, assembled BEFORE measurement from the declarative
       keys, so no passable clause-(a)+(b) verdict is ever left standing_ok
       silently None.
    6. verify the stored generation through
       :func:`heart.reporting.data_methods.gate_verdicts.verify_gate_generation`
       (every declared pair logged exactly once under the real split, params
       matching the declared constants) — a loud local import is impossible:
       the reporting module is part of the same package.

    Returns the aggregate :class:`~heart.eval.treatment_gate.GateReport`
    (the union of every measured outcome with its MLflow run id).
    """
    from heart.reporting.data_methods.gate_verdicts import verify_gate_generation

    pairs = declared_pairs(candidate_keys)
    dataset = load_dataset(allow_download=False)
    logger.info(
        "Real gate preamble: pinned dataset %s sha256=%s rows=%d",
        dataset.source_path,
        dataset.sha256,
        len(dataset.frame),
    )
    train, test = load_split_frames(version=REAL_SPLIT_VERSION)
    logger.info(
        "Real gate preamble: split_version=%s train=%d test=%d rows "
        "(target %r, %d feature columns)",
        REAL_SPLIT_VERSION,
        len(train),
        len(test),
        TARGET_COLUMN,
        len(FEATURE_COLUMNS),
    )
    X = train.loc[:, list(FEATURE_COLUMNS)]
    y = train[TARGET_COLUMN]
    committed = load_committed_fit_params(ledger_path)
    # The ledger's best_params are MERGED effective params; the fitters accept
    # only the tuned search-space keys (fixed params are pinned by the specs),
    # so the gate's fit supply is narrowed per member. (The QDA variance
    # floor reg_param stays: it IS a tuned key — MEM100.)
    fit_params = {
        spec.model_type: {
            key: value
            for key, value in committed[spec.model_type].items()
            if key in set(spec.space.keys)
        }
        for spec in BATTERY_MODELS
    }
    dropped = {
        spec.model_type: sorted(
            set(committed[spec.model_type]) - set(fit_params[spec.model_type])
        )
        for spec in BATTERY_MODELS
        if set(committed[spec.model_type]) - set(fit_params[spec.model_type])
    }
    battery_standings = ensure_real_standings(
        sorted({pair.model_type for pair in pairs}),
        train_frame=train,
        test_frame=test,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=REAL_SPLIT_VERSION,
        ledger_path=ledger_path,
    )
    standings_union: dict[str, tuple[float, int]] = {}
    for pair in pairs:
        standings_union.update(
            standings_for_pair(pair.pair_keys, battery_standings)
        )
    logger.info(
        "Real gate: %d declared pair(s) across %d battery member(s); "
        "clause-(c) standings wired for all pairs before measurement.",
        len(pairs),
        len({pair.model_type for pair in pairs}),
    )
    if dropped:
        logger.info(
            "Real gate: narrowed committed params to tuned space keys; "
            "fixed-pinned keys left out per member: %s",
            dropped,
        )
    report = run_gate(
        X,
        y,
        candidate_keys=candidate_keys,
        config=config,
        log_to_mlflow=True,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        fit_params=fit_params,
        standings=standings_union,
        split_version=REAL_SPLIT_VERSION,
    )
    verify_gate_generation(
        candidate_keys=candidate_keys,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=REAL_SPLIT_VERSION,
    )
    return report




# ---------------------------------------------------------------------------
# Acceptance orchestrator: the MEM077/D039 branch, executed by evidence (T03)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceDisposition:
    """The recorded outcome of the gate's acceptance branch (D039).

    Serialized verbatim into
    ``reports/data-methods/treatment_gate_verdicts.json`` as the top-level
    ``acceptance`` record and surfaced via logging: the branch taken, the
    accepted pairs, the treated battery generation's run ids, the
    leaderboard regeneration + winner read-back, the significance
    re-verification reading, and the ship-path trigger decision with its
    written reasoning. Every field is derived from executed evidence — no
    field is a projection of a verdict text.
    """

    branch: str
    split_version: str
    measured_pair_count: int
    rejected_pair_count: int
    accepted_pairs: tuple[dict[str, str], ...]
    declared_coverage: tuple[str, ...]
    pre_generation_winner_model_type: str | None
    post_generation_winner_model_type: str | None
    winner_model_type_changed: bool | None
    winner_in_accepted_coverage: bool | None
    ship_path_triggered: bool
    ship_path_reasoning: str
    ship_evidence: dict[str, object] | None
    generation: dict[str, object] | None
    selection_annotations_cleared: tuple[dict[str, str], ...]
    annotations_restored: dict[str, object] | None
    provenance_inherited: tuple[dict[str, str], ...]
    leaderboard_report_path: str | None
    regenerated_standings: dict[str, dict[str, object]] | None
    significance: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "branch": self.branch,
            "split_version": self.split_version,
            "measured_pair_count": self.measured_pair_count,
            "rejected_pair_count": self.rejected_pair_count,
            "accepted_pairs": [dict(pair) for pair in self.accepted_pairs],
            "declared_coverage": list(self.declared_coverage),
            "pre_generation_winner_model_type": (
                self.pre_generation_winner_model_type
            ),
            "post_generation_winner_model_type": (
                self.post_generation_winner_model_type
            ),
            "winner_model_type_changed": self.winner_model_type_changed,
            "winner_in_accepted_coverage": self.winner_in_accepted_coverage,
            "ship_path_triggered": self.ship_path_triggered,
            "ship_path_reasoning": self.ship_path_reasoning,
            "ship_evidence": self.ship_evidence,
            "generation": self.generation,
            "selection_annotations_cleared": [
                dict(entry) for entry in self.selection_annotations_cleared
            ],
            "annotations_restored": self.annotations_restored,
            "provenance_inherited": [
                dict(entry) for entry in self.provenance_inherited
            ],
            "leaderboard_report_path": self.leaderboard_report_path,
            "regenerated_standings": self.regenerated_standings,
            "significance": [dict(entry) for entry in self.significance],
        }


def _read_leaderboard_winner(
    *,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    split_version: str = REAL_SPLIT_VERSION,
) -> str:
    """Read the leaderboard's #1 model type from the store (module-level).

    Kept a small module-level function (rather than inlined) so tests can
    monkeypatch the simulated winner coverage without ever touching the real
    store — the competitive-guard refusal test pins the loud refusal branch.
    """
    config = LeaderboardConfig(
        experiment_name=experiment_name,
        run_kind=DEFAULT_RUN_KIND,
        split_version=split_version,
        latest_only=True,
        strict=False,
    )
    rows, _considered = select_leaderboard_rows(config=config, tracking_dir=tracking_dir)
    if not rows:
        raise RealStandingsError(
            f"The store at u51 (experiment {experiment_name!r}) holds no "
            f"{DEFAULT_RUN_KIND!r} run rows under split_version={split_version!r}; "
            "the acceptance branch reads the selection winner's model type "
            "from the authoritative leaderboard — run the battery first."
        )
    return str(rows[0].model_type)


def _accepted_pairs_from_report(
    report: GateReport | Mapping[str, object],
) -> tuple[list[AcceptedPairLike], int, int]:
    """Extract accepted pairs + measured/rejected counts from a gate report.

    Accepts either a :class:`heart.eval.treatment_gate.GateReport` (or any
    duck-typed report exposing ``measured_outcomes`` — the gate runner's
    in-memory output shape) or the committed verdicts-artifact payload
    (``outcomes`` + ``accepted_treated_keys`` + counts). Cross-checks the two
    sources when both are present: a payload whose ``accepted_treated_keys``
    disagrees with its ``outcomes`` verdicts is a loud caller error, never a
    silently narrowed acceptance list.
    """
    from heart.models.treatment_final import accepted_pairs_from_keys

    measured = 0
    rejected = 0
    accepted_keys: list[str] = []
    if hasattr(report, "measured_outcomes"):
        pairs: list[tuple[str, str]] = []
        for outcome in report.measured_outcomes:
            measured += 1
            if outcome.verdict == VERDICT_ACCEPTED:
                if outcome.model_type is None or outcome.candidate_key is None:
                    raise TreatmentGateRealError(
                        "Measured accepted outcome lacks candidate/model "
                        f"identity ({outcome!r}); cannot enter the acceptance "
                        "branch."
                    )
                pairs.append((str(outcome.candidate_key), str(outcome.model_type)))
            else:
                rejected += 1
        return accepted_pairs_from_keys(pairs), measured, rejected
    if not isinstance(report, Mapping):
        raise TreatmentGateRealError(
            f"report must be a GateReport or the verdicts-artifact payload "
            f"mapping, got {type(report).__name__}."
        )
    raw_keys = report.get("accepted_treated_keys")
    if raw_keys is not None and not isinstance(raw_keys, list):
        raise TreatmentGateRealError(
            f"payload 'accepted_treated_keys' must be a list, got "
            f"{type(raw_keys).__name__}."
        )
    outcome_verdicts: dict[str, str] = {}
    outcomes = report.get("outcomes")
    if outcomes is None:
        outcomes = []
    if not isinstance(outcomes, list):
        raise TreatmentGateRealError(
            f"payload 'outcomes' must be a list, got {type(outcomes).__name__}."
        )
    mismatched = []
    for entry in outcomes:
        if not isinstance(entry, dict):
            raise TreatmentGateRealError(
                f"payload outcomes entry {entry!r} is not an object."
            )
        treated_key = entry.get("treated_key")
        if not entry.get("re_measured"):
            continue
        measured += 1
        verdict = str(entry.get("verdict", ""))
        if verdict != VERDICT_ACCEPTED:
            rejected += 1
        if isinstance(treated_key, str) and treated_key:
            outcome_verdicts[str(treated_key)] = verdict
    if raw_keys is not None:
        for key in raw_keys:
            if not isinstance(key, str) or not key.startswith("treatment:"):
                raise TreatmentGateRealError(
                    f"accepted_treated_keys entry {key!r} is not a "
                    "'treatment:<variant>/<model_type>' key."
                )
            recorded = outcome_verdicts.get(key)
            if recorded is not None and recorded != VERDICT_ACCEPTED:
                mismatched.append((key, recorded))
        if mismatched:
            raise TreatmentGateRealError(
                f"payload accepted_treated_keys disagree(s) with the "
                f"outcomes verdict(s): {mismatched}; a mismatched acceptance "
                "list is a loud caller error, never a silently narrowed one."
            )
        accepted_keys = [str(key) for key in raw_keys]
    else:
        accepted_keys = [
            key for key, verdict in outcome_verdicts.items()
            if verdict == VERDICT_ACCEPTED
        ]
    supplied_measured = report.get("measured_pair_count")
    if supplied_measured is not None:
        measured = int(supplied_measured)  # type: ignore[arg-type]
    supplied_rejected = report.get("rejected_pair_count")
    if supplied_rejected is not None:
        rejected = int(supplied_rejected)  # type: ignore[arg-type]
    return accepted_pairs_from_keys(accepted_keys), measured, rejected


def _stored_significance_readings(
    accepted_treated_keys: Sequence[str],
    *,
    tracking_dir: str | Path | None,
    experiment_name: str,
    split_version: str,
) -> tuple[dict[str, object], ...]:
    """Read the accepted pairs' significance readings from their stored runs.

    The gate's paired repeated-CV comparison (the corrected resampled t-test
    at the declared alpha) is recorded on each ``run_kind=treatment-candidate``
    run (the ``significance_reading`` param plus the pooled-OOF gate
    artifact). Re-verification re-reads those stored readings and refuses
    loudly when an accepted pair's storage does not show a TRUE reading.
    """
    from heart.reporting.data_methods.gate_verdicts import collect_candidate_runs

    runs = {
        run.treated_key: run
        for run in collect_candidate_runs(
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
            split_version=split_version,
        )
    }
    records: list[dict[str, object]] = []
    for key in accepted_treated_keys:
        run = runs.get(key)
        if run is None:
            raise TreatmentGateRealError(
                f"Accepted pair {key!r} has no stored treatment-candidate run "
                f"under split_version={split_version!r}; the significance "
                "re-verification reads each accepted pair's own stored "
                "repeated-CV comparison — nothing to read."
            )
        reading = run.params.get("significance_reading", "")
        if reading != "True":
            raise TreatmentGateRealError(
                f"Accepted pair {key!r} stores significance_reading="
                f"{reading!r}; an accepted verdict reaching the acceptance "
                "branch must carry a TRUE stored reading (clause (b))."
            )
        if run.report_outcome.get("significant") is not True:
            raise TreatmentGateRealError(
                f"Accepted pair {key!r} gate artifact records significant="
                f"{run.report_outcome.get('significant')!r}; disagrees with "
                "the stored significance reading — the acceptance evidence "
                "cannot be reconciled."
            )
        records.append(
            {
                "treated_key": key,
                "delta": float(run.params.get("delta", "nan")),
                "significant": True,
                "reading_source": (
                    "stored repeated-CV comparison (treatment-candidate run "
                    "significance_reading param + pooled-OOF gate artifact); "
                    "re-verified from the store, not re-measured"
                ),
            }
        )
    return tuple(records)


def apply_acceptances(
    report: GateReport | Mapping[str, object],
    *,
    train_frame: pd.DataFrame | None = None,
    test_frame: pd.DataFrame | None = None,
    split_version: str = REAL_SPLIT_VERSION,
    ledger_path: str | Path = DEFAULT_BATTERY_LEDGER_PATH,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    members: Sequence[str] | None = None,
    leaderboard_report_path: str | Path | None = None,
    regenerate_leaderboard: bool = True,
    ship_executor: Callable[..., dict[str, object]] | None = None,
) -> AcceptanceDisposition:
    """Execute the MEM077/D039 acceptance branch on the gate's outcome.

    Branch semantics (the disposition ALWAYS names its branch and reasoning):

    - **no-acceptance** (zero accepted pairs): the disposition records
      "no-acceptance: ship path not triggered" with the measured/rejected
      pair counts. ZERO ``run_kind=final`` runs are written — the committed
      battery generation is today's coherent generation, identical in the
      store.
    - **treated-generation** (>=1 accepted pair): the stale ``s6.*``
      selection annotations are cleared from final runs FIRST (S06 hygiene
      trap — never let two annotated rows exist), the winner's model type is
      read from the leaderboard, ONE coherent treated full-battery
      generation is re-run through
      :func:`heart.models.treatment_final.run_treated_final_generation`
      (fresh ``run_kind=final`` runs tagged ``treatment_generation`` /
      ``treatment_variant``), ``reports/leaderboard.md`` is regenerated
      through the leaderboard module's own CLI semantics, and the winner is
      re-read:

      * winner model type outside every declared candidate's
        ``applies_to`` set (the competitive guard) — the disposition
        records that written claim with the coverage union as evidence; the
        accepted treatment(s) update reports and standings only;
      * winner model type covered by an accepted pair — the competitive
        ship path is REQUIRED: with ``ship_executor=None`` this refuses
        loudly (:class:`heart.models.treatment_final.CompetitiveGuardError`)
        rather than record a false "not triggered"; with the executor wired
        the existing selection/serving machinery runs and its evidence is
        recorded.

    Frames default to the real declared split
    (:func:`heart.data.split.load_split_frames`); tests pass fixture frames.
    ``members`` forwards to the generation (full battery when ``None``).

    Returns the JSON-serializable :class:`AcceptanceDisposition`.
    """
    import heart.models.treatment_final as treatment_final
    from heart.models.treatment_final import (
        COMPETITIVE_SHIP_REASONING,
        NON_COMPETITIVE_ACCEPTANCE_REASONING,
        NO_ACCEPTANCE_REASONING,
        CompetitiveGuardError,
    )

    pairs, measured, rejected = _accepted_pairs_from_report(report)
    accepted_keys = [pair.treated_key for pair in pairs]
    coverage = sorted(treatment_final.declared_coverage())
    logger.info(
        "Acceptance orchestrator: %d accepted pair(s) of %d measured "
        "(%d rejected) -> %s branch; declared coverage %s",
        len(pairs),
        measured,
        rejected,
        "treated-generation" if pairs else "no-acceptance",
        coverage,
    )
    if not pairs:
        reasoning = NO_ACCEPTANCE_REASONING.format(
            measured=measured,
            alpha=PAIRED_SIGNIFICANCE_ALPHA,
            min_delta=GATE_MIN_DELTA_ROC_AUC,
            accepted=0,
        )
        written_path: str | None = None
        current_winner = _read_leaderboard_winner(
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
            split_version=split_version,
        )
        winner_covered, _winner_variants = treatment_final.coverage_of(
            current_winner
        )
        if winner_covered:
            raise TreatmentGateRealError(
                f"no-acceptance branch inconsistency: the store's current "
                f"leaderboard winner {current_winner!r} IS inside declared "
                f"coverage {coverage} while NO pair was accepted; refusing "
                "to record a disposition from an incoherent store."
            )
        annotations_cleared = treatment_final.clear_stale_selection_annotations(
            split_version=split_version,
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
        )
        annotations_restored = treatment_final.restore_winner_annotations(
            expected_winner=None,
            split_version=split_version,
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
        )
        provenance_inherited = treatment_final.inherit_prior_generation_provenance(
            split_version=split_version,
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
        )
        if regenerate_leaderboard:
            from heart.reporting.leaderboard import DEFAULT_REPORT_PATH

            board = treatment_final.regenerate_leaderboard_report(
                tracking_dir=tracking_dir,
                split_version=split_version,
                report_path=(
                    leaderboard_report_path
                    if leaderboard_report_path is not None
                    else DEFAULT_REPORT_PATH
                ),
            )
            written_path = str(board.report_path)
        reasoning_full = (
            reasoning
            + " The store's leaderboard winner "
            + repr(current_winner)
            + " is not in declared coverage "
            + str(coverage)
            + " (competitive guard holds with zero acceptances); the S06 "
            "selection narrative and the battery provenance tags were "
            "restated from the store onto the current generation before the "
            f"regeneration (restoration status {annotations_restored.get('status')!r})."
        )
        logger.info(
            "ApplyAcceptance branch=no-acceptance ship_path_triggered=False "
            "measured=%d rejected=%d winner=%r annotations=%s/%d "
            "provenance=%d leaderboard=%s",
            measured,
            rejected,
            current_winner,
            annotations_restored.get("status"),
            len(annotations_cleared),
            len(provenance_inherited),
            written_path,
        )
        return AcceptanceDisposition(
            branch="no-acceptance",
            split_version=split_version,
            measured_pair_count=measured,
            rejected_pair_count=rejected,
            accepted_pairs=(),
            declared_coverage=tuple(coverage),
            pre_generation_winner_model_type=None,
            post_generation_winner_model_type=current_winner,
            winner_model_type_changed=None,
            winner_in_accepted_coverage=False,
            ship_path_triggered=False,
            ship_path_reasoning=reasoning_full,
            ship_evidence=None,
            generation=None,
            selection_annotations_cleared=tuple(
                dict(entry) for entry in annotations_cleared
            ),
            annotations_restored=annotations_restored,
            provenance_inherited=tuple(dict(x) for x in provenance_inherited),
            leaderboard_report_path=written_path,
            regenerated_standings=None,
            significance=(),
        )

    # --- treated-generation branch (>=1 accepted pair) ---------------------
    annotations_cleared = treatment_final.clear_stale_selection_annotations(
        split_version=split_version,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
    )
    pre_winner = _read_leaderboard_winner(
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=split_version,
    )
    logger.info(
        "Acceptance orchestrator: pre-generation leaderboard winner "
        "model_type=%r (declared coverage %s)",
        pre_winner,
        coverage,
    )
    if train_frame is None or test_frame is None:
        train_frame, test_frame = load_split_frames(version=split_version)
    generation = treatment_final.run_treated_final_generation(
        train_frame,
        test_frame,
        pairs,
        split_version=split_version,
        ledger_path=ledger_path,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        members=members,
    )
    post_winner = _read_leaderboard_winner(
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=split_version,
    )
    winner_changed = post_winner != pre_winner
    winner_covered, winner_variant_keys = treatment_final.coverage_of(post_winner)
    logger.info(
        "Acceptance orchestrator: post-generation leaderboard winner "
        "model_type=%r changed=%s inacceptedcoverage=%s (covering variants %s)",
        post_winner,
        winner_changed,
        winner_covered,
        winner_variant_keys,
    )
    ship_evidence: dict[str, object] | None = None
    ship_triggered = bool(winner_covered)
    significance: tuple[dict[str, object], ...] = ()
    annotations_restored: dict[str, object] | None = None
    provenance_inherited: tuple[dict[str, str], ...] = ()
    if ship_triggered:
        reasoning = COMPETITIVE_SHIP_REASONING.format(
            model_type=post_winner,
            treated=accepted_keys,
        )
        if ship_executor is None:
            raise CompetitiveGuardError(
                f"Competitive ship path REQUIRED but no ship executor is "
                f"wired: the regenerated leaderboard's winner model type "
                f"{post_winner!r} is covered by accepted pair(s) "
                f"{accepted_keys} (covering variants "
                f"{winner_variant_keys}). The branch must run the existing "
                "selection/serving machinery end to end — refusing to "
                "record a false 'ship path not triggered'. Wire "
                "ship_executor=heart.models.treatment_final."
                "run_competitive_ship_path (or an explicit override) and "
                "re-run; see the disposition's guard semantics."
            )
        ship_evidence = ship_executor(
            split_version=split_version, tracking_dir=tracking_dir
        )
    else:
        reasoning = NON_COMPETITIVE_ACCEPTANCE_REASONING.format(
            model_type=post_winner,
            coverage=coverage,
            treated=accepted_keys,
        )
        annotations_restored = treatment_final.restore_winner_annotations(
            expected_winner=None,
            split_version=split_version,
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
        )
        provenance_inherited = tuple(
            dict(x)
            for x in treatment_final.inherit_prior_generation_provenance(
                split_version=split_version,
                tracking_dir=tracking_dir,
                experiment_name=experiment_name,
            )
        )
        significance = _stored_significance_readings(
            accepted_keys,
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
            split_version=split_version,
        )
    written_path: str | None = None
    if regenerate_leaderboard:
        from heart.reporting.leaderboard import DEFAULT_REPORT_PATH

        board = treatment_final.regenerate_leaderboard_report(
            tracking_dir=tracking_dir,
            split_version=split_version,
            report_path=(
                leaderboard_report_path
                if leaderboard_report_path is not None
                else DEFAULT_REPORT_PATH
            ),
        )
        written_path = str(board.report_path)
    try:
        regenerated_standings = build_real_standings(
            members if members is not None else [s.model_type for s in BATTERY_MODELS],
            tracking_dir=tracking_dir,
            split_version=split_version,
        )
    except (RealStandingsError, _LeaderboardError) as exc:
        raise RealStandingsError(
            f"Post-generation standings read-back failed: {exc}"
        ) from exc
    standings_projection = {
        key.removeprefix("incumbent:"): {
            "roc_auc": float(entry[0]),
            "rank": int(entry[1]),
        }
        for key, entry in regenerated_standings.items()
    }
    return AcceptanceDisposition(
        branch="treated-generation",
        split_version=split_version,
        measured_pair_count=measured,
        rejected_pair_count=rejected,
        accepted_pairs=tuple(pair.to_dict() for pair in pairs),
        declared_coverage=tuple(coverage),
        pre_generation_winner_model_type=pre_winner,
        post_generation_winner_model_type=post_winner,
        winner_model_type_changed=winner_changed,
        winner_in_accepted_coverage=winner_covered,
        ship_path_triggered=ship_triggered,
        ship_path_reasoning=reasoning,
        ship_evidence=ship_evidence,
        generation={
            **generation.to_dict(),
            "members": list(
                members
                if members is not None
                else [spec.model_type for spec in BATTERY_MODELS]
            ),
        },
        selection_annotations_cleared=tuple(
            dict(entry) for entry in annotations_cleared
        ),
        annotations_restored=annotations_restored,
        provenance_inherited=provenance_inherited,
        leaderboard_report_path=written_path,
        regenerated_standings=standings_projection,
        significance=significance,
    )


def apply_acceptances_from_store(
    *,
    split_version: str = REAL_SPLIT_VERSION,
    ledger_path: str | Path = DEFAULT_BATTERY_LEDGER_PATH,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    candidate_keys: Sequence[str] | None = None,
    model_types: Sequence[str] | None = None,
    leaderboard_report_path: str | Path | None = None,
    regenerate_leaderboard: bool = True,
    ship_executor: Callable[..., dict[str, object]] | None = None,
) -> AcceptanceDisposition:
    """Run the acceptance orchestrator FROM the verified stored generation.

    The store is the only source: :func:`verify_gate_generation` proves every
    measured declared pair is stored exactly once under ``split_version``
    with constant-faithful params, then the pair-level verdicts feed
    :func:`apply_acceptances` (no gate re-measurement, no hand-assembled
    acceptance list).
    """
    from heart.reporting.data_methods.gate_verdicts import verify_gate_generation

    verification = verify_gate_generation(
        candidate_keys=candidate_keys,
        model_types=model_types,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=split_version,
    )
    verdicts: dict[str, str] = dict(verification["verdicts"])
    accepted_keys = [
        key for key, verdict in verdicts.items() if verdict == VERDICT_ACCEPTED
    ]
    pairs_view = {
        "accepted_treated_keys": accepted_keys,
        "measured_pair_count": int(verification["run_count"]),
        "rejected_pair_count": int(verification["run_count"]) - len(accepted_keys),
    }
    return apply_acceptances(
        pairs_view,
        split_version=split_version,
        ledger_path=ledger_path,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        leaderboard_report_path=leaderboard_report_path,
        regenerate_leaderboard=regenerate_leaderboard,
        ship_executor=ship_executor,
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI for the one-command real-data gate (the slice's runtime seam)."""
    parser = argparse.ArgumentParser(
        prog="python -m heart.eval.treatment_gate_real",
        description="Run the M006 treatment gate against the real data and "
        "the real MLflow store (S03; D037-D039).",
    )
    parser.add_argument(
        "--candidate-keys",
        default=None,
        help="Optional comma-separated subset of declared variant keys.",
    )
    parser.add_argument(
        "--tracking-dir",
        default=None,
        help="Optional MLflow store directory override (default: the "
        "worktree store at experiments/mlruns).",
    )
    parser.add_argument(
        "--out",
        default="reports/data-methods/treatment_gate_verdicts.json",
        help="Where the MLflow-derived verdicts artifact is written.",
    )
    parser.add_argument(
        "--apply-acceptances",
        action="store_true",
        help="Skip the gate re-measurement and run ONLY the acceptance "
        "orchestrator from the verified stored generation (T03): apply the "
        "MEM077/D039 branch and rewrite the verdicts artifact with its "
        "top-level 'acceptance' record.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate (or the acceptance orchestrator), exit 0."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    candidate_keys = (
        [key.strip() for key in args.candidate_keys.split(",") if key.strip()]
        if args.candidate_keys
        else None
    )
    if args.apply_acceptances:
        from heart.models.treatment_final import run_competitive_ship_path
        from heart.reporting.data_methods.gate_verdicts import (
            write_gate_verdicts_artifact as _write_artifact,
        )

        disposition = apply_acceptances_from_store(
            tracking_dir=args.tracking_dir,
            ship_executor=run_competitive_ship_path,
        )
        payload = _write_artifact(
            path=args.out,
            tracking_dir=args.tracking_dir,
            split_version=REAL_SPLIT_VERSION,
            acceptance=disposition.to_dict(),
        )
        logger.info(
            "Acceptance record written: branch=%s ship_path_triggered=%s "
            "measured=%d rejected=%d accepted=%d artifact=%s",
            disposition.branch,
            disposition.ship_path_triggered,
            disposition.measured_pair_count,
            disposition.rejected_pair_count,
            len(disposition.accepted_pairs),
            args.out,
        )
        return 0
    report = run_real_gate(candidate_keys, tracking_dir=args.tracking_dir)
    from heart.reporting.data_methods.gate_verdicts import (
        write_gate_verdicts_artifact,
    )

    payload = write_gate_verdicts_artifact(
        path=args.out,
        tracking_dir=args.tracking_dir,
        experiment_name=DEFAULT_EXPERIMENT,
        split_version=REAL_SPLIT_VERSION,
    )
    outcomes = payload.get("outcomes", [])
    logger.info(
        "Gate verdicts artifact written to %s (%d measured outcome(s)): "
        "verdict tallies ranked by clause reading.",
        args.out,
        len(outcomes),
    )
    for outcome in outcomes:
        logger.info(
            "PAIRVERDICT %s delta=%+.6f significant=%s verdict=%s",
            outcome.get("treated_key"),
            float(outcome.get("delta") or 0.0),
            outcome.get("significant"),
            outcome.get("verdict"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
