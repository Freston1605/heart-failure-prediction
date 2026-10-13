"""MLflow-derived treatment-gate verdicts artifact (M006 S03/T02; D038).

Why this module exists
----------------------
The real-data gate command (:func:`heart.eval.treatment_gate_real.run_real_gate`)
logs exactly one ``run_kind=treatment-candidate`` / ``split_version=v1`` MLflow
run per declared measurable (candidate, model_type) pair. This module derives
``reports/data-methods/treatment_gate_verdicts.json`` FROM that store — the
D025/D032 rule that a report number is never hand-assembled: every outcome in
the artifact is read back from the recorded run params/tags plus the run's own
``gate_report.json`` artifact, and any mismatch, missing pair, duplicate pair,
or constant contradiction fails with a named :class:`GateVerdictsError`.

One namespace owns the loud checks (T02 step 3):

- :func:`collect_candidate_runs` reads every candidate run for a split version
  and resolves its stored ``gate_report.json`` outcome entry,
- :func:`verify_gate_generation` proves every declared pair is stored exactly
  once with params/tags matching the declared constants (used both by the gate
  command post-run and by tests),
- :func:`build_gate_verdicts_artifact` assembles the Top-Level payload keyed in
  the D025 house style (``source_module``, ``derived_from_mlflow``,
  ``gate_sentence`` verbatim ``gate_text()``, ``config_fingerprint``,
  ``standings_supplied``, ``split_version``, ``outcomes``, ``mlflow``),
- :func:`write_gate_verdicts_artifact` writes it deterministically (sorted
  keys, indent 2, trailing newline, no timestamps).

Everything here is a store→projection mapping; it re-implements no gate clause,
threshold, or significance logic.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

from heart.config import PROJECT_ROOT
from heart.eval.treatment_declared import (
    DECLARED_CANDIDATES,
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
    MLFLOW_TAG_TREATMENT_VARIANT,
    PAIRED_SIGNIFICANCE_ALPHA,
    gate_text,
)
from heart.eval.treatment_gate import GATE_REPORT_ARTIFACT, VERDICT_ACCEPTED
from heart.eval.treatment_gate_real import (
    REAL_GATE_COMMAND,
    REAL_SPLIT_VERSION,
    TreatmentGateRealError,
)
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT
from heart.tuning.runner import MODEL_TYPE_TAG, RUN_KIND_TAG

logger = logging.getLogger(__name__)

#: Surface name part F/the parity probe cite for the verdicts artifact.
GATE_VERDICTS_ARTIFACT: str = "treatment_gate_verdicts.json"

#: Default artifact target (relative to the project root).
DEFAULT_GATE_VERDICTS_PATH: Final[Path] = (
    PROJECT_ROOT / "reports" / "data-methods" / GATE_VERDICTS_ARTIFACT
)


class GateVerdictsError(TreatmentGateRealError):
    """The stored gate generation cannot faithfully yield the verdicts partner.

    Raised (never swallowed) when a declared pair has no candidate run, a pair
    has more than one, a run's params/tags contradict the declared constants,
    the run's ``gate_report.json`` artifact is missing or unusable, or two
    runs disagree on the measurement geometry. The serialization derives the
    published report; anything less than an exact read-back is a loud failure
    rather than a hand-massaged artifact.
    """


@dataclass(frozen=True)
class CandidateRun:
    """Read back one candidate run from the store (the assembly's one source)."""

    treated_key: str
    run_id: str
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, float]
    report_outcome: dict[str, object]
    artifact_path: Path
    gate_fingerprint: str


# Fact SHORT names the parity surface reuses (single definition).
ARTIFACT_TOP_KEYS: Final[tuple[str, ...]] = (
    "acceptance",
    "config_fingerprint",
    "derived_from_mlflow",
    "gate_sentence",
    "measured_pair_count",
    "accepted_treated_keys",
    "mlflow",
    "outcomes",
    "source_module",
    "split_version",
    "standings_supplied",
)


###########
# Reading the store
###########


def _artifact_dir(run) -> Path:
    uri = str(run.info.artifact_uri)
    root = Path(urllib.parse.urlparse(uri).path)
    # log_evaluation_run nests logged artifacts at <run>/artifacts/<name>.
    if (root / "artifacts").is_dir():
        return root / "artifacts"
    return root


def collect_candidate_runs(
    *,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    split_version: str = REAL_SPLIT_VERSION,
) -> list[CandidateRun]:
    """Return every ``run_kind=treatment-candidate`` run for ``split_version``.

    Pure store read; no declared-constants checking here (that is
    :func:`verify_gate_generation`). Runs are ordered deterministically (run
    start time, run id tiebreak). A run without a parseable
    ``gate_report.json`` artifact outcome (or an artifact missing the run's
    treated key) fails loudly — the artifact is part of the run's own
    declaration (S02's contract) and is load-bearing for the projection.
    """
    from heart.reporting.leaderboard import resolve_tracking_uri as _lb_resolve
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = _lb_resolve(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise GateVerdictsError(
            f"No MLflow experiment named {experiment_name!r} exists in the "
            f"store at {tracking_uri}; point the caller at the right store."
        )
    runs = sorted(
        client.search_runs([experiment.experiment_id], max_results=10000),
        key=lambda run: (run.info.start_time or 0, run.info.run_id),
    )
    collected: list[CandidateRun] = []
    for run in runs:
        tags = {str(k): str(v) for k, v in dict(run.data.tags).items()}
        if tags.get(RUN_KIND_TAG) != MLFLOW_RUN_KIND_TREATMENT_CANDIDATE:
            continue
        if tags.get("split_version") != split_version:
            continue
        treated_key = tags.get("model_name", "")
        model_name = tags.get("model_name", "")
        if not model_name.startswith("treatment:"):
            raise GateVerdictsError(
                f"Run {run.info.run_id} is tagged "
                f"run_kind={MLFLOW_RUN_KIND_TREATMENT_CANDIDATE} but its "
                f"model_name tag {model_name!r} is not a "
                "'treatment:<variant>/<model_type>' name."
            )
        artifact_path = _artifact_dir(run) / GATE_REPORT_ARTIFACT
        if not artifact_path.is_file():
            raise GateVerdictsError(
                f"Run {run.info.run_id} ({model_name!r}) has no "
                f"{GATE_REPORT_ARTIFACT!r} artifact at {artifact_path}; the "
                "verdicts projection reads the run's own declaration."
            )
        report = json.loads(artifact_path.read_text(encoding="utf-8"))
        outcomes = report.get("outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise GateVerdictsError(
                f"Run {run.info.run_id} ({model_name!r}) artifact "
                f"{artifact_path} carries no outcomes list."
            )
        matching = [
            entry
            for entry in outcomes
            if isinstance(entry, dict) and entry.get("treated_key") == model_name
        ]
        if len(matching) != 1:
            raise GateVerdictsError(
                f"Run {run.info.run_id} ({model_name!r}) artifact "
                f"{artifact_path} carries {len(matching)} outcome entry/entries "
                f"for treated_key {model_name!r}; expected exactly one."
            )
        params = {str(k): str(v) for k, v in dict(run.data.params).items()}
        collected.append(
            CandidateRun(
                treated_key=model_name,
                run_id=str(run.info.run_id),
                params=params,
                tags=tags,
                metrics={str(k): float(v) for k, v in dict(run.data.metrics).items()},
                report_outcome=dict(matching[0]),
                artifact_path=artifact_path,
                gate_fingerprint=params.get("gate_fingerprint", ""),
            )
        )
    return collected


def _param_float(params: dict[str, str], key: str, run_id: str) -> float:
    value = params.get(key)
    if value is None:
        raise GateVerdictsError(
            f"Run {run_id} lacks the {key!r} param; the candidate-run  params "
            f"contract carries it (params present: {sorted(params)})."
        )
    try:
        number = float(value)
    except ValueError as exc:
        raise GateVerdictsError(
            f"Run {run_id} param {key!r}={value!r} is not a float."
        ) from exc
    return number


###########
# Verification (T02 step 3: the loud post-run checks)
###########


def verify_gate_generation(
    *,
    candidate_keys=None,
    model_types=None,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    split_version: str = REAL_SPLIT_VERSION,
) -> dict[str, object]:
    """Prove the stored generation is complete and constant-faithful.

    Raises :class:`GateVerdictsError` (never returns a partial verdict) when:

    - a DECLARED pair has no candidate run for ``split_version``,
    - a pair has more than one run (duplicates),
    - a run's params contradict the declared constants (``alpha``/
      ``min_delta``/``gate_metric`` parity),
    - a run's variant/model identity (params vs tags vs the declaration)
      disagrees with the treated key,
    - the stored verdict clauses disagree between the run's params and its
      ``gate_report.json`` artifact, or ``standing_ok`` is missing (clause (c)
      is ALWAYS supplied on the real gate — D038), or
    - the runs disagree on the measurement geometry fingerprint.

    Returns
    -------
    dict
        ``{"run_count", "pairs_covered", "verdicts", "gate_fingerprint"}``.
    """
    from heart.eval.treatment_gate_real import declared_pairs

    expected: dict[str, tuple[str, str]] = {}
    for pair in declared_pairs(candidate_keys):
        if model_types is not None and pair.model_type not in set(model_types):
            continue
        expected[
            f"treatment:{pair.candidate.variant_key}/{pair.model_type}"
        ] = (pair.candidate.variant_key, pair.model_type)
    runs = collect_candidate_runs(
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=split_version,
    )
    if not runs and expected:
        raise GateVerdictsError(
            f"The store holds NO run_kind={MLFLOW_RUN_KIND_TREATMENT_CANDIDATE} "
            f"run under split_version={split_version!r}, but {len(expected)} "
            "declared pair(s) were measured; the gate command's logging "
            "aborted before any run landed — inspect the command's error."
        )
    seen: dict[str, list[str]] = {}
    for run in runs:
        seen.setdefault(run.treated_key, []).append(run.run_id)
    duplicates = {key: ids for key, ids in seen.items() if len(ids) > 1}
    if duplicates:
        raise GateVerdictsError(
            f"Duplicated candidate pair(s) under split_version="
            f"{split_version!r}: {duplicates}; exactly one run per pair."
        )
    unexpected = sorted(set(seen) - set(expected))
    if unexpected:
        raise GateVerdictsError(
            f"Stored candidate run(s) outside the declared pair set: "
            f"{unexpected}; the gate logs declared pairs only."
        )
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise GateVerdictsError(
            f"Declared pair(s) missing a split_version={split_version!r} "
            f"candidate run: {missing}."
        )
    fingerprints: set[str] = set()
    verdicts: dict[str, str] = {}
    for run in runs:
        variant_key, model_type = expected[run.treated_key]
        # Identity parity: params vs tags vs the declaration vs the treated key.
        if run.params.get("variant_key", "") != variant_key:
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) param variant_key="
                f"{run.params.get('variant_key')!r} contradicts the declared "
                f"variant {variant_key!r}."
            )
        if run.params.get("model_type", "") != model_type or (
            run.tags.get(MODEL_TYPE_TAG, "") != model_type
        ):
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) model_type "
                f"params={run.params.get('model_type')!r} "
                f"tags={run.tags.get(MODEL_TYPE_TAG)!r} contradict the declared "
                f"applies_to model {model_type!r}."
            )
        if run.tags.get(MLFLOW_TAG_TREATMENT_VARIANT, "") != variant_key:
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) tag "
                f"{MLFLOW_TAG_TREATMENT_VARIANT}="
                f"{run.tags.get(MLFLOW_TAG_TREATMENT_VARIANT)!r} contradicts "
                f"the declared variant {variant_key!r}."
            )
        # Threshold parity: the declared constants, read back from the run.
        alpha = _param_float(run.params, "alpha", run.run_id)
        min_delta = _param_float(run.params, "min_delta", run.run_id)
        if alpha != float(PAIRED_SIGNIFICANCE_ALPHA):
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) param alpha="
                f"{alpha!r} contradicts the declared "
                f"PAIRED_SIGNIFICANCE_ALPHA={PAIRED_SIGNIFICANCE_ALPHA!r}."
            )
        if min_delta != float(GATE_MIN_DELTA_ROC_AUC):
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) param min_delta="
                f"{min_delta!r} contradicts the declared "
                f"GATE_MIN_DELTA_ROC_AUC={GATE_MIN_DELTA_ROC_AUC!r}."
            )
        if run.params.get("gate_metric", "") != GATE_METRIC:
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) param gate_metric="
                f"{run.params.get('gate_metric')!r} contradicts the declared "
                f"GATE_METRIC={GATE_METRIC!r}."
            )
        if not run.gate_fingerprint:
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) lacks the "
                "'gate_fingerprint' param."
            )
        fingerprints.add(run.gate_fingerprint)
        if len(fingerprints) > 1:
            raise GateVerdictsError(
                f"Stored candidate runs disagree on the measurement geometry "
                f"({sorted(fingerprints)}); one gate generation shares one "
                "fingerprint."
            )
        # Clause parity: params vs the run's own gate_report.json artifact.
        outcome = run.report_outcome
        artifact_verdict = str(outcome.get("verdict", ""))
        param_verdict = run.params.get("verdict", "")
        tag_verdict = run.tags.get("gate_verdict", "")
        if not (artifact_verdict == param_verdict == tag_verdict):
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) verdict disagreement: "
                f"artifact={artifact_verdict!r} params={param_verdict!r} "
                f"tags={tag_verdict!r}."
            )
        delta_artifact = outcome.get("delta")
        delta_param = _param_float(run.params, "delta", run.run_id)
        if delta_artifact is None or abs(float(delta_artifact) - delta_param) > 1e-12:
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) delta disagreement: "
                f"artifact={delta_artifact!r} param={delta_param!r}."
            )
        standing = run.params.get("standing_ok", "null")
        if standing == "null":
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) records standing_ok="
                f"{standing!r}; clause (c) standings are ALWAYS supplied on "
                "the real gate (D038) — no incomplete-evidence verdict is "
                "possible."
            )
        if str(outcome.get("standing_ok")) != (
            "True" if standing == "True" else "False"
        ):
            raise GateVerdictsError(
                f"Run {run.run_id} ({run.treated_key!r}) standing_ok "
                f"disagreement: artifact={outcome.get('standing_ok')!r} "
                f"param={standing!r}."
            )
        verdicts[run.treated_key] = artifact_verdict
    logger.info(
        "Gate generation verified: %d run(s), %d declared pair(s), "
        "fingerprint %r, verdicts %s",
        len(runs),
        len(seen),
        next(iter(fingerprints), ""),
        verdicts,
    )
    return {
        "run_count": len(runs),
        "pairs_covered": len(seen),
        "verdicts": verdicts,
        "gate_fingerprint": next(iter(fingerprints), ""),
    }


###########
# The artifact projection (store -> payload)
###########


def build_gate_verdicts_artifact(
    *,
    candidate_keys=None,
    model_types=None,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    split_version: str = REAL_SPLIT_VERSION,
    acceptance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the D025-style verifications artifact payload from the store.

    Inputs: verified candidate runs (:func:`verify_gate_generation`). The
    ``outcomes`` list mirrors the recorded ``GateOutcome.to_dict()`` field
    vector per pair (taken from each run's own ``gate_report.json`` artifact
    entry, cross-checked against the run's params/tags), with the stored
    ``run_id`` filled in. Payload keys are house-style D025/D032:
    ``source_module``, ``derived_from_mlflow``, ``gate_sentence`` (verbatim
    ``gate_text()``), ``config_fingerprint``, ``standings_supplied``,
    ``split_version``, ``outcomes``, ``mlflow``. No timestamps.
    """
    from heart.eval.treatment_declared import DECLARED_CANDIDATES as _dc

    verification = verify_gate_generation(
        candidate_keys=candidate_keys,
        model_types=model_types,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=split_version,
    )
    fingerprint = str(verification["gate_fingerprint"])
    runs = {
        run.treated_key: run
        for run in collect_candidate_runs(
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
            split_version=split_version,
        )
    }
    # Declared order: candidate's declaration order, then its applies_to order.
    order: dict[str, int] = {}
    for candidate in _dc:
        for index, model_type in enumerate(candidate.applies_to):
            order[f"treatment:{candidate.variant_key}/{model_type}"] = (
                candidate.order * 100 + index
            )
    outcomes: list[dict[str, object]] = []
    for treated_key in sorted(runs, key=lambda key: order.get(key, 10**6)):
        run = runs[treated_key]
        outcome = dict(run.report_outcome)
        # Mirror the recorded GateOutcome.to_dict() field vector; the stored
        # run_id is filled from the store (the artifact's entry records the
        # pre-logging None).
        projection = {
            key: outcome.get(key)
            for key in (
                "candidate_key",
                "model_type",
                "mechanism",
                "group",
                "re_measured",
                "verdict",
                "verdict_text",
                "delta",
                "significant",
                "delta_ok",
                "standing_ok",
                "n_folds",
                "n_repeats",
                "pooled_oof_metrics",
                "treated_key",
            )
        }
        projection["run_id"] = run.run_id
        outcomes.append(projection)
    accepted = [
        key for key, value in verification["verdicts"].items()
        if value == VERDICT_ACCEPTED
    ]
    if acceptance is not None and not isinstance(acceptance, Mapping):
        raise GateVerdictsError(
            f"acceptance record must be a mapping, got "
            f"{type(acceptance).__name__}."
        )
    payload: dict[str, object] = {
        "source_module": REAL_GATE_COMMAND,
        "derived_from_mlflow": True,
        "gate_sentence": gate_text(),
        "config_fingerprint": fingerprint,
        "standings_supplied": True,
        "split_version": split_version,
        "measured_pair_count": len(outcomes),
        "accepted_treated_keys": sorted(accepted),
        "outcomes": outcomes,
        "mlflow": {
            "experiment": experiment_name,
            "run_kind": MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
            "tags": {
                RUN_KIND_TAG: MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
                MODEL_TYPE_TAG: "battery model type per pair",
                MLFLOW_TAG_TREATMENT_VARIANT: "declared variant key",
            },
        },
    }
    if acceptance is not None:
        # The interpretive half of the D039 acceptance branch (T03): the
        # top-level record the artifact's consumers read alongside the
        # measured outcomes. Omitted only when no branch was executed.
        payload["acceptance"] = dict(acceptance)
    return payload


def write_gate_verdicts_artifact(
    *,
    path: str | Path = DEFAULT_GATE_VERDICTS_PATH,
    candidate_keys=None,
    model_types=None,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    split_version: str = REAL_SPLIT_VERSION,
    acceptance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build and atomically write the verdicts artifact; return the payload."""
    payload = build_gate_verdicts_artifact(
        candidate_keys=candidate_keys,
        model_types=model_types,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        split_version=split_version,
        acceptance=acceptance,
    )
    destination = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(destination)
    logger.info(
        "Wrote %d outcome(s) to %s (derived_from_mlflow=True)",
        len(payload["outcomes"]),
        destination,
    )
    return payload
