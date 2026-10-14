"""MLflow parity probe: data-methods vs leaderboard/selection/stores (M006 S03/T04).

Why this module exists
----------------------
Decision D025 forbids any hand-assembled number in the generated report, so
the overlapping quantities between ``reports/data-methods.md`` (parts B-F),
its generated artifacts, and the MLflow-derived ``reports/leaderboard.md`` /
``reports/selection.md`` must be *proven* to agree by an executable probe —
not by eyeballing two reports. This module is that probe: pure parity
functions over plain dicts/strings (no ``.gsd`` access anywhere), one
structured :class:`ParityReport` on success, and a loud
:class:`ParityProbeError` naming the offending quantity on any mismatch.

Overlap checks (the scout-inventoried overlap table):

- **alpha**: the plan gate's ``paired_significance_alpha`` equals
  :data:`PAIRED_SIGNIFICANCE_ALPHA`, and the ``alpha=0.05`` token appears in
  both the committed ``selection.md`` text and the rendered data-methods
  text (parts E/F render it from the same constant).
- **gate geometry**: the plan gate's ``min_delta_roc_auc``/``metric`` equal
  :data:`GATE_MIN_DELTA_ROC_AUC`/:data:`GATE_METRIC`, and every verdicts
  outcome carries ``n_folds == N_SPLITS`` and
  ``n_repeats == DEFAULT_N_REPEATS``.
- **split sizes**: ``treatment_audit`` fitted ``row_count`` (734 train) plus
  the leaderboard ``N`` (184 test) equals the manifest ``row_count`` (918).
- **tuned-params key coverage**: the battery table's ``tuned_space_keys``
  for the winner ⊇ the winner's ``tuned_params`` keys from
  ``leading_model.json``.
- **calibration flag**: the ``calibration_did_not_improve_brier`` string is
  identical across the rendered part E text and the selection report flags.
- **metric schema version**: every ``run_kind=treatment-candidate`` /
  ``split_version=v1`` run's ``metric_schema_version`` tag equals
  :data:`METRIC_SCHEMA_VERSION`.
- **MLflow generation**: :func:`verify_gate_generation` proves every declared
  pair is stored exactly once with params/tags matching the declared
  constants (alpha/min_delta/gate_metric parity), and the verdicts artifact's
  outcomes equal a fresh store-derived build outcome-for-outcome.

Runtime surface: :func:`run_parity` reads the committed reports tree plus the
real store; ``python -m heart.reporting.data_methods.parity`` exits 0 on a
green probe and 1 with the offending quantity named on red.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from heart.config import REPORTS_DIR
from heart.data.split import N_SPLITS
from heart.eval.contract import METRIC_SCHEMA_VERSION
from heart.eval.repeated_cv import DEFAULT_N_REPEATS
from heart.eval.treatment_declared import (
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    PAIRED_SIGNIFICANCE_ALPHA,
)
from heart.reporting.data_methods.gate_verdicts import (
    GATE_VERDICTS_ARTIFACT,
    GateVerdictsError,
    build_gate_verdicts_artifact,
    collect_candidate_runs,
    verify_gate_generation,
)
from heart.reporting.data_methods.narrative import CALIBRATION_FLAG_NAME
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT

logger = logging.getLogger(__name__)

__all__ = [
    "ParityProbeError",
    "ParityCheck",
    "ParityReport",
    "check_alpha",
    "check_gate_geometry",
    "check_split_sizes",
    "check_tuned_key_coverage",
    "check_calibration_flag",
    "check_metric_schema_version",
    "check_candidate_run_constants",
    "check_verdicts_match_store",
    "parse_leaderboard_n",
    "run_parity",
    "main",
]


class ParityProbeError(Exception):
    """Two surfaces that must agree do not — the message names the offender.

    Raised (never swallowed) with the offending quantity, both sides, and
    where each side came from, so a red probe is actionable without a
    debugger.
    """


@dataclass(frozen=True)
class ParityCheck:
    """One green overlap assertion: what was compared and what both sides said."""

    name: str
    expected: str
    actual: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Every stored check is a passing assertion (failures raise)."""
        return True


@dataclass
class ParityReport:
    """The green-probe record: every check that passed, in run order."""

    checks: list[ParityCheck] = field(default_factory=list)
    measured_pair_count: int = 0

    def summary(self) -> str:
        names = ", ".join(check.name for check in self.checks)
        return (
            f"parity PASS: {len(self.checks)} check(s) "
            f"({names}); {self.measured_pair_count} measured pair(s)"
        )


def _alpha_token() -> str:
    return f"alpha={PAIRED_SIGNIFICANCE_ALPHA:g}"


def check_alpha(
    plan_gate: Mapping[str, Any],
    selection_text: str,
    methods_text: str,
) -> ParityCheck:
    """Prove the significance alpha is one number on all three surfaces."""
    declared = plan_gate.get("paired_significance_alpha")
    if declared != PAIRED_SIGNIFICANCE_ALPHA:
        raise ParityProbeError(
            "parity alpha: treatment_plan.json gate paired_significance_alpha="
            f"{declared!r} contradicts PAIRED_SIGNIFICANCE_ALPHA="
            f"{PAIRED_SIGNIFICANCE_ALPHA!r}."
        )
    token = _alpha_token()
    if token not in selection_text:
        raise ParityProbeError(
            f"parity alpha: reports/selection.md carries no {token!r} token; "
            "the selection flags must cite the same alpha the gate measured."
        )
    if token not in methods_text:
        raise ParityProbeError(
            f"parity alpha: reports/data-methods.md carries no {token!r} token; "
            "parts E/F render the alpha from PAIRED_SIGNIFICANCE_ALPHA."
        )
    return ParityCheck(
        name="alpha",
        expected=f"plan={PAIRED_SIGNIFICANCE_ALPHA:g}",
        actual=f"selection+methods carry {token!r}",
    )


def check_gate_geometry(
    plan_gate: Mapping[str, Any],
    outcomes: list[Mapping[str, Any]],
) -> ParityCheck:
    """Prove the plan gate constants match code and every outcome geometry."""
    min_delta = plan_gate.get("min_delta_roc_auc")
    if min_delta != GATE_MIN_DELTA_ROC_AUC:
        raise ParityProbeError(
            "parity gate threshold: treatment_plan.json gate "
            f"min_delta_roc_auc={min_delta!r} contradicts "
            f"GATE_MIN_DELTA_ROC_AUC={GATE_MIN_DELTA_ROC_AUC!r}."
        )
    metric = plan_gate.get("metric")
    if metric != GATE_METRIC:
        raise ParityProbeError(
            "parity gate metric: treatment_plan.json gate metric="
            f"{metric!r} contradicts GATE_METRIC={GATE_METRIC!r}."
        )
    if not outcomes:
        raise ParityProbeError(
            "parity fold geometry: the verdicts artifact carries no outcomes; "
            "expected one outcome per measured (candidate, model_type) pair."
        )
    for outcome in outcomes:
        treated = outcome.get("treated_key", outcome.get("candidate_key", "?"))
        if outcome.get("n_folds") != N_SPLITS or outcome.get("n_repeats") != (
            DEFAULT_N_REPEATS
        ):
            raise ParityProbeError(
                f"parity fold geometry: outcome {treated!r} records "
                f"n_folds={outcome.get('n_folds')!r} "
                f"n_repeats={outcome.get('n_repeats')!r}, contradicting "
                f"N_SPLITS={N_SPLITS} x DEFAULT_N_REPEATS={DEFAULT_N_REPEATS}."
            )
    return ParityCheck(
        name="gate-geometry",
        expected=f"delta=+{GATE_MIN_DELTA_ROC_AUC:.4f} metric={GATE_METRIC} "
        f"{N_SPLITS}x{DEFAULT_N_REPEATS}",
        actual=f"{len(outcomes)} outcome(s) share that geometry",
    )


def check_split_sizes(
    audit_row_count: int,
    leaderboard_n: int,
    total_row_count: int,
) -> ParityCheck:
    """Prove train rows (audit) + test N (leaderboard) = manifest rows."""
    if audit_row_count + leaderboard_n != total_row_count:
        raise ParityProbeError(
            "parity split sizes: treatment_audit fitted row_count="
            f"{audit_row_count!r} + leaderboard N={leaderboard_n!r} = "
            f"{audit_row_count + leaderboard_n!r}, contradicting manifest "
            f"row_count={total_row_count!r}."
        )
    return ParityCheck(
        name="split-sizes",
        expected=f"{audit_row_count}+{leaderboard_n}={total_row_count}",
        actual=f"{audit_row_count}+{leaderboard_n}={total_row_count}",
    )


def check_tuned_key_coverage(
    audit_models: list[Mapping[str, Any]],
    winner_model_type: str,
    leading_tuned_keys: list[str],
) -> ParityCheck:
    """Prove the battery table covers every winner tuned-param key."""
    entry = next(
        (model for model in audit_models if model.get("model_type") == winner_model_type),
        None,
    )
    if entry is None:
        raise ParityProbeError(
            "parity tuned-params: treatment_audit.json has no models entry for "
            f"winner {winner_model_type!r}."
        )
    covered = set(entry.get("tuned_space_keys", []))
    missing = sorted(set(leading_tuned_keys) - covered)
    if missing:
        raise ParityProbeError(
            "parity tuned-params: battery table tuned_space_keys for "
            f"{winner_model_type!r} is missing winner tuned_params key(s) "
            f"{missing}; the battery table must cover the selection winner."
        )
    return ParityCheck(
        name="tuned-params",
        expected=f"battery covers {sorted(leading_tuned_keys)}",
        actual=f"battery covers {sorted(leading_tuned_keys)}",
    )


def check_calibration_flag(
    selection_text: str,
    part_e_text: str,
    flag: str = CALIBRATION_FLAG_NAME,
) -> ParityCheck:
    """Prove the calibration honesty flag string is identical on both sides."""
    if flag not in selection_text:
        raise ParityProbeError(
            f"parity calibration flag: reports/selection.md carries no "
            f"{flag!r}; the part E pointer must name a flag the selection "
            "report actually records."
        )
    if flag not in part_e_text:
        raise ParityProbeError(
            f"parity calibration flag: the rendered part E text carries no "
            f"{flag!r}; part E must state the pipeline's honesty flag."
        )
    return ParityCheck(
        name="calibration-flag",
        expected=flag,
        actual=flag,
    )


def check_metric_schema_version(schema_by_run: Mapping[str, str]) -> ParityCheck:
    """Prove every candidate run's schema tag equals the contract constant."""
    if not schema_by_run:
        raise ParityProbeError(
            "parity metric_schema_version: no candidate runs supplied; "
            "expected one schema tag per measured pair."
        )
    for treated_key in sorted(schema_by_run):
        if str(schema_by_run[treated_key]) != str(METRIC_SCHEMA_VERSION):
            raise ParityProbeError(
                "parity metric_schema_version: run "
                f"{treated_key!r} carries "
                f"metric_schema_version={schema_by_run[treated_key]!r}, "
                "contradicting heart.eval.contract.METRIC_SCHEMA_VERSION="
                f"{METRIC_SCHEMA_VERSION!r}."
            )
    return ParityCheck(
        name="metric-schema-version",
        expected=str(METRIC_SCHEMA_VERSION),
        actual=f"{len(schema_by_run)} run(s) carry {METRIC_SCHEMA_VERSION!r}",
    )


def check_candidate_run_constants(
    runs: list[Mapping[str, Any]],
) -> ParityCheck:
    """Prove each candidate run's gate params equal the declared constants.

    Pure over ``{"treated_key": ..., "params": {...}}`` dicts, so unit tests
    cover the MLflow half without touching a store; the real probe feeds
    these from :func:`collect_candidate_runs`.
    """
    if not runs:
        raise ParityProbeError(
            "parity candidate params: no candidate runs supplied; expected "
            "one run per measured pair."
        )
    for run in runs:
        treated = run.get("treated_key", "?")
        params = run.get("params", {})
        if params.get("alpha") != f"{PAIRED_SIGNIFICANCE_ALPHA:g}":
            raise ParityProbeError(
                f"parity candidate params: run {treated!r} param alpha="
                f"{params.get('alpha')!r} contradicts "
                f"PAIRED_SIGNIFICANCE_ALPHA={PAIRED_SIGNIFICANCE_ALPHA!r}."
            )
        if params.get("min_delta") != f"{GATE_MIN_DELTA_ROC_AUC:g}":
            raise ParityProbeError(
                f"parity candidate params: run {treated!r} param min_delta="
                f"{params.get('min_delta')!r} contradicts "
                f"GATE_MIN_DELTA_ROC_AUC={GATE_MIN_DELTA_ROC_AUC!r}."
            )
        if params.get("gate_metric") != GATE_METRIC:
            raise ParityProbeError(
                f"parity candidate params: run {treated!r} param gate_metric="
                f"{params.get('gate_metric')!r} contradicts "
                f"GATE_METRIC={GATE_METRIC!r}."
            )
    return ParityCheck(
        name="candidate-params",
        expected=f"alpha={PAIRED_SIGNIFICANCE_ALPHA:g} "
        f"min_delta={GATE_MIN_DELTA_ROC_AUC:g} metric={GATE_METRIC}",
        actual=f"{len(runs)} run(s) match those constants",
    )


def check_verdicts_match_store(
    artifact_outcomes: list[Mapping[str, Any]],
    fresh_outcomes: list[Mapping[str, Any]],
) -> ParityCheck:
    """Prove the verdicts file equals a fresh store-derived build, pair by pair."""
    fresh_by_key = {outcome["treated_key"]: outcome for outcome in fresh_outcomes}
    artifact_by_key = {outcome["treated_key"]: outcome for outcome in artifact_outcomes}
    if set(artifact_by_key) != set(fresh_by_key):
        raise ParityProbeError(
            "parity verdicts-vs-store: treated_key coverage differs — "
            f"artifact-only={sorted(set(artifact_by_key) - set(fresh_by_key))} "
            f"store-only={sorted(set(fresh_by_key) - set(artifact_by_key))}."
        )
    for treated_key in sorted(artifact_by_key):
        if dict(artifact_by_key[treated_key]) != dict(fresh_by_key[treated_key]):
            raise ParityProbeError(
                f"parity verdicts-vs-store: outcome {treated_key!r} in "
                f"{GATE_VERDICTS_ARTIFACT} differs from the store-derived "
                "build; the artifact must be a verbatim store projection."
            )
    return ParityCheck(
        name="verdicts-vs-store",
        expected=f"{len(fresh_by_key)} store outcome(s)",
        actual=f"artifact matches all {len(fresh_by_key)}",
    )


_LEADERBOARD_RANK_ROW_RE: re.Pattern[str] = re.compile(r"^\| (\d+) \|")


def parse_leaderboard_n(leaderboard_text: str) -> int:
    """Parse the test-set N from the leaderboard table (every row agrees)."""
    sizes: list[int] = []
    for line in leaderboard_text.splitlines():
        if not _LEADERBOARD_RANK_ROW_RE.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        try:
            sizes.append(int(cells[-1]))
        except (ValueError, IndexError) as exc:
            raise ParityProbeError(
                "parity split sizes: leaderboard table row has no integer N "
                f"in its last cell: {line!r}."
            ) from exc
    if not sizes:
        raise ParityProbeError(
            "parity split sizes: reports/leaderboard.md carries no ranked "
            "table rows; cannot read the test-set N."
        )
    if len(set(sizes)) != 1:
        raise ParityProbeError(
            f"parity split sizes: leaderboard rows disagree on N: "
            f"{sorted(set(sizes))}."
        )
    return sizes[0]


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ParityProbeError(f"parity input: expected artifact missing: {path}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParityProbeError(
            f"parity input: artifact is not valid JSON: {path}: {exc}."
        ) from exc


def run_parity(
    *,
    reports_dir: str | Path = REPORTS_DIR,
    tracking_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    check_store: bool = True,
) -> ParityReport:
    """Run every parity check against a reports tree (plus the store).

    Parameters
    ----------
    reports_dir:
        Directory holding ``selection.md``, ``leaderboard.md``,
        ``leading_model.json``, ``data-methods.md`` and the ``data-methods/``
        artifact directory. Tests point this at a tmp-copied artifact set;
        the real probe uses the committed ``reports/`` tree.
    check_store:
        When True (the real probe), the MLflow half runs against the real
        store: generation verification, schema tags, and a fresh-build
        comparison. Tests pass False and cover the store half through the
        pure ``check_*`` functions instead (no real-store reads in tests).
    """
    reports = Path(reports_dir)
    methods_dir = reports / "data-methods"
    plan = _read_json(methods_dir / "treatment_plan.json")
    audit = _read_json(methods_dir / "treatment_audit.json")
    verdicts = _read_json(methods_dir / GATE_VERDICTS_ARTIFACT)
    meta = _read_json(methods_dir / "meta.json")
    leading = _read_json(reports / "leading_model.json")
    for name in ("selection.md", "leaderboard.md", "data-methods.md"):
        if not (reports / name).is_file():
            raise ParityProbeError(f"parity input: expected report missing: {reports / name}.")
    selection_text = (reports / "selection.md").read_text(encoding="utf-8")
    leaderboard_text = (reports / "leaderboard.md").read_text(encoding="utf-8")
    methods_text = (reports / "data-methods.md").read_text(encoding="utf-8")

    checks = [
        check_alpha(plan["gate"], selection_text, methods_text),
        check_gate_geometry(plan["gate"], list(verdicts["outcomes"])),
        check_split_sizes(
            int(audit["preprocessing"]["fitted"]["row_count"]),
            parse_leaderboard_n(leaderboard_text),
            int(meta["row_count"]),
        ),
        check_tuned_key_coverage(
            list(audit["models"]),
            str(leading["model_type"]),
            list(leading["tuned_params"].keys()),
        ),
        check_calibration_flag(selection_text, methods_text),
    ]
    measured = len(verdicts["outcomes"])

    if check_store:
        try:
            verification = verify_gate_generation(
                tracking_dir=tracking_dir, experiment_name=experiment_name
            )
            runs = collect_candidate_runs(
                tracking_dir=tracking_dir, experiment_name=experiment_name
            )
        except GateVerdictsError as exc:
            raise ParityProbeError(
                f"parity store generation: the stored candidate generation "
                f"failed verification: {exc}."
            ) from exc
        checks.append(
            check_candidate_run_constants(
                [
                    {"treated_key": run.treated_key, "params": dict(run.params)}
                    for run in runs
                ]
            )
        )
        checks.append(
            check_metric_schema_version(
                {run.treated_key: run.tags.get("metric_schema_version", "") for run in runs}
            )
        )
        try:
            fresh = build_gate_verdicts_artifact(
                tracking_dir=tracking_dir, experiment_name=experiment_name
            )
        except GateVerdictsError as exc:
            raise ParityProbeError(
                f"parity store rebuild: a fresh store-derived build failed: {exc}."
            ) from exc
        checks.append(
            check_verdicts_match_store(
                list(verdicts["outcomes"]), list(fresh["outcomes"])
            )
        )
        if int(verification["run_count"]) != measured:
            raise ParityProbeError(
                f"parity store count: the store holds "
                f"{verification['run_count']} candidate run(s) but the "
                f"verdicts artifact records {measured} outcome(s)."
            )

    report = ParityReport(checks=checks, measured_pair_count=measured)
    logger.info("parity probe: %s", report.summary())
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: green probe prints PASS and exits 0; red names the offender, exits 1."""
    _ = argv
    try:
        report = run_parity()
    except ParityProbeError as exc:
        print(f"PARITY=FAIL {exc}")
        return 1
    print(f"PARITY=PASS {report.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
