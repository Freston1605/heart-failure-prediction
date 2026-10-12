"""Treatment-candidate gate runner: paired measurement, verdict, report (D034/D036).

This module is the M006 gate engine's runner (decision D034 concern 3). It
measures every declared non-incumbent candidate from
:mod:`heart.eval.treatment_declared` through ONE evaluation path — the shared
:func:`heart.eval.contract.evaluate` producer inside repeated stratified CV
(:func:`heart.eval.repeated_cv.run_repeated_cv`) — compares each candidate
paired against the committed incumbent baseline with the pre-declared
significance layer (:func:`heart.eval.significance.compare_repeated_cv`,
corrected resampled t-test at the declared alpha), and reaches a verdict whose
logic literally mirrors the pre-declared sentence
:func:`heart.eval.treatment_declared.gate_text`.

What it produces
----------------
* :class:`CandidateMeasurement` — one paired repeated-CV measurement of
  (candidate, model_type): treated and incumbent
  :class:`~heart.eval.repeated_cv.RepeatedCVResult` on the *same*
  :class:`~heart.eval.repeated_cv.RepeatedCVConfig` (same fold seeds -> paired
  folds), the :class:`~heart.eval.significance.PairwiseComparison`, and — for
  mechanism=sampling candidates — one
  :class:`~heart.eval.golden_rule.ResampleLineage` per fold plus the fold
  partitions the golden-rule assertions ran against.
* :func:`verdict_treatment` — the three clause booleans of the pre-declared
  gate text, evaluated clause by clause, plus the verdict string; the verdict
  text embeds ``gate_text()`` verbatim so report readers see the exact
  pre-registered sentence.
* :func:`run_gate_core` — the declared-order measurement loop into a
  :class:`GateReport`, including the incumbent rung-0 entry recorded as
  *documented-not-re-measured* (no measurement, no MLflow number) and, per
  measured pair, the pooled-OOF canonical metric dict of D036
  (:func:`heart.eval.repeated_cv.pooled_oof_metrics` — ``compute_metric_dict``
  on the pooled held-out predictions, then ``validate_metric_dict``, which
  must pass; there is no ``validate=False`` escape hatch here).
* :func:`run_gate` — the D036 MLflow logging path: with ``log_to_mlflow=True``
  every measured pair emits EXACTLY ONE MLflow run through
  :func:`heart.tracking.run.log_evaluation_run` (canonical pooled-OOF metric
  dict schema-checked, never ``validate=False``), tagged
  ``run_kind=treatment-candidate`` with the declared
  ``treatment_variant`` tag — excluded from the leaderboard's
  ``run_kind=final`` collection by construction — plus a
  ``gate_report.json`` artifact embedding the verbatim ``gate_text()``
  sentence. ``log_to_mlflow=False`` is the report-only dry mode: the same
  :class:`GateReport` with ``run_id=None`` on every outcome, and ZERO runs
  written.

Golden-rule wiring (D034/D035/T02)
----------------------------------
For mechanism=sampling candidates this runner owns its fold loop
(:func:`_run_sampling_repeated_cv`): the built imblearn pipeline is fit on the
fold's training rows only (imblearn applies ``fit_resample`` internally, on
the transformed fold-train matrix — never on validation or test rows), one
id-level :class:`~heart.eval.golden_rule.ResampleLineage` is recorded per fold
from the fitted sampler's own bookkeeping, and
:func:`heart.eval.golden_rule.assert_golden_rule` runs against the fold
partition; any violation raises :class:`GoldenRuleError` LOUDLY (T05 adds the
runner-level negative test proving a pre-split-resampled fixture fails).

Verdict semantics (pre-declared, no escape hatches)
---------------------------------------------------
``gate_text()`` declares exactly three clauses:

(a) the candidate's repeated-CV ``roc_auc`` delta over the incumbent is at
    least :data:`~heart.eval.treatment_declared.GATE_MIN_DELTA_ROC_AUC`
    (+0.0100);
(b) the paired significance layer (the corrected resampled t-test) reads that
    delta as significant at
    :data:`~heart.eval.treatment_declared.PAIRED_SIGNIFICANCE_ALPHA`;
(c) the winner-standing rule: the treated model's standing versus the
    baseline and the runner-up on the leaderboard does not worsen
    (:data:`~heart.eval.treatment_declared.GATE_WINNER_STANDING_RULE`).

The gate verdict is ``"accepted"`` ONLY when all three clauses evaluate
True. Clause (c) is a pure comparator over a caller-supplied standings
mapping ``{gate_key -> (primary_mean, rank)}`` — the battery leaderboard
standings, which by construction includes the baseline and the runner-up
entries and embeds them in the ranks. It compares the treated variant's
standing (key ``treatment:<variant>/<model_type>``) against its incumbent's
standing (key ``incumbent:<model_type>``): clause (c) fails when the treated
model's rank is worse than its incumbent's rank (rank is 1-based, lower is
better; an equal rank is not "worse"). When no standings are supplied the
clause is recorded as ``None`` (NOT evaluated, NOT silently true) and a
measurement that passes (a) and (b) is reported as
``"rejected-by-incomplete-evidence"`` — never ``"accepted"``. Thresholds and
alpha come straight from the declared constants: :class:`GateConfig` mirrors
``PAIRED_SIGNIFICANCE_ALPHA`` as a READ-ONLY field (any override attempt
raises :class:`TreatmentGateError`) and carries no delta-threshold parameter
at all, so no parameter surface can move the gate's accept thresholds.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from heart.config import RANDOM_SEED
from heart.eval.contract import (
    PRIMARY_METRIC,
    _predict,
    compute_metric_dict,
    validate_metric_dict,
)
from heart.eval.golden_rule import (
    FoldPartition,
    ResampleLineage,
    allocate_synthetic_ids,
    assert_golden_rule,
)
from heart.eval.repeated_cv import (
    DEFAULT_N_FOLDS,
    DEFAULT_N_REPEATS,
    FoldFitError,
    FoldScore,
    RepeatedCVConfig,
    RepeatedCVResult,
    _check_folds_fit_classes,
    _coerce_frame_and_labels,
    _finalise_repeated_cv_result,
    run_repeated_cv,
)
from heart.eval.significance import PairwiseComparison, compare_repeated_cv
from heart.eval.treatment_declared import (
    BATTERY_MEMBERS,
    DECLARED_CANDIDATES,
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    GATE_WINNER_STANDING_RULE,
    INCUMBENT_CANDIDATE,
    MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
    MLFLOW_TAG_TREATMENT_VARIANT,
    PAIRED_SIGNIFICANCE_ALPHA,
    TreatmentCandidate,
    gate_text,
)
from heart.eval.treatment_variants import (
    TreatmentGateError,
    build_candidate,
    candidate_by_key,
    incumbent_factory,
    require_feature_frame,
)
from heart.models.registry import resolve_spec
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import MODEL_TYPE_TAG, RUN_KIND_TAG

logger = logging.getLogger(__name__)

__all__ = [
    "VERDICT_ACCEPTED",
    "VERDICT_REJECTED",
    "VERDICT_REJECTED_INCOMPLETE_EVIDENCE",
    "VERDICT_DOCUMENTED_NOT_RE_MEASURED",
    "TreatmentGateError",
    "GATE_SPLIT_VERSION",
    "GATE_REPORT_ARTIFACT",
    "FAMILY_TAG",
    "GateConfig",
    "CandidateMeasurement",
    "GateVerdict",
    "GateOutcome",
    "GateReport",
    "treatment_key_for",
    "incumbent_key_for",
    "standing_not_worsened",
    "verdict_treatment",
    "measure_pair",
    "run_gate",
    "run_gate_core",
]


# ---------------------------------------------------------------------------
# Verdict vocabulary (exactly four outcomes; never more, never silently true)
# ---------------------------------------------------------------------------

#: All three clauses evaluated True — the candidate treatment is accepted.
VERDICT_ACCEPTED: str = "accepted"

#: At least one evaluatable clause failed — the incumbent pipeline is kept.
VERDICT_REJECTED: str = "rejected"

#: Clauses (a)/(b) passed but clause (c) was never evaluated (no standings
#: supplied) — rejected for incomplete evidence, NOT accepted.
VERDICT_REJECTED_INCOMPLETE_EVIDENCE: str = "rejected-by-incomplete-evidence"

#: The incumbent rung-0 policy: described in the declaration, never re-measured.
VERDICT_DOCUMENTED_NOT_RE_MEASURED: str = "documented-not-re-measured"


# ---------------------------------------------------------------------------
# MLflow candidate-run logging (D036; the frozen tracking convention)
# ---------------------------------------------------------------------------

#: The ``split_version`` a fixture-level gate measurement logs under.
#:
#: S02's contract proof measures the declared candidates on the deterministic
#: tests/helpers_treatment.py fixture, which is NOT a real-data split version;
#: declaring the string ``"fixture"`` keeps the store honest (a reader can
#: immediately tell candidate measurement runs apart from the real-data
#: ``v1``-style battery runs). S03's real-store gate run supplies the real
#: declared split version instead (it calls the runner with its own value).
GATE_SPLIT_VERSION: str = "fixture"

#: Artifact logged alongside ``metrics.json`` / ``run_config.json`` on every
#: measured candidate run: the JSON-serializable :class:`GateReport`
#: projection, embedding the verbatim pre-declared ``gate_text()`` sentence.
GATE_REPORT_ARTIFACT: str = "gate_report.json"

#: MLflow tag key carrying the model family on a candidate run. Declared as a
#: literal here (the eval layer does not import heart.reporting); the
#: cross-check test asserts it matches
#: ``heart.reporting.leaderboard.FAMILY_TAG`` so it can never silently drift.
FAMILY_TAG: str = "family"


def treatment_key_for(candidate: TreatmentCandidate, model_type: str) -> str:
    """Stable gate key for one treated (candidate, model_type) pair."""
    return f"treatment:{candidate.variant_key}/{model_type}"


def incumbent_key_for(model_type: str) -> str:
    """Stable gate key for the committed incumbent pipeline of one model."""
    return f"incumbent:{model_type}"


# ---------------------------------------------------------------------------
# Configuration — thresholds live in treatment_declared, not here
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Declared gate measurement configuration.

    ``alpha`` deliberately REPEATS
    :data:`~heart.eval.treatment_declared.PAIRED_SIGNIFICANCE_ALPHA` as a
    read-only mirror (D034): the field exists so the gate's audit surface can
    show which alpha the significance text used, but there is NO parameter
    surface to change it or the delta threshold — the accept thresholds must
    come straight from :mod:`heart.eval.treatment_declared` constants, so any
    attempted override raises :class:`TreatmentGateError` at construction.

    ``seed``/``n_folds``/``n_repeats`` shape the *measurement* (repeated-CV
    geometry) and are freely configurable per run (tests and S03 may shrink
    them competitively); they are not accept thresholds.
    """

    seed: int = RANDOM_SEED
    n_folds: int = DEFAULT_N_FOLDS
    n_repeats: int = DEFAULT_N_REPEATS
    alpha: float = PAIRED_SIGNIFICANCE_ALPHA

    def __post_init__(self) -> None:
        for name in ("n_folds", "n_repeats", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TreatmentGateError(
                    f"GateConfig.{name} must be an integer, got "
                    f"{type(value).__name__} ({value!r})."
                )
        if self.n_folds < 2:
            raise TreatmentGateError(
                f"GateConfig.n_folds must be at least 2, got {self.n_folds}."
            )
        if self.n_repeats < 1:
            raise TreatmentGateError(
                f"GateConfig.n_repeats must be at least 1, got {self.n_repeats}."
            )
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not float(self.alpha) == float(PAIRED_SIGNIFICANCE_ALPHA)
        ):
            raise TreatmentGateError(
                f"GateConfig.alpha is a READ-ONLY mirror of "
                f"heart.eval.treatment_declared.PAIRED_SIGNIFICANCE_ALPHA "
                f"({PAIRED_SIGNIFICANCE_ALPHA!r}); the gate has no parameter "
                f"surface to change it, so overriding alpha={self.alpha!r} is "
                "rejected outright."
            )

    # -- read-only mirrors of the declared accept thresholds ----------------

    @property
    def min_delta(self) -> float:
        """The declared minimum delta, straight from the declaration module."""
        return GATE_MIN_DELTA_ROC_AUC

    @property
    def gate_metric(self) -> str:
        """The declared gate metric (``roc_auc``; matches ``PRIMARY_METRIC``)."""
        return GATE_METRIC

    # -- projections --------------------------------------------------------

    def to_repeated_cv_config(self) -> RepeatedCVConfig:
        """The :class:`RepeatedCVConfig` the paired folds are carved with.

        Incumbent and treated candidates are run on this exact config so
        ``repeat``/``fold`` keys pair 1:1 and the significance test is
        genuinely paired.
        """
        return RepeatedCVConfig(
            n_folds=self.n_folds, n_repeats=self.n_repeats, seed=self.seed
        )

    def fingerprint(self) -> str:
        return f"{self.to_repeated_cv_config().fingerprint()}@alpha={self.alpha:g}"

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "alpha": self.alpha,
            "min_delta": self.min_delta,
            "gate_metric": self.gate_metric,
        }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateMeasurement:
    """One paired (candidate, model_type) measurement through the gate.

    ``treated`` and ``incumbent`` share one :class:`RepeatedCVConfig`, so
    their ``per_fold`` entries pair by ``(repeat, fold)`` key — the pairing
    :func:`heart.eval.significance.compare_repeated_cv` validates loudly.
    ``fold_lineages``/``fold_partitions`` are populated only for
    mechanism=sampling candidates (one per measured fold, in loop order) and
    are the golden-rule evidence: each lineage has already passed
    :func:`heart.eval.golden_rule.assert_golden_rule` against its partition.
    """

    candidate: TreatmentCandidate
    model_type: str
    treated: RepeatedCVResult
    incumbent: RepeatedCVResult
    comparison: PairwiseComparison
    fold_lineages: tuple[ResampleLineage, ...] = ()
    fold_partitions: tuple[FoldPartition, ...] = ()

    @property
    def treated_key(self) -> str:
        return treatment_key_for(self.candidate, self.model_type)

    @property
    def incumbent_key(self) -> str:
        return incumbent_key_for(self.model_type)

    @property
    def treated_primary_mean(self) -> float:
        return float(self.treated.aggregate.mean[GATE_METRIC])

    @property
    def incumbent_primary_mean(self) -> float:
        return float(self.incumbent.aggregate.mean[GATE_METRIC])

    @property
    def delta(self) -> float:
        """Treated minus incumbent repeated-CV primary mean (clause (a) input)."""
        return self.treated_primary_mean - self.incumbent_primary_mean

    @property
    def significant(self) -> bool:
        """The corrected resampled t-test's significance reading (clause (b))."""
        return bool(self.comparison.corrected_t.significant)

    def pooled_oof_metric_dict(self) -> dict[str, object]:
        """The D036 pooled-OOF canonical metric dict for the treated model.

        Computed via
        :func:`heart.eval.repeated_cv.pooled_oof_metrics` (``compute_metric_dict``
        on pooled held-out predictions then ``validate_metric_dict``, which
        must pass) and ready to hand straight to
        ``heart.tracking.run.log_evaluation_run`` in T05.
        """
        from heart.eval.repeated_cv import pooled_oof_metrics

        return pooled_oof_metrics(self.treated)


# ---------------------------------------------------------------------------
# Sampling fold loop — the runner-level golden-rule enforcement point
# ---------------------------------------------------------------------------


def _build_fold_lineage(
    model: object,
    variant_key: str,
    train_idx: np.ndarray,
    *,
    n_original: int,
    counter: int,
) -> ResampleLineage:
    """Record one fold's id-level resampling lineage from the fitted pipeline.

    The declared sampling candidate is an imblearn pipeline whose ``sampler``
    step ran ``fit_resample`` on the fold's transformed TRAINING matrix only.
    The lineage is read back from the fitted sampler itself, which keeps the
    lineage a record of what actually happened to the fold's rows:

    - :class:`~imblearn.under_sampling.RandomUnderSampler` /
      :class:`~imblearn.over_sampling.RandomOverSampler`: the sampler exposes
      ``sample_indices_`` (output row -> input row position), so kept and
      duplicated REAL fold rows keep their real ids exactly and synthetic ids
      stay empty (duplicating real rows creates no synthetic identity).
    - :class:`~imblearn.over_sampling.SMOTE` / ``ADASYN``: every original
      training row is kept and the fitted ``sampling_strategy_`` gives the
      per-class count of appended synthetic vectors; those vectors live in
      the RESERVED id space (``>= n_original``, allocated per fold via
      :func:`~heart.eval.golden_rule.allocate_synthetic_ids` with a running
      counter so ids never repeat across folds) and the numeric vectors are
      never part of the lineage.

    Raises :class:`TreatmentGateError` if the fitted sampler reports an
    input-row position outside the fold's training rows — anything else would
    itself be a golden-rule breach and must never reach the assertions.
    """
    sampler = getattr(getattr(model, "named_steps", None), "get", lambda _k: None)(
        "sampler"
    )
    if sampler is None:
        raise TreatmentGateError(
            f"Variant {variant_key!r}: the sampling candidate has no 'sampler' "
            "step; the gate runner records lineages from the fitted sampler."
        )
    train_idx = np.asarray(train_idx, dtype=int)
    n_train = int(train_idx.size)
    original_ids = tuple(int(i) for i in train_idx)

    sample_indices = getattr(sampler, "sample_indices_", None)
    if sample_indices is not None:
        # Random under/over-sampling: real rows only, ids exact.
        indices = np.asarray(sample_indices, dtype=int)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_train):
            raise TreatmentGateError(
                f"Variant {variant_key!r}: the fitted sampler reported an "
                f"input-row position outside the fold's training rows "
                f"(positions {int(indices.min())}..{int(indices.max())} for "
                f"{n_train} training rows); resampling did not run on the "
                "fold's training rows alone."
            )
        resampled_ids = tuple(int(train_idx[int(p)]) for p in indices)
        synthetic_ids: tuple[int, ...] = ()
        synthetic_count = 0
    else:
        # SMOTE/ADASYN: all originals kept; synthetic vectors counted from the
        # fitted strategy and given reserved-space ids (never real ids).
        strategy = getattr(sampler, "sampling_strategy_", None)
        if not isinstance(strategy, Mapping):
            raise TreatmentGateError(
                f"Variant {variant_key!r}: the fitted sampler "
                f"({type(sampler).__name__}) exposes neither sample_indices_ "
                "nor a fitted sampling_strategy_; the lineage cannot be "
                "recorded honestly."
            )
        synthetic_count = int(sum(int(v) for v in strategy.values()))
        resampled_ids = original_ids
        synthetic_ids = allocate_synthetic_ids(
            n_original, synthetic_count, counter=counter
        )
    lineage = ResampleLineage(
        variant_key=variant_key,
        mechanism=type(sampler).__name__,
        n_original=int(n_original),
        original_train_ids=original_ids,
        resampled_train_ids=resampled_ids,
        synthetic_ids=synthetic_ids,
        synthetic_count=synthetic_count,
    )
    logger.debug(
        "Fold lineage recorded for %s: mechanism=%s, n_train=%d, "
        "n_resampled=%d, synthetic=%d",
        variant_key,
        lineage.mechanism,
        n_train,
        len(resampled_ids),
        synthetic_count,
    )
    return lineage


def _run_sampling_repeated_cv(
    model_name: str,
    candidate: TreatmentCandidate,
    model_type: str,
    X: pd.DataFrame,
    y: Sequence[object],
    config: RepeatedCVConfig,
    fit_params: Mapping[str, object] | None = None,
) -> tuple[RepeatedCVResult, tuple[ResampleLineage, ...], tuple[FoldPartition, ...]]:
    """Fold loop for mechanism=sampling candidates (golden-rule owner).

    Mirrors :func:`heart.eval.repeated_cv.run_repeated_cv` fold-for-fold —
    same :class:`RepeatedCVConfig`, same :class:`~sklearn.model_selection.
    StratifiedKFold` construction, same (repeat, fold) keys — so the sampling
    candidate's folds pair exactly with the incumbent's. For every fold it:

    1. builds the declared variant (the imblearn pipeline) and fits it on the
       fold's TRAINING rows only (imblearn applies ``fit_resample`` internally,
       on the transformed fold-train matrix);
    2. records the fold's :class:`~heart.eval.golden_rule.ResampleLineage`
       from the fitted sampler and the fold's
       :class:`~heart.eval.golden_rule.FoldPartition`;
    3. runs :func:`~heart.eval.golden_rule.assert_golden_rule` (pre-split
       detector first) — any violation raises :class:`GoldenRuleError` LOUDLY,
       aborting the measurement before any number is produced;
    4. scores the held-out rows through the shared contract producer
       (``compute_metric_dict`` + ``validate_metric_dict`` on the captured
       held-out predictions) and pools them for the D036 metric dict.
    """
    features, labels = _coerce_frame_and_labels(X, y)
    _check_folds_fit_classes(config, labels)
    n_original = int(len(features))

    def factory() -> object:
        return build_candidate(candidate, model_type, fit_params)

    fold_scores: list[FoldScore] = []
    lineages: list[ResampleLineage] = []
    partitions: list[FoldPartition] = []
    pooled_true: list[np.ndarray] = []
    pooled_proba: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    synthetic_counter = 0

    for repeat in range(config.n_repeats):
        splitter = StratifiedKFold(
            n_splits=config.n_folds,
            shuffle=True,
            random_state=config.fold_seed(repeat),
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            try:
                model = factory()
                model.fit(features.iloc[train_idx], labels[train_idx])
            except Exception as exc:  # noqa: BLE001 - fail named, not bare
                raise FoldFitError(
                    f"Candidate {model_name!r} failed to build or fit on "
                    f"repeat {repeat} fold {fold} ({config.fingerprint()}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            # Golden rule: record the fold's id-level lineage and assert it
            # LOUDLY before any metric exists. A leaky lineage must never
            # produce a gate number.
            lineage = _build_fold_lineage(
                model,
                candidate.variant_key,
                train_idx,
                n_original=n_original,
                counter=synthetic_counter,
            )
            synthetic_counter += lineage.synthetic_count
            partition = FoldPartition(
                fold_id=f"{model_name}/repeat{repeat}/fold{fold}",
                train_ids=tuple(int(i) for i in train_idx),
                validation_ids=tuple(int(i) for i in test_idx),
            )
            assert_golden_rule(lineage, partition)
            lineages.append(lineage)
            partitions.append(partition)

            # Contract scoring path on the held-out rows (predictions are
            # captured for the D036 pooled-OOF metric dict).
            predictions, probabilities = _predict(model, features.iloc[test_idx])
            metrics = compute_metric_dict(labels[test_idx], probabilities, predictions)
            validate_metric_dict(metrics)
            fold_scores.append(
                FoldScore(
                    model_name=model_name,
                    repeat=repeat,
                    fold=fold,
                    n_train=int(train_idx.size),
                    n_test=int(test_idx.size),
                    metrics=metrics,
                )
            )
            pooled_true.append(np.asarray(labels[test_idx]))
            pooled_proba.append(np.asarray(probabilities, dtype=float))
            pooled_pred.append(np.asarray(predictions))

    result = _finalise_repeated_cv_result(
        model_name,
        config,
        fold_scores,
        pooled=(np.concatenate(pooled_true), np.concatenate(pooled_proba), np.concatenate(pooled_pred)),
    )
    logger.info(
        "Sampling candidate %s measured on %s with per-fold golden-rule "
        "lineages (%d folds asserted clean).",
        model_name,
        config.fingerprint(),
        len(lineages),
    )
    return result, tuple(lineages), tuple(partitions)


# ---------------------------------------------------------------------------
# Paired measurement
# ---------------------------------------------------------------------------


def _resolve_fit_params(
    fit_params: Mapping[str, Mapping[str, object]] | None, model_type: str
) -> dict[str, object] | None:
    """Resolve the per-``model_type`` estimator params for one paired measurement.

    ``fit_params`` maps ``{model_type: params-or-None}``. It configures the
    measurement's estimator hyperparameters — NEVER the gate's thresholds
    (alpha/delta are constants). Treated and incumbent of the same
    ``model_type`` always receive the SAME params so the pair differs only by
    the declared mechanism. Non-battery keys and non-mapping values raise
    loudly: named failures, never silent coercion.
    """
    if fit_params is None:
        return None
    if not isinstance(fit_params, Mapping):
        raise TreatmentGateError(
            f"fit_params must be a mapping of model_type -> params, got "
            f"{type(fit_params).__name__}."
        )
    unknown = [key for key in fit_params if key not in BATTERY_MEMBERS]
    if unknown:
        raise TreatmentGateError(
            f"fit_params names non-battery model type(s) {unknown}; battery "
            f"members are {list(BATTERY_MEMBERS)}."
        )
    params = fit_params.get(model_type)
    if params is None:
        return None
    if not isinstance(params, Mapping):
        raise TreatmentGateError(
            f"fit_params[{model_type!r}] must be a mapping of estimator "
            f"hyperparameters, got {type(params).__name__}."
        )
    return dict(params)


def measure_pair(
    candidate: TreatmentCandidate,
    model_type: str,
    X: pd.DataFrame,
    y: Sequence[object],
    *,
    config: GateConfig | None = None,
    fit_params: Mapping[str, Mapping[str, object]] | None = None,
) -> CandidateMeasurement:
    """Measure one (candidate, model_type) pair paired against the incumbent.

    The treated variant and the committed incumbent pipeline are run on the
    SAME :class:`RepeatedCVConfig` (same fold seeds) so their fold keys pair
    1:1, both with ``capture_predictions=True`` for the D036 pooled-OOF metric
    dict, and the corrected resampled t-test
    (:func:`heart.eval.significance.compare_repeated_cv`, McNemar omitted —
    the pre-declared significance layer is the corrected t-test) is computed
    at the declared alpha.

    ``fit_params`` optionally supplies per-``model_type`` estimator
    hyperparameters (``{model_type: params-or-None}``) applied IDENTICALLY to
    the treated and the incumbent build of that model — it configures the
    measurement's estimator, never the gate's thresholds (alpha/delta stay
    read-only mirrors). It exists because a committed spec's declared-default
    params can be unfittable on a given data shape: QDA at ``reg_param=0.0``
    is singular on any one-hot-expanded matrix (the level indicators are
    linearly dependent by construction), so a variance floor must be supplied
    for fixture fits (tests use ``{"qda": {"reg_param": 0.1}}``) and S03's
    real-store run supplies the battery's committed params.

    Raises
    ------
    TreatmentGateError
        Unknown candidate shape, the documented incumbent (rung 0 is
        described, not measured), a model_type outside the candidate's
        declared ``applies_to``, or a non-:class:`GateConfig` override.
    FoldFitError, RepeatedCVDataError, GoldenRuleError
        Fold/fit failures, unusable data, and golden-rule violations surface
        under their own names from the measurement loops.
    """
    if not isinstance(candidate, TreatmentCandidate):
        raise TreatmentGateError(
            f"measure_pair expects a TreatmentCandidate, got "
            f"{type(candidate).__name__}."
        )
    if candidate.variant_key == INCUMBENT_CANDIDATE.variant_key:
        raise TreatmentGateError(
            f"Candidate {candidate.variant_key!r} is the documented incumbent "
            "(rung 0); it is described, not re-measured — the gate compares "
            "variants against the committed pipeline via incumbent_factory."
        )
    if model_type not in candidate.applies_to:
        raise TreatmentGateError(
            f"Candidate {candidate.variant_key!r} does not apply to "
            f"{model_type!r}: declared applies_to is {list(candidate.applies_to)}."
        )
    if config is None:
        gate_cfg = GateConfig()
    elif isinstance(config, GateConfig):
        gate_cfg = config
    else:
        raise TreatmentGateError(
            f"config must be a GateConfig, got {type(config).__name__}; the "
            "gate's thresholds are constants, not overridable parameters."
        )

    features = require_feature_frame(X)
    rcv_cfg = gate_cfg.to_repeated_cv_config()
    treated_key = treatment_key_for(candidate, model_type)
    incumbent_key = incumbent_key_for(model_type)
    pair_params = _resolve_fit_params(fit_params, model_type)

    if candidate.mechanism == "sampling":
        treated_result, lineages, partitions = _run_sampling_repeated_cv(
            treated_key, candidate, model_type, features, y, rcv_cfg, pair_params
        )
    else:
        def treated_factory() -> object:
            return build_candidate(candidate, model_type, pair_params)

        treated_result = run_repeated_cv(
            treated_key,
            treated_factory,
            features,
            y,
            config=rcv_cfg,
            capture_predictions=True,
        )
        lineages, partitions = (), ()

    def incumbent_pipeline_factory() -> object:
        return incumbent_factory(model_type, pair_params)

    incumbent_result = run_repeated_cv(
        incumbent_key,
        incumbent_pipeline_factory,
        features,
        y,
        config=rcv_cfg,
        capture_predictions=True,
    )
    comparison = compare_repeated_cv(
        {treated_key: treated_result, incumbent_key: incumbent_result},
        treated_key,
        incumbent_key,
        metric=GATE_METRIC,
        alpha=PAIRED_SIGNIFICANCE_ALPHA,
    )
    measurement = CandidateMeasurement(
        candidate=candidate,
        model_type=model_type,
        treated=treated_result,
        incumbent=incumbent_result,
        comparison=comparison,
        fold_lineages=lineages,
        fold_partitions=partitions,
    )
    logger.info(
        "Measured %s vs %s on %s: delta=%+.6f, %s, verdict clauses pending",
        treated_key,
        incumbent_key,
        gate_cfg.fingerprint(),
        measurement.delta,
        comparison.corrected_t.verdict(),
    )
    return measurement


# ---------------------------------------------------------------------------
# Verdict logic — the executable mirror of gate_text()
# ---------------------------------------------------------------------------


def standing_not_worsened(
    measurement: CandidateMeasurement,
    standings: Mapping[str, tuple[float, int]],
) -> bool:
    """Clause (c): the winner-standing rule against supplied leaderboard standings.

    ``standings`` maps gate key -> ``(primary_mean, rank)`` (1-based rank,
    lower is better). It is the battery leaderboard's standings — by
    construction it includes the baseline and the runner-up entries, whose
    relative positions are embedded in the ranks — and it MUST carry the two
    entries clause (c) compares:

    - the treated variant's standing, keyed
      ``treatment:<variant>/<model_type>`` (:func:`treatment_key_for`),
    - its incumbent's standing, keyed ``incumbent:<model_type>``
      (:func:`incumbent_key_for`).

    The clause holds iff the treated model's standing is NOT worse than its
    incumbent's standing: ``treated_rank <= incumbent_rank``. An equal rank
    is not a worsening; a larger rank is. The comparator deliberately judges
    the supplied standings rather than re-deriving ranks from the means: the
    leaderboard (with ties and its own ordering policy) is authoritative,
    and the declared sentence is about *standing*, not about the raw delta —
    clause (a) already owns the delta question. This keeps clause (c)
    independently evaluable (a supplied treated entry whose rank falls behind
    the incumbent's rank fails alone), which a means-derived rank could never
    express when the treated mean improves over the incumbent.

    Raises
    ------
    TreatmentGateError
        When standings is not a non-empty mapping, or lacks the treated
        variant's or the incumbent's mandatory entry, or carries a rank that
        is not a positive integer.
    """
    if not isinstance(standings, Mapping) or not standings:
        raise TreatmentGateError(
            "standing_not_worsened requires a non-empty standings mapping of "
            "gate key -> (primary_mean, rank); an empty or non-mapping "
            "standings supply is a caller error, not a passable clause."
        )
    required_keys = (measurement.treated_key, measurement.incumbent_key)
    missing = [key for key in required_keys if key not in standings]
    if missing:
        raise TreatmentGateError(
            f"standings lacks the mandatory gate key(s) {missing}; clause (c) "
            f"compares the treated model's standing "
            f"({measurement.treated_key!r}) against its incumbent's standing "
            f"({measurement.incumbent_key!r}) on the full battery leaderboard "
            "(baseline and runner-up included by construction)."
        )

    def _checked_rank(key: str, entry: object) -> int:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TreatmentGateError(
                f"standings entry for {key!r} must be a (primary_mean, rank) "
                f"tuple, got {entry!r}."
            )
        rank = entry[1]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TreatmentGateError(
                f"standings entry for {key!r} carries rank {rank!r} of type "
                f"{type(rank).__name__}; ranks are 1-based integers (lower "
                "is better)."
            )
        if rank < 1:
            raise TreatmentGateError(
                f"standings entry for {key!r} carries rank {rank}; ranks are "
                "1-based (lower is better)."
            )
        return int(rank)

    treated_rank = _checked_rank(
        measurement.treated_key, standings[measurement.treated_key]
    )
    incumbent_rank = _checked_rank(
        measurement.incumbent_key, standings[measurement.incumbent_key]
    )
    logger.info(
        "Clause (c) winner standing for %s on %s: incumbent standing %d vs "
        "treated standing %d -> %s",
        measurement.treated_key,
        measurement.model_type,
        incumbent_rank,
        treated_rank,
        "not-worsened" if treated_rank <= incumbent_rank else "WORSENED",
    )
    return treated_rank <= incumbent_rank


def verdict_treatment(
    measurement: CandidateMeasurement,
    standings: Mapping[str, tuple[float, int]] | None = None,
) -> GateVerdict:
    """Evaluate the pre-declared gate clauses on one measurement.

    Clause booleans mirror :func:`heart.eval.treatment_declared.gate_text`
    exactly: (a) ``delta >= GATE_MIN_DELTA_ROC_AUC``, (b) the corrected
    resampled t-test is significant at the declared alpha, (c)
    :func:`standing_not_worsened`. With ``standings=None`` clause (c) is
    recorded as ``None`` (not evaluated, not silently true); a measurement
    that then passes (a) and (b) is still NOT accepted — it reports
    :data:`VERDICT_REJECTED_INCOMPLETE_EVIDENCE`. The returned verdict text
    embeds ``gate_text()`` verbatim.
    """
    delta_ok = bool(measurement.delta >= GATE_MIN_DELTA_ROC_AUC)
    significant = measurement.significant
    if standings is None:
        standing_ok: bool | None = None
    else:
        standing_ok = standing_not_worsened(measurement, standings)

    if not (delta_ok and significant):
        verdict = VERDICT_REJECTED
    elif standing_ok is None:
        verdict = VERDICT_REJECTED_INCOMPLETE_EVIDENCE
    elif standing_ok:
        verdict = VERDICT_ACCEPTED
    else:
        verdict = VERDICT_REJECTED

    clause_c_text = (
        "not evaluated (no standings supplied)"
        if standing_ok is None
        else str(standing_ok)
    )
    text = (
        f"Gate verdict for {measurement.treated_key} vs "
        f"{measurement.incumbent_key} on {GATE_METRIC}: {verdict}. "
        f"Clause (a) delta {measurement.delta:+.6f} against threshold "
        f"{GATE_MIN_DELTA_ROC_AUC:+.6f} -> {delta_ok}. "
        f"Clause (b) corrected resampled t-test "
        f"{measurement.comparison.corrected_t.verdict()} -> {significant}. "
        f"Clause (c) winner standing ({GATE_WINNER_STANDING_RULE}) -> "
        f"{clause_c_text}. " + gate_text()
    )
    return GateVerdict(
        delta_ok=delta_ok,
        significant=significant,
        standing_ok=standing_ok,
        verdict=verdict,
        text=text,
    )


@dataclass(frozen=True)
class GateVerdict:
    """The three clause booleans plus the gate's verdict sentence.

    ``standing_ok`` is ``None`` when no standings were supplied (clause (c)
    NOT evaluated). ``verdict`` is one of :data:`VERDICT_ACCEPTED`,
    :data:`VERDICT_REJECTED`, :data:`VERDICT_REJECTED_INCOMPLETE_EVIDENCE`.
    ``text`` embeds the verbatim pre-declared ``gate_text()`` sentence.
    """

    delta_ok: bool
    significant: bool
    standing_ok: bool | None
    verdict: str
    text: str


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateOutcome:
    """One candidate's recorded gate outcome (per candidate, or per pair).

    For the incumbent rung-0 entry ``is_incumbent`` is True,
    ``re_measured`` False, ``delta``/``significant``/clauses are ``None``,
    ``model_type`` is ``None`` (the documented policy covers the whole
    battery), and ``pooled_oof_metrics`` is ``None`` — it carries NO gate
    number and NO MLflow identity by construction. Measured pair entries
    carry the full clause record plus the D036 pooled-OOF canonical metric
    dict ready for a downstream ``log_evaluation_run`` call.
    """

    candidate_key: str
    model_type: str | None
    mechanism: str
    group: str
    description: str
    is_incumbent: bool
    re_measured: bool
    verdict: str
    verdict_text: str
    delta: float | None
    significant: bool | None
    delta_ok: bool | None
    standing_ok: bool | None
    n_folds: int
    n_repeats: int
    pooled_oof_metrics: Mapping[str, object] | None
    treated_key: str | None = None
    run_kind: str = MLFLOW_RUN_KIND_TREATMENT_CANDIDATE
    treatment_variant_tag: str = MLFLOW_TAG_TREATMENT_VARIANT
    run_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_key": self.candidate_key,
            "model_type": self.model_type,
            "mechanism": self.mechanism,
            "group": self.group,
            "description": self.description,
            "is_incumbent": self.is_incumbent,
            "re_measured": self.re_measured,
            "verdict": self.verdict,
            "verdict_text": self.verdict_text,
            "delta": self.delta,
            "significant": self.significant,
            "delta_ok": self.delta_ok,
            "standing_ok": self.standing_ok,
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "pooled_oof_metrics": (
                dict(self.pooled_oof_metrics) if self.pooled_oof_metrics else None
            ),
            "treated_key": self.treated_key,
            "run_kind": self.run_kind,
            "treatment_variant_tag": self.treatment_variant_tag,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class GateReport:
    """The ordered outcomes of one gate run plus its measurement geometry.

    Outcomes follow the declared candidate ``order`` (incumbent entry
    included when the filter keeps it). ``gate_sentence`` is the verbatim
    pre-declared :func:`gate_text` sentence the verdicts were judged against.
    """

    config: GateConfig
    outcomes: tuple[GateOutcome, ...]
    gate_sentence: str
    standings_supplied: bool

    @property
    def measured_outcomes(self) -> tuple[GateOutcome, ...]:
        """Outcomes that carried a real measurement (never the rung-0 entry)."""
        return tuple(outcome for outcome in self.outcomes if outcome.re_measured)

    @property
    def incumbent_outcome(self) -> GateOutcome:
        """The rung-0 documented-not-re-measured entry."""
        for outcome in self.outcomes:
            if outcome.is_incumbent:
                return outcome
        raise TreatmentGateError(
            "GateReport has no incumbent entry; run_gate_core always records "
            "the is_incumbent candidate as documented-not-re-measured."
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable projection (the seed of T05's gate_report.json)."""
        return {
            "config": self.config.to_dict(),
            "gate_sentence": self.gate_sentence,
            "standings_supplied": self.standings_supplied,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


def _incumbent_outcome(candidate: TreatmentCandidate, config: GateConfig) -> GateOutcome:
    """The rung-0 entry: described, never re-measured, never logged."""
    return GateOutcome(
        candidate_key=candidate.variant_key,
        model_type=None,
        mechanism=candidate.mechanism,
        group=candidate.group,
        description=candidate.description,
        is_incumbent=True,
        re_measured=False,
        verdict=VERDICT_DOCUMENTED_NOT_RE_MEASURED,
        verdict_text=(
            "Rung 0 documented-not-re-measured: the committed "
            f"{candidate.description[:-1].lower()} The incumbent pipeline "
            "carries no gate measurement number and no MLflow run; variants "
            "are measured against it via incumbent_factory."
        ),
        delta=None,
        significant=None,
        delta_ok=None,
        standing_ok=None,
        n_folds=config.n_folds,
        n_repeats=config.n_repeats,
        pooled_oof_metrics=None,
        treated_key=None,
    )


def _candidate_run_params(
    outcome: GateOutcome, config: GateConfig
) -> dict[str, object]:
    """MLflow params for one measured candidate run (D036).

    Carries the variant identity (``variant_key``/``mechanism``/``group``/
    ``applies_to``/``model_type``), the measurement geometry
    (``n_folds``/``n_repeats``/``seed``/``gate_fingerprint``), the declared
    accept thresholds (``alpha``/``gate_metric``/``min_delta``) as READ-ONLY
    mirrors, and the measured gate outcome: the delta, the significance
    reading, and each verdict clause boolean. Normalised to MLflow-safe
    strings by :func:`heart.tracking.run.log_evaluation_run`.
    """
    candidate = candidate_by_key(outcome.candidate_key)
    return {
        "variant_key": candidate.variant_key,
        "mechanism": outcome.mechanism,
        "group": outcome.group,
        "candidate_group": outcome.group,
        "model_type": outcome.model_type,
        "applies_to": tuple(candidate.applies_to),
        "n_folds": outcome.n_folds,
        "n_repeats": outcome.n_repeats,
        "seed": config.seed,
        "gate_fingerprint": config.fingerprint(),
        "gate_metric": config.gate_metric,
        "alpha": config.alpha,
        "min_delta": config.min_delta,
        "delta": outcome.delta,
        "significance_reading": outcome.significant,
        "delta_ok": outcome.delta_ok,
        "standing_ok": outcome.standing_ok,
        "verdict": outcome.verdict,
    }


def _candidate_run_tags(
    outcome: GateOutcome, config: GateConfig
) -> dict[str, str]:
    """MLflow tags for one measured candidate run (D036).

    The frozen tracking convention plus the D024 extension:
    ``run_kind=treatment-candidate`` (under
    ``heart.tuning.runner.RUN_KIND_TAG``) and the declared
    ``treatment_variant`` tag — the two markers that keep these runs
    excluded from the leaderboard's default ``run_kind=final`` collection BY
    CONSTRUCTION. Also carries ``model_type`` (the battery member), the
    registry ``family`` of that member, the gate verdict, and the candidate
    group for store-side filtering.
    """
    candidate = candidate_by_key(outcome.candidate_key)
    return {
        RUN_KIND_TAG: MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
        MLFLOW_TAG_TREATMENT_VARIANT: candidate.mlflow_treatment_variant,
        MODEL_TYPE_TAG: str(outcome.model_type),
        FAMILY_TAG: resolve_spec(str(outcome.model_type)).family,
        "gate_verdict": outcome.verdict,
        "candidate_group": outcome.group,
    }


def _log_gate_report(
    report: GateReport,
    *,
    split_version: str = GATE_SPLIT_VERSION,
    tracking_dir: str | Path | None = None,
    experiment_name: str | None = None,
) -> dict[str, str]:
    """Log EXACTLY ONE MLflow run per measured pair; return key -> run_id.

    Every measured (candidate, model_type) pair in
    ``report.measured_outcomes`` is logged through
    :func:`heart.tracking.run.log_evaluation_run` with the canonical pooled-OOF
    metric dict (schema-checked inside; ``validate=False`` is never used), the
    ``treatment:<variant_key>/<model_type>`` model name, the supplied
    ``split_version`` (default :data:`GATE_SPLIT_VERSION` = ``"fixture"`` for
    S02's fixture-level contract proof; S03's real-store run threads the
    declared real split version per D038), the D036 params/tags, and
    the ``gate_report.json`` artifact (the JSON-serializable
    :class:`GateReport` projection, which embeds the verbatim
    :func:`heart.eval.treatment_declared.gate_text` sentence). The incumbent
    rung-0 outcome is never logged: it carries no measurement and no MLflow
    identity by construction.

    A tracking failure raises the named tracking error loudly, aborting the
    log loop — a half-logged report must never be passed off as complete.
    """
    run_ids: dict[str, str] = {}
    projection = json.dumps(
        report.to_dict(), indent=2, sort_keys=True, default=str
    )
    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = Path(tmp) / GATE_REPORT_ARTIFACT
        artifact_path.write_text(projection, encoding="utf-8")
        for outcome in report.measured_outcomes:
            result = log_evaluation_run(
                metrics=dict(outcome.pooled_oof_metrics or {}),
                model_name=str(outcome.treated_key),
                split_version=split_version,
                params=_candidate_run_params(outcome, report.config),
                tags=_candidate_run_tags(outcome, report.config),
                artifacts=[artifact_path],
                tracking_dir=tracking_dir,
                experiment_name=experiment_name or DEFAULT_EXPERIMENT,
            )
            run_ids[str(outcome.treated_key)] = result.run_id
            logger.info(
                "Logged treatment-candidate run %s for %s (verdict=%s, "
                "delta=%+.6f, significant=%s)",
                result.run_id,
                outcome.treated_key,
                outcome.verdict,
                float(outcome.delta or 0.0),
                outcome.significant,
            )
    return run_ids


def run_gate_core(
    X: pd.DataFrame,
    y: Sequence[object],
    *,
    candidate_keys: Sequence[str] | None = None,
    model_types: Sequence[str] | None = None,
    config: GateConfig | None = None,
    standings: Mapping[str, tuple[float, int]] | None = None,
    log_to_mlflow: bool = False,
    tracking_dir: str | Path | None = None,
    experiment_name: str | None = None,
    fit_params: Mapping[str, Mapping[str, object]] | None = None,
    split_version: str = GATE_SPLIT_VERSION,
) -> GateReport:
    """Measure the declared candidates in declared order into a :class:`GateReport`.

    Parameters
    ----------
    X, y:
        The measurement input (the fixture for S02's contract proof; the real
        store frames for S03).
    candidate_keys:
        Optional filter of :data:`~heart.eval.treatment_declared.
        DECLARED_CANDIDATE_KEYS`; an unknown key raises loudly. The incumbent
        appears in the report when its key is kept (default keeps all).
    model_types:
        Optional eligible battery model-type filter; unknown battery members
        raise loudly. Each non-incumbent candidate is measured for every
        model_type in its ``applies_to`` within this filter.
    config:
        :class:`GateConfig` (defaults to the declared geometry).
    standings:
        Optional ``{model_name -> (primary_mean, rank)}`` battery standings
        for clause (c); ``None`` records clause (c) as not evaluated.
    log_to_mlflow:
        The D036 MLflow logging path (D036): when True, every measured
        (candidate, model_type) pair emits EXACTLY ONE MLflow run via
        :func:`heart.tracking.run.log_evaluation_run` — canonical pooled-OOF
        metric dict (schema-checked; never ``validate=False``),
        ``model_name="treatment:<variant_key>/<model_type>"``,
        ``split_version`` (the supplied value; default
        :data:`GATE_SPLIT_VERSION`), the declared params/tags (see
        :func:`_candidate_run_params` / :func:`_candidate_run_tags`) and the
        ``gate_report.json`` artifact. The incumbent rung-0 candidate is
        NEVER logged (documented-not-re-measured; it appears in the report
        only). The returned report carries each measured outcome's
        ``run_id`` (``None`` in the dry mode below). When False, NO run is
        written anywhere and every outcome's ``run_id`` is ``None``.
    tracking_dir:
        Optional local MLflow store directory override (tests point it at a
        tmp path); ``None`` keeps the real configured store — S03 calls the
        runner with defaults.
    experiment_name:
        Optional experiment-name override; ``None`` keeps the default
        experiment.
    fit_params:
        Optional per-``model_type`` estimator hyperparameters applied
        identically to the treated and the incumbent build (see
        :func:`measure_pair`); never a threshold surface.
    split_version:
        The ``split_version`` stamp logged on every measured candidate run
        (default :data:`GATE_SPLIT_VERSION` = ``"fixture"`` — S02's
        fixture-level contract proof is not a real-data split version).
        S03's real-store run threads the declared real split version
        (``"v1"``, D038) through this parameter instead of re-stamping
        fixture runs; the default keeps every existing call byte-identical.

    Returns
    -------
    GateReport
        Ordered outcomes; the incumbent is recorded as
        documented-not-re-measured (rung 0, no number, no MLflow identity).
    """
    if config is None:
        gate_cfg = GateConfig()
    elif isinstance(config, GateConfig):
        gate_cfg = config
    else:
        raise TreatmentGateError(
            f"config must be a GateConfig, got {type(config).__name__}."
        )
    if isinstance(split_version, bool) or not isinstance(split_version, str):
        raise TreatmentGateError(
            f"split_version must be a string, got "
            f"{type(split_version).__name__} ({split_version!r})."
        )
    if not split_version.strip():
        raise TreatmentGateError(
            "split_version must be a non-empty string; an empty split stamp "
            "would make candidate runs indistinguishable in the store."
        )
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
    if model_types is None:
        eligible: set[str] | None = None
    else:
        eligible = set(model_types)
        outside = eligible - set(BATTERY_MEMBERS)
        if outside:
            raise TreatmentGateError(
                f"model_types {sorted(outside)} are not battery members "
                f"{list(BATTERY_MEMBERS)}; the gate measures declared "
                "applies_to scope against the real battery only."
            )

    outcomes: list[GateOutcome] = []
    for candidate in DECLARED_CANDIDATES:
        if keys_filter is not None and candidate.variant_key not in keys_filter:
            continue
        if candidate.is_incumbent:
            outcomes.append(_incumbent_outcome(candidate, gate_cfg))
            continue
        for model_type in candidate.applies_to:
            if eligible is not None and model_type not in eligible:
                continue
            measurement = measure_pair(
                candidate,
                model_type,
                X,
                y,
                config=gate_cfg,
                fit_params=fit_params,
            )
            verdict = verdict_treatment(measurement, standings)
            pooled = measurement.pooled_oof_metric_dict()
            outcomes.append(
                GateOutcome(
                    candidate_key=candidate.variant_key,
                    model_type=model_type,
                    mechanism=candidate.mechanism,
                    group=candidate.group,
                    description=candidate.description,
                    is_incumbent=False,
                    re_measured=True,
                    verdict=verdict.verdict,
                    verdict_text=verdict.text,
                    delta=measurement.delta,
                    significant=verdict.significant,
                    delta_ok=verdict.delta_ok,
                    standing_ok=verdict.standing_ok,
                    n_folds=measurement.treated.config.n_folds,
                    n_repeats=measurement.treated.config.n_repeats,
                    pooled_oof_metrics=pooled,
                    treated_key=measurement.treated_key,
                )
            )
    report = GateReport(
        config=gate_cfg,
        outcomes=tuple(outcomes),
        gate_sentence=gate_text(),
        standings_supplied=standings is not None,
    )
    if log_to_mlflow:
        run_ids = _log_gate_report(
            report,
            split_version=split_version,
            tracking_dir=tracking_dir,
            experiment_name=experiment_name,
        )
        outcomes = tuple(
            _dc_replace(
                outcome,
                run_id=run_ids.get(outcome.treated_key)
                if outcome.re_measured
                else None,
            )
            for outcome in report.outcomes
        )
        report = GateReport(
            config=report.config,
            outcomes=outcomes,
            gate_sentence=report.gate_sentence,
            standings_supplied=report.standings_supplied,
        )
    measured = report.measured_outcomes
    logger.info(
        "Gate run complete (%s): %d measured pair(s) across %d outcome(s); "
        "verdicts: %s",
        gate_cfg.fingerprint(),
        len(measured),
        len(outcomes),
        {outcome.candidate_key: outcome.verdict for outcome in report.outcomes},
    )
    return report


def run_gate(
    X: pd.DataFrame,
    y: Sequence[object],
    *,
    candidate_keys: Sequence[str] | None = None,
    model_types: Sequence[str] | None = None,
    config: GateConfig | None = None,
    standings: Mapping[str, tuple[float, int]] | None = None,
    log_to_mlflow: bool = False,
    tracking_dir: str | Path | None = None,
    experiment_name: str | None = None,
    fit_params: Mapping[str, Mapping[str, object]] | None = None,
    split_version: str = GATE_SPLIT_VERSION,
) -> GateReport:
    """Run the M006 treatment gate (the D036 public entry).

    Identical surface to :func:`run_gate_core` — see its documentation. With
    ``log_to_mlflow=True`` exactly one MLflow run per measured
    (candidate, model_type) pair is written (tmp-path store for tests,
    configured defaults for S03's real-store run) under the supplied
    ``split_version`` (default :data:`GATE_SPLIT_VERSION`; S03 threads
    ``"v1"`` per D038); with ``log_to_mlflow=False``
    the report-only dry mode returns the same :class:`GateReport` with
    ``run_id=None`` on every outcome and writes nothing.
    """
    return run_gate_core(
        X,
        y,
        candidate_keys=candidate_keys,
        model_types=model_types,
        config=config,
        standings=standings,
        log_to_mlflow=log_to_mlflow,
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        fit_params=fit_params,
        split_version=split_version,
    )
