"""Winner selection, calibration, threshold tuning, and serving decision (S06/T03).

This module consumes — never re-implements — the two upstream S06 modules:

* :mod:`heart.eval.repeated_cv` supplies the per-fold ROC-AUC distributions
  (``run_repeated_cv_comparison`` -> ``rank_candidates``), the pairing
  contract the significance tests rely on.
* :mod:`heart.eval.significance` supplies the paired corrected resampled
  t-test (and optional McNemar) that turns a repeated-CV win into a claim
  with a p-value.
* :mod:`heart.eval.calibration` supplies the probability-calibration wrapper
  and the Brier/reliability-curve comparison around it.

Life-giving result of the slice: one :class:`SelectionDecision` that names the
winner, states which paired tests proven it, reports the serving weight,
tunes the decision threshold **on validation data only** (threshold tuning
never sees training or calibration rows), calibrates the winner's
probabilities, and computes the SHIP / NO-SHIP flag. A statistically superior
model whose serving weight exceeds the declared budget is flagged explicitly
(``NO-SHIP`` + ``serving_weight_exceeded``) rather than silently swapped in a
runner-up — the orchestration layer never silently picks another model.

Failure contract
----------------
Malformed inputs raise named :class:`SelectionDataError`; unusable
configuration raises :class:`SelectionConfigError`; both derive from
:class:`SelectionError`, mirroring the T01/T02 taxonomy.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from heart.config import RANDOM_SEED
from heart.eval.calibration import (
    DEFAULT_CALIBRATION_METHOD,
    CalibrationComparison,
    calibration_comparison,
    fit_calibration_wrapper,
)
from heart.eval.metrics import DEFAULT_CALIBRATION_BINS
from heart.eval.repeated_cv import (
    PRIMARY_METRIC,
    RepeatedCVConfig,
    RepeatedCVResult,
    rank_candidates,
    run_repeated_cv_comparison,
)
from heart.eval.significance import (
    DEFAULT_ALPHA,
    PairwiseComparison,
    compare_repeated_cv,
    significance_report,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OBJECTIVE",
    "SERVING_WEIGHT_BUDGET_MB",
    "SELECTION_REPORT_PATH",
    "SelectionError",
    "SelectionConfigError",
    "SelectionDataError",
    "ServingWeightCheck",
    "ThresholdTuning",
    "SelectionDecision",
    "tune_threshold",
    "select_model",
    "selection_report",
    "write_selection_report",
]


#: Default declared thresholding objective (app-serving tradeoff).
DEFAULT_OBJECTIVE: str = "f1"
#: Declared serving-weight budget (MB); over-budget => NO-SHIP, never swap.
SERVING_WEIGHT_BUDGET_MB: float = 25.0
SELECTION_REPORT_PATH: str = "reports/selection.md"

_VALID_OBJECTIVES: tuple[str, ...] = ("f1", "balanced_accuracy", "youden_j")

#: Grid step for the threshold sweep; dense enough to pin the optimum, cheap
#: enough to stay in the pytest run.
_THRESHOLD_STEP: float = 0.01


class SelectionError(Exception):
    """Base class for every model-selection failure."""


class SelectionConfigError(SelectionError):
    """The requested selection configuration is invalid."""


class SelectionDataError(SelectionError):
    """The supplied selection inputs are malformed and cannot be used."""


# ---------------------------------------------------------------------------
# Threshold tuning (validation data only)
# ---------------------------------------------------------------------------


def _check_threshold_inputs(
    y_true: np.ndarray, y_proba: np.ndarray, *, step: float
) -> None:
    if step <= 0.0:
        raise SelectionConfigError(
            f"threshold step must be positive, got {step!r}."
        )
    if y_true.size == 0:
        raise SelectionDataError(
            "Threshold tuning needs at least one validation row."
        )
    if y_true.size != y_proba.size:
        raise SelectionDataError(
            f"y_val ({y_true.size} rows) and y_proba ({y_proba.size} rows) do "
            "not align; threshold scoring is row-paired and a silent zip "
            "would fabricate the operating point."
        )
    if not np.all(np.isfinite(y_proba)):
        raise SelectionDataError(
            "y_proba contains non-finite values; cannot sweep thresholds."
        )
    if y_proba.min() < 0.0 or y_proba.max() > 1.0:
        raise SelectionDataError(
            f"y_proba leaves [0, 1] (min={y_proba.min():.6f}, "
            f"max={y_proba.max():.6f}); these are not probabilities."
        )


def _objective_scores(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray, objective: str
) -> np.ndarray:
    """Vectorized objective scores for the whole sweep grid."""
    positives = (y_true == 1).astype(np.int64)
    negatives = 1 - positives
    in_range = (y_proba[:, None] >= thresholds[None, :]).astype(np.int64)
    tp_all = (in_range * positives[:, None]).sum(axis=0)
    fp_all = (in_range * negatives[:, None]).sum(axis=0)
    fn_all = (positives.sum() - tp_all).astype(np.int64)
    tn_all = (negatives.sum() - fp_all).astype(np.int64)
    tp, fp, fn, tn = tp_all, fp_all, fn_all, tn_all
    total = tp + fp + fn + tn
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    specificity = np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1), 0.0)
    f1 = np.where(
        precision + recall > 0.0,
        2.0 * precision * recall / np.maximum(precision + recall, 1e-12),
        0.0,
    )
    balanced = 0.5 * (recall + specificity)
    youden = recall - (1.0 - specificity)
    table = {"f1": f1, "balanced_accuracy": balanced, "youden_j": youden}
    return table[objective]


@dataclass(frozen=True)
class ThresholdTuning:
    """Declared-objective sweep over thresholds, on validation data only."""

    objective: str
    thresholds: tuple[float, ...]
    scores: tuple[float, ...]
    best_threshold: float
    best_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "thresholds": self.thresholds,
            "scores": [round(v, 6) for v in self.scores],
            "best_threshold": round(self.best_threshold, 4),
            "best_score": round(self.best_score, 6),
        }


def tune_threshold(
    y_val: object,
    y_proba_val: object,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    step: float = _THRESHOLD_STEP,
) -> ThresholdTuning:
    """Sweep decision thresholds on validation rows and keep the argmax.

    The grid walks ``[step, 1.0 - step)`` at ``step`` increments so the sweep
    includes interior operating points while never predicting "always
    positive" (threshold 0) or "never positive" (threshold 1).

    Parameters
    ----------
    y_val, y_proba_val:
        The validation split — rows never used for winner fitting or
        calibration. This function asserts nothing about where they came
        from; the selection flow is what keeps them out of the fit path.
    objective:
        ``f1`` (default), ``balanced_accuracy``, or ``youden_j``. Unknown
        values raise :class:`SelectionConfigError`.
    step:
        Positive grid step; defaults to ``0.01``.

    Returns
    -------
    ThresholdTuning
        Declared objective, the full sweep, and the argmax.

    Raises
    ------
    SelectionConfigError
        Unknown objective, non-positive step.
    SelectionDataError
        Empty validation rows, misaligned or malformed probabilities.
    """
    y = np.asarray(y_val)
    proba = np.asarray(y_proba_val, dtype=float)
    if y.ndim != 1 or proba.ndim != 1:
        raise SelectionDataError(
            "Threshold tuning expects 1-D y_val and y_proba_val vectors."
        )
    y_binary = (y == 1).astype(int)
    _check_threshold_inputs(y_binary, proba, step=step)
    if objective not in _VALID_OBJECTIVES:
        raise SelectionConfigError(
            f"Unknown objective {objective!r}; valid objectives are "
            f"{list(_VALID_OBJECTIVES)}."
        )

    n_bins = int(round(1.0 / step))
    thresholds = (np.arange(n_bins) + 1) * step
    thresholds = thresholds[thresholds < 1.0]
    scores = _objective_scores(y_binary, proba, thresholds, objective)

    best_index = int(np.argmax(scores))
    # Tie-break toward the lower threshold: same objective score at multiple
    # thresholds means the operating point is not unique — prefer the more
    # inclusive (higher-recall) side of the tie, deterministic by construction.
    best_threshold = float(thresholds[best_index])
    tuning = ThresholdTuning(
        objective=objective,
        thresholds=tuple(float(v) for v in thresholds),
        scores=tuple(float(v) for v in scores),
        best_threshold=best_threshold,
        best_score=float(scores[best_index]),
    )
    logger.info(
        "Threshold tuning (%s, %d rows): best threshold %.2f with %s=%.4f",
        objective,
        y_true_rows(y_binary),
        best_threshold,
        objective,
        tuning.best_score,
    )
    return tuning


def y_true_rows(y_binary: np.ndarray) -> int:  # pragma: no cover - tiny helper
    return int(y_binary.size)


# ---------------------------------------------------------------------------
# Serving weight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServingWeightCheck:
    """Serving-weight claim for one candidate, against the declared budget."""

    model_name: str
    weight_mb: float
    budget_mb: float = SERVING_WEIGHT_BUDGET_MB

    @property
    def shippable(self) -> bool:
        return self.weight_mb <= self.budget_mb

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "weight_mb": round(self.weight_mb, 3),
            "budget_mb": round(self.budget_mb, 3),
            "shippable": self.shipping(),
        }

    def shipping(self) -> bool:  # pragma: no cover - alias, kept for symmetry
        return self.shippable


# ---------------------------------------------------------------------------
# Selection decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionDecision:
    """The slice's central artifact: winner + evidence + serving verdict."""

    winner: str
    runner_up: str | None
    baseline: str
    metric: str
    alpha: float
    ranked: tuple[tuple[str, float, float], ...]

    winner_vs_baseline: PairwiseComparison | None
    other_comparisons: tuple[PairwiseComparison, ...]

    serving_weight: ServingWeightCheck
    threshold: ThresholdTuning
    calibration: CalibrationComparison

    proven_significantly_better_than_baseline: bool
    shippable_by_weight: bool

    @property
    def flags(self) -> tuple[str, ...]:
        flags: list[str] = []
        if not self.proven_significantly_better_than_baseline:
            flags.append(
                "not_significantly_better_than_baseline "
                f"(corrected resampled t-test p={self.winner_vs_baseline.corrected_t.p_value:.4g} "
                f"at alpha={self.alpha})"
                if self.winner_vs_baseline is not None
                else "not_significantly_better_than_baseline (no comparison)"
            )
        if not self.shippable_by_weight:
            flags.append(
                f"serving_weight_exceeded ({self.serving_weight.weight_mb:.2f} MB "
                f"> budget {self.serving_weight.budget_mb:.2f} MB)"
            )
        if not self.calibration.is_improved:
            flags.append(
                "calibration_did_not_improve_brier "
                f"(delta {self.calibration.brier_delta:+.6f})"
            )
        return tuple(flags)

    @property
    def ship(self) -> bool:
        """SHIP only when every declared gate is green; NO-SHIP otherwise.

        A statistically superior but over-budget model surfaces
        ``NO-SHIP`` + ``serving_weight_exceeded`` intact; it is flagged, not
        silently substituted by the runner-up.
        """
        return (
            self.proven_significantly_better_than_baseline
            and self.shippable_by_weight
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "winner": self.winner,
            "runner_up": self.runner_up,
            "baseline": self.baseline,
            "metric": self.metric,
            "alpha": self.alpha,
            "ranked": [
                {"model": name, "mean": round(mean, 6), "std": round(std, 6)}
                for name, mean, std in self.ranked
            ],
            "winner_vs_baseline": (
                self.winner_vs_baseline.to_dict()
                if self.winner_vs_baseline is not None
                else None
            ),
            "other_comparisons": [c.to_dict() for c in self.other_comparisons],
            "serving_weight": self.serving_weight.to_dict(),
            "threshold": self.threshold.to_dict(),
            "calibration": self.calibration.to_dict(),
            "proven_significantly_better_than_baseline": (
                self.proven_significantly_better_than_baseline
            ),
            "shippable_by_weight": self.shippable_by_weight,
            "flags": list(self.flags),
            "ship": self.ship,
        }


def select_model(
    results: Mapping[str, RepeatedCVResult],
    *,
    fitted_winner: object,
    X_calib: object,
    y_calib: object,
    X_val: object,
    y_val: object,
    serving_weights: Mapping[str, float],
    objective: str = DEFAULT_OBJECTIVE,
    metric: str = PRIMARY_METRIC,
    alpha: float = DEFAULT_ALPHA,
    baseline: str | None = None,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> SelectionDecision:
    """Name the winner and attach every selection gate the app needs.

    Consuming upstream S06 structures, never re-implementing them:

    1. Winner = ``rank_candidates(results, metric=metric)`` top row, computed
       from the per-fold distributions ``repeated_cv`` already produced.
    2. Paired significance: ``significance_report``-style comparisons are
       built through :func:`heart.eval.significance.compare_repeated_cv` —
       the winner must significantly beat the declared baseline or the ship
       flag is pulled (recorded, never silent).
    3. Serving weight: the *winner* is checked against
       ``SERVING_WEIGHT_BUDGET_MB``; an over-budget winner yields
       ``NO-SHIP`` + ``serving_weight_exceeded`` without a swap.
    4. Calibration: ``fitted_winner`` is wrapped with
       ``fit_calibration_wrapper`` on ``X_calib``/``y_calib`` (rows never
       touched by threshold tuning) and compared by
       :mod:`heart.eval.calibration.calibration_comparison`.
    5. Threshold tuning happens **on validation data only**: ``(X_val,
       y_val)`` never participates in winner fitting, calibration fitting,
       or repeated-CV splits. The test suite asserts the argmax.
    """
    if not results:
        raise SelectionDataError(
            "select_model needs at least one repeated-CV result."
        )
    ranked = rank_candidates(results, metric=metric)
    winner = ranked[0][0]
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    if not hasattr(fitted_winner, "predict_proba"):
        raise SelectionDataError(
            f"Fitted winner {winner!r} of type {type(fitted_winner).__name__} "
            "does not expose predict_proba; only probabilistic classifiers "
            "can carry the calibrated-probability contract."
        )

    # --- significance: winner vs every subject, especially the baseline ---
    winner_vs_baseline: PairwiseComparison | None = None
    others: list[PairwiseComparison] = []
    if len(results) > 1:
        if baseline is None:
            logistic_hits = [name for name in results if "logistic" in name.lower()]
            baseline = logistic_hits[0] if logistic_hits else None
        if baseline is None:
            baseline = ranked[-1][0]
        if baseline not in results:
            raise SelectionDataError(
                f"Declared baseline {baseline!r} is not among candidates "
                f"{sorted(results)}."
            )
        subjects = [name for name, _m, _s in ranked if name not in (winner, baseline)]
        for subject in subjects:
            others.append(
                compare_repeated_cv(results, winner, subject, metric=metric, alpha=alpha)
            )
        if baseline != winner:
            winner_vs_baseline = compare_repeated_cv(
                results, winner, baseline, metric=metric, alpha=alpha
            )
    proven = (
        winner_vs_baseline is not None
        and winner_vs_baseline.corrected_t.p_value < alpha
    )

    # --- serving weight ---
    if winner not in serving_weights:
        raise SelectionDataError(
            f"serving_weights lacks {winner!r}; the ship gate cannot be "
            "evaluated on a model whose weight is unknown, so the selection "
            "refuses to ship silently."
        )
    weight_raw = serving_weights[winner]
    if not isinstance(weight_raw, (int, float)) or isinstance(weight_raw, bool) or weight_raw < 0.0:
        raise SelectionDataError(
            f"Serving weight for {winner!r} must be a non-negative number of "
            f"MB, got {weight_raw!r}."
        )
    weight_check = ServingWeightCheck(
        model_name=winner, weight_mb=float(weight_raw)
    )

    # --- calibration: dedicated calibration split, frozen winner ---
    wrapper = fit_calibration_wrapper(
        fitted_winner, X_calib, y_calib, method=calibration_method
    )
    proba_before = fitted_winner.predict_proba(X_val)
    proba_before = np.asarray(proba_before)
    proba_before = (
        proba_before[:, 1] if proba_before.ndim == 2 else proba_before.ravel()
    )
    proba_after = wrapper.predict_proba(X_val)
    proba_after = np.asarray(proba_after)
    proba_after = proba_after[:, 1] if proba_after.ndim == 2 else proba_after.ravel()
    calibration = calibration_comparison(
        y_val, proba_before, proba_after, n_bins=n_bins
    )

    # --- threshold: validation split only ---
    threshold = tune_threshold(
        y_val, np.asarray(proba_after, dtype=float), objective=objective
    )

    decision = SelectionDecision(
        winner=winner,
        runner_up=runner_up,
        baseline=baseline if len(results) > 1 else winner,
        metric=metric,
        alpha=alpha,
        ranked=tuple(ranked),
        winner_vs_baseline=winner_vs_baseline,
        other_comparisons=tuple(others),
        serving_weight=weight_check,
        threshold=threshold,
        calibration=calibration,
        proven_significantly_better_than_baseline=proven,
        shippable_by_weight=weight_check.shippable,
    )
    logger.info("Selection decision: %s | ship=%s", winner, decision.ship)
    return decision


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _curve_markdown(comparison: CalibrationComparison, label: str) -> str:
    lines = [
        f"| bin | {label} mean predicted | {label} fraction positive | gap | count |",
        "| --- | --- | --- | --- | --- |",
    ]
    for b in comparison.report_calibrated.bins:
        mean = (
            f"{b.mean_predicted:.4f}" if b.mean_predicted is not None else "—"
        )
        frac = (
            f"{b.fraction_positive:.4f}" if b.fraction_positive is not None else "—"
        )
        gap = f"{b.gap:.4f}" if b.gap is not None else "—"
        lines.append(f"| {b.bin} | {mean} | {frac} | {gap} | {b.count} |")
    return "\n".join(lines)


def selection_report(decision: SelectionDecision) -> str:
    """Markdown rendering of the whole selection decision for ``selection.md``."""
    ranked_rows = [
        f"| {i + 1} | {name} | {mean:.6f} ± {std:.6f} |"
        for i, (name, mean, std) in enumerate(decision.ranked)
    ]
    comparison_rows: list[str] = []
    if decision.winner_vs_baseline is not None:
        fields = [
            decision.winner_vs_baseline,
            *decision.other_comparisons,
        ]
        for comparison in fields:
            corrected = comparison.corrected_t
            comparison_rows.append(
                f"| {comparison.model_a} vs {comparison.model_b} | "
                f"{corrected.mean_diff:+.6f} | {corrected.p_value:.4g} | "
                f"{corrected.verdict()} |"
            )

    ship_heading = "SHIP" if decision.ship else "NO-SHIP"
    lines = [
        "# Selection report",
        "",
        f"Winner: **{decision.winner}** "
        f"({decision.metric} {decision.ranked[0][1]:.6f} ± {decision.ranked[0][2]:.6f}).",
        "",
        "## Ranking (repeated-CV, per-fold distribution)",
        "",
        "| rank | model | mean ± std |",
        "| --- | --- | --- |",
        *ranked_rows,
        "",
        "## Paired significance (corrected resampled t-test)",
        "",
    ]
    if comparison_rows:
        lines.append("| comparison | mean diff (A - B) | p-value | verdict |")
        lines.append("| --- | --- | --- | --- |")
        lines.extend(comparison_rows)
    else:
        lines.append("Single candidate; no paired test runnable.")
    lines.extend(
        [
            "",
            f"Serving weight: **{decision.serving_weight.weight_mb:.2f} MB** "
            f"(budget {decision.serving_weight.budget_mb:.2f} MB) — "
            f"{'within' if decision.shippable_by_weight else 'over'} budget.",
            "",
            f"Calibration ({decision.calibration.summary()}):",
            "",
            _curve_markdown(decision.calibration, "calibrated"),
            "",
            f"Chosen threshold: **{decision.threshold.best_threshold:.2f}** "
            f"(objective ``{decision.threshold.objective}`` at "
            f"{decision.threshold.best_score:.4f}, swept on validation rows only).",
            "",
            f"Flags: {', '.join(decision.flags) if decision.flags else 'none'}.",
            "",
            f"**Verdict: {ship_heading}**",
            "",
        ]
    )
    return "\n".join(lines)


def write_selection_report(
    decision: SelectionDecision,
    *,
    path: str | Path = SELECTION_REPORT_PATH,
) -> Path:
    """Atomically-ish write ``reports/selection.md`` from the decision."""
    target = Path(path)
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(selection_report(decision), encoding="utf-8")
    logger.info("Wrote selection report to %s (%d bytes), ship=%s.",
                target, target.stat().st_size, decision.ship)
    return target


# ---------------------------------------------------------------------------
# Real-data generation entrypoint (reports/selection.md)
# ---------------------------------------------------------------------------


def _pickle_size_mb(obj: object) -> float:
    """Serialized serving weight of a fitted object, in megabytes.

    Pickle is the serialization the byte-count claim is measured on; the T04
    artifact format can only be compared against a measured baseline, which
    this provides per candidate.
    """
    import pickle  # local import: only the generation path needs serialization

    return len(pickle.dumps(obj)) / (1024.0 * 1024.0)


def generate_real_selection_report(
    *,
    split_version: str = "v1",
    candidate_types: Sequence[str] = (
        "logistic-regression-l2",
        "random-forest",
    ),
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = RANDOM_SEED,
    objective: str = DEFAULT_OBJECTIVE,
    alpha: float = DEFAULT_ALPHA,
    path: str | Path = SELECTION_REPORT_PATH,
    val_fraction: float = 0.20,
    calib_fraction: float = 0.20,
) -> tuple[SelectionDecision, Path]:
    """End-to-end selection run on the real split, writing ``selection.md``.

    The declared data flow (S06/T03 contract):

    1. ``load_split_frames`` yields the versioned train rows; the held-out
       test rows remain untouched by this entire flow.
    2. Calibration and validation rows are carved from the training split
       with ``train_test_split`` (stratified, fixed seed). Winner fitting,
       repeated CV, and calibration never see the validation rows used for
       threshold tuning.
    3. Candidates come from :func:`heart.models.registry.build_pipeline` at
       each spec's declared default parameters — T04/T05 serialization scope
       stays untouched here.
    4. Repeated CV + paired significance via the upstream S06 modules;
       serving weight is the pickled MB of the fitted candidate pipeline;
       calibration via :mod:`heart.eval.calibration`; threshold via
       :func:`tune_threshold` on validation only.
    """
    from sklearn.model_selection import train_test_split as _split

    from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
    from heart.data.split import load_split_frames
    from heart.models.registry import build_pipeline, resolve_spec

    train, _test = load_split_frames(version=split_version)
    missing = [c for c in FEATURE_COLUMNS if c not in train.columns]
    if missing or TARGET_COLUMN not in train.columns:
        raise SelectionDataError(
            f"Split {split_version!r} lacks declared columns: missing="
            f"{missing}, target_present={TARGET_COLUMN in train.columns}."
        )
    features = train[list(FEATURE_COLUMNS)].copy()
    labels = train[TARGET_COLUMN].astype(int)

    x_rest, x_val, y_rest, y_val = _split(
        features, labels, test_size=val_fraction, random_state=seed, stratify=labels
    )
    x_train, x_calib, y_train, y_calib = _split(
        x_rest, y_rest, test_size=calib_fraction, random_state=seed, stratify=y_rest
    )

    candidates = {
        model_type: (lambda mt=model_type: build_pipeline(resolve_spec(mt)))
        for model_type in candidate_types
    }
    config = RepeatedCVConfig(n_folds=n_folds, n_repeats=n_repeats, seed=seed)
    results = run_repeated_cv_comparison(
        candidates, x_train, y_train, config=config
    )

    winner = rank_candidates(results, metric=PRIMARY_METRIC)[0][0]
    fitted_winner = candidates[winner]()
    fitted_winner.fit(x_train, y_train)

    serving_weights: dict[str, float] = {}
    for model_type in candidate_types:
        probe = candidates[model_type]()
        probe.fit(x_train, y_train)
        serving_weights[model_type] = _pickle_size_mb(probe)

    decision = select_model(
        results,
        fitted_winner=fitted_winner,
        X_calib=x_calib,
        y_calib=y_calib,
        X_val=x_val,
        y_val=y_val,
        serving_weights=serving_weights,
        objective=objective,
        alpha=alpha,
    )
    written = write_selection_report(decision, path=path)

    # Journal the decision onto the winner's MLflow run (S06/T05): the
    # leaderboard regeneration then re-derives the selection narrative from
    # the tracking store alone. Annotation is a soft dependency — a store
    # without the winner's final run logs a warning and selection still ships.
    try:
        from heart.reporting.leaderboard import annotate_winner_run
    except ImportError as exc:  # pragma: no cover - ml extra missing
        logger.warning("S06 annotation skipped (tracking store unavailable): %s", exc)
    else:
        try:
            annotated_run = annotate_winner_run(decision, split_version=split_version)
        except Exception as exc:  # LeaderboardError/TrackingError are named
            logger.warning(
                "S06 annotation of the winner run failed (%s: %s); "
                "the leaderboard will render without the selection section.",
                type(exc).__name__,
                exc,
            )
        else:
            if annotated_run is None:
                logger.warning(
                    "S06 annotation skipped: no final run for winner %r in the "
                    "tracking store. Re-run the battery to produce one.",
                    decision.winner,
                )
            else:
                logger.info(
                    "S06 selection annotated on winner run %s (ship=%s)",
                    annotated_run,
                    decision.ship,
                )
    return decision, written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.eval.selection",
        description="Run the S06 winner selection flow and write selection.md.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(_DEFAULT_GENERATION_CANDIDATES),
        help="Registry model types to compare.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--path", default=SELECTION_REPORT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = build_parser().parse_args(argv)
    decision, written = generate_real_selection_report(
        split_version=args.version,
        candidate_types=tuple(args.candidates),
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        objective=args.objective,
        alpha=args.alpha,
        path=args.path,
    )
    print(f"wrote {written} (winner={decision.winner}, ship={decision.ship})")
    return 0


#: Default registry members the generation entrypoint compares.
_DEFAULT_GENERATION_CANDIDATES: tuple[str, ...] = (
    "logistic-regression-l2",
    "random-forest",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
