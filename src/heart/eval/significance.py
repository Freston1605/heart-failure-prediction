"""Paired significance testing for model comparison (S06/T02).

A repeated-CV mean above another model's mean is a claim; this module is the
receipt. Two paired tests answer two different questions:

1. **McNemar's test** — "do the two models disagree with each other more than
   chance on the same held-out rows?" It uses only the *class-agreement*
   contingency: ``b`` = rows where model A is right and model B is wrong,
   ``c`` = rows where A is wrong and B is right. The exact binomial variant
   (``scipy.stats.binomtest`` on the smaller discordant cell against 0.5) is
   used instead of the chi-square approximation because the discordant count
   on this dataset size is small enough that the approximation is unsafe.

2. **Corrected resampled t-test** (Nadeau & Bengio, 2003) — "is the mean
   repeated-CV metric difference real, given that the folds overlap?" The
   naive paired t-test on per-fold scores inflates significance because fold
   test sets share rows. The correction inflates the standard error by
   ``(1 / n_folds + n_test / n_train)``, where ``n_test`` / ``n_train`` are
   the held-out and training row totals for one repeat, so repeated-CV
   comparisons are not over-confident.

Both tests are *paired*: they consume predictions or per-fold metrics drawn
from identical partitions, which is exactly what
:func:`heart.eval.repeated_cv.run_repeated_cv_comparison` produces (same
config -> same fold seeds -> index-aligned fold scores).

Failure contract
----------------
Everything is named, mirroring the T01 taxonomy: malformed vectors raise
:class:`SignificanceDataError`; an unusable configuration (alpha outside
(0, 1), n_folds < 2) raises :class:`SignificanceConfigError`; vectors that do
not align as paired samples raise :class:`PairedAlignmentError` rather than
being silently zipped to the shorter length — a silent zip would manufacture
false significance for T03's winner selection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from heart.eval.repeated_cv import (
    PRIMARY_METRIC,
    RepeatedCVConfig,
    RepeatedCVDataError,
    RepeatedCVResult,
    rank_candidates,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ALPHA",
    "SignificanceError",
    "SignificanceConfigError",
    "SignificanceDataError",
    "PairedAlignmentError",
    "McNemarResult",
    "CorrectedTResult",
    "PairwiseComparison",
    "mcnemar_test",
    "corrected_resampled_t_test",
    "compare_repeated_cv",
    "significance_report",
]


#: Family-wise-friendly significance level used across the S06 selection flow.
DEFAULT_ALPHA: float = 0.05


class SignificanceError(Exception):
    """Base class for every significance-testing failure."""


class SignificanceConfigError(SignificanceError):
    """The requested significance-test configuration is invalid."""


class SignificanceDataError(SignificanceError):
    """The supplied paired samples are malformed and cannot be tested."""


class PairedAlignmentError(SignificanceError):
    """The paired samples do not align; zipping them would fabricate p-values."""


def _check_alpha(alpha: object) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)):
        raise SignificanceConfigError(
            f"alpha must be a number in (0, 1), got "
            f"{type(alpha).__name__} ({alpha!r})."
        )
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise SignificanceConfigError(
            f"alpha must be strictly between 0 and 1, got {alpha!r}."
        )
    return value


def _as_paired_vector(values: object, *, label: str, n_expected: int) -> np.ndarray:
    """Coerce one paired sample to a 1-D finite float vector of ``n_expected``."""
    if isinstance(values, (str, bytes)):
        raise SignificanceDataError(
            f"{label} is a {type(values).__name__}, not a paired numeric sample."
        )
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SignificanceDataError(
            f"{label} could not be coerced to a numeric vector: {exc}"
        ) from exc
    if vector.ndim != 1:
        raise SignificanceDataError(
            f"{label} must be one-dimensional, got shape {vector.shape}."
        )
    if vector.size == 0:
        raise SignificanceDataError(f"{label} is empty; a paired test needs data.")
    if vector.size != n_expected:
        raise PairedAlignmentError(
            f"{label} has {vector.size} entries but the paired sample has "
            f"{n_expected}; pairing misaligned samples would fabricate a "
            "p-value, so this is rejected outright."
        )
    if not np.all(np.isfinite(vector)):
        bad = int(np.count_nonzero(~np.isfinite(vector)))
        raise SignificanceDataError(
            f"{label} contains {bad} non-finite value(s) (NaN or inf); metrics "
            "must be computed before they are tested."
        )
    return vector


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of the exact McNemar test on one pair of prediction vectors.

    ``b`` and ``c`` are the two discordant cells (A right/B wrong and
    A wrong/B right); ties on both-correct and both-wrong rows do not enter
    the statistic. A zero discordant count means the models agree everywhere
    and the result is trivially non-significant (``p = 1.0``).
    """

    b: int
    c: int
    n_samples: int
    n_agree: int
    statistic: int
    p_value: float
    alpha: float
    significant: bool

    @property
    def n_wrong(self) -> int:
        return self.b + self.c

    def verdict(self) -> str:
        if not self.significant:
            return "not significant"
        return f"significant (alpha={self.alpha:g})"

    def to_dict(self) -> dict[str, object]:
        return {
            "test": "mcnemar_exact",
            "b": self.b,
            "c": self.c,
            "n_samples": self.n_samples,
            "n_agree": self.n_agree,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class CorrectedTResult:
    """Outcome of the Nadeau-Bengio corrected resampled t-test.

    ``overfit_correction`` is the ``(1 / n_folds + n_test / n_train)`` factor
    applied to the variance: the larger it is, the more the fold overlap was
    penalised. ``df`` is ``len(diffs) - 1``.
    """

    mean_diff: float
    std_diff: float
    t_statistic: float
    df: int
    n_samples_paired: int
    overfit_correction: float
    p_value: float
    alpha: float
    significant: bool

    def verdict(self) -> str:
        if not self.significant:
            return "not significant"
        direction = "higher" if self.mean_diff > 0 else "lower"
        return (
            f"significant (alpha={self.alpha:g}, A {direction} "
            f"by {self.mean_diff:+.6f})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "test": "corrected_resampled_t",
            "mean_diff": self.mean_diff,
            "std_diff": self.std_diff,
            "t_statistic": self.t_statistic,
            "df": self.df,
            "n_samples_paired": self.n_samples_paired,
            "overfit_correction": self.overfit_correction,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class PairwiseComparison:
    """Both paired tests for one ordered model pair plus the joint verdict."""

    model_a: str
    model_b: str
    metric: str
    agreement: McNemarResult | None
    corrected_t: CorrectedTResult
    alpha: float

    @property
    def significant(self) -> bool:
        return self.corrected_t.significant

    def verdict(self) -> str:
        """The joint, gate-keeping verdict for selection (T03 consumes this)."""
        parts = [f"corrected-t: {self.corrected_t.verdict()}"]
        if self.agreement is not None:
            parts.append(f"McNemar: {self.agreement.verdict()}")
        if not self.corrected_t.significant:
            return "advantage is noise (" + "; ".join(parts) + ")"
        return "advantage is real (" + "; ".join(parts) + ")"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "alpha": self.alpha,
            "agreement": self.agreement.to_dict() if self.agreement else None,
            "corrected_t": self.corrected_t.to_dict(),
            "verdict": self.verdict(),
        }


def mcnemar_test(
    y_true: Sequence[object],
    pred_a: Sequence[object],
    pred_b: Sequence[object],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> McNemarResult:
    """Exact McNemar's test comparing two models' hard predictions.

    Parameters
    ----------
    y_true, pred_a, pred_b:
        Aligned binary 0/1 vectors for the same held-out rows.
    alpha:
        Significance level; both models' agreement cells are tabulated and
        the two-sided exact binomial p-value on the discordant count is
        compared against it.

    Returns
    ------
    McNemarResult
        Named fields for the contingency cells, statistic, p-value and
        significance call. Identical predictions yield ``p = 1.0``.
    """
    checked = _check_alpha(alpha)
    labels = np.asarray(y_true)
    vector_a = np.asarray(pred_a)
    vector_b = np.asarray(pred_b)
    if labels.ndim != 1 or vector_a.ndim != 1 or vector_b.ndim != 1:
        raise SignificanceDataError(
            "y_true, pred_a and pred_b must all be one-dimensional vectors; "
            f"got shapes {labels.shape}, {vector_a.shape}, {vector_b.shape}."
        )
    sizes = {labels.size, vector_a.size, vector_b.size}
    if len(sizes) != 1 and 0 in sizes:
        raise SignificanceDataError(
            "One of the paired vectors is empty; McNemar's test needs data."
        )
    if len(sizes) != 1:
        raise PairedAlignmentError(
            f"Paired vectors have differing lengths {sorted(sizes)}; McNemar "
            "compares predictions on the *same* rows, and zipping mismatched "
            "rows would fabricate the contingency table."
        )
    if labels.size == 0:
        raise SignificanceDataError("McNemar's test needs at least one labelled row.")
    for name, vector in (("y_true", labels), ("pred_a", vector_a), ("pred_b", vector_b)):
        classes = set(np.unique(vector).tolist())
        if not classes.issubset({0, 1}):
            raise SignificanceDataError(
                f"{name} must be binary 0/1, found classes {sorted(classes)}."
            )

    correct_a = vector_a == labels
    correct_b = vector_b == labels
    b = int(np.count_nonzero(correct_a & ~correct_b))
    c = int(np.count_nonzero(~correct_a & correct_b))
    n_agree = int(np.count_nonzero(correct_a == correct_b))
    n_discordant = b + c

    if n_discordant == 0:
        p_value = 1.0
    else:
        result = stats.binomtest(min(b, c), n_discordant, p=0.5)
        p_value = float(result.pvalue)
    significant = bool(p_value < checked)
    result = McNemarResult(
        b=b,
        c=c,
        n_samples=int(labels.size),
        n_agree=n_agree,
        statistic=min(b, c),
        p_value=p_value,
        alpha=checked,
        significant=significant,
    )
    logger.info(
        "McNemar exact test: b=%d, c=%d (agreements=%d/%d) -> p=%.4g "
        f"({'significant' if significant else 'not significant'} at "
        "alpha=%.3g)",
        result.b,
        result.c,
        result.n_agree,
        result.n_samples,
        result.p_value,
        checked,
    )
    return result


def corrected_resampled_t_test(
    diffs: Sequence[object],
    *,
    n_folds: int,
    n_test: float,
    n_train: float,
    alpha: float = DEFAULT_ALPHA,
) -> CorrectedTResult:
    """Nadeau & Bengio corrected resampled t-test on per-(fold, repeat) diffs.

    Parameters
    ----------
    diffs:
        Per-test-set metric differences (model A minus model B) on identical
        partitions, ordered arbitrarily but *paired*; zero-variance diffs
        (identical models) yield ``t = 0`` and ``p = 1.0`` by convention.
    n_folds:
        Folds per repeat (>= 2); enters the correction as ``1 / n_folds``.
    n_test, n_train:
        Held-out and training row totals for one repeat; their ratio is the
        overlap term of the correction (both must be positive and finite).
    alpha:
        Two-sided significance level.

    Raises
    ------
    SignificanceConfigError
        Invalid ``n_folds``, non-positive/non-finite sizes, or bad alpha.
    """
    checked = _check_alpha(alpha)
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise SignificanceConfigError(
            f"n_folds must be an integer, got {type(n_folds).__name__}."
        )
    if int(n_folds) < 2:
        raise SignificanceConfigError(
            f"n_folds must be at least 2 for a corrected resampled t-test, "
            f"got {n_folds}."
        )
    for name, value in (("n_test", n_test), ("n_train", n_train)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise SignificanceConfigError(
                f"{name} must be a positive number, got "
                f"{type(value).__name__} ({value!r})."
            )
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise SignificanceConfigError(
                f"{name} must be a positive finite row count, got {value!r}."
            )

    sample = _as_paired_vector(diffs, label="diffs", n_expected=len(np.asarray(diffs)))
    mean_diff = float(sample.mean())
    n_paired = int(sample.size)
    if n_paired < 2:
        raise SignificanceDataError(
            f"A corrected resampled t-test needs at least 2 paired diffs, got "
            f"{n_paired}."
        )
    std_diff = float(sample.std(ddof=1))
    variance = sample.var(ddof=1)
    overfit_correction = 1.0 / int(n_folds) + float(n_test) / float(n_train)
    if variance <= 0.0:
        # Identical models: zero variance means the difference distribution is
        # a point mass at (almost certainly) zero. Convention per test-text:
        # report t = 0, p = 1 for an exactly-zero mean and t = ±inf, p -> 0
        # otherwise; a point mass off zero is itself evidence (Nadeau & Bengio
        # never observe this with real CV, but it must not crash).
        significant = mean_diff != 0.0
        p_value = 1.0 if mean_diff == 0.0 else 0.0
        t_statistic = 0.0 if mean_diff == 0.0 else math.copysign(
            math.inf, mean_diff
        )
        result = CorrectedTResult(
            mean_diff=mean_diff,
            std_diff=std_diff,
            t_statistic=t_statistic,
            df=n_paired - 1,
            n_samples_paired=n_paired,
            overfit_correction=overfit_correction,
            p_value=p_value,
            alpha=checked,
            significant=significant,
        )
        logger.info(
            "Corrected resampled t-test (degenerate, zero-variance diffs): "
            "mean=%+.6g -> p=%.4g",
            mean_diff,
            p_value,
        )
        return result

    corrected_se = math.sqrt(variance * overfit_correction)
    t_statistic = mean_diff / corrected_se
    df = n_paired - 1
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df))
    p_value = min(max(p_value, 0.0), 1.0)
    significant = bool(p_value < checked)
    result = CorrectedTResult(
        mean_diff=mean_diff,
        std_diff=std_diff,
        t_statistic=float(t_statistic),
        df=df,
        n_samples_paired=n_paired,
        overfit_correction=overfit_correction,
        p_value=p_value,
        alpha=checked,
        significant=significant,
    )
    logger.info(
        "Corrected resampled t-test: mean=%+.6f, s=%.6f, correction=%.4f -> "
        "t=%.4f df=%d p=%.4g (%s at alpha=%.3g)",
        result.mean_diff,
        result.std_diff,
        result.overfit_correction,
        result.t_statistic,
        result.df,
        result.p_value,
        "significant" if significant else "not significant",
        checked,
    )
    return result


def _fold_diffs(
    result_a: RepeatedCVResult, result_b: RepeatedCVResult, *, metric: str
) -> np.ndarray:
    """Primary-metric differences on identical (repeat, fold) partitions."""
    scores_a = {(score.repeat, score.fold): score for score in result_a.per_fold}
    scores_b = {(score.repeat, score.fold): score for score in result_b.per_fold}
    if set(scores_a) != set(scores_b):
        raise PairedAlignmentError(
            f"{result_a.model_name!r} and {result_b.model_name!r} were run "
            "with different fold keys; paired significance requires identical "
            "repeats and folds across candidates."
        )
    if result_a.config != result_b.config:
        raise PairedAlignmentError(
            f"{result_a.model_name!r} (config {result_a.config.fingerprint()}) "
            f"and {result_b.model_name!r} (config "
            f"{result_b.config.fingerprint()}) used different RepeatedCVConfig"
            "s; comparing their fold scores mispairs samples."
        )
    diffs: list[float] = []
    for key in sorted(scores_a):
        value_a = scores_a[key].metrics.get(metric)
        value_b = scores_b[key].metrics.get(metric)
        if value_a is None or value_b is None:
            raise SignificanceDataError(
                f"Metric {metric!r} is absent from a fold of the paired "
                f"candidates; {result_a.model_name!r} "
                f"({'ok' if value_a is not None else 'missing'}) and "
                f"{result_b.model_name!r} "
                f"({'ok' if value_b is not None else 'missing'})."
            )
        diff = float(value_a) - float(value_b)
        if not math.isfinite(diff):
            raise SignificanceDataError(
                f"Fold {key} produced a non-finite {metric!r} difference; the "
                "underlying metric must be recomputed before testing."
            )
        diffs.append(diff)
    return np.asarray(diffs, dtype=float)


def _repeat_row_totals(result: RepeatedCVResult) -> tuple[float, float, int]:
    """Mean per-repeat (test rows, train rows) totals plus the fold count."""
    repeats = sorted({score.repeat for score in result.per_fold})
    fold_counts = {
        repeat: len({score.fold for score in result.per_fold if score.repeat == repeat})
        for repeat in repeats
    }
    if len(set(fold_counts.values())) != 1:
        raise PairedAlignmentError(
            f"{result.model_name!r} has differing fold counts per repeat "
            f"{sorted(set(fold_counts.values()))}; the fold structure is not "
            "a clean rectangular grid and cannot be paired."
        )
    n_folds = next(iter(fold_counts.values()))
    test_totals = []
    train_totals = []
    for repeat in repeats:
        folds = [score for score in result.per_fold if score.repeat == repeat]
        test_totals.append(sum(score.n_test for score in folds))
        train_totals.append(sum(score.n_train for score in folds))
    return float(np.mean(test_totals)), float(np.mean(train_totals)), n_folds


def compare_repeated_cv(
    results: Mapping[str, RepeatedCVResult],
    model_a: str,
    model_b: str,
    *,
    metric: str = PRIMARY_METRIC,
    alpha: float = DEFAULT_ALPHA,
    mcnemar_scores: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    mcnemar_y_true: np.ndarray | None = None,
) -> PairwiseComparison:
    """Paired significance comparison of two repeated-CV candidates.

    The corrected resampled t-test always runs on the per-fold ``metric``
    differences. McNemar additionally runs when ``mcnemar_scores`` and
    ``mcnemar_y_true`` supply aligned held-out predictions for both models
    (and is omitted, not defaulted, otherwise).

    Raises
    ------
    PairedAlignmentError
        Unknown candidate name, mismatched configs, or misaligned fold keys.
    """
    for name in (model_a, model_b):
        if name not in results:
            raise PairedAlignmentError(
                f"Candidate {name!r} is not in the comparison "
                f"{sorted(results)}; comparison is against supplied results "
                "only."
            )
    result_a = results[model_a]
    result_b = results[model_b]
    diffs = _fold_diffs(result_a, result_b, metric=metric)
    n_test, n_train, _ = _repeat_row_totals(result_a)
    corrected = corrected_resampled_t_test(
        diffs,
        n_folds=result_a.config.n_folds,
        n_test=n_test,
        n_train=n_train,
        alpha=alpha,
    )
    agreement: McNemarResult | None = None
    if mcnemar_scores is not None:
        missing = [name for name in (model_a, model_b) if name not in mcnemar_scores]
        if missing:
            raise SignificanceDataError(
                f"mcnemar_scores lacks {missing}; predictions for both "
                "candidates are needed for the agreement test."
            )
        if mcnemar_y_true is None:
            raise SignificanceDataError(
                "mcnemar_scores was supplied without mcnemar_y_true; McNemar "
                "compares against actual labels, not between models alone."
            )
        pred_a, pred_b = mcnemar_scores[model_a], mcnemar_scores[model_b]
        agreement = mcnemar_test(
            mcnemar_y_true, np.asarray(pred_a), np.asarray(pred_b), alpha=alpha
        )
    comparison = PairwiseComparison(
        model_a=model_a,
        model_b=model_b,
        metric=metric,
        agreement=agreement,
        corrected_t=corrected,
        alpha=alpha,
    )
    logger.info(
        "Paired comparison %s vs %s on %s: %s",
        model_a,
        model_b,
        metric,
        comparison.verdict(),
    )
    return comparison


def significance_report(
    results: Mapping[str, RepeatedCVResult],
    *,
    metric: str = PRIMARY_METRIC,
    alpha: float = DEFAULT_ALPHA,
    baseline: str | None = None,
) -> tuple[str, PairwiseComparison | None]:
    """Markdown significance table plus the winner-vs-baseline verdict.

    Every non-winning candidate is tested against the repeated-CV winner
    (ordered by ``rank_candidates`` on ``metric``) and every other candidate
    against the declared baseline (resolved by default to the candidate whose
    name contains ``logistic``, else the lowest-ranked). Returns
    ``(markdown_report, winner_vs_baseline_comparison)`` — the second element
    is ``None`` only when a single candidate was supplied.
    """
    if not results:
        raise RepeatedCVDataError(
            "significance_report needs at least one repeated-CV result."
        )
    checked = _check_alpha(alpha)
    ranked = rank_candidates(results, metric=metric)
    winner = ranked[0][0]
    if baseline is None:
        logistic_hits = [name for name in results if "logistic" in name.lower()]
        baseline = logistic_hits[0] if logistic_hits else ranked[-1][0]
    if baseline not in results:
        raise PairedAlignmentError(
            f"Declared baseline {baseline!r} is not among candidates "
            f"{sorted(results)}."
        )
    winner_config: RepeatedCVConfig = results[winner].config
    lines = [
        "## Paired significance testing",
        "",
        f"Paired on identical partitions at alpha={checked:g}; the corrected "
        "resampled t-test (Nadeau & Bengio) compares per-fold "
        f"`{metric}` differences, and its variance is inflated by the "
        "fold-overlap correction `(1/n_folds + n_test/n_train)`.",
        "",
        "| comparison | mean diff (A - B) | t | df | correction | p-value | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    winner_vs_baseline: PairwiseComparison | None = None
    subjects = [name for name in ranked if name[0] != winner]
    comparisons: list[PairwiseComparison] = []
    for name, _mean, _std in subjects:
        comparisons.append(
            compare_repeated_cv(
                results, winner, name, metric=metric, alpha=checked
            )
        )
    if baseline != winner and not any(
        {c.model_a, c.model_b} == {winner, baseline} for c in comparisons
    ):
        comparisons.append(
            compare_repeated_cv(
                results, winner, baseline, metric=metric, alpha=checked
            )
        )
    for comparison in comparisons:
        corrected = comparison.corrected_t
        signature = (
            f"{comparison.model_a} vs {comparison.model_b}"
            if comparison.model_a == winner
            else f"{comparison.model_a} (vs winner)"
        )
        lines.append(
            f"| {signature} | {corrected.mean_diff:+.6f} | "
            f"{corrected.t_statistic:+.4f} | {corrected.df} | "
            f"{corrected.overfit_correction:.4f} | {corrected.p_value:.4g} | "
            f"{corrected.verdict()} |"
        )
    winner_vs_baseline = next(
        (
            c
            for c in comparisons
            if {c.model_a, c.model_b} == {winner, baseline}
        ),
        winner_vs_baseline,
    )
    lines.extend(
        [
            "",
            f"Winner: **{winner}** "
            f"(repeated-CV {metric} mean {ranked[0][1]:.6f} ± {ranked[0][2]:.6f}, "
            f"{winner_config.fingerprint()}).",
            f"Baseline: **{baseline}**.",
        ]
    )
    if winner_vs_baseline is not None:
        lines.append("")
        lines.append(
            f"Winner vs baseline ({winner} vs {baseline}): "
            f"p={winner_vs_baseline.corrected_t.p_value:.4g} -> "
            f"{winner_vs_baseline.verdict()}"
        )
    return "\n".join(lines), winner_vs_baseline
