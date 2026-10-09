"""Repeated stratified k-fold cross-validation across seeds (S06/T01).

Single-split evaluation is how a lucky test partition becomes a leaderboard
winner. This module replaces the point estimate with a *distribution*: every
candidate model is re-fit and re-scored on ``n_repeats`` differently-seeded
stratified k-fold partitions, and each fold is scored through the shared
evaluation contract (:func:`heart.eval.contract.evaluate`), so a repeated-CV
metric value is drawn from exactly the same producer as the leaderboard.

What it produces
----------------
* ``per_fold`` — one :class:`FoldScore` per (repeat, fold) pair, each holding
  the full canonical metric dict for one candidate;
* :meth:`RepeatedCVResult.aggregate` — mean / std / min / max for every scalar
  metric leaf (flattened, dotted key names) across all folds;
* ``per_repeat_means`` — the per-repeat mean of every scalar metric leaf, the
  paired sample a later paired significance test (S06/T02) consumes;
* :func:`primary_metric_values` — the primary metric (ROC-AUC) per fold, in a
  stable (repeat, fold) order, ready for paired comparison across candidates
  that were run on the same folds;
* :func:`rank_candidates` — candidates ordered by aggregated primary metric.

Determinism contract
--------------------
Fold membership for repeat ``r`` comes from
:class:`~sklearn.model_selection.StratifiedKFold` seeded with
``seed + r``. Running the same candidates on the same arrays with the same
config therefore yields byte-identical per-fold metrics, and candidates run
against the same config share identical fold wording, which is what makes the
comparison *paired*.

Model construction
------------------
Candidates are supplied as ``name -> model_factory`` where
``model_factory() -> fitted-predict object`` returns a *fresh, unfitted*
estimator (typically the registry's :func:`heart.models.registry.build_pipeline`
bound to spec + tuned params). One new instance is fitted per fold; a factory
that mutates or caches a fitted estimator is a determinism violation and is the
caller's responsibility to avoid.

Observability
-------------
:func:`run_repeated_cv` logs per candidate the fold count, elapsed wall time
and the aggregated primary-metric mean and std at ``INFO``. Failures are
named: invalid configuration raises :class:`RepeatedCVConfigError`; malformed
inputs raise :class:`RepeatedCVDataError`; a factory that refuses to build or
fit raises :class:`FoldFitError` (wrapping the underlying exception as
``__cause__``); an empty candidate mapping raises :class:`EmptyCandidateError`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from heart.config import RANDOM_SEED
from heart.eval.contract import (
    EvaluationSplit,
    PRIMARY_METRIC,
    evaluate,
    flatten_metrics,
)
from heart.eval.metrics import METRIC_KEYS

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_N_FOLDS",
    "DEFAULT_N_REPEATS",
    "RepeatedCVError",
    "RepeatedCVConfigError",
    "RepeatedCVDataError",
    "EmptyCandidateError",
    "CandidateNotFoundError",
    "FoldFitError",
    "RepeatedCVConfig",
    "FoldScore",
    "RepeatedCVResult",
    "run_repeated_cv",
    "run_repeated_cv_comparison",
    "primary_metric_values",
    "rank_candidates",
    "repeated_cv_report",
]


# ---------------------------------------------------------------------------
# Declared configuration
# ---------------------------------------------------------------------------

#: Default stratified folds per repeat (~20% held out per fold).
DEFAULT_N_FOLDS: int = 5

#: Default number of differently-seeded repeats; 5x5 = 25 paired samples per
#: candidate, enough for the corrected resampled t-test of T02 to be
#: meaningful on this dataset size.
DEFAULT_N_REPEATS: int = 5

#: Number of digit characters used when rendering aggregated metrics.
_REPORT_DECIMALS: int = 6


class RepeatedCVError(Exception):
    """Base class for every repeated-CV failure."""


class RepeatedCVConfigError(RepeatedCVError):
    """The requested repeated-CV configuration is invalid."""


class RepeatedCVDataError(RepeatedCVError):
    """The evaluation data cannot support stratified k-fold CV."""


class EmptyCandidateError(RepeatedCVError):
    """No candidate models were supplied."""


class CandidateNotFoundError(RepeatedCVError):
    """A named candidate is absent from the supplied mapping."""


class FoldFitError(RepeatedCVError):
    """Building or fitting a candidate on one fold failed."""


@dataclass(frozen=True)
class RepeatedCVConfig:
    """Declared repeated-CV configuration.

    Attributes
    ----------
    n_folds:
        Stratified folds per repeat. Must be at least 2 and may not exceed
        the minority class count (StratifiedKFold enforces that and its
        error is re-raised as :class:`RepeatedCVDataError`).
    n_repeats:
        Differently-seeded repetitions. Must be at least 1.
    seed:
        Base seed; repeat ``r`` uses fold seed ``seed + r`` so the partition
        scheme is a pure function of (seed, repeat index).
    """

    n_folds: int = DEFAULT_N_FOLDS
    n_repeats: int = DEFAULT_N_REPEATS
    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if isinstance(self.n_folds, bool) or not isinstance(self.n_folds, int):
            raise RepeatedCVConfigError(
                f"n_folds must be an integer, got {type(self.n_folds).__name__} "
                f"({self.n_folds!r})."
            )
        if self.n_folds < 2:
            raise RepeatedCVConfigError(
                f"n_folds must be at least 2, got {self.n_folds}; a single fold "
                "cannot hold out and use the same rows."
            )
        if isinstance(self.n_repeats, bool) or not isinstance(self.n_repeats, int):
            raise RepeatedCVConfigError(
                f"n_repeats must be an integer, got "
                f"{type(self.n_repeats).__name__} ({self.n_repeats!r})."
            )
        if self.n_repeats < 1:
            raise RepeatedCVConfigError(
                f"n_repeats must be at least 1, got {self.n_repeats}."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RepeatedCVConfigError(
                f"seed must be an integer, got "
                f"{type(self.seed).__name__} ({self.seed!r})."
            )

    @property
    def n_samples_per_model(self) -> int:
        """Total fit+score evaluations executed per candidate model."""
        return self.n_folds * self.n_repeats

    def fold_seed(self, repeat: int) -> int:
        """The deterministic fold seed for one repeat index."""
        return self.seed + int(repeat)

    def fingerprint(self) -> str:
        return f"k={self.n_folds}x{self.n_repeats}@seed={self.seed}"

    def to_dict(self) -> dict[str, object]:
        return {
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FoldScore:
    """One candidate's canonical metric dict on one (repeat, fold) pair."""

    model_name: str
    repeat: int
    fold: int
    n_train: int
    n_test: int
    metrics: dict[str, object]

    @property
    def primary(self) -> float:
        return float(self.metrics[PRIMARY_METRIC])

    def flat(self) -> dict[str, float]:
        return flatten_metrics(self.metrics)


@dataclass(frozen=True)
class CandidateAggregate:
    """Aggregated scalar-metric distribution over every fold of one model."""

    model_name: str
    mean: dict[str, float]
    std: dict[str, float]
    min_value: dict[str, float]
    max_value: dict[str, float]

    def primary_mean_std(self) -> tuple[float, float]:
        return self.mean[PRIMARY_METRIC], self.std[PRIMARY_METRIC]


@dataclass(frozen=True)
class RepeatedCVResult:
    """One candidate's per-fold metric distribution plus its aggregates."""

    model_name: str
    config: RepeatedCVConfig
    per_fold: tuple[FoldScore, ...] = field(repr=False)
    per_repeat_means: dict[int, dict[str, float]]

    @property
    def aggregate(self) -> CandidateAggregate:
        flat_folds = [score.flat() for score in self.per_fold]
        keys: list[str] = []
        seen: set[str] = set()
        for flat in flat_folds:
            for key in flat:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        vmin: dict[str, float] = {}
        vmax: dict[str, float] = {}
        for key in keys:
            # flatten_metrics skips None placeholders (empty calibration bins),
            # so a scalar-leaf key may be absent from some folds. Aggregate over
            # the folds that carry the value rather than silently dropping the
            # key from the distribution.
            column = np.asarray(
                [flat[key] for flat in flat_folds if key in flat], dtype=float
            )
            if column.size == 0:
                continue
            mean[key] = float(column.mean())
            std[key] = float(column.std(ddof=1)) if column.size > 1 else 0.0
            vmin[key] = float(column.min())
            vmax[key] = float(column.max())
        return CandidateAggregate(
            model_name=self.model_name,
            mean=mean,
            std=std,
            min_value=vmin,
            max_value=vmax,
        )

    def to_dict(self) -> dict[str, object]:
        aggregate = self.aggregate
        return {
            "model_name": self.model_name,
            "config": self.config.to_dict(),
            "n_folds": len(self.per_fold),
            "primary_metric": {
                "mean": aggregate.mean[PRIMARY_METRIC],
                "std": aggregate.std[PRIMARY_METRIC],
                "min": aggregate.min_value[PRIMARY_METRIC],
                "max": aggregate.max_value[PRIMARY_METRIC],
            },
            "mean": {
                key: round(value, 6)
                for key, value in sorted(aggregate.mean.items())
            },
            "std": {
                key: round(value, 6)
                for key, value in sorted(aggregate.std.items())
            },
        }

    def summary(self) -> str:
        aggregate = self.aggregate
        primary_mean, primary_std = aggregate.primary_mean_std()
        return (
            f"{self.model_name}: {PRIMARY_METRIC} "
            f"{primary_mean:.4f} +/- {primary_std:.4f} "
            f"({len(self.per_fold)} folds, {self.config.fingerprint()})"
        )


def _coerce_frame_and_labels(
    X: object, y: object
) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(X, pd.DataFrame):
        raise RepeatedCVDataError(
            f"X must be a pandas.DataFrame, got {type(X).__name__}."
        )
    labels = pd.Series(y).to_numpy()
    if len(X) != labels.size:
        raise RepeatedCVDataError(
            f"X has {len(X)} rows but y has {labels.size} label(s); they must "
            "align."
        )
    if labels.size == 0:
        raise RepeatedCVDataError("Repeated CV needs at least one labelled row.")
    classes = np.unique(labels)
    if classes.size != 2 or not np.all(np.isin(classes, (0, 1))):
        raise RepeatedCVDataError(
            f"y must be a binary 0/1 vector, got labels {classes.tolist()}; "
            "stratified binary CV is undefined otherwise."
        )
    return X, labels


def _check_folds_fit_classes(config: RepeatedCVConfig, y: np.ndarray) -> None:
    min_class = int(np.bincount(y.astype(int)).min())
    if config.n_folds > min_class:
        raise RepeatedCVDataError(
            f"n_folds={config.n_folds} exceeds the minority class count "
            f"({min_class}); every fold needs both classes."
        )


def run_repeated_cv(
    model_name: str,
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: Sequence[object],
    *,
    config: RepeatedCVConfig | None = None,
) -> RepeatedCVResult:
    """Repeated stratified k-fold CV for one candidate, via the shared contract.

    Parameters
    ----------
    model_name:
        Stable candidate key used in results, ranking, and reports.
    model_factory:
        ``() -> object`` returning a fresh, *unfitted* object exposing
        ``fit(X, y)``, ``predict(X)`` and ``predict_proba(X)``.
    X, y:
        The full design matrix and binary labels — folds are carved here, not
        supplied.
    config:
        :class:`RepeatedCVConfig`; defaults to
        ``(n_folds=DEFAULT_N_FOLDS, n_repeats=DEFAULT_N_REPEATS, seed=RANDOM_SEED)``.

    Returns
    -------
    RepeatedCVResult
        Per-fold canonical metric dicts plus per-repeat means.

    Raises
    ------
    RepeatedCVConfigError, RepeatedCVDataError, FoldFitError
        All failure paths are named; :class:`FoldFitError` chains the
        underlying estimator exception as ``__cause__``.
    """
    cfg = config or RepeatedCVConfig()
    features, labels = _coerce_frame_and_labels(X, y)
    _check_folds_fit_classes(cfg, labels)

    if not callable(model_factory):
        raise FoldFitError(
            f"model_factory for {model_name!r} is not callable "
            f"({type(model_factory).__name__}); repeated CV builds one fresh "
            "model per fold."
        )

    fold_scores: list[FoldScore] = []
    per_repeat_means: dict[int, dict[str, float]] = {}
    started = time.perf_counter()

    for repeat in range(cfg.n_repeats):
        splitter = StratifiedKFold(
            n_splits=cfg.n_folds, shuffle=True, random_state=cfg.fold_seed(repeat)
        )
        repeat_flats: list[dict[str, float]] = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
            try:
                model = model_factory()
                model.fit(features.iloc[train_idx], labels[train_idx])
            except Exception as exc:  # noqa: BLE001 - fail named, not bare
                raise FoldFitError(
                    f"Candidate {model_name!r} failed to build or fit on "
                    f"repeat {repeat} fold {fold} ({cfg.fingerprint()}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            split = EvaluationSplit(
                X_test=features.iloc[test_idx],
                y_test=labels[test_idx],
                name=f"{model_name}/repeat{repeat}/fold{fold}",
            )
            metrics = evaluate(model, split)
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
            repeat_flats.append(flatten_metrics(metrics))
        union_keys: list[str] = []
        seen: set[str] = set()
        for flat in repeat_flats:
            for key in flat:
                if key not in seen:
                    seen.add(key)
                    union_keys.append(key)
        per_repeat_means[repeat] = {}
        for key in union_keys:
            values = [flat[key] for flat in repeat_flats if key in flat]
            per_repeat_means[repeat][key] = float(np.mean(values))

    elapsed = time.perf_counter() - started
    result = RepeatedCVResult(
        model_name=model_name,
        config=cfg,
        per_fold=tuple(fold_scores),
        per_repeat_means=per_repeat_means,
    )
    primary_mean, primary_std = result.aggregate.primary_mean_std()
    logger.info(
        "Repeated CV for %s: %d fits in %.1fs — %s=%.4f +/- %.4f",
        model_name,
        cfg.n_samples_per_model,
        elapsed,
        PRIMARY_METRIC,
        primary_mean,
        primary_std,
    )
    return result


def run_repeated_cv_comparison(
    candidates: Mapping[str, Callable[[], object]],
    X: pd.DataFrame,
    y: object,
    *,
    config: RepeatedCVConfig | None = None,
) -> dict[str, RepeatedCVResult]:
    """Repeated CV for every candidate on the *same* folds, keyed by name.

    Running candidates sequentially against one config guarantees they share
    fold membership (deterministic per repeat seed), which is the pairing a
    later significance test relies on.
    """
    if not isinstance(candidates, Mapping):
        raise EmptyCandidateError(
            f"candidates must be a Mapping of name -> factory, got "
            f"{type(candidates).__name__}."
        )
    if not candidates:
        raise EmptyCandidateError(
            "candidates is empty; repeated-CV comparison needs at least one "
            "candidate model."
        )
    cfg = config or RepeatedCVConfig()
    results: dict[str, RepeatedCVResult] = {}
    for name in candidates:
        results[name] = run_repeated_cv(name, candidates[name], X, y, config=cfg)
    return results


def primary_metric_values(results: Mapping[str, RepeatedCVResult]) -> dict[str, np.ndarray]:
    """Per-fold primary-metric vectors, aligned across truly paired candidates.

    Values are ordered by (repeat, fold). When candidates share a config and
    the same underlying partition seeds, index ``i`` of every returned vector
    corresponds to the same held-out rows, so these vectors are paired samples.
    A length mismatch between results means they were not run on the same
    config and is reported loudly: silent pairing of misaligned folds would
    manufacture false significance in T02.
    """
    sizes = {len(result.per_fold) for result in results.values()}
    if len(sizes) > 1:
        raise RepeatedCVDataError(
            f"Candidates have differing fold counts {sorted(sizes)}; paired "
            "values require identical RepeatedCVConfig across candidates."
        )
    return {
        name: np.asarray([score.primary for score in result.per_fold], dtype=float)
        for name, result in results.items()
    }


def rank_candidates(
    results: Mapping[str, RepeatedCVResult],
    *,
    metric: str = PRIMARY_METRIC,
) -> list[tuple[str, float, float]]:
    """Candidates ordered by aggregated ``metric`` mean, strongest first.

    Returns ``(model_name, mean, std)`` tuples. An unknown metric raises
    :class:`RepeatedCVDataError` rather than ranking on an absent column.
    """
    ranked: list[tuple[str, float, float]] = []
    for name, result in results.items():
        aggregate = result.aggregate
        if metric not in aggregate.mean:
            raise RepeatedCVDataError(
                f"Metric {metric!r} is not an aggregated scalar; declared keys "
                f"include {sorted(aggregate.mean)[:4]}..."
            )
        ranked.append(
            (name, aggregate.mean[metric], aggregate.std[metric])
        )
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def repeated_cv_report(
    results: Mapping[str, RepeatedCVResult],
    *,
    config: RepeatedCVConfig | None = None,
    decimals: int = _REPORT_DECIMALS,
) -> str:
    """Render the comparison as a markdown table of aggregated scalar metrics.

    Rows are ranked by aggregated primary metric. The schema of the columns is
    the declared scalar-suite order (:data:`heart.eval.metrics.METRIC_KEYS`
    slice), so regenerating the report cannot invent a metric.
    """
    cfg = config or next(iter(results.values())).config
    flat_folds = [score.flat() for score in next(iter(results.values())).per_fold]
    present_keys = set(flat_folds[0])
    scalar_keys = [key for key in METRIC_KEYS if key in present_keys]
    ordered = rank_candidates(results)
    lines = [
        f"Repeated stratified CV ({cfg.fingerprint()}, "
        f"{cfg.n_samples_per_model} fits per candidate)",
        "",
        "| model | " + " | ".join(scalar_keys) + " |",
        "| --- | " + " | ".join(["---"] * len(scalar_keys)) + " |",
    ]
    winner = ordered[0]
    for name, _mean, _std in ordered:
        row_means = results[name].aggregate.mean
        row_stds = results[name].aggregate.std
        cells = [
            f"{row_means[key]:.{decimals}f} ± {row_stds[key]:.{decimals}f}"
            for key in scalar_keys
        ]
        marker = "*" if name == winner[0] else ""
        marker_note = " (winner)" if name == winner[0] else ""
        lines.append(f"| **{name}**{marker_note} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
