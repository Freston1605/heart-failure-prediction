"""Tests for repeated stratified k-fold CV across seeds (S06/T01).

The contract pinned here:

1. **Distribution, not a point estimate** — a candidate yields one canonical
   metric dict per (repeat, fold) pair, and the aggregate carries mean/std/
   min/max for every scalar metric leaf. Fold and repeat counts match the
   configuration exactly.
2. **Shared evaluation contract** — every fold is scored through
   :func:`heart.eval.contract.evaluate`, so a per-fold ``roc_auc`` comes from
   the same producer as the leaderboard.
3. **Determinism** — the same data, candidate, and config produce identical
   per-fold metrics across two runs with fixed seeds.
4. **Paired alignment** — candidates run on the same config produce per-fold
   primary-metric vectors of identical ordering (the pairing T02's significance
   tests consume); misaligned configs are rejected loudly.
5. **Named failures** — invalid config, malformed data, empty candidates, and
   a broken model factory all raise their named exceptions.

Fixtures are synthetic and self-contained; one integration test runs the
registry's real pipeline over the committed S01 split for two candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold

from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.eval import repeated_cv as rcv
from heart.eval.contract import (
    METRIC_KEYS,
    PRIMARY_METRIC,
    EvaluationSplit,
    evaluate,
)
from heart.eval.repeated_cv import (
    DEFAULT_N_FOLDS,
    DEFAULT_N_REPEATS,
    CandidateNotFoundError,
    EmptyCandidateError,
    FoldFitError,
    FoldScore,
    RepeatedCVConfig,
    RepeatedCVConfigError,
    RepeatedCVDataError,
    RepeatedCVError,
    RepeatedCVResult,
    primary_metric_values,
    rank_candidates,
    repeated_cv_report,
    run_repeated_cv,
    run_repeated_cv_comparison,
)


# ---------------------------------------------------------------------------
# Synthetic data + candidate factories
# ---------------------------------------------------------------------------


def _synthetic_frame(n: int = 300, *, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """A small self-contained binary problem with real (not trivial) signal."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, 3))
    logit = 1.7 * matrix[:, 0] - 1.2 * matrix[:, 1] + 0.3 * matrix[:, 2]
    labels = (rng.random(n) < expit(logit)).astype(int)
    X = pd.DataFrame(matrix, columns=["f0", "f1", "f2"])
    return X, labels


def _fresh_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def _fresh_naive_bayes() -> GaussianNB:
    return GaussianNB()


# ---------------------------------------------------------------------------
# 1. Distribution shape matches configuration
# ---------------------------------------------------------------------------


def test_per_fold_count_matches_folds_and_repeats():
    X, y = _synthetic_frame()
    config = RepeatedCVConfig(n_folds=4, n_repeats=3, seed=1)
    result = run_repeated_cv("logit", _fresh_logistic, X, y, config=config)
    assert isinstance(result, RepeatedCVResult)
    assert len(result.per_fold) == 4 * 3
    assert len(result.per_repeat_means) == 3
    assert all(isinstance(score, FoldScore) for score in result.per_fold)
    # Every canonical metric dict per fold is complete.
    for score in result.per_fold:
        assert PRIMARY_METRIC in score.metrics
        assert set(score.metrics) == set(METRIC_KEYS)
        assert 0.0 <= score.primary <= 1.0
    # Defaults stay wired to the project constants.
    assert DEFAULT_N_FOLDS == 5 and DEFAULT_N_REPEATS == 5


def test_aggregate_carries_mean_std_min_max_for_every_scalar():
    X, y = _synthetic_frame()
    result = run_repeated_cv(
        "nb", _fresh_naive_bayes, X, y, config=RepeatedCVConfig(n_folds=3, n_repeats=2, seed=5)
    )
    aggregate = result.aggregate
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert key in aggregate.mean and key in aggregate.std
        assert key in aggregate.min_value and key in aggregate.max_value
        assert aggregate.min_value[key] <= aggregate.mean[key] <= aggregate.max_value[key]
    primary_mean, primary_std = aggregate.primary_mean_std()
    assert primary_mean == pytest.approx(aggregate.mean[PRIMARY_METRIC])
    assert 0.0 <= primary_std <= 0.5


def test_per_repeat_means_have_one_entry_per_repeat():
    X, y = _synthetic_frame(n=240, seed=3)
    result = run_repeated_cv(
        "logit",
        _fresh_logistic,
        X,
        y,
        config=RepeatedCVConfig(n_folds=3, n_repeats=4, seed=2),
    )
    assert sorted(result.per_repeat_means) == [0, 1, 2, 3]
    for means in result.per_repeat_means.values():
        assert PRIMARY_METRIC in means


# ---------------------------------------------------------------------------
# 2. Runs through the shared evaluation contract
# ---------------------------------------------------------------------------


def test_fold_metrics_match_direct_contract_evaluation():
    """The per-fold roc_auc equals evaluate() on the same rows — same producer."""
    X, y = _synthetic_frame(n=200, seed=13)
    config = RepeatedCVConfig(n_folds=3, n_repeats=1, seed=11)
    result = run_repeated_cv("logit", _fresh_logistic, X, y, config=config)
    score = result.per_fold[0]
    # Recompute the same fold directly through the contract.
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)
    train_idx, test_idx = next(iter(splitter.split(X, y)))
    model = _fresh_logistic()
    model.fit(X.iloc[train_idx], y[train_idx])
    direct = evaluate(
        model,
        EvaluationSplit(
            X_test=X.iloc[test_idx], y_test=y[test_idx], name="direct"
        ),
        n_bins=10,
    )
    assert score.primary == pytest.approx(float(direct[PRIMARY_METRIC]), abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Determinism under fixed seeds
# ---------------------------------------------------------------------------


def test_results_are_deterministic_under_fixed_seeds():
    X, y = _synthetic_frame(n=260, seed=17)
    config = RepeatedCVConfig(n_folds=3, n_repeats=2, seed=9)
    first = run_repeated_cv("nb", _fresh_naive_bayes, X, y, config=config)
    second = run_repeated_cv("nb", _fresh_naive_bayes, X, y, config=config)
    assert [s.primary for s in first.per_fold] == [s.primary for s in second.per_fold]
    assert first.to_dict() == second.to_dict()
    assert first.per_repeat_means == second.per_repeat_means


def test_folds_are_stratified_per_repeat():
    """Each fold holds out a near-constant positive fraction (stratification)."""
    X, y = _synthetic_frame(n=400, seed=23)
    positive_rate = float(np.mean(y))
    config = RepeatedCVConfig(n_folds=5, n_repeats=2, seed=4)
    result = run_repeated_cv("logit", _fresh_logistic, X, y, config=config)
    rates = [score.metrics["prevalence"] for score in result.per_fold]
    for rate in rates:
        assert abs(rate - positive_rate) < 0.1


# ---------------------------------------------------------------------------
# 4. Comparison keys and paired alignment
# ---------------------------------------------------------------------------


def test_comparison_runs_all_candidates_on_same_config():
    X, y = _synthetic_frame(n=200, seed=29)
    results = run_repeated_cv_comparison(
        {"logit": _fresh_logistic, "nb": _fresh_naive_bayes},
        X,
        y,
        config=RepeatedCVConfig(n_folds=3, n_repeats=2, seed=6),
    )
    assert set(results) == {"logit", "nb"}
    aligned = primary_metric_values(results)
    assert set(aligned) == {"logit", "nb"}
    assert aligned["logit"].shape == aligned["nb"].shape == (6,)
    ranked = rank_candidates(results)
    ranked_names = [name for name, _, _ in ranked]
    assert ranked_names == [name for name, _, _ in sorted(ranked, key=lambda t: -t[1])]


def test_paired_values_reject_misaligned_configs():
    X, y = _synthetic_frame(n=150, seed=31)
    logit = run_repeated_cv(
        "logit", _fresh_logistic, X, y, config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1)
    )
    nb = run_repeated_cv(
        "nb", _fresh_naive_bayes, X, y, config=RepeatedCVConfig(n_folds=4, n_repeats=1, seed=1)
    )
    with pytest.raises(RepeatedCVDataError, match="fold counts"):
        primary_metric_values({"logit": logit, "nb": nb})


def test_rank_rejects_unknown_metric():
    X, y = _synthetic_frame(n=120, seed=37)
    result = run_repeated_cv(
        "logit", _fresh_logistic, X, y, config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1)
    )
    with pytest.raises(RepeatedCVDataError, match="aggregated scalar"):
        rank_candidates({"logit": result}, metric="not-a-metric")


def test_report_names_winner_and_renders_markdown_table():
    X, y = _synthetic_frame(n=200, seed=41)
    results = run_repeated_cv_comparison(
        {"logit": _fresh_logistic, "nb": _fresh_naive_bayes},
        X,
        y,
        config=RepeatedCVConfig(n_folds=3, n_repeats=2, seed=8),
    )
    report = repeated_cv_report(results)
    assert report.startswith("Repeated stratified CV (")
    assert "| model |" in report
    assert "(winner)" in report
    top_name = rank_candidates(results)[0][0]
    assert f"**{top_name}** (winner)" in report
    assert "accuracy" in report and "roc_auc" in report


# ---------------------------------------------------------------------------
# 5. Named failures (negative paths)
# ---------------------------------------------------------------------------


def test_invalid_config_is_named():
    with pytest.raises(RepeatedCVConfigError, match="at least 2"):
        RepeatedCVConfig(n_folds=1)
    with pytest.raises(RepeatedCVConfigError, match="at least 1"):
        RepeatedCVConfig(n_repeats=0)
    with pytest.raises(RepeatedCVConfigError, match="n_folds must be an integer"):
        RepeatedCVConfig(n_folds=2.5)


def test_single_class_data_is_rejected():
    X, _ = _synthetic_frame(n=60, seed=43)
    with pytest.raises(RepeatedCVDataError, match="binary"):
        run_repeated_cv(
            "logit", _fresh_logistic, X, np.ones(60, dtype=int),
            config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1),
        )
    with pytest.raises(RepeatedCVDataError, match="binary"):
        run_repeated_cv(
            "logit", _fresh_logistic, X, np.full(60, 2),
            config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1),
        )


def test_folds_exceeding_minority_class_is_rejected():
    X, _y = _synthetic_frame(n=100, seed=47)
    y = np.zeros(100, dtype=int)
    y[:9] = 1  # exactly 9 positives cannot support 10 folds
    with pytest.raises(RepeatedCVDataError, match="minority class count"):
        run_repeated_cv(
            "logit", _fresh_logistic, X, y,
            config=RepeatedCVConfig(n_folds=10, n_repeats=1, seed=1),
        )


def test_misaligned_X_and_y_is_rejected():
    X, _ = _synthetic_frame(n=100, seed=53)
    with pytest.raises(RepeatedCVDataError, match="align"):
        run_repeated_cv(
            "logit", _fresh_logistic, X, np.ones(90, dtype=int),
            config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1),
        )


def test_empty_candidates_is_rejected():
    X, y = _synthetic_frame(n=90, seed=59)
    with pytest.raises(EmptyCandidateError, match="empty"):
        run_repeated_cv_comparison({}, X, y, config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1))
    with pytest.raises(EmptyCandidateError, match="Mapping"):
        run_repeated_cv_comparison(["logit"], X, y, config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1))


def test_broken_factory_raises_named_fold_fit_error_with_cause():
    X, y = _synthetic_frame(n=100, seed=61)

    class Broken:
        def fit(self, *_args, **_kwargs):
            raise RuntimeError("no dice")

    def broken_factory():
        return Broken()

    with pytest.raises(FoldFitError, match="failed to build or fit") as info:
        run_repeated_cv(
            "broken", broken_factory, X, y,
            config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1),
        )
    assert isinstance(info.value.__cause__, RuntimeError)
    with pytest.raises(FoldFitError, match="not callable"):
        run_repeated_cv(
            "notcallable", 1234, X, y,
            config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=1),
        )


def test_candidate_not_found_error_is_exported_in_hierarchy():
    assert issubclass(CandidateNotFoundError, RepeatedCVError)


# ---------------------------------------------------------------------------
# 6. Integration: real pipeline over the committed S01 split
# ---------------------------------------------------------------------------


def test_integration_registry_pipeline_over_split_frames():
    train, _test = load_split_frames(SPLIT_VERSION)
    # Two structurally different candidates with declared preprocessing.
    def logistic_candidate():
        return Pipeline(
            steps=[
                ("preprocess", build_preprocessing_pipeline()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

    def nb_candidate():
        return Pipeline(
            steps=[
                ("preprocess", build_preprocessing_pipeline()),
                ("clf", GaussianNB()),
            ]
        )

    X = train[list(FEATURE_COLUMNS)]
    y = train["HeartDisease"].to_numpy()
    results = run_repeated_cv_comparison(
        {"logit": logistic_candidate, "nb": nb_candidate},
        X,
        y,
        config=RepeatedCVConfig(n_folds=4, n_repeats=2, seed=42),
    )
    ranked = rank_candidates(results)
    ranked_names = [name for name, _, _ in ranked]
    assert all(0.5 <= mean <= 1.0 for _name, mean, _std in ranked)


def test_candidate_not_found_semantics():
    """The documented CandidateNotFoundError exists for missing keys (API surface)."""
    # Kept as a thin API-surface pin because S06/T02 consumes it via name.
    assert rcv.CandidateNotFoundError is CandidateNotFoundError
