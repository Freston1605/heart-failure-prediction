"""Tests for paired significance testing (S06/T02).

The contract pinned here:

1. **Known-outcome synthetic cases** — identical models yield a
   non-significant result on both tests; a clearly superior model yields a
   significant one. The p-values must land on the declared side of alpha.
2. **McNemar's test** — contingency cells are tabulated on the *same* rows;
   the exact binomial p-value is reproduced against hand-computed reference
   cases; malformed alignment is rejected, never zipped silently.
3. **Corrected resampled t-test** — the Nadeau & Bengio overfit correction is
   applied as ``(1/n_folds + n_test/n_train)``; the corrected p-value is at
   least as large as the naive paired t-test p-value (folds overlap, so the
   correction can only make significance harder).
4. **Paired wiring to repeated CV** — comparisons run against
   :func:`heart.eval.repeated_cv.run_repeated_cv_comparison` results, where
   candidates sharing a config share fold seeds; misaligned configs or
   unknown names raise named errors.
5. **Report surface** — ``significance_report`` renders the winner-versus-
   field table and returns the winner-vs-baseline comparison.

Fixtures are synthetic and self-contained; one integration test runs the
registry-backed pipeline over the committed S01 split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from heart.data.pipeline import build_preprocessing_pipeline
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, load_split_frames
from heart.eval.repeated_cv import (
    RepeatedCVConfig,
    RepeatedCVDataError,
    RepeatedCVResult,
    run_repeated_cv_comparison,
)
from heart.eval.significance import (
    DEFAULT_ALPHA,
    CorrectedTResult,
    McNemarResult,
    PairedAlignmentError,
    PairwiseComparison,
    SignificanceConfigError,
    SignificanceDataError,
    compare_repeated_cv,
    corrected_resampled_t_test,
    mcnemar_test,
    significance_report,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _synthetic_frame(n: int = 300, *, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """A small self-contained binary problem with real (not trivial) signal."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, 3))
    logit = 1.7 * matrix[:, 0] - 1.2 * matrix[:, 1] + 0.3 * matrix[:, 2]
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(matrix, columns=["f0", "f1", "f2"])
    return X, labels


def _fresh_logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _fresh_chance_model() -> DummyClassifier:
    """A model that ignores features entirely (expected ROC-AUC near 0.5)."""
    return DummyClassifier(strategy="stratified")


def _paired_vectors(
    n: int = 200, *, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A y_true pair plus two prediction vectors differing on known rows."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    pred_a = y.copy()
    pred_b = y.copy()
    # b = 2 rows where A is right and B is wrong (A mirrors y, B flipped).
    b_rows = np.flatnonzero(y == 0)[:2]
    pred_b[b_rows] = 1
    # c = 1 row where A is wrong and B is right (A flipped, B mirrors y).
    c_row = np.flatnonzero(y == 1)[:1]
    pred_a[c_row] = 0
    pred_b[c_row] = 1
    return y, pred_a, pred_b


# ---------------------------------------------------------------------------
# 1. McNemar's test — known outcomes
# ---------------------------------------------------------------------------


class TestMcNemarKnownOutcomes:
    def test_identical_predictions_are_not_significant(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
        pred_a = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        result = mcnemar_test(y, pred_a, pred_a.copy())
        assert result.b == 0
        assert result.c == 0
        assert result.p_value == pytest.approx(1.0)
        assert result.significant is False
        assert result.n_agree == y.size
        assert "not significant" in result.verdict()

    def test_symmetric_discordance_is_not_significant(self):
        """b == c -> two-sided exact p is exactly 1.0."""
        y = np.zeros(14, dtype=int)
        pred_a = y.copy()
        pred_b = y.copy()
        # 7 rows where A is right and B wrong; 7 rows where A wrong/B right.
        pred_b[:7] = 1  # y=0, A correct (pred 0), B wrong (pred 1) -> b
        y[7:] = 1
        pred_a[7:] = 0  # y=1, A wrong (pred 0) -> c
        pred_b[7:] = 1  # y=1, B correct (pred 1) -> c
        result = mcnemar_test(y, pred_a, pred_b)
        assert result.b == 7
        assert result.c == 7
        expected = float(stats.binomtest(7, 14, p=0.5).pvalue)
        assert result.p_value == pytest.approx(expected)
        assert result.p_value == pytest.approx(1.0, abs=1e-12)
        assert result.significant is False

    def test_asymmetric_discordance_is_significant(self):
        """b=15, c=0 -> p = 2 * 0.5**15 ~ 6.1e-5 < alpha."""
        y = np.zeros(15 + 5, dtype=int)
        pred_a = y.copy()
        pred_b = y.copy()
        pred_b[:15] = 1  # b = 15 discordant rows, all favouring A
        result = mcnemar_test(y, pred_a, pred_b)
        assert result.b == 15
        assert result.c == 0
        assert result.statistic == 0
        assert result.p_value == pytest.approx(float(stats.binomtest(0, 15, p=0.5).pvalue))
        assert result.p_value < 0.01
        assert result.n_wrong == 15
        assert result.n_agree == 5
        assert result.significant is True
        assert "significant" in result.verdict()
        assert "not" not in result.verdict()

    def test_small_asymmetry_is_not_significant(self):
        """b=2, c=1 -> p = 1.0 > alpha; powers-of-two exactness."""
        y, pred_a, pred_b = _paired_vectors()
        result = mcnemar_test(y, pred_a, pred_b)
        assert (result.b, result.c) == (2, 1)
        # exact two-sided p for (1 of 3) is 1.0
        assert result.p_value == pytest.approx(1.0)
        assert result.significant is False


class TestMcNemarValidation:
    def test_length_mismatch_raises_alignment_error(self):
        y = np.array([0, 1])
        pred_a = np.array([0, 1, 0])
        pred_b = np.array([0, 1])
        with pytest.raises(PairedAlignmentError) as excinfo:
            mcnemar_test(y, pred_a, pred_b)
        assert "same" in str(excinfo.value)
        assert "fabricate" in str(excinfo.value)

    def test_empty_vectors_raise_data_error(self):
        with pytest.raises(SignificanceDataError):
            mcnemar_test([], [], [])

    def test_multiclass_labels_raise_data_error(self):
        y = np.array([0, 1, 2])
        pred_a = np.array([0, 1, 2])
        pred_b = np.array([0, 1, 2])
        with pytest.raises(SignificanceDataError) as excinfo:
            mcnemar_test(y, pred_a, pred_b)
        assert "binary 0/1" in str(excinfo.value)

    def test_invalid_alpha_raises_config_error(self):
        y = np.array([0, 1])
        pred_a = np.array([0, 1])
        for bad in (-0.1, 0.0, 1.0, 1.5):
            with pytest.raises(SignificanceConfigError):
                mcnemar_test(y, pred_a, pred_a, alpha=bad)
        with pytest.raises(SignificanceConfigError):
            mcnemar_test(y, pred_a, pred_a, alpha="0.05")

    def test_to_dict_carries_named_fields(self):
        y, pred_a, pred_b = _paired_vectors()
        result = mcnemar_test(y, pred_a, pred_b)
        payload = result.to_dict()
        assert payload["test"] == "mcnemar_exact"
        assert payload["b"] == 2
        assert payload["c"] == 1
        assert payload["significant"] is False
        assert set(payload) == {
            "test", "b", "c", "n_samples", "n_agree", "statistic",
            "p_value", "alpha", "significant",
        }


# ---------------------------------------------------------------------------
# 2. Corrected resampled t-test — known outcomes and the correction
# ---------------------------------------------------------------------------


class TestCorrectedTKnownOutcomes:
    def test_identical_models_yield_non_significant(self):
        diffs = np.zeros(25)
        result = corrected_resampled_t_test(diffs, n_folds=5, n_test=60, n_train=240)
        assert result.p_value == pytest.approx(1.0)
        assert result.significant is False
        assert result.t_statistic == pytest.approx(0.0)
        assert result.df == 24
        assert "not significant" in result.verdict()

    def test_clearly_superior_model_yields_significant(self):
        rng = np.random.default_rng(11)
        diffs = rng.normal(loc=0.08, scale=0.01, size=25)
        result = corrected_resampled_t_test(diffs, n_folds=5, n_test=60, n_train=240)
        assert result.significant is True
        assert result.p_value < 0.001
        assert result.mean_diff > 0
        assert result.overfit_correction == pytest.approx(1 / 5 + 60 / 240)

    def test_noise_differences_are_not_significant(self):
        rng = np.random.default_rng(23)
        diffs = rng.normal(loc=0.0, scale=0.04, size=25)
        result = corrected_resampled_t_test(diffs, n_folds=5, n_test=60, n_train=240)
        assert result.significant is False
        assert result.p_value >= 0.05

    def test_correction_matches_closed_form_and_inflates_p(self):
        """Corrected p >= naive p: the correction penalises fold overlap."""
        rng = np.random.default_rng(31)
        diffs = rng.normal(loc=0.05, scale=0.02, size=25)
        corrected = corrected_resampled_t_test(
            diffs, n_folds=5, n_test=60, n_train=240
        )
        expected_correction = 1.0 / 5 + 60.0 / 240.0
        assert corrected.overfit_correction == pytest.approx(expected_correction)
        naive_t = float(diffs.mean()) / (float(diffs.std(ddof=1)) / np.sqrt(25))
        naive_p = float(2 * stats.t.sf(abs(naive_t), df=24))
        assert corrected.p_value >= naive_p - 1e-12


class TestCorrectedTValidation:
    def test_n_folds_below_two_raises_config_error(self):
        diffs = np.zeros(10)
        with pytest.raises(SignificanceConfigError):
            corrected_resampled_t_test(diffs, n_folds=1, n_test=60, n_train=240)

    def test_non_positive_row_totals_raise_config_error(self):
        diffs = np.zeros(10)
        with pytest.raises(SignificanceConfigError):
            corrected_resampled_t_test(diffs, n_folds=5, n_test=0, n_train=240)
        with pytest.raises(SignificanceConfigError):
            corrected_resampled_t_test(diffs, n_folds=5, n_test=60, n_train=-1)

    def test_alpha_or_type_errors(self):
        diffs = np.array([0.1, 0.2])
        with pytest.raises(SignificanceConfigError):
            corrected_resampled_t_test(diffs, n_folds=5, n_test=1, n_train=1, alpha=0)
        with pytest.raises(SignificanceDataError):
            corrected_resampled_t_test([0.1, 0.2, np.nan], n_folds=5, n_test=1,
                                       n_train=1)
        with pytest.raises(SignificanceDataError):
            corrected_resampled_t_test([0.1], n_folds=5, n_test=1, n_train=1)
        with pytest.raises(SignificanceDataError):
            corrected_resampled_t_test([[0.1, 0.2], [0.3, 0.4]], n_folds=5,
                                       n_test=1, n_train=1)

    def test_non_finite_diffs_rejected(self):
        with pytest.raises(SignificanceDataError):
            corrected_resampled_t_test([0.1, np.inf], n_folds=5, n_test=1, n_train=1)


# ---------------------------------------------------------------------------
# 3. Paired wiring to repeated-CV results
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_repeated_cv():
    """Two candidates on identical folds: informed model vs a feature-blind one."""
    X, y = _synthetic_frame()
    config = RepeatedCVConfig(n_folds=4, n_repeats=2, seed=1)
    results = run_repeated_cv_comparison(
        {"logistic_informed": _fresh_logistic, "chance_model": _fresh_chance_model},
        X,
        y,
        config=config,
    )
    return results, config, X, y


class TestCompareRepeatedCv:
    def test_identical_models_are_not_significant(self):
        """Two independent fits of the same spec must agree fold-for-fold."""
        X, y = _synthetic_frame(seed=99)
        config = RepeatedCVConfig(n_folds=4, n_repeats=2, seed=5)
        results = run_repeated_cv_comparison(
            {"model_a": _fresh_logistic, "model_b": _fresh_logistic},
            X,
            y,
            config=config,
        )
        comparison = compare_repeated_cv(results, "model_a", "model_b")
        assert comparison.corrected_t.mean_diff == pytest.approx(0.0)
        assert comparison.corrected_t.significant is False
        assert comparison.corrected_t.p_value == pytest.approx(1.0)
        assert "noise" in comparison.verdict()

    def test_superior_model_is_significant_over_chance_model(
        self, synthetic_repeated_cv
    ):
        results, _config, _X, _y = synthetic_repeated_cv
        comparison = compare_repeated_cv(results, "logistic_informed", "chance_model")
        assert comparison.significant is True
        assert comparison.corrected_t.p_value < DEFAULT_ALPHA
        assert comparison.corrected_t.mean_diff > 0.3
        assert "real" in comparison.verdict()

    def test_mcnemar_optional_hooks_receiver_predicts(self, synthetic_repeated_cv):
        results, _config, X, y = synthetic_repeated_cv
        informed = _fresh_logistic().fit(X, y)
        chance = _fresh_chance_model().fit(X, y)
        comparison = compare_repeated_cv(
            results,
            "logistic_informed",
            "chance_model",
            mcnemar_scores={
                "logistic_informed": informed.predict(X),
                "chance_model": chance.predict(X),
            },
            mcnemar_y_true=y,
        )
        assert comparison.agreement is not None
        assert isinstance(comparison.agreement, McNemarResult)
        # The informed model disagrees with a stray classifier overwhelmingly.
        assert comparison.agreement.n_wrong > 0
        assert isinstance(comparison.corrected_t, CorrectedTResult)

    def test_mcnemar_without_y_true_raises(self, synthetic_repeated_cv):
        results, _config, X, _y = synthetic_repeated_cv
        informed = _fresh_logistic().fit(X, _y)
        with pytest.raises(SignificanceDataError):
            compare_repeated_cv(
                results,
                "logistic_informed",
                "chance_model",
                mcnemar_scores={"logistic_informed": informed.predict(X)},
            )

    def test_mcnemar_scores_missing_model_raises(self, synthetic_repeated_cv):
        results, _config, X, _y = synthetic_repeated_cv
        informed = _fresh_logistic().fit(X, _y)
        with pytest.raises(SignificanceDataError) as excinfo:
            compare_repeated_cv(
                results,
                "logistic_informed",
                "chance_model",
                mcnemar_scores={"logistic_informed": informed.predict(X)},
                mcnemar_y_true=_y,
            )
        assert "chance_model" in str(excinfo.value)

    def test_unknown_candidate_raises_named_error(self, synthetic_repeated_cv):
        results, _config, _X, _y = synthetic_repeated_cv
        with pytest.raises(PairedAlignmentError):
            compare_repeated_cv(results, "ghost", "chance_model")

    def test_misaligned_configs_raise_alignment_error(self):
        X, y = _synthetic_frame(seed=13)
        results_a = run_repeated_cv_comparison(
            {"model_a": _fresh_logistic},
            X,
            y,
            config=RepeatedCVConfig(n_folds=4, n_repeats=2, seed=5),
        )
        results_b = run_repeated_cv_comparison(
            {"model_b": _fresh_logistic},
            X,
            y,
            config=RepeatedCVConfig(n_folds=5, n_repeats=2, seed=5),
        )
        merged = {**results_a, **results_b}
        with pytest.raises(PairedAlignmentError) as excinfo:
            compare_repeated_cv(merged, "model_a", "model_b")
        assert "fold" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Significance report
# ---------------------------------------------------------------------------


class TestSignificanceReport:
    def test_report_renders_table_and_returns_pairwise(self, synthetic_repeated_cv):
        results, config, _X, _y = synthetic_repeated_cv
        report, winner_vs_baseline = significance_report(
            results, baseline="chance_model"
        )
        assert "## Paired significance testing" in report
        assert "logistic_informed" in report
        assert f"alpha={DEFAULT_ALPHA:g}" in report
        assert config.fingerprint() in report
        assert isinstance(winner_vs_baseline, PairwiseComparison)
        assert {winner_vs_baseline.model_a, winner_vs_baseline.model_b} == {
            "logistic_informed",
            "chance_model",
        }
        # winner should be the informed model on real signal.
        assert winner_vs_baseline.model_a == "logistic_informed"

    def test_report_baseline_default_prefers_logistic_name(self, synthetic_repeated_cv):
        results, _config, _X, _y = synthetic_repeated_cv
        report, winner_vs_baseline = significance_report(results)
        # the logistic-named candidate is the evaluated baseline, but it is also
        # the winner here, so the returned pair is None only when the baseline
        # matches the winner.
        if winner_vs_baseline is None:
            assert "Baseline: **logistic_informed**." in report
        else:
            assert winner_vs_baseline.model_b == "logistic_informed"

    def test_report_unknown_baseline_raises(self, synthetic_repeated_cv):
        results, _config, _X, _y = synthetic_repeated_cv
        with pytest.raises(PairedAlignmentError):
            significance_report(results, baseline="not_a_model")

    def test_report_empty_results_raises(self):
        with pytest.raises(RepeatedCVDataError):
            significance_report({})

    def test_report_single_candidate_has_no_pairwise(self):
        X, y = _synthetic_frame(seed=5)
        results = run_repeated_cv_comparison(
            {"logistic_baseline": _fresh_logistic},
            X,
            y,
            config=RepeatedCVConfig(n_folds=3, n_repeats=1, seed=2),
        )
        report, winner_vs_baseline = significance_report(results)
        assert winner_vs_baseline is None
        assert "logistic_baseline" in report


# ---------------------------------------------------------------------------
# 5. Integration with the committed S01 split
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_registry_pipeline_over_committed_split(self):
        """Real preprocessing path: the committed split's columns are categorical,
        so candidates must go through ``build_preprocessing_pipeline`` (the same
        contract the leaderboard uses)."""
        train, _test = load_split_frames(SPLIT_VERSION)

        def logistic_candidate():
            return Pipeline(
                steps=[
                    ("preprocess", build_preprocessing_pipeline()),
                    ("clf", LogisticRegression(max_iter=1000)),
                ]
            )

        def dumb_candidate():
            return Pipeline(
                steps=[
                    ("preprocess", build_preprocessing_pipeline()),
                    ("clf", DummyClassifier(strategy="uniform", random_state=0)),
                ]
            )

        X = train[list(FEATURE_COLUMNS)]
        y = train[TARGET_COLUMN].to_numpy()
        config = RepeatedCVConfig(n_folds=4, n_repeats=2, seed=1)
        results = run_repeated_cv_comparison(
            {"logistic_baseline": logistic_candidate, "dumb_prior": dumb_candidate},
            X,
            y,
            config=config,
        )
        comparison = compare_repeated_cv(
            results, "logistic_baseline", "dumb_prior"
        )
        assert np.isfinite(comparison.corrected_t.p_value)
        assert "advantage" in comparison.verdict()
        report, winner = significance_report(results, baseline="dumb_prior")
        assert "Winner: **" in report
        assert winner is not None
        assert winner.model_a in ("logistic_baseline", "dumb_prior")
        # The real metrics echo the shared contract (primary = ROC-AUC).
        from heart.eval.repeated_cv import PRIMARY_METRIC

        assert winner.metric == PRIMARY_METRIC
