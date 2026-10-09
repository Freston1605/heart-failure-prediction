"""Tests for src/heart/eval/selection.py (S06/T03).

All fixtures are inline/synthetic. The repeated-CV results underpinning the
selection tests are produced through the real
``heart.eval.repeated_cv.run_repeated_cv_comparison`` on a small synthetic
frame, so selection consumes the actual upstream artifacts the slice demands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from heart.eval.repeated_cv import (
    PRIMARY_METRIC,
    RepeatedCVConfig,
    rank_candidates,
    run_repeated_cv_comparison,
)
from heart.eval.selection import (
    DEFAULT_OBJECTIVE,
    SERVING_WEIGHT_BUDGET_MB,
    SelectionConfigError,
    SelectionDataError,
    SelectionDecision,
    ServingWeightCheck,
    ThresholdTuning,
    select_model,
    selection_report,
    tune_threshold,
    write_selection_report,
)

_ROWS = 260
_WEIGHTS = {"logistic_baseline": 0.4, "random_forest": 2.0}


@pytest.fixture(scope="module")
def synthetic_classical():
    """Synthetic frame + two cheap candidate factories + fitted winners."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(_ROWS, 3))
    x[:, 0] = StandardScaler().fit_transform(x[:, [0]]).ravel()
    # Deliberately nonlinear (interaction term) so the forest must out-learn
    # the linear baseline — the ship gate then has a real winner to check.
    score = 1.6 * x[:, 0] * x[:, 1] + 0.4 * x[:, 2]
    y = (score + rng.normal(scale=0.7, size=_ROWS) > 0.5).astype(int)
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    labels = pd.Series(y)

    candidates = {
        "logistic_baseline": lambda: LogisticRegression(max_iter=200),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=25, random_state=42
        ),
    }
    config = RepeatedCVConfig(n_folds=4, n_repeats=2, seed=42)
    results = run_repeated_cv_comparison(candidates, frame, labels, config=config)

    x_rest, x_val, y_rest, y_val = train_test_split(
        frame, labels, test_size=0.25, random_state=42, stratify=labels
    )
    x_train, x_calib, y_train, y_calib = train_test_split(
        x_rest, y_rest, test_size=0.25, random_state=42, stratify=y_rest
    )
    return {
        "results": results,
        "frame": frame,
        "labels": labels,
        "x_train": x_train,
        "y_train": y_train,
        "x_calib": x_calib,
        "y_calib": y_calib,
        "x_val": x_val,
        "y_val": y_val,
        "fitted_logistic": LogisticRegression(max_iter=200).fit(x_train, y_train),
        "fitted_forest": RandomForestClassifier(
            n_estimators=25, random_state=42
        ).fit(x_train, y_train),
    }


@pytest.fixture(scope="module")
def selection_decision(synthetic_classical):
    fx = synthetic_classical
    decision = select_model(
        fx["results"],
        fitted_winner=fx["fitted_forest"],
        X_calib=fx["x_calib"],
        y_calib=fx["y_calib"],
        X_val=fx["x_val"],
        y_val=fx["y_val"],
        serving_weights=_WEIGHTS,
        objective="f1",
    )
    return fx, decision


# ---------------------------------------------------------------------------
# tune_threshold
# ---------------------------------------------------------------------------


def test_tune_threshold_argmax_matches_declared_objective_on_validation(
    synthetic_classical,
) -> None:
    """The chosen threshold must be the argmax of the objective on y_val."""
    fx = synthetic_classical
    winner = rank_candidates(fx["results"])[0][0]
    fitted = (
        fx["fitted_forest"] if winner == "random_forest" else fx["fitted_logistic"]
    )
    y_proba_val = fitted.predict_proba(fx["x_val"])[:, 1]
    tuning = tune_threshold(fx["y_val"], y_proba_val, objective="f1")
    scores = np.asarray(tuning.scores)
    assert tuning.best_threshold == tuning.thresholds[int(np.argmax(scores))]
    assert tuning.best_score == pytest.approx(scores.max())

    # Independent recomputation of the declared objective at the chosen point.
    y_bin = np.asarray(fx["y_val"], dtype=int)
    predictions = (y_proba_val >= tuning.best_threshold).astype(int)
    tp = int(((predictions == 1) & (y_bin == 1)).sum())
    fp = int(((predictions == 1) & (y_bin == 0)).sum())
    fn = int(((predictions == 0) & (y_bin == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    assert tuning.best_score == pytest.approx(f1)


def test_tune_threshold_returns_full_sweep_and_default_objective(
    synthetic_classical,
) -> None:
    fx = synthetic_classical
    y_proba_val = fx["fitted_forest"].predict_proba(fx["x_val"])[:, 1]
    tuning = tune_threshold(fx["y_val"], y_proba_val)
    assert tuning.objective == DEFAULT_OBJECTIVE == "f1"
    assert tuning.thresholds[0] == pytest.approx(0.01)
    assert tuning.thresholds[-1] < 1.0
    assert len(tuning.thresholds) == len(tuning.scores)
    assert 0.0 < tuning.best_threshold < 1.0
    assert tuning.to_dict()["objective"] == "f1"


@pytest.mark.parametrize("objective", ["balanced_accuracy", "youden_j"])
def test_tune_threshold_respects_declared_objective(
    synthetic_classical, objective: str
) -> None:
    fx = synthetic_classical
    y_proba_val = fx["fitted_forest"].predict_proba(fx["x_val"])[:, 1]
    tuning = tune_threshold(fx["y_val"], y_proba_val, objective=objective)
    y_bin = np.asarray(fx["y_val"], dtype=int)
    predictions = (y_proba_val >= tuning.best_threshold).astype(int)
    tp = int(((predictions == 1) & (y_bin == 1)).sum())
    fp = int(((predictions == 1) & (y_bin == 0)).sum())
    fn = int(((predictions == 0) & (y_bin == 1)).sum())
    tn = int(((predictions == 0) & (y_bin == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    expected = (
        0.5 * (recall + specificity)
        if objective == "balanced_accuracy"
        else recall - (1.0 - specificity)
    )
    assert tuning.best_score == pytest.approx(expected)


def test_tune_threshold_rejects_unknown_objective(synthetic_classical) -> None:
    fx = synthetic_classical
    y_proba_val = fx["fitted_forest"].predict_proba(fx["x_val"])[:, 1]
    with pytest.raises(SelectionConfigError, match="Unknown objective"):
        tune_threshold(fx["y_val"], y_proba_val, objective="mcc")


def test_tune_threshold_rejects_misaligned_lengths(synthetic_classical) -> None:
    fx = synthetic_classical
    y_proba_val = fx["fitted_forest"].predict_proba(fx["x_val"])[:, 1]
    with pytest.raises(SelectionDataError, match="do not align"):
        tune_threshold(fx["y_val"], y_proba_val[:-1])


def test_tune_threshold_rejects_non_finite_and_out_of_range(
    synthetic_classical,
) -> None:
    fx = synthetic_classical
    y_proba_val = fx["fitted_forest"].predict_proba(fx["x_val"])[:, 1]
    nan_proba = y_proba_val.copy()
    nan_proba[0] = np.nan
    with pytest.raises(SelectionDataError, match="non-finite"):
        tune_threshold(fx["y_val"], nan_proba)
    bad = y_proba_val.copy()
    bad[0] = 1.7
    with pytest.raises(SelectionDataError, match="not probabilities"):
        tune_threshold(fx["y_val"], bad)


def test_tune_threshold_rejects_empty_and_bad_step(synthetic_classical) -> None:
    fx = synthetic_classical
    with pytest.raises(SelectionDataError, match="at least one validation row"):
        tune_threshold(np.array([], dtype=int), np.array([], dtype=float))
    y_proba_val = fx["fitted_forest"].predict_proba(fx["x_val"])[:, 1]
    with pytest.raises(SelectionConfigError, match="positive"):
        tune_threshold(fx["y_val"], y_proba_val, step=0.0)


# ---------------------------------------------------------------------------
# Serving weight
# ---------------------------------------------------------------------------


def test_serving_weight_check_boundary() -> None:
    check = ServingWeightCheck(model_name="m", weight_mb=SERVING_WEIGHT_BUDGET_MB)
    assert check.shippable
    over = ServingWeightCheck(
        model_name="m", weight_mb=SERVING_WEIGHT_BUDGET_MB + 0.1
    )
    assert not over.shippable


# ---------------------------------------------------------------------------
# select_model
# ---------------------------------------------------------------------------


def test_select_model_names_winner_and_baseline(synthetic_classical) -> None:
    """Winner must come from rank_candidates; baseline resolved by name."""
    fx = synthetic_classical
    winner = rank_candidates(fx["results"])[0][0]
    decision = select_model(
        fx["results"],
        fitted_winner=fx["fitted_forest"],
        X_calib=fx["x_calib"],
        y_calib=fx["y_calib"],
        X_val=fx["x_val"],
        y_val=fx["y_val"],
        serving_weights=_WEIGHTS,
    )
    assert isinstance(decision, SelectionDecision)
    assert decision.winner == winner
    assert decision.baseline == "logistic_baseline"
    assert decision.metric == PRIMARY_METRIC


def test_select_model_ship_when_gates_green(selection_decision) -> None:
    _fx, decision = selection_decision
    assert decision.shippable_by_weight
    assert decision.ship is True
    # Calibration on the tiny synthetic split can legitimately worsen the
    # Brier score; that is a flag, not a ship gate. Ship gates are
    # significance + serving weight, and both must be green here.
    assert "serving_weight_exceeded" not in "".join(decision.flags)
    assert all("not_significantly_better" not in f for f in decision.flags)


def test_select_model_over_budget_winner_flagged_not_swapped(
    synthetic_classical,
) -> None:
    """Over-budget winner is flagged NO-SHIP, never silently swapped."""
    fx = synthetic_classical
    decision = select_model(
        fx["results"],
        fitted_winner=fx["fitted_forest"],
        X_calib=fx["x_calib"],
        y_calib=fx["y_calib"],
        X_val=fx["x_val"],
        y_val=fx["y_val"],
        serving_weights={"logistic_baseline": 0.4, "random_forest": 400.0},
    )
    assert decision.winner == "random_forest"  # not swapped
    assert not decision.ship
    assert not decision.shippable_by_weight
    assert "serving_weight_exceeded" in "".join(decision.flags)


def test_selection_decision_threshold_uses_validation_only(
    selection_decision,
) -> None:
    _fx, decision = selection_decision
    assert isinstance(decision.threshold, ThresholdTuning)
    assert 0.0 < decision.threshold.best_threshold < 1.0
    assert decision.calibration is not None
    assert decision.calibration.report_calibrated.n_bins == 10


def test_select_model_requires_winner_in_serving_weights(
    synthetic_classical,
) -> None:
    fx = synthetic_classical
    with pytest.raises(SelectionDataError, match="serving_weights lacks"):
        select_model(
            fx["results"],
            fitted_winner=fx["fitted_forest"],
            X_calib=fx["x_calib"],
            y_calib=fx["y_calib"],
            X_val=fx["x_val"],
            y_val=fx["y_val"],
            serving_weights={"logistic_baseline": 0.4},
        )


def test_select_model_rejects_empty_results() -> None:
    with pytest.raises(
        SelectionDataError, match="at least one repeated-CV result"
    ):
        select_model(
            {},
            fitted_winner=None,
            X_calib=None,
            y_calib=None,
            X_val=None,
            y_val=None,
            serving_weights={},
        )


def test_select_model_rejects_unknown_baseline(synthetic_classical) -> None:
    fx = synthetic_classical
    with pytest.raises(SelectionDataError, match="not among candidates"):
        select_model(
            fx["results"],
            fitted_winner=fx["fitted_forest"],
            X_calib=fx["x_calib"],
            y_calib=fx["y_calib"],
            X_val=fx["x_val"],
            y_val=fx["y_val"],
            serving_weights=_WEIGHTS,
            baseline="ghost_model",
        )


# ---------------------------------------------------------------------------
# Report rendering + writing
# ---------------------------------------------------------------------------


def test_selection_report_content_and_shape(selection_decision) -> None:
    _fx, decision = selection_decision
    report = selection_report(decision)
    assert f"Winner: **{decision.winner}**" in report
    assert "## Ranking" in report
    assert "## Paired significance" in report
    assert "Serving weight:" in report
    assert "Calibration" in report
    assert f"Chosen threshold: **{decision.threshold.best_threshold:.2f}**" in report
    assert f"Verdict: {'SHIP' if decision.ship else 'NO-SHIP'}" in report
    assert "Flags:" in report


def test_write_selection_report(selection_decision, tmp_path) -> None:
    _fx, decision = selection_decision
    target = tmp_path / "reports" / "selection.md"
    written = write_selection_report(decision, path=target)
    assert written == target
    contents = target.read_text(encoding="utf-8")
    assert "Winner:" in contents
    assert "Verdict:" in contents
