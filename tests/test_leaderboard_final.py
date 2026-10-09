"""Final leaderboard regeneration with selection annotations (S06/T05).

The published table must carry the statistical evidence behind the winner
rather than ranking alone. This suite pins the whole annotation loop:

1. **Full metric suite** — the final leaderboard still reports every scalar
   leaf of the canonical metric suite, plus the confusion and calibration
   expansions (the S03 contract, re-asserted on the annotated report).
2. **Device columns** — neural runs render their recorded ``device`` tag and
   classical runs render ``-`` (the generic S05 device column).
3. **Selection annotations** — the winner run's ``s6.*`` MLflow tags are read
   back into :class:`SelectionAnnotations` and rendered as the "Winner
   selection (S06)" section: repeated-CV distribution, paired significance
   vs the baseline, the calibrated threshold and its score, the Brier
   movement, the serving weight, named flags, and the SHIP/NO-SHIP verdict.
4. **Regenerable from MLflow data alone** — the annotations live in the
   tracking store as tags; a fresh build reconstructs the identical report
   from the store alone.
5. **Negative surface** — two annotated rows (ambiguous winner), a partial
   annotation set, a winner tag pointing at the wrong run, an invalid ship
   value, an unknown annotation field, a non-finite annotation number, and a
   flag-list overflow all raise named errors instead of rendering a
   half-story.

Every store-backed test runs against a fresh ``tmp_path`` SQLite database, so
the suite never touches the repository's ``experiments/mlruns`` directory.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pytest

from heart.eval.calibration import (
    DEFAULT_CALIBRATION_BINS,
    calibration_comparison,
)
from heart.eval.contract import (
    PRIMARY_METRIC,
    compute_metric_dict,
)
from heart.eval.selection import (
    DEFAULT_ALPHA,
    DEFAULT_OBJECTIVE,
    ServingWeightCheck,
    SelectionDecision,
    ThresholdTuning,
)
from heart.eval.significance import (
    CorrectedTResult,
    McNemarResult,
    PairwiseComparison,
)
from heart.reporting.leaderboard import (
    DEFAULT_RUN_KIND,
    MAX_TAG_VALUE_LENGTH,
    S06LeaderboardDataError,
    SELECTION_FLAGS_SEPARATOR,
    SELECTION_TAG_PREFIX,
    LeaderboardConfig,
    SelectionAnnotationsError,
    annotate_winner_run,
    build_leaderboard,
    read_selection_annotations,
    render_leaderboard,
    selection_annotation_tags,
    selection_tags,
)
from heart.tracking.mlflow_store import configure_tracking
from heart.tracking.run import log_evaluation_run
from heart.tuning.runner import FINAL_RUN_KIND, MODEL_TYPE_TAG, RUN_KIND_TAG

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Store:
    """A freshly configured local MLflow store for one test."""

    def __init__(self, tracking_dir, config):
        self.tracking_dir = tracking_dir
        self.config = config

    @property
    def experiment_name(self) -> str:
        return self.config.experiment_name


@pytest.fixture
def store(tmp_path) -> _Store:
    tracking_dir = tmp_path / "mlruns"
    config = configure_tracking(tracking_dir=tracking_dir)
    mlflow.set_tracking_uri(config.tracking_uri)
    return _Store(tracking_dir=tracking_dir, config=config)


def _sample_metrics(*, roc_auc: float | None = None) -> dict:
    """A complete, schema-valid metric dict built without fitting a model."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 4)
    probabilities = np.clip(0.18 + 0.64 * labels, 0.0, 1.0)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = compute_metric_dict(labels, probabilities, predictions, n_bins=5)
    if roc_auc is not None:
        metrics[PRIMARY_METRIC] = float(roc_auc)
    return metrics


def _log_run(
    store: _Store,
    *,
    model_type: str,
    model_name: str,
    family: str = "test",
    split_version: str = "v1",
    roc_auc: float | None = None,
    tags: dict | None = None,
) -> str:
    """Log one final benchmark run (the battery's tracking convention)."""
    mlflow.set_tracking_uri(store.config.tracking_uri)
    run = log_evaluation_run(
        _sample_metrics(roc_auc=roc_auc),
        model_name=model_name,
        split_version=split_version,
        params={"C": 1.0},
        tags={
            RUN_KIND_TAG: FINAL_RUN_KIND,
            MODEL_TYPE_TAG: model_type,
            "family": family,
            **(tags or {}),
        },
        config=store.config,
    )
    return run.run_id


def _comparison(*, p_value: float = 0.3089, significant: bool = False):
    """A winner-vs-baseline PairwiseComparison with realistic S06 numbers."""
    corrected = CorrectedTResult(
        mean_diff=0.008603,
        std_diff=0.014002,
        t_statistic=1.860699,
        df=14,
        n_samples_paired=15,
        overfit_correction=0.392144,
        p_value=p_value,
        alpha=DEFAULT_ALPHA,
        significant=significant,
    )
    agreement = McNemarResult(
        b=2,
        c=1,
        n_samples=147,
        n_agree=144,
        statistic=1,
        p_value=1.0,
        alpha=DEFAULT_ALPHA,
        significant=False,
    )
    return PairwiseComparison(
        model_a="random-forest",
        model_b="logistic-regression-l2",
        metric=PRIMARY_METRIC,
        agreement=agreement,
        corrected_t=corrected,
        alpha=DEFAULT_ALPHA,
    )


def _decision(
    *,
    p_value: float = 0.3089,
    significant: bool = False,
    proven: bool = False,
    weight_ok: bool = True,
    brier_delta: float = -0.005274,
) -> SelectionDecision:
    """A SelectionDecision assembled from the real S06 result shapes."""
    winner_mean, winner_std = 0.926600, 0.020526
    if proven:
        p_value, significant = 0.0086, True
    calibration = calibration_comparison(
        np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1]),
        np.array([0.10, 0.30, 0.55, 0.40, 0.70, 0.20, 0.80, 0.35, 0.90, 0.60]),
        np.array([0.06, 0.26, 0.60, 0.38, 0.66, 0.18, 0.84, 0.33, 0.94, 0.55]),
        n_bins=DEFAULT_CALIBRATION_BINS,
    )
    calibration = _brier_swapped(calibration, delta=brier_delta)
    return SelectionDecision(
        winner="random-forest",
        runner_up="logistic-regression-l2",
        baseline="logistic-regression-l2",
        metric=PRIMARY_METRIC,
        alpha=DEFAULT_ALPHA,
        ranked=(
            ("random-forest", winner_mean, winner_std),
            ("logistic-regression-l2", 0.917997, 0.022560),
        ),
        winner_vs_baseline=_comparison(p_value=p_value, significant=significant),
        other_comparisons=(),
        serving_weight=ServingWeightCheck(
            model_name="random-forest", weight_mb=2.41
        ),
        threshold=ThresholdTuning(
            objective=DEFAULT_OBJECTIVE,
            thresholds=(0.01, 0.02, 0.33),
            scores=(0.7, 0.75, 0.8603),
            best_threshold=0.33,
            best_score=0.8603,
        ),
        calibration=calibration,
        proven_significantly_better_than_baseline=proven,
        shippable_by_weight=weight_ok,
    )


def _brier_swapped(comparison, *, delta: float):
    from dataclasses import replace

    brier_calibrated = round(float(comparison.brier_score_calibrated), 6)
    return replace(
        comparison,
        brier_delta=delta,
        brier_score_raw=round(brier_calibrated - delta, 6),
    )


# ---------------------------------------------------------------------------
# Producer: selection decision -> s6.* tag payload
# ---------------------------------------------------------------------------


def test_annotation_tags_carry_the_required_story():
    tags = selection_annotation_tags(_decision())
    for field in ("winner", "metric", "cv_mean", "cv_std", "alpha", "ship"):
        assert f"{SELECTION_TAG_PREFIX}{field}" in tags
    annotated = {key: value for key, value in tags.items()}
    assert annotated[f"{SELECTION_TAG_PREFIX}winner"] == "random-forest"
    assert annotated[f"{SELECTION_TAG_PREFIX}ship"] == "NO-SHIP"
    assert annotated[f"{SELECTION_TAG_PREFIX}alpha"] == "0.05"
    assert annotated[f"{SELECTION_TAG_PREFIX}threshold"] == "0.33"
    assert annotated[f"{SELECTION_TAG_PREFIX}p_value"] == "0.3089"
    assert float(annotated[f"{SELECTION_TAG_PREFIX}brier_raw"]) > 0.0
    assert float(annotated[f"{SELECTION_TAG_PREFIX}brier_calibrated"]) > 0.0
    # Flag detail parentheses may contain commas; never the join separator.
    joined = annotated[f"{SELECTION_TAG_PREFIX}flags"]
    assert SELECTION_FLAGS_SEPARATOR in joined
    assert "not_significantly_better_than_baseline" in joined
    assert "calibration_did_not_improve_brier" in joined


def test_shipped_decision_flags_join_to_the_separator():
    tags = selection_annotation_tags(_decision(proven=True, brier_delta=0.01))
    assert tags[f"{SELECTION_TAG_PREFIX}ship"] == "SHIP"
    assert f"{SELECTION_TAG_PREFIX}flags" not in tags


def test_flag_overflow_raises_a_named_error():
    class _huge_flags_decision:
        winner = "random-forest"
        metric = PRIMARY_METRIC
        alpha = DEFAULT_ALPHA
        ranked = (("random-forest", 0.9, 0.01),)
        winner_vs_baseline = None
        other_comparisons = ()
        threshold = None
        calibration = None
        serving_weight = None
        flags = ("x" * (MAX_TAG_VALUE_LENGTH + 1),)
        ship = False

    with pytest.raises(SelectionAnnotationsError):
        selection_annotation_tags(_huge_flags_decision())


# ---------------------------------------------------------------------------
# Consumer: winner run tags -> annotation -> rendered report section
# ---------------------------------------------------------------------------


def _annotate(store: _Store, run_id: str, decision) -> str:
    annotated = annotate_winner_run(
        decision,
        split_version="v1",
        experiment_name=store.experiment_name,
        tracking_dir=store.tracking_dir,
    )
    assert annotated == run_id
    return annotated


def test_annotated_winner_renders_the_full_selection_section(store):
    winner_run = _log_run(
        store,
        model_type="random-forest",
        model_name="Random Forest",
        roc_auc=0.9364,
    )
    _log_run(
        store,
        model_type="logistic-regression-l2",
        model_name="Logistic Regression (L2)",
        roc_auc=0.9293,
    )
    _annotate(store, winner_run, _decision())

    board = build_leaderboard(
        config=LeaderboardConfig(experiment_name=store.experiment_name),
        tracking_dir=store.tracking_dir,
    )
    rendered = render_leaderboard(board)

    assert "## Winner selection (S06)" in rendered
    assert "- winner: **Random Forest** (`random-forest`)" in rendered
    assert "repeated-CV roc_auc: 0.926600 ± 0.020526" in rendered
    assert (
        "- significance vs baseline `logistic-regression-l2`: "
        "p = 0.3089 — not significant" in rendered
    )
    assert "- calibrated threshold: **0.33** (objective `f1` at 0.8603)" in rendered
    assert "calibration Brier: " in rendered
    assert "serving weight: **2.41 MB** (budget 25.00 MB)" in rendered
    assert "not_significantly_better_than_baseline" in rendered
    assert "- ship verdict: **NO-SHIP**" in rendered


def test_shipped_winner_renders_a_ship_verdict_and_no_flags(store):
    winner_run = _log_run(
        store, model_type="random-forest", model_name="Random Forest"
    )
    _annotate(store, winner_run, _decision(proven=True, brier_delta=0.01))

    board = build_leaderboard(
        config=LeaderboardConfig(experiment_name=store.experiment_name),
        tracking_dir=store.tracking_dir,
    )
    rendered = render_leaderboard(board)
    assert "- ship verdict: **SHIP**" in rendered
    assert "- flags: none" in rendered
    assert "not_significantly_better_than_baseline" not in rendered


def test_unannotated_board_renders_without_a_selection_section(store):
    _log_run(store, model_type="random-forest", model_name="Random Forest")
    board = build_leaderboard(
        config=LeaderboardConfig(experiment_name=store.experiment_name),
        tracking_dir=store.tracking_dir,
    )
    assert "## Winner selection (S06)" not in render_leaderboard(board)
    assert read_selection_annotations(board) is None


def test_device_column_renders_the_generic_neural_and_classical_labels(store):
    _log_run(
        store,
        model_type="mlp-deep",
        model_name="MLP (deep)",
        roc_auc=0.9338,
        tags={"device": "cuda:0"},
    )
    _log_run(
        store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9364
    )
    board = build_leaderboard(
        config=LeaderboardConfig(experiment_name=store.experiment_name),
        tracking_dir=store.tracking_dir,
    )
    rendered = render_leaderboard(board)
    assert "| Device |" in rendered
    assert "cuda:0" in rendered
    assert "| - |" in rendered  # classical rows render the no-device label


# ---------------------------------------------------------------------------
# Regenerability: the store is the sole input
# ---------------------------------------------------------------------------


def test_final_report_is_regenerable_from_mlflow_data_alone(store, tmp_path):
    winner_run = _log_run(
        store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9364
    )
    _log_run(
        store,
        model_type="logistic-regression-l2",
        model_name="Logistic Regression (L2)",
        roc_auc=0.9293,
    )
    _annotate(store, winner_run, _decision())

    fixed_at = "2026-01-01T00:00:00+00:00"

    def _regenerate() -> str:
        board = build_leaderboard(
            config=LeaderboardConfig(experiment_name=store.experiment_name),
            tracking_dir=store.tracking_dir,
            generated_at=fixed_at,
        )
        assert board.top is not None and board.top.model_type == "random-forest"
        return render_leaderboard(board)

    first = _regenerate()
    # A fresh build from the store (no page state, no cached decision object)
    # reproduces the same report byte-for-byte, annotations included.
    assert _regenerate() == first
    assert "## Winner selection (S06)" in first
    assert "- ship verdict: **NO-SHIP**" in first


def test_board_to_dict_serialises_the_annotations(store):
    winner_run = _log_run(
        store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9
    )
    _annotate(store, winner_run, _decision())
    board = build_leaderboard(
        config=LeaderboardConfig(experiment_name=store.experiment_name),
        tracking_dir=store.tracking_dir,
    )
    payload = json.loads(json.dumps(board.to_dict()))
    assert payload["n_models"] == 1
    assert payload["annotations"]["ship"] is False
    assert payload["annotations"]["threshold"] == 0.33


# ---------------------------------------------------------------------------
# Negative surface: named failures, never silent half-stories
# ---------------------------------------------------------------------------


def _annotate_raw(store: _Store, run_id: str, tags: dict[str, str]) -> None:
    mlflow.set_tracking_uri(store.config.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)


def _annotated_store(store: _Store, **overrides):
    run_id = _log_run(
        store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9
    )
    tags = {
        f"{SELECTION_TAG_PREFIX}winner": "random-forest",
        f"{SELECTION_TAG_PREFIX}metric": PRIMARY_METRIC,
        f"{SELECTION_TAG_PREFIX}cv_mean": "0.926600",
        f"{SELECTION_TAG_PREFIX}cv_std": "0.020526",
        f"{SELECTION_TAG_PREFIX}alpha": "0.05",
        f"{SELECTION_TAG_PREFIX}ship": "NO-SHIP",
    }
    tags.update(overrides)
    _annotate_raw(store, run_id, tags)
    return run_id


def _building_raises(store: _Store, expected):
    with pytest.raises(expected):
        build_leaderboard(
            config=LeaderboardConfig(experiment_name=store.experiment_name),
            tracking_dir=store.tracking_dir,
        )


def test_partial_annotation_set_is_a_named_data_error(store):
    run_id = _log_run(
        store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9
    )
    _annotate_raw(store, run_id, {f"{SELECTION_TAG_PREFIX}winner": "random-forest"})
    _building_raises(store, S06LeaderboardDataError)


def test_wrong_winner_tag_is_a_named_data_error(store):
    run_id = _log_run(
        store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9
    )
    tags = {
        f"{SELECTION_TAG_PREFIX}winner": "xgboost",
        f"{SELECTION_TAG_PREFIX}metric": PRIMARY_METRIC,
        f"{SELECTION_TAG_PREFIX}cv_mean": "0.9",
        f"{SELECTION_TAG_PREFIX}cv_std": "0.01",
        f"{SELECTION_TAG_PREFIX}alpha": "0.05",
        f"{SELECTION_TAG_PREFIX}ship": "NO-SHIP",
    }
    _annotate_raw(store, run_id, tags)
    _building_raises(store, S06LeaderboardDataError)


def test_invalid_ship_value_is_a_named_data_error(store):
    _annotated_store(store, **{f"{SELECTION_TAG_PREFIX}ship": "maybe"})
    _building_raises(store, S06LeaderboardDataError)


def test_unknown_annotation_field_is_a_named_data_error(store):
    _annotated_store(store, **{f"{SELECTION_TAG_PREFIX}loser": "xgboost"})
    _building_raises(store, S06LeaderboardDataError)


def test_nonfinite_annotation_number_is_a_named_data_error(store):
    _annotated_store(store, **{f"{SELECTION_TAG_PREFIX}cv_mean": "nan"})
    _building_raises(store, S06LeaderboardDataError)


def test_alpha_outside_unit_interval_is_a_named_data_error(store):
    _annotated_store(store, **{f"{SELECTION_TAG_PREFIX}alpha": "1.5"})
    _building_raises(store, S06LeaderboardDataError)


def test_negative_std_is_a_named_data_error(store):
    _annotated_store(store, **{f"{SELECTION_TAG_PREFIX}cv_std": "-0.01"})
    _building_raises(store, S06LeaderboardDataError)


def test_two_annotated_rows_are_an_ambiguous_winner(store):
    _annotated_store(store)
    other = _log_run(
        store, model_type="xgboost", model_name="XGBoost", roc_auc=0.89
    )
    tags = {
        f"{SELECTION_TAG_PREFIX}winner": "xgboost",
        f"{SELECTION_TAG_PREFIX}metric": PRIMARY_METRIC,
        f"{SELECTION_TAG_PREFIX}cv_mean": "0.88",
        f"{SELECTION_TAG_PREFIX}cv_std": "0.01",
        f"{SELECTION_TAG_PREFIX}alpha": "0.05",
        f"{SELECTION_TAG_PREFIX}ship": "SHIP",
    }
    _annotate_raw(store, other, tags)
    _building_raises(store, SelectionAnnotationsError)


def test_tag_value_length_cap_is_enforced(store):
    huge = "x" * (MAX_TAG_VALUE_LENGTH + 1)
    run_id = _log_run(
        store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9
    )
    base = selection_annotation_tags(_decision())
    fields = selection_tags({f"{SELECTION_TAG_PREFIX}{k}": v for k, v in {}})  # noqa: F841
    tags = dict(
        selection_annotation_tags(_decision()),
        **{f"{SELECTION_TAG_PREFIX}flags": huge},
    )
    _annotate_raw(store, run_id, tags)
    _building_raises(store, S06LeaderboardDataError)
