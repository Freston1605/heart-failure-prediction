"""Model-agnostic proof tests (S02/T04).

The claim under test is narrow and load-bearing: the shared evaluation
contract is genuinely model-agnostic, not shaped around logistic regression.
These tests prove it by pushing **structurally different** model families —
the project's logistic-regression baseline, a pruned decision tree, and a
prior-only ``DummyClassifier`` — through the *identical*
:func:`heart.eval.contract.evaluate` call and the *identical*
:func:`heart.tracking.run.log_evaluation_run` path, then asserting the results
are shaped the same.

What is pinned:

1. Every model type returns a metric dict with exactly the canonical keys, in
   the canonical order — the key sets are *identical across model types*.
2. Every model type logs to MLflow under the frozen convention and its
   ``metrics.json`` artifact carries the same key shape.
3. A second, non-linear model family really is different (a tree, not a linear
   model), so the equality is not an artefact of testing the same estimator
   twice.
4. The negative surface: unknown model types, missing frames, missing columns,
   unknown split versions, and models lacking ``predict_proba`` all fail loudly
   rather than producing a silently-degraded metric dict.

Read-only split frames come from the versioned S01 artifacts; every write goes
to ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, SplitNotFoundError, load_split_frames
from heart.eval.contract import (
    METRIC_KEYS,
    PRIMARY_METRIC,
    MissingPredictProbaError,
    evaluate,
    evaluation_split_from_frames,
)
from heart.models import throwaway as throwaway_module
from heart.models.baseline import BASELINE_MODEL_NAME, run_baseline, train_baseline
from heart.models.throwaway import (
    DEFAULT_MODEL_TYPE,
    THROWAWAY_MODEL_TYPES,
    THROWAWAY_SPECS,
    ThrowawayDataError,
    ThrowawayError,
    UnknownThrowawayModelError,
    build_throwaway_model,
    run_all_throwaway,
    run_throwaway,
    train_throwaway,
)
from heart.tracking.run import (
    METRICS_ARTIFACT,
    build_run_name,
    load_metrics_artifact,
    load_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split_frames():
    """The versioned S01 train/test frames (read-only)."""
    return load_split_frames(SPLIT_VERSION)


@pytest.fixture(scope="module")
def eval_split(split_frames):
    _, test = split_frames
    return evaluation_split_from_frames(test, name=f"{SPLIT_VERSION}/test")


@pytest.fixture(scope="module")
def model_metrics(split_frames, eval_split):
    """Metric dicts for the baseline and every throwaway model family."""
    train, _ = split_frames
    baseline = train_baseline(train)
    metrics_by_model = {
        BASELINE_MODEL_NAME: evaluate(baseline, eval_split),
    }
    for model_type in THROWAWAY_MODEL_TYPES:
        model = train_throwaway(model_type, train)
        metrics_by_model[THROWAWAY_SPECS[model_type].model_name] = evaluate(
            model, eval_split
        )
    return metrics_by_model


class _PredictOnlyModel:
    """A model exposing ``predict`` but deliberately not ``predict_proba``."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=int)


# ---------------------------------------------------------------------------
# One shape, many model families
# ---------------------------------------------------------------------------


def test_metric_key_sets_are_identical_across_model_types(model_metrics):
    """The core proof: every model family yields exactly the canonical shape."""
    assert len(model_metrics) >= 3  # baseline + at least two throwaway families

    canonical = tuple(METRIC_KEYS)
    for model_name, metrics in model_metrics.items():
        assert tuple(metrics.keys()) == canonical, (
            f"{model_name} produced a different metric-dict shape: "
            f"{list(metrics.keys())}"
        )
        assert set(metrics.keys()) == set(METRIC_KEYS)

    key_sets = {frozenset(metrics) for metrics in model_metrics.values()}
    assert len(key_sets) == 1


def test_metric_dicts_are_independently_schema_valid(model_metrics):
    """Each family's dict passes the contract's own validator, not just ours."""
    from heart.eval.contract import validate_metric_dict

    for model_name, metrics in model_metrics.items():
        validate_metric_dict(metrics)
        matrix = metrics["confusion_matrix"]
        assert matrix["n_samples"] == (
            matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"]
        ), model_name


def test_structural_difference_is_real(split_frames):
    """A tree and a linear model are genuinely different families."""
    train, _ = split_frames
    tree = train_throwaway("decision-tree", train)
    classifier = tree.named_steps["clf"]
    assert isinstance(classifier, DecisionTreeClassifier)
    assert not classifier.__class__.__name__.startswith("Logistic")

    dummy = train_throwaway("dummy", train)
    assert isinstance(dummy.named_steps["clf"], DummyClassifier)


def test_throwaway_trains_and_predicts_on_the_s01_split(split_frames):
    train, test = split_frames
    assert len(train) == 734
    assert len(test) == 184

    for model_type in THROWAWAY_MODEL_TYPES:
        model = train_throwaway(model_type, train)
        probabilities = model.predict_proba(test[list(FEATURE_COLUMNS)])
        predictions = model.predict(test[list(FEATURE_COLUMNS)])
        assert probabilities.shape == (len(test), 2), model_type
        assert predictions.shape == (len(test),), model_type
        assert np.all(np.isfinite(probabilities)), model_type
        assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
        assert set(np.unique(predictions)).issubset({0, 1}), model_type


def test_build_throwaway_model_uses_the_declared_params():
    tree = build_throwaway_model("decision-tree")
    classifier = tree.named_steps["clf"]
    declared = THROWAWAY_SPECS["decision-tree"].params
    assert classifier.max_depth == declared["max_depth"]
    assert classifier.min_samples_leaf == declared["min_samples_leaf"]
    assert classifier.random_state == declared["random_state"]

    dummy = build_throwaway_model("dummy")
    assert dummy.named_steps["clf"].strategy == "prior"


def test_throwaway_evaluation_is_deterministic(split_frames, eval_split):
    train, _ = split_frames
    for model_type in THROWAWAY_MODEL_TYPES:
        model = train_throwaway(model_type, train)
        assert evaluate(model, eval_split) == evaluate(model, eval_split)


def test_default_model_type_is_declared_and_is_a_tree():
    assert DEFAULT_MODEL_TYPE in THROWAWAY_MODEL_TYPES
    classifier = build_throwaway_model().named_steps["clf"]
    assert isinstance(classifier, DecisionTreeClassifier)


# ---------------------------------------------------------------------------
# The identical tracking path
# ---------------------------------------------------------------------------


def test_both_model_families_log_to_mlflow_with_equal_metric_shapes(tmp_path):
    """Baseline and a throwaway model both appear, shaped identically."""
    tracking_dir = tmp_path / "mlruns"

    baseline = run_baseline(
        tracking_dir=str(tracking_dir), write_report=False, report_path=None
    )
    throwaway = run_throwaway(
        model_type="decision-tree", tracking_dir=str(tracking_dir)
    )

    assert baseline.run_id is not None and throwaway.run_id is not None
    assert baseline.run_id != throwaway.run_id

    baseline_record = load_run(baseline.run_id)
    throwaway_record = load_run(throwaway.run_id)

    assert baseline_record.run_name == "logistic-regression-v1"
    assert throwaway_record.run_name == "throwaway-decision-tree-v1"
    assert (
        baseline_record.experiment_id == throwaway_record.experiment_id
    ), "both models must land in the same experiment to be comparable"

    baseline_artifact = load_metrics_artifact(baseline.run_id)
    throwaway_artifact = load_metrics_artifact(throwaway.run_id)
    assert set(baseline_artifact) == set(throwaway_artifact) == set(METRIC_KEYS)
    assert tuple(baseline_artifact) == tuple(throwaway_artifact)

    for key in ("roc_auc", "accuracy", "confusion_matrix.tp", "calibration.n_bins"):
        assert key in baseline_record.metrics
        assert key in throwaway_record.metrics

    assert baseline_record.tags["model_name"] == BASELINE_MODEL_NAME
    assert throwaway_record.tags["model_name"] == "throwaway-decision-tree"
    assert throwaway_record.tags["role"] == "model-agnostic-proof"
    assert throwaway_record.tags["primary_metric"] == PRIMARY_METRIC
    assert METRICS_ARTIFACT in throwaway_record.artifact_paths


def test_run_all_throwaway_logs_every_declared_model_type(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    results = run_all_throwaway(tracking_dir=str(tracking_dir))

    assert len(results) == len(THROWAWAY_MODEL_TYPES)
    assert {result.model_type for result in results} == set(THROWAWAY_MODEL_TYPES)

    shapes = {tuple(result.metrics.keys()) for result in results}
    assert shapes == {tuple(METRIC_KEYS)}

    run_names = set()
    experiment_ids = set()
    for result in results:
        assert result.run_id is not None
        record = load_run(result.run_id)
        assert record.run_name == build_run_name(result.model_name, SPLIT_VERSION)
        assert set(record.metrics) >= {"roc_auc", "accuracy", "calibration.n_bins"}
        artifact = load_metrics_artifact(result.run_id)
        assert tuple(artifact) == tuple(METRIC_KEYS)
        run_names.add(record.run_name)
        experiment_ids.add(record.experiment_id)

    assert len(run_names) == len(results), run_names
    assert len(experiment_ids) == 1, "all throwaway runs share one experiment"


# ---------------------------------------------------------------------------
# Negative surface
# ---------------------------------------------------------------------------


def test_unknown_model_type_raises():
    with pytest.raises(UnknownThrowawayModelError):
        build_throwaway_model("does-not-exist")
    with pytest.raises(UnknownThrowawayModelError):
        train_throwaway("does-not-exist", pd.DataFrame())


def test_train_throwaway_requires_a_frame():
    with pytest.raises(ThrowawayDataError):
        train_throwaway("decision-tree", None)


def test_train_throwaway_rejects_a_missing_target():
    frame = pd.DataFrame({column: [1, 2, 3] for column in FEATURE_COLUMNS})
    with pytest.raises(ThrowawayDataError):
        train_throwaway("decision-tree", frame)


def test_run_throwaway_unknown_split_version_raises(tmp_path):
    with pytest.raises(SplitNotFoundError):
        run_throwaway(
            model_type="decision-tree",
            split_version="v-does-not-exist",
            tracking_dir=str(tmp_path / "mlruns"),
        )


def test_run_throwaway_can_skip_mlflow(split_frames):
    result = run_throwaway(model_type="dummy", log_to_mlflow=False)
    assert result.run_id is None
    assert set(result.metrics.keys()) == set(METRIC_KEYS)


def test_evaluate_rejects_a_model_without_predict_proba(eval_split):
    """The interface requirement applies to every model family, not just ours."""
    with pytest.raises(MissingPredictProbaError):
        evaluate(_PredictOnlyModel(), eval_split)


def test_run_all_throwaway_fails_loudly_when_shapes_diverge(monkeypatch):
    """The model-agnostic invariant is enforced, not merely asserted in a test."""
    real_evaluate = throwaway_module.evaluate
    calls = {"count": 0}

    def drifting_evaluate(model, split, **kwargs):
        calls["count"] += 1
        metrics = real_evaluate(model, split, **kwargs)
        if calls["count"] >= 2:
            metrics = {
                key: value for key, value in metrics.items() if key != "specificity"
            }
        return metrics

    monkeypatch.setattr(throwaway_module, "evaluate", drifting_evaluate)

    with pytest.raises(ThrowawayError):
        run_all_throwaway(log_to_mlflow=False)
    assert calls["count"] >= 2


def test_throwaway_errors_share_a_base():
    assert issubclass(UnknownThrowawayModelError, ThrowawayError)
    assert issubclass(ThrowawayDataError, ThrowawayError)
