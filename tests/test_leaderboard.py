"""Tests for the MLflow-derived leaderboard generator (S03/T04).

Contracts under test:

1. **Selection** — the generator reads only finished, ``run_kind=final`` runs
   that carry a ``model_type``; it filters by split version, deduplicates a
   re-benchmarked model to one row, and orders rows by ROC-AUC.
2. **The full metric suite** — every row carries the complete flattened metric
   suite promised by the tracking convention; a run with an incomplete suite
   is a named data error, never a silently dropped row.
3. **Regenerable output** — ``reports/leaderboard.md`` is rendered from run
   data; logging a new run makes it appear in the regenerated report with no
   manual edits.
4. **Integration** — a real battery run (via ``heart.models.run_battery``)
   flows straight into the leaderboard with every benchmarked member present.
5. **Negative surface** — missing experiment, incomplete metrics, empty
   selection (strict), bad config, non-Leaderboard inputs, and unwritable
   report paths all raise named errors.

Every store-backed test runs against a fresh ``tmp_path`` SQLite database, so
the suite never touches the repository's ``experiments/mlruns`` directory.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from heart.data.schema import TARGET_COLUMN
from heart.eval.contract import (
    CALIBRATION_KEY,
    CONFUSION_MATRIX_KEY,
    METRIC_KEYS,
    PRIMARY_METRIC,
    compute_metric_dict,
)
from heart.models.registry import BATTERY_MODEL_TYPES
from heart.models.run_battery import BATTERY_EXPERIMENT, run_battery, smoke_config
from heart.reporting.leaderboard import (
    DEFAULT_RUN_KIND,
    REQUIRED_FLAT_KEYS,
    Leaderboard,
    LeaderboardConfig,
    LeaderboardConfigError,
    LeaderboardDataError,
    LeaderboardError,
    LeaderboardExperimentNotFoundError,
    LeaderboardReportError,
    NoBenchmarkedModelsError,
    build_leaderboard,
    build_parser,
    describe_leaderboard,
    generate_leaderboard,
    main,
    render_leaderboard,
    select_leaderboard_rows,
    write_leaderboard,
)
from heart.tracking.mlflow_store import DEFAULT_EXPERIMENT, configure_tracking
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


def _sample_metrics(*, roc_auc: float | None = None, n_bins: int = 5) -> dict:
    """A complete, schema-valid metric dict built without fitting a model."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 4)
    probabilities = np.clip(0.18 + 0.64 * labels, 0.0, 1.0)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = compute_metric_dict(labels, probabilities, predictions, n_bins=n_bins)
    if roc_auc is not None:
        # ROC-AUC is validated independently of the confusion matrix, so an
        # override still produces a schema-valid dict while letting a test pin
        # the leaderboard ordering deterministically.
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
    run_kind: str = FINAL_RUN_KIND,
    params: dict | None = None,
    metrics: dict | None = None,
) -> str:
    mlflow.set_tracking_uri(store.config.tracking_uri)
    run = log_evaluation_run(
        metrics if metrics is not None else _sample_metrics(roc_auc=roc_auc),
        model_name=model_name,
        split_version=split_version,
        params=params if params is not None else {"C": 1.0},
        tags={
            RUN_KIND_TAG: run_kind,
            MODEL_TYPE_TAG: model_type,
            "family": family,
        },
        config=store.config,
    )
    return run.run_id


def _log_raw_run(
    store: _Store,
    *,
    model_type: str,
    run_kind: str = FINAL_RUN_KIND,
    split_version: str = "v1",
    metrics: dict | None = None,
) -> str:
    """Log a run directly through MLflow (for malformed / partial data)."""
    mlflow.set_tracking_uri(store.config.tracking_uri)
    with mlflow.start_run(experiment_id=store.config.experiment_id) as run:
        mlflow.set_tags(
            {
                RUN_KIND_TAG: run_kind,
                MODEL_TYPE_TAG: model_type,
                "split_version": split_version,
            }
        )
        for key, value in (metrics or {}).items():
            mlflow.log_metric(key, float(value))
        return run.info.run_id


def _read(path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_config_targets_final_runs_in_the_shared_experiment():
    config = LeaderboardConfig()
    assert config.experiment_name == DEFAULT_EXPERIMENT
    assert config.run_kind == FINAL_RUN_KIND
    assert config.latest_only is True
    assert config.strict is False
    assert config.split_version is None


@pytest.mark.parametrize("value", ["", "   "])
def test_config_rejects_blank_experiment_name(value):
    with pytest.raises(LeaderboardConfigError):
        LeaderboardConfig(experiment_name=value)


def test_config_rejects_blank_run_kind():
    with pytest.raises(LeaderboardConfigError):
        LeaderboardConfig(run_kind="  ")


def test_config_rejects_blank_split_version():
    with pytest.raises(LeaderboardConfigError):
        LeaderboardConfig(split_version="  ")


def test_config_rejects_non_boolean_flags():
    with pytest.raises(LeaderboardConfigError):
        LeaderboardConfig(latest_only="yes")  # type: ignore[arg-type]
    with pytest.raises(LeaderboardConfigError):
        LeaderboardConfig(strict=1)  # type: ignore[arg-type]


def test_config_serialises_round_trip():
    payload = LeaderboardConfig(split_version="v1", strict=True).to_dict()
    assert payload["split_version"] == "v1"
    assert payload["strict"] is True


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_build_leaderboard_is_empty_for_a_store_without_battery_runs(store):
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert isinstance(board, Leaderboard)
    assert board.n_models == 0
    assert board.rows == ()
    assert board.n_runs_considered == 0
    assert board.top is None


def test_build_leaderboard_raises_for_a_missing_experiment(tmp_path):
    with pytest.raises(LeaderboardExperimentNotFoundError):
        build_leaderboard(
            tracking_dir=tmp_path / "empty",
            config=LeaderboardConfig(experiment_name="does-not-exist"),
        )


def test_selection_counts_candidates_before_deduplication(store):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.80)
    _log_run(store, model_type="qda", model_name="QDA", roc_auc=0.70)
    rows, considered = select_leaderboard_rows(tracking_dir=store.tracking_dir)
    assert considered == 2
    assert len(rows) == 2


def test_leaderboard_is_ordered_by_roc_auc_descending(store):
    _log_run(store, model_type="a-model", model_name="A Model", roc_auc=0.70)
    _log_run(store, model_type="b-model", model_name="B Model", roc_auc=0.95)
    _log_run(store, model_type="c-model", model_name="C Model", roc_auc=0.82)
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert [row.model_type for row in board.rows] == ["b-model", "c-model", "a-model"]
    assert [round(row.primary_metric, 2) for row in board.rows] == [0.95, 0.82, 0.70]


def test_latest_only_deduplicates_a_rebenchmarked_model(store):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.71)
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.93)

    latest = build_leaderboard(tracking_dir=store.tracking_dir)
    assert latest.n_models == 1
    assert latest.n_runs_considered == 2
    assert round(latest.rows[0].primary_metric, 2) == 0.93

    every = build_leaderboard(
        config=LeaderboardConfig(latest_only=False),
        tracking_dir=store.tracking_dir,
    )
    assert every.n_models == 2
    assert {round(row.primary_metric, 2) for row in every.rows} == {0.71, 0.93}


def test_split_version_filter_limits_the_selection(store):
    _log_run(store, model_type="lda", model_name="LDA", split_version="v1", roc_auc=0.80)
    _log_run(store, model_type="qda", model_name="QDA", split_version="v2", roc_auc=0.85)
    board = build_leaderboard(
        config=LeaderboardConfig(split_version="v1"),
        tracking_dir=store.tracking_dir,
    )
    assert board.n_models == 1
    assert board.row_for("lda") is not None
    assert board.row_for("qda") is None


def test_run_kind_filter_excludes_trial_runs(store):
    _log_run(store, model_type="naive-bayes", model_name="Naive Bayes", roc_auc=0.80)
    # A trial run with partial metrics must be excluded by tag, before the
    # required-metric check would ever fire.
    _log_raw_run(
        store,
        model_type="naive-bayes",
        run_kind="trial",
        metrics={"roc_auc": 0.5},
    )
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert board.n_models == 1
    assert board.rows[0].run_id != ""


def test_only_finished_runs_are_considered(store):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.80)

    # A FAILED run is excluded by status *before* the required-metric check
    # would fire, so it may legitimately carry no metrics at all.
    mlflow.set_tracking_uri(store.config.tracking_uri)
    mlflow.start_run(experiment_id=store.config.experiment_id)
    mlflow.set_tags(
        {
            RUN_KIND_TAG: FINAL_RUN_KIND,
            MODEL_TYPE_TAG: "failed-model",
            "split_version": "v1",
        }
    )
    mlflow.end_run(status="FAILED")

    board = build_leaderboard(tracking_dir=store.tracking_dir)
    assert board.row_for("failed-model") is None
    assert board.n_models == 1


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------


def test_every_row_carries_the_complete_flattened_metric_suite(store):
    _log_run(store, model_type="svm", model_name="Support Vector Machine", roc_auc=0.88)
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    row = board.row_for("svm")
    assert row is not None
    missing = [key for key in REQUIRED_FLAT_KEYS if key not in row.flat_metrics]
    assert missing == []
    # The scalar schema is a subset of the flattened suite.
    for key in METRIC_KEYS:
        if key in (CONFUSION_MATRIX_KEY, CALIBRATION_KEY):
            continue
        assert key in row.flat_metrics


def test_render_expands_the_structured_metric_suite(store):
    _log_run(store, model_type="random-forest", model_name="Random Forest", roc_auc=0.9)
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    text = render_leaderboard(board)
    assert "ROC-AUC" in text
    assert "## Confusion matrices" in text
    assert "## Calibration" in text
    assert "Random Forest" in text
    assert "`random-forest`" in text
    assert "Expected calibration error" in text


def test_missing_required_metric_is_a_named_data_error(store):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.8)
    # A "final" run that skipped the shared logging path has no metric suite.
    _log_raw_run(
        store,
        model_type="broken",
        metrics={"roc_auc": 0.9},
    )
    with pytest.raises(LeaderboardDataError) as excinfo:
        build_leaderboard(tracking_dir=store.tracking_dir)
    message = str(excinfo.value)
    assert "required metric" in message
    assert "broken" in message


# ---------------------------------------------------------------------------
# Regeneration — the core requirement
# ---------------------------------------------------------------------------


def test_generate_emits_every_logged_model(store, tmp_path):
    _log_run(store, model_type="naive-bayes", model_name="Naive Bayes", roc_auc=0.81)
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.86)
    _log_run(store, model_type="qda", model_name="QDA", roc_auc=0.79)

    report_path = tmp_path / "reports" / "leaderboard.md"
    board = generate_leaderboard(
        tracking_dir=store.tracking_dir, report_path=report_path
    )
    assert board.report_path == report_path
    assert report_path.exists()

    text = _read(report_path)
    assert "# Heart-Failure Model Leaderboard" in text
    for name in ("Naive Bayes", "LDA", "QDA"):
        assert name in text
    assert "models ranked: **3**" in text
    assert "## Reproduce" in text


def test_newly_logged_run_appears_in_regenerated_output(store, tmp_path):
    _log_run(store, model_type="naive-bayes", model_name="Naive Bayes", roc_auc=0.81)
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.86)

    report_path = tmp_path / "leaderboard.md"
    generate_leaderboard(tracking_dir=store.tracking_dir, report_path=report_path)
    first = _read(report_path)
    assert "K-Nearest Neighbours" not in first
    assert "models ranked: **2**" in first

    # Log one more benchmarked model — no manual edit to the report.
    _log_run(store, model_type="knn", model_name="K-Nearest Neighbours", roc_auc=0.9)

    board = generate_leaderboard(
        tracking_dir=store.tracking_dir, report_path=report_path
    )
    second = _read(report_path)
    assert "K-Nearest Neighbours" in second
    assert "models ranked: **3**" in second
    assert "K-Nearest Neighbours" not in first  # proves it was regenerated
    assert [row.model_type for row in board.rows][0] == "knn"


def test_generate_without_writing_returns_the_board(store):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.8)
    board = generate_leaderboard(
        tracking_dir=store.tracking_dir, report_path=None, write=False
    )
    assert board.report_path is None
    assert board.n_models == 1


def test_board_to_dict_is_json_serialisable(store):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.8)
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    payload = board.to_dict()
    json.dumps(payload)
    assert payload["n_models"] == 1
    assert payload["rows"][0]["model_type"] == "lda"
    assert "flat_metrics" in payload["rows"][0]


def test_describe_leaderboard_names_the_provenance(store):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.8)
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    text = describe_leaderboard(board)
    assert board.experiment_name in text
    assert "run kind" in text
    assert "models ranked: **1**" in text


# ---------------------------------------------------------------------------
# Empty selection / strict mode
# ---------------------------------------------------------------------------


def test_empty_selection_renders_an_explanatory_report(store, tmp_path):
    report_path = tmp_path / "leaderboard.md"
    board = generate_leaderboard(
        tracking_dir=store.tracking_dir, report_path=report_path
    )
    assert board.n_models == 0
    text = _read(report_path)
    assert "No benchmarked models matched" in text


def test_strict_mode_rejects_an_empty_selection(store):
    with pytest.raises(NoBenchmarkedModelsError):
        build_leaderboard(
            config=LeaderboardConfig(strict=True),
            tracking_dir=store.tracking_dir,
        )


# ---------------------------------------------------------------------------
# Negative surface: rendering / writing
# ---------------------------------------------------------------------------


def test_render_rejects_a_non_leaderboard():
    with pytest.raises(LeaderboardReportError):
        render_leaderboard({"not": "a leaderboard"})  # type: ignore[arg-type]


def test_write_rejects_a_non_leaderboard(tmp_path):
    with pytest.raises(LeaderboardReportError):
        write_leaderboard("nope", tmp_path / "x.md")  # type: ignore[arg-type]


def test_write_reports_an_unwritable_destination(store, tmp_path):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.8)
    board = build_leaderboard(tracking_dir=store.tracking_dir)
    directory = tmp_path / "a-directory"
    directory.mkdir()
    with pytest.raises(LeaderboardReportError):
        write_leaderboard(board, directory)


# ---------------------------------------------------------------------------
# Integration with the real battery runner
# ---------------------------------------------------------------------------


def _schema_valid_frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = (rng.random(rows) < 0.55).astype(int)
    return pd.DataFrame(
        {
            "Age": rng.integers(29, 78, rows),
            "Sex": rng.choice(["F", "M"], rows),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], rows),
            "RestingBP": rng.integers(95, 190, rows),
            "Cholesterol": np.where(
                rng.random(rows) < 0.2, 0, rng.integers(120, 340, rows)
            ),
            "FastingBS": rng.integers(0, 2, rows),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], rows),
            "MaxHR": rng.integers(70, 200, rows),
            "ExerciseAngina": rng.choice(["N", "Y"], rows),
            "Oldpeak": rng.uniform(0.0, 4.0, rows).round(2),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], rows),
            TARGET_COLUMN: labels,
        }
    )


def test_battery_runs_flow_into_the_leaderboard(tmp_path):
    tracking_dir = tmp_path / "battery-mlruns"
    result = run_battery(
        _schema_valid_frame(rows=160, seed=7),
        _schema_valid_frame(rows=80, seed=11),
        model_types=["naive-bayes", "lda", "knn"],
        config=smoke_config(n_trials=1, cv_folds=2),
        tracking_dir=tracking_dir,
        split_version="v1",
        battery_id="leaderboard-integration",
    )
    assert result.n_succeeded == 3

    report_path = tmp_path / "leaderboard.md"
    board = generate_leaderboard(
        tracking_dir=tracking_dir, report_path=report_path
    )
    assert board.experiment_name == BATTERY_EXPERIMENT
    assert board.n_models == 3
    assert {row.model_type for row in board.rows} == {"naive-bayes", "lda", "knn"}
    for row in board.rows:
        assert row.run_id
        assert set(REQUIRED_FLAT_KEYS).issubset(row.flat_metrics)
    # Ordered by ROC-AUC — the top row is the maximum.
    scores = [row.primary_metric for row in board.rows]
    assert scores == sorted(scores, reverse=True)
    for model_type in ("naive-bayes", "lda", "knn"):
        assert model_type in _read(report_path)


def test_battery_models_are_a_subset_of_the_registry():
    # Guards the integration test's model selection against registry churn.
    assert {"naive-bayes", "lda", "knn"} <= set(BATTERY_MODEL_TYPES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.experiment == DEFAULT_EXPERIMENT
    assert args.run_kind == DEFAULT_RUN_KIND
    assert args.all_runs is False
    assert args.strict is False
    assert args.no_report is False


def test_main_generates_the_report(store, tmp_path, capsys):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.8)
    report_path = tmp_path / "leaderboard.md"
    rc = main(
        [
            "--tracking-dir",
            str(store.tracking_dir),
            "--report-path",
            str(report_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "top model: LDA" in out
    assert str(report_path) in out
    assert report_path.exists()


def test_main_no_report_skips_writing(store, tmp_path, capsys):
    _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.8)
    report_path = tmp_path / "leaderboard.md"
    rc = main(
        [
            "--tracking-dir",
            str(store.tracking_dir),
            "--report-path",
            str(report_path),
            "--no-report",
        ]
    )
    assert rc == 0
    assert not report_path.exists()


def test_main_returns_one_for_a_missing_experiment(tmp_path, capsys):
    rc = main(
        [
            "--tracking-dir",
            str(tmp_path / "empty"),
            "--experiment",
            "nope",
        ]
    )
    assert rc == 1
    assert "leaderboard error" in capsys.readouterr().out


def test_main_returns_two_for_invalid_config(tmp_path, capsys):
    rc = main(["--tracking-dir", str(tmp_path), "--experiment", "   "])
    assert rc == 2
    assert "configuration error" in capsys.readouterr().out


def test_main_strict_returns_one_when_empty(store, capsys):
    rc = main(["--tracking-dir", str(store.tracking_dir), "--strict"])
    assert rc == 1
    assert "leaderboard error" in capsys.readouterr().out


def test_leaderboard_errors_share_a_base_class():
    for error in (
        LeaderboardConfigError,
        LeaderboardDataError,
        LeaderboardExperimentNotFoundError,
        LeaderboardReportError,
        NoBenchmarkedModelsError,
    ):
        assert issubclass(error, LeaderboardError)
