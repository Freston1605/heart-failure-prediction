"""Runs-page tests (M002/S02).

Two test sections cover the slice's pinned contract:

1. **Lib contract (T01)** — :mod:`app.lib.runs_view` is the never-raise
   boundary between the dashboard and the MLflow store. Against fresh
   ``tmp_path`` stores these tests pin: the ranked rows come back from
   :func:`heart.reporting.leaderboard.select_leaderboard_rows` (D019) with
   ROC-AUC ordering matching the seeded input data; a missing, empty or
   unreadable store yields a *named* error kind and a friendly message
   naming ``make reproduce`` — never an exception; the per-run drilldown
   round-trips params/tags/metrics/artifacts; and the ``RUNS_TRACKING_DIR``
   environment seam points the page at a tmp store without an explicit
   argument.

2. **AppTest render (T02)** — ``AppTest.from_file`` renders of the real
   ``app/pages/3_Runs.py`` against tmp stores: a populated store renders
   zero exceptions with the top-ranked model, the provenance line and one
   drilldown expander per ranked row; an empty or missing store renders
   the friendly named message (the ``make reproduce`` hint) with no
   'Traceback' text leaking anywhere. The real populated-store browser
   render is S03 scope (fixture-level here, matching the roadmap).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib import runs_view  # noqa: E402
from app.lib.runs_view import (  # noqa: E402
    ERROR_KIND_DRILLDOWN_UNKNOWN,
    ERROR_KIND_RUNS_UNKNOWN,
    ERROR_KIND_STORE_EMPTY,
    ERROR_KIND_STORE_MISSING,
    RUNS_TRACKING_DIR_ENV,
    RunDetails,
    RunsView,
    load_runs_view,
    run_details,
)
from heart.eval.contract import PRIMARY_METRIC, compute_metric_dict  # noqa: E402
from heart.tracking.mlflow_store import configure_tracking  # noqa: E402
from heart.tracking.run import log_evaluation_run  # noqa: E402
from heart.tuning.runner import FINAL_RUN_KIND, MODEL_TYPE_TAG, RUN_KIND_TAG  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers (mirroring tests/test_leaderboard.py's seeding pattern)
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
        # override still produces a schema-valid dict while pinning the
        # leaderboard ordering deterministically.
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
    params: dict | None = None,
) -> str:
    """Log one schema-valid final run; returns its run id."""
    mlflow.set_tracking_uri(store.config.tracking_uri)
    run = log_evaluation_run(
        _sample_metrics(roc_auc=roc_auc),
        model_name=model_name,
        split_version=split_version,
        params=params if params is not None else {"C": 1.0},
        tags={
            RUN_KIND_TAG: FINAL_RUN_KIND,
            MODEL_TYPE_TAG: model_type,
            "family": family,
        },
        config=store.config,
    )
    return run.run_id


# ---------------------------------------------------------------------------
# 1. Populated store: the ranked leaderboard view
# ---------------------------------------------------------------------------


class TestLoadRunsViewPopulated:
    """A populated tmp store yields the ranked rows, never an exception."""

    def test_rows_are_ranked_by_roc_auc_descending(self, store) -> None:
        # Logged in deliberately non-ranked ("shuffled") creation order.
        _log_run(store, model_type="m-second", model_name="Second", roc_auc=0.93)
        _log_run(store, model_type="m-third", model_name="Third", roc_auc=0.92)
        _log_run(store, model_type="m-first", model_name="First", roc_auc=0.94)

        view = load_runs_view(tracking_dir=store.tracking_dir)

        assert isinstance(view, RunsView)
        assert view.ok is True
        assert [row.primary_metric for row in view.rows] == [0.94, 0.93, 0.92]
        assert [row.model_name for row in view.rows] == ["First", "Second", "Third"]
        assert view.n_runs_considered == 3
        assert view.error_kind is None
        assert view.error_message is None

    def test_rows_match_select_leaderboard_rows_input_data(self, store) -> None:
        """The view rows ARE select_leaderboard_rows' rows (D019, by identity
        of the producing function): same ids, same primary metric ordering."""
        run_ids = [
            _log_run(store, model_type="a-model", model_name="A", roc_auc=0.80),
            _log_run(store, model_type="b-model", model_name="B", roc_auc=0.95),
            _log_run(store, model_type="c-model", model_name="C", roc_auc=0.82),
        ]

        view = load_runs_view(tracking_dir=store.tracking_dir)

        from heart.reporting.leaderboard import select_leaderboard_rows

        expected_rows, expected_n = select_leaderboard_rows(
            tracking_dir=store.tracking_dir
        )
        assert view.n_runs_considered == expected_n == 3
        assert [row.run_id for row in view.rows] == [row.run_id for row in expected_rows]
        assert set(run_ids) == {row.run_id for row in view.rows}
        scores = [round(row.primary_metric, 2) for row in view.rows]
        assert scores == [0.95, 0.82, 0.80]

    def test_tracking_uri_is_resolved(self, store) -> None:
        _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.9)
        view = load_runs_view(tracking_dir=store.tracking_dir)
        assert view.ok
        assert view.tracking_uri.startswith("sqlite:///")
        assert (store.tracking_dir / "mlflow.db").as_posix() in view.tracking_uri


# ---------------------------------------------------------------------------
# 2. Empty / missing store: named kinds, friendly messages
# ---------------------------------------------------------------------------


class TestLoadRunsViewEmptyStore:
    """Empty and experiment-absent stores map to ``store-empty``."""

    def test_experiment_exists_but_zero_runs(self, store) -> None:
        view = load_runs_view(tracking_dir=store.tracking_dir)

        assert view.ok is False
        assert view.rows == ()
        assert view.error_kind == ERROR_KIND_STORE_EMPTY
        assert "make reproduce" in view.error_message

    def test_experiment_absent_from_an_existing_store(self, tmp_path) -> None:
        # Configure a store whose only experiment is NOT the default one, so
        # the store directory exists but the default experiment does not.
        other_dir = tmp_path / "other-mlruns"
        config = configure_tracking(
            tracking_dir=other_dir, experiment_name="a-different-experiment"
        )
        mlflow.set_tracking_uri(config.tracking_uri)

        view = load_runs_view(tracking_dir=other_dir)

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_STORE_EMPTY
        assert "make reproduce" in view.error_message


class TestLoadRunsViewMissingStore:
    """A missing or unreadable store directory maps to ``store-missing``."""

    def test_directory_does_not_exist(self, tmp_path) -> None:
        view = load_runs_view(tracking_dir=tmp_path / "no-such-mlruns")

        assert view.ok is False
        assert view.rows == ()
        assert view.error_kind == ERROR_KIND_STORE_MISSING
        assert "make reproduce" in view.error_message

    def test_path_is_a_file_not_a_directory(self, tmp_path) -> None:
        file_path = tmp_path / "not-a-dir"
        file_path.write_text("plainly not a store", encoding="utf-8")

        view = load_runs_view(tracking_dir=file_path)

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_STORE_MISSING
        assert "make reproduce" in view.error_message


# ---------------------------------------------------------------------------
# 3. Never-raise boundary
# ---------------------------------------------------------------------------


class TestNeverRaise:
    """Every outcome is a returned kind; no input raises."""

    def test_load_never_raises_on_any_store_shape(self, tmp_path, store) -> None:
        _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.9)
        file_path = tmp_path / "a-file"
        file_path.write_text("x", encoding="utf-8")
        targets = [
            None,  # default (project) store — may exist or not, never raises
            tmp_path / "missing-dir",
            file_path,
            tmp_path,  # exists but has no experiments
        ]
        for target in targets:
            view = load_runs_view(
                tracking_dir=target if target is not None else tmp_path / "maybe"
            )
            assert isinstance(view, RunsView)
            if target is not None and target != file_path and target != (
                tmp_path / "missing-dir"
            ):
                assert isinstance(view.ok, bool)

    def test_drilldown_never_raises_on_unknown_id(self, store) -> None:
        details = run_details("no-such-run-id-0000", tracking_dir=store.tracking_dir)
        assert isinstance(details, RunDetails)
        assert details.ok is False
        assert details.error_kind == ERROR_KIND_DRILLDOWN_UNKNOWN
        assert "make reproduce" in details.error_message


# ---------------------------------------------------------------------------
# 4. Per-run drilldown
# ---------------------------------------------------------------------------


class TestRunDetails:
    """The drilldown round-trips a seeded run's params/tags/metrics/artifacts."""

    def test_seeded_run_round_trips(self, store) -> None:
        params = {"C": 1.0, "max_iter": 500}
        run_id = _log_run(
            store,
            model_type="lda",
            model_name="Linear Discriminant Analysis",
            roc_auc=0.88,
            params=params,
        )

        details = run_details(run_id, tracking_dir=store.tracking_dir)

        assert isinstance(details, RunDetails)
        assert details.ok is True
        assert details.error_kind is None
        assert details.run_id == run_id
        assert details.run_name == "linear-discriminant-analysis-v1"
        assert details.params["C"] == "1.0"
        assert details.params["max_iter"] == "500"
        # Convention tags round-trip alongside caller tags.
        assert details.tags["model_name"] == "Linear Discriminant Analysis"
        assert details.tags[RUN_KIND_TAG] == FINAL_RUN_KIND
        assert details.tags[MODEL_TYPE_TAG] == "lda"
        # Flattened scalar metrics include the primary metric.
        assert "roc_auc" in details.metrics
        assert details.metrics["roc_auc"] == pytest.approx(0.88)
        # The convention artifacts are attached.
        assert "metrics.json" in details.artifact_names
        assert "run_config.json" in details.artifact_names

    def test_unknown_run_id_is_a_named_error(self, store) -> None:
        details = run_details("deadbeef00000000", tracking_dir=store.tracking_dir)

        assert details.ok is False
        assert details.error_kind == ERROR_KIND_DRILLDOWN_UNKNOWN
        assert details.params == {}
        assert details.metrics == {}
        assert "make reproduce" in details.error_message


# ---------------------------------------------------------------------------
# 5. Environment seam (RUNS_TRACKING_DIR)
# ---------------------------------------------------------------------------


class TestRunsTrackingDirEnvSeam:
    """``RUNS_TRACKING_DIR`` points the loader at a tmp store without an arg."""

    def test_env_override_yields_populated_store(self, store, monkeypatch) -> None:
        _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.91)
        monkeypatch.setenv(RUNS_TRACKING_DIR_ENV, str(store.tracking_dir))

        view = load_runs_view()

        assert view.ok is True
        assert view.n_runs_considered == 1
        assert view.rows[0].model_name == "LDA"

    def test_env_override_missing_dir_yields_store_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv(RUNS_TRACKING_DIR_ENV, str(tmp_path / "absent"))

        view = load_runs_view()

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_STORE_MISSING

    def test_drilldown_honours_env_override(self, store, monkeypatch) -> None:
        """Regression: run_details resolved the raw ``tracking_dir=None``
        through ``resolve_tracking_uri`` directly, which skips the env seam
        and silently drilled into the project-default store — the page's
        drilldown read the wrong store in AppTest renders. It must honour
        ``RUNS_TRACKING_DIR`` exactly like ``load_runs_view`` does.
        """
        run_id = _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.90)
        monkeypatch.setenv(RUNS_TRACKING_DIR_ENV, str(store.tracking_dir))

        details = run_details(run_id)

        assert details.ok is True
        assert details.run_id == run_id
        assert details.run_name.endswith("-v1")
        assert "roc_auc" in details.metrics

    def test_explicit_arg_wins_over_env_override(
        self, store, tmp_path, monkeypatch
    ) -> None:
        _log_run(store, model_type="lda", model_name="LDA", roc_auc=0.91)
        monkeypatch.setenv(RUNS_TRACKING_DIR_ENV, str(tmp_path / "absent"))

        view = load_runs_view(tracking_dir=store.tracking_dir)

        assert view.ok is True


# ---------------------------------------------------------------------------
# 6. Module hygiene: plain Python, no streamlit dependency
# ---------------------------------------------------------------------------


def test_runs_view_module_imports_without_streamlit() -> None:
    """The contract module must be pytest-testable without Streamlit (MEM047)."""
    code = (
        "import sys\n"
        "import app.lib.runs_view\n"
        "assert 'streamlit' not in sys.modules, 'streamlit leaked into runs_view'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_unknown_selection_failure_maps_to_runs_unknown(store, monkeypatch) -> None:
    """Anything unexpected from the selection layer lands in ``runs-unknown``."""

    def explode(**_kwargs):
        raise RuntimeError("mlflow backend exploded")

    monkeypatch.setattr(runs_view, "select_leaderboard_rows", explode)
    view = load_runs_view(tracking_dir=store.tracking_dir)

    assert view.ok is False
    assert view.error_kind == ERROR_KIND_RUNS_UNKNOWN
    assert "make reproduce" in view.error_message


# ---------------------------------------------------------------------------
# 7. AppTest render of the real page (T02) — S03 owns the real-store browser
#    render; these are fixture-level renders against tmp stores.
# ---------------------------------------------------------------------------

PAGES = PROJECT_ROOT / "app" / "pages"
RUNS_PAGE = PAGES / "3_Runs.py"


def _page_texts(test) -> str:
    """Every readable text on the rendered page, joined for assertions."""
    element_lists = [
        test.markdown, test.text, test.caption, test.error, test.warning,
        test.info, test.code, test.title, test.header, test.subheader,
        test.json,
    ]
    chunks = [str(el.value) for el_list in element_lists for el in el_list]
    for df_el in test.dataframe:
        try:
            chunks.append(df_el.value.to_csv(index=False))
        except Exception:  # noqa: BLE001 - render/text bookkeeping only
            chunks.append(str(df_el.value))
    return "\n".join(chunks)


def _run_runs_page(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]):
    """Render the real 3_Runs.py with the given env (mirrors the S07
    ``_run_page`` helper: monkeypatched env, AppTest, undo in ``finally``).
    """
    import streamlit.testing.v1 as st_testing

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    test = st_testing.AppTest.from_file(str(RUNS_PAGE), default_timeout=180)
    try:
        test.run()
    finally:
        monkeypatch.undo()
    return test


class TestRunsPageRenderPopulated:
    """A populated tmp store renders the ranked table and drilldowns."""

    def test_zero_exceptions_top_model_provenance_and_expanders(
        self, store, monkeypatch
    ) -> None:
        _log_run(store, model_type="m-second", model_name="Second", roc_auc=0.93)
        _log_run(store, model_type="m-third", model_name="Third", roc_auc=0.92)
        _log_run(store, model_type="m-first", model_name="First", roc_auc=0.94)

        test = _run_runs_page(
            monkeypatch, {RUNS_TRACKING_DIR_ENV: str(store.tracking_dir)}
        )

        assert test.exception == [], str(test.exception)
        texts = _page_texts(test)
        # Top-ranked model appears; provenance line is present; the friendly
        # params/metrics/artifacts drilldown sections render per row.
        assert "First" in texts
        assert "run kind `final`" in texts
        assert "model(s) ranked from 3 candidate run(s)" in texts
        assert len(test.expander) >= 3
        # Drilldown sections render: params, metrics, artifacts per row.
        assert "Hyperparameters" in texts
        assert "Artifacts:" in texts

    def test_run_ids_render_in_drilldown(self, store, monkeypatch) -> None:
        run_id = _log_run(
            store, model_type="lda", model_name="LDA", roc_auc=0.90
        )

        test = _run_runs_page(
            monkeypatch, {RUNS_TRACKING_DIR_ENV: str(store.tracking_dir)}
        )

        assert test.exception == []
        code_values = " ".join(str(c.value) for c in test.code)
        assert run_id in code_values
        # ROC-AUC formatted to 4dp appears in the ranked table.
        texts = _page_texts(test)
        assert "0.9000" in texts


class TestRunsPageRenderEmptyStore:
    """An empty store renders the friendly message, never a stack trace."""

    def test_zero_exceptions_and_make_reproduce_hint(self, store, monkeypatch) -> None:
        # `store` exists but has zero runs: the store-empty named state.
        test = _run_runs_page(
            monkeypatch, {RUNS_TRACKING_DIR_ENV: str(store.tracking_dir)}
        )

        assert test.exception == []
        texts = _page_texts(test)
        assert "make reproduce" in texts
        assert "store-empty" in texts
        assert "Traceback" not in texts
        # The attempted store path is shown, mirroring 2_Predict's pattern.
        assert str(store.tracking_dir.name) in texts or "sqlite:///" in texts


class TestRunsPageRenderMissingStore:
    """A missing store directory renders the named store-missing message."""

    def test_zero_exceptions_and_friendly_message(
        self, tmp_path, monkeypatch
    ) -> None:
        missing = tmp_path / "no-such-mlruns"
        test = _run_runs_page(
            monkeypatch, {RUNS_TRACKING_DIR_ENV: str(missing)}
        )

        assert test.exception == []
        texts = _page_texts(test)
        assert "make reproduce" in texts
        assert "store-missing" in texts
        assert "Traceback" not in texts
