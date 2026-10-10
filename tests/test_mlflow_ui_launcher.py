"""Tests for the MLflow UI launcher contract (M002/S01/T01).

The unit under test is :mod:`scripts.mlflow_ui`, the one documented entry
point for "viewing the training runs". Its observable surface is the printed
status line plus machine-readable exit codes:

* **Preflight states** — :func:`mlflow_ui.preflight` reports missing store,
  empty store, and populated store as *data* (a ``StoreStatus``), never as
  exceptions, and a missing experiment is its own reported state.
* **Port resolution** — :func:`mlflow_ui.resolve_port` maps
  ``MLFLOW_UI_PORT`` to an integer, defaulting to 5000 and rejecting
  non-integer and out-of-range values with a message naming the variable.
* **Launch refusal** — :func:`mlflow_ui.launch` refuses (exit 2) without
  spawning any server when the store is missing or has zero runs, and names
  the populate command on stderr.
* **CLI exit codes** — :func:`mlflow_ui.main` maps ``--status`` to 0
  (populated) or 2 (refusal naming ``make reproduce``), and an unexpected
  error (bad port value) to 10.

Every store-backed test runs against a fresh ``tmp_path`` SQLite store, so
the suite never touches the repository's gitignored ``experiments/mlruns``
directory. ``subprocess.run`` is always mocked: no server is ever spawned
and no network traffic occurs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from heart.tracking.mlflow_store import (
    DEFAULT_EXPERIMENT,
    configure_tracking,
    sqlite_uri,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "mlflow_ui.py"


def _load_mlflow_ui():
    """Load the launcher script as a module, PYTHONPATH-independent."""
    spec = importlib.util.spec_from_file_location("mlflow_ui_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: the script's @dataclass resolves
    # cls.__module__ via sys.modules at class-creation time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mlflow_ui = _load_mlflow_ui()


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path):
    """An isolated SQLite store + artifact root with the experiment ready."""
    directory = tmp_path / "mlruns"
    config = configure_tracking(tracking_dir=directory)
    return config, directory


def seed_runs(client, experiment_id, n_final, n_other=0):
    """Seed ``n_final`` final-tagged and ``n_other`` trial-tagged runs."""
    run_ids = []
    for kind, count in (("final", n_final), ("trial", n_other)):
        for index in range(count):
            run = client.create_run(
                experiment_id,
                tags={"run_kind": kind, "mlflow.runName": f"run-{kind}-{index}"},
            )
            client.set_terminated(run.info.run_id)
            run_ids.append(run.info.run_id)
    return run_ids


def client_for(directory):
    """A client bound to the tmp store, without relying on global state."""
    return MlflowClient(tracking_uri=sqlite_uri(directory))


# ---------------------------------------------------------------------------
# resolve_port
# ---------------------------------------------------------------------------


class TestResolvePort:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(mlflow_ui.PORT_ENV_VAR, raising=False)
        assert mlflow_ui.resolve_port() == mlflow_ui.DEFAULT_UI_PORT == 5000

    def test_override(self, monkeypatch):
        monkeypatch.setenv(mlflow_ui.PORT_ENV_VAR, "7113")
        assert mlflow_ui.resolve_port() == 7113

    def test_empty_string_env_uses_default(self, monkeypatch):
        monkeypatch.setenv(mlflow_ui.PORT_ENV_VAR, "")
        assert mlflow_ui.resolve_port() == 5000

    @pytest.mark.parametrize(
        ("raw", "fragment"),
        [
            ("abc", "MLFLOW_UI_PORT"),
            ("0", "1..65535"),
            ("65536", "1..65535"),
            ("-1", "1..65535"),
        ],
    )
    def test_rejects_bad_values(self, monkeypatch, raw, fragment):
        monkeypatch.setenv(mlflow_ui.PORT_ENV_VAR, raw)
        with pytest.raises(ValueError, match=fragment):
            mlflow_ui.resolve_port()


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_missing_store_is_reported_not_raised(self, tmp_path):
        status = mlflow_ui.preflight(tmp_path / "absent")
        assert status.store_exists is False
        assert status.ok is False
        assert status.problems  # non-empty
        assert "missing" in status.status_line()
        assert "make reproduce" in status.fix_hint()

    def test_empty_store_has_zero_runs(self, tmp_store):
        config, directory = tmp_store
        status = mlflow_ui.preflight(directory)
        assert status.store_exists is True
        assert status.experiment_found is True
        assert status.n_runs_total == 0
        assert status.n_runs_final == 0
        assert status.ok is False
        assert "zero logged runs" in status.status_line()

    def test_populated_store_counts_final_runs(self, tmp_store):
        config, directory = tmp_store
        client = client_for(directory)
        seed_runs(client, config.experiment_id, n_final=2, n_other=1)
        client.create_run(config.experiment_id, tags={"mlflow.runName": "untagged"})
        status = mlflow_ui.preflight(directory)
        assert status.store_exists is True
        assert status.experiment_found is True
        assert status.ok is True
        assert status.n_runs_total == 4
        assert status.n_runs_final == 2
        assert "4 run(s), 2 final" in status.status_line()

    def test_missing_experiment_is_reported(self, tmp_store, monkeypatch):
        config, directory = tmp_store
        monkeypatch.setattr(
            "heart.tracking.mlflow_store.DEFAULT_EXPERIMENT", "no-such-experiment"
        )
        status = mlflow_ui.preflight(directory)
        assert status.store_exists is True
        assert status.experiment_found is False
        assert status.ok is False
        assert status.problems


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


class TestLaunch:
    def test_refuses_missing_store_without_spawning(self, tmp_path, monkeypatch, capsys):
        directory = tmp_path / "absent"

        def must_not_spawn(cmd, **kwargs):
            raise AssertionError("must not spawn")

        monkeypatch.setattr(mlflow_ui.subprocess, "run", must_not_spawn)
        rc = mlflow_ui.launch(directory)
        assert rc == mlflow_ui.HEART_EXIT["empty_store"]
        captured = capsys.readouterr()
        assert "missing" in captured.out
        assert "make reproduce" in captured.err

    def test_refuses_empty_store_without_spawning(self, tmp_store, monkeypatch, capsys):
        config, directory = tmp_store

        def must_not_spawn(cmd, **kwargs):
            raise AssertionError("must not spawn")

        monkeypatch.setattr(mlflow_ui.subprocess, "run", must_not_spawn)
        rc = mlflow_ui.launch(directory)
        assert rc == mlflow_ui.HEART_EXIT["empty_store"]
        captured = capsys.readouterr()
        assert "zero logged runs" in captured.out
        assert "make reproduce" in captured.err

    def test_populated_store_builds_ui_command(self, tmp_store, monkeypatch, capsys):
        config, directory = tmp_store
        client = client_for(directory)
        seed_runs(client, config.experiment_id, n_final=1)
        captured = {}

        class FakeResult:
            returncode = 0

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            return FakeResult()

        monkeypatch.setattr(mlflow_ui.subprocess, "run", fake_run)
        monkeypatch.setenv(mlflow_ui.PORT_ENV_VAR, "5001")

        rc = mlflow_ui.launch(directory)

        assert rc == mlflow_ui.HEART_EXIT["ok"]
        cmd = captured["cmd"]
        assert cmd[cmd.index("--port") + 1] == "5001"
        assert cmd[cmd.index("--backend-store-uri") + 1] == sqlite_uri(directory)
        expected_root = (directory / "artifacts").resolve().as_uri()
        assert cmd[cmd.index("--default-artifact-root") + 1] == expected_root
        assert "http://localhost:5001" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main (CLI exit codes)
# ---------------------------------------------------------------------------


class TestMain:
    def test_status_populated_exits_zero_with_counts(self, tmp_store, capsys):
        config, directory = tmp_store
        client = client_for(directory)
        seed_runs(client, config.experiment_id, n_final=2, n_other=1)

        rc = mlflow_ui.main(["--status", "--tracking-dir", str(directory)])

        assert rc == mlflow_ui.HEART_EXIT["ok"]
        assert "3 run(s), 2 final" in capsys.readouterr().out

    def test_status_missing_store_exits_two_naming_populate_command(
        self, tmp_path, capsys
    ):
        directory = tmp_path / "absent"

        rc = mlflow_ui.main(["--status", "--tracking-dir", str(directory)])

        assert rc == mlflow_ui.HEART_EXIT["empty_store"]
        out = capsys.readouterr().out
        assert "missing" in out
        assert "make reproduce" in out

    def test_launch_with_bad_port_env_exits_ten(self, tmp_store, monkeypatch, capsys):
        config, directory = tmp_store
        client = client_for(directory)
        seed_runs(client, config.experiment_id, n_final=1)

        def must_not_spawn(cmd, **kwargs):
            raise AssertionError("must not spawn")

        monkeypatch.setattr(mlflow_ui.subprocess, "run", must_not_spawn)
        monkeypatch.setenv(mlflow_ui.PORT_ENV_VAR, "abc")

        rc = mlflow_ui.main(["--launch", "--tracking-dir", str(directory)])

        assert rc == mlflow_ui.HEART_EXIT["unexpected"]
        assert "MLFLOW_UI_PORT" in capsys.readouterr().err
