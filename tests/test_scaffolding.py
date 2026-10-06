"""Scaffolding smoke tests.

These assert the package is importable and that configuration paths resolve
to the repository root. They deliberately contain no ML logic — their only
job is to prove the test framework and the importable package surface
established in T01 work.
"""

from pathlib import Path

import heart
from heart import config


def test_package_importable() -> None:
    assert heart.__version__ == "0.1.0"


def test_project_root_is_repository_root() -> None:
    # pyproject.toml lives at the repository root.
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_standard_directories_are_under_project_root() -> None:
    for directory in (config.DATA_DIR, config.REPORTS_DIR):
        assert isinstance(directory, Path)
        assert config.PROJECT_ROOT in directory.parents
    for directory in (
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.APP_DIR,
        config.EXPERIMENTS_DIR,
    ):
        assert config.PROJECT_ROOT in directory.parents


def test_random_seed_is_fixed() -> None:
    assert config.RANDOM_SEED == 42


def test_ensure_project_dirs_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")

    created_first = config.ensure_project_dirs()
    assert len(created_first) == 3
    created_second = config.ensure_project_dirs()
    assert created_second == []
