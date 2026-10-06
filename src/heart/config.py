"""Central project configuration.

Path resolution is anchored to the repository root (the parent of ``src/``)
so that every slice addresses the same locations regardless of the current
working directory. Keep runtime constants here rather than scattering magic
numbers and paths through the pipeline.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Global reproducibility settings
# ---------------------------------------------------------------------------

#: Single seed threaded through every split, sampler, and model for the
#: whole project. Changing it invalidates published leaderboard numbers.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

#: Repository root: ``<repo>/src/heart/config.py`` -> parents[2] == <repo>.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Top-level data directory holding raw and processed artifacts.
DATA_DIR: Path = PROJECT_ROOT / "data"

#: Immutable source data (pinned dataset copy / download cache).
RAW_DATA_DIR: Path = DATA_DIR / "raw"

#: Derived, regenerable artifacts (splits, engineered features).
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

#: Human-inspectable markdown/table reports produced by the pipeline.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

#: Application code (Streamlit dashboard / prediction app).
APP_DIR: Path = PROJECT_ROOT / "app"

#: Ad-hoc experiment scripts and notebooks.
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"


def ensure_project_dirs() -> list[Path]:
    """Create the standard data/report directories if they are missing.

    Returns the list of directories that were created so callers can log
    what changed. Idempotent: existing directories are left untouched.
    """
    created: list[Path] = []
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created
