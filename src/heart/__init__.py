"""Heart failure prediction portfolio.

Reproducible pipeline for the fedesoriano UCI combined 5-site clinical
dataset: data ingestion, quality audit, leakage-safe splits, model
benchmarking, and a served prediction app.
"""

from heart.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REPORTS_DIR,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RANDOM_SEED",
    "RAW_DATA_DIR",
    "REPORTS_DIR",
]
