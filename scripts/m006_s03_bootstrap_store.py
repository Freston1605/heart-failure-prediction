"""D037 one-time bootstrap: copy the main checkout's MLflow store into this worktree.

Runs from the worktree root. The main checkout's store is referenced only via
its relative path (``../../experiments/mlruns``) and is never modified.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

src_dir = Path("../../experiments/mlruns")
dst = Path("experiments/mlruns")

if __name__ == "__main__":
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "mlflow.db", dst / "mlflow.db")
    shutil.copytree(src_dir / "artifacts", dst / "artifacts", dirs_exist_ok=True)
    print(f"copied {src_dir} -> {dst}")
    sys.exit(0)
