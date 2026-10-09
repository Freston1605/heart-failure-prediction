#!/usr/bin/env bash
# One-command reproduction entry point (S08/T03).
#
# From a clean checkout of this repository, this script:
#   1. creates an isolated virtual environment (never reuses the dev env),
#   2. installs the project with its fully pinned dependency set,
#   3. verifies dataset integrity (aborts on digest/schema/split drift),
#   4. runs the full classical battery + hyperparameter tuning (seeded),
#   5. runs the S06 selection flow, annotating the winner's run, and
#   6. regenerates reports/leaderboard.md from the freshly recorded runs.
#
# Usage:
#   ./scripts/reproduce.sh               # full pipeline
#   REPRO_VENV=/tmp/repro-venv ./scripts/reproduce.sh
#
# Exit codes: 0 success; propagates the nonzero exit code of the first
# failing step (2/3 = dataset integrity failure from scripts/verify_dataset.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PATH="${REPRO_VENV:-$PROJECT_ROOT/.repro-venv}"

step() { printf '\n== %s ==\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Fresh, isolated interpreter
# ---------------------------------------------------------------------------
step "1/5 fresh virtual environment at $VENV_PATH"
if [ ! -x "$VENV_PATH/bin/python" ]; then
    python3 -m venv "$VENV_PATH" || {
        echo "FAIL: could not create the virtual environment at $VENV_PATH." >&2
        exit 1
    }
fi
PYTHON="$VENV_PATH/bin/python"

PY_VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VERSION" in
    3.11|3.12|3.13|3.14) ;;
    *) echo "FAIL: unsupported Python $PY_VERSION (need 3.11-3.14)." >&2; exit 1 ;;
esac

"$PYTHON" -m pip install --quiet --upgrade pip

step "2/5 install pinned dependencies"
# The pinned wheel versions in pyproject.toml are the only accepted set; a
# resolver conflict here is a failure to surface, not to paper over.
"$PYTHON" -m pip install --quiet -e ".[ml,dev]"
"$PYTHON" - <<'PY'
import numpy, pandas, scipy, sklearn, mlflow, optuna, xgboost
print("pinned stack:", numpy.__version__, pandas.__version__, scipy.__version__,
      sklearn.__version__, mlflow.__version__, optuna.__version__, xgboost.__version__)
PY

step "3/5 dataset integrity check"
# verify_dataset.py exits 2 (raw) or 3 (splits) on drift and prints the reason;
# set -e propagates it so the pipeline stops before any training happens.
"$PYTHON" scripts/verify_dataset.py

step "4/5 full classical battery + selection"
PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" -m heart.models.run_battery \
    --tracking-dir "$PROJECT_ROOT/experiments/mlruns"
PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" -m heart.eval.selection

step "5/5 regenerate reports/leaderboard.md"
PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" -m heart.reporting.leaderboard

echo
echo "Reproduction finished. Compare the regenerated reports/leaderboard.md"
echo "against the published numbers recorded in reports/reproduction.md (T04)."
