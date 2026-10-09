# One-command reproduction entry points (S08/T03).
#
# The canonical "stranger reproduces the leaderboard from a clean checkout"
# flow is `make reproduce`, which delegates to scripts/reproduce.sh (fresh
# venv, pinned install, dataset integrity gate, full seeded battery +
# selection, leaderboard regeneration). All targets are phony: nothing here
# should be confused with a generated file.

PYTHON ?= python3
VENV ?= .repro-venv
REPRO_SCRIPT := scripts/reproduce.sh

.PHONY: help reproduce check-data leaderboard mlflow-ui mlflow-status clean-repro
.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

reproduce:  ## Full reproducible pipeline: fresh venv -> integrity check -> battery -> selection -> leaderboard
	./$(REPRO_SCRIPT)

# Integrity-only check, usable in the developer environment without the
# reproduction venv. Requires the package importable (PYTHONPATH=src).
check-data:  ## Verify raw dataset + committed split digests; non-zero exit on drift
	PYTHONPATH=src $(PYTHON) scripts/verify_dataset.py

# Regenerate the leaderboard directly from an existing tracking store.
# This is the regenerate step alone; it does NOT re-train anything.
# aborts with a named error if required runs or annotations are missing.
leaderboard:  ## Re-render reports/leaderboard.md from the recorded MLflow runs
	PYTHONPATH=src $(PYTHON) -m heart.reporting.leaderboard

# View the recorded training runs. The UI is pointed at the one SQLite
# tracking store (experiments/mlruns/mlflow.db); it preflights first so an
# empty or missing store is named loudly instead of opening a blank server.
# Override the port with MLFLOW_UI_PORT (default 5000).
mlflow-ui:  ## Launch the MLflow UI against the local tracking store
	PYTHONPATH=src $(PYTHON) scripts/mlflow_ui.py --launch

mlflow-status:  ## Report tracking-store status and run counts (no server)
	PYTHONPATH=src $(PYTHON) scripts/mlflow_ui.py --status

# Remove the reproduction venv created by `make reproduce` (not the mlflow
# run store, which is evidence).
clean-repro:  ## Delete the reproduction virtual environment
	if [ -d "$(VENV)" ]; then rm -rf "$(VENV)"; fi
