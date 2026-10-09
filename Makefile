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

.PHONY: help reproduce check-data leaderboard clean-repro
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

# Remove the reproduction venv created by `make reproduce` (not the mlflow
# run store, which is evidence).
clean-repro:  ## Delete the reproduction virtual environment
	if [ -d "$(VENV)" ]; then rm -rf "$(VENV)"; fi
