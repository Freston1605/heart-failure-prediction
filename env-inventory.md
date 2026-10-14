# Environment Variable Inventory

Project: heart-failure-prediction (branch milestone/M007)
Date: 2026-10-14
Mode: yaml-step — env-audit / step `inventory`

Scope:
- **code**: `app/`, `src/`, `scripts/`, `Makefile`, `tests/` (Python + Make).
- **docs**: `README.md` (no `docs/` dir, no `.env.example` present).
- **ci**: `.github/workflows/` (uses **no** `env:` blocks and **no** `secrets.*`).

Legend: `y` = referenced, `n` = not referenced.

## Variables (alphabetical)

- `container` — code: y, docs: n, ci: n
- `DATA_METHODS_DIR` — code: y, docs: n, ci: n
- `EXPLORE_DATA_PATH` — code: y, docs: n, ci: n
- `HEART_LOG_FORMAT` — code: y, docs: y, ci: n
- `HEART_LOG_LEVEL` — code: y, docs: y, ci: n
- `MKL_NUM_THREADS` — code: y, docs: n, ci: n
- `MLFLOW_UI_PORT` — code: y, docs: y, ci: n
- `MPLBACKEND` — code: y, docs: n, ci: n
- `NUMEXPR_NUM_THREADS` — code: y, docs: n, ci: n
- `OMP_NUM_THREADS` — code: y, docs: n, ci: n
- `OPENBLAS_NUM_THREADS` — code: y, docs: n, ci: n
- `PYTHONPATH` — code: y, docs: y, ci: n
- `RUNS_TRACKING_DIR` — code: y, docs: n, ci: n

## Notes for the audit step

- **`container`** — lowercase runtime marker set by the container/init
  runtime (`src/heart/gpu/device_check.py:361`), not a project-owned config
  var. Third-party provided; audit as informational.
- **`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MPLBACKEND`** — set by `tests/conftest.py`
  (`os.environ.setdefault(...)`) to cap BLAS threads / force the Agg backend.
  Test-harness-only; not read for app config.
- **`DATA_METHODS_DIR`, `EXPLORE_DATA_PATH`, `RUNS_TRACKING_DIR`** — operator
  path-override seams in the app views; **not documented in README** → drift
  candidate (undocumented).
- **`MLFLOW_UI_PORT`, `HEART_LOG_LEVEL`, `HEART_LOG_FORMAT`** — documented in
  README. No drift.
- **`PYTHONPATH`** — standard Python var, used by `Makefile`/README `make`
  targets. Documented.
- **CI** — `.github/workflows/ci.yml` contains zero `env:`/`secrets.*`
  references, so every project var is `ci: n` by design (none are needed to
  build/test the pinned stack).
- **`MLFLOW_TRACKING_URI`** — intentionally **absent**: the tracking URI is set
  programmatically via `heart.tracking.mlflow_store.configure_tracking()`
  (MEM017), never via env. Not a drift gap.
