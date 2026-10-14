# Environment Variable Drift Report

Project: heart-failure-prediction (branch milestone/M007)
Date: 2026-10-14
Mode: yaml-step — env-audit / step `report`
Inputs: `env-inventory.md`, `env-audit.md`

## Bottom line

Low drift. The app reads env vars only for **optional operator path-seams** and
**logging tuning**; every one has a safe default, so **nothing is required to
boot** (local or Cloud). CI uses no env vars at all (by design). The only real
gap was **three undocumented path-override seams** — now closed by a new
`.env.example`.

## Drift by category

### Undocumented (referenced in code, not in docs) — FIXED
| Variable | Where | Risk | Action taken |
|----------|-------|------|--------------|
| `EXPLORE_DATA_PATH` | `app/lib/plots.py:174` | Low — optional override, falls back to committed `data/raw/heart.csv` | documented in `.env.example` |
| `DATA_METHODS_DIR` | `app/lib/data_methods_view.py:88,168` | Low — optional override, falls back to committed dir | documented in `.env.example` |
| `RUNS_TRACKING_DIR` | `app/lib/runs_view.py:83,170` | Low — optional override, falls back to `mlflow_store` default | documented in `.env.example` |

All three are operator-only seams (a web user cannot set them) with safe
fallbacks — no runtime-failure risk, just a documentation gap.

### unused-in-code (documented but never read) — none
Every inventoried variable is referenced in code. No dead documentation.

### missing-in-ci — none that matter
`.github/workflows/ci.yml` has zero `env:`/`secrets.*` references. Every project
var is `ci: n` **by design** — the CI jobs (test/build/deploy-manifest) install
and smoke the pinned stack without needing any of these runtime seams. Not a gap.

### conflicting-default — none
No variable has conflicting defaults across code vs docs. `MLFLOW_UI_PORT`
(5000), `HEART_LOG_LEVEL` (INFO), `HEART_LOG_FORMAT` (text) match README.

## Defaults / required-ness review

- **No variable is required** — all are optional with safe fallbacks. No
  runtime-failure risk from an unset var.
- **Test-only vars** (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `MPLBACKEND`) are set via
  `os.environ.setdefault` in `tests/conftest.py` — intentional (BLAS caps /
  headless backend), correctly not app config. No action.
- **`container`** — lowercase runtime marker set by the container/init runtime
  (`src/heart/gpu/device_check.py:361`); not project-owned. No action.
- **`PYTHONPATH`** — standard interpreter var used by `make` targets; documented
  in README. No action.
- **`MLFLOW_TRACKING_URI`** — correctly absent from env; set programmatically by
  `configure_tracking()` (MEM017). Not a gap.

## Recommended `.env.example` diff (applied)

Created `.env.example` (it was absent; `.gitignore` already un-ignores it via
`!.env.example`) documenting the optional seams and logging/MLflow-UI knobs with
their defaults and a note that the tracking URI is programmatic. Diff summary:

```diff
+ # all vars optional; safe fallbacks when unset
+ #EXPLORE_DATA_PATH=...        (default data/raw/heart.csv)
+ #DATA_METHODS_DIR=...         (default committed data-methods dir)
+ #RUNS_TRACKING_DIR=...        (default mlflow_store dir)
+ #HEART_LOG_LEVEL=INFO
+ #HEART_LOG_FORMAT=text
+ #MLFLOW_UI_PORT=5000
```

## Recommendations

- Keep `.env.example` in sync when a new `os.environ.get("...")` seam is added
  (the three path-seams were invisible until this audit).
- If the deployed app ever needs a non-default log format/level, set
  `HEART_LOG_FORMAT=json` in the Streamlit Cloud secrets UI (not in the repo).
