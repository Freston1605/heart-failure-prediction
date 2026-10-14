# Environment Variable Audit

Project: heart-failure-prediction (branch milestone/M007)
Date: 2026-10-14
Mode: yaml-step — env-audit / step `audit-var` (iterated over `env-inventory.md`)

Overall: the app reads env vars only for **optional operator seams and logging
tuning**; every one has a safe fallback, so none is *required* to boot. The only
drift is **three undocumented path-override seams**. CI uses none (by design).

---

## DATA_METHODS_DIR
- Code refs: `app/lib/data_methods_view.py:88` (const `DATA_METHODS_DIR_ENV="DATA_METHODS_DIR"`), `:168` (`os.environ.get(...)`); set/reset in `tests/test_data_methods_transformations.py:74,76,82,113,114,120,122`
- Documented: no — gap
- CI: no
- Required / optional: optional (override seam for the Data Methods report dir)
- Has a default: yes — unset → falls back to the committed data-methods dir
- Drift: **undocumented**
- Recommendation: document in `.env.example`/README as an optional override (used for error-path demos/tests).

## container
- Code refs: `src/heart/gpu/device_check.py:361` (`os.environ.get("container")`)
- Documented: no (third-party provided)
- CI: no
- Required / optional: informational — set by the container/init runtime, not the project
- Has a default: n/a (presence check only)
- Drift: none (not project-owned; skip)
- Recommendation: none — runtime marker outside project control.

## EXPLORE_DATA_PATH
- Code refs: `app/lib/plots.py:174` (`os.environ.get("EXPLORE_DATA_PATH")`)
- Documented: no — gap
- CI: no
- Required / optional: optional (Explore dataset-path override, "for error-path demos")
- Has a default: yes — unset → committed `data/raw/heart.csv`
- Drift: **undocumented**
- Recommendation: document in `.env.example` as an optional override.

## HEART_LOG_FORMAT
- Code refs: `src/heart/observability.py:72` (`os.environ.get("HEART_LOG_FORMAT", "text")`)
- Documented: yes — README.md:130,134
- CI: no
- Required / optional: optional
- Has a default: yes — `text`
- Drift: none
- Recommendation: none.

## HEART_LOG_LEVEL
- Code refs: `src/heart/observability.py:71` (`os.environ.get("HEART_LOG_LEVEL", "INFO")`)
- Documented: yes — README.md:131,135
- CI: no
- Required / optional: optional
- Has a default: yes — `INFO`
- Drift: none
- Recommendation: none.

## MKL_NUM_THREADS
- Code refs: `tests/conftest.py:24` (`os.environ.setdefault(_var, "1")`)
- Documented: no (test-harness internal)
- CI: no (runs under the test job implicitly)
- Required / optional: test-harness-only (BLAS thread cap, P002)
- Has a default: yes — set to `1` by conftest if unset
- Drift: none (intentional, test-only)
- Recommendation: none.

## MLFLOW_UI_PORT
- Code refs: `app/pages/3_Runs.py:45` (`os.environ.get("MLFLOW_UI_PORT", "5000")`), `scripts/mlflow_ui.py:50` (const `PORT_ENV_VAR`), `:184` (`os.environ.get(PORT_ENV_VAR, "")`)
- Documented: yes — README.md:113,116,178,181
- CI: no
- Required / optional: optional
- Has a default: yes — `5000`
- Drift: none
- Recommendation: none.

## MPLBACKEND
- Code refs: `tests/conftest.py:32` (`os.environ.setdefault("MPLBACKEND", "Agg")`)
- Documented: no (test-harness internal)
- CI: no
- Required / optional: test-harness-only (headless PNG backend)
- Has a default: yes — `Agg`
- Drift: none (intentional, test-only)
- Recommendation: none.

## NUMEXPR_NUM_THREADS
- Code refs: `tests/conftest.py:25` (`os.environ.setdefault(_var, "1")`)
- Documented: no (test-harness internal)
- CI: no
- Required / optional: test-harness-only (BLAS thread cap)
- Has a default: yes — `1`
- Drift: none (intentional, test-only)
- Recommendation: none.

## OMP_NUM_THREADS
- Code refs: `tests/conftest.py:22` (`os.environ.setdefault(_var, "1")`)
- Documented: no (test-harness internal)
- CI: no
- Required / optional: test-harness-only (BLAS thread cap, P002)
- Has a default: yes — `1`
- Drift: none (intentional, test-only)
- Recommendation: none.

## OPENBLAS_NUM_THREADS
- Code refs: `tests/conftest.py:23` (`os.environ.setdefault(_var, "1")`)
- Documented: no (test-harness internal)
- CI: no
- Required / optional: test-harness-only (BLAS thread cap)
- Has a default: yes — `1`
- Drift: none (intentional, test-only)
- Recommendation: none.

## PYTHONPATH
- Code refs: `Makefile` (make targets), `README.md:135` (`PYTHONPATH=src make leaderboard`)
- Documented: yes — README.md:135
- CI: no (the workflow installs the package / sets sys.path itself)
- Required / optional: optional (standard Python import-path var for `make` runs)
- Has a default: n/a (standard interpreter behavior)
- Drift: none
- Recommendation: none.

## RUNS_TRACKING_DIR
- Code refs: `app/lib/runs_view.py:83` (const `RUNS_TRACKING_DIR_ENV="RUNS_TRACKING_DIR"`), `:170` (`os.environ.get(...)`)
- Documented: no — gap
- CI: no
- Required / optional: optional (override seam for the MLflow runs-tracking dir)
- Has a default: yes — unset → the default tracking dir from `mlflow_store`
- Drift: **undocumented**
- Recommendation: document in `.env.example`/README as an optional override.
