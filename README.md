# Heart Failure Prediction Portfolio

A reproducible comparison of statistical, classical-ML, and neural models for
predicting heart disease from the **fedesoriano combined 5-site clinical
dataset** (918 rows), shipped as a live Streamlit dashboard and prediction app.

The central claim — "model X is best" — must survive repeated cross-validation
and paired significance testing against both the runner-up and a plainly named
logistic-regression baseline. A stranger with a clean environment must be able
to regenerate the leaderboard with one documented command.

## Repository layout

```
.
├── pyproject.toml          # pinned dependency manifest + tool config
├── README.md
├── app/                    # Streamlit dashboard + prediction app
│   ├── Home.py
│   ├── lib/                # non-Streamlit contract modules (runs_view, data_methods_view)
│   └── pages/              # sidebar pages (1_Explore.py, 3_Runs.py, 4_Data_Methods.py)
├── src/heart/              # importable package (src layout)
│   ├── config.py           # paths, RANDOM_SEED, shared constants
│   ├── reporting/data_methods/  # four-stage pre-split EDA engine + report renderer
│   └── ...
├── experiments/            # ad-hoc experiment scripts / notebooks
├── reports/                # generated, human-inspectable reports
├── tests/                  # pytest suite
└── data/
    ├── raw/                # pinned dataset copy / download cache
    └── processed/          # derived, regenerable artifacts (splits, features)
```

## Setup

Requires Python >= 3.11 (developed and verified on CPython 3.14).

Install the core classical/scikit-learn stack plus test tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional groups (used by later slices):

```bash
pip install -e ".[ml]"    # mlflow, optuna, xgboost, shap
pip install -e ".[app]"   # streamlit, matplotlib, seaborn
```

> `uv` is not installed in this environment; the commands above use the
> standard-library `venv` + `pip` path. If you have `uv`, `uv sync` works with
> the same `pyproject.toml`.

## Verify the environment

```bash
pytest                                  # test suite (exit 0)
python -c "import heart; print(heart.__version__)"
```

## Reproduce the leaderboard from a clean environment

One command regenerates `reports/leaderboard.md` from a clean checkout —
fresh virtualenv, fully pinned dependencies, dataset integrity gate, full
seeded battery + selection, leaderboard re-render:

```bash
make reproduce        # ~20 min; override the venv path with REPRO_VENV=/path
```

The run aborts before any training if the dataset integrity check fails
(exit 2: raw dataset drift; exit 3: split drift) — it never silently
downloads substitute data. `make check-data` runs the integrity gate alone,
`make leaderboard` re-renders the report from an existing tracking store,
and `make clean-repro` deletes the reproduction venv.

The comparison of the regenerated numbers against the published ones (zero
numeric deviation; only the generation timestamp and MLflow run IDs differ)
is recorded in [`reports/reproduction.md`](reports/reproduction.md), and
`tests/test_reproduction_contract.py` holds it as an executable contract.
The same pinned stack is available containerized via
`containers/Containerfile` (ROCm base image for GPU experiments).

## Viewing the training runs

Every training run (battery runs, tuning trials, the final per-model run) is
logged to the one local MLflow tracking store, `experiments/mlruns/mlflow.db`.
The store stays **gitignored** (it is evidence, not source) and is regenerated
from scratch by `make reproduce`. There is one entry point for both viewing
paths, `scripts/mlflow_ui.py`, exposed as two make targets:

```bash
make mlflow-ui       # launch the MLflow UI against the local store
make mlflow-status   # print store status + run counts, no server started
```

- `make mlflow-ui` prefights the store and launches MLflow's own UI in the
  foreground. When the store is missing or contains zero logged runs, the
  launcher refuses to open a blank UI: it prints the named problem plus the
  populate hint (`make reproduce`) and exits with code `2` instead of
  starting a server that would show nothing.
- `make mlflow-status` is the same preflight without any server: it prints a
  one-line status (e.g. `tracking store ok: ... — 12 run(s), 7 final`) and
  exits `0` on a populated store, or `2` with the populate hint when the
  store is missing or empty.

Exit codes are machine-readable (`0` ok, `2` missing/empty store, `10`
unexpected error) so `make` targets and CI can branch on them.

The UI default port is `5000`; override it with `MLFLOW_UI_PORT`:

```bash
MLFLOW_UI_PORT=7113 make mlflow-ui
```

The UI is then served at `http://localhost:7113` (with the default port:
`http://localhost:5000`).

A third viewer lives in the Streamlit app: `streamlit run app/Home.py` and
pick the **Runs** page in the sidebar — it mirrors
[`reports/leaderboard.md`](reports/leaderboard.md) and refuses the same
friendly way (naming `make reproduce`) when the store has no runs yet. It is
part of this repository's viewer set alongside the MLflow UI above.
## Operating the app & pipeline (observability)

Events are structured JSON lines on **stderr** of the app / pipeline processes.
`HEART_LOG_FORMAT=json` (default `text` locally) selects the schema'd stream;
`HEART_LOG_LEVEL` (default `INFO`) selects verbosity.

```bash
HEART_LOG_FORMAT=json streamlit run app/Home.py          # one JSON record per line
HEART_LOG_LEVEL=DEBUG PYTHONPATH=src make leaderboard    # debug verboseness
```

Record schema: `ts, level, service, logger, event, status, message, ...fields`.
Key events: `app.artifact_load`, `app.predict_submit`, `app.dataset_load`,
`battery.complete`, `leaderboard.complete`; metrics ride along as
`event:"metric"` records (`heart.serving.predict_latency_ms`,
`heart.serving.predict_count{outcome}`, `heart.battery.duration_ms`, ...).

Healthy ranges: warm page renders < 1.5 s, artifact load p95 < 100 ms
(reference CPU), predicting p95 < 250 ms, battery `n_failed == 0`.
Alert on: any `status:error` in `app.*` events; leaderboard missing-experiment
failure; `served` share of `predict_count` below 70%.

Privacy rule (enforced): patient-style feature values are never logged —
`log_event` refuses field keys matching clinical columns.

## Viewing the training runs

Every training run (battery runs, tuning trials, the final per-model run) is
logged to the one local MLflow tracking store, `experiments/mlruns/mlflow.db`.
The store stays **gitignored** (it is evidence, not source) and is regenerated
from scratch by `make reproduce`. There is one entry point for both viewing
paths, `scripts/mlflow_ui.py`, exposed as two make targets:

```bash
make mlflow-ui       # launch the MLflow UI against the local store
make mlflow-status   # print store status + run counts, no server started
```

- `make mlflow-ui` prefights the store and launches MLflow's own UI in the
  foreground. When the store is missing or contains zero logged runs, the
  launcher refuses to open a blank UI: it prints the named problem plus the
  populate hint (`make reproduce`) and exits with code `2` instead of
  starting a server that would show nothing.
- `make mlflow-status` is the same preflight without any server: it prints a
  one-line status (e.g. `tracking store ok: ... — 12 run(s), 7 final`) and
  exits `0` on a populated store, or `2` with the populate hint when the
  store is missing or empty.

Exit codes are machine-readable (`0` ok, `2` missing/empty store, `10`
unexpected error) so `make` targets and CI can branch on them.

The UI default port is `5000`; override it with `MLFLOW_UI_PORT`:

```bash
MLFLOW_UI_PORT=7113 make mlflow-ui
```

The UI is then served at `http://localhost:7113` (with the default port:
`http://localhost:5000`).

A third viewer lives in the Streamlit app: `streamlit run app/Home.py` and
pick the **Runs** page in the sidebar — it mirrors
[`reports/leaderboard.md`](reports/leaderboard.md) and refuses the same
friendly way (naming `make reproduce`) when the store has no runs yet. It is
part of this repository's viewer set alongside the MLflow UI above.

## Data and Methods report

`reports/data-methods.md` (part A) is the four-stage **pre-split** EDA over the
raw dataset, plus a takeaways checklist. Every stage runs before any train/test
split, so no finding can leak target knowledge into the pipeline:

1. **Data hygiene** — schema, dataset provenance (pinned-digest verification),
   missingness (declared zero-as-missing columns), near-duplicate rows,
   impossible values, missingness-pattern clustering.
2. **Univariate** — per-column skewness, outliers (IQR fences), histograms and
   categorical bars.
3. **Bivariate** — feature/target chi-square tests, high-correlation pairs,
   VIF, correlation heatmap.
4. **Pre-split assumptions** — outlier-policy comparison (mean/median/empirical
   logit) and the resulting imputation decision.

**Regenerate with one command:**

```bash
PYTHONPATH=src python -m heart.reporting.data_methods
```

Two runs produce byte-identical output (seeded, deterministic). The machine
artifacts land in `reports/data-methods/` (`meta.json`, four stage JSONs, PNG
figures) and the rendered report is `reports/data-methods.md`. Do not
hand-edit either — they are generated.

**SHA-256 provenance lock:** `reports/data-methods/manifest.json` pins the
digest of every artifact. Any tampering (or an edited report) fails
`verify_manifest` at regeneration time and fails the provenance tests
(`tests/test_data_methods_provenance.py`) — the artifacts can only be replaced
by rerunning the command above.

**View it in the app:** run `streamlit run app/Home.py` and pick **Data
Methods** in the sidebar. The page renders exactly the generated artifacts
under `reports/data-methods/`, with every table and figure caption citing its
artifact path. When the artifacts are missing (or the provenance check fails)
the page shows a named friendly empty-state — what is missing plus the
regeneration command above — never a stack trace.

**Transformation sections (parts B-F):** the same page also renders the
post-split treatment audit from the same provenance-verified artifacts —
`treatment_audit.json`, `treatment_plan.json`, and
`treatment_gate_verdicts.json` — with no statistic recomputed in the page:

- **Part B — Per-model treatment audit** — per-model audit (pattern, split,
  zero policy) from `treatment_audit.json`.
- **Part C — Logistic-regression preparation methodology** — the six-checklist
  LR narrative.
- **Part D — Pre-declared class-imbalance ladder** — the imbalance ladder
  with final dispositions from `treatment_plan.json`.
- **Part E — Validation and calibration** — validation/calibration narrative.
- **Part F — Severity-gate verdicts** — the gate-verdicts table
  (`candidate_key`, delta, significance reading, standing, verdict) built
  verbatim from `treatment_gate_verdicts.json` outcomes (48 pairs, all
  rejected, split v1). The named friendly empty-state above still covers
  missing/unreadable/corrupt treatment artifacts instead of any stack trace.

## Reproducibility conventions

- All randomness flows from `heart.config.RANDOM_SEED` (currently `42`).
- Splits are created **before** any data-dependent transform so that no test
  row can influence a fitted transformer.
- Generated artifacts live under `data/processed/`; human-readable findings
  live under `reports/`.

## Status

Milestone M001 in progress (S08: free-tier deployment + reproducibility
package). The app is live per `reports/deployment.md`; the clean-checkout
leaderboard reproduction proof is in `reports/reproduction.md`.
