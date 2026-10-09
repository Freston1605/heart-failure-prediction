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
├── src/heart/              # importable package (src layout)
│   ├── __init__.py
│   ├── config.py           # paths, RANDOM_SEED, shared constants
│   └── data/               # loaders, schema, quality, splits (S01)
├── app/                    # Streamlit dashboard + prediction app (S07)
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
