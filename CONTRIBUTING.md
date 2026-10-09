# Contributing

## Local checks (what CI runs)

| Check | Command | Notes |
|---|---|---|
| Install | `pip install -e ".[dev,ml,app]"` | Extras are exact-pinned in `pyproject.toml`. |
| Dataset gate | `make check-data` | Fails non-zero on raw-digest or split-manifest drift. |
| Tests | `pytest` | Config lives in `pyproject.toml` (`addopts=-q`, `pythonpath=src`). ~85s locally. |
| Build | `pipx run build` or `python -m build` | Wheel/sdist packaging check. |

Torch-dependent MLP tests self-skip when torch is not installed — that's expected, not a failure.

## What CI does

GitHub Actions (`.github/workflows/ci.yml`) runs on every PR to `main` and every push to `main`:

- **test** — installs the package with dev/ml/app extras (pip cache keyed on `pyproject.toml`), runs the dataset integrity gate, then the full test suite. Requires Python 3.12 on the runner.
- **build** — verifies the project still builds a wheel/sdist.

There is no lint stage yet — the project has no linter configured. Add a job to `.github/workflows/ci.yml` when one is adopted.

`app/requirements.txt` is not used by CI: it is the deliberately narrower dependency manifest for the Streamlit Community Cloud deploy.

## Debugging a red build

1. Open the failing job log via the PR's checks ( gear → "Details"), or:
   `gh run view <run-id> --log-failed`
2. Common causes:
   - **Install step fails with `tomllib.TOMLDecodeError`** — malformed `pyproject.toml`; the TOML must parse before anything else runs.
   - **Dataset gate fails** — `data/raw/heart.csv` or `data/processed/splits/v1/` diverged from the pinned digests. Do not edit generated files; re-run the producer pipeline.
   - **Test failure** — run locally with `pytest -o addopts="" -v <file>` for per-test names (the `addopts=-q` default hides them).
3. First-party `DeprecationWarning`s from `heart.*` are errors by config (`filterwarnings` in `pyproject.toml`) — fix the call site, don't suppress.
