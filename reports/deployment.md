# Deployment

App deployment configuration and deploy record for the free-tier host.

## Chosen host: Streamlit Community Cloud (free tier)

Streamlit Community Cloud is the natural free-tier home for this app: it deploys
directly from this git repository, runs `streamlit run <main-file>` natively,
costs nothing, and constrains exactly what this project needs to respect
(repo-based artifact packaging and app-scoped dependencies).

- **Main file:** `app/Home.py` (multipage; the four pages `app/pages/1_Explore.py`,
  `app/pages/2_Predict.py`, `app/pages/3_Runs.py` and
  `app/pages/4_Data_Methods.py` are auto-discovered and served as the
  `/Explore`, `/Predict`, `/Runs` and `/Data_Methods` routes)
- **Requirements location:** `app/requirements.txt` (Streamlit Cloud falls back
  to the main-file directory when the repo root has no `requirements.txt`).
  It is scoped to the app's true needs (pinned core sklearn/pandas stack +
  streamlit) — no dev/ml extras, no seaborn/matplotlib since all plots use
  native `st.bar_chart`/`st.line_chart` (built-in Altair).
- **Python version:** documented via `app/runtime.txt` (`python-3.11`),
  pinned in the deploy dialog's advanced settings; the module layer requires
  `>=3.11` per `pyproject.toml`.
- **Streamlit theming/server config:** `.streamlit/config.toml`, shared with
  local runs.

## Artifact packaging within size limits

Streamlit Community Cloud constraints and where this app sits against them:

| Constraint | Platform limit | This project | Headroom |
|---|---|---|---|
| Res. repository size | 1 GB | ~3 MB (code + `models/heart-winner-v1.pkl` 2.5 MB + `data/raw/heart.csv` 36 KB) | ~99.7% |
| Model artifact (`models/heart-winner-v1.pkl`) | — | 2.5 MB (2,525,551 bytes) | — |
| App dependencies resolved at build | — | 6 pinned wheels (`app/requirements.txt`) | — |

The serving artifact is **tracked by git** (`models/heart-winner-v1.pkl`), which
the deploy-from-git flow requires, and is loaded at runtime by
`app/lib/artifact_loader.py` (validated, 2.5 MB) — it needs no extra upload step
and fits comfortably inside the platform's storage envelope.

## Deploy steps (as run to publish)

The Cloud deploy flow is interactive by design (owner OAuth on
`share.streamlit.io`); the automation surface cannot hold that session, so the
one human-required step is flagged in the task closeout:

1. Repository on GitHub with this worktree's branch pushed (automation commits
   repository state after task completion).
2. Human account action (owner performed 2026-10-07): open
   <https://share.streamlit.io>, sign in with the GitHub owner account, choose
   **New app → From existing repo**, pick `Freston1605/heart-failure-prediction`,
   branch `milestone/M001`, main file path `app/Home.py`, advanced settings
   Python version `3.11` (matching `app/runtime.txt`).
3. Streamlit Cloud clones the repo, installs `app/requirements.txt`, validates
   the tracked 2.5 MB artifact within the 1 GB envelope, and boots `Home.py`.
4. The public URL emitted in step 3 is the deployment URL recorded below.

## Deployed URL

| Field | Value |
|---|---|
| Public URL | https://heart-failure-prediction-krwy4etmtavrqd233mteym.streamlit.app/ |
| Main file | `app/Home.py` |
| Python runtime | 3.11 (`app/runtime.txt`) |
| Requirements | `app/requirements.txt` (pinned, app-scoped) |
| Deployed | 2026-10-07 (Streamlit Cloud app created from `Freston1605/heart-failure-prediction`, branch `milestone/M001`); URL renewed 2026-10-11 after the original deployment stopped serving anonymously (owner re-published the app at the new subdomain, MEM082) — the original 2026-10-07 subdomain is retired and the live subdomain recorded above is the current one |

Live-verified in a real browser session (2026-10-07):
- Home renders with the green serving banner: "Model artifact loaded and validated", serving model `random-forest`, threshold `0.33`, calibration `isotonic` — the 2.5 MB tracked artifact deployed and deserialized on the host.
- The Explore and Predict pages load via the sidebar nav (real slugs: `/Explore`, `/Predict`); the Predict page shows the audit trail, the visible disclaimer ("Research portfolio demonstration — not a medical device. No output here is medical advice."), and the patient-input form.
- Note: raw `curl` gets HTTP 303 to `share.streamlit.io/-/auth/...` on first hit — that is session/bot gating on the Cloud edge, not an access wall; a normal browser session renders the app.
- Known defect (pre-existing, S07): the custom markdown links in the `Home.py` sidebar (`1 — Explore` → `/1_Exploratory_Data_Analysis`, `2 — Predict` → `/2_Predict`) use stale page slugs and show Streamlit's "Page not found" dialog; the native sidebar nav (`/Explore`, `/Predict`) works. Recorded for the deployed-flow UAT task (T02).

### Redeploy 2026-10-13 (M007 explainability)

| Field | Value |
|---|---|
| Source | PR #8 merge of `milestone/M007` into `main` (merge commit `442fcb9`, merged 2026-10-13T17:49:40Z) |
| Manifest | `app/requirements.txt` now pins `shap==0.52.0`; the CI **deploy-manifest** job (`pip install -r app/requirements.txt` + `assert shap.__version__ == "0.52.0"` + pin-match vs `pyproject.toml`) is green on this head (run `38441195137`, all jobs green) |
| Tracked branch | `main` — confirmed by evidence: pushing the PR #8 merge to `main` triggered a Streamlit Cloud rebuild that began serving the merged code (the boot traceback below is served from `/mount/src/heart-failure-prediction/...`), so the app auto-redeploys on `main` (the earlier "branch `milestone/M001`" wording above is the creation-time record, now superseded by the current `main` tracking) |
| Health probe | `GET /healthz` → `200 {"status": "ok"}` |
| Runtime | Python 3.11 (`app/runtime.txt`), `app/Home.py` unchanged |
| **Deployed `/Predict` cohort marker** | **OBSERVED 2026-10-13 (after the mlflow fix).** Real browser session on `/Predict` (app frame `~/+/Predict`) renders the M007 cohort-drivers subheader `What drives scores across patients?` with the S03 explanation "Mean \|SHAP\| across 200 deterministically sampled rows of the pinned 918-row dataset, computed once per session from the live serving artifact". |

**Boot crash and resolution (2026-10-13):** the first redeploy (PR #8, merge `442fcb9`) crashed the deployed `/Predict` page at boot — `app/pages/2_Predict.py` → `src/heart/tracking/mlflow_store.py:44` raised `ModuleNotFoundError: No module named 'mlflow'`. Root cause: `app/requirements.txt` (the app-scoped manifest Cloud installs) never carried `mlflow`, which the Predict page's import closure (via `app/lib/runs_view.py` and the M002/M006 code path) imports at module load; it only surfaced once PR #8 landed that code on the deployed branch. Fixed by PR #9 (merge `bc619b7`): pinned `mlflow==3.16.1` (exact `pyproject.toml` `[ml]` version) in `app/requirements.txt`, extended the CI `deploy-manifest` job to smoke-import `mlflow` and pin-assert it, and added `tests/test_deploy_manifest.py` guards (`test_manifest_pins_mlflow_matching_pyproject` + an app-import-coverage guard). PR #9 CI green (test/build/deploy-manifest), and the deployed cohort marker above was then observed. The `shap==0.52.0` pin was never at fault.

### Boot crash observed 2026-10-14 (M008/S01 T02, optuna manifest gap)

The live browser UAT session for the Data & Methods slice found **every**
deployed page (including `/Data_Methods`) rendering Streamlit's redacted
`ImportError` box with `Traceback: File "/mount/src/heart-failure-prediction/
src/heart/tuning/runner.py", line 63, in <module>` — the app serves HTTP 200
(the shell renders; `/healthz` still answers `ok`), so HTTP-level reachability
pins stay green while the runtime is down; only the browser half detects it.
Root cause: `heart.tuning.runner` (in the Runs page's import closure) does
`import optuna` at module load, and `app/requirements.txt` never carried
`optuna` — the same bug class as the 2026-10-13 `mlflow` crash above. Fix
carried on the M008 branch: pin `optuna==5.0.0` (exact pyproject version) in
`app/requirements.txt`, guarded by the new
`tests/test_deploy_manifest.py::test_manifest_pins_optuna_matching_pyproject`.
Redeploy requires the normal merge/push flow (Streamlit Cloud re-clones on
`main`); the live `/Data_Methods` UAT evidence for M008/S01 is re-captured
after that redeploy.

**Resolved 2026-10-14 (PR #10):** the fix reached the tracked branch through
the established PR flow — `milestone/M008` (with `origin/main` merged in first,
the same topology as M007's pre-PR merge `d9b133e`) pushed and merged to `main`
as PR #10 (merge commit `4aec9fe`). CI on the PR head: `build` and
`deploy-manifest` (installs `app/requirements.txt`, pin-asserts) **green**; the
`test` job's sole failure was `test_data_methods_evidence_pack_present` — the
not-yet-captured M008 evidence screenshots this record's re-run then produced.
Streamlit Cloud re-cloned `main` and rebooted; the post-redeploy real-browser
session on `/Data_Methods` observed the healthy app (console state machine
`INITIAL -> … -> RUNNING`, no exception box, `Dataset provenance` banner
showing `918 rows × 12 columns`) and captured `reports/uat-evidence/12-…jpg`
and `13-…jpg`. Served surfaces unchanged: the multipage app still serves
`app/pages/1_Explore.py`–`app/pages/4_Data_Methods.py` as `/Explore`,
`/Predict`, `/Runs`, `/Data_Methods`.

## Verifications already performed locally (pre-deploy)

- `app/requirements.txt` pins match the versions verified in development and
  parse cleanly; `.streamlit/config.toml` parses as valid TOML.
- The serving artifact loads through the app boundary
  (`app.lib.artifact_loader.load_winning_artifact`) and validates via
  `heart.serving.artifact.load_artifact` (format version, schema, feature
  order).
- Local `streamlit` headless boot of `app/Home.py` renders the serving banner
  and both pages mount (see `reports/app_verification.md`, slice S07).

## Failure modes when the host fails (operational notes)

- **Artifact missing/mismatch at deploy time:** the app never crashes at load —
  `load_winning_artifact()` returns `ok=False` and Home/Predict render the named
  friendly error (`app.lib.errors.AppError` kinds), not a stack-trace white
  screen.
- **Dependency install failure (pin drift vs unpickled estimator):** the
  sklearn/scipy pins in `app/requirements.txt` are the versions the pickle was
  produced under; a build log failure should be treated as a redeploy of the
  exact pinned set, not a lazy upgrade (unpickling ties binaries to pinned
  versions).
- **Constraint limit breach (repo growth):** the artifact is regenerated by
  `scripts/reproduce.sh`/`make reproduce` (S08 T03) kept under the 1 GB
  envelope; the artifact size above is re-recorded in this file when
  re-exported.
