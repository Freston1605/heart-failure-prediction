# Deployment

App deployment configuration and deploy record for the free-tier host.

## Chosen host: Streamlit Community Cloud (free tier)

Streamlit Community Cloud is the natural free-tier home for this app: it deploys
directly from this git repository, runs `streamlit run <main-file>` natively,
costs nothing, and constrains exactly what this project needs to respect
(repo-based artifact packaging and app-scoped dependencies).

- **Main file:** `app/Home.py` (multipage; `app/pages/1_Explore.py` and
  `app/pages/2_Predict.py` are auto-discovered)
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
| Public URL | https://heart-failure-prediction-c68cacatqtcrhvls7eszt2.streamlit.app/ |
| Main file | `app/Home.py` |
| Python runtime | 3.11 (`app/runtime.txt`) |
| Requirements | `app/requirements.txt` (pinned, app-scoped) |
| Deployed | 2026-10-07 (Streamlit Cloud app created from `Freston1605/heart-failure-prediction`, branch `milestone/M001`) |

Live-verified in a real browser session (2026-10-07):
- Home renders with the green serving banner: "Model artifact loaded and validated", serving model `random-forest`, threshold `0.33`, calibration `isotonic` — the 2.5 MB tracked artifact deployed and deserialized on the host.
- The Explore and Predict pages load via the sidebar nav (real slugs: `/Explore`, `/Predict`); the Predict page shows the audit trail, the visible disclaimer ("Research portfolio demonstration — not a medical device. No output here is medical advice."), and the patient-input form.
- Note: raw `curl` gets HTTP 303 to `share.streamlit.io/-/auth/...` on first hit — that is session/bot gating on the Cloud edge, not an access wall; a normal browser session renders the app.
- Known defect (pre-existing, S07): the custom markdown links in the `Home.py` sidebar (`1 — Explore` → `/1_Exploratory_Data_Analysis`, `2 — Predict` → `/2_Predict`) use stale page slugs and show Streamlit's "Page not found" dialog; the native sidebar nav (`/Explore`, `/Predict`) works. Recorded for the deployed-flow UAT task (T02).

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
