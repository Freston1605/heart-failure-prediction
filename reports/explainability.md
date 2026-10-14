# Explainability in the Deployed App

**Scope:** M007 — the `/Predict` page of the deployed Streamlit app now shows the
calibration seam (raw RF vs calibrated probability), a per-patient SHAP
attribution chart, and a global cohort-drivers section, all served live from the
tracked artifact on Streamlit Community Cloud.

**Deployed URL:** https://heart-failure-prediction-krwy4etmtavrqd233mteym.streamlit.app/

## Why explainability in the deployed app

The audience for this project is portfolio reviewers: people who click a live
link, submit the default patient, and decide in one screen whether the model
story is credible. Until now the app returned a bare probability — the missing
signal was reasoning, not accuracy. M007 closes that gap on the serving surface
itself, so a reviewer can see, in one deployed page: the calibrated probability
that counts, the raw forest vote behind it (with the gap spelled out), which
features pushed *this* patient's score up or down, and what drives scores across
the cohort — with docs, tests, and committed screenshots backing every number.

## The calibration seam: raw RF vs isotonic-calibrated (D041)

The winning artifact wraps a RandomForest in isotonic calibration
(`CalibratedClassifierCV` → frozen estimator → fitted pipeline). Decision
**D041** puts that seam on screen instead of hiding it: the Predict page renders
the served calibrated probability next to the raw pre-calibration RF score, with
the gap line between them.

Pinned default-patient values (committed artifact `models/heart-winner-v1.pkl`,
default form values, asserted by tests and observed on the deployed page):

- Raw RF probability (pre-calibration): **0.1200**
- Served isotonic-calibrated probability: **0.1111** (decision threshold `0.33`)
- Calibration gap: **-0.0089** — rendered as `Raw RF 0.1200 -> isotonic-calibrated 0.1111 (calibration gap -0.0089)`

The served number is always the calibrated one; the raw pair is best-effort
display. Raw access rides `ServingArtifact.predict_raw_positive_proba` /
`ServingArtifact.raw_estimator` (decision **D043**): if a payload exposes no
calibration wrapper, access fails at request time with a named
`ArtifactPayloadError` — artifact format v1 and the load-time contract are
unchanged — and the page degrades to a friendly named notice while the
calibrated probability is still served.

## Per-patient attribution: TreeExplainer at request time (D040)

Decision **D040**: attribute with `shap.TreeExplainer` on the RandomForest
estimator at request time, pinning `shap==0.52.0` in the deploy manifest behind
a CI manifest-install gate. PermutationExplainer (~10–50x slower per request) and
hand-rolled attribution (less credible) were rejected.

How the computation is reached, without a parallel model path:

- The estimator handle comes only through `ServingArtifact.raw_estimator` — the
  unfrozen base estimator behind the calibration wrapper.
- Input goes through the byte-identical serving schema guard
  (`heart.serving.artifact.check_features`); a schema failure surfaces unchanged
  on the raising path.
- The entire SHAP computation is delegated to the repo's single sanctioned
  contract, `heart.interpret.shap_export.compute_shap_values`; a small naming
  adapter (`_as_explainable_pipeline`) renames the serving pipeline's
  `preprocess` step to the `features` name the contract expects — a zero-copy
  view, never a refit and never a second explainer.
- App code never imports shap directly — only `app/lib/explain.py` does
  (asserted by `test_app_sources_do_not_import_shap_directly`).

The contract lives in `app/lib/explain.py`: `explain_patient` is the **only**
entry the page consumes and it **never raises**. Every outcome is an
`ExplanationOutcome` holding either an attribution or a named `ExplanationError`
with a friendly, plain-language `user_message` (no stack trace). Stable error
kinds: `shap-unavailable`, `explanation-unsupported`, `explanation-failed` (plus
glue-level kinds for the cohort path below).

**Honesty of the seam:** the attribution explains the **raw pre-isotonic RF
score**, and the chart caption says so ("SHAP explains the raw pre-calibration
RF score (raw 0.1200, base 0.5557) — the calibrated probability served above is
what counts"). The additive caption reproduces the raw score exactly:
`base 0.5557 + sum of SHAP values -0.4357 = 0.1200`. Determinism is pinned on
known held-out patients in `tests/test_explain.py` (attribution deterministic
across calls, matches a direct TreeExplainer computation, additivity reproduces
the raw score).

On the deployed page (Flow 3 evidence): "Why this score?" renders the 8-bar SHAP
chart and a table headed by `pushed up ChestPainType_ASY 0.0790` /
`pushed down ST_Slope_Up -0.0985` (ST_Slope_Up is the largest-|SHAP| feature
overall).

## Global cohort drivers: cached per session (D042)

Decision **D042**: compute the global cohort drivers once per session via
`st.cache_resource`, over a deterministic sample of the pinned 918-row dataset —
not as a build-time precomputed artifact, so the global numbers stay consistent
with the live model without adding a build step.

- The row cap is `COHORT_MAX_ROWS = 200` in `app/lib/explain.py` (the repo
  default `shap_export.DEFAULT_MAX_SAMPLES` is `None`, so the cap is requested
  explicitly on every call) — this is also the free-tier load-shed control. The
  deployed page caption reads "Mean |SHAP| across 200 deterministically sampled
  rows of the pinned 918-row dataset".
- Sampling is seeded from `heart.config.RANDOM_SEED`, so repeated runs explain
  the same cohort; the ranking ties keep model column order (same stable ranking
  as `ShapSummary._order`).
- `explain_global` is the global twin of `explain_patient`: same byte-identical
  `check_features` guard, same naming adapter, the whole computation delegated
  to `shap_export.compute_shap_values` + `summarize_shap`, and no exception
  escapes.
- Charts stay **native `st.bar_chart`** — matplotlib/seaborn are deliberately
  not in the deploy manifest (`app/requirements.txt`), keeping the free-tier
  install small and the build fast.
- Both the successful outcome and any failed outcome are sticky for the session
  (the cached-error precedent from `app/lib/plots.py`); only
  `clear_cohort_cache` drops them — the cohort SHAP cost (and any failure) is
  paid exactly once per session.

Rendered output: 8-bar mean-|SHAP| chart plus a top-5 table with mean |SHAP| and
mean signed SHAP (direction). Deployed pins: `ChestPainType_ASY 0.0925`,
`ST_Slope_Up 0.0919` (Flow 3, check E1).

## Failure discipline: explanations never block a prediction

The app's never-raise serving discipline extends to every explanation surface:

- A failed explanation degrades the **section**, never the page: the calibrated
  probability above stays served, and a kind-named inline warning explains what
  happened in plain language ("The explanation is unavailable
  (`explanation-failed`). … The prediction above was served normally; only the
  explanation is affected.").
- Structured degraded events accompany every degradation: `app.explain` with
  `outcome=explanation-unavailable` and `error_kind`, `app.cohort` with
  `outcome=cohort-unavailable` and `error_kind`, both at WARNING level. No stack
  trace ever reaches the UI.
- Render failures are guarded too — a chart-formatting failure degrades the same
  way instead of white-screening a page that already served a probability.
- All of this is AppTest-proven (`tests/test_prediction_page.py` sections 6–8):
  forced compute failure, missing shap, unsupported payload, cohort failure, and
  even a call-site exception all degrade to the named notice with the prediction
  intact (`test_cohort_exception_never_white_screens`,
  `test_raw_access_failure_degrades_to_friendly_notice`).

## Deployment: manifest pin + CI gate + Cloud redeploy

- `shap==0.52.0` is pinned in `app/requirements.txt`, byte-identical to the
  `pyproject.toml` `[ml]` extra (`tests/test_deploy_manifest.py` asserts the
  pin match for shap and mlflow).
- The CI **deploy-manifest** job (`.github/workflows/ci.yml`) installs
  `app/requirements.txt` on **every PR**, smoke-imports the pinned stack
  (`shap, streamlit, sklearn, joblib, pandas, numpy, scipy, mlflow`), and
  pin-asserts shap `0.52.0` and mlflow `3.16.1`. This is the pre-deploy gate: a
  broken or drifted deploy pin fails the PR instead of the Cloud boot. CI had
  previously installed only the pyproject extras and skipped the deploy
  manifest, so a manifest breakage would have surfaced first during the Cloud
  build.
- The Cloud redeploy record and its verification live in
  `reports/deployment.md`: PR #8 merge (`442fcb9`) added the shap pin and
  triggered the rebuild; that redeploy crashed `/Predict` at boot
  (`ModuleNotFoundError: mlflow` — the Predict page's import closure imports
  mlflow at module load and the manifest had never carried it), fixed by PR #9
  (`bc619b7`) pinning `mlflow==3.16.1` and extending the CI gate + manifest
  guards. Post-fix verification: `GET /healthz` → `200 {"status": "ok"}` and the
  deployed `/Predict` cohort marker ("What drives scores across patients?")
  observed in a real browser session — the deployed app is serving the merged
  M007 code from the tracked artifact (wake-then-verify; the free tier sleeps).

## Evidence

- **Screenshots** (committed, real browser session against the live URL):
  - `reports/uat-evidence/09-predict-raw-calibrated-pair.jpg` — calibrated
    `0.1111` / raw `0.1200` metrics + calibration-gap line (Flow 3, E3)
  - `reports/uat-evidence/10-predict-shap-attribution.jpg` — "Why this score?"
    SHAP chart + pushed up/down table (E4)
  - `reports/uat-evidence/11-predict-cohort-drivers.jpg` — cohort-drivers
    section rendered pre-submit (E1)
  - plus the original UAT pack `01`–`08` (dashboard + prediction flow)
- **`reports/uat.md` Flow 3** — the explainability flow table (E1–E5), including
  the deployed negative path (acknowledgement gate blocks submit, nothing
  rendered) and the no-exception session scan.
- **Test surfaces:**
  - `tests/test_explain.py` (18 tests) — attribution determinism on known
    held-out patients, direct-TreeExplainer match, additivity, all named error
    kinds, never-raise outcomes, cohort determinism/shape/order pins, and the
    no-direct-shap-import guard.
  - `tests/test_prediction_page.py` sections 6–8 — AppTest render pins: the
    raw/calibrated pair (matches the frozen base-estimator call), attribution
    (matches a direct `explain_patient` call, determinism), cohort section
    (pins to direct `explain_global`, computed once per session), and the
    degradation paths.
  - `tests/uat/test_deployed_app.py` (10 tests) — deployed-surface pins:
    default form `0.1111`, raw/calibrated pair `0.1200`/`0.1111`,
    attribution top features, evidence-pack presence.
  - `tests/test_deploy_manifest.py` (6 tests) — manifest pins match pyproject,
    every third-party app import is carried by the manifest, the manifest stays
    direct-deps only, and CI carries the deploy-manifest gate job on every PR.

## Observability

Every explanation path emits a structured log event and a latency metric:

| Surface | Log events | Metric |
|---|---|---|
| Per-patient attribution | `app.explain` ok (`outcome=served`, `top_feature`, `n_features`) / degraded (`outcome=explanation-unavailable`, `error_kind`, WARNING) | `heart.serving.explain_latency_ms` |
| Cohort drivers | `app.cohort` ok (`outcome=served`, `top_feature`, `n_rows`) / degraded (`outcome=cohort-unavailable`, `error_kind`, WARNING) | `heart.serving.cohort_explain_latency_ms` |

The prediction counter is untouched: `heart.serving.predict_count` keeps exactly
its previous labels (`served`, `not-acknowledged`, `validation-blocked`,
`error`) — no explanation outcome ever changes a prediction count.

## Known limitations

- **The calibration mapping itself is not attributed.** The attribution explains
  the forest's raw vote; the isotonic transform has no ground truth per feature
  and attributing through it would overclaim. The seam is instead surfaced
  honestly (raw + calibrated + gap, captioned on the chart).
- **Free-tier sleep.** The deployed app sleeps when idle; every deployed probe
  follows wake-then-verify (first hit may serve the wake path, the follow-up is
  the real page), so live checks take one extra round trip.
- **No dedicated alert threshold on explanation degraded-rate.** The degraded
  events and latency metrics exist and are queryable, but there is no dashboard
  alerting on them yet; a sustained degraded-rate would be noticed by inspection,
  not alarm.
- **Cohort latency mixes first-compute and cache hits.**
  `heart.serving.cohort_explain_latency_ms` records both the session's first
  (real) SHAP compute and the subsequent `st.cache_resource` hits — that skew is
  by design and worth remembering when reading the metric.
