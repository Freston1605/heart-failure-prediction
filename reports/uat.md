# UAT — deployed app (S08 T02)

**Deployed URL:** https://heart-failure-prediction-krwy4etmtavrqd233mteym.streamlit.app/
**Apps/flows covered:** multipage Streamlit Cloud deployment — dashboard (Home + Explore), prediction (/Predict), and Data & Methods transparency (/Data_Methods).
**Method:** both flows exercised in a real automated browser session (Chromium 1280x800) against the live URL; each step recorded by a DOM assertion and an on-disk screenshot bundle. The machine-checkable half is the reproducible pytest suite `tests/uat/test_deployed_app.py` (13 tests, all passing), which re-asserts the same facts against the deployed URL and the tracked serving modules. M007/S04 added **Flow 3 — explainability (/Predict)**: the same session asserted the raw/calibrated pair, the per-patient SHAP attribution, and the cohort-drivers section on the deployed page (screenshots 09-11). M008/S01 added **Flow 4 — Data & Methods transparency (/Data_Methods)**: the four stage sections and the Part B-F treatment/gate-verdict tables rendered from the manifest-verified bundle on the live route (screenshots 12-13).

Edge context (documented, not a defect): a raw first-hit HTTP GET to the app returns **303 to `share.streamlit.io/-/auth/app`** — the Streamlit Cloud edge's session/bot gate; a browser session transits `/-/login?payload=...` automatically and lands on a 200. `HTTP evidence: pytest test_root_is_reachable_through_documented_auth_chain`.

## Flow 1 — dashboard (Home + Explore)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| D1 | App URLs under `.../` redirect chain (auth → payload → 200) and the app iframe `_stcore/health` answers `ok` | PASS (DOM + network) | browser network log: `GET / → 303`, `GET /-/login?payload=… → 303`, `GET / → 200`, `GET /~/+/_stcore/health → 200 "ok"`; pytest `test_app_health_endpoint_is_ok` |
| D2 | Home renders the green validated-serving banner "Model artifact loaded and validated. The app can serve predictions." | PASS (DOM) | `browser_assert` 5/5 text_visible in app iframe; `reports/uat-evidence/01-home-dashboard.jpg` |
| D3 | Home banner metrics match the tracked artifact metadata: serving model `random-forest`, threshold `0.33` (`f1`), calibration `isotonic` (applied) | PASS (DOM + pytest) | screenshot 01; pytest `test_home_serving_banner_facts_match_deployed_metadata` |
| D4 | Home shows the standing boundary text "not a medical device" | PASS (DOM) | `browser_assert` 5/5; screenshot 01 |
| D5 | Clicking the native sidebar "Explore" navigates to `/Explore` and the dashboard renders: H1 "📊 Explore — dataset insight", "Distributions", "Class balance", and rendered charts (25 canvases in frame) | PASS (DOM + screenshot) | click → URL `.../Explore`, title "1 — Explore · Streamlit"; `browser_assert` text_visible ×3; screenshot `reports/uat-evidence/02-explore-dashboard.jpg`; bundle DOM (25 `<canvas>`) |

## Flow 2 — prediction (/Predict)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| P1 | `/Predict` serves the prediction page: H1 "Predict — risk probability", "Audit trail", "Patient information" form (12 inputs) | PASS (DOM) | `browser_assert` 5/5 in app iframe; `reports/uat-evidence/03-predict-disclaimer.jpg` |
| P2 | **Medical disclaimer is visible on the prediction page**: prominent warning banner "Research portfolio demonstration — **not a medical device**. No output here is medical advice." above the form | PASS (DOM) | `browser_assert` text_visible ×5; screenshot 03 |
| P3 | Disclaimer-acknowledgement checkbox with wording "I understand this tool is a research portfolio demonstration…" renders beside the form | PASS (DOM) | `browser_assert` text_visible ×5; screenshot 03 |
| P4 | **Negative path — submitting without the acknowledgement** yields the guard warning "Please tick the disclaimer acknowledgement above before a prediction is computed. Nothing was predicted from this input." and **no** prediction metric is rendered | PASS (DOM) | click Predict (ack unticked) → warning text_visible; `innerText` contains `NO_PREDICTION_RENDERED` (absence of "Predicted probability of heart disease"); `reports/uat-evidence/04-predict-unacknowledged-blocked.jpg` |
| P5 | Ticking the acknowledgement (aria-checked true) then submitting computes a prediction from the deployed artifact | PASS (DOM) | checkbox → `aria-checked="true"`; click Predict → `browser_wait_for` "Predicted probability of heart disease" |
| P6 | Prediction result: "Predicted probability of heart disease `0.1111`", "Selected threshold (f1) `0.33`", "probability is below the tuned threshold", and the full disclaimer re-rendered next to the result | PASS (DOM) | `browser_assert` 3/3; `reports/uat-evidence/05-predict-success.jpg` |
| P7 | The served **0.1111** is reproducible offline: the same default feature row (Age 54, F, ASY, 130, 223, 0, Normal, 138, N, 0.6, Up) through the tracked `models/heart-winner-v1.pkl` yields exactly `0.1111` — the deployed app serves the real serialized artifact | PASS (pytest) | `test_default_form_prediction_matches_browser_observed_value` |

## Flow 3 — explainability (/Predict)

Rendered on the same `/Predict` session (M007); screenshots committed under `reports/uat-evidence/`.

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| E1 | **Cohort drivers render pre-submit**: section "What drives scores across patients?" with the mean-\|SHAP\| caption (200 deterministically sampled rows of the pinned 918-row dataset), the 8-bar chart, and the top-5 table (ChestPainType_ASY 0.0925, ST_Slope_Up 0.0919, …) — visible before any Predict click | PASS (DOM + screenshot) | `browser_assert` 2/2 text_visible in app iframe before submit; `reports/uat-evidence/11-predict-cohort-drivers.jpg` |
| E2 | **Negative path — the acknowledgement gate still blocks submit**: clicking Predict with the acknowledgement unticked yields the guard warning "Please tick the disclaimer acknowledgement above before a prediction is computed. Nothing was predicted from this input." and **no** prediction metric, raw pair, or attribution renders | PASS (DOM) | guard `text_visible`; frame `innerText` scan: `Predicted probability…` false, `Raw RF probability…` false, `Why this score?` false |
| E3 | **Raw/calibrated pair values**: calibrated metric `0.1111`, threshold `0.33`, raw metric `Raw RF probability (pre-calibration) 0.1200`, and the gap line `Raw RF 0.1200 → isotonic-calibrated 0.1111 (calibration gap -0.0089)` (source `->` renders as `→` in the DOM) | PASS (DOM + screenshot + pytest) | `reports/uat-evidence/09-predict-raw-calibrated-pair.jpg`; `test_default_form_raw_and_calibrated_pair_matches_browser_observed_value` |
| E4 | **Attribution chart top features**: "Why this score?" with the raw-score caption (raw 0.1200, base 0.5557), the 8-bar SHAP chart, and the table headed by `pushed up ChestPainType_ASY 0.0790` / `pushed down ST_Slope_Up -0.0985` (ST_Slope_Up the largest-\|SHAP\| feature overall); additive caption reproduces the raw score (base 0.5557 + Σ -0.4357 = 0.1200) | PASS (DOM + screenshot + pytest) | `reports/uat-evidence/10-predict-shap-attribution.jpg`; `test_attribution_top_features_match_browser_observed_values` |
| E5 | **No exception anywhere in the session**: no `Traceback`, no "The explanation is unavailable (…)" warning, no degradation notice ("…only the explanation is affected.") anywhere in the frame after submit — the app never white-screened | PASS (DOM) | frame `innerText` scan: `traceback=false`, `degradationNotice=false`, `explanationUnavailable=false` |

## Flow 4 — Data & Methods transparency (/Data_Methods)

Rendered on a real-browser session against the live route (M008/S01); screenshots committed under `reports/uat-evidence/`.

**Wake/redeploy cycle (recorded per plan):** the first live hit of this session found **every** deployed page serving Streamlit's redacted `ImportError` box (`src/heart/tuning/runner.py`, line 63 — `import optuna` at module load; `app/requirements.txt` never carried `optuna`), while HTTP-level probes stayed green (200 shell, `/healthz` ok). The fix was pushed through the normal commit flow: `origin/main` merged into `milestone/M008`, branch pushed, **PR #10** opened and merged to the tracked branch `main` (merge commit `4aec9fe`; CI `build` + `deploy-manifest` green — the `test` job's sole failure was the then-missing evidence screenshots this re-run produced), Cloud re-cloned `main` and re-booted, and the session re-woke the app and re-checked. Full incident + resolution record: `reports/deployment.md` ("Boot crash observed 2026-10-14").

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| M1 | **Route 200 via the documented auth chain**: direct `/Data_Methods` hit → 303 auth → app renders (title "4 — Data Methods · Streamlit", sidebar lists Home/Explore/Predict/Runs/Data Methods) | PASS (DOM + network) | browser navigation `GET /Data_Methods → 303 → 200`; pytest `test_app_serves_all_documented_routes` probes `/`, `/Explore`, `/Predict`, `/Data_Methods` |
| M2 | **Zero exception**: no `stException` box, no traceback, no `ModuleNotFoundError` banner anywhere in the frame; console state machine `INITIAL -> … -> RUNNING` with no app-level error (only platform noise: 404 `/api/v2/user/details`, retired `/_stcore` probes under `/~/+/`, blocked heap-analytics script) | PASS (DOM + console) | frame `innerText` scan: `traceback=false`, `stException=false`, `ModuleNotFoundError=false`; console log `INITIAL -> (5, 0, ) -> RUNNING` |
| M3 | **Four stage sections present**: subheaders "Stage 1 — Data hygiene", "Stage 2 — Per-column distributions", "Stage 3 — Column-vs-target relationships", "Stage 4 — Robustness assumptions" | PASS (DOM + screenshot) | `browser_evaluate` string scan 4/4 true; `reports/uat-evidence/12-data-methods-stage-sections.jpg` |
| M4 | **Parts B-F present**: "Part B — Per-model treatment audit", "Part C — Logistic-regression preparation methodology", "Part D — Pre-declared class-imbalance ladder", "Part E — Validation and calibration", "Part F — Severity-gate verdicts" | PASS (DOM + screenshot) | string scan 5/5 true; `reports/uat-evidence/13-data-methods-gate-verdicts.jpg` |
| M5 | **Part F verdicts table non-empty with the 48 rejected pairs**: grid `aria-rowcount` 49 (header + 48), rendered verdict cells all `rejected`, caption "Split `v1` · 48 measured pairs · standings supplied: True …" (Part E narrative: "Measured pairs: **48** — accepted 0, rejected 48 … *incumbent stands*") | PASS (DOM + screenshot + pytest) | grid `aria-rowcount=49`, rejected cells rendered; screenshot 13; pytest `test_data_methods_gate_verdicts_pin_the_deployed_table` (split `v1`, 48/48 `rejected`) |
| M6 | **Dataset provenance banner shows 918 rows**: "**918 rows × 12 columns** · target column `HeartDisease` · random seed 42 … pinned raw-CSV digest verified clean" | PASS (DOM + screenshot) | string scan `918 rows` true; screenshot 12; pytest `test_data_methods_bundle_verifies_clean_for_deployed_host` |
| M7 | **Technical appendix expanded**: "Manifest artifact index (all digests verified clean):" lists the bundle entries (`meta.json`, `stage1_hygiene.json`, `stage2_bar_*.png`, `stage2_hist_*.png`, …) each with its SHA-256 digest | PASS (DOM + screenshot) | expander opened in session; string scan `stage1_hygiene` true; screenshot 13 |
| M8 | **Stage figures render as images, not broken links**: 23 `<img>` elements in the frame, 0 broken (`complete && naturalWidth > 0` for every visible one) | PASS (DOM) | frame image scan: `imgCount=23`, `broken=0` |
| M9 | **No named degradation**: "The Data & Methods report is unavailable (artifacts-missing/…)" with the regeneration hint absent (the page's own failure signal, so a stale deployed branch would have failed here) | PASS (DOM) | innerText scan `unavailable=false` |

## Reproducible checks (pytest)

```
$ python -m pytest tests/uat/test_deployed_app.py -v
13 passed
```

Test list: `test_root_is_reachable_through_documented_auth_chain`, `test_followed_root_resolves_to_a_200_page`, `test_app_health_endpoint_is_ok`, `test_app_serves_all_documented_routes` (now probes `/Data_Methods` too), `test_disclaimer_wording_matches_deployed_serving_module`, `test_home_serving_banner_facts_match_deployed_metadata`, `test_default_form_prediction_matches_browser_observed_value`, `test_default_form_raw_and_calibrated_pair_matches_browser_observed_value`, `test_attribution_top_features_match_browser_observed_values`, `test_data_methods_bundle_verifies_clean_for_deployed_host`, `test_data_methods_gate_verdicts_pin_the_deployed_table`, `test_explainability_evidence_pack_present`, `test_data_methods_evidence_pack_present`.

## Evidence files

- `reports/uat-evidence/01-home-dashboard.jpg` — Home: serving banner + metrics + disclaimer text
- `reports/uat-evidence/02-explore-dashboard.jpg` — Explore dashboard: rendered distributions chart
- `reports/uat-evidence/03-predict-disclaimer.jpg` — Predict: audit trail, NO-SHIP notice, disclaimer banner, patient form
- `reports/uat-evidence/04-predict-unacknowledged-blocked.jpg` — negative path: guard warning, no prediction
- `reports/uat-evidence/05-predict-success.jpg` — prediction served with result metrics + full disclaimer
- `reports/uat-evidence/09-predict-raw-calibrated-pair.jpg` — explainability: calibrated 0.1111 / raw 0.1200 metrics + calibration-gap line (Flow 3, E3)
- `reports/uat-evidence/10-predict-shap-attribution.jpg` — explainability: "Why this score?" SHAP bar chart + pushed up/down table (Flow 3, E4)
- `reports/uat-evidence/11-predict-cohort-drivers.jpg` — explainability: cohort-drivers section rendered pre-submit (Flow 3, E1)
- `reports/uat-evidence/12-data-methods-stage-sections.jpg` — Data & Methods: dataset-provenance banner (918 rows) + Stage 1 hygiene tables/figure + Stage 2 distribution table/figures (Flow 4, M3/M6)
- `reports/uat-evidence/13-data-methods-gate-verdicts.jpg` — Data & Methods: Part E narrative + Part F 48-pair verdicts table with caption + expanded technical appendix manifest index (Flow 4, M4/M5/M7)

(Browser debug bundles — screenshots, accessibility trees, network/console logs — are also on disk under `.artifacts/browser/*uat-*`; the copies above are the committed evidence.)

## Known issues (pre-existing, tracked)

- **Stale custom sidebar links (S07 defect, unchanged by T02):** the Home sidebar's hand-written "1 — Explore" / "2 — Predict" markdown links still point at `/1_Exploratory_Data_Analysis` and `/2_Predict` (Streamlit "Page not found"). The deployment uses the native sidebar nav, and the browser UAT exercised the native links; do not consider the broken markdown links part of either flow.
- The `/Predict` page legitimately renders a **NO-SHIP** notice (flags `not_significantly_better_than_baseline`, `calibration_did_not_improve_brier`) — this is the app's intended honest-boundary contract carried in the artifact metadata, not a UAT failure.

## Verdict

Both dashboard and prediction flows pass end-to-end in a real browser session with the disclaimer confirmed visible on the prediction page, the acknowledgement gate blocking unacknowledged predictions, and a served prediction reproduced exactly by the tracked artifact. The explainability flow (Flow 3, M007/S04) also passes: the cohort-drivers section renders pre-submit, the raw/calibrated pair and the per-patient SHAP attribution on the deployed page reproduce the pinned local artifact values exactly (0.1200 → 0.1111, gap -0.0089; ChestPainType_ASY +0.0790 / ST_Slope_Up -0.0985), and the session never raised an exception. The Data & Methods flow (Flow 4, M008/S01) also passes: the live `/Data_Methods` route renders all four stage sections and Parts B-F from the SHA-256 manifest-verified committed bundle with the Part F table showing 48 measured pairs (48 rejected, incumbent stands), the provenance banner at 918 rows, and the expanded appendix listing the verified artifact index — zero stException, zero traceback, no named degradation, figures all rendering as images. The flow was only reachable after the 2026-10-14 optuna manifest crash was fixed and redeployed (PR #10, wake/redeploy cycle recorded above); the check was never downgraded to make it pass.
