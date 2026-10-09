# UAT — deployed app (S08 T02)

**Deployed URL:** https://heart-failure-prediction-c68cacatqtcrhvls7eszt2.streamlit.app/
**Apps/flows covered:** multipage Streamlit Cloud deployment — dashboard (Home + Explore) and prediction (/Predict).
**Method:** both flows exercised in a real automated browser session (Chromium 1280x800) against the live URL; each step recorded by a DOM assertion and an on-disk screenshot bundle. The machine-checkable half is the reproducible pytest suite `tests/uat/test_deployed_app.py` (7 tests, all passing), which re-asserts the same facts against the deployed URL and the tracked serving modules.

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

## Reproducible checks (pytest)

```
$ python -m pytest tests/uat/test_deployed_app.py -v
7 passed
```

Test list: `test_root_is_reachable_through_documented_auth_chain`, `test_followed_root_resolves_to_a_200_page`, `test_app_health_endpoint_is_ok`, `test_app_iframe_serves_both_flow_routes`, `test_disclaimer_wording_matches_deployed_serving_module`, `test_home_serving_banner_facts_match_deployed_metadata`, `test_default_form_prediction_matches_browser_observed_value`.

## Evidence files

- `reports/uat-evidence/01-home-dashboard.jpg` — Home: serving banner + metrics + disclaimer text
- `reports/uat-evidence/02-explore-dashboard.jpg` — Explore dashboard: rendered distributions chart
- `reports/uat-evidence/03-predict-disclaimer.jpg` — Predict: audit trail, NO-SHIP notice, disclaimer banner, patient form
- `reports/uat-evidence/04-predict-unacknowledged-blocked.jpg` — negative path: guard warning, no prediction
- `reports/uat-evidence/05-predict-success.jpg` — prediction served with result metrics + full disclaimer

(Browser debug bundles — screenshots, accessibility trees, network/console logs — are also on disk under `.artifacts/browser/*uat-*`; the copies above are the committed evidence.)

## Known issues (pre-existing, tracked)

- **Stale custom sidebar links (S07 defect, unchanged by T02):** the Home sidebar's hand-written "1 — Explore" / "2 — Predict" markdown links still point at `/1_Exploratory_Data_Analysis` and `/2_Predict` (Streamlit "Page not found"). The deployment uses the native sidebar nav, and the browser UAT exercised the native links; do not consider the broken markdown links part of either flow.
- The `/Predict` page legitimately renders a **NO-SHIP** notice (flags `not_significantly_better_than_baseline`, `calibration_did_not_improve_brier`) — this is the app's intended honest-boundary contract carried in the artifact metadata, not a UAT failure.

## Verdict

Both deployed flows pass end-to-end in a real browser session with the disclaimer confirmed visible on the prediction page, the acknowledgement gate blocking unacknowledged predictions, and a served prediction reproduced exactly by the tracked artifact.
