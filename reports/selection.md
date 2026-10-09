# Selection report

Winner: **random-forest** (roc_auc 0.930335 ± 0.018704).

## Ranking (repeated-CV, per-fold distribution)

| rank | model | mean ± std |
| --- | --- | --- |
| 1 | random-forest | 0.930335 ± 0.018704 |
| 2 | logistic-regression-l2 | 0.920103 ± 0.020437 |

## Paired significance (corrected resampled t-test)

| comparison | mean diff (A - B) | p-value | verdict |
| --- | --- | --- | --- |
| random-forest vs logistic-regression-l2 | +0.010233 | 0.221 | not significant |

Serving weight: **2.41 MB** (budget 25.00 MB) — within budget.

Calibration (Brier 0.1120 -> 0.1173 (did not improve); ECE 0.0428 -> 0.0627):

| bin | calibrated mean predicted | calibrated fraction positive | gap | count |
| --- | --- | --- | --- | --- |
| 0 | 0.0013 | 0.0345 | 0.0332 | 29 |
| 1 | 0.1111 | 0.1579 | 0.0468 | 19 |
| 2 | — | — | — | 0 |
| 3 | 0.3214 | 0.0000 | 0.3214 | 1 |
| 4 | 0.4635 | 0.5385 | 0.0750 | 13 |
| 5 | 0.5000 | 0.3333 | 0.1667 | 3 |
| 6 | — | — | — | 0 |
| 7 | 0.7500 | 0.6154 | 0.1346 | 26 |
| 8 | 0.8774 | 0.8571 | 0.0203 | 7 |
| 9 | 0.9200 | 0.9592 | 0.0392 | 49 |

Chosen threshold: **0.33** (objective ``f1`` at 0.8603, swept on validation rows only).

Flags: not_significantly_better_than_baseline (corrected resampled t-test p=0.221 at alpha=0.05), calibration_did_not_improve_brier (delta -0.005274).

**Verdict: NO-SHIP**
