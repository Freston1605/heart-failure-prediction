# Clean-Environment Reproduction (S08/T04)

Proof of the project's central claim: **the published leaderboard is
regenerable from a clean checkout with one documented command.** On
2026-10-08 the full pipeline (`scripts/reproduce.sh`) was executed inside an
isolated clean-checkout copy with a freshly created virtualenv and an empty
MLflow store — every model retrained from scratch — and the regenerated
`reports/leaderboard.md` matched the published numbers with **zero numeric
deviation** across all 110 metric cells.

## Environment (fresh, not the development one)

| Property | Value |
| --- | --- |
| Source revision | commit `c09310b` (`git archive HEAD` — tracked files only, so no local state, caches, or run store leaked in) |
| Clean checkout location | `.artifacts/repro-t04/clean-checkout/` (isolated copy under this repo's scratch area; the tracked working tree was not modified) |
| Reproduction venv | `.artifacts/repro-t04/clean-checkout/.repro-venv` — created fresh by the script (no pre-existing venv; the developer interpreter is never used) |
| Python | 3.14.7 (script gate accepts 3.11–3.14) |
| Pinned stack readback (from the run log) | numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, mlflow 3.16.1, optuna 5.0.0, xgboost 3.4.1 |
| Tracking store at start | **empty** (`experiments/mlruns` is not tracked, so the copy shipped none — all 10 final runs were created from scratch) |
| Dataset integrity gate | passed before training (pinned SHA-256 of `data/raw/*.csv`, row count, schema, and committed split digests vs `manifest.json`) |
| Started / finished (UTC) | 2026-10-08T21:09:42Z → 2026-10-08T21:29:44Z (wall clock ≈ 1202 s, incl. pinned install) |
| Pipeline exit code | 0 (exit code 0, `set -euo pipefail`; any stage failure would abort) |
| Raw run log | captured in the T04 execution record (staged log: `.artifacts/repro-t04/reproduce.log`) |

## Method

From the clean-checkout copy, the single documented command was run:

```bash
./scripts/reproduce.sh
```

which performs: fresh venv → pinned install (`pip install -e ".[ml,dev]"`) →
dataset integrity gate → full 10-model seeded battery + hyperparameter tuning
(5-fold CV, 50 trials/model) → S06 selection flow → regeneration of
`reports/leaderboard.md`. The regenerated file was then compared against the
published, committed version programmatically (not by eye).

## Published vs reproduced numbers

`Published` = committed `reports/leaderboard.md` (sha256
`94e18010456aa8536f80460fa65e2df349d637187bfe60236c7edb2622d3d23c`); `Reproduced` =
the file regenerated inside the clean checkout (sha256
`c89a6b27d76c342e8a0dd05471fd592bca5351debb04dcf5afab8f358bd9c20e`). Deviation =
reproduced − published; every one of the 11 metrics × 10 models is **+0.0000**.

| Model | Published ROC-AUC | Reproduced ROC-AUC | Deviation (ROC-AUC) | Published Accuracy | Reproduced Accuracy | Deviation (Accuracy) | Published Precision | Reproduced Precision | Deviation (Precision) | Published Recall | Reproduced Recall | Deviation (Recall) | Published F1 | Reproduced F1 | Deviation (F1) | Published PR-AUC | Reproduced PR-AUC | Deviation (PR-AUC) | Published Specificity | Reproduced Specificity | Deviation (Specificity) | Published NPV | Reproduced NPV | Deviation (NPV) | Published Prevalence | Reproduced Prevalence | Deviation (Prevalence) | Published Brier | Reproduced Brier | Deviation (Brier) | Published ECE | Reproduced ECE | Deviation (ECE) | Published N | Reproduced N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9432 | 0.9432 | +0.0000 | 0.9185 | 0.9185 | +0.0000 | 0.8919 | 0.8919 | +0.0000 | 0.9706 | 0.9706 | +0.0000 | 0.9296 | 0.9296 | +0.0000 | 0.9403 | 0.9403 | +0.0000 | 0.8537 | 0.8537 | +0.0000 | 0.9589 | 0.9589 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0824 | 0.0824 | +0.0000 | 0.0917 | 0.0917 | +0.0000 | 184 | 184 |
| Random Forest | 0.9353 | 0.9353 | +0.0000 | 0.8859 | 0.8859 | +0.0000 | 0.9010 | 0.9010 | +0.0000 | 0.8922 | 0.8922 | +0.0000 | 0.8966 | 0.8966 | +0.0000 | 0.9294 | 0.9294 | +0.0000 | 0.8780 | 0.8780 | +0.0000 | 0.8675 | 0.8675 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0940 | 0.0940 | +0.0000 | 0.1040 | 0.1040 | +0.0000 | 184 | 184 |
| QDA | 0.9351 | 0.9351 | +0.0000 | 0.8967 | 0.8967 | +0.0000 | 0.8952 | 0.8952 | +0.0000 | 0.9216 | 0.9216 | +0.0000 | 0.9082 | 0.9082 | +0.0000 | 0.9190 | 0.9190 | +0.0000 | 0.8659 | 0.8659 | +0.0000 | 0.8987 | 0.8987 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0853 | 0.0853 | +0.0000 | 0.0598 | 0.0598 | +0.0000 | 184 | 184 |
| k-Nearest Neighbours | 0.9308 | 0.9308 | +0.0000 | 0.8967 | 0.8967 | +0.0000 | 0.8952 | 0.8952 | +0.0000 | 0.9216 | 0.9216 | +0.0000 | 0.9082 | 0.9082 | +0.0000 | 0.9175 | 0.9175 | +0.0000 | 0.8659 | 0.8659 | +0.0000 | 0.8987 | 0.8987 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0891 | 0.0891 | +0.0000 | 0.0948 | 0.0948 | +0.0000 | 184 | 184 |
| Logistic Regression (L1) | 0.9307 | 0.9307 | +0.0000 | 0.8696 | 0.8696 | +0.0000 | 0.8900 | 0.8900 | +0.0000 | 0.8725 | 0.8725 | +0.0000 | 0.8812 | 0.8812 | +0.0000 | 0.9272 | 0.9272 | +0.0000 | 0.8659 | 0.8659 | +0.0000 | 0.8452 | 0.8452 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0960 | 0.0960 | +0.0000 | 0.0871 | 0.0871 | +0.0000 | 184 | 184 |
| Logistic Regression (L2) | 0.9305 | 0.9305 | +0.0000 | 0.8967 | 0.8967 | +0.0000 | 0.9029 | 0.9029 | +0.0000 | 0.9118 | 0.9118 | +0.0000 | 0.9073 | 0.9073 | +0.0000 | 0.9276 | 0.9276 | +0.0000 | 0.8780 | 0.8780 | +0.0000 | 0.8889 | 0.8889 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.1000 | 0.1000 | +0.0000 | 0.0927 | 0.0927 | +0.0000 | 184 | 184 |
| Logistic Regression (ElasticNet) | 0.9302 | 0.9302 | +0.0000 | 0.8967 | 0.8967 | +0.0000 | 0.8879 | 0.8879 | +0.0000 | 0.9314 | 0.9314 | +0.0000 | 0.9091 | 0.9091 | +0.0000 | 0.9273 | 0.9273 | +0.0000 | 0.8537 | 0.8537 | +0.0000 | 0.9091 | 0.9091 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0955 | 0.0955 | +0.0000 | 0.0905 | 0.0905 | +0.0000 | 184 | 184 |
| LDA | 0.9289 | 0.9289 | +0.0000 | 0.8913 | 0.8913 | +0.0000 | 0.8796 | 0.8796 | +0.0000 | 0.9314 | 0.9314 | +0.0000 | 0.9048 | 0.9048 | +0.0000 | 0.9245 | 0.9245 | +0.0000 | 0.8415 | 0.8415 | +0.0000 | 0.9079 | 0.9079 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0911 | 0.0911 | +0.0000 | 0.0854 | 0.0854 | +0.0000 | 184 | 184 |
| Naive Bayes | 0.9280 | 0.9280 | +0.0000 | 0.8750 | 0.8750 | +0.0000 | 0.8911 | 0.8911 | +0.0000 | 0.8824 | 0.8824 | +0.0000 | 0.8867 | 0.8867 | +0.0000 | 0.9231 | 0.9231 | +0.0000 | 0.8659 | 0.8659 | +0.0000 | 0.8554 | 0.8554 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.1040 | 0.1040 | +0.0000 | 0.1068 | 0.1068 | +0.0000 | 184 | 184 |
| Support Vector Machine | 0.9249 | 0.9249 | +0.0000 | 0.8859 | 0.8859 | +0.0000 | 0.8932 | 0.8932 | +0.0000 | 0.9020 | 0.9020 | +0.0000 | 0.8976 | 0.8976 | +0.0000 | 0.9212 | 0.9212 | +0.0000 | 0.8659 | 0.8659 | +0.0000 | 0.8765 | 0.8765 | +0.0000 | 0.5543 | 0.5543 | +0.0000 | 0.0985 | 0.0985 | +0.0000 | 0.0686 | 0.0686 | +0.0000 | 184 | 184 |

Beyond the metric table, the comparison also confirmed programmatically:

- **Confusion matrices** (TN/FP/FN/TP for all 10 models): identical.
- **Calibration table** (Brier + expected calibration error, all models): identical.
- **Winner-selection block**: identical after ignoring MLflow run IDs — winner
  Random Forest, repeated-CV ROC-AUC 0.930335 ± 0.018704, significance vs
  baseline p = 0.221 (not significant), calibrated threshold 0.33, Brier
  0.1120 → 0.1173 (did not improve), serving weight 2.41 MB, flags and the
  **NO-SHIP** ship verdict all reproduced exactly.

## Deviations

Numeric deviations: **none** (0 of 110 metric cells differ). The full-file diff
between published and regenerated `leaderboard.md` has exactly two classes of
difference, both non-numeric metadata:

| # | Deviation | Observed | Cause | Impact on the claim |
| --- | --- | --- | --- | --- |
| 1 | `generated at` timestamp | published `2026-10-07T18:40:57+00:00` vs reproduced `2026-10-08T21:29:44+00:00` | The leaderboard stamps its render time (`datetime.now(UTC)`) by design. | None — render time is not a model result. |
| 2 | MLflow run provenance IDs | all 10 run IDs differ (e.g. winner `d51e1f6d…` → `fb1f9fd5…`) | Each reproduction trains from scratch into its own (untracked) `experiments/mlruns` store; MLflow assigns fresh 32-hex run IDs per training run. | None — all run-attached numbers (metrics, CV, trials, split `v1`) are identical; only the store-local identity differs. |

Nothing else differs: ranks, ordering, model types/families, confusion
matrices, calibration, winner selection, significance verdict, threshold, and
ship decision are byte-equal between the two files.

## How a stranger verifies this

```bash
git clone <this-repo-url> && cd heart_failure
make reproduce          # ~20 min: fresh venv -> integrity gate -> battery -> selection -> leaderboard
cp reports/leaderboard.md /tmp/reproduced-leaderboard.md
diff <(grep -vE 'generated at|run `' reports/leaderboard.md) \
     <(grep -vE 'generated at|run `' /tmp/reproduced-leaderboard.md)   # empty diff expected
```

The regenerated `reports/leaderboard.md` must agree with the committed one on
every number; only the `generated at` line and run-provenance IDs may differ
(see Deviations above). `make check-data` verifies the dataset pin alone, and
`tests/test_reproduction_contract.py` holds this comparison as an executable
contract.
