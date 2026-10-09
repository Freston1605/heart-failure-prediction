# Logistic-Regression Baseline

_Published by `heart.models.baseline` (S02/T03). This is the honesty anchor: every tuned model in the portfolio is measured against these numbers._

## Headline

| Metric | Value |
| --- | --- |
| **ROC-AUC (primary)** | **0.9286** |
| pr_auc | 0.9211 |
| accuracy | 0.8913 |
| precision | 0.8796 |
| recall | 0.9314 |
| f1 | 0.9048 |
| specificity | 0.8415 |
| npv | 0.9079 |
| prevalence | 0.5543 |
| brier_score | 0.0925 |
| Expected calibration error | 0.0734 |

## Confusion matrix

| | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| Actual 0 | 69 (TN) | 13 (FP) |
| Actual 1 | 7 (FN) | 95 (TP) |

Evaluated on **184** held-out rows (102 positive / 82 negative).

## Calibration (reliability curve)

Brier score **0.0925**, expected calibration error **0.0734** over 10 equal-width bins.

| Bin | Range | Count | Mean predicted | Observed | Gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00-0.10 | 46 | 0.0537 | 0.0217 | 0.0320 |
| 1 | 0.10-0.20 | 7 | 0.1382 | 0.0000 | 0.1382 |
| 2 | 0.20-0.30 | 14 | 0.2564 | 0.0714 | 0.1850 |
| 3 | 0.30-0.40 | 4 | 0.3554 | 0.2500 | 0.1054 |
| 4 | 0.40-0.50 | 5 | 0.4454 | 0.8000 | 0.3546 |
| 5 | 0.50-0.60 | 11 | 0.5399 | 0.8182 | 0.2783 |
| 6 | 0.60-0.70 | 7 | 0.6587 | 0.8571 | 0.1984 |
| 7 | 0.70-0.80 | 14 | 0.7704 | 0.7857 | 0.0153 |
| 8 | 0.80-0.90 | 20 | 0.8522 | 0.8500 | 0.0022 |
| 9 | 0.90-1.00 | 56 | 0.9565 | 0.9286 | 0.0279 |

## Provenance

- dataset: fedesoriano combined 5-site collection (918 rows, 11 features)
- split: `v1` group-aware, leakage-safe (train 734 / test 184)
- model: zero-as-missing median imputation -> standard scaling + one-hot encoding -> logistic regression
- params: `{'solver': 'lbfgs', 'max_iter': 2000, 'C': 1.0, 'class_weight': None, 'random_state': 42}`
- metric schema: v1
- MLflow run: `e0136f895403438c994d5896c30cd5c4` (experiment `heart-failure-prediction`)
- generated at: 2026-10-06T19:02:37+00:00

## Reproduce

```bash
python -m heart.models.baseline
```
