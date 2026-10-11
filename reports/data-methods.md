# Pre-Split Data Methods (EDA) — Part A

This report is rendered by `heart.reporting.data_methods` from the four-stage pre-split EDA over the raw dataset — every stage runs entirely before any train/test split. Regenerate deterministically with `PYTHONPATH=src python -m heart.reporting.data_methods`.

## Dataset provenance

- source: `data/raw/heart.csv`
- SHA-256: `948420b084d8a3a0ca42b8419fce9aee175879e43f8aedf712377899a67aa49b` matches the pinned digest (verified)
- shape: 918 rows × 12 columns
- target: `HeartDisease`
- random seed: 42
- declared zero-as-missing columns: `RestingBP, Cholesterol`
- artifact directory: `reports/data-methods`

## Stage 1 — Hygiene

The frame shape is checked, then missingness is split into *random* (true NaN cells) and *systematic* (declared sentinel zeros in the zero-as-missing columns), exact duplicate rows are grouped, and impossible values are audited under the declared policy.

### Missingness division (random NaN vs declared sentinel zeros)

| column | random NaN cells | declared sentinel-zero cells |
|---|---|---|
| `Cholesterol` | 0 | 172 |
| `RestingBP` | 0 | 1 |

Total: 0 random-NaN cell(s) and 173 declared sentinel-zero cell(s); 172 row(s) carry any missingness across a 918-row frame.

*Artifact:* `stage1_hygiene.json`

### Exact duplicate groups

| duplicate rows | duplicate groups | grouped columns |
|---|---|---|
| 0 | 0 | `Age, RestingBP, Cholesterol, MaxHR` (+8 more) |

*Artifact:* `stage1_hygiene.json`

### Impossible values under the declared policy

| column | impossible zeros | row share |
|---|---|---|
| `Cholesterol` | 172 | 18.7364% |
| `RestingBP` | 1 | 0.1089% |

*Artifact:* `stage1_hygiene.json`

*Figure:* `stage1_missingness_patterns.png` — missingness patterns, NaN cells vs declared zeros.

## Stage 2 — Univariate distributions

### Numeric features: skew, kurtosis, IQR box outliers

| column | skew | kurtosis | IQR outliers | outlier share |
|---|---|---|---|---|
| `Age` | -0.1959 | -0.3861 | 0 | 0.0000% |
| `Cholesterol` | -0.6101 | 0.1182 | 183 | 19.9346% |
| `MaxHR` | -0.1444 | -0.4482 | 2 | 0.2179% |
| `Oldpeak` | 1.0229 | 1.2031 | 16 | 1.7429% |
| `RestingBP` | 0.1798 | 3.2713 | 28 | 3.0501% |

*Artifact:* `stage2_univariate.json`

### Rare categorical levels

No categorical level falls below the rare-level prevalence floor.

*Artifact:* `stage2_univariate.json`

### Target class distribution

| class | count | share |
|---|---|---|
| `0` | 410 | 44.6623% |
| `1` | 508 | 55.3377% |

Prevalence 0.5534 — imbalance ratio 0.8071.

*Artifact:* `stage2_univariate.json`

*Figures:*

- *Figure:* `stage2_hist_Age.png`
- *Figure:* `stage2_hist_RestingBP.png`
- *Figure:* `stage2_hist_Cholesterol.png`
- *Figure:* `stage2_hist_MaxHR.png`
- *Figure:* `stage2_hist_Oldpeak.png`
- *Figure:* `stage2_bar_Sex.png`
- *Figure:* `stage2_bar_ChestPainType.png`
- *Figure:* `stage2_bar_FastingBS.png`
- *Figure:* `stage2_bar_RestingECG.png`
- *Figure:* `stage2_bar_ExerciseAngina.png`
- *Figure:* `stage2_bar_ST_Slope.png`

## Stage 3 — Bivariate and multivariate structure

### Categorical features vs target: chi-square

| column | chi-square | dof | p value | p < 0.05 |
|---|---|---|---|---|
| `ChestPainType` | 268.0672 | 3 | 0.000000 | no |
| `ExerciseAngina` | 222.2594 | 1 | 0.000000 | no |
| `FastingBS` | 64.3207 | 1 | 0.000000 | no |
| `RestingECG` | 10.9315 | 2 | 0.004229 | yes |
| `ST_Slope` | 355.9184 | 2 | 0.000000 | no |
| `Sex` | 84.1451 | 1 | 0.000000 | no |

1 of 6 categorical features pass the chi-square test at alpha 0.05.

*Artifact:* `stage3_bivariate.json`

### Pairs with |r| above the collinearity threshold

No numeric pair exceeds |r| > 0.8.

*Artifact:* `stage3_bivariate.json`

### Variance inflation factors (numeric features)

| column | VIF | flagged (VIF >= 0.8) |
|---|---|---|
| `Age` | 1.2858 | yes |
| `MaxHR` | 1.2401 | yes |
| `RestingBP` | 1.0995 | yes |
| `Oldpeak` | 1.0963 | yes |
| `Cholesterol` | 1.0865 | yes |

Highest VIF 1.2858 on `Age`.

*Artifact:* `stage3_bivariate.json`

*Figures:*

- *Figure:* `stage3_overlay_Age.png`
- *Figure:* `stage3_overlay_RestingBP.png`
- *Figure:* `stage3_overlay_Cholesterol.png`
- *Figure:* `stage3_overlay_MaxHR.png`
- *Figure:* `stage3_overlay_Oldpeak.png`
- *Figure:* `stage3_correlation_heatmap.png`

## Stage 4 — Assumption checks

### Outlier handling: raw vs trim vs Winsorize vs the declared median-impute policy

Every numeric column is measured on the raw frame under all four policies; Δ columns are relative to the raw baseline, so the declared policy's shift can be compared against the discard-based baselines.

| column | policy | n | mean | Δ mean | std | Δ std |
|---|---|---|---|---|---|---|
| `Age` | raw | 918 | 53.5109 | 0.0000 | 9.4326 | 0.0000 |
| `Age` | trim_iqr | 918 | 53.5109 | 0.0000 | 9.4326 | 0.0000 |
| `Age` | winsorize_iqr | 918 | 53.5109 | 0.0000 | 9.4326 | 0.0000 |
| `RestingBP` | raw | 918 | 132.3965 | 0.0000 | 18.5142 | 0.0000 |
| `RestingBP` | trim_iqr | 890 | 131.1135 | -1.2830 | 15.8397 | -2.6744 |
| `RestingBP` | winsorize_iqr | 918 | 132.1253 | -0.2712 | 16.9933 | -1.5208 |
| `RestingBP` | declared_median_impute | 918 | 132.5381 | 0.1416 | 17.9901 | -0.5240 |
| `Cholesterol` | raw | 918 | 198.7996 | 0.0000 | 109.3841 | 0.0000 |
| `Cholesterol` | trim_iqr | 735 | 241.0381 | 42.2385 | 51.1647 | -58.2194 |
| `Cholesterol` | winsorize_iqr | 918 | 203.9852 | 5.1856 | 95.9223 | -13.4618 |
| `Cholesterol` | declared_median_impute | 918 | 243.2048 | 44.4052 | 53.4013 | -55.9828 |
| `MaxHR` | raw | 918 | 136.8094 | 0.0000 | 25.4603 | 0.0000 |
| `MaxHR` | trim_iqr | 916 | 136.9738 | 0.1644 | 25.2432 | -0.2172 |
| `MaxHR` | winsorize_iqr | 918 | 136.8192 | 0.0098 | 25.4321 | -0.0283 |
| `Oldpeak` | raw | 918 | 0.8874 | 0.0000 | 1.0666 | 0.0000 |
| `Oldpeak` | trim_iqr | 902 | 0.8335 | -0.0539 | 0.9660 | -0.1006 |
| `Oldpeak` | winsorize_iqr | 918 | 0.8778 | -0.0096 | 1.0317 | -0.0349 |

Largest absolute mean shift under the declared median-impute policy: 0.0000.

*Artifact:* `stage4_assumptions.json`

### Empirical logit summaries (decile-binned numeric features)

| column | bins | min logit |
|---|---|---|
| `Age` | 10 | -0.9264 |
| `Cholesterol` | 9 | -0.4717 |
| `MaxHR` | 10 | -1.5074 |
| `Oldpeak` | 7 | -0.7013 |
| `RestingBP` | 9 | -0.2513 |

| column | max bin n | max bin positives |
|---|---|---|
| `Age` | 117 | 75 |
| `Cholesterol` | 184 | 159 |
| `MaxHR` | 108 | 81 |
| `Oldpeak` | 381 | 133 |
| `RestingBP` | 181 | 85 |

*Artifact:* `stage4_assumptions.json`

*Figures:*

- *Figure:* `stage4_empirical_logit_Age.png`
- *Figure:* `stage4_empirical_logit_RestingBP.png`
- *Figure:* `stage4_empirical_logit_Cholesterol.png`
- *Figure:* `stage4_empirical_logit_MaxHR.png`
- *Figure:* `stage4_empirical_logit_Oldpeak.png`

## Takeaways checklist

- [ ] **Hygiene:** the raw frame is 918 rows × 12 columns; random-NaN cells 0, declared sentinel-zero cells 173 (Cholesterol, RestingBP); exact duplicate rows 0 (artifact: `stage1_hygiene.json`).
- [ ] **Policy:** sentinel zeros in Cholesterol, RestingBP are treated as *systematic* missingness and median-imputed per the declared zero-as-missing policy; NaN cells would be *random* missingness (artifact: `stage1_hygiene.json`).
- [ ] **Univariate:** target prevalence 0.5534 (imbalance ratio 0.8071); rare levels none at the 5% floor (artifact: `stage2_univariate.json`).
- [ ] **Bivariate:** chi-square-significant categorical features `ChestPainType`, `ExerciseAngina`, `FastingBS`, `RestingECG`, `ST_Slope`, `Sex` (p < 0.05); 0 pair(s) with |r| > 0.8; highest VIF 1.2858 on `Age` (artifact: `stage3_bivariate.json`).
- [ ] **Assumptions:** empirical logit over 10 decile bins per numeric feature (add-0.5 smoothing); the declared median-impute policy's largest absolute mean shift is 0.0000, benchmarked against trim and Winsorize (artifact: `stage4_assumptions.json`).

## Technical appendix

### Parameters

- random seed: 42
- rare-category floor: 0.05
- high-correlation threshold: |r| > 0.8
- histogram bins: 20
- empirical logit bins: 10 (add-0.5 smoothing)
- IQR fence multiplier: 1.5
- outlier fence definition: q1 - 1.5*IQR / q3 + 1.5*IQR
- chi-square alpha: 0.05

### Regeneration command

```
PYTHONPATH=src python -m heart.reporting.data_methods
```

### Artifact index (provenance manifest)

All tables and figures above are traceable to these generated artifacts; digests are recomputed by the provenance tests.

*Artifact:* `manifest.json`

| artifact | sha256 |
|---|---|
| `meta.json` | `544a68ae86f96ce98a3d6d1feee96e8a2ef55babe384c304d57260798ab808db` |
| `stage1_hygiene.json` | `c275ed50ded4f9d2e2466dcb287d8daf542ab3b0f69438d321b561d9788ab7f0` |
| `stage1_missingness_patterns.png` | `3e10296bf9fcd0fd925c7eb15f3a8338444f162dea18954dfb848836be5ebe9d` |
| `stage2_bar_ChestPainType.png` | `97c582ff02cb76f4f4375f394f6e0cbed53d25041ff500afe4f0bb735359c37c` |
| `stage2_bar_ExerciseAngina.png` | `1fa351158afc45e9a6055bbda75492e60309b68d7e6c8bf8325c3ad67394d5a5` |
| `stage2_bar_FastingBS.png` | `50a9672409f3cf4645300442bcd3c0c6a241b190184147a0f900b818c78d4710` |
| `stage2_bar_RestingECG.png` | `bbb5b487807b74fd59205a5a7e73a4fc267ff316d6f6301a370947a54c2fb52d` |
| `stage2_bar_ST_Slope.png` | `ea4f8953551a25b6ea958350229b2be351a3963ed0d5a8624c8c11d01c78a5df` |
| `stage2_bar_Sex.png` | `015bf8a5938e53b8a0e9fb1a1a8502a89ee5a9db62441404a5fa9fdaea8288c8` |
| `stage2_hist_Age.png` | `c9f48017bebb0ad089389e9e27026aebafaaac43082c4117d8d6f9d8b6710ae2` |
| `stage2_hist_Cholesterol.png` | `9f38340cc142829daffc9cb7fc783d73a860ddb661a7d26ae281c1075729e5d9` |
| `stage2_hist_MaxHR.png` | `e63c398c3f1133ffcbe834f57e91bd35f4893c089424baae2a5ae57363e4bd96` |
| `stage2_hist_Oldpeak.png` | `7b1c3de6bbc12115f066aca10437451534b6329d7536bc9cb578b738595fccc4` |
| `stage2_hist_RestingBP.png` | `199b47bae4748ee649c0f4ffdbe09fa2ab7f79d070e90eed20ef28cb0b1a0dd0` |
| `stage2_univariate.json` | `8215c1ed3dcd94f63ca031c0f98b9c343cc34926ea9f64ab0ff6e540a6bba29e` |
| `stage3_bivariate.json` | `eb97fcdb3526c71475e448c9fb37d514f2d90a7e665eae7514dd2056fa5a0143` |
| `stage3_correlation_heatmap.png` | `83b5cede34575e00e93f27d740290912ea444650019bc4e5147e0378981cd4dd` |
| `stage3_overlay_Age.png` | `2693d044f660b554c886e03816402f941c1aa357685f7683620acc04470345bd` |
| `stage3_overlay_Cholesterol.png` | `7aebafb8af86c1b82d7ccca164c7b124fc24f3067fb9f4f26c15fe356fb04ce4` |
| `stage3_overlay_MaxHR.png` | `6359b8a2d98c544a6c24551d4bca2d2215d3f063b51ec1f63f19cd1db30c338b` |
| `stage3_overlay_Oldpeak.png` | `aa8f3017a2976225cbc3046672007e57a6a0bdfe89040da740dc011065cc2350` |
| `stage3_overlay_RestingBP.png` | `dfac89798c9ffce314d3685be89d783ae017bbb431c4e83754fddfa1d001b81b` |
| `stage4_assumptions.json` | `bd519b5e357b06efd3a746c7d65fdfcd31ae3b87f0a3ecfe7e05e9faf777717b` |
| `stage4_empirical_logit_Age.png` | `f903afc51782af873b5736f10511eaaa71129b99ff0becd0cb0925ec991719c0` |
| `stage4_empirical_logit_Cholesterol.png` | `95f95707219a62aa7b65a4ac510e8f70cfa4933ce86e58b8048dc05c58943fc4` |
| `stage4_empirical_logit_MaxHR.png` | `538ad56be7f8727bcffbd4b22227b53228ad2e5b53dc370b05a0befb17643161` |
| `stage4_empirical_logit_Oldpeak.png` | `79862a1a561616da7e45ccc1b7f20358b0a6e8676934afa3e329daa6d3dd844d` |
| `stage4_empirical_logit_RestingBP.png` | `abca66783065885a57b1b8dad9e0a51a9a3d11d75268b1dded1e73a7edc1686b` |

