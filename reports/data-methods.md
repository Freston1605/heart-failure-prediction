# Data Methods — Parts A-F

This report is rendered by `heart.reporting.data_methods`. Part A is the four-stage pre-split EDA over the raw dataset — every stage runs entirely before any train/test split. Parts B-F document the per-model treatment audit, the logistic-regression preparation methodology, the pre-declared class-imbalance ladder, the validation/calibration strategy, and the pre-declared severity gate. Regenerate deterministically with `PYTHONPATH=src python -m heart.reporting.data_methods`.

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

## Part B — Per-Model Treatment Audit

The audit follows the **Model-Specific Pipeline Pattern** (decision D033): one pinned raw dataset, split first, per-model pipelines fit on training rows only, unified evaluate() scoring. Depth policy is decision D028: the linear family is audited deep, the robust families light with written justifications.

Provenance: the loader pins SHA-256 `948420b084d8a3a0ca42b8419fce9aee175879e43f8aedf712377899a67aa49b`; the committed split is version `v1` (5 stratified folds, test fold 0); fail-closed: heart.data.split.assert_no_group_straddle raises SplitLeakageError if any near-duplicate group spans the train/test boundary.

Preprocessing chain (fit on training rows only): declared zero policy `median_impute` on `RestingBP`, `Cholesterol` (sentinel 0), then a ColumnTransformer with StandardScaler on the numeric columns and OneHotEncoder with handle_unknown='ignore' on the categorical columns.

### Summary table (whole battery)

| model | family | pipeline treatment chain | tuned keys | fixed params |
|---|---|---|---|---|
| `knn` | distance | `zero_policy -> features` -> `knn` | `n_neighbors`, `p`, `weights` | none |
| `lda` | discriminant | `zero_policy -> features` -> `lda` | `shrinkage`, `solver` | none |
| `logistic-regression-elasticnet` | linear | `zero_policy -> features` -> `logistic-regression-elasticnet` | `C`, `class_weight`, `l1_ratio` | max_iter=2000, random_state=42, solver='saga' |
| `logistic-regression-l1` | linear | `zero_policy -> features` -> `logistic-regression-l1` | `C`, `class_weight` | l1_ratio=1.0, max_iter=2000, random_state=42, solver='liblinear' |
| `logistic-regression-l2` | linear | `zero_policy -> features` -> `logistic-regression-l2` | `C`, `class_weight` | l1_ratio=0.0, max_iter=2000, random_state=42, solver='lbfgs' |
| `naive-bayes` | bayes | `zero_policy -> features` -> `naive-bayes` | `var_smoothing` | none |
| `qda` | discriminant | `zero_policy -> features` -> `qda` | `reg_param` | none |
| `random-forest` | ensemble | `zero_policy -> features` -> `random-forest` | `class_weight`, `max_depth`, `max_features`, `min_samples_leaf`, `min_samples_split`, `n_estimators` | random_state=42 |
| `svm` | kernel | `zero_policy -> features` -> `svm` | `C`, `class_weight`, `gamma`, `kernel` | probability=True, random_state=42 |
| `xgboost` | gradient-boosting | `zero_policy -> features` -> `xgboost` | `colsample_bytree`, `learning_rate`, `max_depth`, `n_estimators`, `reg_lambda`, `subsample` | eval_metric='logloss', random_state=42 |

*Artifact:* `treatment_audit.json`

### Deep audit — the linear family (decision D028)

The linear-family members `lda`, `logistic-regression-elasticnet`, `logistic-regression-l1`, `logistic-regression-l2`, `qda` are audited at deep depth: the shared preprocessing chain is rendered step by step from the actual sklearn objects (`heart.data.pipeline`), followed by the fitted train-side statistics.

#### Shared preprocessing chain (actual sklearn objects)

| step | transformer | columns | parameters |
|---|---|---|---|
| `zero_policy` | ZeroMedianImputer | n/a | columns=['RestingBP', 'Cholesterol'], sentinel=0 |
| `features` | ColumnTransformer | n/a (sub-transformers below) | n_jobs=None, remainder='drop', sparse_threshold=0.3, verbose=False, verbose_feature_names_out=True |
| `features:numeric` | StandardScaler | `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak` | copy=True, with_mean=True, with_std=True |
| `features:categorical` | OneHotEncoder | `Sex`, `ChestPainType`, `FastingBS`, `RestingECG`, `ExerciseAngina`, `ST_Slope` | categories='auto', drop=None, dtype='float', feature_name_combiner='concat', handle_unknown='ignore', max_categories=None, min_frequency=None, sparse_output=False |

*Artifact:* `treatment_audit.json`

#### Fitted train-side statistics (fit on the 734 training rows only)

Declared zero policy and fitted medians:

| column | declared sentinel | fitted median |
|---|---|---|
| `Cholesterol` | 0 | 239.5 |
| `RestingBP` | 0 | 130.0 |

*Artifact:* `treatment_audit.json`

StandardScaler on the numeric columns (fitted mean/spread):

| numeric column | fitted mean | fitted scale |
|---|---|---|
| `Age` | 53.483651 | 9.469678 |
| `RestingBP` | 132.726158 | 18.252392 |
| `Cholesterol` | 244.26158 | 52.543278 |
| `MaxHR` | 137.531335 | 25.394403 |
| `Oldpeak` | 0.895095 | 1.085891 |

*Artifact:* `treatment_audit.json`

OneHotEncoder on the categorical columns (observed categories):

| categorical column | observed categories |
|---|---|
| `Sex` | `F`, `M` |
| `ChestPainType` | `ASY`, `ATA`, `NAP`, `TA` |
| `FastingBS` | `0`, `1` |
| `RestingECG` | `LVH`, `Normal`, `ST` |
| `ExerciseAngina` | `N`, `Y` |
| `ST_Slope` | `Down`, `Flat`, `Up` |

Expanded-feature count after the chain: **21** — the count the declared `one-hot-drop-first` candidate would shrink (part C).

*Artifact:* `treatment_audit.json`

#### Deep member: LDA (lda)

- family: discriminant
- estimator: Linear discriminant analysis; solver and shrinkage are tuned (shrinkage only applies to the lsqr solver).
- tuned space keys: `shrinkage`, `solver`
- fixed params: none
- expanded-feature count this member trains on: 21

*Artifact:* `treatment_audit.json`

#### Deep member: Logistic Regression (ElasticNet) (logistic-regression-elasticnet)

- family: linear
- estimator: logistic regression with elastic-net mixing weight tuned in (0, 1) (solver saga); the tuned strength is C and the class-weight policy.
- tuned space keys: `C`, `class_weight`, `l1_ratio`
- fixed params: max_iter=2000, random_state=42, solver='saga'
- expanded-feature count this member trains on: 21

*Artifact:* `treatment_audit.json`

#### Deep member: Logistic Regression (L1) (logistic-regression-l1)

- family: linear
- estimator: logistic regression with l1_ratio=1 (solver liblinear); the tuned strength is C and the class-weight policy.
- tuned space keys: `C`, `class_weight`
- fixed params: l1_ratio=1.0, max_iter=2000, random_state=42, solver='liblinear'
- expanded-feature count this member trains on: 21

*Artifact:* `treatment_audit.json`

#### Deep member: Logistic Regression (L2) (logistic-regression-l2)

- family: linear
- estimator: logistic regression with l1_ratio=0 (solver lbfgs); the tuned strength is C and the class-weight policy.
- tuned space keys: `C`, `class_weight`
- fixed params: l1_ratio=0.0, max_iter=2000, random_state=42, solver='lbfgs'
- expanded-feature count this member trains on: 21

*Artifact:* `treatment_audit.json`

#### Deep member: QDA (qda)

- family: discriminant
- estimator: Quadratic discriminant analysis; covariance regularisation is tuned.
- tuned space keys: `reg_param`
- fixed params: none
- expanded-feature count this member trains on: 21

*Artifact:* `treatment_audit.json`

### Light audit — robust families (decision D028)

The robust-family members `knn`, `naive-bayes`, `random-forest`, `svm`, `xgboost` are audited at light depth. Each carries a declared robustness justification — a written declaration of why the transform/imputation/encoding candidates do not bind for the family, not a measured claim; a battery change that adds an unjustified robust member fails the audit build loudly.

#### Light member: k-Nearest Neighbours (knn)

- family: distance
- estimator: k-nearest neighbours on the scaled feature space; neighbourhood size, weighting, and Minkowski power are tuned.
- tuned space keys: `n_neighbors`, `p`, `weights`
- fixed params: none

Robustness justification: The k-nearest-neighbour vote is taken on distances computed in the standardized space the shared chain produces, so the metric the model uses is already the scaled one; a monotone transform preserves neighbour orderings up to scale and the committed median imputation is what the distances actually see, so the transform and imputation rungs are not re-measured for this family.

*Artifact:* `treatment_audit.json`

#### Light member: Naive Bayes (naive-bayes)

- family: bayes
- estimator: Gaussian Naive Bayes; only the variance-smoothing floor is tuned.
- tuned space keys: `var_smoothing`
- fixed params: none

Robustness justification: Gaussian naive Bayes estimates one mean and one variance per feature, so a monotone power transform would shift those estimates but cannot change the model's shape; the committed median imputation already keeps every row usable, and the one-hot keep-all encoding matches the independent-feature assumption the model declares, so the transform, imputation, and encoding rungs are not re-measured for this family.

*Artifact:* `treatment_audit.json`

#### Light member: Random Forest (random-forest)

- family: ensemble
- estimator: Random forest of pruned trees; ensemble size, tree depth, split counts, feature sampling, and class weighting are tuned.
- tuned space keys: `class_weight`, `max_depth`, `max_features`, `min_samples_leaf`, `min_samples_split`, `n_estimators`
- fixed params: random_state=42

Robustness justification: Axis-aligned leaf splits are invariant to any monotone transform of a single feature, so the power-transform rungs cannot change the trees this family grows; the trees never scale features and the committed median imputation is the only missing-value policy the chain applies, so the transform and encoding rungs are not re-measured for this family.

*Artifact:* `treatment_audit.json`

#### Light member: Support Vector Machine (svm)

- family: kernel
- estimator: Support-vector classifier with calibrated probabilities; C, kernel, and gamma are tuned.
- tuned space keys: `C`, `class_weight`, `gamma`, `kernel`
- fixed params: probability=True, random_state=42

Robustness justification: Kernel evaluations happen in the standardized space the shared chain produces, and the tuned C/gamma budget already absorbs any feature-shape change a transform would introduce; the committed median imputation feeds the same standardized features regardless, so the transform, imputation, and encoding rungs are not re-measured for this family.

*Artifact:* `treatment_audit.json`

#### Light member: XGBoost (xgboost)

- family: gradient-boosting
- estimator: Gradient-boosted trees (XGBoost); boosting rounds, depth, learning rate, subsampling, and L2 regularisation are tuned.
- tuned space keys: `colsample_bytree`, `learning_rate`, `max_depth`, `n_estimators`, `reg_lambda`, `subsample`
- fixed params: eval_metric='logloss', random_state=42

Robustness justification: Split-gain trees are invariant to monotone per-feature transforms and the booster handles whatever the shared imputer hands it, so the power-transform rungs cannot change the learned structure; the committed one-hot keep-all encoding already feeds the booster numeric columns only, so the transform and encoding rungs are not re-measured for this family.

*Artifact:* `treatment_audit.json`

## Part C — Logistic-Regression Preparation Methodology (six-checklist)

The logistic-regression family is prepared through a fixed six-point checklist. Every number in this part is cited from a generated part A artifact or from code constants — nothing is hand-assembled (D025).

### Categorical encoding: one-hot keep-all with handle_unknown=ignore, and the drop-first candidate

The implemented encoding is one-hot *keep-all*: `heart.data.pipeline.build_preprocessing_pipeline` builds `OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)` over the categorical columns, so an unseen level at predict time degrades to an all-zero row instead of crashing and no observed category is dropped. The classic dummy-trap argument (a redundant level collinear with the intercept) targets unpenalized OLS; the linear family here is regularized, so the redundant column's coefficient is shrunk by the same penalty as every other feature. The drop-first alternative is not discarded, though: it is pre-declared as the `one-hot-drop-first` candidate (part D's register) and will be measured, not assumed.

### Standardization under regularization

Standardization sits *inside* the pipeline (the `features` step of the same module), so the regularization penalty applies equally to every feature regardless of raw scale, and the StandardScaler is fit on training rows only (`heart.data.pipeline.fit_preprocessing`); the fitted mean and spread per numeric column are recorded in part B (`treatment_audit.json`), not re-derived here.

### Skew and polynomial handling reconciled with already-engineered features

Part of the skew/shape story is already answered by the engineered features from M001/S04 (`heart/features/engineering.py`): `age_band`, `age_band_sex`, `cholesterol_age_ratio`, `cholesterol_restingbp_ratio`, `exercise_ecg_group`, `hr_reserve`, `max_hr_percent_predicted`, `metabolic_risk_count`, `oldpeak_slope_interaction`, `rate_pressure_product`. The remaining raw skew is not hand-corrected either: the power-transform rungs are pre-declared as the `yeo-johnson-power-transform` and `box-cox-power-transform` candidates and measured through the part F gate. The stage-2 skew table records the raw skews; the largest absolute skews are:

| column | skew (raw, stage 2) |
|---|---|
| `Oldpeak` | 1.0229 |
| `Cholesterol` | -0.6101 |
| `Age` | -0.1959 |

*Artifact:* `stage2_univariate.json`

The empirical-logit decile check over 5 numeric feature(s) is the linearity-of-logit diagnostic behind the same transform candidates.

*Artifact:* `stage4_assumptions.json`

### Imputation toolbox incl. missingness flags

Imputation is the committed `ZeroMedianImputer` (`heart/data/quality.py`): the declared sentinel zeros become missing and are filled with the fit-frame median (the declared policy is recorded in part A stage 1; the fitted medians are in part B). The declared toolbox next rungs are `knn-imputer` (a K-nearest-neighbours imputer fit on training rows only) and `missingness-flag-columns` (one binary missing-indicator column per feature with any observed missing value) — both pre-declared in part D's register, measured rather than assumed.

### Class imbalance runs through the part D ladder

Class imbalance is not handled here: every imbalance action runs through the ordered pre-declared ladder in part D.

### Validation and calibration honesty

Validation and calibration honesty live in part E, including the recorded `calibration_did_not_improve_brier` honesty flag.

## Part D — Pre-Declared Class-Imbalance Ladder

The ladder is rendered strictly from the declared plan artifact (`treatment_plan.json`), which mirrors `heart/eval/treatment_declared.py` 1:1 (decision D032). Rung 0 is the documented incumbent and is not re-measured; every later rung is one declared imbalance candidate, applied inside the training fold only (never before the split).

| rung | ladder label | candidate key | mechanism | applies to |
|---|---|---|---|---|
| 0 | none -- incumbent pipeline with its tuned class-weight policy (documented, not re-measured) | — | none (documented, not re-measured) | whole battery (incumbent pipeline with its tuned class-weight policy) |
| 1 | class weights pinned | `class-weight-balanced` | estimator-param | `logistic-regression-l2`, `logistic-regression-l1`, `logistic-regression-elasticnet` |
| 2 | random undersampling | `random-undersampling` | sampling | `logistic-regression-l2`, `logistic-regression-l1`, `logistic-regression-elasticnet`, `lda`, `qda` |
| 3 | random oversampling | `random-oversampling` | sampling | `logistic-regression-l2`, `logistic-regression-l1`, `logistic-regression-elasticnet`, `lda`, `qda` |
| 4 | SMOTE | `smote-oversampling` | sampling | `logistic-regression-l2`, `logistic-regression-l1`, `logistic-regression-elasticnet`, `lda`, `qda` |
| 5 | ADASYN | `adasyn-oversampling` | sampling | `logistic-regression-l2`, `logistic-regression-l1`, `logistic-regression-elasticnet`, `lda`, `qda` |

*Artifact:* `treatment_plan.json`

### Gate dispositions (measured)

Gate-measured dispositions from `treatment_gate_verdicts.json` (one line per measured pair; the incumbent rung keeps its documented-not-re-measured status):

- rung 0 (incumbent): documented-not-re-measured — no gate outcome recorded.
- gate-measured: `treatment:yeo-johnson-power-transform/lda` — delta -0.0019 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:yeo-johnson-power-transform/logistic-regression-elasticnet` — delta -0.0018 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:yeo-johnson-power-transform/logistic-regression-l1` — delta -0.0016 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:yeo-johnson-power-transform/logistic-regression-l2` — delta -0.0026 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:yeo-johnson-power-transform/qda` — delta +0.0020 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:box-cox-power-transform/lda` — delta +0.0005 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:box-cox-power-transform/logistic-regression-elasticnet` — delta -0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:box-cox-power-transform/logistic-regression-l1` — delta +0.0003 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:box-cox-power-transform/logistic-regression-l2` — delta +0.0005 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:box-cox-power-transform/qda` — delta +0.0009 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:knn-imputer/lda` — delta -0.0000 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:knn-imputer/logistic-regression-elasticnet` — delta +0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:knn-imputer/logistic-regression-l1` — delta -0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:knn-imputer/logistic-regression-l2` — delta -0.0002 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:knn-imputer/qda` — delta -0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:missingness-flag-columns/lda` — delta +0.0061 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:missingness-flag-columns/logistic-regression-elasticnet` — delta +0.0053 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:missingness-flag-columns/logistic-regression-l1` — delta +0.0056 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:missingness-flag-columns/logistic-regression-l2` — delta +0.0051 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:missingness-flag-columns/qda` — delta -0.0014 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:one-hot-drop-first/lda` — delta -0.0019 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:one-hot-drop-first/logistic-regression-elasticnet` — delta -0.0052 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:one-hot-drop-first/logistic-regression-l1` — delta -0.0011 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:one-hot-drop-first/logistic-regression-l2` — delta -0.0069 on roc_auc, significance significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:one-hot-drop-first/qda` — delta -0.0046 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:class-weight-balanced/logistic-regression-elasticnet` — delta +0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:class-weight-balanced/logistic-regression-l1` — delta +0.0000 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:class-weight-balanced/logistic-regression-l2` — delta +0.0000 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-undersampling/lda` — delta -0.0009 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-undersampling/logistic-regression-elasticnet` — delta +0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-undersampling/logistic-regression-l1` — delta +0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-undersampling/logistic-regression-l2` — delta -0.0004 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-undersampling/qda` — delta -0.0007 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-oversampling/lda` — delta -0.0012 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-oversampling/logistic-regression-elasticnet` — delta -0.0006 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-oversampling/logistic-regression-l1` — delta -0.0012 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-oversampling/logistic-regression-l2` — delta -0.0009 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:random-oversampling/qda` — delta +0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:smote-oversampling/lda` — delta -0.0008 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:smote-oversampling/logistic-regression-elasticnet` — delta +0.0001 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:smote-oversampling/logistic-regression-l1` — delta -0.0003 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:smote-oversampling/logistic-regression-l2` — delta -0.0004 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:smote-oversampling/qda` — delta -0.0004 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:adasyn-oversampling/lda` — delta -0.0017 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:adasyn-oversampling/logistic-regression-elasticnet` — delta -0.0004 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:adasyn-oversampling/logistic-regression-l1` — delta -0.0015 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:adasyn-oversampling/logistic-regression-l2` — delta -0.0004 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.
- gate-measured: `treatment:adasyn-oversampling/qda` — delta -0.0014 on roc_auc, significance not significant at alpha=0.05, standing ok, verdict rejected.

## Part E — Validation and Calibration Strategy

Every declared candidate is scored with repeated stratified cross-validation (`heart/eval/repeated_cv.py`): the committed split's 5 folds re-drawn across 5 differently-seeded repeats, every fold model fit on its training fold only, so a candidate's repeated-CV fold deltas stay paired across the same fold structure.

The significance layer reads those paired deltas with the Nadeau-Bengio corrected resampled t-test (`heart/eval/significance.py`, `corrected_resampled_t_test`) at alpha=0.05; the same alpha is fixed in the part F gate constants.

Probability calibration is measured, not assumed: when the calibrated model fails to improve the Brier score, the pipeline records the honesty flag `calibration_did_not_improve_brier` instead of shipping the calibration silently.

The current committed selection report (`reports/selection.md`) carries that flag; this part cites it textually and copies none of its numbers (D025) — MLflow-derived numbers stay in their own reports.

Gate-measured dispositions (artifact `treatment_gate_verdicts.json`): 48 measured pair(s), 0 accepted, 48 rejected at split_version `v1`; every rejection keeps the incumbent pipeline. The committed selection report's winner standing is unchanged by the gate (no acceptance re-ran the battery).

## Part F — Pre-Declared Severity Gate (fixed before any candidate number exists)

Severity gate, pre-declared before any candidate number existed: a candidate treatment is accepted only if (a) its delta on repeated-CV roc_auc against the incumbent is at least +0.0100, (b) the paired significance layer reads that delta as significant at alpha=0.05, and (c) the treated model's standing versus the baseline and the runner-up on the leaderboard does not worsen; every candidate that fails any of the three keeps the incumbent pipeline and is recorded as rejected.

The significance layer reads the delta through the paired significance layer at alpha=0.05: a delta the corrected resampled t-test does not read as significant keeps the incumbent, regardless of size.

| declared constant | value |
|---|---|
| gate metric | `roc_auc` |
| minimum repeated-CV ROC-AUC delta | +0.0100 |
| winner-standing rule | the treated model's standing versus the baseline and the runner-up on the leaderboard does not worsen |
| paired significance alpha | 0.05 |

*Artifact:* `treatment_plan.json`

Declaration of antecedence: the gate metric, the delta threshold, the significance alpha, and the winner-standing rule above were fixed in `heart/eval/treatment_declared.py` (decision D032) before any candidate was measured; the declared plan artifact records `declared_before_any_candidate_number: true`.

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
| `treatment_audit.json` | `791d878939bc5ed6985089f96d2560c0af29a2d929db078d5c58d425cd6e90a6` |
| `treatment_gate_verdicts.json` | `110a10f5db6f04f3cfd7d67284c970bd6840f81e7e24b08d322827f52c0f8811` |
| `treatment_plan.json` | `fa082296e28f95a9345d857c494e4df65295cec8e6746e0ddf223cfb6d711ed0` |

