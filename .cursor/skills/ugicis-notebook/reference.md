# UGICIS Notebook — Full Reference

## Section-by-section flow

| Section | Title | Purpose | Key functions / variables |
|---------|-------|---------|--------------------------|
| Intro | Executive Summary | Validation hierarchy, dual-track explanation, known limitations (N=2,645, no external cohort) | — |
| 1 | Setup | Imports, seed, path constants | `RANDOM_SEED`, `RAW_CSV`, `PROJECT_ROOT`, `TBL_DIR` |
| 2 | Load the Clinical Dataset | `pd.read_csv`, shape check (2645, 19) | `df`, `parse_bp` |
| 3.A | Normality Tests | Shapiro-Wilk, D'Agostino on continuous cols | `scipy.stats.shapiro`, `normaltest` |
| 3.B | Univariate Odds Ratios | Per-feature OR with 95% CI | `statsmodels` GLM Binomial, `univariate_odds_ratios.csv` |
| 3.C | Mutual Information | Feature–target MI ranking | `mutual_info_classif`, `mutual_information.csv` |
| 3.D | Multicollinearity | VIF + condition number | `variance_inflation_factor` |
| 3.E | MAR Test | Missing-at-random logistic test | `inject_mcar`, `inject_mnar` |
| 3.F | Bayesian Prevalence | Beta-Binomial posterior for `htn_now` rate | `prevalence_bootstrap_ci.csv` |
| 3.G | Phenotype Clustering | KMeans k=4 on numeric features | `KMeans`, `phenotype_cluster`, `phenotype_clusters.csv` |
| 3.H | Anomaly Detection | IsolationForest contamination=0.03 | `IsolationForest`, `anomaly_score`, `is_anomaly` |
| 3.I | Facility ICC | Intra-cluster correlation by `hc_code` | `facility_summary.csv` |
| 3.J | Separability | PCA/t-SNE feature-space visualisation | `PCA`, `TSNE` |
| 4 | Data Preprocessing | Categorical cast, 80/20 stratified split, scaling, Yeo-Johnson, imputation comparisons | `CAT_COLS`, `df_train`, `df_test`, `StandardScaler`, `MinMaxScaler`, `PowerTransformer` |
| 5.A | WoE Encodings | Weight-of-evidence per bin | `information_value.csv` |
| 5.B | Correlation Analysis | Pearson / Spearman heatmaps, Cramér's V | — |
| 5.C | Variable Distributions | KDE / histogram overlays | — |
| 5.D | Scatter Plots | Pairwise scatter coloured by `htn_now` | — |
| 5.E | Imputation | Simple mean fill: `df_clean.fillna(df_clean.mean(...))` | `df_clean` |
| 5.F | Feature Selection | Permutation importance + RFECV | `permutation_importance.csv`, `rfecv_selected_features.csv` |
| 5.G | Class Imbalance | Imbalance assessment, optional resampling strategy | — |
| 5.H | Pairplots / Manifold | PairGrid, t-SNE-style views | — |
| 5.I | Feature Summary | Consolidated report card | `feature_report_card.csv` |
| 6 | Build Modelling Matrix | Assemble `X` (47 cols) and `y` | `X`, `y` (prevalence 0.136) |
| 7 | Logistic Regression | 5-fold CV, coefficient bar chart, OOF ROC | `oof_lr`, `coef_accum`, `lr_df` |
| 8 | K-Nearest Neighbors | K-selection curve (k=3–31), best-k 5-fold CV, OOF ROC | `oof_knn`, `best_k`, `knn_df` |
| 9 | Naive Bayes | 5-fold CV, calibration curve, OOF ROC | `oof_nb`, `nb_df` |
| 10.A | Decision Tree | Depth sweep (2–10), 5-fold CV, feature importances | `oof_dt`, `best_depth`, `dt_final` |
| 10.B | Multi-Model Benchmark | LR, RF, LGB, CatBoost, XGB leaderboard across 3 targets | `leaderboard_long`, `oof_store`, `MODEL_NAMES` |
| 11.A–C | Random Forest | Benchmark summary, MDI importances (5-fold avg), confusion matrix | `imp_accum`, `oof_rf` |
| 11.D | LightGBM Tuning | Tuned vs untuned CV comparison | `best_params`, `oof_tuned` |
| 12.A | SVM | RBF kernel, 5-fold CV, support vector count | `oof_svm`, `svm_df` |
| 12.B | MTL Transformer | PyTorch MissAwareMTL, MC-Dropout uncertainty | `MissAwareMTL`, `model_mtl` |
| 13 | Model Comparison Summary | Robustness under missingness scenarios | `lgbm_final`, `rows` |
| 14 | Bar Charts / Clinical Routing | Route function; patient record formatter | `derive_route_from_stage` |
| 15 | Investigating the Best Model | Define `EARLY_RISK_DROP`, build `X_safe`, re-run benchmark on safe features, optional SHAP | `X_safe`, `y_safe`, `SAFE_MODEL_NAMES`, `shap_earlyrisk_lgbm.png` |
| 16 | ROC Curves | ROC plot for all models | `roc_curve`, `auc` |
| 17.A | Strict In-fold Pipelines | Preprocessing inside CV folds | `Pipeline`, `ColumnTransformer`, `SimpleImputer` |
| 17.B | Wilcoxon Test | Statistical comparison of fold AUCs | `scipy.stats.wilcoxon` |
| 17.C | Subgroup Fairness | Performance by demographic subgroups | — |
| 17.D | Multi-seed Stability | Repeat benchmark over `SEEDS` | `SEEDS` |
| 17.E | MTL Route Head Note | Route head zero-weighted regulariser explanation | PyTorch Transformer MTL |
| 18.A | FoldWoE Integration | WoE inside outer CV | `FoldWoE`, `X_safe_woe` |
| 18.B | Simple Classifier Parity | LR / NB parity check in same CV | — |
| 18.C | MTL Loss-weight Ablation | Grid over head loss weights | `train_mtl_with_weights` |
| 18.D | SHAP Pooled Across Folds | Aggregate SHAP values from all folds | `shap.TreeExplainer` |
| 18.E | Review Mapping Table | Maps open review items to notebook sections | — |
| 19.A | Cardiometabolic Ablation | Ablation of `cardiometabolic_composite` on `X_safe` | — |
| 19.B | Smoke/Alcohol Coding | Verify `{1=Yes, 0=No}` recoding | — |
| 19.C | BP Plausibility Bounds | Systolic/diastolic cap check (e.g. [30, 160] mmHg) | — |
| 19.D | Extended MCAR Tests | Pairwise Little's MCAR-style tests | `missingness_simulation_summary.csv` |
| 19.E | IsolationForest Sensitivity | Contamination parameter sweep | `IsolationForest` |
| 19.F | 5×2 CV t-test | Formal method comparison test | `scipy.stats.t` |
| 19.G | Primary Results Table | Final summary of all validated results | CSV export |
| 19.H | Closing Statement | Limitations and future work | — |

---

## Library list

### Core data
| Package | Usage |
|---------|-------|
| `pandas` | DataFrame operations throughout |
| `numpy` | Numerical arrays, random state |
| `pathlib.Path` | Path management |
| `json`, `joblib`, `pickle` | Serialisation |
| `glob`, `re`, `itertools` | Utilities |

### Visualisation
| Package | Usage |
|---------|-------|
| `matplotlib` | Base plots, bar charts, ROC curves |
| `seaborn` | Heatmaps, KDE/histogram, PairGrid |

### scikit-learn
| Module | Usage |
|--------|-------|
| `train_test_split` | 80/20 stratified split |
| `StratifiedKFold`, `KFold` | Cross-validation |
| `StandardScaler`, `MinMaxScaler` | Feature scaling |
| `PowerTransformer` | Yeo-Johnson transform |
| `SimpleImputer` | Missing value imputation |
| `Pipeline`, `ColumnTransformer` | In-fold preprocessing pipelines |
| `PolynomialFeatures`, `PCA` | Feature engineering |
| `TSNE` (manifold) | Dimensionality reduction visualisation |
| `mutual_info_classif` | Feature–target MI |
| `permutation_importance`, `RFECV` | Feature selection |
| `KMeans`, `silhouette_score` | Phenotype clustering |
| `IsolationForest` | Anomaly detection |
| `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB` | Classical classifiers |
| `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC` | Classical classifiers |
| `cross_val_predict` | CV predictions |
| `roc_curve`, `auc`, `average_precision_score`, `brier_score_loss` | Evaluation metrics |

### Boosting
| Package | Class |
|---------|-------|
| `lightgbm` | `LGBMClassifier` |
| `catboost` | `CatBoostClassifier` |
| `xgboost` | `XGBClassifier` |

### Deep learning
| Package | Usage |
|---------|-------|
| `torch` | Tensor operations |
| `torch.nn` | `TransformerEncoder`, `TransformerEncoderLayer`, `Linear`, loss functions |
| `torch.utils.data` | `DataLoader`, `TensorDataset` |

### Explainability
| Package | Usage |
|---------|-------|
| `shap` | `TreeExplainer`, summary_plot, bar_plot |

### Statistics
| Package | Usage |
|---------|-------|
| `scipy.stats` | `shapiro`, `normaltest`, `wilcoxon`, `chi2_contingency`, `t` |
| `statsmodels` | `GLM` (Binomial), `multipletests` (FDR), optional `BinomialBayesMixedGLM` |

---

## Variable and function catalogue

### DataFrames and matrices
| Name | Description |
|------|-------------|
| `df` | Raw loaded DataFrame (2645, 19) |
| `df_clean` | Post-parse, post-recode, mean-imputed DataFrame |
| `df_train` / `df_test` | 80/20 stratified split |
| `X` | Full modelling matrix (47 columns) |
| `y` | Target series for `htn_now` |
| `X_safe` | Leakage-safe matrix (17 columns, BP/TE/ID/cluster features dropped) |
| `y_safe` | Target for early-risk track (same as `y` but paired with `X_safe`) |
| `X_safe_woe` | `X_safe` with `FoldWoE`-encoded columns |
| `woe_input_df` | Intermediate frame fed into `FoldWoE` |

### Functions
| Name | Signature / purpose |
|------|-------------------|
| `parse_bp(bp_str)` | Splits `"120/80"` → `(bp_systolic, bp_diastolic)` |
| `cv_benchmark(X, y, model_names, cv, seed)` | Runs all named models with `StratifiedKFold`, returns per-fold metric dict |
| `evaluate_predictions(y_true, y_pred, y_prob)` | Returns `{roc_auc, pr_auc, brier, ...}` |
| `bootstrap_ci(stat_fn, data, n=1000, seed)` | Percentile bootstrap confidence interval |
| `inject_mcar(df, cols, rate)` | Randomly masks values for MAR/MCAR simulation |
| `inject_mnar(df, col, threshold)` | Masks values conditionally for MNAR simulation |
| `train_mtl_with_weights(X, y_dict, loss_weights, max_ep, patience, seed)` | Trains PyTorch MTL Transformer; returns trained model + metrics |

### Classes
| Name | Description |
|------|-------------|
| `FoldWoE` | sklearn-compatible transformer; fits WoE bins inside each CV fold to prevent leakage |
| PyTorch MTL module | `TransformerEncoder`-based model with `htn_now` head + auxiliary heads (`htn_stage`, `treat` route head) |

---

## Configuration parameter table

| Parameter | Value / note |
|-----------|-------------|
| `RANDOM_SEED` | Fixed integer; used in every random state |
| Train/test split | 80/20, `stratify=htn_stage` |
| CV folds | 5, shuffled, `random_state=RANDOM_SEED` |
| KMeans clusters | `n_clusters=4` (hardcoded) |
| IsolationForest contamination | `0.03` (sensitivity sweep in 19.E) |
| MTL training epochs | `max_ep=80` |
| MTL patience | `patience=12` |
| MTL route head loss weight | `0` (zero-weighted) |
| Multi-seed list | `[RANDOM_SEED, RANDOM_SEED+11, RANDOM_SEED+23]` |
| BP plausibility systolic range | ~[30, 160] mmHg (Section 19.C) |
| `X` column count | 47 |
| `X_safe` column count | 17 |
| Dataset shape | (2645, 19) |
| `htn_now` prevalence | ~13.6% |

---

## Raw dataset column reference (19 columns)

The dataset includes (exact names depend on the CSV header):
- `hc_code` — facility code (high-cardinality ID)
- `patient_id` or similar — patient identifier (excluded from `X_safe`)
- Demographic: `age`, `sex`
- Clinical: `bpfinal` (raw BP string), `htn_now`, `htn_stage`, `treat`
- Lifestyle: `smoke`, `alcohol`, `exercise`, BMI-related fields
- HIV/ART: `cd4` or similar ART markers

Derived on load:
- `bp_systolic`, `bp_diastolic` (from `parse_bp`)
- `MAP = (bp_systolic + 2 * bp_diastolic) / 3`
- `pulse_pressure = bp_systolic - bp_diastolic`
- `lifestyle_risk`, `cardiometabolic_composite`
