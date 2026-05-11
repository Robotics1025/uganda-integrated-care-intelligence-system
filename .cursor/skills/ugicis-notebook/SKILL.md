---
name: ugicis-notebook
description: >-
  Deep context on the Uganda Integrated Care Intelligence System (UGICIS)
  notebook. Covers structure, section flow, variable names, modelling
  conventions, leakage-safe track, domain rules, and the lecturer's exact
  cell-structure style. Use whenever working on UGICIS.ipynb, adding new
  sections, debugging cells, extending models, or interpreting results.
---

# UGICIS Notebook

The **Uganda Integrated Care Intelligence System** notebook (`UGICIS.ipynb`) applies ML and statistics to a de-identified Uganda HIV/ART cohort to support integrated care decisions around hypertension (HTN).

- **Dataset:** `Dataset/Uganda_Int_HIV_HTN_baseline_EUopensci.deident.v3.csv` — N=2,645, 19 columns
- **Primary target:** `htn_now` (binary, ~13.6% prevalence)
- **Secondary targets:** `htn_stage`, `treat` (stratification and MTL)
- **197 cells total:** 64 markdown + 133 code; sections numbered 1–19 preceded by an Executive Summary

For full section-by-section details, library list, and variable catalogue see [reference.md](reference.md).

---

## Lecturer's cell structure style

This notebook follows the same style as the companion lab notebooks (Lab 3–6). Apply this style whenever adding or editing cells.

### Heading conventions

- **Main section:** `## N. Algorithm Name` (e.g., `## 7. Logistic Regression`)
- **Sub-section:** `#### Sub-task Title` in Title Case (e.g., `#### Training the Model`, `#### Confusion Matrix`)
- Keep markdown content minimal — a title and at most 2 lines of description. Do NOT write long prose paragraphs in markdown cells.

### One task per code cell

Each code cell does exactly ONE thing. Never combine training + evaluation + plotting in a single cell:

| Cell | Does only |
|------|-----------|
| `# training the model` | fit the model (5-fold CV loop) |
| `# classification metrics` | compute & print metrics dict |
| `# confusion matrix` | plot `ConfusionMatrixDisplay` |
| `# classification report` | `print(classification_report(...))` |
| `# classification report heatmap` | `sns.heatmap` on report DataFrame |
| `# [algorithm-specific plot]` | one visualisation (k-selection, coef bar, depth sweep, etc.) |

### Comment style

Use **lowercase descriptive comments** as the first line of every code cell:
```python
# training the model
# splitting the data
# classification metrics
# confusion matrix
# feature importance
```

### Standard evaluation for every classifier (sections 7–12)

Use `classification_metrics()` (defined once after Section 6) then always produce:

```python
# 1. metrics dict
metrics = classification_metrics(y, y_pred)

# 2. confusion matrix — its OWN cell
ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot()
plt.show()

# 3. classification report text — its OWN cell
print(classification_report(y, y_pred))

# 4. classification report heatmap — its OWN cell
report_df = pd.DataFrame(
    classification_report(y, y_pred, output_dict=True)
).transpose().drop("support", axis=1, errors="ignore")
sns.heatmap(report_df, annot=True, fmt=".2f", cmap="Blues")
plt.title("Classification Report — AlgorithmName")
plt.show()
```

For UGICIS, `y_pred` = `(oof_array >= 0.5).astype(int)` using the OOF probability array from 5-fold CV.

### Exercises

End each section with a bold exercise block:
```markdown
**Exercise**

1. ...
2. ...
```

---

## Two evaluation tracks

Always distinguish these two tracks before interpreting any result:

| Track | Feature matrix | Purpose | Headline AUC |
|-------|---------------|---------|-------------|
| **Early-risk / leakage-safe** | `X_safe` (17 cols) | Front-desk prioritisation — who gets BP measured | ~0.93–0.94 |
| **Diagnostic / triage** | `X` (47 cols, incl. BP readings, target-encoding, IDs) | Reference only — AUC inflated by label definition | >0.96 |

`X_safe` is built by dropping `EARLY_RISK_DROP` from `X`.

---

## Key constants and paths

```python
RANDOM_SEED          # single seed used everywhere
RAW_CSV              # path to Dataset/Uganda_Int_HIV_HTN_baseline_EUopensci.deident.v3.csv
PROJECT_ROOT         # repo root
TBL_DIR              # output directory for CSV result tables
FIG_DIR              # output directory for figures
EARLY_RISK_DROP      # columns excluded to build X_safe
CAT_COLS             # categorical columns cast before modelling
TARGETS = ["htn_now", "htn_stage", "treat"]
MODEL_NAMES          # ["LogisticRegression","RandomForest","LightGBM","CatBoost","XGBoost"]
SAFE_MODEL_NAMES     # same list used on X_safe
SEEDS = [RANDOM_SEED, RANDOM_SEED+11, RANDOM_SEED+23]
```

---

## Data and split conventions

- `-9` values → `NaN` on load; `parse_bp` splits `bpfinal` → `bp_systolic`, `bp_diastolic`.
- Smoke/alcohol recoded: `{1=Yes, 2=No}` → `{1=Yes, 0=No}`.
- **Train/test split:** 80/20, `stratify=htn_stage` → `df_train`, `df_test`.
- **CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)` throughout.
- All preprocessing (scaling, imputation, WoE, target encoding) fit **on train folds only**.

---

## Individual algorithm section pattern (Sections 7–12)

Each section follows the same template:

| Cell | Type | Content |
|------|------|---------|
| 1 | markdown | `## N. Algorithm Name` + 1-2 line description |
| 2 | markdown | `#### Training the Model` |
| 3 | code | `# training the model` — 5-fold CV, collect `oof_*` array |
| 4 | markdown | `#### Model Evaluation` |
| 5 | code | `# classification metrics` — call `classification_metrics()` |
| 6 | markdown | `#### Confusion Matrix` |
| 7 | code | `# confusion matrix` — `ConfusionMatrixDisplay` |
| 8 | markdown | `#### Classification Report` |
| 9 | code | `# classification report` — print text + heatmap |
| 10 | markdown | `#### [Algorithm-specific Title]` |
| 11 | code | `# [algorithm-specific visualisation]` (own cell) |
| 12 | markdown | `**Exercise**` block |

**OOF variables per section:**

| Section | Algorithm | OOF variable | Algorithm-specific extra |
|---------|-----------|-------------|--------------------------|
| 7 | Logistic Regression | `oof_lr` | Coefficient bar chart |
| 8 | KNN | `oof_knn` | K-selection curve (k=3–31) |
| 9 | Naive Bayes | `oof_nb` | Calibration curve |
| 10.A | Decision Tree | `oof_dt` | Depth sweep (2–10) |
| 10.B | Multi-Model Benchmark | via `oof_store` | Full leaderboard |
| 11.A–C | Random Forest | via benchmark | MDI importance bar chart |
| 12.A | SVM | `oof_svm` | Support vector count |
| 12.B | MTL Transformer | — | Advanced: MC-Dropout uncertainty |

---

## Feature groups

| Group | Examples | In `X_safe`? |
|-------|---------|-------------|
| BP-derived | `bp_systolic`, `bp_diastolic`, `MAP`, `pulse_pressure`, splines | No |
| Lifestyle composite | `lifestyle_risk`, `cardiometabolic_composite` | Yes |
| Cluster / anomaly | `phenotype_cluster`, `anomaly_score`, `is_anomaly` | No |
| Target encoding | `hc_code_te`, `hc_code_te_oof` | No |
| Demographics / clinical | age, sex, BMI, exercise, ART years | Yes |

---

## CV harness functions

```python
classification_metrics(y_true, y_pred)   # accuracy, precision, recall, F1 — defined after Section 6
cv_benchmark(X, y, model_names, cv, seed)
evaluate_predictions(y_true, y_pred, ...)
FoldWoE                                   # WoE fit inside each fold
```

---

## Validation hierarchy (Sections 15 → 17 → 18 → 19)

- **15** — `X_safe`, `EARLY_RISK_DROP`, SHAP for early-risk LightGBM
- **17** — strict in-fold pipelines, Wilcoxon, subgroup fairness, multi-seed stability
- **18** — `FoldWoE`, MTL loss-weight ablation, pooled SHAP across folds
- **19** — composite ablation, BP plausibility, MCAR tests, IsolationForest sensitivity, 5×2 CV t-test, primary-results table

---

## Conventions to always follow

1. **One task per code cell** — no combined train+evaluate+plot cells.
2. **Each graph in its own cell** — never embed `plt.show()` inside a multi-step cell.
3. **`#### Title Case`** sub-headings before each logical step.
4. **`# lowercase comment`** as the first line of every code cell.
5. **Single seed** — use `RANDOM_SEED` for every random state.
6. **Fit on train only** — scalers, imputers, WoE, target encoding must never see validation data.
7. **`classification_metrics()` helper** — always use the shared helper, never re-implement metrics inline.
8. **Report both tracks** — never quote a single AUC without stating which feature matrix was used.

---

## Known maintenance issues

- **Setup cell source ≠ outputs (fixed):** The full import block has been restored in Section 1.
- **Embedded PNG outputs:** Many cells store large base64 PNGs; avoid re-running all cells unnecessarily.
