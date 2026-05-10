#!/usr/bin/env python3
"""Append Sections 15 and 16 to UGICIS.ipynb.

15 = Leakage-Safe Early Risk Track (drops every BP-derived feature).
16 = Clinical Routing as a deterministic rule (explicit scope statement).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "UGICIS.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS: list[dict] = []

# ============================================================================
# Section 15 — Leakage-Safe Early Risk Track
# ============================================================================

CELLS.append(
    md(
        """## 15. Leakage-Safe Early Risk Track (Diagnostic vs Risk Separation)

### Why this section exists

The Dryad data dictionary defines `htn_stage` **literally as ranges of blood-pressure
measurements** (Grade 1 = 140-159 / 90-99 mmHg, Grade 2 = 160-179 / 100-109,
Grade 3 >= 180/110), and `htn_now` includes `new_dx`, which is itself set from BP at the
clinic visit.  Any model that uses **`bp_systolic`, `bp_diastolic`, MAP, pulse pressure,
hypertensive urgency, isolated systolic flag, BP-missingness flags, or features derived
from BP via PCA / clustering / IsolationForest** is therefore predicting the target *from
the target's own definition*.  That is target leakage by construction, and it is the
reason Sections 7-11 reach ROC-AUC > 0.96 on `htn_now`.

We split the modelling problem into two scientifically distinct tracks:

| Track | Inputs | What it answers | Honest interpretation |
|---|---|---|---|
| **A. Diagnostic / Triage** (Sections 7-11) | All features incl. BP | "Given this patient's BP today, which severity bucket are they in?" | Effectively a calculator; useful as a **screening rule**, not as prediction |
| **B. Early Risk** (this section) | **No BP-derived features at all** | "Without measuring BP, how likely is this patient to be hypertensive based on demographics, ART history, lifestyle, and facility?" | True **prediction** task; the headline science |

This is the framing we will take to publication: keep the Diagnostic track as a
deterministic-feeling reference and report the Early Risk track as the model that earns
clinical value (it tells the front desk who to prioritise for BP measurement).
"""
    )
)

CELLS.append(
    code(
        """# Strict leakage exclusion. Anything BP-derived OR target-encoded is dropped.
import re

BP_DERIVED_FEATURES = {
    # raw and engineered BP
    "bp_systolic", "bp_diastolic", "bpfinal",
    "bp_systolic_was_missing", "bp_diastolic_was_missing",
    "bp_systolic_bin", "bp_systolic_bin_woe",
    "MAP", "pulse_pressure", "pp", "pp_x_overweight",
    "hypertensive_urgency", "isolated_systolic",
    "pp_tertile", "pp_tertile_woe",
    # BP-derived cluster / anomaly outputs
    "phenotype_cluster", "phenotype_cluster_woe",
    "anomaly_score", "is_anomaly",
}
# All natural-cubic-spline basis vectors built from BP_systolic (bp_spl_1, bp_spl_2, ...)
BP_DERIVED_FEATURES.update({c for c in X.columns if re.fullmatch(r"bp_spl_\\d+", c)})

# Every Weight-of-Evidence and target-encoding column uses the target during training.
TARGET_ENCODED = {c for c in X.columns if c.endswith("_woe") or c.endswith("_te") or c.endswith("_te_oof")}

# Per-patient and per-facility identifiers can be memorised; drop both.
ID_COLS = {"clinicid", "hc_code"}

EARLY_RISK_DROP = sorted((BP_DERIVED_FEATURES | TARGET_ENCODED | ID_COLS) & set(X.columns))
print(f"Dropping {len(EARLY_RISK_DROP)} leakage-prone features:")
for c in EARLY_RISK_DROP:
    print(f"  - {c}")

X_safe = X.drop(columns=EARLY_RISK_DROP).copy()
y_safe = y.copy()
print(f"\\nLeakage-safe matrix: X_safe shape {X_safe.shape}  |  y prevalence {y_safe.mean():.3f}")
print(f"Remaining features ({X_safe.shape[1]}): {list(X_safe.columns)}")
"""
    )
)

CELLS.append(
    code(
        """# Re-use the same cv_benchmark / evaluate_predictions harness from Section 10.
SAFE_MODEL_NAMES = ["LogisticRegression", "RandomForest", "LightGBM", "CatBoost", "XGBoost"]
print("Running 5-fold CV on the leakage-safe matrix (htn_now only)...")
safe_long, safe_oof = cv_benchmark(X_safe, y_safe, SAFE_MODEL_NAMES)
safe_long["target"] = "htn_now_norisk"

safe_summary = (safe_long.groupby("model")[["ROC_AUC", "PR_AUC", "Brier"]]
                          .mean().round(4).sort_values("ROC_AUC", ascending=False))
print("\\nLeakage-safe baselines (mean over folds):")
print(safe_summary.to_string())
"""
    )
)

CELLS.append(
    code(
        """# Aggregate + persist
safe_metrics = ["ROC_AUC", "PR_AUC", "Recall@90%Prec", "Brier", "CalibSlope", "fit_seconds"]
safe_leaderboard = (safe_long
                    .groupby(["target", "model"])[safe_metrics]
                    .agg(["mean", "std"])
                    .round(4))
safe_leaderboard.columns = [f"{m}_{stat}" for m, stat in safe_leaderboard.columns]
safe_leaderboard = safe_leaderboard.reset_index()

OUT = Path("reports/tables/leakage_safe_leaderboard.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
safe_leaderboard.to_csv(OUT, index=False)
print(f"Wrote {OUT}")
safe_leaderboard
"""
    )
)

CELLS.append(
    code(
        """# Honest head-to-head: Diagnostic (with BP) vs Early Risk (no BP), per model
diag = leaderboard[leaderboard.target == "htn_now"][["model", "ROC_AUC_mean", "PR_AUC_mean", "Brier_mean"]].copy()
diag.columns = ["model", "ROC_AUC_diag", "PR_AUC_diag", "Brier_diag"]
risk = safe_leaderboard[["model", "ROC_AUC_mean", "PR_AUC_mean", "Brier_mean"]].copy()
risk.columns = ["model", "ROC_AUC_risk", "PR_AUC_risk", "Brier_risk"]
combo = diag.merge(risk, on="model")
combo["ROC_AUC_drop"] = (combo["ROC_AUC_diag"] - combo["ROC_AUC_risk"]).round(4)
combo["PR_AUC_drop"] = (combo["PR_AUC_diag"] - combo["PR_AUC_risk"]).round(4)
combo = combo.sort_values("ROC_AUC_risk", ascending=False)
combo.to_csv("reports/tables/diagnostic_vs_earlyrisk.csv", index=False)
print("Diagnostic-track ROC-AUC vs honest Early-Risk ROC-AUC (drop = leakage value):")
combo
"""
    )
)

CELLS.append(
    code(
        """# Visualise the two tracks side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = combo.sort_values("ROC_AUC_risk", ascending=True)["model"].tolist()
x = np.arange(len(order)); w = 0.38
diag_vals = [combo.set_index("model").loc[m, "ROC_AUC_diag"] for m in order]
risk_vals = [combo.set_index("model").loc[m, "ROC_AUC_risk"] for m in order]
axes[0].barh(x - w/2, diag_vals, w, label="Diagnostic (with BP)", color="#cccccc")
axes[0].barh(x + w/2, risk_vals, w, label="Early Risk (no BP)", color="#1f77b4")
axes[0].set_yticks(x); axes[0].set_yticklabels(order)
axes[0].set_xlabel("ROC-AUC (5-fold CV mean)")
axes[0].set_xlim(0.5, 1.0)
axes[0].axvline(0.5, color="grey", lw=0.6, ls="--")
axes[0].set_title("htn_now: Diagnostic vs Early Risk", fontweight="bold")
axes[0].legend(loc="lower right")

# OOF ROC of best Early-Risk model
from sklearn.metrics import roc_curve
best_safe = combo.iloc[0]["model"]
p_safe = safe_oof[best_safe]
fpr, tpr, _ = roc_curve(y_safe, p_safe)
auc_safe = roc_auc_score(y_safe, p_safe)
axes[1].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Early Risk: {best_safe} (AUC={auc_safe:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.6)
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
axes[1].set_title("Honest OOF ROC — best Early Risk model", fontweight="bold")
axes[1].legend(loc="lower right"); axes[1].grid(alpha=0.3)
plt.tight_layout()
fig_path = Path("reports/figures/modelling/diagnostic_vs_earlyrisk.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")
"""
    )
)

CELLS.append(
    code(
        """# Tune LightGBM on the leakage-safe matrix (smaller study, same protocol)
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS_SAFE = 30
spw_safe = float((len(y_safe) - y_safe.sum()) / max(y_safe.sum(), 1))


def lgb_objective_safe(trial):
    params = {
        "objective": "binary", "metric": "average_precision",
        "boosting_type": "gbdt", "verbose": -1, "n_jobs": -1,
        "random_state": RANDOM_SEED, "scale_pos_weight": spw_safe,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + trial.number)
    fold_aps = []
    for k, (tr, va) in enumerate(inner.split(X_safe, y_safe)):
        X_tr, X_va = X_safe.iloc[tr], X_safe.iloc[va]
        y_tr, y_va = y_safe.iloc[tr], y_safe.iloc[va]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, p)
        fold_aps.append(ap)
        trial.report(float(np.mean(fold_aps)), step=k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_aps))


storage_safe = Path("reports/tables/optuna_lightgbm_htn_now_norisk.db")
study_safe = optuna.create_study(
    study_name="ugicis_lgbm_norisk",
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_SEED, n_startup_trials=8),
    pruner=MedianPruner(n_warmup_steps=1, n_startup_trials=5),
    storage=f"sqlite:///{storage_safe.as_posix()}",
    load_if_exists=True,
)
print(f"Optuna (Early-Risk track): {N_TRIALS_SAFE} trials -> {storage_safe}")
study_safe.optimize(lgb_objective_safe, n_trials=N_TRIALS_SAFE,
                    show_progress_bar=False, gc_after_trial=True)
print(f"Best inner-CV PR-AUC (Early-Risk): {study_safe.best_value:.4f}")
best_safe_params = study_safe.best_params
Path("reports/tables/best_params_lightgbm_norisk.json").write_text(json.dumps(best_safe_params, indent=2))
print("Best params saved.")
"""
    )
)

CELLS.append(
    code(
        """# Outer 5-fold CV with tuned LightGBM on the leakage-safe matrix
print("Refitting tuned LightGBM under outer 5-fold CV on the leakage-safe matrix...")
skf_safe = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
tuned_safe_rows, oof_safe_tuned = [], np.zeros(len(y_safe))
for fold, (tr, va) in enumerate(skf_safe.split(X_safe, y_safe), start=1):
    X_tr, X_va = X_safe.iloc[tr], X_safe.iloc[va]
    y_tr, y_va = y_safe.iloc[tr], y_safe.iloc[va]
    spw = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    m = lgb.LGBMClassifier(objective="binary", random_state=RANDOM_SEED,
                           n_jobs=-1, verbose=-1, scale_pos_weight=spw, **best_safe_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict_proba(X_va)[:, 1]
    oof_safe_tuned[va] = p
    row = {"fold": fold, **evaluate_predictions(y_va.values, p)}
    tuned_safe_rows.append(row)
tuned_safe_df = pd.DataFrame(tuned_safe_rows)
print("\\nPer-fold tuned LightGBM (Early Risk):")
print(tuned_safe_df.round(4).to_string(index=False))
print("\\nMean +/- SD:")
print(tuned_safe_df.drop(columns=["fold"]).agg(["mean", "std"]).round(4).T.to_string())
tuned_safe_df.to_csv("reports/tables/early_risk_tuned_lightgbm_metrics.csv", index=False)

import joblib
final_lgb_safe = lgb.LGBMClassifier(objective="binary", random_state=RANDOM_SEED,
                                    n_jobs=-1, verbose=-1, scale_pos_weight=spw_safe,
                                    **best_safe_params)
final_lgb_safe.fit(X_safe, y_safe)
joblib.dump(final_lgb_safe, "reports/tables/final_lgbm_htn_now_norisk.joblib")
print("Saved reports/tables/final_lgbm_htn_now_norisk.joblib")
"""
    )
)

CELLS.append(
    code(
        """# SHAP for the tuned Early-Risk LightGBM (the model with real clinical value)
import shap
expl = shap.TreeExplainer(final_lgb_safe)
sample = X_safe.sample(min(800, len(X_safe)), random_state=RANDOM_SEED)
sv = expl.shap_values(sample)
if isinstance(sv, list):  # binary may return list of two arrays in older shap
    sv = sv[1]
plt.figure(figsize=(8, 5))
shap.summary_plot(sv, sample, plot_type="bar", show=False)
plt.title("Early Risk track — SHAP feature importance (tuned LightGBM, no BP)",
          fontweight="bold")
plt.tight_layout()
fig_path = Path("reports/figures/modelling/shap_earlyrisk_lgbm.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")
"""
    )
)

# ============================================================================
# Section 16 — Routing scope
# ============================================================================

CELLS.append(
    md(
        """## 16. Clinical Routing — Deterministic Rule and Scope Statement

### What routing IS

A **deterministic rule** that maps the predicted hypertension severity (`htn_stage`)
plus the facility level (`hc4` from the Dryad dictionary: 1 = Health Center 4 facility,
0 = lower-tier clinic) onto a four-class **referral pathway**:

| Predicted `htn_stage` | Rule                                              | Route               |
|---|---|---|
| 0 | Normal BP                                          | `self_manage`       |
| 1 | Grade 1 (140-159 / 90-99)                          | `treat_at_facility` |
| 2 | Grade 2 (160-179 / 100-109)                        | `treat_at_facility` if `hc4 == 1` else `refer_district` |
| 3 | Grade 3 (>=180/110) — clinical urgency             | `refer_tertiary`    |

These routing rules are taken from the cluster-randomized HIV/HTN integrated-care trial
that produced the dataset (EDCTP-funded study, 52 Ugandan facilities); the staging
thresholds match the **2020 American Society of Hypertension / International Society of
Hypertension** guidelines, which the Dryad dictionary explicitly cites for `htn_stage`.

### What routing is NOT

* **Not a learned model.**  No classifier predicts the route directly; routing is a
  stage -> route lookup with a single facility-level branch.  The MissAware-MTL `route`
  head exists only as an auxiliary task during training (multi-task regularization);
  the production decision is the rule above.
* **Not geographic.**  The dataset has no GPS coordinates, no inter-facility distances,
  and no transport-time fields.  "Nearest facility" logic is **out of scope for v1**;
  we only know whether the patient's current clinic is HC4 (`hc4 == 1`) or lower.
* **Not a substitute for clinician judgment.**  A `review_clinician = True` flag fires
  when MC-Dropout uncertainty on `htn_now` exceeds the operational threshold (see
  Section 14); those cases are sent to a clinician regardless of the rule.

### Future work for routing

1. Acquire facility geo-coordinates and patient catchment data to add a true distance /
   travel-time term.
2. Augment with facility-level capability data (drug stock-outs, BP machine availability,
   on-site clinician hours) to make `treat_at_facility` recommendations safer.
3. Replace the deterministic stage -> route map with a constrained learned policy once
   labelled outcome-after-referral data is available.

This scope statement is referenced from the project README so that no reviewer can
mistake the routing component for a learned recommender or a geographic optimiser.
"""
    )
)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    nb["cells"].extend(CELLS)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Appended {len(CELLS)} cells -> {NB_PATH} (total cells {len(nb['cells'])})")


if __name__ == "__main__":
    main()
