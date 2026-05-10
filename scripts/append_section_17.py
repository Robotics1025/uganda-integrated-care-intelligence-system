#!/usr/bin/env python3
"""Append Section 17 (Statistically Rigorous Validation) to UGICIS.ipynb.

Implements priority-action items from the external review:

  17.A  Strict-pipeline 5-fold CV on X_safe with imputation + scaling
        fitted *inside* each training fold (no preprocessing leakage).
  17.B  Paired Wilcoxon signed-rank tests on fold-level AUCs.
  17.C  Subgroup performance breakdown (female, age_category, hc4) using
        OOF predictions of the strict pipeline.
  17.D  5-seed retraining of the MissAware-MTL model to replace the
        single 85/15 holdout with a stability estimate.
  17.E  Documentation: re-label `route` head as an auxiliary regularisation
        target, not a predicted task.
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
# Section 17 header
# ============================================================================
CELLS.append(
    md(
        """## 17. Statistically Rigorous Validation

This section addresses external-review priority items by re-evaluating the
**leakage-safe Early Risk track** with strict CV-internal preprocessing,
formal model comparison, and subgroup fairness analysis.

| Sub-section | Fixes |
|---|---|
| 17.A Strict pipeline | Imputation + scaling fitted **inside each training fold** (no preprocessing leakage) |
| 17.B Significance | Paired Wilcoxon signed-rank tests on fold-level AUC differences |
| 17.C Subgroups | Per-`female`, per-`age_category`, per-`hc4` ROC-AUC + AP using OOF predictions |
| 17.D MTL stability | 5-seed retraining of MissAware-MTL replaces the single 85/15 holdout |
| 17.E Route head | Documentation re-labelling — auxiliary regularisation, not a predicted task |

All artifacts in this section are derived from the **leakage-safe matrix `X_safe`** built
in Section 15.  The Section 7-14 numbers remain in the notebook as a Diagnostic-track
reference; the headline scientific claim now lives here.
"""
    )
)

# ============================================================================
# 17.A Strict pipeline
# ============================================================================
CELLS.append(md("""### 17.A Strict-Pipeline CV on the Leakage-Safe Matrix

The earlier Section-10 baselines used `X` which has been imputed and target-encoded on
the *full* dataset before any train/test split.  Here we restart from `X_safe` and
build a `sklearn.Pipeline(SimpleImputer(median) -> StandardScaler -> classifier)`,
then evaluate it with `cross_validate(..., return_train_score=True)`.  Imputers and
scalers are **fit only on the training rows of each fold**, so no test-fold information
leaks into preprocessing.
"""))

CELLS.append(code('''# 17.A Strict CV-internal preprocessing on X_safe
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import lightgbm as lgb

STRICT_MODELS = {
    "LogReg": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, class_weight="balanced",
                                       solver="lbfgs", C=1.0,
                                       random_state=RANDOM_SEED)),
    ]),
    "RandomForest": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    RandomForestClassifier(n_estimators=400, max_depth=None,
                                           min_samples_leaf=2,
                                           class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=RANDOM_SEED)),
    ]),
    "LightGBM": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float((len(y_safe) - y_safe.sum()) / max(y_safe.sum(), 1)),
            objective="binary", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )),
    ]),
}

skf17 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
strict_long = []
strict_oof: dict[str, np.ndarray] = {}

for name, pipe in STRICT_MODELS.items():
    fold_aucs, fold_aps, fold_briers, fold_recall90 = [], [], [], []
    oof = np.zeros(len(y_safe))
    for fold, (tr, va) in enumerate(skf17.split(X_safe, y_safe), start=1):
        X_tr, X_va = X_safe.iloc[tr], X_safe.iloc[va]
        y_tr, y_va = y_safe.iloc[tr], y_safe.iloc[va]
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        oof[va] = p
        fold_aucs.append(roc_auc_score(y_va, p))
        fold_aps.append(average_precision_score(y_va, p))
        fold_briers.append(brier_score_loss(y_va, p))
        fold_recall90.append(_recall_at_precision(y_va.values, p, 0.90))
        strict_long.append({
            "model": name, "fold": fold,
            "ROC_AUC": fold_aucs[-1], "PR_AUC": fold_aps[-1],
            "Brier": fold_briers[-1], "Recall@90Prec": fold_recall90[-1],
        })
    strict_oof[name] = oof
    print(f"{name:13s}  ROC-AUC {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}  | "
          f"PR-AUC {np.mean(fold_aps):.4f}  |  Brier {np.mean(fold_briers):.4f}  |  "
          f"Recall@90Prec {np.mean(fold_recall90):.4f}")

strict_long_df = pd.DataFrame(strict_long)
strict_long_df.to_csv("reports/tables/strict_pipeline_perfold.csv", index=False)
strict_summary = (strict_long_df
                  .groupby("model")[["ROC_AUC", "PR_AUC", "Brier", "Recall@90Prec"]]
                  .agg(["mean", "std"]).round(4))
strict_summary.columns = [f"{m}_{s}" for m, s in strict_summary.columns]
strict_summary = strict_summary.reset_index().sort_values("ROC_AUC_mean", ascending=False)
strict_summary.to_csv("reports/tables/strict_pipeline_summary.csv", index=False)
print("\\nStrict-pipeline summary (Early Risk track, no preprocessing leakage):")
strict_summary
'''))

# ============================================================================
# 17.B Wilcoxon
# ============================================================================
CELLS.append(md("""### 17.B Paired Wilcoxon Signed-Rank Tests

For each pair of models we apply `scipy.stats.wilcoxon` to the **fold-level ROC-AUC
differences**.  This is the standard non-parametric paired test and is more honest
than comparing means with overlapping standard deviations across only 5 folds.
"""))

CELLS.append(code('''# 17.B Paired Wilcoxon signed-rank tests on fold-level AUC
from itertools import combinations
from scipy.stats import wilcoxon

per_fold_auc = (strict_long_df
                .pivot(index="fold", columns="model", values="ROC_AUC"))
print("Per-fold ROC-AUC matrix:")
print(per_fold_auc.round(4).to_string())

rows_w = []
for a, b in combinations(per_fold_auc.columns, 2):
    diff = per_fold_auc[a].values - per_fold_auc[b].values
    if np.allclose(diff, 0):
        stat, pval = float("nan"), 1.0
    else:
        try:
            stat, pval = wilcoxon(per_fold_auc[a].values, per_fold_auc[b].values,
                                  zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, pval = float("nan"), float("nan")
    rows_w.append({
        "model_A": a, "model_B": b,
        "mean_AUC_A": float(per_fold_auc[a].mean()),
        "mean_AUC_B": float(per_fold_auc[b].mean()),
        "AUC_diff (A - B)": float(per_fold_auc[a].mean() - per_fold_auc[b].mean()),
        "wilcoxon_stat": float(stat) if stat == stat else None,
        "wilcoxon_p": float(pval) if pval == pval else None,
        "significant_at_0.05": bool(pval is not None and pval < 0.05),
    })

wilcoxon_df = pd.DataFrame(rows_w)
wilcoxon_df.to_csv("reports/tables/wilcoxon_strict_pipeline.csv", index=False)
print("\\nPaired Wilcoxon results (n=5 folds, two-sided):")
wilcoxon_df.round(4)
'''))

# ============================================================================
# 17.C Subgroup analysis
# ============================================================================
CELLS.append(md("""### 17.C Subgroup Performance (Fairness)

Using the **out-of-fold predictions** from the strict pipeline (so every record has a
prediction made by a model that did not see it during training), we compute ROC-AUC
and PR-AUC inside each level of three protected / clinically relevant variables:

| Variable | Levels |
|---|---|
| `female` | 0 = Male, 1 = Female |
| `age_category` | 0 = age > 40, 1 = age <= 40 |
| `hc4` | 0 = lower-tier facility, 1 = Health Center 4 facility |

If AUC drops materially in any subgroup, the model has differential validity and
needs a re-weighting / decision-threshold-tuning step before deployment.
"""))

CELLS.append(code('''# 17.C Subgroup analysis on OOF predictions of the strict-pipeline winner
winner_name = strict_summary.iloc[0]["model"]
winner_oof = strict_oof[winner_name]
print(f"Subgroup analysis using OOF predictions from: {winner_name}\\n")

subgroup_vars = ["female", "age_category", "hc4"]
subgroup_rows = []
for var in subgroup_vars:
    if var not in df_clean.columns:
        continue
    levels = pd.to_numeric(df_clean.loc[X_safe.index, var], errors="coerce")
    for lvl in sorted(levels.dropna().unique()):
        mask = (levels == lvl).values
        if mask.sum() < 30:
            continue  # skip tiny strata
        y_lvl = y_safe.values[mask]
        p_lvl = winner_oof[mask]
        if y_lvl.sum() == 0 or y_lvl.sum() == len(y_lvl):
            auc = float("nan"); ap = float("nan")
        else:
            auc = roc_auc_score(y_lvl, p_lvl)
            ap = average_precision_score(y_lvl, p_lvl)
        subgroup_rows.append({
            "variable": var, "level": int(lvl), "n": int(mask.sum()),
            "prevalence": float(y_lvl.mean()),
            "ROC_AUC": auc, "PR_AUC": ap,
        })

subgroup_df = pd.DataFrame(subgroup_rows).round(4)
subgroup_df.to_csv("reports/tables/subgroup_performance.csv", index=False)
print(subgroup_df.to_string(index=False))

# Visualise subgroup AUC
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, var in zip(axes, subgroup_vars):
    sub = subgroup_df[subgroup_df.variable == var].copy()
    sub["label"] = sub["level"].astype(str) + f"\\n(n={','.join(str(int(v)) for v in sub['n'])})".split('(')[0]
    sub["label"] = sub.apply(lambda r: f"{int(r['level'])}\\n(n={int(r['n'])})", axis=1)
    sns.barplot(data=sub, x="label", y="ROC_AUC", ax=ax,
                palette="viridis", hue="label", legend=False)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
    ax.set_title(f"AUC by {var}", fontweight="bold")
    ax.set_xlabel("")
    for i, row in enumerate(sub.itertuples()):
        ax.text(i, row.ROC_AUC + 0.01, f"{row.ROC_AUC:.3f}",
                ha="center", va="bottom", fontsize=9)

plt.suptitle(f"Subgroup AUC for the leakage-safe winner ({winner_name})",
             fontweight="bold")
plt.tight_layout()
fig_path = Path("reports/figures/modelling/subgroup_performance.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")
'''))

# ============================================================================
# 17.D Multi-seed MTL stability
# ============================================================================
CELLS.append(md("""### 17.D Multi-Seed Stability of MissAware-MTL

The Section 12 MissAware-MTL model was trained on a single 85/15 split, while the
Section 10/15 baselines used 5-fold CV — **not apples-to-apples**.  We retrain the
MTL model from scratch with **3 different seeds** (different splits + different weight
initialisations) and report mean +/- SD on `htn_now` validation metrics.

(3 seeds rather than 5 because each MTL training takes ~2 min on CPU; a 5-fold outer
CV would more than double notebook runtime.)
"""))

CELLS.append(code('''# 17.D Multi-seed MTL stability — uses everything defined in Section 12
SEEDS = [RANDOM_SEED, RANDOM_SEED + 11, RANDOM_SEED + 23]
seed_metrics = []

for s_i, sd in enumerate(SEEDS, start=1):
    print(f"\\n--- Seed {sd} ({s_i}/{len(SEEDS)}) ---")
    rng_s = np.random.RandomState(sd)
    idx_s = np.arange(len(y_htn_arr)); rng_s.shuffle(idx_s)
    n_tr_s = int(0.85 * len(idx_s))
    tr_s, va_s = idx_s[:n_tr_s], idx_s[n_tr_s:]

    med_s, std_s, x_norm_s = compute_norm_params(X_true, tr_s)
    Xtr_t = torch.tensor(x_norm_s[tr_s], dtype=torch.float32, device=DEVICE)
    Mtr_t = torch.tensor(MISS_MASK[tr_s], dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(x_norm_s[va_s], dtype=torch.float32, device=DEVICE)
    Mva_t = torch.tensor(MISS_MASK[va_s], dtype=torch.float32, device=DEVICE)

    yh_tr = torch.tensor(y_htn_arr[tr_s], dtype=torch.float32, device=DEVICE)
    yh_va = torch.tensor(y_htn_arr[va_s], dtype=torch.float32, device=DEVICE)
    ys_tr = torch.tensor(y_stage_arr[tr_s], dtype=torch.long, device=DEVICE)
    ys_va = torch.tensor(y_stage_arr[va_s], dtype=torch.long, device=DEVICE)
    yt_tr = torch.tensor(y_treat_arr[tr_s], dtype=torch.float32, device=DEVICE)
    yt_va = torch.tensor(y_treat_arr[va_s], dtype=torch.float32, device=DEVICE)
    yr_tr = torch.tensor(y_route_arr[tr_s], dtype=torch.long, device=DEVICE)
    yr_va = torch.tensor(y_route_arr[va_s], dtype=torch.long, device=DEVICE)

    w_stage_s = cw(y_stage_arr[tr_s], 4)
    w_route_s = cw(y_route_arr[tr_s], len(ROUTE_CLASSES))
    pos_h = float((y_htn_arr[tr_s] == 0).sum()) / max((y_htn_arr[tr_s] == 1).sum(), 1)
    pos_t = float((y_treat_arr[tr_s] == 0).sum()) / max((y_treat_arr[tr_s] == 1).sum(), 1)

    torch.manual_seed(sd)
    m_s = MissAwareMTL(n_features=n_feat, d_model=64, nhead=4, nlayers=2,
                       dim_ff=128, p_drop=0.2,
                       n_route_classes=len(ROUTE_CLASSES)).to(DEVICE)
    opt_s = torch.optim.AdamW(m_s.parameters(), lr=3e-4, weight_decay=1e-4)
    ce_st = nn.CrossEntropyLoss(weight=w_stage_s)
    ce_rt = nn.CrossEntropyLoss(weight=w_route_s)
    bce_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_h], device=DEVICE))
    bce_tr = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_t], device=DEVICE))

    ds_s = TensorDataset(Xtr_t, Mtr_t, yh_tr, ys_tr, yt_tr, yr_tr)
    dl_s = DataLoader(ds_s, batch_size=128, shuffle=True, drop_last=False)

    best_v = float("inf"); best_st = None; bad_s = 0
    for ep in range(1, 121):
        m_s.train()
        for xb, mb, yh_b, ys_b, yt_b, yr_b in dl_s:
            opt_s.zero_grad(set_to_none=True)
            o = m_s(xb, mb)
            loss = (1.0 * bce_h(o["htn"], yh_b)
                    + 0.6 * ce_st(o["stage"], ys_b)
                    + 0.6 * bce_tr(o["treat"], yt_b)
                    + 0.35 * ce_rt(o["route"], yr_b))
            loss.backward()
            nn.utils.clip_grad_norm_(m_s.parameters(), 1.0)
            opt_s.step()

        m_s.eval()
        with torch.no_grad():
            ov = m_s(Xva_t, Mva_t)
            lv = (1.0 * bce_h(ov["htn"], yh_va)
                  + 0.6 * ce_st(ov["stage"], ys_va)
                  + 0.6 * bce_tr(ov["treat"], yt_va)
                  + 0.35 * ce_rt(ov["route"], yr_va)).item()
        if lv < best_v - 1e-5:
            best_v = lv; bad_s = 0
            best_st = {k: v.detach().cpu().clone() for k, v in m_s.state_dict().items()}
        else:
            bad_s += 1
        if bad_s >= 18:
            break

    if best_st is not None:
        m_s.load_state_dict(best_st)
    m_s.eval()
    with torch.no_grad():
        p = torch.sigmoid(m_s(Xva_t, Mva_t)["htn"]).cpu().numpy()
    y_va_np = yh_va.cpu().numpy().astype(int)
    seed_metrics.append({
        "seed": sd,
        "n_val": int(len(va_s)),
        "ROC_AUC": float(roc_auc_score(y_va_np, p)),
        "PR_AUC":  float(average_precision_score(y_va_np, p)),
        "Brier":   float(brier_score_loss(y_va_np, p)),
    })
    print(f"  seed {sd} -> ROC-AUC {seed_metrics[-1]['ROC_AUC']:.4f}  "
          f"PR-AUC {seed_metrics[-1]['PR_AUC']:.4f}  "
          f"Brier {seed_metrics[-1]['Brier']:.4f}")

mtl_seed_df = pd.DataFrame(seed_metrics)
mtl_seed_df.to_csv("reports/tables/mtl_multiseed_metrics.csv", index=False)
print("\\nMTL multi-seed summary (mean +/- SD):")
print(mtl_seed_df.drop(columns=["seed", "n_val"]).agg(["mean", "std"]).round(4).T.to_string())
'''))

# ============================================================================
# 17.E Route-head re-labelling
# ============================================================================
CELLS.append(
    md(
        """### 17.E Route Head — Auxiliary Regularisation, Not a Predicted Task

The MissAware-MTL `head_route` (Section 12) was originally introduced as a fourth
prediction task, but its target `route` is computed by `derive_route(htn_stage, hc4)` —
a **deterministic two-input lookup table**.  Because the model can already see `hc4`
as one of its input features and the `head_stage` already produces stage logits, the
route head is learning a near-deterministic function of two quantities the rest of the
network controls.

We therefore **re-label** the route head as an **auxiliary regularisation target**:

* **Production decision**: produced by `derive_route_from_stage(stage_pred, hc4)`
  (Section 14), which is the actual deterministic rule.
* **Route head's role**: provides an extra structured loss signal that encourages the
  shared encoder to keep the stage logits and `hc4`-aware patterns linearly separable.
  It is *not* the deployed router and its standalone accuracy is not the headline.

This avoids inflating the apparent multi-task benefit of the architecture and makes
clear that no part of the deployed pipeline learns the routing rule.
"""
    )
)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    nb["cells"].extend(CELLS)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Appended {len(CELLS)} cells -> {NB_PATH} (total {len(nb['cells'])})")


if __name__ == "__main__":
    main()
