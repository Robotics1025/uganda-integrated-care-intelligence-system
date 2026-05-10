#!/usr/bin/env python3
"""Append Section 18 (Definitive Validation Layer) to UGICIS.ipynb.

Closes the remaining external-review items:

  18.A  Custom WoE / target-encoding transformer that fits INSIDE the
        cross-validation training fold only -- removes preprocessing leakage
        for ALL target-encoded features (not just imputation).
  18.B  MissAware-MTL evaluated under the SAME 5-fold outer stratified CV
        as the classical baselines (apples-to-apples).
  18.C  Multi-task loss-weight ablation: small grid over (W_STG, W_TRT, W_RTE)
        with W_HTN=1.0 fixed; reports htn_now PR-AUC for each cell.
  18.D  SHAP values pooled across all 5 outer folds (replaces the
        Section-8 single-fold SHAP), giving an honest global summary.
  18.E  Closing summary table mapping each review item to its remediation.
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
# Section 18 header
# ============================================================================
CELLS.append(
    md(
        """## 18. Definitive Validation Layer

Closes the remaining external-review priority items:

| Sub-section | Review item closed |
|---|---|
| 18.A WoE-inside-CV | "WoE encoding and MICE imputation computed on the full dataset" -- now ALL target-encoded features are fit per fold |
| 18.B MTL 5-fold CV | "MissAware-MTL evaluated on a single 85/15 split vs 5-fold CV for baselines" -- now apples-to-apples |
| 18.C MTL loss-weight ablation | "Multi-task loss weights are fixed heuristics with no ablation" |
| 18.D Pooled-fold SHAP | "SHAP computed on the last fold's model" -- now pooled across all 5 outer folds |
| 18.E Review-to-section map | Closing summary: every priority item -> notebook section |

After this section the leakage-safe Early Risk track in the notebook is the
**definitive scientific result**; everything earlier is supporting / Diagnostic-track context.
"""
    )
)

# ============================================================================
# 18.A WoE-inside-CV
# ============================================================================
CELLS.append(
    md(
        """### 18.A Custom WoE Transformer (Fit-Inside-Fold)

The Section 5.B WoE columns were computed once on the full `df_clean`, which
includes test-fold rows.  Here we expose the same encoding as a sklearn-compatible
transformer that **only sees the training half of each fold** during `fit`.

We rebuild a parallel feature matrix `X_safe_woe` that contains the demographic
features from `X_safe` plus *fold-internal* WoE encodings of the same low-cardinality
clinical features the original Section 5.B used, then re-run the strict 5-fold CV
(`Pipeline(SimpleImputer -> WoE -> StandardScaler -> classifier)`) to confirm the
honest Early Risk numbers do **not** rise once preprocessing leakage is fully
excluded.
"""
    )
)

CELLS.append(
    code(
        '''# 18.A WoE encoder that fits inside each training fold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class FoldWoE(BaseEstimator, TransformerMixin):
    """Compute Weight-of-Evidence per category, fit only on training rows.

    For each input column treated as categorical, we estimate
    ``WoE(level) = log( P(level | y=1) / P(level | y=0) )``
    with additive smoothing.  Unseen levels at transform time map to 0.
    """

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True).astype(int)
        self.columns_ = list(X.columns)
        self.maps_: dict[str, dict] = {}
        pos = float(max(y.sum(), 1))
        neg = float(max(len(y) - y.sum(), 1))
        for c in self.columns_:
            col = X[c].astype(object).where(X[c].notna(), "__nan__")
            tbl = pd.crosstab(col, y)
            tbl = tbl.reindex(columns=[0, 1], fill_value=0).astype(float)
            tbl[1] = (tbl[1] + self.smoothing) / (pos + self.smoothing * len(tbl))
            tbl[0] = (tbl[0] + self.smoothing) / (neg + self.smoothing * len(tbl))
            self.maps_[c] = dict(np.log(tbl[1] / tbl[0]))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for c in self.columns_:
            m = self.maps_[c]
            col = X[c].astype(object).where(X[c].notna(), "__nan__")
            out[f"{c}_woe"] = col.map(m).fillna(0.0).astype(float)
        return out.values

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_woe" for c in self.columns_])


# Pick the same low-cardinality clinical fields the original Section 5.B used
WOE_COLS = [c for c in
            ["age_category", "marital_status", "overweight", "alcohol",
             "bpmdate6mo", "exercise"]
            if c in df_clean.columns]
NUM_COLS = [c for c in X_safe.columns if c not in WOE_COLS]
print(f"WOE columns (fit per fold): {WOE_COLS}")
print(f"Numeric columns: {len(NUM_COLS)}")
'''
    )
)

CELLS.append(
    code(
        '''# 18.A Build the truly-leakage-free matrix and run strict CV
def make_strict_woe_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), NUM_COLS),
            ("woe", Pipeline([
                ("woe",   FoldWoE(smoothing=0.5)),
                ("scale", StandardScaler()),
            ]), WOE_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


# Build a feature frame that has the raw WoE source columns AND all numeric features
woe_input_df = pd.concat(
    [X_safe[NUM_COLS],
     df_clean.loc[X_safe.index, WOE_COLS].reset_index(drop=True)
                                      .set_index(X_safe.index)],
    axis=1,
)

print(f"WoE-input frame shape: {woe_input_df.shape}")

WOE_MODELS = {
    "LogReg":   LogisticRegression(max_iter=2000, class_weight="balanced",
                                   solver="lbfgs", C=1.0,
                                   random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                            class_weight="balanced_subsample",
                                            random_state=RANDOM_SEED),
}
try:
    import lightgbm as _lgb
    WOE_MODELS["LightGBM"] = _lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=float((len(y_safe) - y_safe.sum()) / max(y_safe.sum(), 1)),
        objective="binary", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
except Exception as e:
    print(f"LightGBM not available, skipping: {e}")

skf18 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
woe_long = []
woe_oof: dict[str, np.ndarray] = {}

for name, est in WOE_MODELS.items():
    oof = np.zeros(len(y_safe))
    fold_aucs, fold_aps, fold_briers, fold_r90 = [], [], [], []
    for fold, (tr, va) in enumerate(skf18.split(woe_input_df, y_safe), start=1):
        pipe = make_strict_woe_pipeline(est)
        Xtr, Xva = woe_input_df.iloc[tr], woe_input_df.iloc[va]
        ytr, yva = y_safe.iloc[tr], y_safe.iloc[va]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        oof[va] = p
        fold_aucs.append(roc_auc_score(yva, p))
        fold_aps.append(average_precision_score(yva, p))
        fold_briers.append(brier_score_loss(yva, p))
        fold_r90.append(_recall_at_precision(yva.values, p, 0.90))
        woe_long.append({"model": name, "fold": fold,
                         "ROC_AUC": fold_aucs[-1], "PR_AUC": fold_aps[-1],
                         "Brier": fold_briers[-1], "Recall@90Prec": fold_r90[-1]})
    woe_oof[name] = oof
    print(f"{name:13s}  ROC-AUC {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}  | "
          f"PR-AUC {np.mean(fold_aps):.4f}  |  Brier {np.mean(fold_briers):.4f}  |  "
          f"Recall@90Prec {np.mean(fold_r90):.4f}")

woe_long_df = pd.DataFrame(woe_long)
woe_long_df.to_csv("reports/tables/strict_pipeline_with_woe_perfold.csv", index=False)
woe_summary = (woe_long_df.groupby("model")[["ROC_AUC", "PR_AUC", "Brier", "Recall@90Prec"]]
                          .agg(["mean", "std"]).round(4))
woe_summary.columns = [f"{m}_{s}" for m, s in woe_summary.columns]
woe_summary = woe_summary.reset_index().sort_values("ROC_AUC_mean", ascending=False)
woe_summary.to_csv("reports/tables/strict_pipeline_with_woe_summary.csv", index=False)
print("\\nStrict-pipeline + WoE-inside-fold summary:")
woe_summary
'''
    )
)

# ============================================================================
# 18.B MTL inside 5-fold outer CV
# ============================================================================
CELLS.append(
    md(
        """### 18.B MissAware-MTL Inside the SAME 5-Fold Outer CV

Section 17.D reported 3 random seeds, which controls for split variance but is still
not the harness used for the classical baselines.  Here we drop the MTL into the
**identical `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)`**
splits used in Sections 10/15/17.A and report mean +/- SD.

Each fold trains the same architecture (Section 12) with early stopping on the
fold's own validation slice; we then take the *out-of-fold* htn_now probability for
that slice.  Total compute: ~5 x 2 minutes on CPU.
"""
    )
)

CELLS.append(
    code(
        '''# 18.B MissAware-MTL with 5-fold outer CV (apples-to-apples vs the baselines)
mtl_cv_rows = []
mtl_cv_oof_htn = np.full(len(y_htn_arr), np.nan)

skfM = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Review item #3: zero out route-head loss weight so the route head
# (which learns derive_route(stage, hc4) -- a deterministic lookup) cannot
# inflate apparent multi-task signal in this evaluation.
W_HTN_CV, W_STG_CV, W_TRT_CV, W_RTE_CV = 1.0, 0.6, 0.6, 0.0
print(f"Loss weights for 5-fold CV: HTN={W_HTN_CV}  STG={W_STG_CV}  TRT={W_TRT_CV}  RTE={W_RTE_CV}  (route head silenced)")

for fold, (tr_f, va_f) in enumerate(skfM.split(np.arange(len(y_htn_arr)), y_htn_arr),
                                     start=1):
    print(f"\\n--- Fold {fold}/5  (n_train={len(tr_f)} | n_val={len(va_f)}) ---")
    med_f, std_f, x_norm_f = compute_norm_params(X_true, tr_f)
    Xtr_t = torch.tensor(x_norm_f[tr_f], dtype=torch.float32, device=DEVICE)
    Mtr_t = torch.tensor(MISS_MASK[tr_f], dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(x_norm_f[va_f], dtype=torch.float32, device=DEVICE)
    Mva_t = torch.tensor(MISS_MASK[va_f], dtype=torch.float32, device=DEVICE)

    yh_tr = torch.tensor(y_htn_arr[tr_f], dtype=torch.float32, device=DEVICE)
    yh_va = torch.tensor(y_htn_arr[va_f], dtype=torch.float32, device=DEVICE)
    ys_tr = torch.tensor(y_stage_arr[tr_f], dtype=torch.long, device=DEVICE)
    ys_va = torch.tensor(y_stage_arr[va_f], dtype=torch.long, device=DEVICE)
    yt_tr = torch.tensor(y_treat_arr[tr_f], dtype=torch.float32, device=DEVICE)
    yt_va = torch.tensor(y_treat_arr[va_f], dtype=torch.float32, device=DEVICE)
    yr_tr = torch.tensor(y_route_arr[tr_f], dtype=torch.long, device=DEVICE)
    yr_va = torch.tensor(y_route_arr[va_f], dtype=torch.long, device=DEVICE)

    w_st = cw(y_stage_arr[tr_f], 4)
    w_rt = cw(y_route_arr[tr_f], len(ROUTE_CLASSES))
    pos_h = float((y_htn_arr[tr_f] == 0).sum()) / max((y_htn_arr[tr_f] == 1).sum(), 1)
    pos_t = float((y_treat_arr[tr_f] == 0).sum()) / max((y_treat_arr[tr_f] == 1).sum(), 1)

    torch.manual_seed(RANDOM_SEED + fold)
    m_f = MissAwareMTL(n_features=n_feat, d_model=64, nhead=4, nlayers=2,
                       dim_ff=128, p_drop=0.2,
                       n_route_classes=len(ROUTE_CLASSES)).to(DEVICE)
    opt_f = torch.optim.AdamW(m_f.parameters(), lr=3e-4, weight_decay=1e-4)
    ce_st = nn.CrossEntropyLoss(weight=w_st)
    ce_rt = nn.CrossEntropyLoss(weight=w_rt)
    bce_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_h], device=DEVICE))
    bce_tr = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_t], device=DEVICE))

    ds_f = TensorDataset(Xtr_t, Mtr_t, yh_tr, ys_tr, yt_tr, yr_tr)
    dl_f = DataLoader(ds_f, batch_size=128, shuffle=True, drop_last=False)

    best_v = float("inf"); best_state = None; bad = 0
    for ep in range(1, 121):
        m_f.train()
        for xb, mb, yh_b, ys_b, yt_b, yr_b in dl_f:
            opt_f.zero_grad(set_to_none=True)
            o = m_f(xb, mb)
            loss = (W_HTN_CV * bce_h(o["htn"], yh_b)
                    + W_STG_CV * ce_st(o["stage"], ys_b)
                    + W_TRT_CV * bce_tr(o["treat"], yt_b)
                    + W_RTE_CV * ce_rt(o["route"], yr_b))
            loss.backward()
            nn.utils.clip_grad_norm_(m_f.parameters(), 1.0)
            opt_f.step()
        m_f.eval()
        with torch.no_grad():
            ov = m_f(Xva_t, Mva_t)
            lv = (W_HTN_CV * bce_h(ov["htn"], yh_va)
                  + W_STG_CV * ce_st(ov["stage"], ys_va)
                  + W_TRT_CV * bce_tr(ov["treat"], yt_va)
                  + W_RTE_CV * ce_rt(ov["route"], yr_va)).item()
        if lv < best_v - 1e-5:
            best_v = lv; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in m_f.state_dict().items()}
        else:
            bad += 1
        if bad >= 18:
            break

    if best_state is not None:
        m_f.load_state_dict(best_state)
    m_f.eval()
    with torch.no_grad():
        p_h = torch.sigmoid(m_f(Xva_t, Mva_t)["htn"]).cpu().numpy()
    mtl_cv_oof_htn[va_f] = p_h
    y_va_np = yh_va.cpu().numpy().astype(int)
    fold_metrics = {
        "fold": fold, "n_val": int(len(va_f)),
        "ROC_AUC": float(roc_auc_score(y_va_np, p_h)),
        "PR_AUC":  float(average_precision_score(y_va_np, p_h)),
        "Brier":   float(brier_score_loss(y_va_np, p_h)),
        "Recall@90Prec": float(_recall_at_precision(y_va_np, p_h, 0.90)),
    }
    mtl_cv_rows.append(fold_metrics)
    print(f"  Fold {fold} -> "
          f"ROC-AUC {fold_metrics['ROC_AUC']:.4f}  "
          f"PR-AUC {fold_metrics['PR_AUC']:.4f}  "
          f"Brier {fold_metrics['Brier']:.4f}")

mtl_cv_df = pd.DataFrame(mtl_cv_rows)
mtl_cv_df.to_csv("reports/tables/mtl_5fold_metrics.csv", index=False)
print("\\nMissAware-MTL 5-fold CV summary (htn_now):")
print(mtl_cv_df.drop(columns=["fold", "n_val"]).agg(["mean", "std"]).round(4).T.to_string())

# OOF AUC
oof_mask = ~np.isnan(mtl_cv_oof_htn)
print(f"\\nMTL OOF ROC-AUC (across all {oof_mask.sum()} patients): "
      f"{roc_auc_score(y_htn_arr[oof_mask], mtl_cv_oof_htn[oof_mask]):.4f}")
'''
    )
)

# ============================================================================
# 18.C Loss-weight ablation
# ============================================================================
CELLS.append(
    md(
        """### 18.C Multi-Task Loss-Weight Ablation

Original Section 12 used `(W_HTN, W_STG, W_TRT, W_RTE) = (1.0, 0.6, 0.6, 0.35)` as
heuristics.  Here we sweep a small grid for each auxiliary head while holding
`W_HTN = 1.0` fixed and using a **single** stratified 80/20 split (kept identical
across runs) so differences are solely due to the loss-weight choice.

We report `htn_now` PR-AUC on the validation slice; the configuration that wins is
called out for re-use in production.
"""
    )
)

CELLS.append(
    code(
        '''# 18.C Loss-weight ablation -- single fixed split, vary one weight at a time
from sklearn.model_selection import train_test_split

idx_ab = np.arange(len(y_htn_arr))
tr_ab, va_ab = train_test_split(idx_ab, test_size=0.2, stratify=y_htn_arr,
                                 random_state=RANDOM_SEED)
med_ab, std_ab, x_norm_ab = compute_norm_params(X_true, tr_ab)

Xtr = torch.tensor(x_norm_ab[tr_ab], dtype=torch.float32, device=DEVICE)
Mtr = torch.tensor(MISS_MASK[tr_ab], dtype=torch.float32, device=DEVICE)
Xva = torch.tensor(x_norm_ab[va_ab], dtype=torch.float32, device=DEVICE)
Mva = torch.tensor(MISS_MASK[va_ab], dtype=torch.float32, device=DEVICE)

yh_tr_a = torch.tensor(y_htn_arr[tr_ab],   dtype=torch.float32, device=DEVICE)
yh_va_a = torch.tensor(y_htn_arr[va_ab],   dtype=torch.float32, device=DEVICE)
ys_tr_a = torch.tensor(y_stage_arr[tr_ab], dtype=torch.long,    device=DEVICE)
ys_va_a = torch.tensor(y_stage_arr[va_ab], dtype=torch.long,    device=DEVICE)
yt_tr_a = torch.tensor(y_treat_arr[tr_ab], dtype=torch.float32, device=DEVICE)
yt_va_a = torch.tensor(y_treat_arr[va_ab], dtype=torch.float32, device=DEVICE)
yr_tr_a = torch.tensor(y_route_arr[tr_ab], dtype=torch.long,    device=DEVICE)
yr_va_a = torch.tensor(y_route_arr[va_ab], dtype=torch.long,    device=DEVICE)

w_st_a = cw(y_stage_arr[tr_ab], 4)
w_rt_a = cw(y_route_arr[tr_ab], len(ROUTE_CLASSES))
pos_h_a = float((y_htn_arr[tr_ab] == 0).sum()) / max((y_htn_arr[tr_ab] == 1).sum(), 1)
pos_t_a = float((y_treat_arr[tr_ab] == 0).sum()) / max((y_treat_arr[tr_ab] == 1).sum(), 1)


def train_mtl_with_weights(W_STG, W_TRT, W_RTE, max_ep=80, patience=12, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    mdl = MissAwareMTL(n_features=n_feat, d_model=64, nhead=4, nlayers=2,
                       dim_ff=128, p_drop=0.2,
                       n_route_classes=len(ROUTE_CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=1e-4)
    ce_s = nn.CrossEntropyLoss(weight=w_st_a)
    ce_r = nn.CrossEntropyLoss(weight=w_rt_a)
    bce_h = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_h_a], device=DEVICE))
    bce_t = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_t_a], device=DEVICE))
    ds = TensorDataset(Xtr, Mtr, yh_tr_a, ys_tr_a, yt_tr_a, yr_tr_a)
    dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=False)

    best_v = float("inf"); best_st = None; bad = 0
    for _ in range(max_ep):
        mdl.train()
        for xb, mb, yh_b, ys_b, yt_b, yr_b in dl:
            opt.zero_grad(set_to_none=True)
            o = mdl(xb, mb)
            loss = (1.0 * bce_h(o["htn"], yh_b)
                    + W_STG * ce_s(o["stage"], ys_b)
                    + W_TRT * bce_t(o["treat"], yt_b)
                    + W_RTE * ce_r(o["route"], yr_b))
            loss.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
        mdl.eval()
        with torch.no_grad():
            ov = mdl(Xva, Mva)
            lv = (1.0 * bce_h(ov["htn"], yh_va_a)
                  + W_STG * ce_s(ov["stage"], ys_va_a)
                  + W_TRT * bce_t(ov["treat"], yt_va_a)
                  + W_RTE * ce_r(ov["route"], yr_va_a)).item()
        if lv < best_v - 1e-5:
            best_v = lv; bad = 0
            best_st = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_st is not None:
        mdl.load_state_dict(best_st)
    mdl.eval()
    with torch.no_grad():
        p = torch.sigmoid(mdl(Xva, Mva)["htn"]).cpu().numpy()
    y_va_np = yh_va_a.cpu().numpy().astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y_va_np, p)),
        "PR_AUC":  float(average_precision_score(y_va_np, p)),
        "Brier":   float(brier_score_loss(y_va_np, p)),
    }


ABLATION_GRID = [
    # (W_STG, W_TRT, W_RTE)
    (0.6, 0.6, 0.35),    # original Section 12
    (0.0, 0.0, 0.0),     # single-task htn_now baseline
    (0.6, 0.6, 0.0),     # drop the route head entirely
    (1.0, 1.0, 0.5),     # heavier multi-task
    (0.3, 0.3, 0.2),     # lighter multi-task
]
ablation_rows = []
for w_stg, w_trt, w_rte in ABLATION_GRID:
    print(f"  trying W_STG={w_stg} W_TRT={w_trt} W_RTE={w_rte} ...", end=" ", flush=True)
    metrics = train_mtl_with_weights(w_stg, w_trt, w_rte)
    ablation_rows.append({"W_STG": w_stg, "W_TRT": w_trt, "W_RTE": w_rte, **metrics})
    print(f"PR-AUC={metrics['PR_AUC']:.4f}  ROC-AUC={metrics['ROC_AUC']:.4f}")

ablation_df = pd.DataFrame(ablation_rows).sort_values("PR_AUC", ascending=False)
ablation_df.to_csv("reports/tables/mtl_loss_weight_ablation.csv", index=False)
print("\\nLoss-weight ablation (sorted by PR-AUC):")
ablation_df.round(4)
'''
    )
)

# ============================================================================
# 18.D Pooled-fold SHAP
# ============================================================================
CELLS.append(
    md(
        """### 18.D SHAP Pooled Across All 5 Outer Folds

The Section-8 SHAP plot used "the last fold's model" -- which is whichever fold the
loop happened to exit on, and is *not* guaranteed to be representative.  Here we
re-train the strict-pipeline winner on each of the 5 outer training folds and pool
the SHAP values computed on each fold's **held-out validation slice**.  Every patient
contributes one SHAP vector from a model that did not see that patient during
training, so the resulting global summary is an honest description of feature
attributions across the whole dataset.
"""
    )
)

CELLS.append(
    code(
        '''# 18.D Pooled-fold SHAP for the strict-pipeline winner on X_safe
import shap

# Use a tree-based estimator for fast TreeExplainer (LightGBM if present, else RF)
try:
    import lightgbm as _lgb_p
    BASE_EST = _lgb_p.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        scale_pos_weight=float((len(y_safe) - y_safe.sum()) / max(y_safe.sum(), 1)),
        objective="binary", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    EST_NAME = "LightGBM"
except Exception:
    BASE_EST = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
    )
    EST_NAME = "RandomForest"

print(f"Pooled-fold SHAP using {EST_NAME} on X_safe (no preprocessing leakage)")

skf_sh = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
shap_chunks: list[np.ndarray] = []
sample_chunks: list[pd.DataFrame] = []

for fold, (tr_sh, va_sh) in enumerate(skf_sh.split(X_safe, y_safe), start=1):
    print(f"  fold {fold}/5: training, then computing SHAP on validation slice...")
    X_tr = X_safe.iloc[tr_sh].fillna(X_safe.iloc[tr_sh].median(numeric_only=True)).fillna(0)
    y_tr = y_safe.iloc[tr_sh]
    X_va = X_safe.iloc[va_sh].fillna(X_safe.iloc[tr_sh].median(numeric_only=True)).fillna(0)
    est = type(BASE_EST)(**BASE_EST.get_params())
    est.fit(X_tr, y_tr)
    expl = shap.TreeExplainer(est)
    sv = expl.shap_values(X_va)
    if isinstance(sv, list):
        sv = sv[1]
    shap_chunks.append(np.asarray(sv))
    sample_chunks.append(X_va)

shap_all = np.concatenate(shap_chunks, axis=0)
sample_all = pd.concat(sample_chunks, axis=0).reset_index(drop=True)

mean_abs = pd.Series(np.abs(shap_all).mean(axis=0), index=sample_all.columns)\\
              .sort_values(ascending=False).round(4)
mean_abs.to_csv("reports/tables/shap_pooled_mean_abs.csv", header=["mean_abs_shap"])

print("\\nTop 10 mean-|SHAP| features (pooled across all 5 folds):")
print(mean_abs.head(10).to_string())

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_all, sample_all, plot_type="bar", show=False, max_display=15)
plt.title(f"Pooled-fold SHAP (Early Risk track, {EST_NAME})", fontweight="bold")
plt.tight_layout()
fig_path = Path("reports/figures/modelling/shap_pooled_folds.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")
'''
    )
)

# ============================================================================
# 18.E Closing summary
# ============================================================================
CELLS.append(
    md(
        """### 18.E Review Item -> Notebook Section Map

| Review item | Severity | Remediation in this notebook |
|---|---|---|
| 1. Target leakage in Sections 7-14 | Critical | Section 15 added the **leakage-safe Early Risk track** (`X_safe`); Sections 7-14 are now reported as "Diagnostic / triage reference" only |
| 2. WoE / MICE on full dataset | Critical | Section **18.A** -- WoE encoder fit inside each CV fold via `FoldWoE` transformer |
| 3. IterativeImputer fit before split | Critical | Section **17.A** + **18.A** -- imputers placed inside `Pipeline` and refitted per fold |
| 4. KMeans on full data feeding features | Critical | `phenotype_cluster` (and its WoE) explicitly listed in Section 15 leakage drop set |
| 5. MTL on single split vs CV baselines | Major | Section **17.D** (3 seeds) and **18.B** (full 5-fold outer CV, apples-to-apples) |
| 6. `route` head learns deterministic function | Major | Section **17.E** re-labels it as auxiliary regularisation; production routing is the deterministic rule of Section 14 |
| 7. KMeans `k=4` hardcoded | Major | Documented limitation; k-selection from silhouette is recommended for Section 3.G v2 |
| 8. SHAP from last fold only | Major | Section **18.D** -- SHAP pooled across all 5 outer folds |
| 9. EDA references features that don't exist yet | Major | Documented limitation; EDA was authored top-down and assumes Section 5 features. Production version should re-order |
| 10. Multi-task loss weights without ablation | Major | Section **18.C** -- 5-cell ablation grid |
| 11. Cardiometabolic composite uncalibrated | Moderate | Documented; ablation requires re-running Section 5 with/without -- recommended for v2 |
| 12. Little's MCAR test on 3 columns | Moderate | Documented limitation in Section 3.A |
| 13. Robustness scenarios overlap with training | Moderate | Documented in Section 13 -- requires external cohort for true generalisation |
| 14. No formal model significance testing | Moderate | Section **17.B** -- paired Wilcoxon signed-rank tests on fold-level AUC |
| 15. No subgroup / fairness analysis | Moderate | Section **17.C** -- per-`female`, per-`age_category`, per-`hc4` AUC + PR-AUC |
| 16. t-SNE without interpretability caveats | Minor | Documented; t-SNE distances are local-only; recommend UMAP with `min_dist` annotation for v2 |

Items not closed in code (would require re-architecting the existing Sections 3-9):

* Reordering EDA cells so feature-engineering precedes correlation / VIF (item 9)
* Replacing `IterativeImputer(sample_posterior=True)` with a known-stable MICE backend (Minor item)
* Calibration plot extended to all 5 folds (Minor item)

The headline scientific claim now lives in **Sections 15, 17, and 18**.  Sections 7-14
remain in the notebook as supporting / Diagnostic-track context but should not be cited
as the model's predictive performance.
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
