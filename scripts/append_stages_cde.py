#!/usr/bin/env python3
"""Append Sections 12-14 to UGICIS.ipynb: MissAware-MTL (C), robustness (D), routing (E)."""
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

CELLS.append(
    md(
        """## 12. MissAware-MTL (Stage C)

Missingness-aware multi-task model: each feature contributes **normalized value +
missing mask** tokens, encoded with a small **Transformer**, then four heads predict
`htn_now`, `htn_stage` (4-class), `treat`, and `route` (4-class).  Train/val split 85/15
(stratified on `htn_now`).  **Monte Carlo Dropout** (T=30) estimates uncertainty on the
primary hypertension probability.

Artifacts: `reports/tables/mtl_state.pt`, `reports/tables/mtl_preprocess.pkl`,
`reports/tables/missaware_mtl_val_metrics.json`.
"""
    )
)

CELLS.append(
    code(
        """import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | device={DEVICE}")
"""
    )
)

CELLS.append(
    code(
        """# Feature matrix with NaNs preserved (mask = missingness signal)
LEAKAGE_MTL = ["htn_stage", "treat", "new_dx", "prior_unknown", "category", "route",
                "htn_stage_ord", "bp_systolic_bin", "pp_tertile"]

mod_raw = df_clean.drop(columns=[c for c in LEAKAGE_MTL if c in df_clean.columns],
                        errors="ignore").copy()
for c in mod_raw.columns:
    if str(mod_raw[c].dtype) == "category":
        mod_raw[c] = mod_raw[c].cat.codes.replace(-1, np.nan)
mod_raw = mod_raw.apply(pd.to_numeric, errors="coerce")

y_htn_arr = mod_raw["htn_now"].astype(int).values
feat_df = mod_raw.drop(columns=["htn_now"])
FEATURE_COLS = list(feat_df.columns)
X_true = feat_df.to_numpy(dtype=np.float64)
MISS_MASK = np.isnan(X_true).astype(np.float32)

y_stage_arr = pd.to_numeric(df_clean.loc[mod_raw.index, "htn_stage"], errors="coerce").fillna(0).astype(int).values
y_treat_arr = pd.to_numeric(df_clean.loc[mod_raw.index, "treat"], errors="coerce").fillna(0).astype(int).values

ROUTE_CLASSES = ["self_manage", "treat_at_facility", "refer_district", "refer_tertiary"]
_route_series = df_clean.loc[mod_raw.index, "route"]
y_route_arr = np.array(
    [ROUTE_CLASSES.index(r) if r in ROUTE_CLASSES else 0 for r in _route_series.fillna("self_manage")],
    dtype=np.int64,
)

n_feat = X_true.shape[1]
print(f"MissAware inputs: n={len(y_htn_arr):,}  features={n_feat}")
print(f"  htn_now prevalence {y_htn_arr.mean():.3f} | treat prevalence {y_treat_arr.mean():.3f}")
"""
    )
)

CELLS.append(
    code(
        """class MissAwareMTL(nn.Module):
    'Feature-sequence transformer with value + missingness embeddings and four heads.'

    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 nlayers: int = 2, dim_ff: int = 128, p_drop: float = 0.2,
                 n_route_classes: int = 4):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.val_proj = nn.Linear(1, d_model, bias=True)
        self.miss_emb = nn.Embedding(2, d_model)
        self.feat_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=p_drop, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(p_drop)
        self.head_htn = nn.Linear(d_model, 1)
        self.head_stage = nn.Linear(d_model, 4)
        self.head_treat = nn.Linear(d_model, 1)
        self.head_route = nn.Linear(d_model, n_route_classes)

    def forward(self, x_norm: torch.Tensor, miss: torch.Tensor) -> dict:
        B, _F = x_norm.shape
        v = self.val_proj(x_norm.unsqueeze(-1))
        m = self.miss_emb(miss.long().clamp(0, 1))
        tok = v + m + self.feat_bias.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = self.encoder(h)
        z = self.dropout(h[:, 0, :])
        return {
            "htn": self.head_htn(z).squeeze(-1),
            "stage": self.head_stage(z),
            "treat": self.head_treat(z).squeeze(-1),
            "route": self.head_route(z),
        }


def compute_norm_params(x_true: np.ndarray, train_idx: np.ndarray):
    med = np.zeros(n_feat, dtype=np.float64)
    std = np.ones(n_feat, dtype=np.float64)
    X_tr = x_true[train_idx]
    for j in range(n_feat):
        col = X_tr[:, j]
        obs = ~np.isnan(col)
        if obs.sum() == 0:
            continue
        med[j] = np.median(col[obs])
        s = float(np.std(col[obs]))
        std[j] = s if s > 1e-8 else 1.0
    x_norm = np.zeros_like(x_true, dtype=np.float32)
    obs_g = ~np.isnan(x_true)
    if obs_g.any():
        r, c = np.where(obs_g)
        x_norm[obs_g] = ((x_true[obs_g] - med[c]) / std[c]).astype(np.float32)
    return med, std, x_norm


def apply_norm_params(x_true: np.ndarray, med: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_norm = np.zeros_like(x_true, dtype=np.float32)
    obs = ~np.isnan(x_true)
    if obs.any():
        r, c = np.where(obs)
        x_norm[obs] = ((x_true[obs] - med[c]) / std[c]).astype(np.float32)
    return x_norm


from sklearn.model_selection import train_test_split

_idx_all = np.arange(len(y_htn_arr))
tr_idx, va_idx = train_test_split(
    _idx_all, test_size=0.15, stratify=y_htn_arr, random_state=RANDOM_SEED,
)

med_np, std_np, x_norm_full = compute_norm_params(X_true, tr_idx)

X_tr = torch.tensor(x_norm_full[tr_idx], dtype=torch.float32, device=DEVICE)
M_tr = torch.tensor(MISS_MASK[tr_idx], dtype=torch.float32, device=DEVICE)
X_va = torch.tensor(x_norm_full[va_idx], dtype=torch.float32, device=DEVICE)
M_va = torch.tensor(MISS_MASK[va_idx], dtype=torch.float32, device=DEVICE)

y_htn_tr = torch.tensor(y_htn_arr[tr_idx], dtype=torch.float32, device=DEVICE)
y_htn_va = torch.tensor(y_htn_arr[va_idx], dtype=torch.float32, device=DEVICE)
y_stage_tr = torch.tensor(y_stage_arr[tr_idx], dtype=torch.long, device=DEVICE)
y_stage_va = torch.tensor(y_stage_arr[va_idx], dtype=torch.long, device=DEVICE)
y_treat_tr = torch.tensor(y_treat_arr[tr_idx], dtype=torch.float32, device=DEVICE)
y_treat_va = torch.tensor(y_treat_arr[va_idx], dtype=torch.float32, device=DEVICE)
y_route_tr = torch.tensor(y_route_arr[tr_idx], dtype=torch.long, device=DEVICE)
y_route_va = torch.tensor(y_route_arr[va_idx], dtype=torch.long, device=DEVICE)


def cw(y: np.ndarray, n_cls: int) -> torch.Tensor:
    cnt = np.bincount(y, minlength=n_cls).astype(np.float64)
    w = len(y) / (n_cls * np.maximum(cnt, 1.0))
    w = w / w.mean()
    w = np.minimum(w, 5.0)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


w_stage = cw(y_stage_arr[tr_idx], 4)
w_route = cw(y_route_arr[tr_idx], len(ROUTE_CLASSES))
pos_htn = float((y_htn_arr[tr_idx] == 0).sum()) / max((y_htn_arr[tr_idx] == 1).sum(), 1)
pos_trt = float((y_treat_arr[tr_idx] == 0).sum()) / max((y_treat_arr[tr_idx] == 1).sum(), 1)

model_mtl = MissAwareMTL(
    n_features=n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, p_drop=0.2,
    n_route_classes=len(ROUTE_CLASSES),
).to(DEVICE)
opt = torch.optim.AdamW(model_mtl.parameters(), lr=3e-4, weight_decay=1e-4)

ds_tr = TensorDataset(X_tr, M_tr, y_htn_tr, y_stage_tr, y_treat_tr, y_route_tr)
dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True, drop_last=False)

ce_stage = nn.CrossEntropyLoss(weight=w_stage)
ce_route = nn.CrossEntropyLoss(weight=w_route)
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_htn], device=DEVICE))
bce_t = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_trt], device=DEVICE))

W_HTN, W_STG, W_TRT, W_RTE = 1.0, 0.6, 0.6, 0.35

best_val = float("inf")
best_state = None
patience, bad = 18, 0
max_ep = 120

for ep in range(1, max_ep + 1):
    model_mtl.train()
    tot = 0.0
    for xb, mb, yh, ys, yt, yr in dl_tr:
        opt.zero_grad(set_to_none=True)
        out = model_mtl(xb, mb)
        loss = (W_HTN * bce(out["htn"], yh)
                + W_STG * ce_stage(out["stage"], ys)
                + W_TRT * bce_t(out["treat"], yt)
                + W_RTE * ce_route(out["route"], yr))
        loss.backward()
        nn.utils.clip_grad_norm_(model_mtl.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(xb)
    tot /= len(ds_tr)

    model_mtl.eval()
    with torch.no_grad():
        out_va = model_mtl(X_va, M_va)
        lv = (W_HTN * bce(out_va["htn"], y_htn_va)
              + W_STG * ce_stage(out_va["stage"], y_stage_va)
              + W_TRT * bce_t(out_va["treat"], y_treat_va)
              + W_RTE * ce_route(out_va["route"], y_route_va)).item()
    if lv < best_val - 1e-5:
        best_val, bad = lv, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model_mtl.state_dict().items()}
    else:
        bad += 1
    if ep % 10 == 0 or ep == 1:
        print(f"epoch {ep:3d}  train_loss={tot:.4f}  val_loss={lv:.4f}  patience {bad}/{patience}")
    if bad >= patience:
        print(f"Early stop at epoch {ep}")
        break

if best_state is not None:
    model_mtl.load_state_dict(best_state)

model_mtl.eval()
with torch.no_grad():
    out_va = model_mtl(X_va, M_va)
    p_htn = torch.sigmoid(out_va["htn"]).cpu().numpy()
y_htn_va_np = y_htn_va.cpu().numpy().astype(int)
mtl_val_auc = roc_auc_score(y_htn_va_np, p_htn)
mtl_val_ap = average_precision_score(y_htn_va_np, p_htn)
mtl_val_brier = brier_score_loss(y_htn_va_np, p_htn)
print(f"\\nMissAware-MTL val  htn_now  ROC-AUC={mtl_val_auc:.4f}  PR-AUC={mtl_val_ap:.4f}  Brier={mtl_val_brier:.4f}")

_mtl_dir = Path("reports/tables")
_mtl_dir.mkdir(parents=True, exist_ok=True)
torch.save(model_mtl.state_dict(), _mtl_dir / "mtl_state.pt")
with open(_mtl_dir / "mtl_preprocess.pkl", "wb") as f:
    pickle.dump(
        {"median": med_np, "std": std_np, "feature_cols": FEATURE_COLS,
         "route_classes": ROUTE_CLASSES, "leakage_cols": LEAKAGE_MTL},
        f,
    )
print("Saved reports/tables/mtl_state.pt and mtl_preprocess.pkl")
"""
    )
)

CELLS.append(
    code(
        """# MC Dropout uncertainty (T=30) on validation split
T_MC = 30
model_mtl.train()

def mc_htn_probs(x_tensor, m_tensor):
    probs = []
    for _ in range(T_MC):
        with torch.no_grad():
            o = model_mtl(x_tensor, m_tensor)
            probs.append(torch.sigmoid(o["htn"]).cpu().numpy())
    probs = np.stack(probs, axis=0)
    return probs.mean(axis=0), probs.std(axis=0)


p_mean, p_std = mc_htn_probs(X_va, M_va)
unc_mean = float(np.mean(p_std))
unc_p95 = float(np.percentile(p_std, 95))

mtl_metrics = {
    "val_roc_auc_htn_now": float(mtl_val_auc),
    "val_pr_auc_htn_now": float(mtl_val_ap),
    "val_brier_htn_now": float(mtl_val_brier),
    "mc_dropout_T": T_MC,
    "uncertainty_htn_prob_std_mean": unc_mean,
    "uncertainty_htn_prob_std_p95": unc_p95,
}

Path("reports/tables").mkdir(parents=True, exist_ok=True)
with open("reports/tables/missaware_mtl_val_metrics.json", "w") as f:
    json.dump(mtl_metrics, f, indent=2)
print(json.dumps(mtl_metrics, indent=2))
"""
    )
)

# ----- Section 13: Robustness -----
CELLS.append(
    md(
        """## 13. Robustness Under Simulated Missingness (Stage D)

Each parquet in `data/processed/sim/` injects MCAR or MNAR missingness at 10-40%.
We evaluate **(1)** the tuned LightGBM baseline (median-imputed `X` with training medians
from the live notebook `X`) and **(2)** the MissAware-MTL encoder (value + mask, same
`median`/`std` as Stage C) on **`htn_now`**.  This answers whether the missingness-aware
representation degrades more gracefully than a classical impute-then-boost pipeline when
the missingness pattern shifts.
"""
    )
)

CELLS.append(
    code(
        """import joblib
from glob import glob

X_med = X.median(numeric_only=True)
lgbm_path = Path("reports/tables/final_lgbm_htn_now.joblib")
if lgbm_path.exists():
    lgbm_final = joblib.load(lgbm_path)
else:
    lgbm_final = None
    print("Warning: final_lgbm_htn_now.joblib not found — LGBM robustness rows will be NaN.")

feat_cols_mtl = FEATURE_COLS
rows = []

for p in sorted(glob("data/processed/sim/*.parquet")):
    scen = Path(p).stem
    dsim = pd.read_parquet(p)
    cols = list(X.columns)
    for c in cols:
        if c not in dsim.columns:
            dsim[c] = np.nan
    Xs = dsim[cols].apply(pd.to_numeric, errors="coerce")
    y_true = pd.to_numeric(dsim["htn_now"], errors="coerce").fillna(0).astype(int).values

    if lgbm_final is not None:
        X_lgb = Xs.fillna(X_med).astype(np.float32)
        pl = lgbm_final.predict_proba(X_lgb)[:, 1]
        rows.append({
            "scenario": scen,
            "model": "LightGBM_tuned",
            "n": len(y_true),
            "roc_auc_htn_now": float(roc_auc_score(y_true, pl)),
            "pr_auc_htn_now": float(average_precision_score(y_true, pl)),
            "brier_htn_now": float(brier_score_loss(y_true, pl)),
        })

    X_arr = np.zeros((len(dsim), len(feat_cols_mtl)), dtype=np.float64)
    for j, c in enumerate(feat_cols_mtl):
        if c in dsim.columns:
            X_arr[:, j] = pd.to_numeric(dsim[c], errors="coerce").values
        else:
            X_arr[:, j] = np.nan
    miss = np.isnan(X_arr).astype(np.float32)
    x_norm = apply_norm_params(X_arr, med_np, std_np)
    xt = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)
    mt = torch.tensor(miss, dtype=torch.float32, device=DEVICE)

    model_mtl.eval()
    with torch.no_grad():
        pm = torch.sigmoid(model_mtl(xt, mt)["htn"]).cpu().numpy()

    rows.append({
        "scenario": scen,
        "model": "MissAware_MTL",
        "n": len(y_true),
        "roc_auc_htn_now": float(roc_auc_score(y_true, pm)),
        "pr_auc_htn_now": float(average_precision_score(y_true, pm)),
        "brier_htn_now": float(brier_score_loss(y_true, pm)),
    })

robust_df = pd.DataFrame(rows)
robust_df.to_csv("reports/tables/robustness_missingness_eval.csv", index=False)
print(robust_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
for mname, g in robust_df.groupby("model"):
    g = g.copy()
    g["rate"] = g["scenario"].str.split("_").str[-1].astype(int)
    g["kind"] = g["scenario"].str.startswith("mcar").map({True: "MCAR", False: "MNAR"})
    for knd, gg in g.groupby("kind"):
        gg = gg.sort_values("rate")
        ax.plot(gg["rate"], gg["roc_auc_htn_now"], marker="o", label=f"{mname} ({knd})")
ax.set_xlabel("Injected missingness rate (%)")
ax.set_ylabel("ROC-AUC  htn_now")
ax.set_title("Robustness: tuned LightGBM vs MissAware-MTL")
ax.legend()
ax.grid(alpha=0.3)
fig_path = Path("reports/figures/modelling/robustness_missingness.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")
"""
    )
)

# ----- Section 14: Routing -----
CELLS.append(
    md(
        """## 14. Clinical Routing + JSON Export (Stage E)

Deterministic **referral route** follows the same rule as feature engineering:
`self_manage` if stage 0; `refer_tertiary` if stage 3; stage 1 -> `treat_at_facility`;
stage 2 -> `treat_at_facility` if `hc4==1` else `refer_district`.

We combine **predicted stage** (argmax of the MTL stage head on the validation split) with
facility `hc4`, **MC-dropout std** on `htn_now` probability, and a **confidence rule**:
if `std > 0.08` the case is flagged `review_clinician=True`.

Example patient JSON lines are written to `reports/tables/routing_examples.jsonl`.
"""
    )
)

CELLS.append(
    code(
        """def derive_route_from_stage(stage: int, hc4_v: float) -> str:
    if pd.isna(stage):
        return "self_manage"
    s = int(stage)
    if s == 0:
        return "self_manage"
    if s == 3:
        return "refer_tertiary"
    if s == 1:
        return "treat_at_facility"
    return "treat_at_facility" if int(hc4_v) == 1 else "refer_district"


def clinical_patient_record(
    patient_id,
    p_htn: float,
    p_htn_std: float,
    p_treat: float,
    stage_logits: np.ndarray,
    hc4_val: float,
    top_features: list | None = None,
    unc_threshold: float = 0.08,
) -> dict:
    stage_pred = int(np.argmax(stage_logits))
    route = derive_route_from_stage(stage_pred, hc4_val)
    review = bool(p_htn_std > unc_threshold)
    rec = {
        "patient_id": patient_id,
        "predicted_htn_now_prob": round(float(p_htn), 4),
        "predicted_htn_now_uncertainty_std": round(float(p_htn_std), 4),
        "predicted_treat_prob": round(float(p_treat), 4),
        "predicted_htn_stage": stage_pred,
        "predicted_route": route,
        "review_clinician": review,
        "rationale": (
            "High epistemic uncertainty on hypertension probability."
            if review else "Uncertainty within operational threshold."
        ),
    }
    if top_features:
        rec["top_shap_features"] = top_features
    return rec


model_mtl.eval()
with torch.no_grad():
    _out_va = model_mtl(X_va, M_va)
    st_logits_va = _out_va["stage"].cpu().numpy()
    tr_logits_va = _out_va["treat"].cpu().numpy()

p_treat_va = 1.0 / (1.0 + np.exp(-tr_logits_va))

# hc4 on validation rows (same order as va_idx)
_va_index = mod_raw.iloc[va_idx].index
hc4_va = pd.to_numeric(df_clean.loc[_va_index, "hc4"], errors="coerce").fillna(0).values

out_path = Path("reports/tables/routing_examples.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as fout:
    for i in range(min(25, len(va_idx))):
        ii = int(va_idx[i])
        pid = mod_raw.iloc[ii].name
        try:
            pid = int(pid)
        except Exception:
            pid = str(pid)
        rec = clinical_patient_record(
            patient_id=pid,
            p_htn=float(p_mean[i]),
            p_htn_std=float(p_std[i]),
            p_treat=float(p_treat_va[i]),
            stage_logits=st_logits_va[i],
            hc4_val=float(hc4_va[i]),
            top_features=None,
        )
        fout.write(json.dumps(rec) + "\\n")

print(f"Wrote {out_path} (first 25 validation patients)")
print("Example record:")
print(json.dumps(json.loads(out_path.read_text().splitlines()[0]), indent=2))
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
