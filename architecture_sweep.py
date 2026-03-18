#!/usr/bin/env python3
"""
architecture_sweep.py — Comprehensive stacking architecture experiment
=======================================================================
Tests every combination of meta-learner × feature allocation to find
the optimal way to integrate lineup features into the ensemble.

THE PROBLEM:
  Lineup features improve every individual learner by 0.1-0.2 pts
  but DEGRADE the stack by 0.05 pts because all learners converge
  on the same prediction → Ridge can't diversify.

SOLUTIONS TESTED:

  Meta-learners:
    A) Ridge (current)
    B) ElasticNet (L1 can zero out redundant learner contributions)
    C) LightGBM meta (non-linear conditional weighting)
    D) Simple average (no meta-learner at all)
    E) Weighted average (optimized fixed weights)

  Feature allocations:
    1) All 146 to all (current production baseline — no lineup)
    2) All 150 to all (lineup to everyone — known to fail with Ridge)
    3) Lineup to CatBoost only (feature subspacing)
    4) Market to CatBoost only, lineup to XGB only (specialist split)
    5) SHAP-guided: each learner gets unique top features

  Bonus:
    6) Residual stacking: base 146 model + CatBoost error corrector with lineup

Run:
    python3 -u architecture_sweep.py
    nohup python3 -u architecture_sweep.py > arch_sweep_results.txt 2>&1 &

Estimated time: ~30 min (50-fold CV, ~40K games, multiple configs)
"""

import sys, os, json, copy, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import load_cached

N_SPLITS = 50
DEPTH = 4  # confirmed optimal

# ═══════════════════════════════════════════════════════════════
# DATA LOADING (same as production retrain)
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  ARCHITECTURE SWEEP — Meta-Learners × Feature Subspacing")
print("=" * 70)

df = load_cached()
df = df[df["actual_home_score"].notna()].copy()

# ESPN odds fallback
if "espn_spread" in df.columns:
    espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df.loc[fill, "market_spread_home"] = espn_s[fill]

# Quality filter (production)
_quality_cols = ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.8
if "referee_1" in df.columns:
    _keep = _keep & df["referee_1"].notna() & (df["referee_1"] != "")
df = df.loc[_keep].reset_index(drop=True)

df = _ncaa_backfill_heuristic(df)

with open("referee_profiles.json") as f:
    ncaa_build_features._ref_profiles = json.load(f)

df = df.dropna(subset=["actual_home_score", "actual_away_score"])
X_all = ncaa_build_features(df)
y = df["actual_home_score"].values - df["actual_away_score"].values
print(f"  {len(X_all)} games × {X_all.shape[1]} features")

# ═══════════════════════════════════════════════════════════════
# DEFINE FEATURE SETS
# ═══════════════════════════════════════════════════════════════

all_cols = list(X_all.columns)
lineup_cols = [c for c in all_cols if 'lineup' in c or 'starter_experience' in c]
# Remove new_starter_either (0% SHAP)
lineup_cols = [c for c in lineup_cols if 'new_starter' not in c]
base_cols = [c for c in all_cols if c not in lineup_cols and 'new_starter' not in c]

# Market-related features
market_cols = [c for c in base_cols if 'mkt_' in c or 'spread_movement' in c or 
               'total_movement' in c or 'espn_' in c or 'has_mkt' in c or 'has_spread' in c]
no_market_cols = [c for c in base_cols if c not in market_cols]

print(f"  Base features: {len(base_cols)}")
print(f"  Lineup features: {lineup_cols}")
print(f"  Market features: {market_cols}")
print(f"  Non-market features: {len(no_market_cols)}")

# ═══════════════════════════════════════════════════════════════
# OOF HELPER
# ═══════════════════════════════════════════════════════════════

def time_series_oof(model, X, y, n_splits=N_SPLITS):
    n = len(X)
    oof = np.zeros(n)
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(train_end + fold_size, n)
        if val_start >= n: break
        m = copy.deepcopy(model)
        if isinstance(m, CatBoostRegressor):
            m.fit(X[:train_end], y[:train_end], verbose=0)
        else:
            m.fit(X[:train_end], y[:train_end])
        oof[val_start:val_end] = m.predict(X[val_start:val_end])
    return oof

def make_xgb():
    return XGBRegressor(n_estimators=175, max_depth=DEPTH, learning_rate=0.10,
                        random_state=42, tree_method="hist")

def make_cat():
    return CatBoostRegressor(n_estimators=175, depth=DEPTH, learning_rate=0.10,
                             random_seed=42, verbose=0)

def make_mlp():
    return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                        validation_fraction=0.1, random_state=42)

# ═══════════════════════════════════════════════════════════════
# PRE-COMPUTE OOF FOR EACH (model, feature_set) PAIR
# ═══════════════════════════════════════════════════════════════

print("\n  Pre-computing OOF predictions for all (model, feature_set) combos...")
print("  This avoids redundant training across configs.\n")

# Scale each feature set
scalers = {}
scaled = {}

feature_sets = {
    "base": base_cols,                              # 146 features (production)
    "full": base_cols + lineup_cols,                 # 149 features (base + 3 lineup)
    "no_market": no_market_cols,                     # base minus market
    "no_market_plus_lineup": no_market_cols + lineup_cols,  # no market + lineup
    "market_plus_lineup": market_cols + lineup_cols, # market + lineup only (specialist)
}

for name, cols in feature_sets.items():
    sc = StandardScaler()
    scaled[name] = sc.fit_transform(X_all[cols])
    scalers[name] = sc
    print(f"  Scaled '{name}': {len(cols)} features")

# OOF cache: key = (model_name, feature_set_name)
oof_cache = {}

def get_oof(model_name, feat_name, model_factory):
    key = (model_name, feat_name)
    if key not in oof_cache:
        X_data = scaled[feat_name]
        print(f"  Training {model_name} on '{feat_name}' ({X_data.shape[1]} feat)...", end=" ", flush=True)
        t0 = time.time()
        oof = time_series_oof(model_factory(), X_data, y)
        mae = mean_absolute_error(y, oof)
        elapsed = time.time() - t0
        print(f"MAE: {mae:.4f} ({elapsed:.0f}s)")
        oof_cache[key] = oof
    return oof_cache[key]

# Pre-compute all needed OOF combinations
print("\n─── Training individual models ───")
# Base features (production)
get_oof("xgb", "base", make_xgb)
get_oof("cat", "base", make_cat)
get_oof("mlp", "base", make_mlp)

# Full features (base + lineup)
get_oof("xgb", "full", make_xgb)
get_oof("cat", "full", make_cat)
get_oof("mlp", "full", make_mlp)

# Specialist: no-market features
get_oof("xgb", "no_market", make_xgb)
get_oof("mlp", "no_market", make_mlp)

# Specialist: market + lineup (for CatBoost specialist)
get_oof("cat", "market_plus_lineup", make_cat)

# Specialist: no-market + lineup (for XGB specialist)
get_oof("xgb", "no_market_plus_lineup", make_xgb)

# ═══════════════════════════════════════════════════════════════
# META-LEARNER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def stack_ridge(oof_list, y):
    S = np.column_stack(oof_list)
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta.fit(S, y)
    preds = meta.predict(S)
    return mean_absolute_error(y, preds), list(meta.coef_.round(4)), "Ridge"

def stack_elasticnet(oof_list, y):
    S = np.column_stack(oof_list)
    meta = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], alphas=[0.001, 0.01, 0.1, 1.0],
                        max_iter=5000, cv=5)
    meta.fit(S, y)
    preds = meta.predict(S)
    return mean_absolute_error(y, preds), list(meta.coef_.round(4)), f"ElasticNet(l1={meta.l1_ratio_:.2f},α={meta.alpha_:.4f})"

def stack_lgbm(oof_list, y):
    from lightgbm import LGBMRegressor
    S = np.column_stack(oof_list)
    # Use time-series OOF for the meta-learner too (prevent leakage)
    n = len(y)
    meta_oof = np.zeros(n)
    fold_size = n // (N_SPLITS + 1)
    for i in range(N_SPLITS):
        te = fold_size * (i + 2)
        vs, ve = te, min(te + fold_size, n)
        if vs >= n: break
        m = LGBMRegressor(n_estimators=50, max_depth=2, learning_rate=0.1,
                          random_state=42, verbose=-1)
        m.fit(S[:te], y[:te])
        meta_oof[vs:ve] = m.predict(S[vs:ve])
    return mean_absolute_error(y, meta_oof), [0, 0, 0], "LightGBM(d=2)"

def stack_average(oof_list, y):
    preds = np.mean(oof_list, axis=0)
    return mean_absolute_error(y, preds), [round(1/len(oof_list), 3)]*len(oof_list), "Average"

def stack_weighted_opt(oof_list, y):
    """Brute-force optimized fixed weights (grid search)."""
    S = np.column_stack(oof_list)
    best_mae, best_w = 999, None
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0: continue
            preds = S[:, 0]*w1 + S[:, 1]*w2 + S[:, 2]*w3
            mae = mean_absolute_error(y, preds)
            if mae < best_mae:
                best_mae = mae
                best_w = [round(w1, 2), round(w2, 2), round(w3, 2)]
    return best_mae, best_w, f"OptWeights({best_w})"

meta_learners = {
    "Ridge": stack_ridge,
    "ElasticNet": stack_elasticnet,
    "LightGBM": stack_lgbm,
    "Average": stack_average,
    "OptWeights": stack_weighted_opt,
}

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGS
# ═══════════════════════════════════════════════════════════════

configs = [
    # ── SECTION 1: Current production baseline ──
    {
        "label": "1A: Production baseline (146 feat, Ridge)",
        "oof": [("xgb", "base"), ("cat", "base"), ("mlp", "base")],
        "metas": ["Ridge"],
    },

    # ── SECTION 2: All features to all learners (what we tested before) ──
    {
        "label": "2A: All 149 to all, Ridge",
        "oof": [("xgb", "full"), ("cat", "full"), ("mlp", "full")],
        "metas": ["Ridge"],
    },
    {
        "label": "2B: All 149 to all, ElasticNet",
        "oof": [("xgb", "full"), ("cat", "full"), ("mlp", "full")],
        "metas": ["ElasticNet"],
    },
    {
        "label": "2C: All 149 to all, LightGBM meta",
        "oof": [("xgb", "full"), ("cat", "full"), ("mlp", "full")],
        "metas": ["LightGBM"],
    },
    {
        "label": "2D: All 149 to all, Average",
        "oof": [("xgb", "full"), ("cat", "full"), ("mlp", "full")],
        "metas": ["Average"],
    },
    {
        "label": "2E: All 149 to all, OptWeights",
        "oof": [("xgb", "full"), ("cat", "full"), ("mlp", "full")],
        "metas": ["OptWeights"],
    },

    # ── SECTION 3: Lineup to CatBoost only ──
    {
        "label": "3A: Lineup→Cat only, Ridge",
        "oof": [("xgb", "base"), ("cat", "full"), ("mlp", "base")],
        "metas": ["Ridge"],
    },
    {
        "label": "3B: Lineup→Cat only, ElasticNet",
        "oof": [("xgb", "base"), ("cat", "full"), ("mlp", "base")],
        "metas": ["ElasticNet"],
    },
    {
        "label": "3C: Lineup→Cat only, LightGBM meta",
        "oof": [("xgb", "base"), ("cat", "full"), ("mlp", "base")],
        "metas": ["LightGBM"],
    },

    # ── SECTION 4: Specialist split (market isolation) ──
    # XGB: no market, no lineup (pure efficiency/momentum)
    # Cat: market + lineup (specialist)
    # MLP: base (all features)
    {
        "label": "4A: Specialist split, Ridge",
        "oof": [("xgb", "no_market"), ("cat", "market_plus_lineup"), ("mlp", "base")],
        "metas": ["Ridge"],
    },
    {
        "label": "4B: Specialist split, ElasticNet",
        "oof": [("xgb", "no_market"), ("cat", "market_plus_lineup"), ("mlp", "base")],
        "metas": ["ElasticNet"],
    },
    {
        "label": "4C: Specialist split, LightGBM meta",
        "oof": [("xgb", "no_market"), ("cat", "market_plus_lineup"), ("mlp", "base")],
        "metas": ["LightGBM"],
    },

    # ── SECTION 5: Lineup to XGB, market to Cat (reversed) ──
    {
        "label": "5A: Lineup→XGB, Market→Cat, Ridge",
        "oof": [("xgb", "no_market_plus_lineup"), ("cat", "base"), ("mlp", "base")],
        "metas": ["Ridge"],
    },
    {
        "label": "5B: Lineup→XGB, Market→Cat, ElasticNet",
        "oof": [("xgb", "no_market_plus_lineup"), ("cat", "base"), ("mlp", "base")],
        "metas": ["ElasticNet"],
    },

    # ── SECTION 6: Production baseline with alt meta-learners ──
    {
        "label": "6A: Production 146, ElasticNet",
        "oof": [("xgb", "base"), ("cat", "base"), ("mlp", "base")],
        "metas": ["ElasticNet"],
    },
    {
        "label": "6B: Production 146, LightGBM meta",
        "oof": [("xgb", "base"), ("cat", "base"), ("mlp", "base")],
        "metas": ["LightGBM"],
    },
    {
        "label": "6C: Production 146, Average",
        "oof": [("xgb", "base"), ("cat", "base"), ("mlp", "base")],
        "metas": ["Average"],
    },
    {
        "label": "6D: Production 146, OptWeights",
        "oof": [("xgb", "base"), ("cat", "base"), ("mlp", "base")],
        "metas": ["OptWeights"],
    },
]

# ═══════════════════════════════════════════════════════════════
# SECTION 7: RESIDUAL STACKING (two-stage)
# ═══════════════════════════════════════════════════════════════
# Stage 1: Production stack (146 features) → get prediction
# Stage 2: CatBoost predicts the ERROR using lineup features
# Final = Stage 1 + Stage 2 correction

# ═══════════════════════════════════════════════════════════════
# RUN ALL CONFIGS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  RUNNING STACKING EXPERIMENTS")
print("=" * 70)

results = []

for cfg in configs:
    label = cfg["label"]
    oof_specs = cfg["oof"]
    
    # Gather OOF predictions
    oof_list = [oof_cache[spec] for spec in oof_specs]
    
    for meta_name in cfg["metas"]:
        meta_fn = meta_learners[meta_name]
        mae, weights, meta_desc = meta_fn(oof_list, y)
        
        results.append({
            "label": label,
            "meta": meta_desc,
            "mae": mae,
            "weights": weights,
        })
        print(f"  {label:50s} → MAE: {mae:.4f}  w={weights}")

# ── SECTION 7: Residual stacking ──
print(f"\n{'─'*70}")
print(f"  RESIDUAL STACKING (two-stage)")
print(f"{'─'*70}")

# Stage 1: production stack OOF
oof_base = [oof_cache[("xgb", "base")], oof_cache[("cat", "base")], oof_cache[("mlp", "base")]]
S_base = np.column_stack(oof_base)
meta_base = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
meta_base.fit(S_base, y)
stage1_pred = meta_base.predict(S_base)
stage1_mae = mean_absolute_error(y, stage1_pred)
print(f"  Stage 1 (production Ridge): MAE {stage1_mae:.4f}")

# Stage 2: predict the residual using lineup features + stage1 prediction
residuals = y - stage1_pred
X_residual_cols = lineup_cols + ["mkt_spread"]  # lineup + context
X_residual_cols = [c for c in X_residual_cols if c in X_all.columns]
print(f"  Stage 2 features: {X_residual_cols}")

sc_res = StandardScaler()
X_res = sc_res.fit_transform(X_all[X_residual_cols])

# Add stage1 prediction as a feature for the corrector
X_res_with_pred = np.column_stack([X_res, stage1_pred])

print(f"  Training residual corrector (CatBoost d=3)...", end=" ", flush=True)
residual_oof = time_series_oof(
    CatBoostRegressor(n_estimators=100, depth=3, learning_rate=0.05, random_seed=42, verbose=0),
    X_res_with_pred, residuals
)
# Blend: final = stage1 + alpha * residual_correction
# Find optimal alpha
best_alpha, best_residual_mae = 0, stage1_mae
for alpha in np.arange(0.0, 1.05, 0.05):
    combined = stage1_pred + alpha * residual_oof
    mae = mean_absolute_error(y, combined)
    if mae < best_residual_mae:
        best_residual_mae = mae
        best_alpha = alpha

print(f"MAE: {best_residual_mae:.4f} (α={best_alpha:.2f})")
results.append({
    "label": "7A: Residual stacking (Stage1+CatBoost corrector)",
    "meta": f"Residual(α={best_alpha:.2f})",
    "mae": best_residual_mae,
    "weights": [f"base_ridge", f"α={best_alpha:.2f}"],
})

# Residual with ElasticNet stage 1
S_base_en = np.column_stack(oof_base)
meta_base_en = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], alphas=[0.001, 0.01, 0.1, 1.0],
                             max_iter=5000, cv=5)
meta_base_en.fit(S_base_en, y)
stage1_en_pred = meta_base_en.predict(S_base_en)
residuals_en = y - stage1_en_pred

X_res_with_en = np.column_stack([X_res, stage1_en_pred])
residual_oof_en = time_series_oof(
    CatBoostRegressor(n_estimators=100, depth=3, learning_rate=0.05, random_seed=42, verbose=0),
    X_res_with_en, residuals_en
)
best_alpha_en, best_res_en_mae = 0, mean_absolute_error(y, stage1_en_pred)
for alpha in np.arange(0.0, 1.05, 0.05):
    combined = stage1_en_pred + alpha * residual_oof_en
    mae = mean_absolute_error(y, combined)
    if mae < best_res_en_mae:
        best_res_en_mae = mae
        best_alpha_en = alpha

results.append({
    "label": "7B: Residual stacking (ElasticNet base + corrector)",
    "meta": f"Residual_EN(α={best_alpha_en:.2f})",
    "mae": best_res_en_mae,
    "weights": [f"base_en", f"α={best_alpha_en:.2f}"],
})
print(f"  7B ElasticNet base + residual: MAE {best_res_en_mae:.4f} (α={best_alpha_en:.2f})")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)

production = 8.807  # current deployed best
best = min(results, key=lambda r: r["mae"])

print(f"\n  {'Config':<55} {'MAE':>8} {'Δ vs 8.807':>10}")
print(f"  {'─'*55} {'─'*8} {'─'*10}")

for r in sorted(results, key=lambda x: x["mae"]):
    delta = r["mae"] - production
    star = " ★" if r["mae"] == best["mae"] else ""
    beat = " ✅" if r["mae"] < production else ""
    print(f"  {r['label']:<55} {r['mae']:>8.4f} {delta:>+10.4f}{star}{beat}")

print(f"\n  Production (146 feat, d=4, Ridge):  {production}")
print(f"  Best found:                         {best['mae']:.4f}")
print(f"  Config:                             {best['label']}")
print(f"  Meta:                               {best['meta']}")

if best["mae"] < production:
    print(f"\n  ✅ NEW BEST! Improvement: {production - best['mae']:.4f}")
    print(f"     Deploy this config as production.")
else:
    gap = best["mae"] - production
    print(f"\n  ⚪ Production 8.807 still best (gap: {gap:.4f})")
    print(f"     Lineup features confirmed valuable but current architecture is optimal.")

# ── Section breakdown ──
print(f"\n  {'─'*55}")
print(f"  KEY INSIGHTS:")

# Best per section
sections = {}
for r in results:
    sec = r["label"].split(":")[0]
    if sec not in sections or r["mae"] < sections[sec]["mae"]:
        sections[sec] = r

for sec in sorted(sections.keys()):
    r = sections[sec]
    print(f"    Section {sec}: {r['mae']:.4f} — {r['label']}")

# Meta-learner comparison on same features
print(f"\n  Meta-learner comparison (all 149 features to all learners):")
for r in sorted(results, key=lambda x: x["mae"]):
    if r["label"].startswith("2"):
        print(f"    {r['meta']:30s} MAE: {r['mae']:.4f}")

print(f"\n  Meta-learner comparison (production 146 features):")
for r in sorted(results, key=lambda x: x["mae"]):
    if r["label"].startswith("1") or r["label"].startswith("6"):
        print(f"    {r['meta']:30s} MAE: {r['mae']:.4f}")

print(f"\n  Feature subspacing comparison (best meta per config):")
for sec in ["1A", "3", "4", "5"]:
    matching = [r for r in results if r["label"].startswith(sec)]
    if matching:
        best_m = min(matching, key=lambda x: x["mae"])
        print(f"    {best_m['label']:50s} MAE: {best_m['mae']:.4f}")

print("\n  Done.")
