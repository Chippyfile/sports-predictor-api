#!/usr/bin/env python3
"""
optimization_sweep.py — Final push to crack sub-8.8
=====================================================
Tests data curation, depth, CatBoost iterations, and season weighting.

Configs:
  DATA CURATION:
    A) All seasons (current production)
    B) Drop 2021 (COVID — HCA 5.91 vs 7-8 normal)
    C) 2019+ only (modern game, transfer portal era)
    D) 2019+ no 2021

  DEPTH:
    d=3, d=4 (current), d=5

  CATBOOST ITERATIONS:
    175 (current), 300, 500

  RECENCY WEIGHTING:
    Current (0.5 for 2015), Aggressive (0.0 for pre-2020)

Run:
    python3 -u optimization_sweep.py
    nohup python3 -u optimization_sweep.py > opt_sweep_results.txt 2>&1 &
"""

import sys, os, json, copy, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import load_cached

N_SPLITS = 50

def time_series_oof(model, X, y, w=None, n_splits=N_SPLITS):
    n = len(X)
    oof = np.zeros(n)
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        te = fold_size * (i + 2)
        vs, ve = te, min(te + fold_size, n)
        if vs >= n: break
        m = copy.deepcopy(model)
        if w is not None and hasattr(m, 'fit'):
            try:
                m.fit(X[:te], y[:te], sample_weight=w[:te])
            except TypeError:
                m.fit(X[:te], y[:te])
        else:
            m.fit(X[:te], y[:te])
        oof[vs:ve] = m.predict(X[vs:ve])
    return oof

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  OPTIMIZATION SWEEP — Data × Depth × Iterations")
print("=" * 70)

df_raw = load_cached()
df_raw = df_raw[df_raw["actual_home_score"].notna()].copy()

# ESPN odds fallback
if "espn_spread" in df_raw.columns:
    es = pd.to_numeric(df_raw["espn_spread"], errors="coerce")
    ms = pd.to_numeric(df_raw.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fl = (ms.isna() | (ms == 0)) & es.notna()
    df_raw.loc[fl, "market_spread_home"] = es[fl]

print(f"  Raw data: {len(df_raw)} games")

# ═══════════════════════════════════════════════════════════════
# DATA CURATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def apply_quality_filter(df):
    """Standard 80% quality + ref filter."""
    qc = ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"]
    qc = [c for c in qc if c in df.columns]
    qm = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in qc})
    k = qm.mean(axis=1) >= 0.8
    if "referee_1" in df.columns:
        k = k & df["referee_1"].notna() & (df["referee_1"] != "")
    return df.loc[k].reset_index(drop=True)

def build_features(df, weight_scheme="current"):
    """Full pipeline: backfill → ref profiles → features → target."""
    df = _ncaa_backfill_heuristic(df.copy())
    df = df.dropna(subset=["actual_home_score", "actual_away_score"])
    X = ncaa_build_features(df)
    y = df["actual_home_score"].values - df["actual_away_score"].values
    
    from datetime import datetime
    current_year = datetime.utcnow().year
    
    if weight_scheme == "current":
        # Current: 1.0, 0.9, 0.75, 0.6, 0.5
        w = df["season"].apply(lambda s: 1.0 if (current_year-s)<=0 else 0.9 if (current_year-s)==1 
            else 0.75 if (current_year-s)==2 else 0.6 if (current_year-s)==3 else 0.5).values
    elif weight_scheme == "aggressive":
        # Aggressive: recent=1.0, older rapidly decays
        w = df["season"].apply(lambda s: 1.0 if (current_year-s)<=0 else 0.9 if (current_year-s)==1 
            else 0.7 if (current_year-s)==2 else 0.4 if (current_year-s)==3 
            else 0.2 if (current_year-s)==4 else 0.1).values
    elif weight_scheme == "equal":
        w = np.ones(len(df))
    else:
        w = np.ones(len(df))
    
    return X, y, w

# Load ref profiles once
with open("referee_profiles.json") as f:
    ncaa_build_features._ref_profiles = json.load(f)

# ═══════════════════════════════════════════════════════════════
# DEFINE EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

data_configs = {
    "all": lambda df: df,
    "no_2021": lambda df: df[df["season"] != 2021],
    "2019+": lambda df: df[df["season"] >= 2019],
    "2019+_no21": lambda df: df[(df["season"] >= 2019) & (df["season"] != 2021)],
    "2022+": lambda df: df[df["season"] >= 2022],
}

# Test matrix (reduced to avoid 10-hour runs)
experiments = [
    # ── Current production baseline ──
    ("all", 4, 175, "current", "BASELINE: current production"),
    
    # ── Data curation (depth 4, 175 iters) ──
    ("no_2021", 4, 175, "current", "Drop 2021 COVID season"),
    ("2019+", 4, 175, "current", "2019+ only (modern era)"),
    ("2019+_no21", 4, 175, "current", "2019+ no COVID"),
    ("2022+", 4, 175, "current", "2022+ only (recent)"),
    
    # ── Depth sweep on best data (depth 3-5) ──
    ("all", 3, 175, "current", "Depth 3"),
    ("all", 5, 175, "current", "Depth 5"),
    
    # ── CatBoost iterations (depth 4) ──
    ("all", 4, 300, "current", "Cat iterations 300"),
    ("all", 4, 500, "current", "Cat iterations 500"),
    
    # ── Recency weighting ──
    ("all", 4, 175, "aggressive", "Aggressive recency weighting"),
    ("all", 4, 175, "equal", "Equal weighting (no recency)"),
    
    # ── Combined: best data + best depth candidates ──
    ("no_2021", 3, 175, "current", "No 2021 + depth 3"),
    ("no_2021", 4, 300, "current", "No 2021 + Cat 300 iters"),
    ("2019+_no21", 3, 175, "current", "2019+ no COVID + depth 3"),
    ("2019+_no21", 4, 175, "aggressive", "2019+ no COVID + aggressive weights"),
]

# ═══════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

results = []

for data_key, depth, cat_iters, weight_scheme, label in experiments:
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  data={data_key} depth={depth} cat_iters={cat_iters} weights={weight_scheme}")
    print(f"{'─'*70}")
    
    t0 = time.time()
    
    # Prepare data
    df_subset = data_configs[data_key](df_raw.copy())
    df_filtered = apply_quality_filter(df_subset)
    
    if len(df_filtered) < 5000:
        print(f"  ⚠️ Only {len(df_filtered)} games — skipping")
        continue
    
    X, y, w = build_features(df_filtered, weight_scheme)
    n = len(X)
    print(f"  {n} games × {X.shape[1]} features")
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    # Train models
    print(f"  XGB d={depth}...", end=" ", flush=True)
    xgb = XGBRegressor(n_estimators=175, max_depth=depth, learning_rate=0.10,
                        random_state=42, tree_method="hist")
    oof_xgb = time_series_oof(xgb, X_s, y, w)
    xgb_mae = mean_absolute_error(y, oof_xgb)
    print(f"MAE: {xgb_mae:.4f}")
    
    print(f"  Cat d={depth} n={cat_iters}...", end=" ", flush=True)
    cat = CatBoostRegressor(n_estimators=cat_iters, depth=depth, learning_rate=0.10,
                            random_seed=42, verbose=0)
    oof_cat = time_series_oof(cat, X_s, y, w)
    cat_mae = mean_absolute_error(y, oof_cat)
    print(f"MAE: {cat_mae:.4f}")
    
    print(f"  MLP...", end=" ", flush=True)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                       validation_fraction=0.1, random_state=42)
    oof_mlp = time_series_oof(mlp, X_s, y)
    mlp_mae = mean_absolute_error(y, oof_mlp)
    print(f"MAE: {mlp_mae:.4f}")
    
    # Stack
    S = np.column_stack([oof_xgb, oof_cat, oof_mlp])
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta.fit(S, y)
    stacked = meta.predict(S)
    stack_mae = mean_absolute_error(y, stacked)
    elapsed = time.time() - t0
    wt = meta.coef_
    
    print(f"  Stacked MAE: {stack_mae:.4f}")
    print(f"  Weights: xgb={wt[0]:.3f}, cat={wt[1]:.3f}, mlp={wt[2]:.3f}")
    print(f"  Time: {elapsed:.0f}s")
    
    results.append({
        "label": label, "data": data_key, "depth": depth,
        "cat_iters": cat_iters, "weights": weight_scheme,
        "n_games": n, "xgb": xgb_mae, "cat": cat_mae, "mlp": mlp_mae,
        "stacked": stack_mae, "meta_weights": list(wt.round(4)),
        "time": elapsed,
    })

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)

production = 8.807
best = min(results, key=lambda r: r["stacked"])

print(f"\n  {'Label':<40} {'Games':>6} {'XGB':>8} {'Cat':>8} {'MLP':>8} {'Stack':>8} {'Δ':>8}")
print(f"  {'─'*40} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

for r in sorted(results, key=lambda x: x["stacked"]):
    delta = r["stacked"] - production
    star = " ★" if r["stacked"] == best["stacked"] else ""
    beat = " ✅" if r["stacked"] < production else ""
    print(f"  {r['label']:<40} {r['n_games']:>6} {r['xgb']:>8.4f} {r['cat']:>8.4f} {r['mlp']:>8.4f} {r['stacked']:>8.4f} {delta:>+8.4f}{star}{beat}")

print(f"\n  Production: {production} (159 feat, d=4, 175 iters, all seasons)")
print(f"  Best found: {best['stacked']:.4f} — {best['label']}")
print(f"  Meta: {best['meta_weights']}")

# ── Breakdown by factor ──
print(f"\n  {'─'*50}")
print(f"  FACTOR ANALYSIS:")

# Best per data curation
print(f"\n  Data curation (depth=4, 175 iters):")
for dk in ["all", "no_2021", "2019+", "2019+_no21", "2022+"]:
    matching = [r for r in results if r["data"] == dk and r["depth"] == 4 and r["cat_iters"] == 175 and r["weights"] == "current"]
    if matching:
        r = matching[0]
        print(f"    {dk:20s} {r['n_games']:>6} games → {r['stacked']:.4f}")

# Best per depth
print(f"\n  Depth (all seasons, 175 iters):")
for d in [3, 4, 5]:
    matching = [r for r in results if r["data"] == "all" and r["depth"] == d and r["cat_iters"] == 175 and r["weights"] == "current"]
    if matching:
        r = matching[0]
        print(f"    depth={d} → {r['stacked']:.4f}")

# CatBoost iterations
print(f"\n  CatBoost iterations (all seasons, depth=4):")
for ci in [175, 300, 500]:
    matching = [r for r in results if r["data"] == "all" and r["depth"] == 4 and r["cat_iters"] == ci and r["weights"] == "current"]
    if matching:
        r = matching[0]
        print(f"    n={ci} → Cat={r['cat']:.4f} Stack={r['stacked']:.4f}")

# Weighting
print(f"\n  Recency weighting (all seasons, depth=4):")
for ws in ["current", "aggressive", "equal"]:
    matching = [r for r in results if r["data"] == "all" and r["depth"] == 4 and r["cat_iters"] == 175 and r["weights"] == ws]
    if matching:
        r = matching[0]
        print(f"    {ws:15s} → {r['stacked']:.4f}")

if best["stacked"] < production:
    print(f"\n  ✅ NEW BEST! Improvement: {production - best['stacked']:.4f}")
else:
    print(f"\n  ⚪ Production still best (gap: {best['stacked'] - production:.4f})")

print("\n  Done.")
