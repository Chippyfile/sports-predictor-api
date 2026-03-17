"""Quick depth sweep — test depths 4-8 with 50-fold CV on 40K games."""
import sys, os, json, time, copy, warnings
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
        m.fit(X[:train_end], y[:train_end])
        oof[val_start:val_end] = m.predict(X[val_start:val_end])
    return oof

# ── Load data (same as retrain) ──
print("=" * 60)
print("  DEPTH SWEEP (50-fold, depths 4-8)")
print("=" * 60)

df = load_cached()
df = df[df["actual_home_score"].notna()].copy()

if "espn_spread" in df.columns:
    espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df.loc[fill, "market_spread_home"] = espn_s[fill]

# Same quality filter as production
_quality_cols = ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.8
if "referee_1" in df.columns:
    _keep = _keep & df["referee_1"].notna() & (df["referee_1"] != "")
df = df.loc[_keep].reset_index(drop=True)
print(f"  {len(df)} games")

df = _ncaa_backfill_heuristic(df)

with open("referee_profiles.json") as f:
    ncaa_build_features._ref_profiles = json.load(f)

df = df.dropna(subset=["actual_home_score", "actual_away_score"])
X = ncaa_build_features(df)
y = df["actual_home_score"].values - df["actual_away_score"].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
print(f"  {len(X)} games × {X.shape[1]} features\n")

# ── MLP (depth-independent, only run once) ──
print("  MLP-128-64 (shared across all depths)...", end=" ", flush=True)
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                   validation_fraction=0.1, random_state=42)
oof_mlp = time_series_oof(mlp, X_s, y)
mlp_mae = mean_absolute_error(y, oof_mlp)
print(f"MAE: {mlp_mae:.3f}")

# ── Sweep depths ──
results = []
for depth in [4, 5, 6, 7, 8]:
    print(f"\n{'─'*60}")
    print(f"  DEPTH {depth}")
    print(f"{'─'*60}")
    
    t0 = time.time()
    
    print(f"  XGBoost d={depth}...", end=" ", flush=True)
    xgb = XGBRegressor(n_estimators=175, max_depth=depth, learning_rate=0.10, random_state=42, tree_method="hist")
    oof_xgb = time_series_oof(xgb, X_s, y)
    xgb_mae = mean_absolute_error(y, oof_xgb)
    print(f"MAE: {xgb_mae:.3f}")
    
    print(f"  CatBoost d={depth}...", end=" ", flush=True)
    cat = CatBoostRegressor(n_estimators=175, depth=depth, learning_rate=0.10, random_seed=42, verbose=0)
    oof_cat = time_series_oof(cat, X_s, y)
    cat_mae = mean_absolute_error(y, oof_cat)
    print(f"MAE: {cat_mae:.3f}")
    
    # Stack
    oof_stack = np.column_stack([oof_xgb, oof_cat, oof_mlp])
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta.fit(oof_stack, y)
    stacked = meta.predict(oof_stack)
    stack_mae = mean_absolute_error(y, stacked)
    
    elapsed = time.time() - t0
    w = meta.coef_
    
    print(f"  Stacked MAE: {stack_mae:.4f}")
    print(f"  Weights: xgb={w[0]:.3f}, cat={w[1]:.3f}, mlp={w[2]:.3f}")
    print(f"  Time: {elapsed:.0f}s")
    
    results.append({"depth": depth, "xgb": xgb_mae, "cat": cat_mae, "mlp": mlp_mae,
                     "stacked": stack_mae, "weights": list(w.round(4)), "time": elapsed})

# ── Summary ──
print("\n" + "=" * 60)
print("  DEPTH SWEEP RESULTS")
print("=" * 60)
print(f"\n  {'Depth':>5} {'XGB':>8} {'Cat':>8} {'MLP':>8} {'Stacked':>9} {'Δ vs d=7':>9}")
print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*9}")

d7_mae = next((r["stacked"] for r in results if r["depth"] == 7), 0)
best = min(results, key=lambda r: r["stacked"])

for r in results:
    delta = r["stacked"] - d7_mae
    star = " ★" if r["depth"] == best["depth"] else ""
    print(f"  {r['depth']:>5} {r['xgb']:>8.3f} {r['cat']:>8.3f} {r['mlp']:>8.3f} {r['stacked']:>9.4f} {delta:>+9.4f}{star}")

print(f"\n  Best: depth={best['depth']} → Stacked MAE {best['stacked']:.4f}")
if best["depth"] != 7:
    print(f"  ✅ Depth {best['depth']} beats current d=7 by {d7_mae - best['stacked']:.4f}")
else:
    print(f"  ⚪ Depth 7 is optimal")
