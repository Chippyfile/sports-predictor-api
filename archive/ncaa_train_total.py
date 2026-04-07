#!/usr/bin/env python3
"""
NCAA O/U Total Model — Possession-based scoring prediction.

Predicts actual game total using tempo/efficiency features.
Trained as RESIDUAL (actual_total - market_total) to learn where Vegas is wrong,
then also stores a direct total prediction for display.

Integrates into O/U v4 as a 4th agreement signal.
"""
import sys, os, warnings, time
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from collections import defaultdict, deque
import json, pickle, io, base64, joblib

from dump_training_data import load_cached

SEED = 42; N_FOLDS = 30

# ══════════════════════════════════════════════════════════════
#  FEATURE DEFINITIONS — Possession/efficiency focused
# ══════════════════════════════════════════════════════════════

# Core tempo/efficiency features
TEMPO_FEATURES = [
    "home_tempo", "away_tempo", "tempo_avg", "tempo_diff",
    "home_adj_oe", "away_adj_oe", "home_adj_de", "away_adj_de",
    "adj_oe_diff", "adj_de_diff",
    "home_ppg", "away_ppg", "home_opp_ppg", "away_opp_ppg",
    "ppg_sum", "opp_ppg_sum",
]

# Shooting features (drive total variance)
SHOOTING_FEATURES = [
    "home_fgpct", "away_fgpct", "fgpct_avg",
    "home_threepct", "away_threepct", "threepct_avg",
    "home_ftpct", "away_ftpct",
    "home_three_rate", "away_three_rate", "three_rate_avg",
]

# Style/context features
STYLE_FEATURES = [
    "home_orb_pct", "away_orb_pct", "orb_pct_sum",
    "home_drb_pct", "away_drb_pct",
    "home_blocks", "away_blocks", "blocks_sum",
    "home_assist_rate", "away_assist_rate", "assist_rate_sum",
    "neutral_site",
    "is_early",
]

# Market context
MARKET_FEATURES = [
    "market_ou_total", "abs_spread",
]

# Interaction features (computed below)
INTERACTION_FEATURES = [
    "tempo_x_oe_home", "tempo_x_oe_away",  # fast + efficient = high scoring
    "tempo_x_de_opp",  # fast pace vs bad defense
    "oe_product",       # combined offensive power
    "de_product",       # combined defensive weakness
    "pace_mismatch",    # tempo difference (fast vs slow)
    "efficiency_gap",   # total efficiency gap
    "estimated_possessions",
    "formula_total",    # KenPom-style estimate
]


def _col(df, name, default=0):
    """Safe column accessor — returns Series of defaults if column missing."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def build_total_features(df):
    """Build feature matrix for total prediction."""
    X = pd.DataFrame(index=df.index)
    
    # ── Core tempo/efficiency ──
    X["home_tempo"] = _col(df, "home_tempo", 68)
    X["away_tempo"] = _col(df, "away_tempo", 68)
    X["tempo_avg"] = (X["home_tempo"] + X["away_tempo"]) / 2
    X["tempo_diff"] = X["home_tempo"] - X["away_tempo"]
    
    X["home_adj_oe"] = _col(df, "home_adj_oe", 100)
    X["away_adj_oe"] = _col(df, "away_adj_oe", 100)
    X["home_adj_de"] = _col(df, "home_adj_de", 100)
    X["away_adj_de"] = _col(df, "away_adj_de", 100)
    X["adj_oe_diff"] = X["home_adj_oe"] - X["away_adj_oe"]
    X["adj_de_diff"] = X["home_adj_de"] - X["away_adj_de"]
    
    X["home_ppg"] = _col(df, "home_ppg", 74)
    X["away_ppg"] = _col(df, "away_ppg", 74)
    X["home_opp_ppg"] = _col(df, "home_opp_ppg", 72)
    X["away_opp_ppg"] = _col(df, "away_opp_ppg", 72)
    X["ppg_sum"] = X["home_ppg"] + X["away_ppg"]
    X["opp_ppg_sum"] = X["home_opp_ppg"] + X["away_opp_ppg"]
    
    # ── Shooting ──
    X["home_fgpct"] = _col(df, "home_fgpct", 0.44)
    X["away_fgpct"] = _col(df, "away_fgpct", 0.44)
    X["fgpct_avg"] = (X["home_fgpct"] + X["away_fgpct"]) / 2
    
    X["home_threepct"] = _col(df, "home_threepct", 0.34)
    X["away_threepct"] = _col(df, "away_threepct", 0.34)
    X["threepct_avg"] = (X["home_threepct"] + X["away_threepct"]) / 2
    
    X["home_ftpct"] = _col(df, "home_ftpct", 0.72)
    X["away_ftpct"] = _col(df, "away_ftpct", 0.72)
    
    X["home_three_rate"] = _col(df, "home_three_rate", 0.35)
    X["away_three_rate"] = _col(df, "away_three_rate", 0.35)
    X["three_rate_avg"] = (X["home_three_rate"] + X["away_three_rate"]) / 2
    
    # ── Style ──
    X["home_orb_pct"] = _col(df, "home_orb_pct", 0.28)
    X["away_orb_pct"] = _col(df, "away_orb_pct", 0.28)
    X["orb_pct_sum"] = X["home_orb_pct"] + X["away_orb_pct"]
    X["home_drb_pct"] = _col(df, "home_drb_pct", 0.72)
    X["away_drb_pct"] = _col(df, "away_drb_pct", 0.72)
    
    X["home_blocks"] = _col(df, "home_blocks", 3)
    X["away_blocks"] = _col(df, "away_blocks", 3)
    X["blocks_sum"] = X["home_blocks"] + X["away_blocks"]
    
    X["home_assist_rate"] = _col(df, "home_assist_rate", 0.50)
    X["away_assist_rate"] = _col(df, "away_assist_rate", 0.50)
    X["assist_rate_sum"] = X["home_assist_rate"] + X["away_assist_rate"]
    
    X["neutral_site"] = _col(df, "neutral_site", 0)
    
    # is_early: November/early December games
    gd = pd.to_datetime(df.get("game_date", "2026-01-01"), errors="coerce")
    X["is_early"] = ((gd.dt.month == 11) | ((gd.dt.month == 12) & (gd.dt.day <= 15))).astype(int)
    
    # ── Market ──
    X["market_ou_total"] = _col(df, "market_ou_total", 145)
    X["abs_spread"] = _col(df, "market_spread_home", 0).abs()
    
    # ── Interactions (where the edge is) ──
    LG_AVG = 100.0  # average efficiency ~100 per KenPom convention
    
    # Fast + efficient = explosive
    X["tempo_x_oe_home"] = X["tempo_avg"] * X["home_adj_oe"] / LG_AVG
    X["tempo_x_oe_away"] = X["tempo_avg"] * X["away_adj_oe"] / LG_AVG
    
    # Fast pace vs bad defense
    X["tempo_x_de_opp"] = X["tempo_avg"] * (X["home_adj_de"] + X["away_adj_de"]) / (2 * LG_AVG)
    
    # Combined offensive/defensive power
    X["oe_product"] = X["home_adj_oe"] * X["away_adj_oe"] / LG_AVG
    X["de_product"] = X["home_adj_de"] * X["away_adj_de"] / LG_AVG
    
    # Pace mismatch
    X["pace_mismatch"] = (X["home_tempo"] - X["away_tempo"]).abs()
    
    # Total efficiency gap 
    X["efficiency_gap"] = (X["home_adj_oe"] - X["home_adj_de"]) + (X["away_adj_oe"] - X["away_adj_de"])
    
    # KenPom-style formula estimate
    X["estimated_possessions"] = X["tempo_avg"]
    home_pts = X["estimated_possessions"] * X["home_adj_oe"] * X["away_adj_de"] / (LG_AVG ** 2)
    away_pts = X["estimated_possessions"] * X["away_adj_oe"] * X["home_adj_de"] / (LG_AVG ** 2)
    X["formula_total"] = home_pts + away_pts
    
    return X


def walk_forward_total(X_arr, y, mkt, n_folds, weights):
    """Walk-forward validation predicting residual (actual - market)."""
    n = len(X_arr)
    fold_size = n // (n_folds + 1)
    min_train = fold_size * 2
    
    residual = y - mkt
    oof_residual = np.full(n, np.nan)
    oof_direct = np.full(n, np.nan)
    
    models_cfg = {
        "Lasso": lambda: Lasso(alpha=0.05, max_iter=5000),
        "Ridge": lambda: Ridge(alpha=1.0),
        "LGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
    }
    
    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break
        
        preds_res = []
        preds_dir = []
        for name, builder in models_cfg.items():
            m_res = builder()
            m_dir = builder()
            
            # Residual model (actual - market)
            try: m_res.fit(X_arr[:ts], residual[:ts], sample_weight=weights[:ts])
            except TypeError: m_res.fit(X_arr[:ts], residual[:ts])
            preds_res.append(m_res.predict(X_arr[ts:te]))
            
            # Direct total model
            try: m_dir.fit(X_arr[:ts], y[:ts], sample_weight=weights[:ts])
            except TypeError: m_dir.fit(X_arr[:ts], y[:ts])
            preds_dir.append(m_dir.predict(X_arr[ts:te]))
        
        oof_residual[ts:te] = np.mean(preds_res, axis=0)
        oof_direct[ts:te] = np.mean(preds_dir, axis=0)
    
    return oof_residual, oof_direct


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("  NCAA TOTAL MODEL — Possession-Based Scoring Prediction")
print("=" * 70)

df = load_cached()
if df is None:
    print("No cache!"); sys.exit(1)

# Filter
df = df[df["actual_home_score"].notna() & df["actual_away_score"].notna()].copy()
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()

# Quality filter: need market O/U total (use fillna for tempo/efficiency instead of dropping)
df["market_ou_total"] = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce")
df = df[df["market_ou_total"].notna() & (df["market_ou_total"] > 80) & (df["market_ou_total"] < 220)].copy()

# Target
df["actual_total"] = df["actual_home_score"].astype(float) + df["actual_away_score"].astype(float)
df = df[(df["actual_total"] > 60) & (df["actual_total"] < 250)].copy()

# Fillna for tempo/efficiency (don't drop — defaults are reasonable)
for col, default in [("home_tempo", 68), ("away_tempo", 68), ("home_adj_oe", 100), ("away_adj_oe", 100)]:
    df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").fillna(default)

# Sort chronologically
df = df.sort_values("game_date").reset_index(drop=True)
print(f"  {len(df)} games with tempo + efficiency + market total")
print(f"  Seasons: {sorted(df['season'].unique())}")

# Build features
X = build_total_features(df)
all_features = list(X.columns)
print(f"  {len(all_features)} features")

# Arrays
y = df["actual_total"].values
mkt = df["market_ou_total"].values
seasons = df["season"].values.astype(int)

# Weights — recent scheme (best for O/U)
weights = np.array([{2026:1.0,2025:1.0,2024:0.5,2023:0.25}.get(s, 0.1) for s in seasons])

# Fill NaN
X = X.fillna(0)
X_arr = StandardScaler().fit_transform(X.values)

print(f"\n  Market total MAE: {mean_absolute_error(y, mkt):.3f}")
print(f"  PPG sum MAE:      {mean_absolute_error(y, X['ppg_sum'].values):.3f}")
print(f"  Formula total MAE:{mean_absolute_error(y, X['formula_total'].values):.3f}")

# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════
print(f"\n  Walk-forward ({N_FOLDS} folds)...")
oof_res, oof_dir = walk_forward_total(X_arr, y, mkt, N_FOLDS, weights)

valid = ~np.isnan(oof_res)
pred_total_res = mkt[valid] + oof_res[valid]
pred_total_dir = oof_dir[valid]
actual = y[valid]
mkt_v = mkt[valid]

res_mae = mean_absolute_error(actual, pred_total_res)
dir_mae = mean_absolute_error(actual, pred_total_dir)
mkt_mae = mean_absolute_error(actual, mkt_v)

# Blend: weighted average of residual and direct
for blend_w in [0.5, 0.6, 0.7, 0.8]:
    blended = blend_w * pred_total_res + (1 - blend_w) * pred_total_dir
    blend_mae = mean_absolute_error(actual, blended)
    print(f"  Blend {blend_w:.0%} res / {1-blend_w:.0%} dir: MAE {blend_mae:.3f}")

best_blend = 0.7  # typical sweet spot, will verify
blended = best_blend * pred_total_res + (1 - best_blend) * pred_total_dir
blend_mae = mean_absolute_error(actual, blended)

print(f"\n  Results ({sum(valid)} games):")
print(f"    Market total MAE:        {mkt_mae:.3f}")
print(f"    Residual model MAE:      {res_mae:.3f} ({'✅ beats' if res_mae < mkt_mae else '❌ loses to'} market)")
print(f"    Direct model MAE:        {dir_mae:.3f}")
print(f"    Blended (70/30) MAE:     {blend_mae:.3f}")

# ── O/U accuracy at thresholds ──
print(f"\n  O/U Accuracy (residual model):")
res_pred = pred_total_res
for thresh in [0, 1, 2, 3, 4, 5]:
    # Predict OVER when model total > market + thresh
    over_mask = res_pred > (mkt_v + thresh)
    under_mask = res_pred < (mkt_v - thresh)
    
    over_correct = (actual[over_mask] > mkt_v[over_mask]).sum() if over_mask.sum() > 0 else 0
    under_correct = (actual[under_mask] < mkt_v[under_mask]).sum() if under_mask.sum() > 0 else 0
    
    n_over = over_mask.sum()
    n_under = under_mask.sum()
    
    over_pct = over_correct / max(n_over, 1)
    under_pct = under_correct / max(n_under, 1)
    total_pct = (over_correct + under_correct) / max(n_over + n_under, 1)
    
    print(f"    Thresh ±{thresh}: OVER {over_pct:.1%} ({n_over}) | UNDER {under_pct:.1%} ({n_under}) | Combined {total_pct:.1%} ({n_over + n_under})")

# ── Lasso feature importance ──
print(f"\n  Lasso feature selection (residual target):")
residual_target = y - mkt
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
lasso = Lasso(alpha=0.05, max_iter=5000)
lasso.fit(X_scaled, residual_target, sample_weight=weights)
coefs = sorted(zip(all_features, lasso.coef_), key=lambda x: -abs(x[1]))
kept = [(f, c) for f, c in coefs if abs(c) > 0.001]
dropped = [(f, c) for f, c in coefs if abs(c) <= 0.001]
print(f"    Kept: {len(kept)}/{len(all_features)}")
for f, c in kept:
    print(f"      {f:>30s}: {c:+.4f}")
print(f"    Dropped: {[f for f, c in dropped]}")

# ══════════════════════════════════════════════════════════════
#  PRODUCTION TRAINING
# ══════════════════════════════════════════════════════════════
if "--upload" in sys.argv:
    print(f"\n{'=' * 70}")
    print(f"  PRODUCTION TRAINING")
    print(f"{'=' * 70}")
    
    # Use kept features only
    kept_features = [f for f, c in kept]
    X_kept = X[kept_features].fillna(0)
    X_scaled_kept = StandardScaler().fit_transform(X_kept.values)
    
    residual = y - mkt
    
    # Train residual ensemble
    res_models = []
    for name, cls in [("Lasso", Lasso(alpha=0.05, max_iter=5000)),
                       ("Ridge", Ridge(alpha=1.0)),
                       ("LGBM", LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                               subsample=0.8, verbose=-1, random_state=SEED))]:
        try: cls.fit(X_scaled_kept, residual, sample_weight=weights)
        except TypeError: cls.fit(X_scaled_kept, residual)
        res_preds = cls.predict(X_scaled_kept)
        mae = mean_absolute_error(residual, res_preds)
        print(f"    {name} residual MAE: {mae:.3f}")
        res_models.append((name, cls))
    
    # Train direct ensemble
    dir_models = []
    for name, cls in [("Lasso", Lasso(alpha=0.05, max_iter=5000)),
                       ("Ridge", Ridge(alpha=1.0)),
                       ("LGBM", LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                               subsample=0.8, verbose=-1, random_state=SEED))]:
        try: cls.fit(X_scaled_kept, y, sample_weight=weights)
        except TypeError: cls.fit(X_scaled_kept, y)
        dir_preds = cls.predict(X_scaled_kept)
        mae = mean_absolute_error(y, dir_preds)
        print(f"    {name} direct MAE: {mae:.3f}")
        dir_models.append((name, cls))
    
    scaler_prod = StandardScaler()
    scaler_prod.fit(X_kept.values)
    
    bundle = {
        "version": "total_v1",
        "features": kept_features,
        "scaler": scaler_prod,
        "res_models": res_models,
        "dir_models": dir_models,
        "blend_weight": best_blend,  # 0.7 res + 0.3 dir
        "market_mae": mkt_mae,
        "res_mae": res_mae,
        "dir_mae": dir_mae,
        "blend_mae": blend_mae,
        "n_train": len(df),
    }
    
    # Save locally
    local_path = "ncaa_total_v1.pkl"
    with open(local_path, "wb") as f:
        joblib.dump(bundle, f, compress=3)
    size_kb = os.path.getsize(local_path) / 1024
    print(f"\n  Saved: {local_path} ({size_kb:.0f} KB)")
    
    # Upload to Supabase
    print("  Uploading to Supabase as 'ncaa_total'...")
    import requests
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    raw = buf.getvalue()
    encoded = base64.b64encode(raw).decode()
    
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
    SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
    
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/model_store",
        headers=headers,
        json={"name": "ncaa_total", "data": encoded},
        timeout=60)
    if r.ok or r.status_code == 201:
        print(f"  ✅ Upload successful ({len(raw)/1024:.0f} KB)")
    else:
        # Try PATCH
        r2 = requests.patch(
            f"{SUPABASE_URL}/rest/v1/model_store?name=eq.ncaa_total",
            headers={**headers, "Prefer": "return=minimal"},
            json={"data": encoded},
            timeout=60)
        if r2.ok:
            print(f"  ✅ Upload successful via PATCH ({len(raw)/1024:.0f} KB)")
        else:
            print(f"  ❌ Upload failed: {r2.status_code} {r2.text[:200]}")
    
    print(f"\n{'=' * 70}")
    print(f"  NCAA TOTAL MODEL v1 COMPLETE")
    print(f"  Features: {len(kept_features)} | Games: {len(df)}")
    print(f"  Market MAE: {mkt_mae:.3f} | Model MAE: {blend_mae:.3f}")
    print(f"{'=' * 70}")

else:
    print(f"\n  Run with --upload to train production model and upload to Supabase")
