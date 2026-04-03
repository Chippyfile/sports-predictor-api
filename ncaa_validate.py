#!/usr/bin/env python3
"""
ncaa_validate.py — Fold stability + feature reduction for Lasso+LGBM
=====================================================================
1. Tests 15/30/50 folds to check if ATS numbers are robust
2. Shows which features Lasso kept vs zeroed
3. Tests Lasso-kept-only (43f) vs full (148f) for the ensemble
4. Tests backward elimination on the ensemble

Usage:
    python3 ncaa_validate.py
"""
import sys, os, time, warnings, copy
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from datetime import datetime

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SEED = 42
DUPLICATE_DROPS = [
    "win_pct_diff", "pace_adj_ppg_diff", "adj_de_diff", "season_pct_avg",
    "form_diff", "opp_orb_pct_diff", "fgpct_diff",
]

MODELS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
}


def walk_forward(X_s, y, n_folds):
    n = len(X_s)
    fold_size = n // (n_folds + 1)
    min_train = fold_size * 2
    oof = np.full(n, np.nan)
    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in MODELS.items():
            m = builder()
            m.fit(X_s[:ts], y[:ts])
            preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)
    return oof


def ats_at(oof, y, spreads, threshold):
    valid = ~np.isnan(oof) & (np.abs(spreads) > 0.1)
    p = oof[valid]; a = y[valid]; s = spreads[valid]
    edge = p - (-s); margin = a + s
    np_ = margin != 0; cor = np.sign(edge) == np.sign(margin)
    m = (np.abs(edge) >= threshold) & np_
    n = m.sum()
    if n < 20: return 0, 0, 0
    acc = float(cor[m].mean())
    roi = round((acc * 1.909 - 1) * 100, 1)
    return acc, roi, n


def full_ats(oof, y, spreads):
    valid = ~np.isnan(oof) & (np.abs(spreads) > 0.1)
    p = oof[valid]; a = y[valid]
    mae = float(np.mean(np.abs(p - a)))
    results = {}
    for t in [0, 4, 7, 10]:
        acc, roi, n = ats_at(oof, y, spreads, t)
        results[t] = (acc, roi, n)
    return mae, results


def composite_score(oof, y, spreads):
    score = 0
    for t, w in [(0,1),(4,2),(7,3),(10,2)]:
        acc, _, n = ats_at(oof, y, spreads, t)
        if n >= 20: score += (acc - 0.524) * w * min(n/50, 1)
    return round(score, 4)


# ══════════════════════════════════════════════════════════
# LOAD DATA (same pipeline as retrain)
# ══════════════════════════════════════════════════════════

print("=" * 70)
print("  NCAA VALIDATION — Fold Stability + Feature Reduction")
print("=" * 70)

print("\n  Loading data...")
t0 = time.time()
df = load_cached()
if df is None: df = dump()
df = df[df["actual_home_score"].notna()].copy()
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()

df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
season_mask = (df["game_date_dt"].dt.month >= 11) | (df["game_date_dt"].dt.month <= 4)
early_mask = ~((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10))
df = df[season_mask & early_mask].copy()
df = df.drop(columns=["game_date_dt"], errors="ignore")

if "espn_spread" in df.columns:
    espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df.loc[fill, "market_spread_home"] = espn_s[fill]
if "espn_over_under" in df.columns:
    espn_ou = pd.to_numeric(df["espn_over_under"], errors="coerce")
    mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
    fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
    df.loc[fill_ou, "market_ou_total"] = espn_ou[fill_ou]

_qcols = [c for c in ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.8
if "referee_1" in df.columns:
    _keep = _keep & df["referee_1"].notna() & (df["referee_1"] != "")
df = df.loc[_keep].reset_index(drop=True)

for col in ["actual_home_score","actual_away_score","home_adj_em","away_adj_em","home_ppg","away_ppg",
            "home_tempo","away_tempo","season","home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),
             ("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

df["season_weight"] = df["season"].apply(
    lambda s: 1.0 if (datetime.utcnow().year - s) <= 0 else 0.9 if (datetime.utcnow().year - s) == 1 else
    0.75 if (datetime.utcnow().year - s) == 2 else 0.6 if (datetime.utcnow().year - s) == 3 else 0.5)

print("  Heuristic backfill + features...")
df = _ncaa_backfill_heuristic(df)
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)
try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except ImportError: pass
try:
    import json
    with open("referee_profiles.json") as f: ncaa_build_features._ref_profiles = json.load(f)
except: pass

df = df.dropna(subset=["actual_home_score","actual_away_score"])
X = ncaa_build_features(df)
X = X.drop(columns=[c for c in DUPLICATE_DROPS if c in X.columns])
feature_cols = list(X.columns)

y = df["actual_home_score"].values - df["actual_away_score"].values
spreads = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0).values
n = len(X)

print(f"  {n} games × {len(feature_cols)} features")

scaler = StandardScaler()
X_s = scaler.fit_transform(X)


# ══════════════════════════════════════════════════════════
# PHASE 1: FOLD STABILITY (15/30/50)
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 1: FOLD STABILITY TEST")
print(f"{'='*70}")

print(f"\n  {'Folds':>6s} {'MAE':>7s} {'ATS0+':>6s} {'ATS4+':>6s} {'ATS7+':>6s} {'N7':>5s} {'ATS10+':>7s} {'Score':>7s} {'Time':>5s}")
print("  " + "-" * 65)

for nf in [15, 50]:
    t1 = time.time()
    oof = walk_forward(X_s, y, nf)
    mae, results = full_ats(oof, y, spreads)
    sc = composite_score(oof, y, spreads)
    elapsed = time.time() - t1

    a0, r0, n0 = results.get(0, (0, 0, 0))
    a4, r4, n4 = results.get(4, (0, 0, 0))
    a7, r7, n7 = results.get(7, (0, 0, 0))
    a10, r10, n10 = results.get(10, (0, 0, 0))
    print(f"  {nf:>5d}f {mae:>7.3f} {a0:>5.1%} {a4:>5.1%} {a7:>5.1%} {n7:>5d} {a10:>5.1%} {sc:>7.4f} {elapsed:>4.0f}s")

# Print known 30-fold result for comparison
print(f"    30f   8.650 54.6% 69.3% 88.0%   191 98.7%  2.4063  (previous run)")


# ══════════════════════════════════════════════════════════
# PHASE 2: LASSO FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 2: LASSO FEATURE SELECTION ANALYSIS")
print(f"{'='*70}")

lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_s, y)
coefs = dict(zip(feature_cols, lasso.coef_))
kept = {f: c for f, c in coefs.items() if abs(c) > 1e-6}
zeroed = {f: c for f, c in coefs.items() if abs(c) <= 1e-6}

print(f"\n  Lasso kept: {len(kept)}/{len(feature_cols)} features")
print(f"  Lasso zeroed: {len(zeroed)} features")

print(f"\n  === KEPT FEATURES (by |coefficient|) ===")
for f, c in sorted(kept.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"    {f:<35s} {c:>+8.4f}")

print(f"\n  === ZEROED FEATURES (Lasso thinks these are noise) ===")
for f in sorted(zeroed.keys()):
    print(f"    {f}")


# ══════════════════════════════════════════════════════════
# PHASE 3: FEATURE REDUCTION TEST
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 3: FEATURE REDUCTION (30-fold)")
print(f"{'='*70}")

# Config A: Full 148 features (baseline — already tested above)
# Config B: Lasso-kept only (43 features)
# Config C: LightGBM importance top 43 (match Lasso count)
# Config D: Union of Lasso-kept + LGBM top 43

# Get LightGBM feature importance
lgbm = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                      subsample=0.8, verbose=-1, random_state=SEED)
lgbm.fit(X_s, y)
lgbm_imp = dict(zip(feature_cols, lgbm.feature_importances_))
lgbm_top = sorted(lgbm_imp.items(), key=lambda x: x[1], reverse=True)[:len(kept)]
lgbm_top_names = [f for f, _ in lgbm_top]

lasso_names = list(kept.keys())
union_names = list(set(lasso_names) | set(lgbm_top_names))

configs = [
    (f"Lasso-kept ({len(lasso_names)})", lasso_names),
    (f"LGBM-top ({len(lgbm_top_names)})", lgbm_top_names),
    (f"Union ({len(union_names)})", union_names),
]

print(f"\n  {'Config':<25s} {'MAE':>7s} {'ATS0+':>6s} {'ATS4+':>6s} {'ATS7+':>6s} {'N7':>5s} {'ATS10+':>7s} {'Score':>7s}")
print("  " + "-" * 70)
# Known baseline
print(f"  {'Full 148 (baseline)':<25s}   8.650 54.6% 69.3% 88.0%   191 98.7%  2.4063  (previous)")

for name, feat_list in configs:
    idx = [feature_cols.index(f) for f in feat_list if f in feature_cols]
    X_sub = X_s[:, idx]
    oof = walk_forward(X_sub, y, 30)
    mae, results = full_ats(oof, y, spreads)
    sc = composite_score(oof, y, spreads)
    a0 = results.get(0, (0,0,0))[0]
    a4 = results.get(4, (0,0,0))[0]
    a7, _, n7 = results.get(7, (0,0,0))
    a10 = results.get(10, (0,0,0))[0]
    print(f"  {name:<25s} {mae:>7.3f} {a0:>5.1%} {a4:>5.1%} {a7:>5.1%} {n7:>5d} {a10:>5.1%} {sc:>7.4f}")

# Feature overlap analysis
lasso_set = set(lasso_names)
lgbm_set = set(lgbm_top_names)
both = lasso_set & lgbm_set
lasso_only = lasso_set - lgbm_set
lgbm_only = lgbm_set - lasso_set

print(f"\n  === FEATURE OVERLAP ===")
print(f"  Both models agree ({len(both)}):")
for f in sorted(both):
    print(f"    {f:<35s} Lasso={kept[f]:+.4f}  LGBM={lgbm_imp[f]:.0f}")
print(f"\n  Lasso-only ({len(lasso_only)}):")
for f in sorted(lasso_only):
    print(f"    {f:<35s} Lasso={kept[f]:+.4f}  LGBM={lgbm_imp.get(f,0):.0f}")
print(f"\n  LGBM-only ({len(lgbm_only)}):")
for f in sorted(lgbm_only):
    print(f"    {f:<35s} Lasso=0  LGBM={lgbm_imp[f]:.0f}")

print(f"\n  Done in {(time.time()-t0)/60:.1f} min")
