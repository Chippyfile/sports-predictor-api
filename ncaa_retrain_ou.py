#!/usr/bin/env python3
"""
ncaa_retrain_ou.py — NCAA O/U Model: CatBoost + LightGBM
=========================================================
Architecture: CatBoost(d=3, 650i) + LightGBM(300) → simple average
Features: 20 (v1 hand-picked set — validated to beat 41-feature data-driven set)
Data filters: No 2020/2021, no games before Nov 10 of any season

Usage:
    python3 ncaa_retrain_ou.py              # Train + evaluate
    python3 ncaa_retrain_ou.py --upload     # Train + upload to Supabase as 'ncaa_ou'
    python3 ncaa_retrain_ou.py --refresh    # Pull fresh data first
"""
import sys, os, time, warnings, copy
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, joblib, io, base64, requests
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import shap

from ml_utils import StackedRegressor
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')

SEED = 42
N_FOLDS = 30

# v1 hand-picked O/U features (beat 41-feature data-driven set)
OU_FEATURES = [
    'mkt_total', 'tempo_avg', 'ref_ou_bias', 'venue_advantage',
    'hca_pts', 'starter_balance_diff', 'lineup_changes_diff',
    'roll_ats_margin_gated', 'to_margin_diff', 'matchup_efg',
    'roll_top3_share_diff', 'assist_rate_diff',
    'mkt_spread_vs_model', 'matchup_orb',
    'starter_experience_diff', 'opp_fta_rate_diff',
    'lineup_stability_diff', 'ref_foul_rate',
    'info_gain_diff', 'eff_vol_diff',
]


def time_series_oof(models_dict, X_s, y, n_splits):
    n = len(X_s)
    oof = np.full(n, np.nan)
    fold_size = n // (n_splits + 1)
    min_train = fold_size * 2
    for fold in range(n_splits):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in models_dict.items():
            m = builder()
            m.fit(X_s[:ts], y[:ts])
            preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)
    return oof


def ou_analysis(oof, y_total, mkt_total):
    valid = ~np.isnan(oof) & (mkt_total > 0)
    pred = oof[valid]; actual = y_total[valid]; mkt = mkt_total[valid]
    mae = float(np.mean(np.abs(pred - actual)))
    mkt_mae = float(np.mean(np.abs(mkt - actual)))

    print(f"\n  Walk-forward MAE: {mae:.3f}")
    print(f"  Market MAE:       {mkt_mae:.3f}")
    print(f"  Model vs market:  {mae - mkt_mae:+.3f}")
    print(f"  Games with O/U:   {valid.sum()}")

    edge = pred - mkt
    print(f"\n  {'Edge':>6s} {'Games':>7s} {'Acc':>6s} {'ROI':>7s} {'OVER':>8s} {'UNDER':>8s}")
    print("  " + "-" * 50)

    results = {}
    for t in [2, 3, 4, 5, 6, 8]:
        mask = np.abs(edge) >= t
        push = actual == mkt; decided = mask & ~push; n_games = decided.sum()
        if n_games < 20: continue
        co = (edge[decided] > 0) & (actual[decided] > mkt[decided])
        cu = (edge[decided] < 0) & (actual[decided] < mkt[decided])
        acc = float((co | cu).mean())
        roi = round((acc * 1.909 - 1) * 100, 1)
        # Directional
        n_over = (edge[decided] > 0).sum(); n_under = (edge[decided] < 0).sum()
        ov_acc = co.sum() / max(n_over, 1); un_acc = cu.sum() / max(n_under, 1)
        tag = "YES" if acc > 0.524 else "no"
        print(f"  {t:>5d}+ {n_games:>7d} {acc:>5.1%} {roi:>+6.1f}%  {ov_acc:.0%}({n_over}) {un_acc:.0%}({n_under})  {tag}")
        results[t] = {"acc": acc, "n": n_games, "roi": roi}

    return mae, results


# ══════════════════════════════════════════════════════════
# LOAD + FILTER DATA
# ══════════════════════════════════════════════════════════

print("=" * 70)
print("  NCAA O/U MODEL — CatBoost + LightGBM (20 features)")
print("=" * 70)

upload = "--upload" in sys.argv
refresh = "--refresh" in sys.argv

print("\n  Loading data...")
t0 = time.time()

if refresh:
    df = dump()
else:
    df = load_cached()
    if df is None: df = dump()

df = df[df["actual_home_score"].notna()].copy()

# ── Data filters ──
n_before = len(df)
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()
n_covid = n_before - len(df)
if n_covid > 0:
    print(f"  Dropped {n_covid} games from 2020/2021")

n_before = len(df)
df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
season_mask = (df["game_date_dt"].dt.month >= 11) | (df["game_date_dt"].dt.month <= 4)
early_mask = ~((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10))
df = df[season_mask & early_mask].copy()
n_early = n_before - len(df)
if n_early > 0:
    print(f"  Dropped {n_early} games before Nov 10 / off-season")
df = df.drop(columns=["game_date_dt"], errors="ignore")

print(f"  {len(df)} games after filters ({time.time()-t0:.0f}s)")

# ESPN odds fallback
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

# Quality filter
_qcols = [c for c in ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.8
if "referee_1" in df.columns:
    _has_ref = df["referee_1"].notna() & (df["referee_1"] != "")
    _keep = _keep & _has_ref
df = df.loc[_keep].reset_index(drop=True)

for col in ["actual_home_score","actual_away_score","home_adj_em","away_adj_em",
            "home_ppg","away_ppg","home_tempo","away_tempo","season",
            "home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),
             ("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

df["season_weight"] = df["season"].apply(
    lambda s: 1.0 if (datetime.utcnow().year - s) <= 0 else 0.9 if (datetime.utcnow().year - s) == 1 else
    0.75 if (datetime.utcnow().year - s) == 2 else 0.6 if (datetime.utcnow().year - s) == 3 else 0.5)

print("  Heuristic backfill...")
df = _ncaa_backfill_heuristic(df)
print("  Computing crowd_shock + h2h/conf/form...")
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)

try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except ImportError: pass

try:
    import json
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
except FileNotFoundError: pass

df = df.dropna(subset=["actual_home_score","actual_away_score"])

# Build full features, then select O/U subset
print("  Building features...")
X_full = ncaa_build_features(df)

ou_avail = [f for f in OU_FEATURES if f in X_full.columns]
missing = [f for f in OU_FEATURES if f not in X_full.columns]
if missing:
    print(f"  ⚠️ Missing O/U features: {missing}")
X = X_full[ou_avail]
feature_cols = ou_avail

y_total = df["actual_home_score"].values + df["actual_away_score"].values
mkt_total = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0).values
weights = df["season_weight"].values
n = len(X)

print(f"  {n} games × {len(feature_cols)} O/U features")

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ══════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════

LEARNERS = {
    "CatBoost": lambda: CatBoostRegressor(n_estimators=650, depth=3, learning_rate=0.10,
                                           random_seed=SEED, verbose=0),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
}

print(f"\n  {N_FOLDS}-fold walk-forward (CatBoost + LightGBM avg)...")
oof = time_series_oof(LEARNERS, X_s, y_total, N_FOLDS)
mae, ou_results = ou_analysis(oof, y_total, mkt_total)

# ══════════════════════════════════════════════════════════
# PRODUCTION TRAINING
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PRODUCTION TRAINING")
print(f"{'='*70}")

cat = CatBoostRegressor(n_estimators=650, depth=3, learning_rate=0.10, random_seed=SEED, verbose=0)
cat.fit(X_s, y_total)
print(f"  CatBoost MAE: {mean_absolute_error(y_total, cat.predict(X_s)):.3f}")

lgbm = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                      subsample=0.8, verbose=-1, random_state=SEED)
lgbm.fit(X_s, y_total)
print(f"  LightGBM MAE: {mean_absolute_error(y_total, lgbm.predict(X_s)):.3f}")

avg_pred = (cat.predict(X_s) + lgbm.predict(X_s)) / 2
avg_mae = mean_absolute_error(y_total, avg_pred)
print(f"  Ensemble MAE: {avg_mae:.3f}")

bias = float(np.mean(y_total[~np.isnan(oof)] - oof[~np.isnan(oof)]))
print(f"  Bias: {bias:+.3f}")

# SHAP
print("  Building SHAP explainer...")
explainer = shap.TreeExplainer(lgbm)

# ══════════════════════════════════════════════════════════
# BUNDLE + SAVE
# ══════════════════════════════════════════════════════════

ou5 = ou_results.get(5, {})

bundle = {
    "scaler": scaler,
    "_ensemble_models": [cat, lgbm],
    "explainer": explainer,
    "feature_cols": feature_cols,
    "ou_feature_cols": feature_cols,
    "n_train": n,
    "mae_cv": round(mae, 4),
    "model_type": "Cat_LGBM_OU_v2",
    "architecture": "CatBoost_d3_650 + LightGBM_300 (simple avg)",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "bias_correction": round(bias, 4),
    "data_filters": "no 2020/2021, no games before Nov 10",
}

local_path = "ncaa_ou_model_v2.pkl"
joblib.dump(bundle, local_path, compress=3)
size_kb = os.path.getsize(local_path) / 1024
print(f"\n  Saved: {local_path} ({size_kb:.0f} KB)")

if upload:
    print("  Uploading to Supabase as 'ncaa_ou'...")
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    resp = requests.post(
        f'{SUPABASE_URL}/rest/v1/model_store',
        headers={'apikey': KEY, 'Authorization': f'Bearer {KEY}',
                 'Content-Type': 'application/json', 'Prefer': 'resolution=merge-duplicates'},
        json={'name': 'ncaa_ou', 'data': b64,
              'metadata': {'trained_at': bundle['trained_at'], 'mae_cv': bundle['mae_cv'],
                           'n_train': n, 'model_type': bundle['model_type'],
                           'size_bytes': len(buf.getvalue())},
              'updated_at': datetime.now(timezone.utc).isoformat()},
        timeout=300)
    if resp.ok:
        print(f'  ✅ Upload successful ({len(buf.getvalue())//1024} KB)')
    else:
        print(f'  ❌ Failed: {resp.status_code} {resp.text[:300]}')
else:
    print(f"  To upload: python3 ncaa_retrain_ou.py --upload")

print(f"\n{'='*70}")
print(f"  NCAA O/U MODEL COMPLETE")
print(f"  Architecture: CatBoost + LightGBM (simple avg)")
print(f"  Features: {len(feature_cols)}")
print(f"  Games: {n}")
print(f"  Walk-forward MAE: {mae:.3f}")
print(f"  O/U 5+: {ou5.get('acc',0):.1%} ({ou5.get('n',0)} picks)")
print(f"  Bias: {bias:+.3f}")
print(f"{'='*70}")
