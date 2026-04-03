#!/usr/bin/env python3
"""
ncaa_retrain_ou.py — NCAA O/U Model v3 (Rolling HCA + σ=6.5 + cascade spread + weights)
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
SEED = 42; N_FOLDS = 30; SIGMA = 6.5

OU_FEATURES = [
    'mkt_total', 'tempo_avg', 'ref_ou_bias', 'venue_advantage',
    'hca_pts', 'starter_balance_diff', 'lineup_changes_diff',
    'roll_ats_margin_gated', 'to_margin_diff', 'matchup_efg',
    'roll_top3_share_diff', 'assist_rate_diff',
    'mkt_spread_vs_model', 'matchup_orb',
    'starter_experience_diff', 'opp_fta_rate_diff',
    'lineup_stability_diff', 'ref_foul_rate',
    'info_gain_diff', 'eff_vol_diff',
    'rolling_hca',
]

LEARNERS = {
    "CatBoost": lambda: CatBoostRegressor(n_estimators=650, depth=3, learning_rate=0.10, random_seed=SEED, verbose=0),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED),
}

def compute_rolling_hca_col(df, window=20):
    df = df.sort_values("game_date").reset_index(drop=True)
    margins = df["actual_home_score"].values - df["actual_away_score"].values
    h_ids = df["home_team_id"].astype(str).values
    a_ids = df["away_team_id"].astype(str).values
    from collections import defaultdict
    home_margins = defaultdict(list); away_margins = defaultdict(list)
    rolling_hca = np.full(len(df), 6.6)
    for i in range(len(df)):
        hid, aid = h_ids[i], a_ids[i]
        h_hist = home_margins[hid][-window:]; a_hist = away_margins[hid][-window:]
        if len(h_hist) >= 5 and len(a_hist) >= 5:
            rolling_hca[i] = float(np.mean(h_hist) - np.mean(a_hist)) / 2
        home_margins[hid].append(margins[i]); away_margins[aid].append(-margins[i])
    df["rolling_hca"] = rolling_hca
    print(f"  Rolling HCA: mean={df['rolling_hca'].mean():+.2f}, coverage={(rolling_hca!=6.6).sum()/len(df)*100:.1f}%")
    return df

def walk_forward(models_dict, X_s, y, n_splits, weights=None):
    n = len(X_s); fold_size = n // (n_splits + 1); min_train = fold_size * 2
    oof = np.full(n, np.nan)
    for fold in range(n_splits):
        ts = min_train + fold * fold_size; te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in models_dict.items():
            m = builder()
            if weights is not None:
                try: m.fit(X_s[:ts], y[:ts], sample_weight=weights[:ts])
                except TypeError: m.fit(X_s[:ts], y[:ts])
            else: m.fit(X_s[:ts], y[:ts])
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
        n_over = (edge[decided] > 0).sum(); n_under = (edge[decided] < 0).sum()
        ov_acc = co.sum() / max(n_over, 1); un_acc = cu.sum() / max(n_under, 1)
        tag = "YES" if acc > 0.524 else "no"
        print(f"  {t:>5d}+ {n_games:>7d} {acc:>5.1%} {roi:>+6.1f}%  {ov_acc:.0%}({n_over}) {un_acc:.0%}({n_under})  {tag}")
        results[t] = {"acc": acc, "n": n_games, "roi": roi}
    return mae, results

# ══════════════════════════════════════════════════════════
print("=" * 70)
print("  NCAA O/U MODEL v3 — CatBoost + LightGBM + Rolling HCA")
print("=" * 70)
upload = "--upload" in sys.argv
if "--refresh" in sys.argv:
    df = dump()
else:
    df = load_cached()
    if df is None or len(df) == 0:
        df = dump()
df = df[df["actual_home_score"].notna()].copy()

# Force numeric
for col in ["market_spread_home","market_ou_total","espn_spread","espn_over_under",
            "dk_spread_close","dk_total_close","odds_api_spread_close","odds_api_total_close",
            "actual_home_score","actual_away_score","home_adj_em","away_adj_em",
            "home_ppg","away_ppg","home_tempo","away_tempo","season"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

df["season"] = df["season"].fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()
df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
m = df["game_date_dt"].dt.month
df = df[((m>=11)|(m<=4)) & ~((m==11)&(df["game_date_dt"].dt.day<10))].copy()
df = df.drop(columns=["game_date_dt"], errors="ignore")
print(f"  {len(df)} games after filters")

# Cascade spread/total backfill
for sc, tc in [("espn_spread","espn_over_under"),("dk_spread_close","dk_total_close"),("odds_api_spread_close","odds_api_total_close")]:
    for col, tgt in [(sc,"market_spread_home"),(tc,"market_ou_total")]:
        if col in df.columns:
            src = pd.to_numeric(df[col], errors="coerce")
            cur = pd.to_numeric(df.get(tgt, pd.Series(dtype=float)), errors="coerce")
            fill = (cur.isna()|(cur==0)) & src.notna() & (src!=0)
            if fill.sum() > 0:
                df.loc[fill, tgt] = src[fill]
                print(f"  {tgt} backfill from {col}: {fill.sum():,}")

# Quality filter
_qcols = [c for c in ["home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c]!=0 if c in ["market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.75
if "referee_1" in df.columns:
    _keep = _keep & df["referee_1"].notna() & (df["referee_1"]!="") & (df["referee_1"]!="None")
df = df.loc[_keep].reset_index(drop=True)

for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

current_year = 2026
seasons = pd.to_numeric(df["season"], errors="coerce").values
weights = np.array([{2026:1.0,2025:0.9,2024:0.75,2023:0.6}.get(s,0.5) for s in seasons])

print("  Heuristic backfill + features...")
df = _ncaa_backfill_heuristic(df)
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)
try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except: pass
try:
    import json
    with open("referee_profiles.json") as f: ncaa_build_features._ref_profiles = json.load(f)
except: pass
df = df.dropna(subset=["actual_home_score","actual_away_score"])
df = compute_rolling_hca_col(df)

X_full = ncaa_build_features(df)
X_full['rolling_hca'] = df['rolling_hca'].values
ou_avail = [f for f in OU_FEATURES if f in X_full.columns]
missing = [f for f in OU_FEATURES if f not in X_full.columns]
if missing: print(f"  ⚠️ Missing: {missing}")
X = X_full[ou_avail]; feature_cols = ou_avail

y_total = df["actual_home_score"].values + df["actual_away_score"].values
mkt_total = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce").fillna(0).values
n = len(X)
print(f"  {n} games × {len(feature_cols)} features")

scaler = StandardScaler(); X_s = scaler.fit_transform(X)
print(f"\n  {N_FOLDS}-fold walk-forward (weighted)...")
oof = walk_forward(LEARNERS, X_s, y_total, N_FOLDS, weights)
mae, ou_results = ou_analysis(oof, y_total, mkt_total)

# Production training
print(f"\n  Training final models...")
cat = CatBoostRegressor(n_estimators=650, depth=3, learning_rate=0.10, random_seed=SEED, verbose=0)
cat.fit(X_s, y_total, sample_weight=weights)
lgbm = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
lgbm.fit(X_s, y_total, sample_weight=weights)
avg_mae = mean_absolute_error(y_total, (cat.predict(X_s)+lgbm.predict(X_s))/2)
bias = float(np.mean(y_total[~np.isnan(oof)] - oof[~np.isnan(oof)]))
print(f"  Ensemble MAE: {avg_mae:.3f}, Bias: {bias:+.3f}")

explainer = shap.TreeExplainer(lgbm)
ou5 = ou_results.get(5, {})
bundle = {
    "scaler": scaler, "_ensemble_models": [cat, lgbm], "explainer": explainer,
    "feature_cols": feature_cols, "ou_feature_cols": feature_cols,
    "n_train": n, "mae_cv": round(mae, 4), "model_type": "Cat_LGBM_OU_v3",
    "architecture": "CatBoost_d3_650 + LightGBM_300 (weighted avg)",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "bias_correction": round(bias, 4), "sigma": SIGMA,
}
joblib.dump(bundle, "ncaa_ou_model_v3.pkl", compress=3)

if upload:
    print("  Uploading to Supabase as 'ncaa_ou'...")
    buf = io.BytesIO(); joblib.dump(bundle, buf, compress=3)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    resp = requests.post(f'{SUPABASE_URL}/rest/v1/model_store',
        headers={'apikey': KEY, 'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json', 'Prefer': 'resolution=merge-duplicates'},
        json={'name': 'ncaa_ou', 'data': b64, 'metadata': {'trained_at': bundle['trained_at'], 'mae_cv': bundle['mae_cv'], 'n_train': n, 'model_type': bundle['model_type'], 'size_bytes': len(buf.getvalue())}, 'updated_at': datetime.now(timezone.utc).isoformat()},
        timeout=300)
    print(f"  {'✅ Upload successful' if resp.ok else '❌ Failed: ' + resp.text[:200]}")

print(f"\n  Done. MAE={mae:.3f}, O/U@5+: {ou5.get('acc',0):.1%} ({ou5.get('n',0)} picks), Bias={bias:+.3f}")
