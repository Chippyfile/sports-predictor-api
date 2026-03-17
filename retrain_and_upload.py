"""Retrain locally with ml_utils classes so pickle paths match Railway."""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))

import numpy as np, pandas as pd, joblib, io, base64, requests, warnings, time
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.isotonic import IsotonicRegression
# from sklearn.ensemble import RandomForestRegressor  # replaced by LGBM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap
warnings.filterwarnings("ignore")

# Import FROM ml_utils so pickle references match Railway
from ml_utils import StackedRegressor, StackedClassifier

# Import feature builder
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic

# Import local cache support
from dump_training_data import dump, load_cached, check_cache

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ['SUPABASE_ANON_KEY']

def sb_get(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"}, timeout=60)
        if not r.ok: break
        data = r.json()
        if not data: break
        all_data.extend(data)
        if len(data) < limit: break
        offset += limit
    return all_data

def time_series_oof(model, X, y, n_splits=5):
    import copy
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

print("=" * 70)
print("  RETRAIN + UPLOAD (ml_utils classes)")
print("=" * 70)

print("\n  Loading data...")
t0 = time.time()

# ── Local cache support: --refresh dumps fresh, otherwise use cache ──
USE_REFRESH = "--refresh" in sys.argv
if USE_REFRESH:
    print("  --refresh: Pulling fresh data from Supabase...")
    df = dump()  # dump_training_data.py → parquet + returns DataFrame
else:
    df = load_cached()
    if df is None:
        print("  No local cache found. Pulling from Supabase (first run)...")
        df = dump()

# Filter to rows with actual scores
df = df[df["actual_home_score"].notna()].copy()
# v24: Drop 2021 COVID season (HCA 5.91 vs 7-8 normal, empty arenas, opt-outs)
# Validated: dropping 2021 improved MAE from 8.8037 → 8.7753
_before_covid = len(df)
df = df[df["season"] != 2021].copy()
print(f"  Dropped {_before_covid - len(df)} COVID 2021 games")
print(f"  Loaded {len(df)} rows in {time.time()-t0:.0f}s")

# ESPN/DraftKings odds fallback
if "espn_spread" in df.columns:
    espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df.loc[fill, "market_spread_home"] = espn_s[fill]
    print(f"  ESPN odds fallback: {int(fill.sum())} spreads filled")
if "espn_over_under" in df.columns:
    espn_ou = pd.to_numeric(df["espn_over_under"], errors="coerce")
    mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
    fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
    df.loc[fill_ou, "market_ou_total"] = espn_ou[fill_ou]
rows = df.to_dict("records")

df = pd.DataFrame(rows)

# Quality filter: 80% threshold on key columns (v22 — produced MAE 8.830)
_quality_cols = ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_row_q = _qmat.mean(axis=1)
_keep = _row_q >= 0.8
# v23: Also require referee data (ref features are #4 SHAP but were 100% zero without this)
if "referee_1" in df.columns:
    _has_ref = df["referee_1"].notna() & (df["referee_1"] != "")
    _before = int(_keep.sum())
    _keep = _keep & _has_ref
    print(f"  Quality filter: keeping {int(_keep.sum())}/{len(df)} games (dropped {int((~_keep).sum())}, {_before - int(_keep.sum())} had no ref data)")
else:
    print(f"  Quality filter: keeping {int(_keep.sum())}/{len(df)} games (dropped {int((~_keep).sum())} below 80%)")
df = df.loc[_keep].reset_index(drop=True)
rows = df.to_dict("records")
for col in ["actual_home_score","actual_away_score","home_adj_em","away_adj_em",
            "home_adj_oe","away_adj_oe","home_adj_de","away_adj_de",
            "home_ppg","away_ppg","home_opp_ppg","away_opp_ppg",
            "home_tempo","away_tempo","home_rank","away_rank","season",
            "home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "home_record_wins" in df.columns and "home_wins" not in df.columns:
    df["home_wins"] = df["home_record_wins"]
if "away_record_wins" in df.columns and "away_wins" not in df.columns:
    df["away_wins"] = df["away_record_wins"]
if "home_record_losses" in df.columns and "home_losses" not in df.columns:
    df["home_losses"] = df["home_record_losses"]
if "away_record_losses" in df.columns and "away_losses" not in df.columns:
    df["away_losses"] = df["away_record_losses"]

current_year = datetime.utcnow().year
df["season_weight"] = df["season"].apply(
    lambda s: 1.0 if (current_year - s) <= 0 else 0.9 if (current_year - s) == 1 else
    0.75 if (current_year - s) == 2 else 0.6 if (current_year - s) == 3 else 0.5)

print("  Heuristic backfill...")
df = _ncaa_backfill_heuristic(df)

try:
    import json as _json
    with open("referee_profiles.json") as _rf:
        from sports.ncaa import ncaa_build_features
        ncaa_build_features._ref_profiles = _json.load(_rf)
    print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
except FileNotFoundError:
    print("  referee_profiles.json not found - ref features zero")
print("  Building features...")
df = df.dropna(subset=["actual_home_score", "actual_away_score"])
X = ncaa_build_features(df)

# ── v23: No features pruned — validated all 3 learners independently ──
# Every feature has >0 importance in at least one of XGB/CatBoost/MLP.
# mkt_spread_vs_model: 0 in XGB but 10.8% in CatBoost, 8.2% in MLP
# is_revenge_game: 0 in XGB/Cat but 0.3% in MLP
print(f"  {X.shape[1]} features (all validated across 3 learners)")
y_margin = df["actual_home_score"].values - df["actual_away_score"].values
y_win = (y_margin > 0).astype(int)
weights = df["season_weight"].values
n = len(X)
print(f"  {n} games × {X.shape[1]} features")

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print("\n  CatBoost solo (validated: beats 3-model stack on 2026 holdout)...", end=" ", flush=True)
cat = CatBoostRegressor(n_estimators=125, depth=4, learning_rate=0.10, random_seed=42, verbose=0)
oof_cat = time_series_oof(cat, X_s, y_margin, n_splits=50)
cat.fit(X_s, y_margin, sample_weight=weights)
mae = mean_absolute_error(y_margin, oof_cat)
print(f"MAE: {mae:.3f}")

bias = float(np.mean(y_margin - oof_cat))

explainer = shap.TreeExplainer(cat)

# Wrap in StackedRegressor for Railway compatibility (single model, passthrough meta)
from sklearn.linear_model import Ridge as _Ridge
_passthrough_meta = _Ridge(alpha=0.01, fit_intercept=False)
_passthrough_meta.fit(oof_cat.reshape(-1, 1), y_margin)
print(f"  Meta weight: {_passthrough_meta.coef_[0]:.4f} (should be ~1.0)")
reg = StackedRegressor([cat], _passthrough_meta)

# ═══ CLASSIFIER: CatBoost solo + isotonic calibration ═══
from catboost import CatBoostClassifier
print("  Training classifier (CatBoost solo, OOF)...")

cat_c = CatBoostClassifier(n_estimators=125, depth=4, learning_rate=0.10, random_seed=42, verbose=0)

# OOF probabilities
oof_probs_cat = np.zeros(n)
fold_size = n // 51  # 50-fold time series
for i in range(50):
    train_end = fold_size * (i + 2)
    val_start = train_end
    val_end = min(train_end + fold_size, n)
    if val_start >= n:
        break
    X_tr, X_val = X_s[:train_end], X_s[val_start:val_end]
    y_tr, w_tr = y_win[:train_end], weights[:train_end]
    import copy
    cc = copy.deepcopy(cat_c); cc.fit(X_tr, y_tr, sample_weight=w_tr)
    oof_probs_cat[val_start:val_end] = cc.predict_proba(X_val)[:, 1]

valid_mask = oof_probs_cat != 0
oof_probs_valid = oof_probs_cat[valid_mask]
y_win_valid = y_win[valid_mask]

# Isotonic calibration on OOF probs
isotonic = IsotonicRegression(out_of_bounds="clip")
isotonic.fit(oof_probs_valid, y_win_valid)

# Brier score
from sklearn.metrics import brier_score_loss
brier_raw = brier_score_loss(y_win_valid, oof_probs_valid)
brier_cal = brier_score_loss(y_win_valid, isotonic.predict(oof_probs_valid))
print(f"  Brier (raw): {brier_raw:.4f}  Brier (calibrated): {brier_cal:.4f}")

# Refit on full data
print("  Refitting classifier on full data...")
cat_c.fit(X_s, y_win, sample_weight=weights)

# Wrap in StackedClassifier for Railway compatibility
_passthrough_meta_clf = LogisticRegression(max_iter=2000)
_passthrough_meta_clf.fit(oof_probs_valid.reshape(-1, 1), y_win_valid)
clf = StackedClassifier([cat_c], _passthrough_meta_clf)

bundle = {
    "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
    "feature_cols": list(X.columns), "n_train": n,
    "mae_cv": round(mae, 3), "model_type": "StackedEnsemble_LOCAL_FULL",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "bias_correction": round(bias, 3), "isotonic": isotonic,
    "meta_weights": list(_passthrough_meta.coef_.round(4)),
}

print(f"\n  Verifying class paths...")
print(f"    reg: {type(bundle['reg']).__module__}.{type(bundle['reg']).__name__}")
print(f"    clf: {type(bundle['clf']).__module__}.{type(bundle['clf']).__name__}")

print("  Compressing...")
buf = io.BytesIO()
joblib.dump(bundle, buf, compress=3)
compressed = buf.getvalue()
print(f"  Size: {len(compressed)/1024:.0f} KB")

b64 = base64.b64encode(compressed).decode('ascii')
print("  Uploading...")
resp = requests.post(
    f'{SUPABASE_URL}/rest/v1/model_store',
    headers={'apikey': KEY, 'Authorization': f'Bearer {KEY}',
             'Content-Type': 'application/json', 'Prefer': 'resolution=merge-duplicates'},
    json={'name': 'ncaa', 'data': b64,
          'metadata': {'trained_at': bundle['trained_at'], 'mae_cv': bundle['mae_cv'],
                       'n_train': n, 'model_type': 'StackedEnsemble_LOCAL_FULL',
                       'size_bytes': len(compressed)},
          'updated_at': datetime.now(timezone.utc).isoformat()},
    timeout=300)
if resp.ok:
    print(f'  ✅ Upload successful ({len(compressed)/1024:.0f} KB)')
else:
    print(f'  ❌ Failed: {resp.status_code} {resp.text[:300]}')

joblib.dump(bundle, "ncaa_model_local.pkl", compress=3)
print(f"\n  Done. n_train={n}, MAE={mae:.3f}")
