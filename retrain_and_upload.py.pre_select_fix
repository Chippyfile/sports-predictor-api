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
from lightgbm import LGBMRegressor
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
rows = sb_get("ncaa_historical", "actual_home_score=not.is.null&select=*&order=season.asc")
print(f"  Loaded {len(rows)} rows in {time.time()-t0:.0f}s")

# ESPN/DraftKings odds fallback
import pandas as pd, numpy as np
df = pd.DataFrame(rows)
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

# Quality filter: drop rows with <80% real data
_quality_cols = ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_row_q = _qmat.mean(axis=1)
_keep = _row_q >= 0.8
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
y_margin = df["actual_home_score"].values - df["actual_away_score"].values
y_win = (y_margin > 0).astype(int)
weights = df["season_weight"].values
n = len(X)
print(f"  {n} games × {X.shape[1]} features")

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print("\n  XGBoost...", end=" ", flush=True)
xgb = XGBRegressor(n_estimators=175, max_depth=7, learning_rate=0.10, random_state=42, tree_method="hist")
oof_xgb = time_series_oof(xgb, X_s, y_margin, n_splits=50)
xgb.fit(X_s, y_margin, sample_weight=weights)
print(f"MAE: {mean_absolute_error(y_margin, oof_xgb):.3f}")

print("  CatBoost...", end=" ", flush=True)
cat = CatBoostRegressor(n_estimators=175, depth=7, learning_rate=0.10, random_seed=42, verbose=0)
oof_cat = time_series_oof(cat, X_s, y_margin, n_splits=50)
cat.fit(X_s, y_margin, sample_weight=weights)
print(f"MAE: {mean_absolute_error(y_margin, oof_cat):.3f}")

print("  LightGBM...", end=" ", flush=True)
lgbm = LGBMRegressor(n_estimators=175, max_depth=7, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1)
oof_lgbm = time_series_oof(lgbm, X_s, y_margin, n_splits=50)
lgbm.fit(X_s, y_margin, sample_weight=weights)
print(f"MAE: {mean_absolute_error(y_margin, oof_lgbm):.3f}")

print("  MLP-128-64...", end=" ", flush=True)
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                   validation_fraction=0.1, random_state=42)
oof_mlp = time_series_oof(mlp, X_s, y_margin, n_splits=50)
mlp.fit(X_s, y_margin)
print(f"MAE: {mean_absolute_error(y_margin, oof_mlp):.3f}")

print("\n  Stacking...")
oof_stacked = np.column_stack([oof_xgb, oof_cat, oof_lgbm, oof_mlp])
meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
meta_reg.fit(oof_stacked, y_margin)
print(f"  Meta weights: {list(meta_reg.coef_.round(4))}")
stacked_preds = meta_reg.predict(oof_stacked)
mae = mean_absolute_error(y_margin, stacked_preds)
print(f"  Stacked MAE: {mae:.3f}")

meta_clf = LogisticRegression(max_iter=2000)
meta_clf.fit(oof_stacked, y_win)

bias = float(np.mean(y_margin - stacked_preds))
oof_probs = meta_clf.predict_proba(oof_stacked)[:, 1]
isotonic = IsotonicRegression(out_of_bounds="clip")
isotonic.fit(oof_probs, y_win)

explainer = shap.TreeExplainer(xgb)

# Build with ml_utils classes (THIS IS THE KEY FIX)
reg = StackedRegressor([xgb, cat, lgbm, mlp], meta_reg)
# Train classifiers (StackedClassifier needs predict_proba)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
print("  Training classifiers...")
xgb_c = XGBClassifier(n_estimators=175, max_depth=7, learning_rate=0.10, random_state=42, tree_method="hist", eval_metric="logloss")
xgb_c.fit(X_s, y_win, sample_weight=weights)
cat_c = CatBoostClassifier(n_estimators=175, depth=7, learning_rate=0.10, random_seed=42, verbose=0)
cat_c.fit(X_s, y_win, sample_weight=weights)
lgbm_c = LGBMClassifier(n_estimators=175, max_depth=7, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1)
lgbm_c.fit(X_s, y_win, sample_weight=weights)
mlp_c = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                      validation_fraction=0.1, random_state=42)
mlp_c.fit(X_s, y_win)
oof_probs_stack = np.column_stack([xgb_c.predict_proba(X_s)[:,1], cat_c.predict_proba(X_s)[:,1], lgbm_c.predict_proba(X_s)[:,1], mlp_c.predict_proba(X_s)[:,1]])
meta_clf = LogisticRegression(max_iter=2000)
meta_clf.fit(oof_probs_stack, y_win)
clf = StackedClassifier([xgb_c, cat_c, lgbm_c, mlp_c], meta_clf)

bundle = {
    "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
    "feature_cols": list(X.columns), "n_train": n,
    "mae_cv": round(mae, 3), "model_type": "StackedEnsemble_LOCAL_FULL",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "bias_correction": round(bias, 3), "isotonic": isotonic,
    "meta_weights": list(meta_reg.coef_.round(4)),
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
