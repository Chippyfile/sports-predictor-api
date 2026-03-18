"""
NBA Local Retrain + Upload
Mirrors retrain_and_upload.py (NCAA) for NBA.

Usage:
  python retrain_nba.py              # Use cached parquet
  python retrain_nba.py --refresh    # Pull fresh from Supabase

Requirements:
  - nba_elo.py in same directory (or sports-predictor-api root)
  - SUPABASE_ANON_KEY env var set
  - catboost, xgboost, shap installed
"""

import sys, os, copy, time, io, base64, json, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))

import numpy as np, pandas as pd, joblib, requests
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss
from catboost import CatBoostRegressor, CatBoostClassifier
import shap
warnings.filterwarnings("ignore")

from ml_utils import StackedRegressor, StackedClassifier

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


def time_series_oof(model, X, y, n_splits=50, weights=None):
    """50-fold expanding-window time series OOF predictions."""
    n = len(X)
    oof = np.zeros(n)
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(train_end + fold_size, n)
        if val_start >= n: break
        m = copy.deepcopy(model)
        X_tr, y_tr = X[:train_end], y[:train_end]
        if weights is not None:
            m.fit(X_tr, y_tr, sample_weight=weights[:train_end])
        else:
            m.fit(X_tr, y_tr)
        oof[val_start:val_end] = m.predict(X[val_start:val_end])
    return oof


def time_series_oof_proba(model, X, y, n_splits=50, weights=None):
    """50-fold expanding-window time series OOF probabilities."""
    n = len(X)
    oof = np.zeros(n)
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(train_end + fold_size, n)
        if val_start >= n: break
        m = copy.deepcopy(model)
        X_tr, y_tr = X[:train_end], y[:train_end]
        if weights is not None:
            m.fit(X_tr, y_tr, sample_weight=weights[:train_end])
        else:
            m.fit(X_tr, y_tr)
        oof[val_start:val_end] = m.predict_proba(X[val_start:val_end])[:, 1]
    return oof


def dump_nba_data():
    """Pull NBA data from Supabase and save to parquet."""
    print("  Pulling NBA data from Supabase...")
    
    # Historical
    hist_rows = sb_get("nba_historical",
                       "is_outlier_season=eq.false&actual_home_score=not.is.null&select=*&order=game_date.asc")
    hist_df = pd.DataFrame(hist_rows) if hist_rows else pd.DataFrame()
    print(f"    nba_historical: {len(hist_df)} rows")
    
    # Current season predictions with results
    pred_rows = sb_get("nba_predictions",
                       "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    pred_df = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
    print(f"    nba_predictions (completed): {len(pred_df)} rows")
    
    # Add season to predictions if missing
    if len(pred_df) > 0 and "season" not in pred_df.columns:
        pred_df["season"] = pred_df["game_date"].apply(
            lambda d: int(d[:4]) + 1 if int(d[5:7]) >= 10 else int(d[:4])
        )
    
    # Combine
    if len(hist_df) > 0 and len(pred_df) > 0:
        combined = pd.concat([hist_df, pred_df], ignore_index=True)
    elif len(hist_df) > 0:
        combined = hist_df
    else:
        combined = pred_df
    
    combined = combined.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")
    combined = combined.sort_values("game_date").reset_index(drop=True)
    
    combined.to_parquet("nba_training_data.parquet", index=False)
    print(f"    Saved {len(combined)} rows to nba_training_data.parquet")
    return combined


def load_cached():
    """Load cached parquet if available."""
    path = "nba_training_data.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  Loaded {len(df)} rows from {path}")
        return df
    return None


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("  NBA RETRAIN + UPLOAD (v22 features, CatBoost solo, NCAA parity)")
print("=" * 70)

# ── Load data ──
t0 = time.time()
USE_REFRESH = "--refresh" in sys.argv

if USE_REFRESH:
    df = dump_nba_data()
else:
    df = load_cached()
    if df is None:
        print("  No cache found — pulling from Supabase...")
        df = dump_nba_data()

# Filter to completed games
df = df[df["actual_home_score"].notna()].copy()

# Ensure numeric
for col in df.columns:
    if col not in ['home_team', 'away_team', 'game_date', 'id', 'season',
                   'is_outlier_season', 'home_win', 'result_entered', 'game_id',
                   'home_team_name', 'away_team_name']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"  Loaded {len(df)} completed games in {time.time()-t0:.0f}s")

# ── Season weights ──
current_year = datetime.utcnow().year
if "season" in df.columns:
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(current_year)
    df["season_weight"] = df["season"].apply(
        lambda s: 1.0 if (current_year - s) <= 0 else
        1.0 if (current_year - s) == 1 else
        0.9 if (current_year - s) == 2 else
        0.8 if (current_year - s) == 3 else
        0.7 if (current_year - s) == 4 else 0.6
    )
else:
    df["season_weight"] = 1.0

# ── Compute Elo ratings ──
print("\n  Computing Elo ratings...")
try:
    from nba_elo import compute_all_elo, merge_elo_into_df
    elo_df, current_ratings = compute_all_elo(df)
    df = merge_elo_into_df(df, elo_df)
    
    # Save ratings for live prediction
    with open("nba_elo_ratings.json", "w") as f:
        json.dump({"ratings": current_ratings,
                    "n_games": len(elo_df),
                    "last_updated": str(pd.Timestamp.now())}, f, indent=2)
    print(f"  Elo computed: {len(elo_df)} games, saved ratings for {len(current_ratings)} teams")
except Exception as e:
    print(f"  WARNING: Elo computation failed ({e}), using defaults")
    df["home_elo"] = 1500
    df["away_elo"] = 1500
    df["elo_diff"] = 0.0

# ── Heuristic backfill ──
# Import and run the backfill to populate pred_home_score, win_pct_home, etc.
print("\n  Running heuristic backfill...")
try:
    from sports.nba import _nba_backfill_heuristic
    df = _nba_backfill_heuristic(df)
except Exception as e:
    print(f"  WARNING: Backfill failed ({e}), computing minimal features")
    if "pred_home_score" not in df.columns:
        df["pred_home_score"] = df.get("home_ppg", 110)
    if "pred_away_score" not in df.columns:
        df["pred_away_score"] = df.get("away_ppg", 110)
    if "win_pct_home" not in df.columns:
        df["win_pct_home"] = 0.5
    if "model_ml_home" not in df.columns:
        df["model_ml_home"] = 0

# ── Quality filter ──
# Run v2 enrichment if available (adds momentum, distribution, matchup features)
try:
    from enrich_nba_v2 import enrich as enrich_v2
    print("\n  Running v2 enrichment (momentum, distributions, matchups)...")
    t_enrich = time.time()
    df = enrich_v2(df)
    print(f"  v2 enrichment took {time.time()-t_enrich:.1f}s")
except ImportError:
    print("  WARNING: enrich_nba_v2.py not found — using basic enrichment only")

_quality_cols = ["home_ppg", "away_ppg", "home_fgpct", "away_fgpct",
                 "market_spread_home", "market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
if _qcols:
    _qmat = pd.DataFrame({
        c: df[c].notna() & (df[c] != 0 if c in ["market_spread_home", "market_ou_total"] else True)
        for c in _qcols
    })
    _row_q = _qmat.mean(axis=1)
    _keep = _row_q >= 0.60  # Slightly more permissive than NCAA's 0.80
    print(f"  Quality filter: keeping {int(_keep.sum())}/{len(df)} games")
    df = df.loc[_keep].reset_index(drop=True)

# ── Build features ──
print("\n  Building features...")
try:
    from nba_build_features_v22 import nba_build_features
    print("  Using v22 expanded feature builder (120+ features)")
except ImportError:
    try:
        from nba_build_features_v21 import nba_build_features
        print("  Using v21 feature builder (50 features)")
    except ImportError:
        from sports.nba import nba_build_features
        print("  Using current feature builder (v20, 27 features)")

X = nba_build_features(df)
y_margin = df["actual_home_score"].values - df["actual_away_score"].values
y_win = (y_margin > 0).astype(int)
weights = df["season_weight"].values if "season_weight" in df.columns else None
n = len(X)
print(f"  {n} games × {X.shape[1]} features")
print(f"  Features: {list(X.columns)}")

# ── Scale ──
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ══════════════════════════════════════════════════════════════
# REGRESSOR: CatBoost solo (validated as optimal for NCAA)
# ══════════════════════════════════════════════════════════════
print(f"\n  Training CatBoost regressor (50-fold TS-CV)...", end=" ", flush=True)

# Depth sweep
best_mae, best_depth = 999, 4
for depth in [3, 4, 5, 6]:
    cat_test = CatBoostRegressor(n_estimators=125, depth=depth, learning_rate=0.10,
                                  random_seed=42, verbose=0)
    oof_test = time_series_oof(cat_test, X_s, y_margin, n_splits=50, weights=weights)
    mae_test = mean_absolute_error(y_margin, oof_test)
    print(f"\n    depth={depth}: MAE={mae_test:.3f}", end="")
    if mae_test < best_mae:
        best_mae = mae_test
        best_depth = depth

print(f"\n  Best depth: {best_depth} (MAE={best_mae:.3f})")

cat = CatBoostRegressor(n_estimators=125, depth=best_depth, learning_rate=0.10,
                         random_seed=42, verbose=0)
oof_cat = time_series_oof(cat, X_s, y_margin, n_splits=50, weights=weights)
cat.fit(X_s, y_margin, sample_weight=weights)
mae = mean_absolute_error(y_margin, oof_cat)
print(f"  Final regressor MAE: {mae:.3f}")

bias = float(np.mean(y_margin - oof_cat))
print(f"  Bias correction: {bias:+.3f} pts")

# Wrap for Railway compatibility
_passthrough_meta = Ridge(alpha=0.01, fit_intercept=False)
_passthrough_meta.fit(oof_cat.reshape(-1, 1), y_margin)
print(f"  Meta weight: {_passthrough_meta.coef_[0]:.4f} (should be ~1.0)")
reg = StackedRegressor([cat], _passthrough_meta)

# SHAP analysis
print("\n  Computing SHAP values...")
explainer = shap.TreeExplainer(cat)
shap_vals = explainer.shap_values(X_s[:min(2000, n)])
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
shap_importance = sorted(zip(X.columns, mean_abs_shap), key=lambda x: -x[1])
print("\n  SHAP Top 15:")
total_shap = sum(v for _, v in shap_importance)
for feat, val in shap_importance[:15]:
    print(f"    {feat:30s}  {val:.3f}  ({val/total_shap*100:.1f}%)")

# Check for zero-importance features
zero_feats = [f for f, v in shap_importance if v < 0.001]
if zero_feats:
    print(f"\n  WARNING: {len(zero_feats)} features with near-zero SHAP:")
    for f in zero_feats:
        print(f"    {f}")

# ══════════════════════════════════════════════════════════════
# CLASSIFIER: CatBoost solo + isotonic calibration
# ══════════════════════════════════════════════════════════════
print("\n  Training CatBoost classifier (50-fold TS-CV)...")
cat_c = CatBoostClassifier(n_estimators=125, depth=best_depth, learning_rate=0.10,
                            random_seed=42, verbose=0)
oof_probs = time_series_oof_proba(cat_c, X_s, y_win, n_splits=50, weights=weights)

# Only calibrate on non-zero OOF values
valid_mask = oof_probs != 0
oof_probs_valid = oof_probs[valid_mask]
y_win_valid = y_win[valid_mask]

# Isotonic calibration
isotonic = IsotonicRegression(out_of_bounds="clip")
isotonic.fit(oof_probs_valid, y_win_valid)

brier_raw = brier_score_loss(y_win_valid, oof_probs_valid)
brier_cal = brier_score_loss(y_win_valid, isotonic.predict(oof_probs_valid))
print(f"  Brier (raw): {brier_raw:.4f}  Brier (calibrated): {brier_cal:.4f}")

# Refit on full data
cat_c.fit(X_s, y_win, sample_weight=weights)

# Wrap for Railway compatibility
_meta_clf = LogisticRegression(max_iter=2000)
_meta_clf.fit(oof_probs_valid.reshape(-1, 1), y_win_valid)
clf = StackedClassifier([cat_c], _meta_clf)

# ══════════════════════════════════════════════════════════════
# HOLDOUT VALIDATION (last season)
# ══════════════════════════════════════════════════════════════
if "season" in df.columns:
    latest_season = df["season"].max()
    holdout_mask = df["season"] == latest_season
    if holdout_mask.sum() >= 50:
        X_holdout = X_s[holdout_mask]
        y_holdout_margin = y_margin[holdout_mask]
        y_holdout_win = y_win[holdout_mask]
        
        # Train on all prior seasons
        train_mask = ~holdout_mask
        cat_holdout = CatBoostRegressor(n_estimators=125, depth=best_depth,
                                        learning_rate=0.10, random_seed=42, verbose=0)
        cat_holdout.fit(X_s[train_mask], y_margin[train_mask],
                       sample_weight=weights[train_mask] if weights is not None else None)
        holdout_preds = cat_holdout.predict(X_holdout)
        holdout_mae = mean_absolute_error(y_holdout_margin, holdout_preds)
        
        # ATS check
        mkt_spread = X.loc[holdout_mask, "market_spread"].values
        has_mkt = X.loc[holdout_mask, "has_market"].values
        mkt_games = has_mkt == 1
        if mkt_games.sum() > 0:
            model_covers = (holdout_preds[mkt_games] > mkt_spread[mkt_games]) == (y_holdout_margin[mkt_games] > mkt_spread[mkt_games])
            ats_acc = model_covers.mean()
        else:
            ats_acc = None
        
        print(f"\n  HOLDOUT (season {int(latest_season)}, {holdout_mask.sum()} games):")
        print(f"    MAE: {holdout_mae:.3f}")
        if ats_acc is not None:
            print(f"    ATS: {ats_acc:.1%} ({mkt_games.sum()} games with market)")

# ══════════════════════════════════════════════════════════════
# SAVE + UPLOAD
# ══════════════════════════════════════════════════════════════
bundle = {
    "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
    "feature_cols": list(X.columns), "n_train": n,
    "mae_cv": round(mae, 3), "model_type": "StackedEnsemble_NBA_v22_LOCAL",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "bias_correction": round(bias, 3), "isotonic": isotonic,
    "meta_weights": list(_passthrough_meta.coef_.round(4)),
}

print(f"\n  Verifying class paths...")
print(f"    reg: {type(bundle['reg']).__module__}.{type(bundle['reg']).__name__}")
print(f"    clf: {type(bundle['clf']).__module__}.{type(bundle['clf']).__name__}")

# Save locally
joblib.dump(bundle, "nba_model_local.pkl", compress=3)

# Compress for upload
print("  Compressing...")
buf = io.BytesIO()
joblib.dump(bundle, buf, compress=3)
compressed = buf.getvalue()
print(f"  Size: {len(compressed)/1024:.0f} KB")

# Upload to Supabase
b64 = base64.b64encode(compressed).decode('ascii')
print("  Uploading to Supabase model_store...")
resp = requests.post(
    f'{SUPABASE_URL}/rest/v1/model_store',
    headers={'apikey': KEY, 'Authorization': f'Bearer {KEY}',
             'Content-Type': 'application/json', 'Prefer': 'resolution=merge-duplicates'},
    json={'name': 'nba', 'data': b64,
          'metadata': {'trained_at': bundle['trained_at'], 'mae_cv': bundle['mae_cv'],
                       'n_train': n, 'model_type': 'StackedEnsemble_NBA_LOCAL',
                       'n_features': len(bundle['feature_cols']),
                       'features': bundle['feature_cols'],
                       'size_bytes': len(compressed)},
          'updated_at': datetime.now(timezone.utc).isoformat()},
    timeout=300)

if resp.ok:
    print(f'  ✅ Upload successful ({len(compressed)/1024:.0f} KB)')
else:
    print(f'  ❌ Failed: {resp.status_code} {resp.text[:300]}')

print(f"\n{'='*70}")
print(f"  DONE: n_train={n}, features={X.shape[1]}, MAE={mae:.3f}, "
      f"Brier(raw)={brier_raw:.4f}, Brier(cal)={brier_cal:.4f}")
print(f"{'='*70}")
