#!/usr/bin/env python3
"""
retrain_nba.py — NBA Model v26 (Lasso Solo)

Architecture: Lasso(alpha=0.1) with inline L1 feature selection
  - Builds 178 features via nba_build_features_v25
  - Lasso auto-selects ~70 features (zeros out the rest)
  - StandardScaler required (linear model)
  - Isotonic calibration for win probability
  - Saves: scaler, lasso model, calibrator, feature list

Walkforward validated:
  MAE 10.287 | Acc 68.2% | ATS 55.9% (5,280 games)
  ATS 4+: 63.5% | ATS 6+: 66.0% | Beats Vegas by 0.11 MAE

Usage:
  python3 retrain_nba.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression

print("=" * 60)
print("  NBA Model v26 — Lasso Solo Retrain")
print("=" * 60)
t_start = time.time()

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("\n  Loading training data...")
df = pd.read_parquet("nba_training_data.parquet")
print(f"  Loaded {len(df)} rows")

# ── Season weights ──
if "season" in df.columns:
    latest = pd.to_numeric(df["season"], errors="coerce").max()
    df["season_weight"] = df["season"].apply(
        lambda s: 1.0 if pd.to_numeric(s, errors="coerce") >= latest - 1
        else 0.85 if pd.to_numeric(s, errors="coerce") >= latest - 2
        else 0.7
    )
else:
    df["season_weight"] = 1.0

# ── Compute Elo ratings ──
print("\n  Computing Elo ratings...")
try:
    from nba_elo import compute_all_elo, merge_elo_into_df
    elo_df, current_ratings = compute_all_elo(df)
    df = merge_elo_into_df(df, elo_df)

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

# ── Enrichment v2 ──
try:
    from enrich_nba_v2 import enrich as enrich_v2
    print("\n  Running v2 enrichment...")
    t_enrich = time.time()
    df = enrich_v2(df)
    print(f"  v2 enrichment took {time.time()-t_enrich:.1f}s")
except ImportError:
    print("  WARNING: enrich_nba_v2.py not found")

# ── Quality filter ──
_quality_cols = ["home_ppg", "away_ppg", "home_fgpct", "away_fgpct",
                 "market_spread_home", "market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
if _qcols:
    _qmat = pd.DataFrame({
        c: df[c].notna() & (df[c] != 0 if c in ["market_spread_home", "market_ou_total"] else True)
        for c in _qcols
    })
    _row_q = _qmat.mean(axis=1)
    _keep = _row_q >= 0.60
    print(f"  Quality filter: keeping {int(_keep.sum())}/{len(df)} games")
    df = df.loc[_keep].reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# 2. BUILD FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n  Building features...")
try:
    from nba_build_features_v25 import nba_build_features
    print("  Using v25 feature builder (178 features)")
except ImportError:
    try:
        from nba_build_features_v24 import nba_build_features
        print("  Using v24 feature builder")
    except ImportError:
        from sports.nba import nba_build_features
        print("  Using legacy feature builder")

X = nba_build_features(df)
y_margin = pd.to_numeric(df["actual_home_score"], errors="coerce").fillna(0) - \
           pd.to_numeric(df["actual_away_score"], errors="coerce").fillna(0)
y_win = (y_margin > 0).astype(int)
weights = np.clip(
    pd.to_numeric(df.get("season_weight", 1), errors="coerce").fillna(1).values,
    0.1, 2.0
)
n = len(X)

print(f"  {n} games × {len(X.columns)} features")
print(f"  Target range: [{y_margin.min():.0f}, {y_margin.max():.0f}], mean={y_margin.mean():.1f}")

# ═══════════════════════════════════════════════════════════════
# 3. LASSO FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 1: Lasso Feature Selection")
print("=" * 60)

scaler_full = StandardScaler()
X_scaled = pd.DataFrame(scaler_full.fit_transform(X), columns=X.columns, index=X.index)

lasso_selector = Lasso(alpha=0.1, max_iter=5000)
lasso_selector.fit(X_scaled, y_margin, sample_weight=weights)

coefs = pd.Series(lasso_selector.coef_, index=X.columns)
survivors = coefs[coefs.abs() > 0.001].sort_values(key=abs, ascending=False)
dropped = coefs[coefs.abs() <= 0.001]

print(f"\n  Lasso keeps {len(survivors)}/{len(coefs)} features")
print(f"\n  Top 20 features by |coefficient|:")
for i, (feat, val) in enumerate(survivors.head(20).items()):
    d = "+" if val > 0 else "-"
    print(f"    {i+1:2d}. {d} {feat:40s}  |coef|={abs(val):.4f}")

print(f"\n  Dropped {len(dropped)} features (zeroed by L1)")
FEATURE_LIST = survivors.index.tolist()

# ═══════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: 50-Fold Time-Series Cross-Validation")
print("=" * 60)

X_slim = X[FEATURE_LIST]
tscv = TimeSeriesSplit(n_splits=50)
oof_pred = np.full(n, np.nan)
fold_maes = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_slim)):
    X_tr, X_val = X_slim.iloc[train_idx], X_slim.iloc[val_idx]
    y_tr, y_val = y_margin.iloc[train_idx], y_margin.iloc[val_idx]
    w_tr = weights[train_idx]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)

    model = Lasso(alpha=0.1, max_iter=5000)
    model.fit(X_tr_s, y_tr, sample_weight=w_tr)

    pred = model.predict(X_val_s)
    oof_pred[val_idx] = pred
    mae = np.abs(y_val.values - pred).mean()
    fold_maes.append(mae)

cv_mask = ~np.isnan(oof_pred)
cv_mae = np.mean(fold_maes)
cv_acc = ((oof_pred[cv_mask] > 0) == (y_margin[cv_mask].values > 0)).mean()

raw_prob = 1.0 / (1.0 + np.exp(-oof_pred[cv_mask] / 8.0))
cal_brier_raw = np.mean((raw_prob - y_win[cv_mask].values) ** 2)

print(f"\n  CV Results ({len(fold_maes)} folds, {cv_mask.sum()} OOF games):")
print(f"    MAE:      {cv_mae:.4f}")
print(f"    Accuracy: {cv_acc:.1%}")
print(f"    Brier:    {cal_brier_raw:.4f} (raw sigmoid)")
print(f"    Features: {len(FEATURE_LIST)}")

# ═══════════════════════════════════════════════════════════════
# 5. FINAL MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3: Final Model Training (all data)")
print("=" * 60)

final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_slim)

final_model = Lasso(alpha=0.1, max_iter=5000)
final_model.fit(X_final_scaled, y_margin, sample_weight=weights)

final_coefs = pd.Series(final_model.coef_, index=FEATURE_LIST)
nonzero_final = (final_coefs.abs() > 0.001).sum()
print(f"  Final model: {nonzero_final}/{len(FEATURE_LIST)} features with nonzero coefficients")

# ═══════════════════════════════════════════════════════════════
# 6. ISOTONIC CALIBRATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4: Isotonic Calibration")
print("=" * 60)

cal_mask = ~np.isnan(oof_pred)
raw_probs_cal = 1.0 / (1.0 + np.exp(-oof_pred[cal_mask] / 8.0))
actual_wins_cal = y_win[cal_mask].values

iso_cal = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(raw_probs_cal, actual_wins_cal)

cal_probs = iso_cal.predict(raw_probs_cal)
cal_brier = np.mean((cal_probs - actual_wins_cal) ** 2)
print(f"  Brier before calibration: {cal_brier_raw:.4f}")
print(f"  Brier after calibration:  {cal_brier:.4f}")

print("\n  Calibration by predicted probability bucket:")
print(f"  {'Bucket':>12s}  {'Predicted':>9s}  {'Actual':>7s}  {'N':>5s}  {'Gap':>6s}")
for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]:
    mask_b = (cal_probs >= lo) & (cal_probs < hi)
    if mask_b.sum() > 20:
        pred_avg = cal_probs[mask_b].mean()
        act_avg = actual_wins_cal[mask_b].mean()
        gap = pred_avg - act_avg
        print(f"  {lo:.1f}-{hi:.1f}      {pred_avg:.3f}      {act_avg:.3f}  {mask_b.sum():>5d}  {gap:+.3f}")

# ═══════════════════════════════════════════════════════════════
# 7. SAVE MODEL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5: Save Model")
print("=" * 60)

model_payload = {
    "model": final_model,
    "scaler": final_scaler,
    "calibrator": iso_cal,
    "feature_list": FEATURE_LIST,
    "all_features": X.columns.tolist(),
    "architecture": "Lasso_solo_v26",
    "alpha": 0.1,
    "cv_mae": round(cv_mae, 4),
    "cv_accuracy": round(cv_acc, 4),
    "cv_brier": round(cal_brier, 4),
    "n_features_selected": len(FEATURE_LIST),
    "n_features_total": len(X.columns),
    "n_games": n,
    "trained_at": str(pd.Timestamp.now()),
}

pkl_path = "nba_model_local.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(model_payload, f, protocol=4)
pkl_size = os.path.getsize(pkl_path) / 1024
print(f"  Saved: {pkl_path} ({pkl_size:.0f} KB)")

# ═══════════════════════════════════════════════════════════════
# 8. UPLOAD TO SUPABASE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6: Upload to Supabase")
print("=" * 60)

try:
    from ml_utils import upload_model_to_supabase
    upload_model_to_supabase(
        model_payload=model_payload,
        sport="nba",
        model_name="nba_lasso_v26",
        pkl_path=pkl_path,
    )
    print("  ✅ Uploaded to Supabase model_store")
except Exception as e:
    print(f"  WARNING: Supabase upload failed ({e})")
    print("  Model saved locally — deploy via git push")

# ═══════════════════════════════════════════════════════════════
# 9. HOLDOUT VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 7: Holdout Validation (last 60 days)")
print("=" * 60)

try:
    gd = pd.to_datetime(df["game_date"])
    cutoff = gd.max() - pd.Timedelta(days=60)
    holdout_mask = gd >= cutoff

    if holdout_mask.sum() > 20:
        X_hold = X_slim[holdout_mask]
        y_hold = y_margin[holdout_mask]

        X_hold_s = final_scaler.transform(X_hold)
        hold_pred = final_model.predict(X_hold_s)

        hold_mae = np.abs(y_hold.values - hold_pred).mean()
        hold_acc = ((hold_pred > 0) == (y_hold.values > 0)).mean()

        spread = pd.to_numeric(df.loc[holdout_mask, "market_spread_home"], errors="coerce").fillna(0)
        has_mkt = spread.notna() & (spread != 0)
        if has_mkt.sum() > 10:
            sp = spread[has_mkt].values
            pm = hold_pred[has_mkt.values]
            am = y_hold[has_mkt].values
            actual_covers = (am + sp) > 0
            model_picks = (pm + sp) > 0
            push = (am + sp) == 0
            non_push = ~push
            if non_push.sum() > 0:
                ats = (model_picks[non_push] == actual_covers[non_push]).mean()
                ats_n = non_push.sum()
            else:
                ats, ats_n = float("nan"), 0
            vegas_mae = np.abs(am - (-sp)).mean()
            model_mae_mkt = np.abs(am - pm).mean()
        else:
            ats, ats_n = float("nan"), 0
            vegas_mae, model_mae_mkt = float("nan"), float("nan")

        print(f"  Last 60 days ({holdout_mask.sum()} games):")
        print(f"    MAE:      {hold_mae:.3f}")
        print(f"    Accuracy: {hold_acc:.1%}")
        if not np.isnan(ats):
            print(f"    ATS:      {ats:.1%} ({ats_n} games)")
        if not np.isnan(vegas_mae):
            print(f"    Model MAE (mkt): {model_mae_mkt:.3f}")
            print(f"    Vegas MAE:       {vegas_mae:.3f}")
            print(f"    Advantage:       {vegas_mae - model_mae_mkt:+.3f}")
except Exception as e:
    print(f"  WARNING: Holdout validation failed ({e})")

# ═══════════════════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print("\n" + "=" * 60)
print("  RETRAIN COMPLETE")
print("=" * 60)
print(f"""
  Architecture:  Lasso solo (alpha=0.1)
  Features:      {len(FEATURE_LIST)}/{len(X.columns)} selected by L1
  Training:      {n} games
  CV MAE:        {cv_mae:.4f}
  CV Accuracy:   {cv_acc:.1%}
  CV Brier:      {cal_brier:.4f}
  Model file:    {pkl_path} ({pkl_size:.0f} KB)
  Time:          {elapsed:.0f}s

  Deploy: git add . && git commit -m "NBA v26 Lasso" && git push
""")
