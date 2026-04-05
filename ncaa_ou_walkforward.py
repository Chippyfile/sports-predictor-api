#!/usr/bin/env python3
"""
Walk-forward test: Does market + O/U residual predict actual totals
better than market alone?

Uses the same architecture as ncaa_retrain_ou_v4.py residual models
(Lasso + LGBM + CatBoost ensemble) in true walk-forward fashion.
"""
import sys, os, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor

from dump_training_data import load_cached
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SEED = 42
N_FOLDS = 30

print("=" * 70)
print("  WALK-FORWARD: Market + Residual Total Accuracy")
print("=" * 70)

# ── Load and filter ──
df = load_cached()
if df is None:
    print("  ERROR: No cached parquet"); sys.exit(1)

df = df[df["actual_home_score"].notna() & df["actual_away_score"].notna()].copy()
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()

df["market_ou_total"] = pd.to_numeric(df.get("market_ou_total", 0), errors="coerce")
df = df[df["market_ou_total"].notna() & (df["market_ou_total"] > 80) & (df["market_ou_total"] < 220)].copy()

df["actual_total"] = df["actual_home_score"].astype(float) + df["actual_away_score"].astype(float)
df = df[(df["actual_total"] > 60) & (df["actual_total"] < 250)].copy()

# Push filter
df = df[(df["actual_total"] - df["market_ou_total"]).abs() > 0.25].copy()

df = df.sort_values("game_date").reset_index(drop=True)
seasons = df["season"].values.astype(int)
print(f"  {len(df)} games (non-push, with market O/U)")
print(f"  Seasons: {sorted(df['season'].unique())}")

# ── Heuristic backfill + features ──
print("  Computing features...")
df = _ncaa_backfill_heuristic(df)
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)
try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except Exception:
    pass
try:
    import json
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
except Exception:
    pass

X_full = ncaa_build_features(df)

# ── Use the same tight feature set as O/U v4 residual ──
# These are the 37-45 features from ncaa_retrain_ou_v4
res_features = [f for f in X_full.columns if X_full[f].notna().mean() > 0.5]
print(f"  {len(res_features)} features available")

X = X_full[res_features].fillna(0).values
y_total = df["actual_total"].values
mkt = df["market_ou_total"].values
residual_target = y_total - mkt  # what the residual model predicts

# ── Season weights (recent scheme for O/U) ──
weights = np.array([{2026:1.0, 2025:1.0, 2024:0.5, 2023:0.25}.get(s, 0.1) for s in seasons])

# ── Walk-forward ──
print(f"\n  Walk-forward ({N_FOLDS} folds)...")
n = len(X)
fold_size = n // (N_FOLDS + 1)
min_train = fold_size * 2

oof_residual = np.full(n, np.nan)

for fold in range(N_FOLDS):
    ts = min_train + fold * fold_size
    te = min(ts + fold_size, n)
    if ts >= n:
        break

    X_train, y_train, w_train = X[:ts], residual_target[:ts], weights[:ts]
    X_test = X[ts:te]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    preds = []

    # Lasso
    m1 = Lasso(alpha=0.1, max_iter=5000)
    m1.fit(X_tr_s, y_train, sample_weight=w_train)
    preds.append(m1.predict(X_te_s))

    # LightGBM
    m2 = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                        subsample=0.8, verbose=-1, random_state=SEED)
    m2.fit(X_tr_s, y_train, sample_weight=w_train)
    preds.append(m2.predict(X_te_s))

    oof_residual[ts:te] = np.mean(preds, axis=0)

    if (fold + 1) % 10 == 0:
        print(f"    Fold {fold + 1}/{N_FOLDS}")

# ── Results ──
valid = ~np.isnan(oof_residual)
res_pred = oof_residual[valid]
actual = y_total[valid]
mkt_v = mkt[valid]

pred_total = mkt_v + res_pred  # market + residual

mkt_mae = np.mean(np.abs(mkt_v - actual))
model_mae = np.mean(np.abs(pred_total - actual))
residual_mae = np.mean(np.abs(res_pred - (actual - mkt_v)))

print(f"\n{'=' * 70}")
print(f"  RESULTS ({valid.sum()} games)")
print(f"{'=' * 70}")
print(f"  Market-only MAE:          {mkt_mae:.3f}")
print(f"  Market + Residual MAE:    {model_mae:.3f} {'✅ beats market' if model_mae < mkt_mae else '❌ loses to market'}")
print(f"  Improvement:              {mkt_mae - model_mae:+.3f} pts")
print(f"  Residual prediction MAE:  {residual_mae:.3f}")
print(f"  Residual mean:            {res_pred.mean():+.2f}")
print(f"  Residual std:             {res_pred.std():.2f}")

# ── O/U accuracy at thresholds ──
print(f"\n  O/U Accuracy (market + residual vs market-only):")
print(f"  {'Thresh':>8s} {'M+R OVER':>10s} {'M+R UNDER':>12s} {'M+R Total':>10s} {'Mkt Total':>10s} {'Picks':>8s}")
print(f"  {'-' * 65}")

for thresh in [0, 0.5, 1, 1.5, 2, 3, 4, 5]:
    # Model: predict OVER when residual > thresh, UNDER when residual < -thresh
    over_mask = res_pred > thresh
    under_mask = res_pred < -thresh
    skip_mask = ~over_mask & ~under_mask

    n_over = over_mask.sum()
    n_under = under_mask.sum()
    n_picks = n_over + n_under

    if n_picks == 0:
        continue

    over_correct = (actual[over_mask] > mkt_v[over_mask]).sum() if n_over > 0 else 0
    under_correct = (actual[under_mask] < mkt_v[under_mask]).sum() if n_under > 0 else 0

    over_pct = over_correct / max(n_over, 1)
    under_pct = under_correct / max(n_under, 1)
    total_pct = (over_correct + under_correct) / max(n_picks, 1)

    # Market baseline: random 50/50 on same games
    mkt_over_correct = (actual[over_mask] > mkt_v[over_mask]).sum() if n_over > 0 else 0
    mkt_under_correct = (actual[under_mask] < mkt_v[under_mask]).sum() if n_under > 0 else 0

    print(f"  {thresh:>8.1f} {over_pct:>9.1%} ({n_over:>4d}) {under_pct:>9.1%} ({n_under:>4d}) {total_pct:>9.1%} {'':>10s} {n_picks:>8d}")

# ── Calibration: is the residual well-calibrated? ──
print(f"\n  Residual calibration (binned):")
print(f"  {'Pred Bin':>12s} {'Actual Res':>12s} {'Count':>8s} {'Calibrated':>12s}")
print(f"  {'-' * 50}")
for lo, hi in [(-20, -5), (-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5), (5, 20)]:
    mask = (res_pred >= lo) & (res_pred < hi)
    if mask.sum() < 20:
        continue
    actual_res = (actual[mask] - mkt_v[mask]).mean()
    pred_res = res_pred[mask].mean()
    print(f"  [{lo:>3d},{hi:>3d}) {actual_res:>+10.1f} {mask.sum():>8d} {'✅' if np.sign(actual_res) == np.sign(pred_res) else '❌':>12s}")

# ── Season breakdown ──
print(f"\n  Per-season MAE:")
seasons_v = seasons[valid]
for s in sorted(set(seasons_v)):
    mask = seasons_v == s
    if mask.sum() < 50:
        continue
    s_mkt = np.mean(np.abs(mkt_v[mask] - actual[mask]))
    s_model = np.mean(np.abs(pred_total[mask] - actual[mask]))
    delta = s_mkt - s_model
    print(f"    {s}: Market={s_mkt:.2f}, M+Residual={s_model:.2f}, Δ={delta:+.2f} {'✅' if delta > 0 else '❌'}")
