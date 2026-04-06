#!/usr/bin/env python3
"""
nba_retrain_v27.py — NBA Model v27 Retrain (Post-Audit Fixes)
==============================================================
Architecture: Lasso α=0.1 on 38 V27 features
σ=7.0 (validated via Brier sweep)
Isotonic calibration for win probability

Fixes applied:
  - CRIT-1: home_after_loss, away_after_loss now in feature set
  - CRIT-2: crowd_pct uses real attendance (training) / team fill rates (serve)
  - roll_dreb_diff, roll_fast_break_diff restored
  - σ=7.0 (was 8.0)

Usage:
    python3 nba_retrain_v27.py                # Train + validate
    python3 nba_retrain_v27.py --upload       # Train + upload to Supabase
"""
import sys, os, time, argparse, warnings, pickle, io
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from nba_build_features_v27 import load_training_data, build_features

# ── Configuration ──
LASSO_ALPHA = 0.1
SIGMA = 7.0  # Validated via walk-forward Brier sweep (beat 8.0)
N_FOLDS = 30

V27_FEATURE_SET = [
    "lineup_value_diff", "win_pct_diff", "scoring_hhi_diff", "espn_pregame_wp",
    "ceiling_diff", "matchup_efg", "ml_implied_spread", "sharp_spread_signal",
    "efg_diff", "opp_suppression_diff", "net_rtg_diff", "steals_to_diff",
    "threepct_diff", "b2b_diff", "ftpct_diff", "ou_gap",
    "roll_dreb_diff", "ts_regression_diff", "roll_paint_pts_diff", "ref_home_whistle",
    "opp_ppg_diff", "roll_max_run_avg", "away_is_public_team", "away_after_loss",
    "games_last_14_diff", "h2h_total_games", "three_pt_regression_diff", "games_diff",
    "ref_foul_proxy", "roll_fast_break_diff", "crowd_pct", "matchup_to",
    "overround", "roll_ft_trip_rate_diff", "home_after_loss", "rest_diff",
    "spread_juice_imbalance", "vig_uncertainty",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload to Supabase")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA v27 RETRAIN — Post-Audit Fixes")
    print("  Lasso α=0.1, 38 features, σ=7.0")
    print("=" * 70)
    t0 = time.time()

    # ── Load data ──
    df = load_training_data("nba_training_data.parquet")
    X_all, all_features = build_features(df)
    y = df["target_margin"].values

    # Select V27 features
    available = [f for f in V27_FEATURE_SET if f in X_all.columns]
    missing = [f for f in V27_FEATURE_SET if f not in X_all.columns]
    if missing:
        print(f"\n  WARNING: {len(missing)} V27 features not in training data:")
        for f in missing:
            print(f"    ❌ {f}")

    feature_cols = available
    X = X_all[feature_cols]
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Games: {len(X)}")
    print(f"  Target: mean={y.mean():.2f}, std={y.std():.2f}")

    # ── Sort by date ──
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort().values
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    df = df.iloc[sort_idx].reset_index(drop=True)
    spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values

    # ── Walk-forward validation ──
    print(f"\n  Walk-forward validation ({N_FOLDS} folds)...")
    fold_size = len(X) // (N_FOLDS + 2)
    min_train = fold_size * 2
    oof_preds = np.full(len(X), np.nan)

    for fold in range(N_FOLDS):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, len(X))
        if ts >= len(X):
            break
        sc = StandardScaler()
        X_tr = sc.fit_transform(X.iloc[:ts])
        X_te = sc.transform(X.iloc[ts:te])
        model = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
        model.fit(X_tr, y[:ts])
        oof_preds[ts:te] = model.predict(X_te)

    valid = ~np.isnan(oof_preds)
    mae = np.mean(np.abs(oof_preds[valid] - y[valid]))
    actual_home_win = (y[valid] > 0).astype(float)

    # Win probability at σ=7.0
    probs = 1.0 / (1.0 + np.exp(-oof_preds[valid] / SIGMA))
    brier = float(np.mean((probs - actual_home_win) ** 2))
    acc = float(np.mean((probs > 0.5) == actual_home_win.astype(bool)))

    print(f"\n  MAE:    {mae:.3f}")
    print(f"  Acc:    {acc:.1%}")
    print(f"  Brier:  {brier:.4f}  (σ={SIGMA})")

    # ── ATS analysis ──
    print(f"\n  ATS Analysis:")
    v_spreads = spreads[valid]
    v_preds = oof_preds[valid]
    v_actual = y[valid]
    has_spread = np.abs(v_spreads) > 0.1

    for threshold in [0, 4, 7, 10]:
        ats_edge = v_preds - (-v_spreads)
        mask = has_spread & (np.abs(ats_edge) >= threshold)
        ats_result = v_actual + v_spreads
        not_push = ats_result != 0
        decidable = mask & not_push
        n = decidable.sum()
        if n < 20:
            continue
        correct = np.sign(ats_edge[decidable]) == np.sign(ats_result[decidable])
        ats_acc = correct.mean()
        roi = (ats_acc * 1.909 - 1) * 100
        print(f"    ATS {threshold:>2d}+: {ats_acc:.1%} ({n} picks) → ROI {roi:+.1f}%")

    # ── Feature importance ──
    # Train one model on all data to see coefficients
    sc_full = StandardScaler()
    X_s_full = sc_full.fit_transform(X)
    model_full = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
    model_full.fit(X_s_full, y)

    n_active = sum(1 for c in model_full.coef_ if abs(c) > 1e-6)
    coefs = sorted(zip(feature_cols, model_full.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Active features: {n_active}/{len(feature_cols)}")
    print(f"\n  Top 15 coefficients:")
    for feat, coef in coefs[:15]:
        bar = "█" * min(int(abs(coef) * 5), 30)
        print(f"    {feat:35s} {coef:+7.3f}  {bar}")

    zeroed = [f for f, c in coefs if abs(c) < 1e-6]
    if zeroed:
        print(f"\n  Zeroed by Lasso ({len(zeroed)}):")
        for f in zeroed:
            print(f"    ⚪ {f}")

    # ── Isotonic calibration ──
    raw_probs = 1.0 / (1.0 + np.exp(-oof_preds[valid] / SIGMA))
    sort_cal = np.argsort(raw_probs)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(raw_probs[sort_cal], actual_home_win[sort_cal])
    cal_probs = iso.predict(raw_probs)
    cal_brier = float(np.mean((cal_probs - actual_home_win) ** 2))
    print(f"\n  Isotonic calibration: Brier {brier:.4f} → {cal_brier:.4f}")

    # ── Classifier for win probability ──
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_s_full, (y > 0).astype(int))

    # ── Bias correction ──
    bias = float(np.mean(oof_preds[valid] - y[valid]))
    print(f"  Bias correction: {bias:+.4f}")

    # ── Build bundle ──
    bundle = {
        "reg": model_full,
        "scaler": sc_full,
        "clf": clf,
        "calibrator": iso,
        "isotonic": iso,
        "feature_cols": feature_cols,
        "feature_list": feature_cols,
        "model_type": "Lasso_v27_audit",
        "architecture": f"Lasso(alpha={LASSO_ALPHA})",
        "n_features": len(feature_cols),
        "n_active": n_active,
        "n_games": len(X),
        "cv_mae": round(mae, 4),
        "cv_brier": round(cal_brier, 4),
        "cv_acc": round(acc, 4),
        "sigma": SIGMA,
        "bias_correction": round(bias, 4),
        "trained_at": pd.Timestamp.now().isoformat(),
        "audit_fixes": ["CRIT-1_after_loss", "CRIT-2_crowd_pct", "HIGH-1_sigma_7.0",
                        "roll_dreb_diff", "roll_fast_break_diff"],
    }

    # ── Save locally ──
    local_path = "nba_v27_audit.pkl"
    import joblib
    joblib.dump(bundle, local_path)
    pkl_size = os.path.getsize(local_path) / 1024
    print(f"\n  Saved: {local_path} ({pkl_size:.0f} KB)")

    # ── Upload to Supabase ──
    if args.upload:
        print(f"\n  Uploading to Supabase model_store as 'nba'...")
        try:
            from db import save_model
            save_model("nba", bundle)
            print(f"  ✅ Uploaded successfully")

            # Verify
            from db import load_model
            check = load_model("nba")
            if check:
                print(f"  ✅ Verified: {check.get('model_type')}, {check.get('n_features')} features")
            else:
                print(f"  ❌ Verification failed — model not found after upload")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            print(f"  Manual upload: use save_model('nba', bundle) from Python")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RETRAIN COMPLETE")
    print(f"{'='*70}")
    print(f"""
  Architecture:  Lasso (α={LASSO_ALPHA})
  Features:      {len(feature_cols)} total, {n_active} active
  Games:         {len(X)}
  MAE:           {mae:.3f}
  Brier:         {cal_brier:.4f} (isotonic, σ={SIGMA})
  Accuracy:      {acc:.1%}
  Sigma:         {SIGMA}
  Bias:          {bias:+.4f}
  File:          {local_path} ({pkl_size:.0f} KB)
  Time:          {elapsed:.0f}s

  Deploy: git add . && git commit -m "NBA v27 audit retrain" && git push
  Then:   curl -s -X POST https://sports-predictor-api-production.up.railway.app/debug/reload-model/nba
""")


if __name__ == "__main__":
    main()
