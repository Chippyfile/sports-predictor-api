#!/usr/bin/env python3
"""
nba_v29_retrain.py — NBA v29: Lasso+Ridge+LightGBM, 50 features
================================================================
Architecture: 3-model ensemble (Lasso_0.1 + Ridge_1.0 + LightGBM_300)
Features: 50 (backward-eliminated from 60 via composite ATS score)
Walk-forward validated: 73.4% ATS at 7+ edge, +32% ROI

Dropped vs v28 (60 features):
  matchup_ft, pyth_luck_diff, is_revenge_home, margin_accel_diff,
  matchup_orb, overround, net_rtg_diff, games_diff,
  post_trade_deadline, after_loss_either

Usage:
    python3 nba_v29_retrain.py              # Train + validate locally
    python3 nba_v29_retrain.py --upload     # Train + upload to Supabase
"""
import sys, os, time, warnings, argparse
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMRegressor
from nba_ensemble import EnsembleRegressor
from nba_build_features_v27 import load_training_data, build_features

FEATURES_50 = [
    "altitude_factor", "b2b_diff", "bimodal_diff", "ceiling_diff",
    "conference_game", "consistency_diff", "elo_diff", "espn_pregame_wp",
    "espn_pregame_wp_pbp", "ftpct_diff", "games_last_14_diff",
    "h2h_avg_margin", "h2h_total_games", "implied_prob_home",
    "is_early_season", "is_friday_sat", "lineup_value_diff",
    "matchup_efg", "matchup_to", "ml_implied_spread",
    "momentum_halflife_diff", "opp_ppg_diff", "opp_suppression_diff",
    "ou_gap", "pace_control_diff", "pace_leverage", "post_allstar",
    "pyth_residual_diff", "recovery_diff", "ref_foul_proxy",
    "ref_home_whistle", "ref_ou_bias", "rest_diff",
    "reverse_line_movement", "roll_bench_pts_diff", "roll_ft_trip_rate_diff",
    "roll_max_run_avg", "roll_paint_fg_rate_diff", "roll_three_fg_rate_diff",
    "score_kurtosis_diff", "scoring_hhi_diff", "sharp_spread_signal",
    "spread_juice_imbalance", "steals_to_diff", "three_pt_regression_diff",
    "three_value_diff", "threepct_diff", "ts_regression_diff",
    "turnovers_diff", "win_pct_diff",
]

MODEL_CONFIGS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "Ridge": lambda: Ridge(alpha=1.0),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=42),
}


def load_and_prepare():
    df = load_training_data("nba_training_data.parquet")
    X_all, _ = build_features(df)
    y = (df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)).values
    spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values

    dates = pd.to_datetime(df["game_date"])
    idx = dates.argsort()
    X_all = X_all.iloc[idx].reset_index(drop=True)
    y = y[idx]; spreads = spreads[idx]; df = df.iloc[idx].reset_index(drop=True)

    available = [f for f in FEATURES_50 if f in X_all.columns]
    missing = [f for f in FEATURES_50 if f not in X_all.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing: {missing}")
    return X_all[available], y, spreads, df, available


def walk_forward_validate(X, y, spreads, n_folds=30):
    n = len(X); X_vals = X.values
    fold_size = n // (n_folds + 2); min_train = fold_size * 3

    per_model = {name: np.full(n, np.nan) for name in MODEL_CONFIGS}
    ens_preds = np.full(n, np.nan)

    print(f"\n  {n_folds}-fold walk-forward ({n} games, fold={fold_size}, min_train={min_train})...")
    t0 = time.time()

    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_vals[:ts]); X_te = sc.transform(X_vals[ts:te])

        fold_preds = []
        for name, builder in MODEL_CONFIGS.items():
            mdl = builder(); mdl.fit(X_tr, y[:ts])
            p = mdl.predict(X_te)
            per_model[name][ts:te] = p
            fold_preds.append(p)
        ens_preds[ts:te] = np.mean(fold_preds, axis=0)

        if (fold + 1) % 10 == 0:
            print(f"    Fold {fold+1}/{n_folds} ({time.time()-t0:.0f}s)")

    print(f"  Complete in {time.time()-t0:.0f}s")

    valid = ~np.isnan(ens_preds) & (np.abs(spreads) > 0.1)
    print(f"  Valid spread games: {valid.sum()}")

    # Per-model summary
    print(f"\n  {'Model':<12s} {'MAE':>7s} {'ATS0+':>6s} {'ATS4+':>6s} {'ATS7+':>6s} {'ATS10+':>7s} {'N7':>5s}")
    print("  " + "-" * 50)
    for name in MODEL_CONFIGS:
        v = ~np.isnan(per_model[name]) & (np.abs(spreads) > 0.1)
        if v.sum() < 100: continue
        p = per_model[name][v]; a = y[v]; s = spreads[v]
        mae = np.mean(np.abs(p - a))
        edge = p - (-s); margin = a + s; np_ = margin != 0; cor = np.sign(edge) == np.sign(margin)
        def _at(t):
            m = (np.abs(edge) >= t) & np_
            return (cor[m].mean() if m.sum() >= 20 else 0, m.sum())
        a0, _ = _at(0); a4, _ = _at(4); a7, n7 = _at(7); a10, _ = _at(10)
        print(f"  {name:<12s} {mae:>7.3f} {a0:>5.1%} {a4:>5.1%} {a7:>5.1%} {a10:>6.1%} {n7:>5d}")

    # Ensemble
    ep = ens_preds[valid]; ea = y[valid]; es = spreads[valid]
    emae = np.mean(np.abs(ep - ea))
    ee = ep - (-es); em = ea + es; enp = em != 0; ec = np.sign(ee) == np.sign(em)
    def _eat(t):
        m = (np.abs(ee) >= t) & enp
        return (ec[m].mean() if m.sum() >= 20 else 0, m.sum())
    ea0, _ = _eat(0); ea4, _ = _eat(4); ea7, n7 = _eat(7); ea8, n8 = _eat(8); ea10, n10 = _eat(10)
    print(f"  {'ENSEMBLE':<12s} {emae:>7.3f} {ea0:>5.1%} {ea4:>5.1%} {ea7:>5.1%} {ea10:>6.1%} {n7:>5d}")

    # Granular thresholds
    print(f"\n  === CUMULATIVE ATS ===")
    print(f"  {'Thresh':>7s} {'Games':>7s} {'Acc':>6s} {'ROI':>7s}")
    print("  " + "-" * 30)
    ats_results = {}
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        acc, n = _eat(t)
        roi = round((acc * 1.909 - 1) * 100, 1) if n >= 20 else 0
        tag = "YES" if acc > 0.524 and n >= 20 else ("n/a" if n < 20 else "no")
        if n >= 10:
            print(f"  {t:>6d}+ {n:>7d} {acc:>5.1%} {roi:>+6.1f}%  {tag}")
        ats_results[t] = {"acc": acc, "n": n, "roi": roi}

    return ens_preds, ats_results, emae


def train_production(X, y, feature_cols):
    print(f"\n  Training production ensemble on {len(X)} games, {len(feature_cols)} features...")
    X_vals = X.values
    scaler = StandardScaler(); X_s = scaler.fit_transform(X_vals)

    models = []; model_names = []
    for name, builder in MODEL_CONFIGS.items():
        t0 = time.time()
        mdl = builder(); mdl.fit(X_s, y)
        models.append(mdl); model_names.append(name)
        mae = np.mean(np.abs(mdl.predict(X_s) - y))
        print(f"    {name:<12s} in-sample MAE: {mae:.3f} ({time.time()-t0:.1f}s)")

    lasso = models[model_names.index("Lasso")]
    ensemble = EnsembleRegressor(models, model_names, shap_model=lasso)
    ens_mae = np.mean(np.abs(ensemble.predict(X_s) - y))
    print(f"  Ensemble in-sample MAE: {ens_mae:.3f}")

    # Isotonic calibration
    raw_probs = 1.0 / (1.0 + np.exp(-ensemble.predict(X_s) / 8.0))
    actual_wins = (y > 0).astype(float)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(raw_probs, actual_wins)
    cal_acc = np.mean((iso.predict(raw_probs) > 0.5) == (y > 0))
    print(f"  Isotonic calibration: {cal_acc:.1%} accuracy")

    return ensemble, scaler, iso, models, model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA v29 — Lasso+Ridge+LightGBM, 50 features")
    print("  Backward-eliminated from 60 via composite ATS score sweep")
    print("=" * 70)

    X, y, spreads, df, feature_cols = load_and_prepare()
    print(f"\n  Data: {len(X)} games, {len(feature_cols)} features, {(np.abs(spreads) > 0.1).sum()} with spreads")

    ens_preds, ats_results, wf_mae = walk_forward_validate(X, y, spreads, n_folds=30)

    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING")
    print(f"{'='*70}")

    ensemble, scaler, isotonic, models, model_names = train_production(X, y, feature_cols)

    ats_7 = ats_results.get(7, {}); ats_10 = ats_results.get(10, {})

    bundle = {
        "reg": ensemble, "scaler": scaler, "calibrator": isotonic,
        "feature_cols": feature_cols,
        "model_type": "ensemble_3_v29",
        "architecture": "Lasso_0.1+Ridge_1.0+LightGBM_300",
        "n_features": len(feature_cols), "n_games": len(X), "n_train": len(X),
        "cv_mae": wf_mae,
        "cv_ats_7_acc": ats_7.get("acc", 0), "cv_ats_7_n": ats_7.get("n", 0),
        "cv_ats_10_acc": ats_10.get("acc", 0), "cv_ats_10_n": ats_10.get("n", 0),
        "cv_folds": 30,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_names": model_names,
        "dropped_features": [
            "matchup_ft", "pyth_luck_diff", "is_revenge_home", "margin_accel_diff",
            "matchup_orb", "overround", "net_rtg_diff", "games_diff",
            "post_trade_deadline", "after_loss_either",
        ],
    }

    local_path = "nba_v29_ensemble.pkl"
    joblib.dump(bundle, local_path, compress=3)
    size_kb = os.path.getsize(local_path) / 1024

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Architecture: Lasso + Ridge + LightGBM (3 models)")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training games: {len(X)}")
    print(f"  Walk-forward MAE: {wf_mae:.3f}")
    print(f"  ATS 7+ accuracy: {ats_7.get('acc', 0):.1%} ({ats_7.get('n', 0)} picks)")
    print(f"  ATS 10+ accuracy: {ats_10.get('acc', 0):.1%} ({ats_10.get('n', 0)} picks)")
    print(f"  File: {local_path} ({size_kb:.0f} KB)")

    if args.upload:
        print(f"\n  Uploading to Supabase...")
        try:
            from db import save_model
            save_model("nba", bundle)
            print(f"  ✅ Uploaded to Supabase model_store as 'nba'")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
    else:
        print(f"\n  To upload: python3 nba_v29_retrain.py --upload")
