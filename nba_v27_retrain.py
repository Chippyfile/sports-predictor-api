#!/usr/bin/env python3
"""
nba_v27_retrain.py — Train 55-feature 5-model ensemble with full validation.

Architecture: Lasso_0.1 + Ridge_1.0 + CatBoost_d6_800 + LightGBM_300 + GBM_200
Validation: 30-fold walk-forward with granular ATS bucket analysis

Run from ~/Desktop/sports-predictor-api/:
    python3 nba_v27_retrain.py              # Train + validate + save locally
    python3 nba_v27_retrain.py --upload     # Also upload to Supabase
    python3 nba_v27_retrain.py --validate   # Only validate, no save
"""
import sys, os, time, warnings, argparse, json
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from nba_ensemble import EnsembleRegressor

FEATURES_55 = [
    "after_loss_either", "altitude_factor", "b2b_diff", "bimodal_diff",
    "ceiling_diff", "conference_game", "consistency_diff", "elo_diff",
    "espn_pregame_wp", "espn_pregame_wp_pbp", "ftpct_diff", "games_diff",
    "games_last_14_diff", "h2h_avg_margin", "h2h_total_games",
    "implied_prob_home", "is_early_season", "is_friday_sat",
    "is_revenge_home", "lineup_value_diff", "margin_accel_diff",
    "matchup_efg", "matchup_ft", "matchup_orb", "momentum_halflife_diff",
    "opp_suppression_diff", "ou_gap", "overround", "pace_control_diff",
    "pace_leverage", "post_allstar", "post_trade_deadline",
    "pyth_luck_diff", "pyth_residual_diff", "recovery_diff",
    "ref_foul_proxy", "ref_home_whistle", "ref_ou_bias",
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
    "CatBoost": lambda: CatBoostRegressor(
        depth=6, iterations=800, learning_rate=0.03, l2_leaf_reg=5,
        random_seed=42, verbose=0),
    "LightGBM": lambda: LGBMRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, verbose=-1, random_state=42),
    "GBM": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42),
}


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

def load_and_prepare():
    from nba_build_features_v27 import load_training_data, build_features

    df = load_training_data()
    X_all, feature_names = build_features(df)
    y = (df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)).values
    spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values

    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X_all = X_all.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    spreads = spreads[sort_idx]
    df = df.iloc[sort_idx].reset_index(drop=True)

    available = [f for f in FEATURES_55 if f in X_all.columns]
    missing = [f for f in FEATURES_55 if f not in X_all.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing: {missing}")

    X = X_all[available]
    return X, y, spreads, df, available


# ═══════════════════════════════════════════════════════════
# GRANULAR ATS ANALYSIS
# ═══════════════════════════════════════════════════════════

def granular_ats_analysis(preds, actual, spreads):
    """Analyze ATS accuracy across granular edge buckets."""

    # Only games with real spreads
    has_spread = np.abs(spreads) > 0.1
    preds = preds[has_spread]
    actual = actual[has_spread]
    spr = spreads[has_spread]

    ats_edge = preds - (-spr)  # positive = model says home covers
    ats_margin = actual + spr
    not_push = ats_margin != 0
    ats_correct = np.sign(ats_edge) == np.sign(ats_margin)

    # Filter out pushes
    mask = not_push
    edge = ats_edge[mask]
    correct = ats_correct[mask]

    mae = round(float(np.mean(np.abs(preds - actual))), 3)
    overall_acc = float(correct.mean())

    print(f"\n  Overall: {len(edge)} graded games, MAE={mae}, ATS={overall_acc:.1%}")

    abs_edge = np.abs(edge)

    # ── Cumulative ATS by threshold ──
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    print(f"\n  === CUMULATIVE ATS BY EDGE THRESHOLD ===")
    print(f"  {'Threshold':>10s} {'Games':>7s} {'ATS %':>7s} {'ROI':>8s} {'Profitable':>11s}")
    print("  " + "-" * 50)

    threshold_results = {}
    for t in thresholds:
        tmask = abs_edge >= t
        n = int(tmask.sum())
        if n >= 10:
            acc = float(correct[tmask].mean())
            roi = round((acc * 1.909 - 1) * 100, 1)
            be = "YES" if acc > 0.524 else "no"
        else:
            acc = 0
            roi = 0
            be = "n/a"
        threshold_results[t] = {"n": n, "acc": acc, "roi": roi}
        print(f"  {t:>10d}+ {n:>7d} {acc:>6.1%} {roi:>+7.1f}% {be:>11s}")

    # ── Signed edge distribution ──
    print(f"\n  === SIGNED EDGE DISTRIBUTION ===")
    print(f"  (Negative = model closer to 0 than spread; Positive = model sees more value)")
    print(f"\n  {'Edge Range':>12s} {'N':>6s} {'ATS%':>7s} {'ROI':>8s} {'Interpretation'}")
    print("  " + "-" * 65)

    signed_buckets = [
        (-999, -10, "Strong anti-value (fade)"),
        (-10, -7, "Moderate anti-value"),
        (-7, -4, "Slight anti-value"),
        (-4, -2, "Marginal anti-value"),
        (-2, 0, "Near-neutral (no edge)"),
        (0, 2, "Near-neutral (tiny edge)"),
        (2, 4, "Marginal value"),
        (4, 7, "Moderate value (bet zone)"),
        (7, 10, "Strong value (high-conf)"),
        (10, 999, "Extreme value (max conf)"),
    ]

    for lo, hi, interp in signed_buckets:
        bmask = (edge > lo) & (edge <= hi)
        n = int(bmask.sum())
        if n >= 10:
            acc = float(correct[bmask].mean())
            roi = round((acc * 1.909 - 1) * 100, 1)
        else:
            acc = 0
            roi = 0
        label = f"{lo:+d} to {hi:+d}" if lo > -999 and hi < 999 else (f"  <= {hi:+d}" if lo == -999 else f"  >= {lo:+d}")
        print(f"  {label:>12s} {n:>6d} {acc:>6.1%} {roi:>+7.1f}%  {interp}")

    # ── Win probability calibration ──
    print(f"\n  === WIN PROBABILITY CALIBRATION ===")
    print(f"  (Does predicted win% match actual win%?)")

    raw_probs = 1.0 / (1.0 + np.exp(-preds[has_spread][mask] / 8.0))
    actual_wins = (actual[mask] > 0).astype(float)

    prob_buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5),
                    (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.0)]

    print(f"\n  {'Pred WP%':>12s} {'N':>6s} {'Actual Win%':>12s} {'Error':>8s} {'Quality'}")
    print("  " + "-" * 55)

    for lo, hi in prob_buckets:
        pmask = (raw_probs >= lo) & (raw_probs < hi)
        n = int(pmask.sum())
        if n >= 10:
            actual_rate = float(actual_wins[pmask].mean())
            expected = (lo + hi) / 2
            cal_error = round(actual_rate - expected, 3)
            quality = "GOOD" if abs(cal_error) < 0.03 else ("over" if cal_error > 0 else "under")
            print(f"  {lo:.0%}-{hi:.0%} {n:>6d} {actual_rate:>11.1%} {cal_error:>+7.3f}  {quality}")

    # ── Season-by-season consistency ──
    print(f"\n  === SEASON CONSISTENCY ===")
    # We need season info — approximate from game count
    n_total = len(preds)
    games_per_season = n_total // 8  # ~8 seasons in data
    print(f"  (Approximate: ~{games_per_season} spread-games per season chunk)")

    chunk_size = max(games_per_season, 200)
    n_chunks = max(1, len(edge) // chunk_size)
    print(f"\n  {'Chunk':>8s} {'Games':>7s} {'ATS%':>7s} {'ATS 7+':>8s} {'ROI 7+':>8s}")
    print("  " + "-" * 45)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(edge))
        chunk_edge = edge[start:end]
        chunk_correct = correct[start:end]
        chunk_n = len(chunk_edge)
        if chunk_n < 50:
            continue
        chunk_acc = float(chunk_correct.mean())
        m7 = np.abs(chunk_edge) >= 7
        if m7.sum() >= 10:
            a7 = float(chunk_correct[m7].mean())
            r7 = round((a7 * 1.909 - 1) * 100, 1)
        else:
            a7 = 0
            r7 = 0
        print(f"  {i+1:>8d} {chunk_n:>7d} {chunk_acc:>6.1%} {a7:>7.1%} {r7:>+7.1f}%")

    return threshold_results, mae


# ═══════════════════════════════════════════════════════════
# 30-FOLD WALK-FORWARD
# ═══════════════════════════════════════════════════════════

def walk_forward_validate(X, y, spreads, n_folds=30):
    n = len(X)
    fold_size = n // (n_folds + 2)
    min_train = fold_size * 3

    X_vals = X.values if hasattr(X, 'values') else X

    per_model = {name: np.full(n, np.nan) for name in MODEL_CONFIGS}
    ensemble_preds = np.full(n, np.nan)

    print(f"\n  30-fold walk-forward ({n} games, fold_size={fold_size}, min_train={min_train})...")
    t0 = time.time()

    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n or te <= ts:
            break

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_vals[:ts])
        X_te = sc.transform(X_vals[ts:te])

        fold_preds = []
        for name, builder in MODEL_CONFIGS.items():
            mdl = builder()
            mdl.fit(X_tr, y[:ts])
            p = mdl.predict(X_te)
            per_model[name][ts:te] = p
            fold_preds.append(p)

        ensemble_preds[ts:te] = np.mean(fold_preds, axis=0)

        if (fold + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Fold {fold+1}/{n_folds} ({elapsed:.0f}s)")

    print(f"  Walk-forward complete in {time.time()-t0:.0f}s")

    valid = ~np.isnan(ensemble_preds) & (np.abs(spreads) > 0.1)
    print(f"  Valid test games with spreads: {valid.sum()}")

    # Per-model summary
    print(f"\n  {'Model':<15s} {'MAE':>7s} {'ATS':>6s} {'ATS 4+':>7s} {'ATS 7+':>7s} {'ATS 10+':>7s} {'N7':>5s}")
    print("  " + "-" * 55)

    for name in MODEL_CONFIGS:
        v = ~np.isnan(per_model[name]) & (np.abs(spreads) > 0.1)
        if v.sum() < 100: continue
        p = per_model[name][v]; a = y[v]; s = spreads[v]
        mae = np.mean(np.abs(p - a))
        edge = p - (-s); margin = a + s; np_ = margin != 0
        cor = np.sign(edge) == np.sign(margin)
        def _at(t):
            m = (np.abs(edge) >= t) & np_
            return (cor[m].mean() if m.sum()>=20 else 0, m.sum())
        a0,_ = _at(0); a4,_ = _at(4); a7,n7 = _at(7); a10,_ = _at(10)
        print(f"  {name:<15s} {mae:>7.3f} {a0:>5.1%} {a4:>6.1%} {a7:>6.1%} {a10:>6.1%} {n7:>5d}")

    # Ensemble row
    ep = ensemble_preds[valid]; ea = y[valid]; es = spreads[valid]
    emae = np.mean(np.abs(ep - ea))
    ee = ep - (-es); em = ea + es; enp = em != 0; ec = np.sign(ee) == np.sign(em)
    def _eat(t):
        m = (np.abs(ee) >= t) & enp
        return (ec[m].mean() if m.sum()>=20 else 0, m.sum())
    ea0,_ = _eat(0); ea4,_ = _eat(4); ea7,n7 = _eat(7); ea10,_ = _eat(10)
    print(f"  {'ENSEMBLE':<15s} {emae:>7.3f} {ea0:>5.1%} {ea4:>6.1%} {ea7:>6.1%} {ea10:>6.1%} {n7:>5d}")

    return ensemble_preds, per_model


# ═══════════════════════════════════════════════════════════
# TRAIN PRODUCTION
# ═══════════════════════════════════════════════════════════

def train_production(X, y, feature_cols):
    print(f"\n  Training production ensemble on {len(X)} games, {len(feature_cols)} features...")

    X_vals = X.values if hasattr(X, 'values') else X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals)

    models = []; model_names = []
    t0 = time.time()
    for name, builder in MODEL_CONFIGS.items():
        t1 = time.time()
        mdl = builder()
        mdl.fit(X_scaled, y)
        models.append(mdl)
        model_names.append(name)
        in_mae = np.mean(np.abs(mdl.predict(X_scaled) - y))
        print(f"    {name:<15s} trained in {time.time()-t1:.1f}s (in-sample MAE: {in_mae:.3f})")

    print(f"  All 5 models trained in {time.time()-t0:.0f}s")

    lasso_model = models[model_names.index("Lasso")]
    ensemble = EnsembleRegressor(models, model_names, shap_model=lasso_model)
    print(f"  Ensemble in-sample MAE: {np.mean(np.abs(ensemble.predict(X_scaled) - y)):.3f}")

    raw_probs = 1.0 / (1.0 + np.exp(-ensemble.predict(X_scaled) / 8.0))
    actual_wins = (y > 0).astype(float)
    sort_idx = np.argsort(raw_probs)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(raw_probs[sort_idx], actual_wins[sort_idx])
    print(f"  Isotonic calibration: {np.mean((iso.predict(raw_probs) > 0.5) == (y > 0)):.1%} accuracy")

    return ensemble, scaler, iso, models, model_names


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA v27 RETRAIN — 55 Features, 5-Model Ensemble (depth 6)")
    print("  Lasso + Ridge + CatBoost_d6_800 + LightGBM_300 + GBM_200")
    print("=" * 70)

    X, y, spreads, df, feature_cols = load_and_prepare()
    print(f"\n  Data: {len(X)} games, {len(feature_cols)} features, {(np.abs(spreads) > 0.1).sum()} with spreads")

    # Walk-forward
    ensemble_preds, per_model = walk_forward_validate(X, y, spreads, n_folds=30)

    # Granular analysis
    print(f"\n{'='*70}")
    print(f"  GRANULAR ATS ANALYSIS (30-fold walk-forward, spread games only)")
    print(f"{'='*70}")

    valid = ~np.isnan(ensemble_preds) & (np.abs(spreads) > 0.1)
    threshold_results, wf_mae = granular_ats_analysis(
        ensemble_preds[valid], y[valid], spreads[valid])

    if args.validate:
        print("\n  --validate mode: skipping production training")
        sys.exit(0)

    # Train
    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING")
    print(f"{'='*70}")

    ensemble, scaler, isotonic, models, model_names = train_production(X, y, feature_cols)

    ats_7 = threshold_results.get(7, {})
    ats_10 = threshold_results.get(10, {})

    bundle = {
        "reg": ensemble,
        "scaler": scaler,
        "calibrator": isotonic,
        "feature_cols": feature_cols,
        "model_type": "ensemble_5_v27",
        "architecture": "Lasso+Ridge+CatBoost_d6_800+LightGBM_300+GBM_200",
        "n_features": len(feature_cols),
        "n_games": len(X),
        "n_train": len(X),
        "cv_mae": wf_mae,
        "cv_ats_7_acc": ats_7.get("acc", 0),
        "cv_ats_7_n": ats_7.get("n", 0),
        "cv_ats_10_acc": ats_10.get("acc", 0),
        "cv_ats_10_n": ats_10.get("n", 0),
        "cv_folds": 30,
        "catboost_depth": 6,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_names": model_names,
    }

    local_path = "nba_v27_ensemble.pkl"
    joblib.dump(bundle, local_path, compress=3)
    size_kb = os.path.getsize(local_path) / 1024

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Architecture: {bundle['architecture']}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training games: {len(X)}")
    print(f"  Walk-forward MAE: {wf_mae:.3f}")
    print(f"  ATS 7+ accuracy: {ats_7.get('acc', 0):.1%} ({ats_7.get('n', 0)} picks)")
    print(f"  ATS 10+ accuracy: {ats_10.get('acc', 0):.1%} ({ats_10.get('n', 0)} picks)")
    print(f"  Local file: {local_path} ({size_kb:.0f} KB)")

    if args.upload:
        print(f"\n  Uploading to Supabase...")
        try:
            from db import save_model
            save_model("nba", bundle)
            print(f"  ✅ Uploaded to Supabase model_store as 'nba'")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
    else:
        print(f"\n  To upload: python3 nba_v27_retrain.py --upload")
