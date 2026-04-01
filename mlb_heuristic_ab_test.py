#!/usr/bin/env python3
"""
mlb_heuristic_ab_test.py — Test if dropping heuristic features improves accuracy
================================================================================
Compares two models:
  A: Current 32 features (includes run_diff_pred + spread_vs_market)
  B: 30 features (drops run_diff_pred + spread_vs_market — pure raw stats)

If B matches or beats A on ATS accuracy, drop them permanently.
If B loses > 1%, keep them.

Usage:
    python3 mlb_heuristic_ab_test.py
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from catboost import CatBoostRegressor
from scipy.stats import norm as _norm

SEED = 42
N_FOLDS = 20
SIGMA = 4.0

from mlb_retrain import build_features as build_41_features, load_data

# Current 32-feature set
FEATURES_A = [
    "woba_diff", "fip_diff", "k_bb_diff", "bullpen_era_diff",
    "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff", "sp_fip_spread",
    "sp_relative_fip_diff",
    "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold",
    "temp_x_park",
    "rest_diff", "run_diff_pred",
    "market_spread", "spread_vs_market",
    "woba_x_park",
    "platoon_diff",
    "pyth_residual_diff", "babip_luck_diff", "scoring_entropy_diff",
    "first_inn_rate_diff", "clutch_divergence_diff", "opp_adj_form_diff",
    "ump_run_env", "series_game_num",
    "scoring_entropy_combined", "first_inn_rate_combined",
]

# Without heuristic features (drop run_diff_pred + spread_vs_market)
FEATURES_B = [f for f in FEATURES_A if f not in ("run_diff_pred", "spread_vs_market")]


def walk_forward(X, y, market_spread, weights, n_folds=N_FOLDS):
    """Walk-forward ensemble, returns (predictions, metrics)."""
    n = len(X)
    fold_size = n // n_folds
    preds = np.full(n, np.nan)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n) if fold < n_folds - 1 else n
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < 200:
            continue

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        w_tr = weights[train_idx] if weights is not None else None

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        m1 = Lasso(alpha=0.01, random_state=SEED, max_iter=5000)
        m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=5000)
        m3 = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.03,
                                subsample=0.8, random_seed=SEED, verbose=0)
        m1.fit(X_tr_s, y_tr, sample_weight=w_tr)
        m2.fit(X_tr_s, y_tr, sample_weight=w_tr)
        m3.fit(X_tr_s, y_tr, sample_weight=w_tr)
        preds[test_idx] = (m1.predict(X_te_s) + m2.predict(X_te_s) + m3.predict(X_te_s)) / 3

    return preds


def evaluate(name, preds, y, market_spread):
    """Print ATS accuracy at key thresholds."""
    valid = ~np.isnan(preds)
    pv, tv, ms = preds[valid], y[valid], market_spread[valid]

    mae = np.mean(np.abs(pv - tv))
    win_acc = np.mean(((pv > 0) == (tv > 0)))

    # ATS
    has_rl = ms != 0
    if has_rl.sum() < 100:
        print(f"  {name}: MAE={mae:.3f}, Win={win_acc:.1%}, ATS=insufficient market data")
        return mae, 0, 0

    model_margin = pv[has_rl]
    mkt_implied = -ms[has_rl]
    actual_margin = tv[has_rl]
    edge = np.abs(model_margin - mkt_implied)
    model_side = model_margin - mkt_implied
    actual_side = actual_margin - mkt_implied
    ats_correct = (model_side > 0) == (actual_side > 0)
    not_push = actual_side != 0

    print(f"\n  {'='*60}")
    print(f"  {name} ({len(pv)} games)")
    print(f"  {'='*60}")
    print(f"  MAE: {mae:.4f}   Win: {win_acc:.1%}")
    print(f"  {'Edge':>6} {'Games':>7} {'Correct':>8} {'Acc%':>7} {'ROI%':>7}")
    print(f"  {'─'*6} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")

    results = {}
    for thresh in [0, 0.5, 1.0, 1.5, 2.0]:
        mask = (edge >= thresh) & not_push
        if mask.sum() < 20:
            continue
        correct = ats_correct[mask].sum()
        total = mask.sum()
        acc = correct / total * 100
        roi = (correct * 100 - (total - correct) * 110) / (total * 110) * 100
        print(f"  {thresh:>5.1f}+ {total:>7d} {correct:>8d} {acc:>6.1f}% {roi:>+6.1f}%")
        results[thresh] = acc

    # Moneyline at key thresholds (using Gaussian CDF)
    win_prob = _norm.cdf(pv / SIGMA)
    home_won = tv > 0
    print(f"\n  Moneyline: ≥65% home = ", end="")
    mask65 = win_prob >= 0.65
    if mask65.sum() >= 20:
        print(f"{home_won[mask65].mean():.1%} ({mask65.sum()}g)")
    else:
        print("insufficient")

    return mae, results.get(0.5, 0), results.get(1.5, 0)


def main():
    print("=" * 60)
    print("  MLB HEURISTIC FEATURE A/B TEST")
    print("  A: 32 features (with run_diff_pred + spread_vs_market)")
    print("  B: 30 features (raw stats only — no heuristic)")
    print("=" * 60)

    df = load_data()
    y = df["target_margin"].values
    weights = df["season_weight"].values if "season_weight" in df.columns else None

    X_full = build_41_features(df)
    market_spread = pd.to_numeric(
        df.get("market_spread_home", pd.Series(0, index=df.index)),
        errors="coerce").fillna(0).values

    print(f"\n  Games: {len(df)}")
    print(f"  Games with market spread: {(market_spread != 0).sum()}")

    # ── Model A: Current (with heuristic features) ──
    print(f"\n  Running Model A (32 features, includes heuristic)...")
    X_a = X_full[FEATURES_A].values
    t0 = time.time()
    preds_a = walk_forward(X_a, y, market_spread, weights)
    print(f"  Complete in {time.time()-t0:.0f}s")
    mae_a, ats05_a, ats15_a = evaluate("MODEL A (with heuristic)", preds_a, y, market_spread)

    # ── Model B: Without heuristic features ──
    print(f"\n  Running Model B (30 features, raw stats only)...")
    X_b = X_full[FEATURES_B].values
    t0 = time.time()
    preds_b = walk_forward(X_b, y, market_spread, weights)
    print(f"  Complete in {time.time()-t0:.0f}s")
    mae_b, ats05_b, ats15_b = evaluate("MODEL B (no heuristic)", preds_b, y, market_spread)

    # ── Comparison ──
    print(f"\n  {'='*60}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"  {'='*60}")
    print(f"  {'Metric':<25} {'A (heuristic)':>15} {'B (raw only)':>15} {'Delta':>10}")
    print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*10}")
    print(f"  {'MAE':<25} {mae_a:>15.4f} {mae_b:>15.4f} {mae_b-mae_a:>+10.4f}")
    print(f"  {'ATS 0.5+ acc':<25} {ats05_a:>14.1f}% {ats05_b:>14.1f}% {ats05_b-ats05_a:>+9.1f}%")
    print(f"  {'ATS 1.5+ acc':<25} {ats15_a:>14.1f}% {ats15_b:>14.1f}% {ats15_b-ats15_a:>+9.1f}%")

    if mae_b <= mae_a + 0.02 and ats05_b >= ats05_a - 1.0:
        print(f"\n  ✅ VERDICT: DROP heuristic features. Model B matches or beats A.")
        print(f"     Remove 'run_diff_pred' and 'spread_vs_market' from FEATURE_COLS")
        print(f"     in mlb_ensemble_retrain.py, then retrain with --upload")
    else:
        print(f"\n  ❌ VERDICT: KEEP heuristic features. Model A is measurably better.")
        print(f"     The heuristic encodes non-linear signal the ensemble can't replicate.")


if __name__ == "__main__":
    main()
