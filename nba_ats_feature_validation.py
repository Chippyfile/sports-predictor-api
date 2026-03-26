#!/usr/bin/env python3
"""
nba_ats_feature_validation.py — Verify the 38 v27 features are optimal for ATS.

Walk-forward cross-validation with Lasso (matching production architecture).
Tests:
  1. Baseline: all 38 features → MAE, ATS%, ATS at 4+/7+ unit edges
  2. Drop-one: remove each feature → measure ATS impact
  3. Add-back: test candidate features from v20 builder not in v27

Run from ~/Desktop/sports-predictor-api/:
    python3 nba_ats_feature_validation.py

Requires: nba_historical + nba_predictions in Supabase, sports/nba.py
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

# ── Import project modules ──
from db import sb_get
from sports.nba import nba_build_features, _nba_merge_historical, _nba_backfill_heuristic

try:
    from dynamic_constants import compute_nba_league_averages, NBA_DEFAULT_AVERAGES
    _nba_lg = compute_nba_league_averages()
    if _nba_lg:
        nba_build_features._league_averages = _nba_lg
        print(f"  Using dynamic NBA averages ({len(_nba_lg)} stats)")
except Exception:
    print("  Using static NBA averages")

# ═══════════════════════════════════════════════════════════
# V27 FEATURE LIST (must match nba_v27_features_live.py)
# ═══════════════════════════════════════════════════════════

V27_FEATURES = [
    "lineup_value_diff", "win_pct_diff", "scoring_hhi_diff",
    "espn_pregame_wp", "ceiling_diff", "matchup_efg",
    "ml_implied_spread", "sharp_spread_signal", "efg_diff",
    "opp_suppression_diff", "net_rtg_diff", "steals_to_diff",
    "threepct_diff", "b2b_diff", "ftpct_diff", "ou_gap",
    "roll_dreb_diff", "ts_regression_diff", "roll_paint_pts_diff",
    "ref_home_whistle", "opp_ppg_diff", "roll_max_run_avg",
    "away_is_public_team", "away_after_loss", "games_last_14_diff",
    "h2h_total_games", "three_pt_regression_diff", "games_diff",
    "ref_foul_proxy", "roll_fast_break_diff", "crowd_pct",
    "matchup_to", "overround", "spread_juice_imbalance",
    "vig_uncertainty", "roll_ft_trip_rate_diff", "home_after_loss",
    "rest_diff",
]

# ═══════════════════════════════════════════════════════════
# CANDIDATE FEATURES (from v20 builder, not in v27)
# ═══════════════════════════════════════════════════════════

V20_CANDIDATES = [
    "ppg_diff", "fgpct_diff", "assists_diff", "turnovers_diff",
    "blocks_diff", "steals_diff", "orb_pct_diff", "fta_rate_diff",
    "ato_ratio_diff", "opp_fgpct_diff", "opp_threepct_diff",
    "to_margin_diff", "form_diff", "tempo_avg",
    "away_travel", "market_spread", "market_total",
    "spread_vs_market", "has_market", "score_diff_pred",
    "total_pred", "home_fav", "win_pct_home",
]


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("  NBA ATS FEATURE VALIDATION — v27 Lasso (38 features)")
print("=" * 70)

print("\nLoading data...")
rows = sb_get("nba_predictions",
              "result_entered=eq.true&actual_home_score=not.is.null&select=*")
current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
df, sample_weights, n_historical = _nba_merge_historical(current_df)
print(f"  {len(df)} games loaded ({n_historical} historical + {len(current_df)} current)")

# Build v20 features (superset that contains both v27 and candidates)
print("Building features...")
X_full = nba_build_features(df)

# Target
y_margin = (df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)).values

# Market spread (for ATS evaluation)
market_spread = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce").fillna(0).values
has_market = np.abs(market_spread) > 0.1

# Season for walk-forward splits
season = pd.to_numeric(df.get("season", pd.Series(dtype=float)), errors="coerce").fillna(2025).astype(int).values

# Weights
weights = sample_weights if sample_weights is not None else np.ones(len(df))

print(f"  Features available: {len(X_full.columns)}")
print(f"  Games with market spread: {has_market.sum()}/{len(df)}")
print(f"  Seasons: {sorted(set(season))}")


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════

def walk_forward_ats(X, y, spread, has_mkt, seasons, w, alpha=0.1, min_train=500):
    """
    Walk-forward Lasso: train on seasons < S, test on season S.
    Returns dict with MAE, ATS metrics.
    """
    unique_seasons = sorted(set(seasons))
    all_pred = np.full(len(y), np.nan)

    for test_season in unique_seasons:
        train_mask = seasons < test_season
        test_mask = seasons == test_season

        if train_mask.sum() < min_train or test_mask.sum() < 10:
            continue

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr = y[train_mask]
        w_tr = w[train_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_tr_s, y_tr, sample_weight=w_tr)

        all_pred[test_mask] = model.predict(X_te_s)

    # Evaluate only where we have predictions AND market spread
    valid = ~np.isnan(all_pred) & has_mkt
    if valid.sum() < 50:
        return None

    pred_v = all_pred[valid]
    actual_v = y[valid]
    spread_v = spread[valid]

    mae = float(np.mean(np.abs(pred_v - actual_v)))

    # ATS: model picks home to cover if pred_margin > -spread
    model_home_cover = pred_v > -spread_v
    actual_ats = actual_v + spread_v
    actual_home_cover = actual_ats > 0
    non_push = actual_ats != 0

    ats_correct = (model_home_cover[non_push] == actual_home_cover[non_push])
    ats_pct = float(ats_correct.mean()) if len(ats_correct) > 0 else 0.5
    ats_n = int(non_push.sum())

    # ATS by edge size (model disagree with market)
    edge = np.abs(pred_v - (-spread_v))
    results = {
        "mae": mae,
        "ats": ats_pct,
        "ats_n": ats_n,
        "n_predicted": int(valid.sum()),
    }

    for threshold in [4, 7, 10]:
        mask_t = non_push & (edge >= threshold)
        if mask_t.sum() >= 20:
            ats_t = float((model_home_cover[mask_t] == actual_home_cover[mask_t]).mean())
            results[f"ats_{threshold}u"] = ats_t
            results[f"ats_{threshold}u_n"] = int(mask_t.sum())
        else:
            results[f"ats_{threshold}u"] = None
            results[f"ats_{threshold}u_n"] = 0

    return results


# ═══════════════════════════════════════════════════════════
# 1. BASELINE (all 38 v27 features)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PHASE 1: BASELINE — all 38 v27 features")
print("=" * 70)

# Select v27 features that exist in v20 builder output
v27_available = [f for f in V27_FEATURES if f in X_full.columns]
v27_missing = [f for f in V27_FEATURES if f not in X_full.columns]
if v27_missing:
    print(f"\n  WARNING: {len(v27_missing)} v27 features missing from v20 builder:")
    for f in v27_missing:
        print(f"    - {f}")
    print("  These will be zeroed (matching live behavior when data missing)")
    # Add missing features as zeros
    for f in v27_missing:
        X_full[f] = 0.0
    v27_available = V27_FEATURES

X_v27 = X_full[v27_available].fillna(0).values

baseline = walk_forward_ats(X_v27, y_margin, market_spread, has_market, season, weights)
if baseline:
    print(f"\n  Baseline Results:")
    print(f"    MAE:       {baseline['mae']:.3f}")
    print(f"    ATS:       {baseline['ats']:.1%} ({baseline['ats_n']} games)")
    for t in [4, 7, 10]:
        v = baseline.get(f"ats_{t}u")
        n = baseline.get(f"ats_{t}u_n", 0)
        if v is not None:
            roi = (v * 2 - 1) * 100  # simplified ROI
            print(f"    ATS {t}+u:   {v:.1%} ({n} games, ~{roi:+.1f}% ROI)")
else:
    print("  ERROR: Baseline failed (not enough data)")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# 2. DROP-ONE ANALYSIS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PHASE 2: DROP-ONE — impact of removing each feature")
print("=" * 70)

drop_results = []
t0 = time.time()

for i, feat in enumerate(v27_available):
    remaining = [f for f in v27_available if f != feat]
    X_drop = X_full[remaining].fillna(0).values
    r = walk_forward_ats(X_drop, y_margin, market_spread, has_market, season, weights)
    if r:
        delta_ats = r["ats"] - baseline["ats"]
        delta_mae = r["mae"] - baseline["mae"]
        drop_results.append({
            "feature": feat,
            "ats_without": r["ats"],
            "delta_ats": delta_ats,
            "mae_without": r["mae"],
            "delta_mae": delta_mae,
            "ats_4u": r.get("ats_4u"),
        })
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(v27_available) - i - 1)
        print(f"  [{i+1}/{len(v27_available)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

# Sort by ATS impact (most harmful to remove = most valuable)
drop_results.sort(key=lambda x: x["delta_ats"])

print(f"\n  {'Feature':<30} {'ATS w/o':>8} {'ΔATS':>8} {'MAE w/o':>8} {'ΔMAE':>8} {'Verdict':>10}")
print("  " + "-" * 85)
for r in drop_results:
    delta = r["delta_ats"]
    if delta < -0.005:
        verdict = "KEEP ✓"
    elif delta > 0.005:
        verdict = "DROP ✗"
    else:
        verdict = "neutral"
    print(f"  {r['feature']:<30} {r['ats_without']:>7.1%} {delta:>+7.1%} "
          f"{r['mae_without']:>8.3f} {r['delta_mae']:>+7.3f}  {verdict}")

# Identify features to drop
droppers = [r for r in drop_results if r["delta_ats"] > 0.005]
keepers = [r for r in drop_results if r["delta_ats"] < -0.005]
neutral = [r for r in drop_results if -0.005 <= r["delta_ats"] <= 0.005]

print(f"\n  Summary:")
print(f"    Must-keep ({len(keepers)}): removing hurts ATS by >0.5%")
print(f"    Neutral   ({len(neutral)}): <0.5% impact either way")
print(f"    Droppable  ({len(droppers)}): removing IMPROVES ATS by >0.5%")

if droppers:
    print(f"\n  Features that HURT ATS (removing improves it):")
    for r in droppers:
        print(f"    - {r['feature']} → +{r['delta_ats']:.1%} ATS without it")


# ═══════════════════════════════════════════════════════════
# 3. CUMULATIVE DROP (remove all harmful features at once)
# ═══════════════════════════════════════════════════════════

if droppers:
    print("\n" + "=" * 70)
    print("  PHASE 3: CUMULATIVE DROP — remove all harmful features together")
    print("=" * 70)

    drop_names = [r["feature"] for r in droppers]
    remaining = [f for f in v27_available if f not in drop_names]
    X_slim = X_full[remaining].fillna(0).values
    slim_result = walk_forward_ats(X_slim, y_margin, market_spread, has_market, season, weights)
    if slim_result:
        print(f"\n  Dropped: {', '.join(drop_names)}")
        print(f"  Remaining: {len(remaining)} features")
        print(f"  ATS:  {slim_result['ats']:.1%} (was {baseline['ats']:.1%}, Δ{slim_result['ats']-baseline['ats']:+.1%})")
        print(f"  MAE:  {slim_result['mae']:.3f} (was {baseline['mae']:.3f}, Δ{slim_result['mae']-baseline['mae']:+.3f})")
        for t in [4, 7, 10]:
            v = slim_result.get(f"ats_{t}u")
            bv = baseline.get(f"ats_{t}u")
            n = slim_result.get(f"ats_{t}u_n", 0)
            if v is not None and bv is not None:
                print(f"  ATS {t}+u: {v:.1%} (was {bv:.1%}, Δ{v-bv:+.1%}, {n} games)")


# ═══════════════════════════════════════════════════════════
# 4. ADD-BACK ANALYSIS (test v20 candidates not in v27)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PHASE 4: ADD-BACK — test v20 candidates not in v27")
print("=" * 70)

candidates_available = [f for f in V20_CANDIDATES if f in X_full.columns and f not in v27_available]
print(f"\n  Testing {len(candidates_available)} candidate features...")

add_results = []
for feat in candidates_available:
    test_feats = v27_available + [feat]
    X_add = X_full[test_feats].fillna(0).values
    r = walk_forward_ats(X_add, y_margin, market_spread, has_market, season, weights)
    if r:
        delta_ats = r["ats"] - baseline["ats"]
        delta_mae = r["mae"] - baseline["mae"]
        add_results.append({
            "feature": feat,
            "ats_with": r["ats"],
            "delta_ats": delta_ats,
            "mae_with": r["mae"],
            "delta_mae": delta_mae,
        })

add_results.sort(key=lambda x: -x["delta_ats"])

print(f"\n  {'Candidate':<30} {'ATS w/':>8} {'ΔATS':>8} {'MAE w/':>8} {'ΔMAE':>8} {'Verdict':>10}")
print("  " + "-" * 85)
for r in add_results:
    delta = r["delta_ats"]
    if delta > 0.005:
        verdict = "ADD ✓"
    elif delta < -0.005:
        verdict = "skip ✗"
    else:
        verdict = "neutral"
    print(f"  {r['feature']:<30} {r['ats_with']:>7.1%} {delta:>+7.1%} "
          f"{r['mae_with']:>8.3f} {r['delta_mae']:>+7.3f}  {verdict}")

adders = [r for r in add_results if r["delta_ats"] > 0.005]
if adders:
    print(f"\n  Features worth ADDING (>0.5% ATS improvement):")
    for r in adders:
        print(f"    + {r['feature']} → +{r['delta_ats']:.1%} ATS")


# ═══════════════════════════════════════════════════════════
# 5. OPTIMAL SET (drop harmful + add beneficial)
# ═══════════════════════════════════════════════════════════

if droppers or adders:
    print("\n" + "=" * 70)
    print("  PHASE 5: OPTIMAL SET — drop harmful + add beneficial")
    print("=" * 70)

    drop_names = [r["feature"] for r in droppers] if droppers else []
    add_names = [r["feature"] for r in adders] if adders else []

    optimal_feats = [f for f in v27_available if f not in drop_names] + add_names
    X_opt = X_full[optimal_feats].fillna(0).values
    opt_result = walk_forward_ats(X_opt, y_margin, market_spread, has_market, season, weights)

    if opt_result:
        print(f"\n  Optimal set: {len(optimal_feats)} features")
        if drop_names:
            print(f"  Dropped: {', '.join(drop_names)}")
        if add_names:
            print(f"  Added:   {', '.join(add_names)}")
        print(f"\n  ATS:  {opt_result['ats']:.1%} (baseline {baseline['ats']:.1%}, Δ{opt_result['ats']-baseline['ats']:+.1%})")
        print(f"  MAE:  {opt_result['mae']:.3f} (baseline {baseline['mae']:.3f}, Δ{opt_result['mae']-baseline['mae']:+.3f})")
        for t in [4, 7, 10]:
            v = opt_result.get(f"ats_{t}u")
            bv = baseline.get(f"ats_{t}u")
            n = opt_result.get(f"ats_{t}u_n", 0)
            if v is not None and bv is not None:
                print(f"  ATS {t}+u: {v:.1%} (was {bv:.1%}, Δ{v-bv:+.1%}, {n} games)")

        print(f"\n  OPTIMAL FEATURE LIST ({len(optimal_feats)}):")
        for f in sorted(optimal_feats):
            print(f"    \"{f}\",")

print(f"\n{'=' * 70}")
print(f"  DONE — {time.time() - t0:.0f}s total")
print(f"{'=' * 70}")
