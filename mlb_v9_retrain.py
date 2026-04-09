#!/usr/bin/env python3
"""
mlb_v9_retrain.py — MLB v9 ensemble with lineup features
=========================================================
Adds lineup_woba_diff, lineup_ops_diff, lineup_iso_diff, top3_woba_diff
to the existing v8 ensemble (Lasso + ElasticNet + CatBoost).

Usage:
  python mlb_v9_retrain.py                # compare v8 vs v9
  python mlb_v9_retrain.py --select       # feature selection with lineup features
  python mlb_v9_retrain.py --upload       # train + upload to Supabase
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)
N_FOLDS = 20

# v8 features (baseline)
V8_FEATURES = [
    "woba_diff", "fip_diff", "k_bb_diff", "bullpen_era_diff",
    "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff", "sp_fip_spread",
    "sp_relative_fip_diff",
    "park_factor", "temp_f", "wind_mph", "wind_out", "is_warm", "is_cold",
    "temp_x_park",
    "rest_diff",
    "market_spread",
    "woba_x_park",
    "platoon_diff",
    "pyth_residual_diff", "scoring_entropy_diff",
    "first_inn_rate_diff", "clutch_divergence_diff", "opp_adj_form_diff",
    "ump_run_env", "series_game_num",
    "scoring_entropy_combined", "first_inn_rate_combined",
]

# v9 adds lineup features (raw + advanced)
LINEUP_RAW = [
    "lineup_woba_diff",
    "lineup_ops_diff",
    "lineup_iso_diff",
    "top3_woba_diff",
]

LINEUP_ADVANCED = [
    # CHANGE signals (key batter missing proxy)
    "lineup_delta_diff",       # r=0.034 with margin (ATS)
    "lineup_delta_sum",        # r=0.144 with total (O/U) ⭐
    "home_woba_vs_rolling",    # r=0.106 with total (O/U) ⭐
    "away_woba_vs_rolling",    # r=0.112 with total (O/U) ⭐
    # STRUCTURE signals
    "lineup_bot3_diff",        # r=0.074 with margin (ATS) ⭐
    "lineup_top_heavy_diff",
    "lineup_consistency_diff",
    # FORM signals
    "lineup_trend_diff",
    "lineup_trend_sum",
    # O/U totals
    "lineup_total_woba",
    "lineup_total_iso",
    "lineup_total_top3",
]

LINEUP_FEATURES = LINEUP_RAW + LINEUP_ADVANCED


def load_data():
    from mlb_retrain import load_data as _load, build_features

    df = _load()
    X = build_features(df)

    # ── Merge raw lineup features ──
    lineup_file = "mlb_lineup_backfill.parquet"
    if not os.path.exists(lineup_file):
        print(f"  ERROR: {lineup_file} not found")
        sys.exit(1)

    lineup = pd.read_parquet(lineup_file)
    print(f"  Lineup data: {len(lineup)} games")

    lineup["_key"] = lineup["game_date"] + "|" + lineup["home_abbr"]
    df["_key"] = df["game_date"].astype(str) + "|" + df["home_team"].astype(str)

    raw_cols = [c for c in LINEUP_RAW if c in lineup.columns]
    if raw_cols:
        lineup_sub = lineup[["_key"] + raw_cols].drop_duplicates(subset="_key", keep="first")
        merged = df[["_key"]].merge(lineup_sub, on="_key", how="left")
        for col in raw_cols:
            X[col] = merged[col].fillna(0).values
        print(f"  Raw lineup matched: {merged[raw_cols[0]].notna().sum()}/{len(df)}")

    # ── Merge advanced lineup features ──
    adv_file = "mlb_lineup_features_advanced.parquet"
    if os.path.exists(adv_file):
        adv = pd.read_parquet(adv_file)
        adv["_key"] = adv["game_date"] + "|" + adv["home_abbr"]
        adv_cols = [c for c in LINEUP_ADVANCED if c in adv.columns]
        if adv_cols:
            adv_sub = adv[["_key"] + adv_cols].drop_duplicates(subset="_key", keep="first")
            merged_adv = df[["_key"]].merge(adv_sub, on="_key", how="left")
            for col in adv_cols:
                X[col] = merged_adv[col].fillna(0).values
            print(f"  Advanced lineup matched: {merged_adv[adv_cols[0]].notna().sum()}/{len(df)}")
    else:
        print(f"  WARNING: {adv_file} not found — advanced features zero")
        for col in LINEUP_ADVANCED:
            X[col] = 0

    y = (df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)).values
    spreads = X["market_spread"].values if "market_spread" in X.columns else np.zeros(len(X))
    has_spread = np.abs(spreads) > 0.1

    seasons = pd.to_numeric(df.get("season", 2026), errors="coerce").fillna(2026).astype(int).values
    w = np.array([{0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8}.get(2026 - s, 0.7) for s in seasons])

    return X, y, spreads, has_spread, w, seasons, df


def walk_forward(X_df, y, w, features, n_folds=N_FOLDS, label=""):
    """3-model ensemble walk-forward. Returns per-model + ensemble OOF."""
    X = X_df[features].values
    n = len(X); fs = n // (n_folds + 1); mt = max(fs * 3, 1000)

    oof_l = np.full(n, np.nan)
    oof_en = np.full(n, np.nan)
    oof_cb = np.full(n, np.nan)

    for fold in range(n_folds):
        ts = mt + fold * fs; te = min(ts + fs, n)
        if ts >= n: break
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:ts]); Xte = sc.transform(X[ts:te])

        m1 = Lasso(alpha=0.01, max_iter=5000, random_state=SEED)
        m1.fit(Xtr, y[:ts]); oof_l[ts:te] = m1.predict(Xte)

        m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=SEED)
        m2.fit(Xtr, y[:ts]); oof_en[ts:te] = m2.predict(Xte)

        m3 = CatBoostRegressor(depth=6, iterations=200, learning_rate=0.03,
                               subsample=0.8, min_data_in_leaf=20, random_seed=SEED, verbose=0)
        m3.fit(Xtr, y[:ts], sample_weight=w[:ts]); oof_cb[ts:te] = m3.predict(Xte)

        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{n_folds}")

    oof_avg = (oof_l + oof_en + oof_cb) / 3.0
    return oof_l, oof_en, oof_cb, oof_avg


def ats_eval(preds, y, spreads, has_spread, label=""):
    """Evaluate ATS at various edge thresholds."""
    valid = ~np.isnan(preds) & has_spread
    ats = y + spreads; push = ats == 0
    edge = np.abs(preds[valid] - (-spreads[valid]))

    print(f"\n  {label}")
    print(f"  {'Threshold':>10} {'Games':>7} {'ATS%':>7} {'ROI%':>7} {'ML%':>7}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    results = {}
    p = preds[valid]; yv = y[valid]; sv = spreads[valid]
    atsv = ats[valid]; pv = push[valid]
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        mask = ~pv & (np.abs(p - (-sv)) >= t)
        if mask.sum() < 30: continue
        model_home = p[mask] > (-sv[mask])
        actual_home = atsv[mask] > 0
        ats_correct = (model_home == actual_home).mean()
        ml_correct = ((p[mask] > 0) == (yv[mask] > 0)).mean()
        roi = (ats_correct * 1.909 - 1) * 100
        print(f"  {t:>9}+ {mask.sum():>7} {ats_correct:>6.1%} {roi:>+6.1f}% {ml_correct:>6.1%}")
        results[t] = {"n": int(mask.sum()), "ats": float(ats_correct), "roi": float(roi), "ml": float(ml_correct)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  MLB v9 RETRAIN — Lineup Features")
    print("=" * 70)

    X, y, spreads, has_spread, w, seasons, df = load_data()

    v8_feats = [f for f in V8_FEATURES if f in X.columns]
    v9_feats = v8_feats + [f for f in LINEUP_FEATURES if f in X.columns]

    lineup_coverage = (X[LINEUP_FEATURES].abs().sum(axis=1) > 0).mean()
    print(f"\n  v8 features: {len(v8_feats)}")
    print(f"  v9 features: {len(v9_feats)} (+{len(v9_feats)-len(v8_feats)} lineup)")
    print(f"  Lineup feature coverage: {lineup_coverage:.1%}")
    print(f"  Lineup feature means:")
    for f in LINEUP_FEATURES:
        if f in X.columns:
            nz = (X[f].abs() > 0.001).sum()
            print(f"    {f:25s} mean={X[f].mean():+.4f} std={X[f].std():.4f} nonzero={nz}")

    # ── v8 baseline ──
    print(f"\n{'='*70}")
    print(f"  v8 BASELINE ({len(v8_feats)} features)")
    print(f"{'='*70}")
    l8, en8, cb8, avg8 = walk_forward(X, y, w, v8_feats)
    r8 = ats_eval(avg8, y, spreads, has_spread, "v8 Ensemble (Lasso+EN+CatBoost)")

    # ── v9 with lineup ──
    print(f"\n{'='*70}")
    print(f"  v9 WITH LINEUP ({len(v9_feats)} features)")
    print(f"{'='*70}")
    l9, en9, cb9, avg9 = walk_forward(X, y, w, v9_feats)
    r9 = ats_eval(avg9, y, spreads, has_spread, "v9 Ensemble (+ lineup features)")

    # ── v9 on lineup-available games only ──
    lineup_mask = X[LINEUP_FEATURES[0]].abs() > 0.001
    print(f"\n{'='*70}")
    print(f"  v9 ON LINEUP-AVAILABLE GAMES ONLY ({lineup_mask.sum()} games)")
    print(f"{'='*70}")
    ats_eval(avg8, y, spreads, has_spread & lineup_mask, "v8 (on lineup games)")
    ats_eval(avg9, y, spreads, has_spread & lineup_mask, "v9 (on lineup games)")

    # ── Comparison ──
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Tier':>6} {'v8 ATS':>10} {'v9 ATS':>10} {'Lift':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*8}")
    for t in [0.5, 1.0, 1.5, 2.0]:
        a8 = r8.get(t, {}).get("ats", 0)
        a9 = r9.get(t, {}).get("ats", 0)
        n8 = r8.get(t, {}).get("n", 0)
        n9 = r9.get(t, {}).get("n", 0)
        lift = (a9 - a8) * 100
        print(f"  {t:>5}+ {a8:>8.1%}/{n8:<4} {a9:>8.1%}/{n9:<4} {lift:>+6.1f}%")

    # ── Per-model lineup feature importance (CatBoost) ──
    print(f"\n  CatBoost feature importance (v9):")
    sc = StandardScaler()
    Xall = sc.fit_transform(X[v9_feats].values)
    cb_full = CatBoostRegressor(depth=6, iterations=200, learning_rate=0.03,
                                subsample=0.8, min_data_in_leaf=20, random_seed=SEED, verbose=0)
    cb_full.fit(Xall, y, sample_weight=w)
    imp = sorted(zip(v9_feats, cb_full.feature_importances_), key=lambda x: -x[1])
    for f, i in imp[:15]:
        marker = " ← LINEUP" if f in LINEUP_FEATURES else ""
        print(f"    {f:30s} {i:>6.2f}{marker}")

    if args.upload:
        print(f"\n  --upload: TODO — implement production training + upload")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
