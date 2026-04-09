#!/usr/bin/env python3
"""
mlb_v9_deploy.py — Production MLB v9 deploy
============================================
Architecture: CatBoost(15 feats, d8/i300/lr0.03) × 0.8 + Lasso(10 feats, α=0.01) × 0.2
Training: 2022+ only (pre-COVID data hurts accuracy)
Tiers: 1u = edge ≥ 2.0 + agree (74.8% ATS)
       2u = edge ≥ 2.5 + agree (76.5% ATS)
Sigma: 4.0 (Brier-optimal)

Usage:
  python mlb_v9_deploy.py            # validate only
  python mlb_v9_deploy.py --upload   # train + upload to Supabase
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from catboost import CatBoostRegressor
from datetime import datetime, timezone

SEED = 42
np.random.seed(SEED)
N_FOLDS = 30

CB_WEIGHT = 0.8
LASSO_WEIGHT = 0.2

# CatBoost config (calibrated)
CB_PARAMS = dict(depth=8, iterations=300, learning_rate=0.03,
                 subsample=0.8, min_data_in_leaf=20, random_seed=SEED, verbose=0)

# Lasso config
LASSO_ALPHA = 0.01

# Unit tiers
TIERS = {
    1: {"edge_min": 2.0, "require_agree": True},
    2: {"edge_min": 2.5, "require_agree": True},
}

# Feature sets
CB_FEATURES = [
    "k_bb_diff", "woba_x_park", "woba_diff", "market_spread", "sp_relative_fip_diff",
    "sp_ip_diff", "home_woba_vs_rolling", "away_woba_vs_rolling",
    "lineup_delta_sum", "sp_form_combined", "is_warm",
    "lineup_ops_diff", "lineup_delta_diff", "lineup_top_heavy_diff", "lineup_total_top3",
]

LASSO_FEATURES = [
    "k_bb_diff", "woba_x_park", "market_spread", "sp_relative_fip_diff", "sp_ip_diff",
    "lineup_delta_sum", "home_woba_vs_rolling", "away_woba_vs_rolling",
    "sp_form_combined", "lineup_woba_diff",
]

LINEUP_RAW = ["lineup_woba_diff", "lineup_ops_diff", "lineup_iso_diff", "top3_woba_diff"]
LINEUP_ADVANCED = [
    "lineup_delta_diff", "lineup_delta_sum",
    "home_woba_vs_rolling", "away_woba_vs_rolling",
    "lineup_bot3_diff", "lineup_top_heavy_diff", "lineup_consistency_diff",
    "lineup_trend_diff", "lineup_trend_sum",
    "lineup_total_woba", "lineup_total_iso", "lineup_total_top3",
]


def load_data():
    from mlb_retrain import load_data as _load, build_features

    df = _load()
    X = build_features(df)

    # Merge lineup features
    df["_key"] = df["game_date"].astype(str) + "|" + df["home_team"].astype(str)
    for src, cols in [("mlb_lineup_backfill.parquet", LINEUP_RAW),
                      ("mlb_lineup_features_advanced.parquet", LINEUP_ADVANCED)]:
        if not os.path.exists(src):
            for c in cols: X[c] = 0
            continue
        tmp = pd.read_parquet(src)
        tmp["_key"] = tmp["game_date"] + "|" + tmp["home_abbr"]
        avail = [c for c in cols if c in tmp.columns]
        if avail:
            sub = tmp[["_key"] + avail].drop_duplicates(subset="_key", keep="first")
            merged = df[["_key"]].merge(sub, on="_key", how="left")
            for c in avail:
                X[c] = merged[c].fillna(0).values

    y = (df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)).values
    sp = X["market_spread"].values if "market_spread" in X.columns else np.zeros(len(X))
    has_sp = np.abs(sp) > 0.1
    seasons = pd.to_numeric(df.get("season", 2026), errors="coerce").fillna(2026).astype(int).values
    w = np.ones(len(X))  # equal weights for 2022+ (all recent, no decay needed)

    # Filter to 2022+ only (pre-COVID data hurts accuracy)
    keep = seasons >= 2022
    X = X[keep].reset_index(drop=True)
    y = y[keep]; sp = sp[keep]; has_sp = has_sp[keep]; w = w[keep]; seasons = seasons[keep]
    df = df[keep].reset_index(drop=True)

    X = X.fillna(0)
    print(f"  Data: {len(X)} games (2022+ only)")
    return X, y, sp, has_sp, w, seasons, df


def validate(X, y, sp, has_sp, w):
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION ({N_FOLDS}-fold)")
    print(f"  CatBoost: {len(CB_FEATURES)} feats (d{CB_PARAMS['depth']}/i{CB_PARAMS['iterations']}/lr{CB_PARAMS['learning_rate']}) × {CB_WEIGHT}")
    print(f"  Lasso: {len(LASSO_FEATURES)} feats (α={LASSO_ALPHA}) × {LASSO_WEIGHT}")
    print(f"{'='*70}")

    n = len(X); fs = n // (N_FOLDS + 1); mt = max(fs * 3, 1000)
    oof_c = np.full(n, np.nan)
    oof_l = np.full(n, np.nan)

    for fold in range(N_FOLDS):
        ts = mt + fold * fs; te = min(ts + fs, n)
        if ts >= n: break

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[CB_FEATURES].values[:ts])
        Xte = sc.transform(X[CB_FEATURES].values[ts:te])
        mc = CatBoostRegressor(**CB_PARAMS)
        mc.fit(Xtr, y[:ts], sample_weight=w[:ts])
        oof_c[ts:te] = mc.predict(Xte)

        sc2 = StandardScaler()
        Xtr2 = sc2.fit_transform(X[LASSO_FEATURES].values[:ts])
        Xte2 = sc2.transform(X[LASSO_FEATURES].values[ts:te])
        ml = Lasso(alpha=LASSO_ALPHA, max_iter=5000, random_state=SEED)
        ml.fit(Xtr2, y[:ts])
        oof_l[ts:te] = ml.predict(Xte2)

        if (fold + 1) % 10 == 0:
            print(f"    Fold {fold+1}/{N_FOLDS}")

    blend = CB_WEIGHT * oof_c + LASSO_WEIGHT * oof_l
    agree = ((oof_c > 0) & (oof_l > 0)) | ((oof_c < 0) & (oof_l < 0))
    ats = y + sp; push = ats == 0
    valid = ~np.isnan(blend) & ~push & has_sp
    edge = np.abs(blend - (-sp))

    # ── Full blend breakdown ──
    print(f"\n  BLEND (CB×{CB_WEIGHT} + Lasso×{LASSO_WEIGHT}):")
    print(f"  {'Thresh':>7} {'Games':>6} {'ATS%':>6} {'ROI%':>7} {'ML%':>6}")
    print(f"  {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*6}")
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        mask = valid & (edge >= t)
        if mask.sum() < 25: continue
        ats_c = ((blend[mask] > (-sp[mask])) == (ats[mask] > 0)).mean()
        ml_c = ((blend[mask] > 0) == (y[mask] > 0)).mean()
        roi = (ats_c * 1.909 - 1) * 100
        print(f"  {t:>6}+ {mask.sum():>6} {ats_c:>5.1%} {roi:>+6.1f}% {ml_c:>5.1%}")

    # ── Tier performance with agreement ──
    print(f"\n  UNIT TIERS (agreement-gated):")
    print(f"  {'Tier':>5} {'Rule':>30} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
    print(f"  {'─'*5} {'─'*30} {'─'*6} {'─'*6} {'─'*7}")
    for units in [1, 2]:
        tier = TIERS[units]
        mask = valid & (edge >= tier["edge_min"])
        if tier["require_agree"]:
            mask = mask & agree
        if mask.sum() < 25: continue
        correct = ((blend[mask] > (-sp[mask])) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100
        rule = f"edge≥{tier['edge_min']:.1f} + agree"
        print(f"  {units:>4}u {rule:>30} {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")

    # ── Unit-weighted profit ──
    print(f"\n  UNIT-WEIGHTED PROFIT:")
    total_profit = 0; total_wagered = 0
    for units in [1, 2]:
        tier = TIERS[units]
        mask = valid & (edge >= tier["edge_min"]) & agree
        # Exclude picks in higher tier
        for higher_u in range(units + 1, 3):
            ht = TIERS.get(higher_u)
            if ht:
                higher_mask = (edge >= ht["edge_min"]) & agree
                mask = mask & ~higher_mask
        if mask.sum() == 0: continue
        correct = ((blend[mask] > (-sp[mask])) == (ats[mask] > 0))
        n_bets = mask.sum()
        profit_per = correct.mean() * 1.909 - 1
        tier_profit = profit_per * units * n_bets
        tier_wagered = units * n_bets
        total_profit += tier_profit
        total_wagered += tier_wagered
        print(f"    {units}u: {n_bets} bets × {units}u × {correct.mean():.1%} ATS = {tier_profit:+.1f} units profit")
    if total_wagered > 0:
        print(f"    TOTAL: {total_profit:+.1f} units profit on {total_wagered} wagered = {total_profit/total_wagered*100:+.1f}% ROI")

    return oof_c, oof_l, blend, agree


def train_and_upload(X, y, sp, w):
    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING")
    print(f"{'='*70}")

    # CatBoost
    sc_c = StandardScaler()
    Xc = sc_c.fit_transform(X[CB_FEATURES].values)
    m_c = CatBoostRegressor(**CB_PARAMS)
    m_c.fit(Xc, y, sample_weight=w)
    print(f"  CatBoost: {len(CB_FEATURES)} features (d{CB_PARAMS['depth']}/i{CB_PARAMS['iterations']})")

    # Feature importance
    imp = sorted(zip(CB_FEATURES, m_c.feature_importances_), key=lambda x: -x[1])
    print(f"\n  CatBoost feature importance:")
    for f, i in imp[:10]:
        print(f"    {f:35s} {i:>6.2f}")

    # Lasso
    sc_l = StandardScaler()
    Xl = sc_l.fit_transform(X[LASSO_FEATURES].values)
    m_l = Lasso(alpha=LASSO_ALPHA, max_iter=5000, random_state=SEED)
    m_l.fit(Xl, y, sample_weight=w)
    n_active = np.sum(np.abs(m_l.coef_) > 1e-6)
    print(f"\n  Lasso: {len(LASSO_FEATURES)} features ({n_active} active)")

    coefs = sorted(zip(LASSO_FEATURES, m_l.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Lasso coefficients:")
    for f, c in coefs[:10]:
        if abs(c) > 0.001:
            print(f"    {f:35s} {c:+.4f}")

    bundle = {
        "models": [m_l, m_c],
        "scalers": [sc_l, sc_c],
        "feature_sets": [LASSO_FEATURES, CB_FEATURES],
        "model_names": [f"Lasso_{LASSO_ALPHA}", f"CatBoost_d{CB_PARAMS['depth']}_i{CB_PARAMS['iterations']}"],
        "model_weights": [LASSO_WEIGHT, CB_WEIGHT],
        "tiers": TIERS,
        "cb_params": CB_PARAMS,
        "lasso_alpha": LASSO_ALPHA,
        "n_train": len(X),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "mlb_v9_cb_lasso",
        "architecture": f"CatBoost(d{CB_PARAMS['depth']}/i{CB_PARAMS['iterations']})×{CB_WEIGHT}+Lasso×{LASSO_WEIGHT}",
        "tier_rules": {
            "1u": "edge≥2.0 + both agree",
            "2u": "edge≥2.5 + both agree",
        },
        "sigma": 4.0,
        "lineup_features": True,
        "training_seasons": "2022+",
    }

    from db import save_model
    save_model("mlb_ats_v9", bundle)
    print(f"\n  ✅ Saved to Supabase as 'mlb_ats_v9'")
    print(f"  Architecture: CatBoost(d{CB_PARAMS['depth']}/i{CB_PARAMS['iterations']})×{CB_WEIGHT} + Lasso×{LASSO_WEIGHT}")
    print(f"  Tiers: 1u=edge≥2.0+agree | 2u=edge≥2.5+agree")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  MLB v9 DEPLOY (2022+ training)")
    print(f"  CatBoost(d8/i300/lr0.03) × {CB_WEIGHT} + Lasso × {LASSO_WEIGHT}")
    print(f"  1u: edge≥2.0+agree | 2u: edge≥2.5+agree")
    print("=" * 70)

    X, y, sp, has_sp, w, seasons, df = load_data()
    oof_c, oof_l, blend, agree = validate(X, y, sp, has_sp, w)

    if args.upload:
        train_and_upload(X, y, sp, w)

    print(f"\n  Done. Add --upload to deploy.")


if __name__ == "__main__":
    main()
