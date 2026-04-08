#!/usr/bin/env python3
"""
nba_v28_deploy.py — Train and deploy NBA v28 Residual ATS model
================================================================
Architecture: CatBoost (19 features) × 0.7 + Lasso (19 features) × 0.3
Unit sizing:
  1u: blend ≥ 3
  2u: blend ≥ 4 AND both models agree on direction
  3u: blend ≥ 5 AND both models agree on direction

Run:
  python nba_v28_deploy.py                # validate only
  python nba_v28_deploy.py --upload       # train + upload to Supabase
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from catboost import CatBoostRegressor
from datetime import datetime, timezone

SEED = 42
np.random.seed(SEED)
N_FOLDS = 25

CB_WEIGHT = 0.7
LASSO_WEIGHT = 0.3

# Unit sizing: blend threshold + agreement gate
TIERS = {
    1: {"blend_min": 3.0, "require_agree": False},
    2: {"blend_min": 4.0, "require_agree": True},
    3: {"blend_min": 5.0, "require_agree": True},
}


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

def load_data():
    from nba_build_features_v27 import load_training_data, build_features

    df = load_training_data()
    X_all, _ = build_features(df)

    pf = pd.read_parquet("nba_player_features_for_training.parquet")
    pf_cols = [c for c in pf.columns if c not in ["game_id", "game_date", "home_team", "away_team", "_match_key"]]
    pf = pf.drop_duplicates(subset="_match_key", keep="first")
    df["_match_key"] = (pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d") + "|" +
                        df.get("home_team", df.get("home_team_name", "")).astype(str))
    merged = df[["_match_key"]].merge(pf[["_match_key"] + pf_cols], on="_match_key", how="left").iloc[:len(X_all)]
    for c in pf_cols:
        X_all[c] = merged[c].fillna(0).values

    am = (df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)).values
    sp = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values
    hs = np.abs(sp) > 0.1; yr = am + sp
    seasons = pd.to_numeric(df.get("season", 2026), errors="coerce").fillna(2026).astype(int).values
    w = np.array([{0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8}.get(2026 - s, 0.7) for s in seasons])

    idx = pd.to_datetime(df["game_date"], errors="coerce").argsort()
    X_all = X_all.iloc[idx].reset_index(drop=True)
    am = am[idx]; sp = sp[idx]; hs = hs[idx]; yr = yr[idx]; w = w[idx]

    m = hs
    X = X_all[m].reset_index(drop=True); am = am[m]; sp = sp[m]; yr = yr[m]; w = w[m]

    exclude = {"market_spread", "spread_vs_market", "score_diff_pred", "total_pred", "home_fav", "has_market"}
    cands = [f for f in X.columns if f not in exclude]
    X = X[cands].fillna(0)

    print(f"  Data: {len(X)} games with spreads")
    return X, am, sp, yr, w


def load_feats(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════

def validate(X, lf, cf, yr, w, am, sp):
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION ({N_FOLDS}-fold)")
    print(f"  CatBoost: {len(cf)} features × {CB_WEIGHT}")
    print(f"  Lasso:    {len(lf)} features × {LASSO_WEIGHT}")
    print(f"{'='*70}")

    n = len(X); fs = n // (N_FOLDS + 1); mt = max(fs * 3, 500)
    oof_l = np.full(n, np.nan)
    oof_c = np.full(n, np.nan)

    for fold in range(N_FOLDS):
        ts = mt + fold * fs; te = min(ts + fs, n)
        if ts >= n: break

        # Lasso
        sc_l = StandardScaler()
        Xtr_l = sc_l.fit_transform(X[lf].values[:ts])
        Xte_l = sc_l.transform(X[lf].values[ts:te])
        ml = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        ml.fit(Xtr_l, yr[:ts], sample_weight=w[:ts])
        oof_l[ts:te] = ml.predict(Xte_l)

        # CatBoost
        sc_c = StandardScaler()
        Xtr_c = sc_c.fit_transform(X[cf].values[:ts])
        Xte_c = sc_c.transform(X[cf].values[ts:te])
        mc = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                               l2_leaf_reg=5, random_seed=SEED, verbose=0)
        mc.fit(Xtr_c, yr[:ts], sample_weight=w[:ts])
        oof_c[ts:te] = mc.predict(Xte_c)

        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{N_FOLDS}")

    blend = CB_WEIGHT * oof_c + LASSO_WEIGHT * oof_l
    agree = ((oof_c > 0) & (oof_l > 0)) | ((oof_c < 0) & (oof_l < 0))
    ats = am + sp; push = ats == 0

    # ── Raw blend accuracy ──
    print(f"\n  BLEND ({CB_WEIGHT:.0%} CB / {LASSO_WEIGHT:.0%} Lasso):")
    print(f"  {'Thresh':>7} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
    print(f"  {'─'*7} {'─'*6} {'─'*6} {'─'*7}")
    for t in [0, 1, 2, 3, 4, 5, 7, 10]:
        mask = ~np.isnan(blend) & ~push & (np.abs(blend) >= t)
        if mask.sum() < 20: continue
        correct = ((blend[mask] > 0) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100
        print(f"  {t:>6}+ {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")

    # ── Unit-sized tier performance ──
    print(f"\n  UNIT-SIZED TIERS:")
    print(f"  {'Tier':>5} {'Rule':>35} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
    print(f"  {'─'*5} {'─'*35} {'─'*6} {'─'*6} {'─'*7}")

    for units in [1, 2, 3]:
        tier = TIERS[units]
        mask = ~np.isnan(blend) & ~push & (np.abs(blend) >= tier["blend_min"])
        if tier["require_agree"]:
            mask = mask & agree
        if mask.sum() < 20:
            continue
        correct = ((blend[mask] > 0) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100
        rule = f"blend≥{tier['blend_min']:.0f}" + (" + agree" if tier["require_agree"] else "")
        print(f"  {units:>4}u {rule:>35} {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")

    # ── Effective unit-weighted ROI ──
    print(f"\n  EFFECTIVE ROI (unit-weighted):")
    total_units = 0; total_profit = 0
    for units in [1, 2, 3]:
        tier = TIERS[units]
        # This tier ONLY (not overlapping with higher tiers)
        mask = ~np.isnan(blend) & ~push & (np.abs(blend) >= tier["blend_min"])
        if tier["require_agree"]:
            mask = mask & agree
        # Exclude picks that qualify for a higher tier
        for higher_u in range(units + 1, 4):
            ht = TIERS[higher_u]
            higher_mask = np.abs(blend) >= ht["blend_min"]
            if ht["require_agree"]:
                higher_mask = higher_mask & agree
            mask = mask & ~higher_mask
        if mask.sum() == 0:
            continue
        correct = ((blend[mask] > 0) == (ats[mask] > 0)).mean()
        n_bets = mask.sum()
        profit_per_bet = correct * 1.909 - 1
        tier_profit = profit_per_bet * units * n_bets
        tier_units = units * n_bets
        total_profit += tier_profit
        total_units += tier_units
        print(f"    {units}u: {n_bets} bets × {units}u × {correct:.1%} ATS = {tier_profit:+.1f} units profit")

    if total_units > 0:
        print(f"    TOTAL: {total_profit:+.1f} units profit on {total_units} units wagered = {total_profit/total_units*100:+.1f}% ROI")

    # ── Asymmetric home/away ──
    print(f"\n  ASYMMETRIC (at 2u tier: blend≥4 + agree):")
    for label, cond in [("HOME cover", blend > TIERS[2]["blend_min"]),
                         ("AWAY cover", blend < -TIERS[2]["blend_min"])]:
        mask = ~np.isnan(blend) & ~push & cond & agree
        if mask.sum() < 20: continue
        if "HOME" in label:
            correct = (ats[mask] > 0).mean()
        else:
            correct = (ats[mask] < 0).mean()
        roi = (correct * 1.909 - 1) * 100
        print(f"    {label}: {correct:.1%} ATS, {roi:+.1f}% ROI ({mask.sum()} picks)")

    return oof_l, oof_c, blend, agree


# ═══════════════════════════════════════════════════════════
# PRODUCTION TRAINING + UPLOAD
# ═══════════════════════════════════════════════════════════

def train_and_upload(X, lf, cf, yr, w):
    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING")
    print(f"{'='*70}")

    # Lasso
    sc_l = StandardScaler()
    Xl = sc_l.fit_transform(X[lf].values)
    m_l = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
    m_l.fit(Xl, yr, sample_weight=w)
    n_active = np.sum(np.abs(m_l.coef_) > 1e-6)
    print(f"  Lasso: {len(lf)} features ({n_active} active)")

    # CatBoost
    sc_c = StandardScaler()
    Xc = sc_c.fit_transform(X[cf].values)
    m_c = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                            l2_leaf_reg=5, random_seed=SEED, verbose=0)
    m_c.fit(Xc, yr, sample_weight=w)
    print(f"  CatBoost: {len(cf)} features")

    # Top Lasso features
    coefs = sorted(zip(lf, m_l.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Lasso top coefficients:")
    for f, c in coefs[:10]:
        if abs(c) > 0.001:
            print(f"    {f:40s} {c:+.4f}")

    # CatBoost feature importance
    cb_imp = sorted(zip(cf, m_c.feature_importances_), key=lambda x: -x[1])
    print(f"\n  CatBoost top importance:")
    for f, imp in cb_imp[:10]:
        print(f"    {f:40s} {imp:.2f}")

    bundle = {
        "models": [m_l, m_c],
        "scalers": [sc_l, sc_c],
        "feature_sets": [lf, cf],
        "model_names": ["Lasso_0.1", "CatBoost_d4_300"],
        "model_weights": [LASSO_WEIGHT, CB_WEIGHT],
        "tiers": TIERS,
        "n_train": len(X),
        "n_active_lasso": int(n_active),
        "cb_weight": CB_WEIGHT,
        "lasso_weight": LASSO_WEIGHT,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "residual_ats_cb_lasso_v28",
        "architecture": f"CatBoost×{CB_WEIGHT}+Lasso×{LASSO_WEIGHT}, agreement-gated tiers",
        "tier_rules": {
            "1u": "blend≥3",
            "2u": "blend≥4 + both agree",
            "3u": "blend≥5 + both agree",
        },
    }

    from db import save_model
    save_model("nba_ats_residual", bundle)
    print(f"\n  ✅ Saved to Supabase as 'nba_ats_residual'")
    print(f"  Architecture: CatBoost×{CB_WEIGHT} + Lasso×{LASSO_WEIGHT}")
    print(f"  Tiers: 1u=blend≥3 | 2u=blend≥4+agree | 3u=blend≥5+agree")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA v28 RESIDUAL ATS — DEPLOY")
    print(f"  CatBoost × {CB_WEIGHT} + Lasso × {LASSO_WEIGHT}")
    print(f"  1u: blend≥3 | 2u: blend≥4+agree | 3u: blend≥5+agree")
    print("=" * 70)

    X, am, sp, yr, w = load_data()

    lf = load_feats("v28_lasso_features.txt")
    cf = load_feats("v28_catboost_features.txt")
    print(f"  Lasso features: {len(lf)}")
    print(f"  CatBoost features: {len(cf)}")

    oof_l, oof_c, blend, agree = validate(X, lf, cf, yr, w, am, sp)

    if args.upload:
        train_and_upload(X, lf, cf, yr, w)

    print(f"\n  Done. Add --upload to deploy.")


if __name__ == "__main__":
    main()
