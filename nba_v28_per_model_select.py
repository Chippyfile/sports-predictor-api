#!/usr/bin/env python3
"""
nba_v28_per_model_select.py — Independent feature selection per model
=====================================================================
Runs forward selection 3 times (Lasso, Ridge, CatBoost) independently.
Each model gets its OWN optimal feature set.
Final ensemble averages predictions where each model uses its own features.

Run:
  python nba_v28_per_model_select.py
  python nba_v28_per_model_select.py --upload
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse, json
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)
N_FOLDS = 15


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

def load_data():
    from nba_build_features_v27 import load_training_data, build_features

    df = load_training_data()
    X_all, all_feats = build_features(df)

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
    candidates = [f for f in X.columns if f not in exclude]
    X = X[candidates].fillna(0)

    print(f"  Data: {len(X)} games, {len(candidates)} candidate features")
    return X, candidates, am, sp, yr, w


# ═══════════════════════════════════════════════════════════
# SINGLE-MODEL WALK-FORWARD SCORE
# ═══════════════════════════════════════════════════════════

def make_model(name):
    if name == "Lasso":
        return Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
    elif name == "Ridge":
        return Ridge(alpha=1.0, random_state=SEED)
    elif name == "CatBoost":
        return CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                                 l2_leaf_reg=5, random_seed=SEED, verbose=0)
    raise ValueError(f"Unknown model: {name}")


def wf_score(model_name, Xv, yr, w, am, sp, n_folds=N_FOLDS):
    n = len(Xv); fs = n // (n_folds + 1); mt = max(fs * 3, 500)
    oof = np.full(n, np.nan)

    for fold in range(n_folds):
        ts = mt + fold * fs; te = min(ts + fs, n)
        if ts >= n: break
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xv[:ts]); Xte = sc.transform(Xv[ts:te])
        m = make_model(model_name)
        m.fit(Xtr, yr[:ts], sample_weight=w[:ts])
        oof[ts:te] = m.predict(Xte)

    ats = am + sp; push = ats == 0; r = {}
    for t in [0, 3, 4, 5, 7]:
        mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
        r[t] = ((oof[mask] > 0) == (ats[mask] > 0)).mean() if mask.sum() >= 20 else 0.5

    composite = r[4] * 0.4 + r[5] * 0.3 + r[7] * 0.3
    return composite, r


# ═══════════════════════════════════════════════════════════
# FORWARD SELECTION FOR ONE MODEL
# ═══════════════════════════════════════════════════════════

def forward_select_model(model_name, X, candidates, yr, w, am, sp):
    core = [
        "lineup_value_diff", "matchup_efg", "three_value_diff",
        "ref_foul_proxy", "scoring_hhi_diff", "threepct_diff",
        "rest_diff", "away_minutes_load", "roll_pts_off_to_diff",
    ]
    core = [f for f in core if f in candidates]
    remaining = [f for f in candidates if f not in core]

    print(f"\n  ── {model_name} FORWARD SELECTION ──")
    core_score, cd = wf_score(model_name, X[core].values, yr, w, am, sp)
    print(f"  Core ({len(core)}): {core_score:.5f} (4+={cd[4]:.1%} 5+={cd[5]:.1%} 7+={cd[7]:.1%})")

    selected = list(core)
    best_score = core_score
    added = []

    t0 = time.time()
    for i, feat in enumerate(remaining):
        trial = selected + [feat]
        score, detail = wf_score(model_name, X[trial].values, yr, w, am, sp)
        delta = score - best_score

        if delta > 0.00005:
            selected.append(feat)
            best_score = score
            added.append((feat, delta, detail))

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 1
        rem = (len(remaining) - i - 1) / rate / 60
        if (i + 1) % 15 == 0:
            print(f"    [{i+1:3d}/{len(remaining)}] {len(selected)} feats, score={best_score:.5f} ~{rem:.1f}m")

    print(f"\n  {model_name} RESULT: {len(selected)} features, score={best_score:.5f} ({best_score-core_score:+.5f})")
    print(f"  Added {len(added)} features:")
    for f, d, det in added:
        print(f"    {f:40s} d={d:+.6f} (5+={det[5]:.1%} 7+={det[7]:.1%})")

    return selected, added, best_score


# ═══════════════════════════════════════════════════════════
# MULTI-MODEL ENSEMBLE VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_independent_ensemble(X, lasso_feats, ridge_feats, cb_feats, yr, w, am, sp, n_folds=25):
    """Each model uses its OWN features. Predictions averaged."""
    print(f"\n{'='*70}")
    print(f"  INDEPENDENT ENSEMBLE VALIDATION (25-fold)")
    print(f"  Lasso: {len(lasso_feats)} features")
    print(f"  Ridge: {len(ridge_feats)} features")
    print(f"  CatBoost: {len(cb_feats)} features")
    print(f"{'='*70}")

    n = len(X); fs = n // (n_folds + 1); mt = max(fs * 3, 500)
    oof_l = np.full(n, np.nan)
    oof_r = np.full(n, np.nan)
    oof_c = np.full(n, np.nan)

    for fold in range(n_folds):
        ts = mt + fold * fs; te = min(ts + fs, n)
        if ts >= n: break

        # Lasso (own features)
        sc_l = StandardScaler()
        Xtr_l = sc_l.fit_transform(X[lasso_feats].values[:ts])
        Xte_l = sc_l.transform(X[lasso_feats].values[ts:te])
        m_l = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        m_l.fit(Xtr_l, yr[:ts], sample_weight=w[:ts])
        oof_l[ts:te] = m_l.predict(Xte_l)

        # Ridge (own features)
        sc_r = StandardScaler()
        Xtr_r = sc_r.fit_transform(X[ridge_feats].values[:ts])
        Xte_r = sc_r.transform(X[ridge_feats].values[ts:te])
        m_r = Ridge(alpha=1.0, random_state=SEED)
        m_r.fit(Xtr_r, yr[:ts], sample_weight=w[:ts])
        oof_r[ts:te] = m_r.predict(Xte_r)

        # CatBoost (own features)
        sc_c = StandardScaler()
        Xtr_c = sc_c.fit_transform(X[cb_feats].values[:ts])
        Xte_c = sc_c.transform(X[cb_feats].values[ts:te])
        m_c = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                                l2_leaf_reg=5, random_seed=SEED, verbose=0)
        m_c.fit(Xtr_c, yr[:ts], sample_weight=w[:ts])
        oof_c[ts:te] = m_c.predict(Xte_c)

        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{n_folds}")

    # Individual model results
    ats = am + sp; push = ats == 0

    for label, oof in [("Lasso", oof_l), ("Ridge", oof_r), ("CatBoost", oof_c)]:
        print(f"\n  {label} (solo):")
        print(f"  {'Thresh':>7} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
        for t in [0, 3, 5, 7, 10]:
            mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
            if mask.sum() < 20: continue
            correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
            roi = (correct * 1.909 - 1) * 100
            print(f"  {t:>6}+ {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")

    # Ensemble: average all 3
    oof_ens = np.nanmean([oof_l, oof_r, oof_c], axis=0)

    print(f"\n  INDEPENDENT ENSEMBLE (each model uses own features):")
    print(f"  {'Thresh':>7} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
    print(f"  {'─'*7} {'─'*6} {'─'*6} {'─'*7}")
    results = {}
    for t in [0, 1, 2, 3, 4, 5, 7, 10]:
        mask = ~np.isnan(oof_ens) & ~push & (np.abs(oof_ens) >= t)
        if mask.sum() < 20: continue
        correct = ((oof_ens[mask] > 0) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100
        print(f"  {t:>6}+ {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")
        results[t] = {"n": int(mask.sum()), "acc": round(float(correct), 4)}

    # Tier calibration
    print(f"\n  TIER CALIBRATION:")
    tiers = {}
    for t in np.arange(0.5, 12.5, 0.5):
        mask = ~np.isnan(oof_ens) & ~push & (np.abs(oof_ens) >= t)
        if mask.sum() < 20: continue
        correct = ((oof_ens[mask] > 0) == (ats[mask] > 0)).mean()
        if correct >= 0.65 and 3 not in tiers:
            tiers[3] = t
        elif correct >= 0.60 and 2 not in tiers:
            tiers[2] = t
        elif correct >= 0.55 and 1 not in tiers:
            tiers[1] = t

    for u in [1, 2, 3]:
        if u in tiers:
            t = tiers[u]
            mask = ~np.isnan(oof_ens) & ~push & (np.abs(oof_ens) >= t)
            correct = ((oof_ens[mask] > 0) == (ats[mask] > 0)).mean()
            print(f"    {u}u: residual ≥ {t:.1f} → {correct:.1%} ATS ({mask.sum()} picks)")

    # Feature overlap analysis
    all_feats = set(lasso_feats) | set(ridge_feats) | set(cb_feats)
    common = set(lasso_feats) & set(ridge_feats) & set(cb_feats)
    print(f"\n  FEATURE OVERLAP:")
    print(f"    Lasso: {len(lasso_feats)}, Ridge: {len(ridge_feats)}, CatBoost: {len(cb_feats)}")
    print(f"    Union: {len(all_feats)}, All-3 intersection: {len(common)}")
    print(f"    Common: {sorted(common)}")

    return oof_ens, tiers, oof_l, oof_r, oof_c


# ═══════════════════════════════════════════════════════════
# TRAIN & UPLOAD
# ═══════════════════════════════════════════════════════════

def train_and_upload(X, lasso_feats, ridge_feats, cb_feats, yr, w, tiers):
    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING — Independent Feature Sets")
    print(f"{'='*70}")

    # Lasso
    sc_l = StandardScaler()
    Xl = sc_l.fit_transform(X[lasso_feats].values)
    m_l = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
    m_l.fit(Xl, yr, sample_weight=w)
    n_active = np.sum(np.abs(m_l.coef_) > 1e-6)
    print(f"  Lasso: {len(lasso_feats)} features ({n_active} active)")

    # Ridge
    sc_r = StandardScaler()
    Xr = sc_r.fit_transform(X[ridge_feats].values)
    m_r = Ridge(alpha=1.0, random_state=SEED)
    m_r.fit(Xr, yr, sample_weight=w)
    print(f"  Ridge: {len(ridge_feats)} features")

    # CatBoost
    sc_c = StandardScaler()
    Xc = sc_c.fit_transform(X[cb_feats].values)
    m_c = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                            l2_leaf_reg=5, random_seed=SEED, verbose=0)
    m_c.fit(Xc, yr, sample_weight=w)
    print(f"  CatBoost: {len(cb_feats)} features")

    # Classifier (uses union of all features)
    all_feats = list(dict.fromkeys(lasso_feats + ridge_feats + cb_feats))
    sc_cls = StandardScaler()
    Xcls = sc_cls.fit_transform(X[all_feats].values)
    cls = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)
    yc = (yr > 0).astype(int)
    cls.fit(Xcls, yc, sample_weight=w)

    bundle = {
        "models": [m_l, m_r, m_c],
        "scalers": [sc_l, sc_r, sc_c],
        "feature_sets": [lasso_feats, ridge_feats, cb_feats],
        "model_names": ["Lasso_0.1", "Ridge_1.0", "CatBoost_d4_300"],
        "cls_model": cls,
        "cls_scaler": sc_cls,
        "cls_features": all_feats,
        "tiers": tiers,
        "n_train": len(X),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_type": "residual_ats_independent_ensemble_v28",
        "architecture": "Lasso(own) + Ridge(own) + CatBoost(own) → avg",
    }

    from db import save_model
    save_model("nba_ats_residual", bundle)
    print(f"\n  ✅ Saved to Supabase as 'nba_ats_residual'")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA v28 — PER-MODEL FEATURE SELECTION")
    print("  Each model gets its OWN optimal feature set")
    print("=" * 70)

    X, candidates, am, sp, yr, w = load_data()

    # Forward select for each model independently
    lasso_feats, lasso_added, lasso_score = forward_select_model(
        "Lasso", X, candidates, yr, w, am, sp)

    ridge_feats, ridge_added, ridge_score = forward_select_model(
        "Ridge", X, candidates, yr, w, am, sp)

    cb_feats, cb_added, cb_score = forward_select_model(
        "CatBoost", X, candidates, yr, w, am, sp)

    # Save feature lists
    for name, feats in [("lasso", lasso_feats), ("ridge", ridge_feats), ("catboost", cb_feats)]:
        with open(f"v28_{name}_features.txt", "w") as f:
            for feat in feats:
                f.write(feat + "\n")
        print(f"  Saved {len(feats)} features to v28_{name}_features.txt")

    # Validate independent ensemble
    oof_ens, tiers, oof_l, oof_r, oof_c = validate_independent_ensemble(
        X, lasso_feats, ridge_feats, cb_feats, yr, w, am, sp)

    # Compare: independent vs shared features
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':20s} {'Features':>8s} {'Score':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8}")
    print(f"  {'Lasso (own)':20s} {len(lasso_feats):>8d} {lasso_score:>8.5f}")
    print(f"  {'Ridge (own)':20s} {len(ridge_feats):>8d} {ridge_score:>8.5f}")
    print(f"  {'CatBoost (own)':20s} {len(cb_feats):>8d} {cb_score:>8.5f}")

    if args.upload:
        train_and_upload(X, lasso_feats, ridge_feats, cb_feats, yr, w, tiers)

    print(f"\n  Done. Add --upload to deploy.")


if __name__ == "__main__":
    main()
