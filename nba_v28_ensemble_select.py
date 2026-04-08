#!/usr/bin/env python3
"""
nba_v28_ensemble_select.py — Forward selection using the FULL ensemble
=====================================================================
Scores each feature addition with Lasso+Ridge+CatBoost ensemble average.
Finds the optimal feature set for the actual production architecture.

Run:
  python nba_v28_ensemble_select.py                  # forward selection
  python nba_v28_ensemble_select.py --upload          # train + deploy
  python nba_v28_ensemble_select.py --validate-only   # validate saved features
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from catboost import CatBoostRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

SEED = 42
np.random.seed(SEED)
N_FOLDS = 15
FEATURES_FILE = "v28_ensemble_features.txt"


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
    hs = np.abs(sp) > 0.1
    yr = am + sp
    seasons = pd.to_numeric(df.get("season", 2026), errors="coerce").fillna(2026).astype(int).values
    w = np.array([{0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8}.get(2026 - s, 0.7) for s in seasons])

    idx = pd.to_datetime(df["game_date"], errors="coerce").argsort()
    X_all = X_all.iloc[idx].reset_index(drop=True)
    am = am[idx]; sp = sp[idx]; hs = hs[idx]; yr = yr[idx]; w = w[idx]

    m = hs
    X = X_all[m].reset_index(drop=True)
    am = am[m]; sp = sp[m]; yr = yr[m]; w = w[m]

    exclude = {"market_spread", "spread_vs_market", "score_diff_pred", "total_pred", "home_fav", "has_market"}
    candidates = [f for f in X.columns if f not in exclude]
    X = X[candidates].fillna(0)

    print(f"  Data: {len(X)} games, {len(candidates)} candidate features")
    return X, candidates, am, sp, yr, w, pf_cols


# ═══════════════════════════════════════════════════════════
# ENSEMBLE WALK-FORWARD SCORE
# ═══════════════════════════════════════════════════════════

def wf_ensemble_score(Xv, yr, w, am, sp, n_folds=N_FOLDS):
    """
    Walk-forward with Lasso+Ridge+CatBoost ensemble.
    Returns composite score and per-threshold details.
    """
    n = len(Xv)
    fs = n // (n_folds + 1)
    mt = max(fs * 3, 500)

    oof_lasso = np.full(n, np.nan)
    oof_ridge = np.full(n, np.nan)
    oof_cb = np.full(n, np.nan)

    for fold in range(n_folds):
        ts = mt + fold * fs
        te = min(ts + fs, n)
        if ts >= n:
            break

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xv[:ts])
        Xte = sc.transform(Xv[ts:te])
        wt = w[:ts]
        yt = yr[:ts]

        # Lasso
        m1 = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        m1.fit(Xtr, yt, sample_weight=wt)
        oof_lasso[ts:te] = m1.predict(Xte)

        # Ridge
        m2 = Ridge(alpha=1.0, random_state=SEED)
        m2.fit(Xtr, yt, sample_weight=wt)
        oof_ridge[ts:te] = m2.predict(Xte)

        # CatBoost
        m3 = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                               l2_leaf_reg=5, random_seed=SEED, verbose=0)
        m3.fit(Xtr, yt, sample_weight=wt)
        oof_cb[ts:te] = m3.predict(Xte)

    # Ensemble average
    oof = np.nanmean([oof_lasso, oof_ridge, oof_cb], axis=0)

    ats = am + sp
    push = ats == 0
    r = {}
    for t in [0, 3, 4, 5, 7]:
        mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
        r[t] = ((oof[mask] > 0) == (ats[mask] > 0)).mean() if mask.sum() >= 20 else 0.5

    composite = r[4] * 0.4 + r[5] * 0.3 + r[7] * 0.3
    return composite, r, oof


# ═══════════════════════════════════════════════════════════
# FORWARD SELECTION
# ═══════════════════════════════════════════════════════════

def forward_select(X, candidates, yr, w, am, sp):
    # Core: intersection of both pass top-15
    core = [
        "lineup_value_diff",
        "matchup_efg",
        "three_value_diff",
        "ref_foul_proxy",
        "scoring_hhi_diff",
        "threepct_diff",
        "rest_diff",
        "away_minutes_load",
        "roll_pts_off_to_diff",
    ]
    core = [f for f in core if f in candidates]
    remaining = [f for f in candidates if f not in core]

    print(f"\n{'='*70}")
    print(f"  ENSEMBLE FORWARD SELECTION")
    print(f"  Core: {len(core)} features (both-pass intersection)")
    print(f"  Candidates: {len(remaining)}")
    print(f"  Scoring: Lasso + Ridge + CatBoost (averaged)")
    print(f"{'='*70}")

    core_score, cd, _ = wf_ensemble_score(X[core].values, yr, w, am, sp)
    print(f"\n  Core score: {core_score:.5f} (4+={cd[4]:.1%} 5+={cd[5]:.1%} 7+={cd[7]:.1%})")

    selected = list(core)
    best_score = core_score
    added = []
    skipped = []

    t0 = time.time()
    for i, feat in enumerate(remaining):
        trial = selected + [feat]
        score, detail, _ = wf_ensemble_score(X[trial].values, yr, w, am, sp)
        delta = score - best_score

        if delta > 0.00005:
            selected.append(feat)
            best_score = score
            added.append((feat, delta, detail))
            tag = "✅"
        else:
            skipped.append((feat, delta))
            tag = "❌"

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 1
        rem = (len(remaining) - i - 1) / rate / 60
        if (i + 1) % 5 == 0 or delta > 0.001:
            print(f"  [{i+1:3d}/{len(remaining)}] {tag} {feat:35s} d={delta:+.6f} "
                  f"now={best_score:.5f} ({len(selected)} feats) ~{rem:.1f}m")

    return selected, added, core_score, best_score


# ═══════════════════════════════════════════════════════════
# FINAL VALIDATION (25 folds, full detail)
# ═══════════════════════════════════════════════════════════

def final_validate(X, feats, yr, w, am, sp):
    print(f"\n{'='*70}")
    print(f"  FINAL VALIDATION — {len(feats)} features, 25-fold ensemble")
    print(f"{'='*70}")

    _, _, oof = wf_ensemble_score(X[feats].values, yr, w, am, sp, n_folds=25)

    ats = am + sp
    push = ats == 0

    print(f"\n  {'Thresh':>7} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
    print(f"  {'─'*7} {'─'*6} {'─'*6} {'─'*7}")
    for t in [0, 1, 2, 3, 4, 5, 7, 10]:
        mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
        if mask.sum() < 20:
            continue
        correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100
        print(f"  {t:>6}+ {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")

    # Tier calibration
    print(f"\n  TIER CALIBRATION:")
    tiers = {}
    for t in np.arange(0.5, 12.5, 0.5):
        mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
        if mask.sum() < 20:
            continue
        correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
        if correct >= 0.65 and 3 not in tiers:
            tiers[3] = t
        elif correct >= 0.60 and 2 not in tiers:
            tiers[2] = t
        elif correct >= 0.55 and 1 not in tiers:
            tiers[1] = t

    for u in [1, 2, 3]:
        if u in tiers:
            t = tiers[u]
            mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
            correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
            print(f"    {u}u: residual ≥ {t:.1f} → {correct:.1%} ATS ({mask.sum()} picks)")

    # Asymmetric
    print(f"\n  ASYMMETRIC:")
    if 2 in tiers:
        t = tiers[2]
        for label, cond in [("HOME", oof > t), ("AWAY", oof < -t)]:
            mask = ~np.isnan(oof) & ~push & cond
            if mask.sum() < 20:
                continue
            if label == "HOME":
                correct = (ats[mask] > 0).mean()
            else:
                correct = (ats[mask] < 0).mean()
            roi = (correct * 1.909 - 1) * 100
            print(f"    {label} cover: {correct:.1%} ATS, {roi:+.1f}% ROI ({mask.sum()} picks)")

    return oof, tiers


# ═══════════════════════════════════════════════════════════
# TRAIN & UPLOAD
# ═══════════════════════════════════════════════════════════

def train_and_upload(X, feats, yr, w, tiers):
    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING — {len(feats)} features")
    print(f"{'='*70}")

    Xv = X[feats].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(Xv)

    # Train all 3 ensemble members
    lasso = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
    lasso.fit(X_s, yr, sample_weight=w)
    n_active = np.sum(np.abs(lasso.coef_) > 1e-6)

    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_s, yr, sample_weight=w)

    cb = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                           l2_leaf_reg=5, random_seed=SEED, verbose=0)
    cb.fit(X_s, yr, sample_weight=w)

    # Classifier
    yc = (yr > 0).astype(int)
    cls = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)
    cls.fit(X_s, yc, sample_weight=w)

    print(f"  Lasso: {n_active} active features")
    print(f"  Ridge: {len(feats)} features")
    print(f"  CatBoost: depth=4, 300 iterations")

    # Top features by Lasso coefficient
    coefs = sorted(zip(feats, lasso.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 15 Lasso coefficients:")
    for f, c in coefs[:15]:
        if abs(c) > 0.001:
            print(f"    {f:40s} {c:+.4f}")

    bundle = {
        "models": [lasso, ridge, cb],
        "model_names": ["Lasso_0.1", "Ridge_1.0", "CatBoost_d4_300"],
        "cls_model": cls,
        "scaler": scaler,
        "feature_cols": feats,
        "tiers": tiers,
        "n_train": len(X),
        "n_active_lasso": int(n_active),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_type": "residual_ats_ensemble_v28",
        "architecture": "Lasso+Ridge+CatBoost → residual ATS",
    }

    from db import save_model
    save_model("nba_ats_residual", bundle)
    print(f"\n  ✅ Saved to Supabase as 'nba_ats_residual'")
    print(f"  Features: {len(feats)}, Tiers: {tiers}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA v28 ENSEMBLE FEATURE SELECTION")
    print("  Scoring: Lasso + Ridge + CatBoost (averaged)")
    print("=" * 70)

    X, candidates, am, sp, yr, w, pf_cols = load_data()

    if args.validate_only and os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE) as f:
            feats = [line.strip() for line in f if line.strip()]
        print(f"\n  Loaded {len(feats)} features from {FEATURES_FILE}")
    else:
        selected, added, core_score, final_score = forward_select(X, candidates, yr, w, am, sp)

        print(f"\n{'='*70}")
        print(f"  SELECTION SUMMARY")
        print(f"{'='*70}")
        print(f"  Core (9) → {core_score:.5f}")
        print(f"  Final ({len(selected)}) → {final_score:.5f} ({final_score-core_score:+.5f})")
        print(f"\n  ACCEPTED features (in order added):")
        for f, d, det in added:
            print(f"    {f:40s} d={d:+.6f} (5+={det[5]:.1%} 7+={det[7]:.1%})")

        with open(FEATURES_FILE, "w") as fout:
            for f in selected:
                fout.write(f + "\n")
        print(f"\n  Saved {len(selected)} features to {FEATURES_FILE}")
        feats = selected

    # Final validation
    oof, tiers = final_validate(X, feats, yr, w, am, sp)

    if args.upload:
        train_and_upload(X, feats, yr, w, tiers)

    print(f"\n  Done. Add --upload to deploy.")


if __name__ == "__main__":
    main()
