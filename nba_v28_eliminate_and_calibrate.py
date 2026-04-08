#!/usr/bin/env python3
"""
nba_v28_eliminate_and_calibrate.py
===================================
Phase 1: Drop-one feature elimination on residual ATS target
Phase 2: Forward selection for optimal feature set
Phase 3: Multi-alpha sweep (Lasso, Ridge, ElasticNet)
Phase 4: Walk-forward validation of final model
Phase 5: Tier calibration (unit sizing thresholds)

Run:
  python nba_v28_eliminate_and_calibrate.py
  python nba_v28_eliminate_and_calibrate.py --upload   # train + deploy
"""

import numpy as np
import pandas as pd
import os, sys, time, copy, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.calibration import IsotonicRegression

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except:
    HAS_CB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except:
    HAS_LGBM = False

SEED = 42
np.random.seed(SEED)
N_FOLDS = 15  # fewer folds for speed during elimination


# ═══════════════════════════════════════════════════════════
# LOAD DATA (same as the comparison run)
# ═══════════════════════════════════════════════════════════

def load_data():
    from nba_build_features_v27 import load_training_data, build_features

    df = load_training_data()
    X_all, all_feats = build_features(df)

    # Merge player features
    pf = pd.read_parquet("nba_player_features_for_training.parquet")
    pf_cols = [c for c in pf.columns if c not in ["game_id", "game_date", "home_team", "away_team", "_match_key"]]
    pf = pf.drop_duplicates(subset="_match_key", keep="first")

    df["_match_key"] = (pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d") + "|" +
                        df.get("home_team", df.get("home_team_name", "")).astype(str))
    merged = df[["_match_key"]].merge(pf[["_match_key"] + pf_cols], on="_match_key", how="left")
    merged = merged.iloc[:len(X_all)]
    for c in pf_cols:
        X_all[c] = merged[c].fillna(0).values

    actual_margin = (df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)).values
    spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values
    has_spread = np.abs(spreads) > 0.1
    y_residual = actual_margin + spreads

    seasons = pd.to_numeric(df.get("season", 2026), errors="coerce").fillna(2026).astype(int).values
    weights = np.array([{0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8}.get(2026 - s, 0.7) for s in seasons])

    dates = pd.to_datetime(df["game_date"], errors="coerce")
    idx = dates.argsort()
    X_all = X_all.iloc[idx].reset_index(drop=True)
    actual_margin = actual_margin[idx]; spreads = spreads[idx]
    has_spread = has_spread[idx]; y_residual = y_residual[idx]; weights = weights[idx]

    m = has_spread
    X = X_all[m].reset_index(drop=True)
    am = actual_margin[m]; sp = spreads[m]; yr = y_residual[m]; w = weights[m]

    exclude = {"market_spread", "spread_vs_market", "score_diff_pred", "total_pred", "home_fav", "has_market"}
    feats = [f for f in X.columns if f not in exclude]
    X = X[feats].fillna(0)

    print(f"  Data: {len(X)} games, {len(feats)} features")
    return X, feats, am, sp, yr, w, pf_cols


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD ATS SCORE
# ═══════════════════════════════════════════════════════════

def wf_ats_score(X_vals, y_res, weights, am, sp, n_folds=N_FOLDS, alpha=0.1):
    """Walk-forward ATS accuracy at 4+ threshold (our primary metric)."""
    n = len(X_vals)
    fold_size = n // (n_folds + 1)
    min_train = max(fold_size * 3, 500)
    oof = np.full(n, np.nan)

    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_vals[:ts])
        Xte = sc.transform(X_vals[ts:te])
        m = Lasso(alpha=alpha, max_iter=5000, random_state=SEED)
        m.fit(Xtr, y_res[:ts], sample_weight=weights[:ts])
        oof[ts:te] = m.predict(Xte)

    ats = am + sp
    push = ats == 0
    valid = ~np.isnan(oof) & ~push

    results = {}
    for t in [0, 3, 4, 5, 7]:
        mask = valid & (np.abs(oof) >= t)
        if mask.sum() < 20:
            results[t] = 0.5
            continue
        correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
        results[t] = correct

    # Composite: weighted average of ATS at key thresholds
    composite = results.get(4, 0.5) * 0.4 + results.get(5, 0.5) * 0.3 + results.get(7, 0.5) * 0.3
    return composite, results


# ═══════════════════════════════════════════════════════════
# PHASE 1: DROP-ONE ELIMINATION
# ═══════════════════════════════════════════════════════════

def phase1_drop_one(X, feats, yr, w, am, sp):
    print(f"\n{'='*70}")
    print(f"  PHASE 1: DROP-ONE ELIMINATION ({len(feats)} features)")
    print(f"{'='*70}")

    baseline_score, baseline_detail = wf_ats_score(X.values, yr, w, am, sp)
    print(f"  Baseline composite: {baseline_score:.5f}")
    print(f"  Baseline ATS: 4+={baseline_detail[4]:.1%} 5+={baseline_detail[5]:.1%} 7+={baseline_detail[7]:.1%}")

    results = {}
    t0 = time.time()
    for i, feat in enumerate(feats):
        cols = [f for f in feats if f != feat]
        X_drop = X[cols].values
        score, detail = wf_ats_score(X_drop, yr, w, am, sp)
        delta = baseline_score - score  # positive = removing hurts (feature is useful)
        results[feat] = {"score": score, "delta": delta, "detail": detail}

        elapsed = time.time() - t0
        remaining = (len(feats) - i - 1) * (elapsed / (i + 1)) / 60
        tag = "✅" if delta > 0.002 else ("❌" if delta < -0.002 else "⚪")
        if (i + 1) % 10 == 0 or i < 5:
            print(f"    [{i+1:3d}/{len(feats)}] {tag} {feat:40s} Δ={delta:+.5f} (~{remaining:.1f}m left)")

    # Sort
    sorted_feats = sorted(results.items(), key=lambda x: -x[1]["delta"])

    print(f"\n  TOP 15 (removing hurts most — keep these):")
    for f, r in sorted_feats[:15]:
        print(f"    {f:40s} Δ={r['delta']:+.5f}")

    print(f"\n  BOTTOM 10 (removing helps — drop these):")
    for f, r in sorted_feats[-10:]:
        print(f"    {f:40s} Δ={r['delta']:+.5f}")

    # Keep features where removing hurts or is neutral (delta >= -0.001)
    keep = [f for f, r in sorted_feats if r["delta"] >= -0.001]
    drop = [f for f, r in sorted_feats if r["delta"] < -0.001]
    print(f"\n  Keep: {len(keep)}, Drop: {len(drop)}")

    return keep, drop, baseline_score, results


# ═══════════════════════════════════════════════════════════
# PHASE 2: FORWARD SELECTION
# ═══════════════════════════════════════════════════════════

def phase2_forward(X, keep_feats, yr, w, am, sp):
    print(f"\n{'='*70}")
    print(f"  PHASE 2: FORWARD SELECTION (from {len(keep_feats)} survivors)")
    print(f"{'='*70}")

    selected = []
    remaining = list(keep_feats)
    best_score = 0.5

    for step in range(min(60, len(remaining))):
        best_feat = None
        best_step_score = best_score

        for feat in remaining:
            trial = selected + [feat]
            score, _ = wf_ats_score(X[trial].values, yr, w, am, sp)
            if score > best_step_score:
                best_step_score = score
                best_feat = feat

        if best_feat is None:
            print(f"    Step {step+1}: No improvement — stopping")
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_step_score
        print(f"    Step {step+1}: +{best_feat:40s} score={best_score:.5f}")

        if step >= 10 and best_step_score - best_score < 0.0001:
            break

    print(f"\n  Selected {len(selected)} features")
    return selected


# ═══════════════════════════════════════════════════════════
# PHASE 3: ALPHA SWEEP
# ═══════════════════════════════════════════════════════════

def phase3_alpha_sweep(X, feats, yr, w, am, sp):
    print(f"\n{'='*70}")
    print(f"  PHASE 3: ALPHA SWEEP ({len(feats)} features)")
    print(f"{'='*70}")

    print(f"  {'Model':20s} {'Alpha':>7s} {'Comp':>8s} {'ATS@4':>7s} {'ATS@5':>7s} {'ATS@7':>7s}")
    print(f"  {'─'*20} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")

    best = None
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        score, detail = wf_ats_score(X[feats].values, yr, w, am, sp, alpha=alpha)
        print(f"  {'Lasso':20s} {alpha:>7.2f} {score:>8.5f} {detail[4]:>6.1%} {detail[5]:>6.1%} {detail[7]:>6.1%}")
        if best is None or score > best[0]:
            best = (score, alpha, detail)

    print(f"\n  ✅ Best alpha: {best[1]} (composite={best[0]:.5f})")
    return best[1]


# ═══════════════════════════════════════════════════════════
# PHASE 4: FINAL WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════

def phase4_validate(X, feats, yr, w, am, sp, alpha=0.1):
    print(f"\n{'='*70}")
    print(f"  PHASE 4: FINAL VALIDATION — {len(feats)} features, α={alpha}")
    print(f"{'='*70}")

    n = len(X); n_folds = 25
    fold_size = n // (n_folds + 1)
    min_train = max(fold_size * 3, 500)
    oof = np.full(n, np.nan)
    oof_cls = np.full(n, np.nan)

    Xv = X[feats].values

    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xv[:ts]); Xte = sc.transform(Xv[ts:te])

        m = Lasso(alpha=alpha, max_iter=5000, random_state=SEED)
        m.fit(Xtr, yr[:ts], sample_weight=w[:ts])
        oof[ts:te] = m.predict(Xte)

        yc = (yr[:ts] > 0).astype(int)
        if yc.sum() > 10:
            c = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)
            c.fit(Xtr, yc, sample_weight=w[:ts])
            oof_cls[ts:te] = c.predict_proba(Xte)[:, 1]

    ats = am + sp; push = ats == 0

    print(f"\n  {'Thresh':>7} {'Games':>6} {'ATS%':>6} {'ROI%':>7}")
    print(f"  {'─'*7} {'─'*6} {'─'*6} {'─'*7}")
    for t in [0, 1, 2, 3, 4, 5, 7, 10]:
        mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
        if mask.sum() < 20: continue
        correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100
        print(f"  {t:>6}+ {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}%")

    return oof, oof_cls


# ═══════════════════════════════════════════════════════════
# PHASE 5: TIER CALIBRATION
# ═══════════════════════════════════════════════════════════

def phase5_calibrate(oof, oof_cls, am, sp):
    print(f"\n{'='*70}")
    print(f"  PHASE 5: TIER CALIBRATION (unit sizing)")
    print(f"{'='*70}")

    ats = am + sp; push = ats == 0
    valid = ~np.isnan(oof) & ~push

    # Find optimal thresholds for 1u / 2u / 3u
    print(f"\n  Searching for 1u/2u/3u thresholds (target: 55%/60%/65%)...")
    print(f"  {'Residual':>9} {'Games':>6} {'ATS%':>6} {'ROI%':>7} {'Tier':>5}")
    print(f"  {'─'*9} {'─'*6} {'─'*6} {'─'*7} {'─'*5}")

    tiers = {}
    for t in np.arange(0.5, 12.5, 0.5):
        mask = valid & (np.abs(oof) >= t)
        if mask.sum() < 20: continue
        correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100

        tier = ""
        if correct >= 0.65 and 3 not in tiers:
            tiers[3] = t; tier = "← 3u"
        elif correct >= 0.60 and 2 not in tiers:
            tiers[2] = t; tier = "← 2u"
        elif correct >= 0.55 and 1 not in tiers:
            tiers[1] = t; tier = "← 1u"

        print(f"  {t:>8.1f}+ {mask.sum():>6} {correct:>5.1%} {roi:>+6.1f}% {tier:>5}")

    print(f"\n  CALIBRATED THRESHOLDS:")
    for u in [1, 2, 3]:
        if u in tiers:
            t = tiers[u]
            mask = valid & (np.abs(oof) >= t)
            correct = ((oof[mask] > 0) == (ats[mask] > 0)).mean()
            print(f"    {u}u: residual ≥ {t:.1f} → {correct:.1%} ATS ({mask.sum()} picks)")
        else:
            print(f"    {u}u: not reached")

    # Asymmetric: home vs away
    print(f"\n  ASYMMETRIC (home cover vs away cover at 2u threshold):")
    if 2 in tiers:
        t = tiers[2]
        for label, cond in [("HOME cover", oof > t), ("AWAY cover", oof < -t)]:
            mask = valid & cond
            if mask.sum() < 20: continue
            if "HOME" in label:
                correct = (ats[mask] > 0).mean()
            else:
                correct = (ats[mask] < 0).mean()
            roi = (correct * 1.909 - 1) * 100
            print(f"    {label}: {correct:.1%} ATS, {roi:+.1f}% ROI ({mask.sum()} picks)")

    return tiers


# ═══════════════════════════════════════════════════════════
# PRODUCTION TRAINING + UPLOAD
# ═══════════════════════════════════════════════════════════

def train_and_upload(X, feats, yr, w, am, sp, alpha, tiers):
    print(f"\n{'='*70}")
    print(f"  PRODUCTION TRAINING — {len(feats)} features, α={alpha}")
    print(f"{'='*70}")

    Xv = X[feats].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(Xv)

    res_model = Lasso(alpha=alpha, max_iter=5000, random_state=SEED)
    res_model.fit(X_s, yr, sample_weight=w)
    n_active = np.sum(np.abs(res_model.coef_) > 1e-6)
    print(f"  Residual Lasso: {n_active} active features")

    yc = (yr > 0).astype(int)
    cls_model = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)
    cls_model.fit(X_s, yc, sample_weight=w)

    # Top features
    coefs = sorted(zip(feats, res_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 20 features:")
    for f, c in coefs[:20]:
        if abs(c) > 0.001:
            print(f"    {f:40s} {c:+.4f}")

    bundle = {
        "res_model": res_model,
        "cls_model": cls_model,
        "scaler": scaler,
        "feature_cols": feats,
        "tiers": tiers,
        "n_train": len(X),
        "n_active": int(n_active),
        "alpha": alpha,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_type": "residual_ats_v28",
        "architecture": f"Lasso_α{alpha}_res + LogReg_cls",
    }

    from db import save_model
    save_model("nba_ats_residual", bundle)
    print(f"\n  ✅ Saved to Supabase as 'nba_ats_residual'")
    print(f"  Features: {len(feats)}, Active: {n_active}")
    print(f"  Tiers: {tiers}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--skip-elim", action="store_true", help="Skip elimination, use all features")
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA v28 RESIDUAL ATS — ELIMINATION + CALIBRATION")
    print("=" * 70)

    X, feats, am, sp, yr, w, pf_cols = load_data()

    if args.skip_elim:
        final_feats = feats
        best_alpha = 0.1
    else:
        # Phase 1: Drop-one
        keep, drop, baseline, results = phase1_drop_one(X, feats, yr, w, am, sp)

        # Phase 2: Forward selection (optional — skip if drop-one is clean)
        if len(drop) > 20:
            final_feats = phase2_forward(X, keep, yr, w, am, sp)
        else:
            final_feats = keep
            print(f"\n  Clean elimination — using {len(keep)} survivors directly")

        # Phase 3: Alpha sweep
        best_alpha = phase3_alpha_sweep(X, final_feats, yr, w, am, sp)

    # Phase 4: Final validation
    oof, oof_cls = phase4_validate(X, final_feats, yr, w, am, sp, alpha=best_alpha)

    # Phase 5: Tier calibration
    tiers = phase5_calibrate(oof, oof_cls, am, sp)

    # Upload
    if args.upload:
        train_and_upload(X, final_feats, yr, w, am, sp, best_alpha, tiers)

    print(f"\n  Done. Add --upload to deploy production model.")


if __name__ == "__main__":
    main()
