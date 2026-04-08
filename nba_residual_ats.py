#!/usr/bin/env python3
"""
nba_residual_ats.py — Residual ATS Model for NBA
=================================================
Uses the v27 data pipeline (11K+ games, 105 features).
Target: actual_margin + market_spread (where Vegas is wrong)

Run:
  python nba_residual_ats.py --compare --folds 20
  python nba_residual_ats.py --upload
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.calibration import IsotonicRegression
from datetime import datetime, timezone

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════
# LOAD DATA (same pipeline as v27 retrain)
# ═══════════════════════════════════════════════════════════

def load_and_prepare():
    from nba_build_features_v27 import load_training_data, build_features

    df = load_training_data()
    X_all, all_feature_names = build_features(df)

    # Targets
    actual_margin = (df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)).values
    spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values
    has_spread = np.abs(spreads) > 0.1
    y_residual = actual_margin + spreads  # positive = home covered ATS

    # Season weights
    seasons = pd.to_numeric(df.get("season", 2026), errors="coerce").fillna(2026).astype(int).values
    current_year = 2026
    weights = np.array([{0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8}.get(current_year - s, 0.7) for s in seasons])

    # Sort by date
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    sort_idx = dates.argsort()
    X_all = X_all.iloc[sort_idx].reset_index(drop=True)
    actual_margin = actual_margin[sort_idx]
    spreads = spreads[sort_idx]
    has_spread = has_spread[sort_idx]
    y_residual = y_residual[sort_idx]
    weights = weights[sort_idx]
    seasons = seasons[sort_idx]

    # Filter to games WITH market spread (can't compute residual without it)
    mask = has_spread
    X_spread = X_all[mask].reset_index(drop=True)
    actual_m = actual_margin[mask]
    spr = spreads[mask]
    y_res = y_residual[mask]
    w = weights[mask]

    # Feature selection for residual model:
    # EXCLUDE market_spread itself (target is derived from it — circular)
    # EXCLUDE spread_vs_market, score_diff_pred, total_pred (model outputs)
    # KEEP everything else — let Lasso decide what matters
    exclude = {"market_spread", "spread_vs_market", "score_diff_pred", "total_pred",
               "home_fav", "has_market"}
    res_features = [f for f in all_feature_names if f not in exclude]
    available = [f for f in res_features if f in X_spread.columns]
    X = X_spread[available].fillna(0)

    print(f"\n  Residual dataset: {len(X)} games with spreads")
    print(f"  Features: {len(available)} (excluded {len(exclude)} market-derived)")
    print(f"  ATS residual: mean={y_res.mean():+.2f}, std={y_res.std():.2f}")
    print(f"  Home cover rate: {(y_res > 0).mean():.1%}")
    print(f"  Seasons: {sorted(set(seasons[mask]))}")

    return X, available, actual_m, spr, y_res, w


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════

def walk_forward(X, y_residual, y_margin, spreads, weights, n_folds=20):
    n = len(X)
    fold_size = n // (n_folds + 1)
    min_train = max(fold_size * 3, 500)

    oof_res = np.full(n, np.nan)
    oof_direct = np.full(n, np.nan)
    oof_cls = np.full(n, np.nan)
    oof_res_ridge = np.full(n, np.nan)
    oof_res_cb = np.full(n, np.nan)

    print(f"\n  Walk-forward: {n_folds} folds, ~{fold_size}/fold, min_train={min_train}, total={n}")

    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n:
            break

        X_tr, X_te = X[:ts], X[ts:te]
        yr_tr = y_residual[:ts]
        ym_tr = y_margin[:ts]
        w_tr = weights[:ts]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Residual: Lasso (primary)
        m1 = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        m1.fit(X_tr_s, yr_tr, sample_weight=w_tr)
        oof_res[ts:te] = m1.predict(X_te_s)

        # Residual: Ridge
        m2 = Ridge(alpha=1.0, random_state=SEED)
        m2.fit(X_tr_s, yr_tr, sample_weight=w_tr)
        oof_res_ridge[ts:te] = m2.predict(X_te_s)

        # Residual: CatBoost
        if HAS_CB:
            m3 = CatBoostRegressor(depth=4, iterations=300, learning_rate=0.05,
                                   l2_leaf_reg=5, random_seed=SEED, verbose=0)
            m3.fit(X_tr_s, yr_tr, sample_weight=w_tr)
            oof_res_cb[ts:te] = m3.predict(X_te_s)

        # Direct margin: Lasso (for comparison)
        m_dir = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        m_dir.fit(X_tr_s, ym_tr, sample_weight=w_tr)
        oof_direct[ts:te] = m_dir.predict(X_te_s)

        # Classifier: P(home covers)
        y_cov = (yr_tr > 0).astype(int)
        if y_cov.sum() > 10 and (1 - y_cov).sum() > 10:
            cls = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)
            cls.fit(X_tr_s, y_cov, sample_weight=w_tr)
            oof_cls[ts:te] = cls.predict_proba(X_te_s)[:, 1]

        if (fold + 1) % 5 == 0:
            print(f"    Fold {fold+1}/{n_folds} done (train={ts}, test={te-ts})")

    # Ensemble residual
    models_used = [oof_res]
    if not np.all(np.isnan(oof_res_ridge)):
        models_used.append(oof_res_ridge)
    if HAS_CB and not np.all(np.isnan(oof_res_cb)):
        models_used.append(oof_res_cb)
    oof_ensemble = np.nanmean(models_used, axis=0)

    return oof_res, oof_direct, oof_cls, oof_ensemble, oof_res_ridge, oof_res_cb


# ═══════════════════════════════════════════════════════════
# ATS EVALUATION
# ═══════════════════════════════════════════════════════════

def ats_eval(preds, actual_margin, spreads, label="Model"):
    ats_result = actual_margin + spreads
    not_push = ats_result != 0

    print(f"\n  {'─'*65}")
    print(f"  {label}")
    print(f"  {'─'*65}")
    print(f"  {'Threshold':>10} {'Games':>7} {'Correct':>8} {'ATS%':>7} {'ROI%':>7}")
    print(f"  {'─'*10} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")

    results = {}
    for threshold in [0, 0.5, 1, 2, 3, 4, 5, 7, 10]:
        mask = (np.abs(preds) >= threshold) & not_push & ~np.isnan(preds)
        n_picks = mask.sum()
        if n_picks < 20:
            continue
        model_says_home = preds[mask] > 0
        home_covered = ats_result[mask] > 0
        correct = model_says_home == home_covered
        acc = correct.mean()
        roi = (acc * 1.909 - 1) * 100
        print(f"  {threshold:>9}+ {n_picks:>7} {correct.sum():>8} {acc:>6.1%} {roi:>+6.1f}%")
        results[threshold] = {"n": int(n_picks), "acc": round(float(acc), 4), "roi": round(float(roi), 1)}

    return results


def compare_models(oof_res, oof_direct, oof_ensemble, actual_margin, spreads):
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD: Residual vs Direct vs Ensemble")
    print(f"{'='*70}")

    ats_eval(oof_res, actual_margin, spreads, "RESIDUAL (Lasso solo)")
    ats_eval(oof_ensemble, actual_margin, spreads, "RESIDUAL (Lasso+Ridge+CatBoost ensemble)")

    direct_edge = oof_direct - (-spreads)
    ats_eval(direct_edge, actual_margin, spreads, "DIRECT (predicts margin, compares to spread)")

    valid = ~np.isnan(oof_res) & ~np.isnan(oof_direct)
    if valid.sum() > 100:
        res_mae = np.mean(np.abs(oof_res[valid] - (actual_margin[valid] + spreads[valid])))
        dir_mae = np.mean(np.abs(oof_direct[valid] - actual_margin[valid]))
        mkt_mae = np.mean(np.abs(-spreads[valid] - actual_margin[valid]))
        print(f"\n  Residual MAE (predicting ATS result): {res_mae:.3f}")
        print(f"  Direct MAE (predicting margin):        {dir_mae:.3f}")
        print(f"  Market MAE (spread vs actual margin):   {mkt_mae:.3f}")
        print(f"  Direct beats market by:                 {mkt_mae - dir_mae:+.3f}")


# ═══════════════════════════════════════════════════════════
# ASYMMETRIC TIERS
# ═══════════════════════════════════════════════════════════

def find_tiers(oof_res, oof_cls, actual_margin, spreads):
    print(f"\n{'='*70}")
    print(f"  ASYMMETRIC TIER SEARCH (HOME vs AWAY cover)")
    print(f"{'='*70}")

    ats_result = actual_margin + spreads
    not_push = ats_result != 0
    valid = ~np.isnan(oof_res) & not_push

    res = oof_res[valid]
    cls = oof_cls[valid]
    actual_cover = (ats_result[valid] > 0).astype(int)

    print(f"  Base: {actual_cover.mean():.1%} home cover ({valid.sum()} games)")

    for direction, dir_label in [("home", "HOME COVER"), ("away", "AWAY COVER")]:
        print(f"\n  {dir_label} tiers:")
        print(f"  {'Res_thresh':>10} {'+Cls':>5} {'Games':>7} {'Acc%':>7} {'ROI%':>7}")
        print(f"  {'─'*10} {'─'*5} {'─'*7} {'─'*7} {'─'*7}")

        for res_t in [0.5, 1, 2, 3, 4, 5, 7]:
            for cls_gate in [None, 0.52, 0.55]:
                if direction == "home":
                    mask = res > res_t
                    if cls_gate and not np.all(np.isnan(cls)):
                        mask = mask & (cls > cls_gate)
                    acc_val = actual_cover[mask].mean() if mask.sum() > 0 else 0
                else:
                    mask = res < -res_t
                    if cls_gate and not np.all(np.isnan(cls)):
                        mask = mask & (cls < (1 - cls_gate))
                    acc_val = (1 - actual_cover[mask]).mean() if mask.sum() > 0 else 0

                n = mask.sum()
                if n < 20:
                    continue
                roi = (acc_val * 1.909 - 1) * 100
                cls_str = f"{cls_gate}" if cls_gate else "none"
                print(f"  {res_t:>9}+ {cls_str:>5} {n:>7} {acc_val:>6.1%} {roi:>+6.1f}%")


# ═══════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════

def show_feature_importance(X, y_residual, weights, feature_names):
    print(f"\n{'='*70}")
    print(f"  FEATURE IMPORTANCE (full-data Lasso)")
    print(f"{'='*70}")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    m = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
    m.fit(X_s, y_residual, sample_weight=weights)

    coefs = sorted(zip(feature_names, m.coef_), key=lambda x: abs(x[1]), reverse=True)
    active = [(f, c) for f, c in coefs if abs(c) > 0.001]
    print(f"  Active features: {len(active)}/{len(feature_names)}")
    print(f"\n  {'Feature':40s} {'Coef':>8s}")
    print(f"  {'─'*40} {'─'*8}")
    for f, c in active[:30]:
        print(f"  {f:40s} {c:>+7.4f}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--folds", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print("  NBA RESIDUAL ATS MODEL")
    print("  Target: actual_margin + spread (where Vegas is wrong)")
    print("  Data: v27 training pipeline (11K+ games)")
    print("=" * 70)

    X, feature_names, actual_margin, spreads, y_residual, weights = load_and_prepare()

    # Walk-forward
    oof_res, oof_direct, oof_cls, oof_ensemble, oof_ridge, oof_cb = walk_forward(
        X.values, y_residual, actual_margin, spreads, weights, n_folds=args.folds
    )

    # Results
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD RESULTS")
    print(f"{'='*70}")

    res_results = ats_eval(oof_res, actual_margin, spreads, "RESIDUAL (Lasso)")
    ens_results = ats_eval(oof_ensemble, actual_margin, spreads, "RESIDUAL (Ensemble)")

    if args.compare:
        compare_models(oof_res, oof_direct, oof_ensemble, actual_margin, spreads)

    # Tiers
    find_tiers(oof_ensemble, oof_cls, actual_margin, spreads)

    # Feature importance
    show_feature_importance(X.values, y_residual, weights, feature_names)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for label, results in [("Lasso", res_results), ("Ensemble", ens_results)]:
        if results:
            best = max(results.items(), key=lambda x: x[1].get("roi", -100))
            print(f"  {label} best: {best[0]}+ → {best[1]['acc']:.1%} ATS, "
                  f"{best[1]['roi']:+.1f}% ROI ({best[1]['n']} picks)")

    # Upload
    if args.upload:
        print(f"\n  Training production model...")
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X.values)

        res_model = Lasso(alpha=0.1, max_iter=5000, random_state=SEED)
        res_model.fit(X_s, y_residual, sample_weight=weights)

        y_covers = (y_residual > 0).astype(int)
        cls_model = LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)
        cls_model.fit(X_s, y_covers, sample_weight=weights)

        bundle = {
            "res_model": res_model, "cls_model": cls_model, "scaler": scaler,
            "feature_cols": feature_names,
            "n_train": len(X), "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "residual_ats_lasso", "architecture": "Lasso_res + LogReg_cls",
        }
        from db import save_model
        save_model("nba_ats_residual", bundle)
        print(f"  ✅ Saved to Supabase as 'nba_ats_residual'")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
