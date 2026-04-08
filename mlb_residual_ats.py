#!/usr/bin/env python3
"""
mlb_residual_ats.py — Residual ATS Model for MLB
=================================================
Same architecture as NBA v28: predict WHERE the run line is wrong.
Target: actual_margin + market_spread (run line residual)

Current MLB ATS: 60.2% ML, 70.2% ATS at 1+ run edge
Goal: beat this with the residual approach.

Run:
  python mlb_residual_ats.py --compare --folds 20
  python mlb_residual_ats.py --per-model           # independent feature selection
  python mlb_residual_ats.py --upload               # train + deploy
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from catboost import CatBoostRegressor
from datetime import datetime, timezone

SEED = 42
np.random.seed(SEED)
N_FOLDS = 20


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

def load_and_prepare():
    from mlb_retrain import load_data, build_features, FEATURE_COLS

    df = load_data()
    X_full = build_features(df)

    actual_margin = (df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)).values
    spreads = pd.to_numeric(df.get("market_spread", X_full.get("market_spread", 0)), errors="coerce").fillna(0).values

    # MLB run line: sometimes in market_spread_home column
    if "market_spread_home" in df.columns:
        sp2 = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values
        # Use whichever has more non-zero values
        if (np.abs(sp2) > 0.1).sum() > (np.abs(spreads) > 0.1).sum():
            spreads = sp2

    has_spread = np.abs(spreads) > 0.1
    y_residual = actual_margin + spreads

    # Season weights
    if "season" in df.columns:
        seasons = pd.to_numeric(df["season"], errors="coerce").fillna(2026).astype(int).values
        w = np.array([{0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8}.get(2026 - s, 0.7) for s in seasons])
    else:
        w = np.ones(len(df))
        seasons = np.full(len(df), 2026)

    # Sort by date
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    idx = dates.argsort()
    X_full = X_full.iloc[idx].reset_index(drop=True)
    actual_margin = actual_margin[idx]
    spreads = spreads[idx]
    has_spread = has_spread[idx]
    y_residual = y_residual[idx]
    w = w[idx]
    seasons = seasons[idx]

    # Filter to games with run line
    m = has_spread
    X = X_full[m].reset_index(drop=True)
    am = actual_margin[m]; sp = spreads[m]; yr = y_residual[m]; wt = w[m]

    # Exclude circular features
    exclude = {"market_spread", "spread_vs_market", "run_diff_pred", "has_heuristic",
               "has_market", "market_total"}
    all_feats = [f for f in X.columns if f not in exclude]
    avail = [f for f in all_feats if f in X.columns]
    X = X[avail].fillna(0)

    print(f"\n  MLB Residual dataset: {len(X)} games with run lines")
    print(f"  Features: {len(avail)} (excluded {len(exclude)} market-derived)")
    print(f"  Run line residual: mean={yr.mean():+.3f}, std={yr.std():.2f}")
    print(f"  Home cover rate: {(yr > 0).mean():.1%}")
    print(f"  Seasons: {sorted(set(seasons[m]))}")

    return X, avail, am, sp, yr, wt


# ═══════════════════════════════════════════════════════════
# SINGLE-MODEL WALK-FORWARD SCORE
# ═══════════════════════════════════════════════════════════

def make_model(name):
    if name == "Lasso":
        return Lasso(alpha=0.01, max_iter=5000, random_state=SEED)
    elif name == "Ridge":
        return Ridge(alpha=1.0, random_state=SEED)
    elif name == "ElasticNet":
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=SEED)
    elif name == "CatBoost":
        return CatBoostRegressor(depth=6, iterations=200, learning_rate=0.03,
                                 subsample=0.8, min_data_in_leaf=20,
                                 random_seed=SEED, verbose=0)


def wf_score(model_name, Xv, yr, w, am, sp, n_folds=N_FOLDS):
    n = len(Xv); fs = n // (n_folds + 1); mt = max(fs * 3, 1000)
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
    for t in [0, 0.5, 1, 1.5, 2, 3]:
        mask = ~np.isnan(oof) & ~push & (np.abs(oof) >= t)
        r[t] = ((oof[mask] > 0) == (ats[mask] > 0)).mean() if mask.sum() >= 30 else 0.5
    # MLB composite: 1+ and 1.5+ are the money tiers
    composite = r.get(1, 0.5) * 0.4 + r.get(1.5, 0.5) * 0.3 + r.get(2, 0.5) * 0.3
    return composite, r, oof


# ═══════════════════════════════════════════════════════════
# ATS EVALUATION
# ═══════════════════════════════════════════════════════════

def ats_eval(preds, am, sp, label="Model"):
    ats = am + sp; push = ats == 0
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  {'Threshold':>10} {'Games':>7} {'ATS%':>7} {'ROI%':>7}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7}")
    results = {}
    for t in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]:
        mask = ~np.isnan(preds) & ~push & (np.abs(preds) >= t)
        if mask.sum() < 30: continue
        correct = ((preds[mask] > 0) == (ats[mask] > 0)).mean()
        roi = (correct * 1.909 - 1) * 100
        print(f"  {t:>9}+ {mask.sum():>7} {correct:>6.1%} {roi:>+6.1f}%")
        results[t] = {"n": int(mask.sum()), "acc": round(float(correct), 4), "roi": round(float(roi), 1)}
    return results


# ═══════════════════════════════════════════════════════════
# FORWARD SELECTION (per model)
# ═══════════════════════════════════════════════════════════

def forward_select(model_name, X, candidates, yr, w, am, sp):
    # Start with top MLB features from existing v8
    core = [
        "woba_diff", "fip_diff", "k_bb_diff", "bullpen_era_diff",
        "park_factor", "sp_fip_spread", "sp_relative_fip_diff",
    ]
    core = [f for f in core if f in candidates]
    remaining = [f for f in candidates if f not in core]

    print(f"\n  ── {model_name} FORWARD SELECTION ──")
    core_score, cd, _ = wf_score(model_name, X[core].values, yr, w, am, sp)
    print(f"  Core ({len(core)}): {core_score:.5f} (1+={cd.get(1,0):.1%} 1.5+={cd.get(1.5,0):.1%} 2+={cd.get(2,0):.1%})")

    selected = list(core)
    best_score = core_score
    added = []

    t0 = time.time()
    for i, feat in enumerate(remaining):
        trial = selected + [feat]
        score, detail, _ = wf_score(model_name, X[trial].values, yr, w, am, sp)
        delta = score - best_score
        if delta > 0.00005:
            selected.append(feat)
            best_score = score
            added.append((feat, delta, detail))
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 1
        rem = (len(remaining) - i - 1) / rate / 60
        if (i + 1) % 5 == 0:
            print(f"    [{i+1:3d}/{len(remaining)}] {len(selected)} feats, score={best_score:.5f} ~{rem:.1f}m")

    print(f"\n  {model_name}: {len(selected)} features, score={best_score:.5f} ({best_score-core_score:+.5f})")
    print(f"  Added {len(added)} features:")
    for f, d, det in added:
        print(f"    {f:35s} d={d:+.6f} (1+={det.get(1,0):.1%} 1.5+={det.get(1.5,0):.1%})")

    return selected, added, best_score


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--per-model", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--folds", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print("  MLB RESIDUAL ATS MODEL")
    print("  Target: actual_margin + run_line (where run line is wrong)")
    print("=" * 70)

    X, feats, am, sp, yr, w = load_and_prepare()

    if args.per_model:
        # Independent feature selection per model
        lasso_feats, _, ls = forward_select("Lasso", X, feats, yr, w, am, sp)
        cb_feats, _, cs = forward_select("CatBoost", X, feats, yr, w, am, sp)

        # Save
        for name, fl in [("lasso", lasso_feats), ("catboost", cb_feats)]:
            with open(f"mlb_v2_{name}_features.txt", "w") as f:
                for feat in fl:
                    f.write(feat + "\n")
            print(f"  Saved {len(fl)} features to mlb_v2_{name}_features.txt")

        # Validate ensemble
        print(f"\n{'='*70}")
        print(f"  INDEPENDENT ENSEMBLE VALIDATION (25-fold)")
        print(f"{'='*70}")

        n = len(X); nf = 25; fs = n // (nf + 1); mt = max(fs * 3, 1000)
        oof_l = np.full(n, np.nan); oof_c = np.full(n, np.nan)
        for fold in range(nf):
            ts = mt + fold * fs; te = min(ts + fs, n)
            if ts >= n: break
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[lasso_feats].values[:ts])
            Xte = sc.transform(X[lasso_feats].values[ts:te])
            ml = Lasso(alpha=0.01, max_iter=5000, random_state=SEED)
            ml.fit(Xtr, yr[:ts], sample_weight=w[:ts])
            oof_l[ts:te] = ml.predict(Xte)

            sc2 = StandardScaler()
            Xtr2 = sc2.fit_transform(X[cb_feats].values[:ts])
            Xte2 = sc2.transform(X[cb_feats].values[ts:te])
            mc = CatBoostRegressor(depth=6, iterations=200, learning_rate=0.03,
                                   subsample=0.8, min_data_in_leaf=20,
                                   random_seed=SEED, verbose=0)
            mc.fit(Xtr2, yr[:ts], sample_weight=w[:ts])
            oof_c[ts:te] = mc.predict(Xte2)
            if (fold + 1) % 5 == 0:
                print(f"    Fold {fold+1}/{nf}")

        # Weight sweep
        print(f"\n  WEIGHT SWEEP:")
        print(f"  {'Weights':>20} {'ATS@1':>7} {'ATS@1.5':>8} {'ATS@2':>7}")
        best_w = None; best_comp = 0
        ats = am + sp; push = ats == 0
        for cw in np.arange(0.1, 1.0, 0.1):
            lw = round(1.0 - cw, 2)
            blend = cw * oof_c + lw * oof_l
            r = {}
            for t in [1, 1.5, 2]:
                mask = ~np.isnan(blend) & ~push & (np.abs(blend) >= t)
                r[t] = ((blend[mask] > 0) == (ats[mask] > 0)).mean() if mask.sum() >= 30 else 0.5
            comp = r[1] * 0.4 + r[1.5] * 0.3 + r[2] * 0.3
            if comp > best_comp:
                best_comp = comp; best_w = (cw, lw)
            print(f"  CB={cw:.1f}/L={lw:.1f}      {r[1]:>6.1%} {r[1.5]:>7.1%} {r[2]:>6.1%}  comp={comp:.5f}")

        print(f"\n  ✅ Best weights: CB={best_w[0]:.1f} / Lasso={best_w[1]:.1f}")

        # Full eval at best weights
        blend = best_w[0] * oof_c + best_w[1] * oof_l
        ats_eval(blend, am, sp, f"ENSEMBLE CB={best_w[0]:.1f}/L={best_w[1]:.1f}")
        ats_eval(oof_c, am, sp, "CatBoost solo")
        ats_eval(oof_l, am, sp, "Lasso solo")

    elif args.compare:
        # Quick comparison: residual vs direct on ALL features
        print(f"\n  Running {args.folds}-fold walk-forward comparison...")
        _, _, oof_res = wf_score("CatBoost", X.values, yr, w, am, sp, n_folds=args.folds)
        _, _, oof_dir = wf_score("CatBoost", X.values, am, w, am, sp, n_folds=args.folds)

        ats_eval(oof_res, am, sp, "RESIDUAL CatBoost (predicts where run line is wrong)")
        direct_edge = oof_dir - (-sp)
        ats_eval(direct_edge, am, sp, "DIRECT CatBoost (predicts margin, compares to RL)")

        valid = ~np.isnan(oof_res)
        if valid.sum() > 100:
            res_mae = np.mean(np.abs(oof_res[valid] - (am[valid] + sp[valid])))
            dir_mae = np.mean(np.abs(oof_dir[valid] - am[valid]))
            mkt_mae = np.mean(np.abs(-sp[valid] - am[valid]))
            print(f"\n  Residual MAE: {res_mae:.3f}")
            print(f"  Direct MAE:   {dir_mae:.3f}")
            print(f"  Market MAE:   {mkt_mae:.3f}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
