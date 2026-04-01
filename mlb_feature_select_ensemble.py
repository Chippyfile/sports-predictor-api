#!/usr/bin/env python3
"""
mlb_feature_select_ensemble.py — Feature selection against the actual 3-model ensemble
Tests dropping each feature from Lasso+ElasticNet+CatBoost ensemble.
Also tests adding back Lasso-zeroed features to see if ensemble benefits.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from catboost import CatBoostRegressor

from mlb_retrain import build_features, load_data, FEATURE_COLS

SEED = 42
N_FOLDS = 15

def ensemble_walk_forward(X_df, y, weights, features, n_folds=N_FOLDS):
    """Walk-forward eval with the 3-model ensemble on given feature subset."""
    X = X_df[features].values
    fold_size = len(X) // (n_folds + 3)
    min_train = fold_size * 3
    
    preds_lasso = np.full(len(X), np.nan)
    preds_en = np.full(len(X), np.nan)
    preds_cat = np.full(len(X), np.nan)
    
    for fold in range(n_folds):
        te_s = min_train + fold * fold_size
        te_e = min(te_s + fold_size, len(X))
        if te_s >= len(X):
            break
        
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:te_s])
        Xte = sc.transform(X[te_s:te_e])
        wt = weights[:te_s] if weights is not None else None
        
        # Lasso
        m1 = Lasso(alpha=0.01, max_iter=5000)
        m1.fit(Xtr, y[:te_s])
        preds_lasso[te_s:te_e] = m1.predict(Xte)
        
        # ElasticNet
        m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        m2.fit(Xtr, y[:te_s])
        preds_en[te_s:te_e] = m2.predict(Xte)
        
        # CatBoost
        m3 = CatBoostRegressor(
            depth=6, iterations=200, learning_rate=0.03,
            subsample=0.8, min_data_in_leaf=20,
            random_seed=SEED, verbose=0,
        )
        m3.fit(Xtr, y[:te_s], sample_weight=wt)
        preds_cat[te_s:te_e] = m3.predict(Xte)
    
    valid = ~np.isnan(preds_lasso)
    avg = (preds_lasso[valid] + preds_en[valid] + preds_cat[valid]) / 3
    tv = y[valid]
    
    mae = float(np.mean(np.abs(avg - tv)))
    win_acc = float(((avg > 0) == (tv > 0)).sum() / (tv != 0).sum() * 100)
    
    return {"mae": mae, "win_acc": win_acc}


def main():
    print("=" * 70)
    print("  MLB ENSEMBLE FEATURE SELECTION")
    print("  Lasso_0.01 + ElasticNet_a0.1_r0.5 + CatBoost_d6_i200_lr0.03")
    print("=" * 70)
    
    df = load_data()
    y = df["target_margin"].values
    weights = df["season_weight"].values if "season_weight" in df.columns else None
    X_df = build_features(df)
    all_features = list(X_df.columns)
    
    # ── Phase 1: Baseline with all 41 ──
    print(f"\n  Phase 1: Baseline (all {len(all_features)} features)...")
    t0 = time.time()
    baseline = ensemble_walk_forward(X_df, y, weights, all_features)
    print(f"  Baseline: MAE={baseline['mae']:.4f}, Win={baseline['win_acc']:.2f}% ({time.time()-t0:.0f}s)")
    
    # ── Phase 2: Drop each feature one at a time ──
    print(f"\n  Phase 2: Ablation (drop each feature from ensemble)...")
    ablation = []
    for i, feat in enumerate(all_features):
        subset = [f for f in all_features if f != feat]
        t1 = time.time()
        result = ensemble_walk_forward(X_df, y, weights, subset)
        delta_mae = result["mae"] - baseline["mae"]
        delta_win = result["win_acc"] - baseline["win_acc"]
        ablation.append({
            "feature": feat,
            "mae_without": result["mae"],
            "delta_mae": delta_mae,
            "win_without": result["win_acc"],
            "delta_win": delta_win,
        })
        elapsed = time.time() - t1
        tag = "HELPS→KEEP" if delta_mae > 0.001 else "HURTS→DROP" if delta_mae < -0.001 else "NEUTRAL"
        print(f"    [{i+1}/{len(all_features)}] Drop {feat:<35} ΔMAE={delta_mae:>+.4f} ΔWin={delta_win:>+.2f}%  {tag}  ({elapsed:.0f}s)")
    
    # Sort by impact
    ablation.sort(key=lambda x: x["delta_mae"], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"  ABLATION RESULTS (sorted by impact on ensemble)")
    print(f"{'='*70}")
    
    helpful = []
    neutral = []
    harmful = []
    
    print(f"\n  {'Feature':<35} {'ΔMAE':>8} {'ΔWin%':>8} {'Verdict':<15}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*15}")
    for r in ablation:
        if r["delta_mae"] > 0.001:
            verdict = "✅ KEEP (helps)"
            helpful.append(r["feature"])
        elif r["delta_mae"] < -0.001:
            verdict = "❌ DROP (hurts)"
            harmful.append(r["feature"])
        else:
            verdict = "⚪ NEUTRAL"
            neutral.append(r["feature"])
        print(f"  {r['feature']:<35} {r['delta_mae']:>+8.4f} {r['delta_win']:>+8.2f}% {verdict}")
    
    print(f"\n  Summary: {len(helpful)} helpful, {len(neutral)} neutral, {len(harmful)} harmful")
    
    # ── Phase 3: Progressive elimination of harmful features ──
    if harmful:
        print(f"\n  Phase 3: Dropping all {len(harmful)} harmful features and re-testing...")
        keep = [f for f in all_features if f not in harmful]
        trimmed = ensemble_walk_forward(X_df, y, weights, keep)
        print(f"  All 41:     MAE={baseline['mae']:.4f}, Win={baseline['win_acc']:.2f}%")
        print(f"  Trimmed {len(keep)}:  MAE={trimmed['mae']:.4f}, Win={trimmed['win_acc']:.2f}%")
        print(f"  Δ MAE: {trimmed['mae'] - baseline['mae']:+.4f}")
        print(f"  Δ Win: {trimmed['win_acc'] - baseline['win_acc']:+.2f}%")
        
        # Try also dropping neutrals with very small positive delta_mae (redundant)
        keep_tight = [f for f in all_features if f in helpful or (f in neutral and 
            any(a["feature"] == f and a["delta_mae"] >= 0 for a in ablation))]
        if len(keep_tight) < len(keep):
            tight = ensemble_walk_forward(X_df, y, weights, keep_tight)
            print(f"\n  Tight {len(keep_tight)}:   MAE={tight['mae']:.4f}, Win={tight['win_acc']:.2f}%")
    
    # ── Phase 4: Final recommendation ──
    print(f"\n{'='*70}")
    print(f"  RECOMMENDED FEATURE SET")
    print(f"{'='*70}")
    final = helpful + neutral  # keep helpful + neutral, drop harmful
    print(f"\n  {len(final)} features (dropped {len(harmful)} harmful):")
    print(f"  FEATURE_COLS = [")
    for f in final:
        tag = "  # strong signal" if f in helpful[:10] else ""
        print(f'    "{f}",{tag}')
    print(f"  ]")
    
    if harmful:
        print(f"\n  DROPPED ({len(harmful)}):")
        for f in harmful:
            r = next(a for a in ablation if a["feature"] == f)
            print(f"    {f}: ΔMAE={r['delta_mae']:+.4f} (removing IMPROVES ensemble)")
    
    # Save
    import json
    with open("mlb_ensemble_feature_results.json", "w") as f:
        json.dump({
            "baseline": baseline,
            "ablation": ablation,
            "helpful": helpful,
            "neutral": neutral,
            "harmful": harmful,
            "recommended": final,
        }, f, indent=2)
    print(f"\n  Saved to mlb_ensemble_feature_results.json")


if __name__ == "__main__":
    main()
