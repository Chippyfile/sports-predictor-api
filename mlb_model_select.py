#!/usr/bin/env python3
"""
mlb_model_select.py — Test multiple learners + hyperparameters + ensembles
═══════════════════════════════════════════════════════════════════════════
Learners:
  - CatBoost (current) — multiple depth/iter/lr combos
  - XGBoost
  - LightGBM
  - Ridge / Lasso / ElasticNet (linear — won on NBA data)
  - MLP (neural net)
  - Random Forest
  - Ensemble combinations

All evaluated via walk-forward validation (no data leakage).

Usage:
  python3 mlb_model_select.py               # Full sweep
  python3 mlb_model_select.py --quick       # Top configs only (~5 min)
  python3 mlb_model_select.py --ensemble    # Just test ensemble of top models
"""
import sys, time, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("⚠️ CatBoost not installed — skipping CatBoost configs")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed — pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM not installed — pip install lightgbm")

from mlb_retrain import build_features, load_data

SEED = 42
N_FOLDS = 15

# ═══════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════
CONFIGS = []

# ── CatBoost variants ──
if HAS_CAT:
    for depth in [3, 4, 5, 6]:
        for iters in [50, 100, 200, 400]:
            for lr in [0.03, 0.06, 0.1]:
                CONFIGS.append({
                    "name": f"CatBoost_d{depth}_i{iters}_lr{lr}",
                    "family": "CatBoost",
                    "model": lambda d=depth, i=iters, l=lr: CatBoostRegressor(
                        depth=d, iterations=i, learning_rate=l,
                        subsample=0.8, min_data_in_leaf=20,
                        random_seed=SEED, verbose=0,
                    ),
                    "needs_scale": True,
                })

# ── XGBoost variants ──
if HAS_XGB:
    for depth in [3, 4, 5, 6]:
        for n_est in [100, 200, 400]:
            for lr in [0.03, 0.06, 0.1]:
                CONFIGS.append({
                    "name": f"XGB_d{depth}_n{n_est}_lr{lr}",
                    "family": "XGBoost",
                    "model": lambda d=depth, n=n_est, l=lr: XGBRegressor(
                        max_depth=d, n_estimators=n, learning_rate=l,
                        subsample=0.8, min_child_weight=20,
                        random_state=SEED, verbosity=0,
                    ),
                    "needs_scale": True,
                })

# ── LightGBM variants ──
if HAS_LGB:
    for leaves in [15, 31, 63]:
        for n_est in [100, 200, 400]:
            for lr in [0.03, 0.06, 0.1]:
                CONFIGS.append({
                    "name": f"LGBM_l{leaves}_n{n_est}_lr{lr}",
                    "family": "LightGBM",
                    "model": lambda lv=leaves, n=n_est, l=lr: LGBMRegressor(
                        num_leaves=lv, n_estimators=n, learning_rate=l,
                        subsample=0.8, min_child_samples=20,
                        random_state=SEED, verbose=-1,
                    ),
                    "needs_scale": True,
                })

# ── Linear models ──
for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    CONFIGS.append({
        "name": f"Ridge_{alpha}",
        "family": "Ridge",
        "model": lambda a=alpha: Ridge(alpha=a),
        "needs_scale": True,
    })
    CONFIGS.append({
        "name": f"Lasso_{alpha}",
        "family": "Lasso",
        "model": lambda a=alpha: Lasso(alpha=a, max_iter=5000),
        "needs_scale": True,
    })

for alpha in [0.01, 0.1, 0.5, 1.0]:
    for ratio in [0.3, 0.5, 0.7, 0.9]:
        CONFIGS.append({
            "name": f"ElasticNet_a{alpha}_r{ratio}",
            "family": "ElasticNet",
            "model": lambda a=alpha, r=ratio: ElasticNet(alpha=a, l1_ratio=r, max_iter=5000),
            "needs_scale": True,
        })

# ── MLP variants ──
for hidden in [(128, 64), (256, 128), (256, 128, 64), (512, 256, 128)]:
    for lr in [0.001, 0.003]:
        name = "-".join(str(h) for h in hidden)
        CONFIGS.append({
            "name": f"MLP_{name}_lr{lr}",
            "family": "MLP",
            "model": lambda h=hidden, l=lr: MLPRegressor(
                hidden_layer_sizes=h, learning_rate_init=l,
                max_iter=500, early_stopping=True,
                random_state=SEED, verbose=False,
            ),
            "needs_scale": True,
        })

# ── Random Forest ──
for n_est in [100, 200, 500]:
    for depth in [6, 10, 15, None]:
        CONFIGS.append({
            "name": f"RF_n{n_est}_d{depth}",
            "family": "RF",
            "model": lambda n=n_est, d=depth: RandomForestRegressor(
                n_estimators=n, max_depth=d, min_samples_leaf=20,
                random_state=SEED, n_jobs=-1,
            ),
            "needs_scale": False,
        })

# ── GBM (sklearn) ──
for n_est in [100, 200]:
    for depth in [3, 4, 5]:
        for lr in [0.05, 0.1]:
            CONFIGS.append({
                "name": f"GBM_n{n_est}_d{depth}_lr{lr}",
                "family": "GBM",
                "model": lambda n=n_est, d=depth, l=lr: GradientBoostingRegressor(
                    n_estimators=n, max_depth=d, learning_rate=l,
                    min_samples_leaf=20, random_state=SEED,
                ),
                "needs_scale": False,
            })

QUICK_CONFIGS = [c for c in CONFIGS if any(q in c["name"] for q in [
    # Best-guess configs per family
    "CatBoost_d4_i100_lr0.06", "CatBoost_d4_i200_lr0.06", "CatBoost_d5_i200_lr0.06",
    "CatBoost_d4_i400_lr0.03", "CatBoost_d6_i200_lr0.03", "CatBoost_d3_i200_lr0.1",
    "XGB_d4_n200_lr0.06", "XGB_d5_n200_lr0.03", "XGB_d4_n400_lr0.03",
    "LGBM_l31_n200_lr0.06", "LGBM_l63_n200_lr0.03", "LGBM_l15_n400_lr0.06",
    "Ridge_0.1", "Ridge_1.0", "Ridge_10.0",
    "Lasso_0.01", "Lasso_0.1", "Lasso_1.0",
    "ElasticNet_a0.1_r0.5", "ElasticNet_a0.1_r0.9",
    "MLP_256-128_lr0.001", "MLP_256-128-64_lr0.001",
    "RF_n200_d10", "RF_n200_d15",
    "GBM_n200_d4_lr0.1", "GBM_n200_d5_lr0.05",
])]


def walk_forward_eval(X, y, weights, model_fn, needs_scale=True, n_folds=N_FOLDS):
    """Walk-forward validation for a model. Returns metrics dict."""
    fold_size = len(X) // (n_folds + 3)
    min_train = fold_size * 3
    preds = np.full(len(X), np.nan)
    
    for fold in range(n_folds):
        te_s = min_train + fold * fold_size
        te_e = min(te_s + fold_size, len(X))
        if te_s >= len(X):
            break
        
        if needs_scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[:te_s])
            Xte = sc.transform(X[te_s:te_e])
        else:
            Xtr = X[:te_s]
            Xte = X[te_s:te_e]
        
        wt = weights[:te_s] if weights is not None else None
        m = model_fn()
        try:
            if hasattr(m, 'fit') and 'sample_weight' in m.fit.__code__.co_varnames:
                m.fit(Xtr, y[:te_s], sample_weight=wt)
            else:
                m.fit(Xtr, y[:te_s])
        except TypeError:
            m.fit(Xtr, y[:te_s])
        preds[te_s:te_e] = m.predict(Xte)
    
    valid = ~np.isnan(preds)
    pv, tv = preds[valid], y[valid]
    mae = float(np.mean(np.abs(pv - tv)))
    win_acc = float(((pv > 0) == (tv > 0)).sum() / (tv != 0).sum() * 100)
    bias = float(np.mean(tv - pv))
    
    return {"mae": mae, "win_acc": win_acc, "bias": bias}


def test_ensembles(X, y, weights, top_models):
    """Test ensemble combinations of top models."""
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE TESTING (top {len(top_models)} models)")
    print(f"{'='*70}")
    
    fold_size = len(X) // (N_FOLDS + 3)
    min_train = fold_size * 3
    
    # Collect per-fold predictions for each model
    model_preds = {}
    for config in top_models:
        preds = np.full(len(X), np.nan)
        for fold in range(N_FOLDS):
            te_s = min_train + fold * fold_size
            te_e = min(te_s + fold_size, len(X))
            if te_s >= len(X):
                break
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[:te_s])
            Xte = sc.transform(X[te_s:te_e])
            wt = weights[:te_s] if weights is not None else None
            m = config["model"]()
            try:
                m.fit(Xtr, y[:te_s], sample_weight=wt)
            except TypeError:
                m.fit(Xtr, y[:te_s])
            preds[te_s:te_e] = m.predict(Xte)
        model_preds[config["name"]] = preds
    
    valid = ~np.isnan(list(model_preds.values())[0])
    tv = y[valid]
    
    # Test all pairs and triples
    names = list(model_preds.keys())
    results = []
    
    # All pairs
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            avg = (model_preds[names[i]][valid] + model_preds[names[j]][valid]) / 2
            mae = float(np.mean(np.abs(avg - tv)))
            win = float(((avg > 0) == (tv > 0)).sum() / (tv != 0).sum() * 100)
            results.append({"combo": f"{names[i]} + {names[j]}", "n": 2, "mae": mae, "win_acc": win})
    
    # All triples (from top 6 only to keep tractable)
    top6 = names[:6]
    for i in range(len(top6)):
        for j in range(i+1, len(top6)):
            for k in range(j+1, len(top6)):
                avg = (model_preds[top6[i]][valid] + model_preds[top6[j]][valid] + model_preds[top6[k]][valid]) / 3
                mae = float(np.mean(np.abs(avg - tv)))
                win = float(((avg > 0) == (tv > 0)).sum() / (tv != 0).sum() * 100)
                results.append({"combo": f"{top6[i]} + {top6[j]} + {top6[k]}", "n": 3, "mae": mae, "win_acc": win})
    
    # Top 4 and 5 model ensembles
    for n in [4, 5]:
        if len(names) >= n:
            avg = np.mean([model_preds[names[i]][valid] for i in range(n)], axis=0)
            mae = float(np.mean(np.abs(avg - tv)))
            win = float(((avg > 0) == (tv > 0)).sum() / (tv != 0).sum() * 100)
            results.append({"combo": " + ".join(names[:n]), "n": n, "mae": mae, "win_acc": win})
    
    results.sort(key=lambda x: x["mae"])
    
    print(f"\n  {'Rank':<5} {'Ensemble':<60} {'MAE':>7} {'Win%':>7}")
    print(f"  {'─'*5} {'─'*60} {'─'*7} {'─'*7}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:<5} {r['combo']:<60} {r['mae']:>7.4f} {r['win_acc']:>6.2f}%")
    
    return results


def main():
    quick = "--quick" in sys.argv
    ensemble_only = "--ensemble" in sys.argv
    
    print("=" * 70)
    print("  MLB MODEL SELECTION + HYPERPARAMETER SWEEP")
    print("=" * 70)
    
    df = load_data()
    y = df["target_margin"].values
    weights = df["season_weight"].values if "season_weight" in df.columns else None
    X_df = build_features(df)
    X = X_df.values
    print(f"\n  Features: {len(X_df.columns)}, Games: {len(X)}")
    
    configs = QUICK_CONFIGS if quick else CONFIGS
    print(f"  Testing {len(configs)} configurations {'(quick mode)' if quick else '(full sweep)'}")
    
    if not ensemble_only:
        results = []
        t0 = time.time()
        
        for i, config in enumerate(configs):
            ct = time.time()
            try:
                metrics = walk_forward_eval(X, y, weights, config["model"], config.get("needs_scale", True))
                metrics["name"] = config["name"]
                metrics["family"] = config["family"]
                results.append(metrics)
                elapsed = time.time() - ct
                total_elapsed = time.time() - t0
                eta = (total_elapsed / (i+1)) * (len(configs) - i - 1)
                print(f"  [{i+1}/{len(configs)}] {config['name']:<45} MAE={metrics['mae']:.4f}  Win={metrics['win_acc']:.2f}%  ({elapsed:.1f}s, ETA {eta/60:.0f}m)")
            except Exception as e:
                print(f"  [{i+1}/{len(configs)}] {config['name']:<45} ❌ {str(e)[:50]}")
        
        # Sort by MAE
        results.sort(key=lambda x: x["mae"])
        
        print(f"\n{'='*70}")
        print(f"  TOP 20 MODELS (by MAE)")
        print(f"{'='*70}")
        print(f"\n  {'Rank':<5} {'Model':<45} {'Family':<12} {'MAE':>7} {'Win%':>7} {'Bias':>7}")
        print(f"  {'─'*5} {'─'*45} {'─'*12} {'─'*7} {'─'*7} {'─'*7}")
        for i, r in enumerate(results[:20]):
            marker = " ← BEST" if i == 0 else ""
            print(f"  {i+1:<5} {r['name']:<45} {r['family']:<12} {r['mae']:>7.4f} {r['win_acc']:>6.2f}% {r['bias']:>+6.3f}{marker}")
        
        # Best per family
        print(f"\n  BEST PER FAMILY:")
        seen = set()
        for r in results:
            if r["family"] not in seen:
                seen.add(r["family"])
                print(f"    {r['family']:<12}: {r['name']:<45} MAE={r['mae']:.4f} Win={r['win_acc']:.2f}%")
        
        # Save results
        with open("mlb_model_selection_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to mlb_model_selection_results.json")
    else:
        # Load previous results for ensemble testing
        with open("mlb_model_selection_results.json") as f:
            results = json.load(f)
    
    # ── Ensemble testing with top models ──
    # Pick best from each family
    top_for_ensemble = []
    seen_families = set()
    for r in results:
        if r["family"] not in seen_families and len(top_for_ensemble) < 8:
            seen_families.add(r["family"])
            # Find the matching config
            matching = [c for c in CONFIGS if c["name"] == r["name"]]
            if matching:
                top_for_ensemble.append(matching[0])
    
    # Also add top 3 overall if not already included
    for r in results[:3]:
        matching = [c for c in CONFIGS if c["name"] == r["name"]]
        if matching and matching[0] not in top_for_ensemble:
            top_for_ensemble.append(matching[0])
    
    if len(top_for_ensemble) >= 2:
        ensemble_results = test_ensembles(X, y, weights, top_for_ensemble)
        
        with open("mlb_ensemble_results.json", "w") as f:
            json.dump(ensemble_results[:20], f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  COMPLETE — check mlb_model_selection_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
