#!/usr/bin/env python3
"""
nba_ats_eliminate.py — ATS-optimized feature selection pipeline

Step 1: Correlation audit → group features with |r| > 0.7
Step 2: Solo ATS lift → rank each feature by standalone ATS improvement over spread-only baseline
Step 3: Greedy forward selection → add features one at a time, keep only if ATS improves
Step 4: Backward pruning → try removing each feature, drop if ATS doesn't decrease
Step 5: Final validation on optimal subset

All evaluation uses walk-forward CV with ATS at 4+/7+/10u thresholds.

Usage:
    python3 nba_ats_eliminate.py                  # Full pipeline
    python3 nba_ats_eliminate.py --step1          # Correlation audit only
    python3 nba_ats_eliminate.py --step2          # Solo lift only
    python3 nba_ats_eliminate.py --step3          # Forward selection (needs step2 results)
    python3 nba_ats_eliminate.py --step4          # Backward pruning (needs step3 results)
    python3 nba_ats_eliminate.py --final          # Final validation
"""

import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from nba_build_features_v27 import load_training_data, build_features

RESULTS_FILE = "nba_ats_elimination_results.json"
CATBOOST_PARAMS = {"depth": 4, "iterations": 500, "learning_rate": 0.05,
                   "l2_leaf_reg": 3, "random_seed": 42, "verbose": 0}


# ═══════════════════════════════════════════════════════════
# CORE: Walk-forward ATS evaluator
# ═══════════════════════════════════════════════════════════

def ats_eval(X, y, spreads, n_folds=10, model_type="catboost", return_details=False):
    """Walk-forward ATS evaluation. Returns dict of threshold → accuracy.
    
    Uses fewer folds for speed during elimination (10 vs 30 for final).
    """
    from sklearn.preprocessing import StandardScaler
    
    n = len(X)
    fold_size = n // (n_folds + 2)
    min_train = fold_size * 3  # need enough training data
    
    all_preds = []
    all_actual = []
    all_spreads = []
    
    for fold in range(n_folds):
        test_start = min_train + fold * fold_size
        test_end = min(test_start + fold_size, n)
        if test_end > n or test_start >= n:
            break
        
        X_train = X[:test_start]
        y_train = y[:test_start]
        X_test = X[test_start:test_end]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        if model_type == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**CATBOOST_PARAMS)
            model.fit(X_train_s, y_train)
        else:
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=0.1, max_iter=5000)
            model.fit(X_train_s, y_train)
        
        preds = model.predict(X_test_s)
        all_preds.extend(preds)
        all_actual.extend(y[test_start:test_end])
        all_spreads.extend(spreads[test_start:test_end])
    
    all_preds = np.array(all_preds)
    all_actual = np.array(all_actual)
    all_spreads = np.array(all_spreads)
    
    # ATS metrics
    ats_edge = all_preds - (-all_spreads)  # model margin vs spread
    ats_margin = all_actual + all_spreads    # actual ATS result
    not_push = ats_margin != 0
    ats_correct = np.sign(ats_edge) == np.sign(ats_margin)
    
    mae = np.mean(np.abs(all_preds - all_actual))
    
    results = {"mae": round(mae, 3), "n_games": len(all_preds)}
    
    for threshold in [0, 2, 4, 7, 10]:
        mask = (np.abs(ats_edge) >= threshold) & not_push
        n_picks = mask.sum()
        if n_picks >= 20:  # minimum sample
            acc = ats_correct[mask].mean()
            roi = (acc * 1.909 - 1) * 100
            results[f"ats_{threshold}"] = round(acc, 4)
            results[f"roi_{threshold}"] = round(roi, 1)
            results[f"n_{threshold}"] = int(n_picks)
        else:
            results[f"ats_{threshold}"] = 0
            results[f"roi_{threshold}"] = -100
            results[f"n_{threshold}"] = int(n_picks)
    
    # Composite ATS score: weighted average of ROI at different thresholds
    # Weight higher thresholds more (where edge matters most)
    w = {4: 0.3, 7: 0.4, 10: 0.3}
    composite = sum(results.get(f"roi_{t}", -100) * wt for t, wt in w.items())
    results["composite_ats"] = round(composite, 2)
    
    if return_details:
        results["preds"] = all_preds.tolist()
        results["actual"] = all_actual.tolist()
    
    return results


# ═══════════════════════════════════════════════════════════
# STEP 1: Correlation Audit
# ═══════════════════════════════════════════════════════════

def step1_correlation_audit(X, feature_names, threshold=0.7):
    """Find groups of highly correlated features."""
    print(f"\n{'='*70}")
    print(f"  STEP 1: Correlation Audit (|r| > {threshold})")
    print(f"{'='*70}")
    
    corr = X.corr().abs()
    
    # Find all pairs above threshold
    pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            r = corr.iloc[i, j]
            if r > threshold:
                pairs.append((feature_names[i], feature_names[j], round(r, 3)))
    
    pairs.sort(key=lambda x: -x[2])
    
    print(f"\n  Found {len(pairs)} highly correlated pairs:")
    for f1, f2, r in pairs[:30]:
        print(f"    {f1:35s} ↔ {f2:35s}  r={r:.3f}")
    if len(pairs) > 30:
        print(f"    ... and {len(pairs) - 30} more")
    
    # Build groups using union-find
    parent = {f: f for f in feature_names}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    for f1, f2, r in pairs:
        union(f1, f2)
    
    groups = {}
    for f in feature_names:
        root = find(f)
        if root not in groups:
            groups[root] = []
        groups[root].append(f)
    
    # Only show groups with 2+ members
    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    
    print(f"\n  Correlated feature groups ({len(multi_groups)} groups with 2+ members):")
    group_list = []
    for i, (root, members) in enumerate(sorted(multi_groups.items(), key=lambda x: -len(x[1]))):
        print(f"\n  Group {i+1} ({len(members)} features):")
        for m in members:
            print(f"    • {m}")
        group_list.append(members)
    
    # Singletons
    singletons = [f for f in feature_names if len(groups[find(f)]) == 1]
    print(f"\n  Uncorrelated singletons: {len(singletons)} features")
    
    return {"pairs": pairs, "groups": group_list, "singletons": singletons}


# ═══════════════════════════════════════════════════════════
# STEP 2: Solo ATS Lift
# ═══════════════════════════════════════════════════════════

def step2_solo_ats_lift(X, y, spreads, feature_names):
    """Test each feature alone (with market_spread as anchor) for ATS lift."""
    print(f"\n{'='*70}")
    print(f"  STEP 2: Solo ATS Lift (each feature + market_spread)")
    print(f"{'='*70}")
    
    # Baseline: market_spread alone
    X_base = X[["market_spread"]].values
    print(f"\n  Computing baseline (market_spread only)...")
    baseline = ats_eval(X_base, y, spreads, n_folds=10)
    print(f"  Baseline: MAE {baseline['mae']}, ATS 4+: {baseline.get('ats_4', 0):.1%}, "
          f"ATS 7+: {baseline.get('ats_7', 0):.1%}, composite: {baseline['composite_ats']:.1f}")
    
    # Test each feature paired with market_spread
    results = []
    total = len(feature_names)
    
    for i, feat in enumerate(feature_names):
        if feat == "market_spread":
            continue
        
        t0 = time.time()
        cols = ["market_spread", feat]
        X_test = X[cols].values
        
        r = ats_eval(X_test, y, spreads, n_folds=10)
        r["feature"] = feat
        r["lift_composite"] = round(r["composite_ats"] - baseline["composite_ats"], 2)
        r["lift_ats4"] = round((r.get("ats_4", 0) - baseline.get("ats_4", 0)) * 100, 2)
        r["lift_ats7"] = round((r.get("ats_7", 0) - baseline.get("ats_7", 0)) * 100, 2)
        r["lift_ats10"] = round((r.get("ats_10", 0) - baseline.get("ats_10", 0)) * 100, 2)
        results.append(r)
        
        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == total - 1:
            remaining = elapsed * (total - i - 1)
            print(f"  [{i+1}/{total}] {feat:40s} composite_lift={r['lift_composite']:+.1f}  "
                  f"ATS4+: {r['lift_ats4']:+.1f}pp  ATS7+: {r['lift_ats7']:+.1f}pp  "
                  f"(~{remaining/60:.0f}m left)")
    
    # Rank by composite lift
    results.sort(key=lambda x: -x["lift_composite"])
    
    print(f"\n  {'='*70}")
    print(f"  Solo ATS Lift Rankings (vs spread-only baseline)")
    print(f"  {'='*70}")
    print(f"  Baseline composite: {baseline['composite_ats']:.1f}")
    print(f"\n  {'Rank':>4s} {'Feature':40s} {'Composite':>10s} {'ATS4+':>8s} {'ATS7+':>8s} {'ATS10+':>8s}")
    print(f"  {'-'*74}")
    
    positive_lift = 0
    for i, r in enumerate(results):
        marker = "✅" if r["lift_composite"] > 0 else "❌"
        if r["lift_composite"] > 0:
            positive_lift += 1
        print(f"  {i+1:4d} {r['feature']:40s} {r['lift_composite']:+10.1f} "
              f"{r['lift_ats4']:+8.1f} {r['lift_ats7']:+8.1f} {r['lift_ats10']:+8.1f} {marker}")
    
    print(f"\n  Features with positive ATS lift: {positive_lift}/{len(results)}")
    
    # Harmful features (negative lift)
    harmful = [r for r in results if r["lift_composite"] < -1]
    if harmful:
        print(f"\n  HARMFUL features (composite lift < -1):")
        for r in harmful:
            print(f"    ❌ {r['feature']:40s} {r['lift_composite']:+.1f}")
    
    return {"baseline": baseline, "solo_results": results}


# ═══════════════════════════════════════════════════════════
# STEP 3: Greedy Forward Selection
# ═══════════════════════════════════════════════════════════

def step3_forward_selection(X, y, spreads, feature_names, solo_results, 
                            corr_groups, max_features=80):
    """Greedy forward selection: add features one at a time, keep only if ATS improves."""
    print(f"\n{'='*70}")
    print(f"  STEP 3: Forward Selection (ATS-optimized)")
    print(f"{'='*70}")
    
    # Start with market_spread (always included)
    selected = ["market_spread"]
    
    # Rank candidates by solo lift (positive only)
    candidates = [r["feature"] for r in solo_results 
                  if r["lift_composite"] > 0 and r["feature"] != "market_spread"]
    
    print(f"  Candidates with positive solo lift: {len(candidates)}")
    print(f"  Starting with: {selected}")
    
    # Track which correlated group members are already selected
    # If we select one from a group, skip others in that group
    group_membership = {}
    for group in corr_groups:
        for f in group:
            group_membership[f] = set(group)
    
    current_score = ats_eval(X[selected].values, y, spreads, n_folds=10)["composite_ats"]
    print(f"  Starting composite ATS: {current_score:.1f}")
    
    selection_log = [{"step": 0, "feature": "market_spread", "composite": current_score, 
                      "n_features": 1}]
    
    # Track correlated features already covered
    covered_groups = set()
    
    stall_count = 0
    max_stall = 10  # stop if 10 consecutive features don't improve
    
    for rank, feat in enumerate(candidates):
        if len(selected) >= max_features:
            print(f"\n  Hit max features ({max_features}), stopping")
            break
        
        if stall_count >= max_stall:
            print(f"\n  {max_stall} consecutive non-improvements, stopping")
            break
        
        # Skip if a correlated partner is already selected
        if feat in group_membership:
            group = group_membership[feat]
            already_in = group & set(selected)
            if already_in:
                print(f"  [{len(selected)+1}] SKIP {feat:40s} (correlated with {already_in.pop()})")
                continue
        
        # Test adding this feature
        test_features = selected + [feat]
        X_test = X[test_features].values
        
        r = ats_eval(X_test, y, spreads, n_folds=10)
        new_score = r["composite_ats"]
        delta = new_score - current_score
        
        if delta > 0.1:  # minimum improvement threshold
            selected.append(feat)
            current_score = new_score
            stall_count = 0
            selection_log.append({
                "step": len(selected), "feature": feat,
                "composite": round(current_score, 2),
                "delta": round(delta, 2),
                "mae": r["mae"],
                "ats_4": r.get("ats_4", 0),
                "ats_7": r.get("ats_7", 0),
                "ats_10": r.get("ats_10", 0),
                "n_features": len(selected)
            })
            print(f"  [{len(selected):3d}] ADD  {feat:40s} composite={current_score:+.1f} "
                  f"(Δ={delta:+.1f}) ATS7+: {r.get('ats_7', 0):.1%}")
        else:
            stall_count += 1
            print(f"  [   ] SKIP {feat:40s} composite={new_score:.1f} (Δ={delta:+.1f})")
    
    print(f"\n  Forward selection complete: {len(selected)} features")
    print(f"  Final composite ATS: {current_score:.1f}")
    
    return {"selected": selected, "log": selection_log}


# ═══════════════════════════════════════════════════════════
# STEP 4: Backward Pruning
# ═══════════════════════════════════════════════════════════

def step4_backward_pruning(X, y, spreads, selected_features):
    """Try removing each feature. Drop if ATS doesn't decrease."""
    print(f"\n{'='*70}")
    print(f"  STEP 4: Backward Pruning")
    print(f"{'='*70}")
    
    current = list(selected_features)
    
    # Get baseline score
    base_r = ats_eval(X[current].values, y, spreads, n_folds=10)
    base_score = base_r["composite_ats"]
    print(f"  Starting: {len(current)} features, composite: {base_score:.1f}")
    
    removed = []
    
    # Try removing each non-anchor feature
    for feat in list(current):
        if feat == "market_spread":
            continue  # never remove anchor
        
        test_features = [f for f in current if f != feat]
        r = ats_eval(X[test_features].values, y, spreads, n_folds=10)
        new_score = r["composite_ats"]
        delta = new_score - base_score
        
        if delta >= -0.1:  # removing doesn't hurt (or even helps)
            current.remove(feat)
            base_score = new_score
            removed.append(feat)
            marker = "IMPROVES" if delta > 0 else "neutral"
            print(f"  REMOVE {feat:40s} Δ={delta:+.1f} ({marker}) → {len(current)} features")
        else:
            print(f"  KEEP   {feat:40s} Δ={delta:+.1f}")
    
    print(f"\n  Pruning complete: {len(current)} features (removed {len(removed)})")
    print(f"  Final composite: {base_score:.1f}")
    
    if removed:
        print(f"\n  Removed features:")
        for f in removed:
            print(f"    - {f}")
    
    return {"final_features": current, "removed": removed}


# ═══════════════════════════════════════════════════════════
# STEP 5: Final Validation
# ═══════════════════════════════════════════════════════════

def step5_final_validation(X, y, spreads, final_features, df):
    """Full 30-fold walk-forward validation on final feature set."""
    print(f"\n{'='*70}")
    print(f"  STEP 5: Final Validation ({len(final_features)} features, 30 folds)")
    print(f"{'='*70}")
    
    X_final = X[final_features].values
    
    # CatBoost d=4
    print(f"\n  >>> CatBoost (depth=4, 650 iters)")
    r_cat = ats_eval(X_final, y, spreads, n_folds=30)
    print(f"  MAE: {r_cat['mae']}")
    for t in [0, 2, 4, 7, 10]:
        acc = r_cat.get(f"ats_{t}", 0)
        roi = r_cat.get(f"roi_{t}", -100)
        n = r_cat.get(f"n_{t}", 0)
        print(f"  ATS {t}+: {acc:.1%} ({n} picks) → ROI {roi:+.1f}%")
    print(f"  Composite: {r_cat['composite_ats']:.1f}")
    
    # Compare with all 102 features
    print(f"\n  >>> Comparison: All {X.shape[1]} features")
    r_all = ats_eval(X.values, y, spreads, n_folds=30)
    print(f"  MAE: {r_all['mae']}")
    for t in [0, 2, 4, 7, 10]:
        acc = r_all.get(f"ats_{t}", 0)
        roi = r_all.get(f"roi_{t}", -100)
        n = r_all.get(f"n_{t}", 0)
        print(f"  ATS {t}+: {acc:.1%} ({n} picks) → ROI {roi:+.1f}%")
    print(f"  Composite: {r_all['composite_ats']:.1f}")
    
    # Summary
    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")
    print(f"  All features:      {X.shape[1]:3d} feats → composite {r_all['composite_ats']:.1f}")
    print(f"  Optimized features: {len(final_features):3d} feats → composite {r_cat['composite_ats']:.1f}")
    delta = r_cat["composite_ats"] - r_all["composite_ats"]
    print(f"  Improvement: {delta:+.1f}")
    
    print(f"\n  Final feature set ({len(final_features)}):")
    for i, f in enumerate(final_features):
        print(f"    {i+1:3d}. {f}")
    
    return {"optimized": r_cat, "all_features": r_all, "final_features": final_features}


# ═══════════════════════════════════════════════════════════
# SAVE / LOAD STATE
# ═══════════════════════════════════════════════════════════

def save_state(state):
    with open(RESULTS_FILE, "w") as f:
        # Filter out non-serializable numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            return obj
        
        json.dump(state, f, indent=2, default=convert)
    print(f"  Saved state to {RESULTS_FILE}")


def load_state():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1", action="store_true", help="Correlation audit only")
    parser.add_argument("--step2", action="store_true", help="Solo ATS lift only")
    parser.add_argument("--step3", action="store_true", help="Forward selection")
    parser.add_argument("--step4", action="store_true", help="Backward pruning")
    parser.add_argument("--final", action="store_true", help="Final validation")
    args = parser.parse_args()
    
    run_all = not (args.step1 or args.step2 or args.step3 or args.step4 or args.final)
    
    # Load data
    df = load_training_data()
    X, feature_names = build_features(df)
    y = df["target_margin"].values
    spreads = df["market_spread_home"].values
    
    # Sort by date for walk-forward
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    spreads = spreads[sort_idx]
    df = df.iloc[sort_idx].reset_index(drop=True)
    
    state = load_state()
    
    # STEP 1: Correlation Audit
    if run_all or args.step1:
        t0 = time.time()
        corr_results = step1_correlation_audit(X, feature_names)
        state["step1"] = {
            "pairs": [(a, b, float(r)) for a, b, r in corr_results["pairs"]],
            "groups": corr_results["groups"],
            "singletons": corr_results["singletons"],
            "elapsed": round(time.time() - t0, 1)
        }
        save_state(state)
    
    # STEP 2: Solo ATS Lift
    if run_all or args.step2:
        t0 = time.time()
        solo_results = step2_solo_ats_lift(X, y, spreads, feature_names)
        state["step2"] = {
            "baseline": solo_results["baseline"],
            "solo_results": solo_results["solo_results"],
            "elapsed": round(time.time() - t0, 1)
        }
        save_state(state)
    
    # STEP 3: Forward Selection
    if run_all or args.step3:
        if "step2" not in state:
            print("ERROR: Need step2 results first. Run with --step2")
            exit(1)
        if "step1" not in state:
            print("ERROR: Need step1 results first. Run with --step1")
            exit(1)
        
        t0 = time.time()
        solo = state["step2"]["solo_results"]
        groups = state["step1"]["groups"]
        
        fwd_results = step3_forward_selection(X, y, spreads, feature_names, solo, groups)
        state["step3"] = {
            "selected": fwd_results["selected"],
            "log": fwd_results["log"],
            "elapsed": round(time.time() - t0, 1)
        }
        save_state(state)
    
    # STEP 4: Backward Pruning
    if run_all or args.step4:
        if "step3" not in state:
            print("ERROR: Need step3 results first. Run with --step3")
            exit(1)
        
        t0 = time.time()
        selected = state["step3"]["selected"]
        prune_results = step4_backward_pruning(X, y, spreads, selected)
        state["step4"] = {
            "final_features": prune_results["final_features"],
            "removed": prune_results["removed"],
            "elapsed": round(time.time() - t0, 1)
        }
        save_state(state)
    
    # STEP 5: Final Validation
    if run_all or args.final:
        if "step4" not in state:
            if "step3" in state:
                final_features = state["step3"]["selected"]
            else:
                print("ERROR: Need step3 or step4 results first")
                exit(1)
        else:
            final_features = state["step4"]["final_features"]
        
        t0 = time.time()
        val_results = step5_final_validation(X, y, spreads, final_features, df)
        state["step5"] = {
            "optimized": val_results["optimized"],
            "all_features": val_results["all_features"],
            "final_features": val_results["final_features"],
            "elapsed": round(time.time() - t0, 1)
        }
        save_state(state)
    
    # Total time
    total = sum(state.get(f"step{i}", {}).get("elapsed", 0) for i in range(1, 6))
    print(f"\n  Total elapsed: {total/60:.1f} minutes")
