#!/usr/bin/env python3
"""
nba_v27_eliminate.py — Comprehensive feature elimination for NBA v27

5 Learners: CatBoost, LightGBM, Lasso, Ridge, ElasticNet
3 Targets:  ML (margin), ATS (margin + spread), O/U (total points)

Pipeline:
  Phase 1: Correlation audit → group features with |r| > 0.7
  Phase 2: Multi-learner drop-one → score each feature across all 5 learners × 3 targets
  Phase 3: Consensus filter → keep features that help majority of learner×target combos
  Phase 4: ATS forward selection → from survivors, build optimal ATS set
  Phase 5: Final validation → 30-fold walk-forward on all 3 targets

Usage:
  python3 nba_v27_eliminate.py                     # Full pipeline
  python3 nba_v27_eliminate.py --phase1            # Correlation only
  python3 nba_v27_eliminate.py --phase2            # Multi-learner drop-one
  python3 nba_v27_eliminate.py --phase3            # Consensus filter
  python3 nba_v27_eliminate.py --phase4            # ATS forward selection
  python3 nba_v27_eliminate.py --phase5            # Final validation
  python3 nba_v27_eliminate.py --analyze           # Re-read saved results
"""

import pandas as pd
import numpy as np
import json, os, copy, time, sys, warnings, argparse
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed — pip install lightgbm")

from nba_build_features_v27 import load_training_data, build_features

RESULTS_FILE = "nba_v27_elimination_results.json"


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def save_state(state):
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, set): return list(obj)
        return obj
    with open(RESULTS_FILE, "w") as f:
        json.dump(state, f, indent=2, default=convert)


def load_state():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def build_learners():
    """Build all 5 learners."""
    learners = {
        "CatBoost": CatBoostRegressor(
            depth=4, iterations=500, learning_rate=0.05,
            l2_leaf_reg=3, random_seed=42, verbose=0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    }
    if HAS_LGBM:
        learners["LightGBM"] = LGBMRegressor(
            max_depth=4, n_estimators=500, learning_rate=0.05,
            reg_lambda=3, random_state=42, verbose=-1)
    return learners


def prepare_targets(df):
    """Build all 3 target vectors."""
    margin = (df["actual_home_score"] - df["actual_away_score"]).values
    spread = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values
    total = (df["actual_home_score"] + df["actual_away_score"]).values

    return {
        "ML": margin,                    # predict home margin
        "ATS": margin + spread,           # predict ATS margin (positive = home covers)
        "OU": total,                      # predict total points
    }


def cv_score(learner, X, y, n_splits=10, name="", scale=True):
    """Walk-forward CV returning MAE + ATS accuracy (if spread provided)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if scale and name in ("Lasso", "Ridge", "ElasticNet"):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)

        mdl = copy.deepcopy(learner)
        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_val)
        maes.append(np.mean(np.abs(pred - y_val)))

    return round(np.mean(maes), 5)


def ats_eval_detailed(X, y_margin, spreads, n_folds=10, model_type="catboost"):
    """Walk-forward ATS evaluation at multiple thresholds."""
    n = len(X)
    fold_size = n // (n_folds + 2)
    min_train = fold_size * 3

    all_preds, all_actual, all_spreads = [], [], []

    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if te > n: break

        sc = StandardScaler()
        X_tr = sc.fit_transform(X[:ts])
        X_te = sc.transform(X[ts:te])

        if model_type == "catboost":
            mdl = CatBoostRegressor(depth=4, iterations=500, learning_rate=0.05,
                                     l2_leaf_reg=3, random_seed=42, verbose=0)
        else:
            mdl = Lasso(alpha=0.1, max_iter=5000)

        mdl.fit(X_tr, y_margin[:ts])
        preds = mdl.predict(X_te)

        all_preds.extend(preds)
        all_actual.extend(y_margin[ts:te])
        all_spreads.extend(spreads[ts:te])

    preds = np.array(all_preds)
    actual = np.array(all_actual)
    spr = np.array(all_spreads)

    ats_edge = preds - (-spr)
    ats_margin = actual + spr
    not_push = ats_margin != 0
    ats_correct = np.sign(ats_edge) == np.sign(ats_margin)

    mae = round(np.mean(np.abs(preds - actual)), 3)
    results = {"mae": mae}

    for t in [0, 2, 4, 7, 10]:
        mask = (np.abs(ats_edge) >= t) & not_push
        n_picks = int(mask.sum())
        if n_picks >= 20:
            acc = round(float(ats_correct[mask].mean()), 4)
            roi = round((acc * 1.909 - 1) * 100, 1)
        else:
            acc, roi = 0.0, -100.0
        results[f"ats_{t}"] = acc
        results[f"roi_{t}"] = roi
        results[f"n_{t}"] = n_picks

    # Composite: weighted ROI at 4+/7+/10u
    composite = (results["roi_4"] * 0.3 + results["roi_7"] * 0.4 + results["roi_10"] * 0.3)
    results["composite"] = round(composite, 2)

    return results


# ═══════════════════════════════════════════════════════════
# PHASE 1: Correlation Audit
# ═══════════════════════════════════════════════════════════

def phase1_correlation(X, feature_names, threshold=0.7):
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Correlation Audit (|r| > {threshold})")
    print(f"{'='*70}")

    corr = X.corr().abs()
    pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            r = corr.iloc[i, j]
            if r > threshold:
                pairs.append((feature_names[i], feature_names[j], round(float(r), 3)))
    pairs.sort(key=lambda x: -x[2])

    print(f"\n  {len(pairs)} highly correlated pairs:")
    for f1, f2, r in pairs[:25]:
        print(f"    {f1:35s} ↔ {f2:35s}  r={r:.3f}")
    if len(pairs) > 25:
        print(f"    ... and {len(pairs)-25} more")

    # Union-find grouping
    parent = {f: f for f in feature_names}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for f1, f2, _ in pairs:
        union(f1, f2)

    groups = {}
    for f in feature_names:
        root = find(f)
        groups.setdefault(root, []).append(f)
    multi = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"\n  {len(multi)} correlated groups:")
    group_list = []
    for i, (_, members) in enumerate(sorted(multi.items(), key=lambda x: -len(x[1]))):
        print(f"  Group {i+1}: {members}")
        group_list.append(members)

    singletons = [f for f in feature_names if len(groups[find(f)]) == 1]
    print(f"  Singletons: {len(singletons)}")

    return {"pairs": pairs, "groups": group_list, "singletons": singletons}


# ═══════════════════════════════════════════════════════════
# PHASE 2: Multi-Learner Drop-One (3 targets × 5 learners)
# ═══════════════════════════════════════════════════════════

def phase2_drop_one(X, targets, feature_names, n_folds=10):
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Multi-Learner Drop-One ({len(feature_names)} features)")
    print(f"  Learners × Targets = {5 if HAS_LGBM else 4} × {len(targets)} = "
          f"{(5 if HAS_LGBM else 4) * len(targets)} evaluations per feature")
    print(f"{'='*70}")

    learners = build_learners()
    X_arr = X.values

    # Baselines
    print("\n  Computing baselines (all features)...")
    baselines = {}
    for tname, y in targets.items():
        baselines[tname] = {}
        for lname, learner in learners.items():
            t0 = time.time()
            mae = cv_score(copy.deepcopy(learner), X_arr, y, n_folds, lname)
            dt = time.time() - t0
            baselines[tname][lname] = mae
            print(f"    {tname:4s} × {lname:12s}: MAE={mae:.4f} ({dt:.1f}s)")

    # Drop each feature
    print(f"\n  Testing {len(feature_names)} features...")
    results = {}
    total = len(feature_names)
    t_start = time.time()

    for i, feat in enumerate(feature_names):
        feat_idx = feature_names.index(feat)
        X_drop = np.delete(X_arr, feat_idx, axis=1)

        scores = {}
        for tname, y in targets.items():
            scores[tname] = {}
            for lname, learner in learners.items():
                mae = cv_score(copy.deepcopy(learner), X_drop, y, n_folds, lname)
                delta = mae - baselines[tname][lname]
                scores[tname][lname] = round(delta, 5)

        # Aggregate: how many learner×target combos does removing this feature HURT?
        all_deltas = []
        for tname in targets:
            for lname in learners:
                all_deltas.append(scores[tname][lname])

        n_combos = len(all_deltas)
        n_hurts = sum(1 for d in all_deltas if d > 0.001)   # removing hurts = feature is useful
        n_helps = sum(1 for d in all_deltas if d < -0.001)   # removing helps = feature is harmful
        avg_delta = round(np.mean(all_deltas), 5)

        # Per-target averages
        target_avgs = {}
        for tname in targets:
            target_avgs[tname] = round(np.mean([scores[tname][l] for l in learners]), 5)

        results[feat] = {
            "scores": scores,
            "avg_delta": avg_delta,
            "target_avgs": target_avgs,
            "n_hurts": n_hurts,
            "n_helps": n_helps,
            "n_combos": n_combos,
            "pct_hurts": round(n_hurts / n_combos, 2),
        }

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 1
        remaining = (total - i - 1) / rate / 60

        if (i + 1) % 5 == 0 or i < 3:
            tag = "✅" if avg_delta > 0.002 else ("❌" if avg_delta < -0.002 else "⚪")
            ml_avg = target_avgs.get("ML", 0)
            ats_avg = target_avgs.get("ATS", 0)
            ou_avg = target_avgs.get("OU", 0)
            print(f"    [{i+1:3d}/{total}] {tag} {feat:35s} "
                  f"ML:{ml_avg:+.4f} ATS:{ats_avg:+.4f} OU:{ou_avg:+.4f} "
                  f"hurts:{n_hurts}/{n_combos} (~{remaining:.0f}m left)")

    return baselines, results


# ═══════════════════════════════════════════════════════════
# PHASE 3: Consensus Filter
# ═══════════════════════════════════════════════════════════

def phase3_consensus(results, feature_names):
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Consensus Filter")
    print(f"{'='*70}")

    # Categorize
    strong_keep = []    # hurts majority to remove
    safe_keep = []      # neutral or mixed
    drop = []           # helps majority to remove
    conflicts = []      # helps some targets, hurts others significantly

    for feat, r in results.items():
        pct = r["pct_hurts"]
        avg = r["avg_delta"]
        ta = r["target_avgs"]

        # Check for cross-target conflicts
        target_signs = [1 if v > 0.001 else (-1 if v < -0.001 else 0) for v in ta.values()]
        has_conflict = (max(target_signs) > 0 and min(target_signs) < 0)

        if pct >= 0.60 and avg > 0:
            strong_keep.append((feat, r))
        elif avg < -0.002 and pct <= 0.30:
            drop.append((feat, r))
        elif has_conflict:
            conflicts.append((feat, r))
        else:
            safe_keep.append((feat, r))

    strong_keep.sort(key=lambda x: -x[1]["avg_delta"])
    drop.sort(key=lambda x: x[1]["avg_delta"])

    print(f"\n  STRONG KEEP ({len(strong_keep)} — removing hurts 60%+ of learner×target combos):")
    for f, r in strong_keep[:30]:
        ta = r["target_avgs"]
        print(f"    {f:40s} avg={r['avg_delta']:+.5f}  ML:{ta['ML']:+.4f} ATS:{ta['ATS']:+.4f} OU:{ta['OU']:+.4f}  hurts:{r['n_hurts']}/{r['n_combos']}")

    print(f"\n  DROP ({len(drop)} — removing helps 70%+ of combos):")
    for f, r in drop:
        ta = r["target_avgs"]
        print(f"    {f:40s} avg={r['avg_delta']:+.5f}  ML:{ta['ML']:+.4f} ATS:{ta['ATS']:+.4f} OU:{ta['OU']:+.4f}")

    print(f"\n  CONFLICTS ({len(conflicts)} — helps some targets, hurts others):")
    for f, r in conflicts[:15]:
        ta = r["target_avgs"]
        print(f"    {f:40s} avg={r['avg_delta']:+.5f}  ML:{ta['ML']:+.4f} ATS:{ta['ATS']:+.4f} OU:{ta['OU']:+.4f}")

    print(f"\n  SAFE/MARGINAL ({len(safe_keep)})")

    # Survivors = strong_keep + safe_keep + conflicts (keep conflicts for now, let ATS selection decide)
    survivors = [f for f, _ in strong_keep] + [f for f, _ in safe_keep] + [f for f, _ in conflicts]
    dropped = [f for f, _ in drop]

    print(f"\n  Summary: {len(strong_keep)} strong + {len(safe_keep)} safe + {len(conflicts)} conflicts = {len(survivors)} survivors")
    print(f"  Dropped: {len(dropped)}")

    return {"survivors": survivors, "dropped": dropped,
            "strong_keep": [f for f, _ in strong_keep],
            "conflicts": [f for f, _ in conflicts]}


# ═══════════════════════════════════════════════════════════
# PHASE 4: ATS Forward Selection (from survivors)
# ═══════════════════════════════════════════════════════════

def phase4_ats_forward(X, y_margin, spreads, survivors, corr_groups):
    print(f"\n{'='*70}")
    print(f"  PHASE 4: ATS Forward Selection ({len(survivors)} survivors)")
    print(f"{'='*70}")

    # Build group membership for correlation dedup
    group_map = {}
    for group in corr_groups:
        for f in group:
            group_map[f] = set(group)

    # Start with market_spread
    selected = ["market_spread"]
    if "market_spread" not in survivors:
        survivors = ["market_spread"] + survivors

    # Get baseline
    X_sel = X[selected].values
    base = ats_eval_detailed(X_sel, y_margin, spreads, n_folds=10)
    current_composite = base["composite"]
    print(f"  Baseline (spread only): composite={current_composite:.1f}")

    # Rank survivors by phase 2 avg_delta (best features first)
    # We'll try them in this order
    candidates = [f for f in survivors if f != "market_spread" and f in X.columns]

    log = [{"step": 0, "feature": "market_spread", "composite": current_composite}]
    stall = 0

    for feat in candidates:
        if stall >= 12:
            print(f"\n  12 consecutive non-improvements, stopping")
            break

        # Skip if correlated partner already selected
        if feat in group_map:
            already = group_map[feat] & set(selected)
            if already:
                continue

        test = selected + [feat]
        X_test = X[test].values
        r = ats_eval_detailed(X_test, y_margin, spreads, n_folds=10)
        delta = r["composite"] - current_composite

        if delta > 0.2:  # meaningful improvement
            selected.append(feat)
            current_composite = r["composite"]
            stall = 0
            log.append({"step": len(selected), "feature": feat,
                        "composite": round(current_composite, 2), "delta": round(delta, 2),
                        "ats_4": r.get("ats_4", 0), "ats_7": r.get("ats_7", 0),
                        "ats_10": r.get("ats_10", 0)})
            print(f"  [{len(selected):3d}] ADD  {feat:40s} comp={current_composite:.1f} "
                  f"Δ={delta:+.1f}  ATS7={r.get('ats_7',0):.1%}")
        else:
            stall += 1
            if stall <= 3 or stall % 5 == 0:
                print(f"  [   ] skip {feat:40s} Δ={delta:+.1f}")

    print(f"\n  Forward selection: {len(selected)} features, composite={current_composite:.1f}")

    # Backward prune
    print(f"\n  Backward pruning...")
    removed = []
    for feat in list(selected):
        if feat == "market_spread": continue
        test = [f for f in selected if f != feat]
        r = ats_eval_detailed(X[test].values, y_margin, spreads, n_folds=10)
        delta = r["composite"] - current_composite
        if delta >= -0.2:  # removing doesn't hurt
            selected.remove(feat)
            current_composite = r["composite"]
            removed.append(feat)
            tag = "HELPS" if delta > 0 else "neutral"
            print(f"  PRUNE {feat:40s} Δ={delta:+.1f} ({tag}) → {len(selected)}f")
        else:
            print(f"  KEEP  {feat:40s} Δ={delta:+.1f}")

    print(f"\n  After pruning: {len(selected)} features (removed {len(removed)})")
    return {"selected": selected, "log": log, "pruned": removed}


# ═══════════════════════════════════════════════════════════
# PHASE 5: Final Validation (30-fold, all 3 targets)
# ═══════════════════════════════════════════════════════════

def phase5_final(X, targets, spreads, final_features, all_features):
    print(f"\n{'='*70}")
    print(f"  PHASE 5: Final Validation")
    print(f"{'='*70}")

    configs = [
        ("Optimized", final_features),
        ("All features", all_features),
    ]

    for label, feats in configs:
        print(f"\n  >>> {label} ({len(feats)} features)")
        X_sub = X[[f for f in feats if f in X.columns]].values

        # ATS detailed (CatBoost)
        r = ats_eval_detailed(X_sub, targets["ML"], spreads, n_folds=20, model_type="catboost")
        print(f"  MAE: {r['mae']}")
        for t in [0, 2, 4, 7, 10]:
            print(f"  ATS {t}+: {r.get(f'ats_{t}',0):.1%} ({r.get(f'n_{t}',0)} picks) → ROI {r.get(f'roi_{t}',-100):+.1f}%")
        print(f"  Composite: {r['composite']:.1f}")

        # Per-target MAE (all learners)
        learners = build_learners()
        for tname, y in targets.items():
            maes = {}
            for lname, learner in learners.items():
                mae = cv_score(copy.deepcopy(learner), X_sub, y, 10, lname)
                maes[lname] = mae
            avg = np.mean(list(maes.values()))
            parts = " ".join(f"{l}:{m:.3f}" for l, m in maes.items())
            print(f"  {tname:4s} MAE: avg={avg:.3f}  [{parts}]")

    print(f"\n  FINAL FEATURE SET ({len(final_features)}):")
    for i, f in enumerate(final_features):
        print(f"    {i+1:3d}. {f}")

    return final_features


# ═══════════════════════════════════════════════════════════
# ANALYZE SAVED RESULTS
# ═══════════════════════════════════════════════════════════

def analyze():
    state = load_state()
    if not state:
        print("No results file found. Run the pipeline first.")
        return

    if "phase2" in state:
        p2 = state["phase2"]
        results = p2["results"]
        print(f"\n{'='*70}")
        print(f"  Phase 2 Results: {len(results)} features evaluated")
        print(f"{'='*70}")

        sorted_feats = sorted(results.items(), key=lambda x: -x[1]["avg_delta"])
        print(f"\n  Top 20 (most important — removing hurts):")
        print(f"  {'Rank':>4s} {'Feature':40s} {'Avg Δ':>8s} {'ML':>8s} {'ATS':>8s} {'OU':>8s} {'Hurts':>6s}")
        for i, (f, r) in enumerate(sorted_feats[:20]):
            ta = r["target_avgs"]
            print(f"  {i+1:4d} {f:40s} {r['avg_delta']:+.5f} "
                  f"{ta['ML']:+.5f} {ta['ATS']:+.5f} {ta['OU']:+.5f} "
                  f"{r['n_hurts']}/{r['n_combos']}")

        print(f"\n  Bottom 10 (least important — removing helps):")
        for i, (f, r) in enumerate(sorted_feats[-10:]):
            ta = r["target_avgs"]
            print(f"  {len(sorted_feats)-9+i:4d} {f:40s} {r['avg_delta']:+.5f} "
                  f"{ta['ML']:+.5f} {ta['ATS']:+.5f} {ta['OU']:+.5f}")

    if "phase4" in state:
        p4 = state["phase4"]
        print(f"\n  ATS-optimized set: {len(p4['selected'])} features")
        for i, f in enumerate(p4["selected"]):
            print(f"    {i+1:3d}. {f}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--phase3", action="store_true")
    parser.add_argument("--phase4", action="store_true")
    parser.add_argument("--phase5", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()

    if args.analyze:
        analyze()
        sys.exit(0)

    run_all = not (args.phase1 or args.phase2 or args.phase3 or args.phase4 or args.phase5)

    # Load
    print("="*70)
    print("  NBA v27 COMPREHENSIVE ELIMINATION")
    print(f"  5 Learners × 3 Targets × {args.folds}-fold CV")
    print("="*70)

    df = load_training_data()
    X, feature_names = build_features(df)
    targets = prepare_targets(df)
    spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values

    # Sort by date
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    for tname in targets:
        targets[tname] = targets[tname][sort_idx]
    spreads = spreads[sort_idx]
    df = df.iloc[sort_idx].reset_index(drop=True)

    state = load_state()

    # ── PHASE 1 ──
    if run_all or args.phase1:
        t0 = time.time()
        p1 = phase1_correlation(X, feature_names)
        state["phase1"] = {
            "pairs": p1["pairs"], "groups": p1["groups"],
            "singletons": p1["singletons"],
            "elapsed": round(time.time()-t0, 1)
        }
        save_state(state)

    # ── PHASE 2 ──
    if run_all or args.phase2:
        t0 = time.time()
        baselines, results = phase2_drop_one(X, targets, feature_names, n_folds=args.folds)
        state["phase2"] = {
            "baselines": {t: {l: float(v) for l, v in lm.items()} for t, lm in baselines.items()},
            "results": results,
            "elapsed": round(time.time()-t0, 1)
        }
        save_state(state)
        print(f"\n  Phase 2 complete: {state['phase2']['elapsed']/60:.1f} min")

    # ── PHASE 3 ──
    if run_all or args.phase3:
        if "phase2" not in state:
            print("ERROR: Run --phase2 first"); sys.exit(1)
        t0 = time.time()
        p3 = phase3_consensus(state["phase2"]["results"], feature_names)
        state["phase3"] = {
            "survivors": p3["survivors"], "dropped": p3["dropped"],
            "strong_keep": p3["strong_keep"], "conflicts": p3["conflicts"],
            "elapsed": round(time.time()-t0, 1)
        }
        save_state(state)

    # ── PHASE 4 ──
    if run_all or args.phase4:
        if "phase3" not in state:
            print("ERROR: Run --phase3 first"); sys.exit(1)
        if "phase1" not in state:
            print("ERROR: Run --phase1 first"); sys.exit(1)
        t0 = time.time()

        survivors = state["phase3"]["survivors"]
        # Sort survivors by phase2 avg_delta (best first)
        p2_results = state["phase2"]["results"]
        survivors.sort(key=lambda f: -p2_results.get(f, {}).get("avg_delta", 0))

        corr_groups = state["phase1"]["groups"]
        p4 = phase4_ats_forward(X, targets["ML"], spreads, survivors, corr_groups)
        state["phase4"] = {
            "selected": p4["selected"], "log": p4["log"],
            "pruned": p4["pruned"],
            "elapsed": round(time.time()-t0, 1)
        }
        save_state(state)

    # ── PHASE 5 ──
    if run_all or args.phase5:
        if "phase4" not in state:
            print("ERROR: Run --phase4 first"); sys.exit(1)
        t0 = time.time()
        final = state["phase4"]["selected"]
        phase5_final(X, targets, spreads, final, feature_names)
        state["phase5"] = {"elapsed": round(time.time()-t0, 1), "final_features": final}
        save_state(state)

    # Summary
    total_time = sum(state.get(f"phase{i}", {}).get("elapsed", 0) for i in range(1, 6))
    print(f"\n  Total time: {total_time/60:.1f} minutes")
