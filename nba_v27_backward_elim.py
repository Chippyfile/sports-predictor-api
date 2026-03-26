#!/usr/bin/env python3
"""
nba_v27_backward_elim.py — Backward elimination from Phase 3 survivors.

Starts with ALL 91 survivors (preserving synergies), then removes features
one-at-a-time if removal improves or doesn't hurt ATS composite.

Uses multi-learner consensus (CatBoost + GBM + Lasso + Ridge) to avoid
architecture bias.

Run from ~/Desktop/sports-predictor-api/:
    python3 nba_v27_backward_elim.py
"""
import sys, os, json, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("WARNING: CatBoost not installed — using Lasso+Ridge only")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_GBM = True
except ImportError:
    HAS_GBM = False


# ═══════════════════════════════════════════════════════════
# LOAD STATE + DATA
# ═══════════════════════════════════════════════════════════

STATE_FILE = "nba_v27_elimination_results.json"
if not os.path.exists(STATE_FILE):
    print(f"ERROR: {STATE_FILE} not found.")
    sys.exit(1)

with open(STATE_FILE) as f:
    state = json.load(f)

if "phase3" not in state:
    print("ERROR: Phase 3 not complete. Run nba_v27_eliminate.py --phase3 first.")
    sys.exit(1)

survivors = state["phase3"]["survivors"]
strong_keep = state["phase3"].get("strong_keep", [])
dropped_p3 = state["phase3"].get("dropped", [])
p2_results = state.get("phase2", {}).get("results", {})

print("=" * 70)
print("  NBA v27 BACKWARD ELIMINATION — Multi-Learner Consensus")
print("=" * 70)
print(f"\n  Phase 3 survivors: {len(survivors)}")
print(f"  Strong keep: {len(strong_keep)}")
print(f"  Already dropped in Phase 3: {len(dropped_p3)}")

# Load training data
print("\nLoading training data...")
from nba_v27_eliminate import load_training_data, build_features, prepare_targets

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

y_margin = targets["ML"]
has_spread = np.abs(spreads) > 0.1
n = len(X)

print(f"  {n} games, {len(feature_names)} features, {has_spread.sum()} with spreads")

# Filter survivors to those that exist in X
survivors = [f for f in survivors if f in X.columns]
# Add newly computed features that weren't in original Phase 3 survivors
for _pf in ["h2h_avg_margin", "conference_game", "is_revenge_home"]:
    if _pf in X.columns and _pf not in survivors:
        survivors.append(_pf)
        print(f"  Added new feature to survivors: {_pf}")
print(f"  Available survivors: {len(survivors)}")


# ═══════════════════════════════════════════════════════════
# MULTI-LEARNER ATS EVALUATION
# ═══════════════════════════════════════════════════════════

def _score_ats(preds, actual, spr):
    """Compute ATS metrics."""
    ats_edge = preds - (-spr)
    ats_margin = actual + spr
    not_push = ats_margin != 0
    ats_correct = np.sign(ats_edge) == np.sign(ats_margin)

    mae = round(float(np.mean(np.abs(preds - actual))), 3)
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

    composite = (results["roi_4"] * 0.3 + results["roi_7"] * 0.4 + results["roi_10"] * 0.3)
    results["composite"] = round(composite, 2)
    return results


def _wf_predict(X_sub, y, n_folds, model_builder):
    """Walk-forward predictions for a single model type."""
    n = len(X_sub)
    fold_size = n // (n_folds + 2)
    min_train = fold_size * 3

    all_preds, all_idx = [], []
    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n or te <= ts:
            break

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_sub[:ts])
        X_te = sc.transform(X_sub[ts:te])

        mdl = model_builder(X_tr.shape[1])
        mdl.fit(X_tr, y[:ts])
        preds = mdl.predict(X_te)
        all_preds.extend(preds)
        all_idx.extend(range(ts, te))

    return np.array(all_preds), np.array(all_idx)


def ats_eval(X_sub, y, spr, n_folds=15):
    """Multi-learner consensus ATS evaluation."""
    if isinstance(X_sub, pd.DataFrame):
        X_sub = X_sub.values
    if X_sub.ndim == 1:
        X_sub = X_sub.reshape(-1, 1)

    n_f = X_sub.shape[1]

    def make_lasso(nf): return Lasso(alpha=0.1, max_iter=5000)
    def make_ridge(nf): return Ridge(alpha=1.0)
    def make_cat(nf): return CatBoostRegressor(depth=min(4, nf), iterations=500,
        learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0)
    def make_gbm(nf): return GradientBoostingRegressor(n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42)

    learners = {"Lasso": make_lasso, "Ridge": make_ridge}
    if HAS_CAT and n_f >= 2:
        learners["CatBoost"] = make_cat
    if HAS_GBM and n_f >= 2:
        learners["GBM"] = make_gbm

    # Run walk-forward for each learner
    all_learner_preds = {}
    common_idx = None
    for name, builder in learners.items():
        preds, idx = _wf_predict(X_sub, y, n_folds, builder)
        all_learner_preds[name] = (preds, idx)
        if common_idx is None:
            common_idx = set(idx.tolist())
        else:
            common_idx = common_idx & set(idx.tolist())

    if not common_idx:
        return {"composite": -100, "mae": 99, "ats_0": 0.5}

    common_idx = sorted(common_idx)
    idx_map = {v: i for i, v in enumerate(common_idx)}

    ensemble_preds = np.zeros(len(common_idx))
    n_learners = 0
    for name, (preds, idx) in all_learner_preds.items():
        learner_aligned = np.zeros(len(common_idx))
        for p, i in zip(preds, idx):
            if i in idx_map:
                learner_aligned[idx_map[i]] = p
        ensemble_preds += learner_aligned
        n_learners += 1
    ensemble_preds /= n_learners

    actual_common = y[common_idx]
    spr_common = spr[common_idx]

    return _score_ats(ensemble_preds, actual_common, spr_common)


# ═══════════════════════════════════════════════════════════
# BASELINE: All survivors
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  BASELINE: All {len(survivors)} survivors")
print(f"{'='*70}")

t0 = time.time()
X_surv = X[survivors].values
baseline = ats_eval(X_surv, y_margin, spreads)

print(f"\n  MAE:       {baseline['mae']:.3f}")
print(f"  ATS:       {baseline['ats_0']:.1%}")
print(f"  ATS 4+:    {baseline['ats_4']:.1%} ({baseline['n_4']} picks, ROI {baseline['roi_4']:+.1f}%)")
print(f"  ATS 7+:    {baseline['ats_7']:.1%} ({baseline['n_7']} picks, ROI {baseline['roi_7']:+.1f}%)")
print(f"  ATS 10+:   {baseline['ats_10']:.1%} ({baseline['n_10']} picks, ROI {baseline['roi_10']:+.1f}%)")
print(f"  Composite: {baseline['composite']:.1f}")
print(f"  Time: {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# ROUND 1: Score every feature's removal impact
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  ROUND 1: Score all {len(survivors)} features for removal")
print(f"{'='*70}")

current_set = list(survivors)
current_composite = baseline["composite"]

removal_scores = []
t0 = time.time()

for i, feat in enumerate(current_set):
    test = [f for f in current_set if f != feat]
    r = ats_eval(X[test].values, y_margin, spreads)
    delta = r["composite"] - current_composite
    removal_scores.append({
        "feature": feat,
        "delta": delta,
        "composite_without": r["composite"],
        "ats_7_without": r.get("ats_7", 0),
        "mae_without": r["mae"],
        "is_strong": feat in strong_keep,
    })

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(current_set) - i - 1)
        print(f"  [{i+1}/{len(current_set)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

# Sort: most beneficial to remove first (highest positive delta)
removal_scores.sort(key=lambda x: -x["delta"])

print(f"\n  {'Rank':>4} {'Feature':<40s} {'ΔComp':>8s} {'ΔMAE':>8s} {'Strong':>6s} {'Action'}")
print("  " + "-" * 90)

for i, r in enumerate(removal_scores):
    delta = r["delta"]
    dmae = r["mae_without"] - baseline["mae"]
    is_strong = "✓" if r["is_strong"] else ""
    if delta > 0.5:
        action = "REMOVE ✗"
    elif delta > 0.0:
        action = "maybe rm"
    elif delta > -0.5:
        action = "neutral"
    elif delta > -2.0:
        action = "useful"
    else:
        action = "CRITICAL ✓"
    print(f"  {i+1:4d} {r['feature']:<40s} {delta:>+7.2f} {dmae:>+7.3f}  {is_strong:>5s}  {action}")


# ═══════════════════════════════════════════════════════════
# ITERATIVE BACKWARD ELIMINATION
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  ITERATIVE BACKWARD ELIMINATION")
print(f"  Remove features where removal improves or barely hurts composite")
print(f"{'='*70}")

# Sort by removal benefit (most positive delta first)
remove_candidates = [r for r in removal_scores if r["delta"] > -0.3]
remove_candidates.sort(key=lambda x: -x["delta"])

removed = []
kept_neutral = []

for r in remove_candidates:
    feat = r["feature"]
    if feat not in current_set:
        continue

    test = [f for f in current_set if f != feat]
    result = ats_eval(X[test].values, y_margin, spreads)
    delta = result["composite"] - current_composite

    if delta > -0.3:  # removal doesn't significantly hurt
        current_set.remove(feat)
        old_composite = current_composite
        current_composite = result["composite"]
        removed.append(feat)
        tag = "HELPS" if delta > 0.1 else "neutral"
        print(f"  REMOVE {feat:<40s} Δ={delta:+.2f} ({tag}) comp={current_composite:.1f} → {len(current_set)}f")
    else:
        kept_neutral.append(feat)
        print(f"  KEEP   {feat:<40s} Δ={delta:+.2f}")

print(f"\n  Removed {len(removed)} features → {len(current_set)} remaining")
print(f"  Composite: {current_composite:.1f} (was {baseline['composite']:.1f})")


# ═══════════════════════════════════════════════════════════
# SECOND PASS: Try removing again (cascading effects)
# ═══════════════════════════════════════════════════════════

print(f"\n  Second pass: retrying {len(current_set)} features...")
pass2_removed = 0
for feat in list(current_set):
    if feat == "market_spread":  # always keep
        continue
    test = [f for f in current_set if f != feat]
    result = ats_eval(X[test].values, y_margin, spreads)
    delta = result["composite"] - current_composite

    if delta > 0.1:  # Only remove if clearly helps (tighter threshold for pass 2)
        current_set.remove(feat)
        current_composite = result["composite"]
        removed.append(feat)
        pass2_removed += 1
        print(f"  REMOVE2 {feat:<38s} Δ={delta:+.2f} comp={current_composite:.1f} → {len(current_set)}f")

print(f"  Pass 2 removed {pass2_removed} more → {len(current_set)} remaining")


# ═══════════════════════════════════════════════════════════
# FINAL VALIDATION: Head-to-Head
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  FINAL VALIDATION — Head-to-Head (20-fold walk-forward)")
print(f"{'='*70}")

# Also get the forward-selected sets for comparison
original_5 = state.get("phase4", {}).get("selected", [])

configs = {
    f"Backward-eliminated ({len(current_set)}f)": current_set,
    f"All survivors ({len(survivors)}f)": survivors,
    f"Original fwd-sel (5f)": original_5,
    f"All 102 features": feature_names,
    "Spread only (1f)": ["market_spread"],
}

print(f"\n  {'Config':<45s} {'MAE':>7s} {'ATS':>7s} {'ATS4+':>7s} {'ATS7+':>7s} {'ATS10+':>7s} {'Comp':>7s} {'N7':>5s}")
print("  " + "-" * 105)

best_config = None
best_composite = -999
config_results = {}

for label, feats in configs.items():
    available = [f for f in feats if f in X.columns]
    if not available:
        continue

    r = ats_eval(X[available].values, y_margin, spreads, n_folds=20)
    config_results[label] = r
    print(f"  {label:<45s} {r['mae']:>7.3f} {r['ats_0']:>6.1%} {r['ats_4']:>6.1%} "
          f"{r['ats_7']:>6.1%} {r['ats_10']:>6.1%} {r['composite']:>7.1f} {r['n_7']:>5d}")

    if r["composite"] > best_composite:
        best_composite = r["composite"]
        best_config = label

print(f"\n  BEST: {best_config} (composite={best_composite:.1f})")

# Per-learner breakdown for backward-eliminated set
print(f"\n  Per-learner breakdown (backward-eliminated set):")
X_final = X[current_set].values

def _single_eval(X_sub, y, spr, n_folds, builder):
    preds, idx = _wf_predict(X_sub, y, n_folds, builder)
    if len(preds) == 0: return None
    return _score_ats(preds, y[idx], spr[idx])

print(f"  {'Learner':<15s} {'MAE':>7s} {'ATS':>7s} {'ATS4+':>7s} {'ATS7+':>7s} {'ATS10+':>7s} {'Comp':>7s}")
print("  " + "-" * 65)

learner_fns = {
    "Lasso": lambda nf: Lasso(alpha=0.1, max_iter=5000),
    "Ridge": lambda nf: Ridge(alpha=1.0),
}
if HAS_CAT:
    learner_fns["CatBoost"] = lambda nf: CatBoostRegressor(depth=min(4, nf), iterations=500,
        learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0)
if HAS_GBM:
    learner_fns["GBM"] = lambda nf: GradientBoostingRegressor(n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42)

for name, builder in learner_fns.items():
    r = _single_eval(X_final, y_margin, spreads, 20, builder)
    if r:
        print(f"  {name:<15s} {r['mae']:>7.3f} {r['ats_0']:>6.1%} {r['ats_4']:>6.1%} "
              f"{r['ats_7']:>6.1%} {r['ats_10']:>6.1%} {r['composite']:>7.1f}")

br = config_results.get(f"Backward-eliminated ({len(current_set)}f)", {})
print(f"  {'CONSENSUS':<15s} {br.get('mae',0):>7.3f} {br.get('ats_0',0):>6.1%} {br.get('ats_4',0):>6.1%} "
      f"{br.get('ats_7',0):>6.1%} {br.get('ats_10',0):>6.1%} {br.get('composite',0):>7.1f}")


# ═══════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  FINAL FEATURE SET ({len(current_set)} features)")
print(f"{'='*70}")

for i, f in enumerate(sorted(current_set)):
    cov = (X[f] != 0).sum() / len(X) * 100
    is_strong = " ★" if f in strong_keep else ""
    print(f"  {i+1:3d}. {f:45s} cov={cov:.0f}%{is_strong}")

print(f"\n  Removed ({len(removed)}):")
for f in sorted(removed):
    print(f"    - {f}")

# Save
output = {
    "final_set": sorted(current_set),
    "removed": sorted(removed),
    "baseline_composite": baseline["composite"],
    "final_composite": current_composite,
    "n_survivors": len(survivors),
    "n_final": len(current_set),
}
with open("nba_v27_backward_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved to nba_v27_backward_results.json")

total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"  DONE — {total_time/60:.0f} min total")
print(f"{'='*70}")
