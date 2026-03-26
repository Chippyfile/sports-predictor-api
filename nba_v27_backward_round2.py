#!/usr/bin/env python3
"""
nba_v27_backward_round2.py — Full validation of backward elimination survivors.

Three phases:
  1. FULL RE-TEST: Test removing every survivor (no auto-keep cutoff)
  2. GROUP ADD-BACK: Add all borderline removed features at once
  3. NUCLEAR: Shuffle removal order 5×, take intersection

Run from ~/Desktop/sports-predictor-api/ AFTER nba_v27_backward_elim.py finishes:
    python3 nba_v27_backward_round2.py
"""
import sys, os, json, time, random, warnings, copy
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

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_GBM = True
except ImportError:
    HAS_GBM = False


# ═══════════════════════════════════════════════════════════
# LOAD RESULTS + DATA
# ═══════════════════════════════════════════════════════════

BACKWARD_RESULTS = "nba_v27_backward_results.json"
if not os.path.exists(BACKWARD_RESULTS):
    print(f"ERROR: {BACKWARD_RESULTS} not found. Run nba_v27_backward_elim.py first.")
    sys.exit(1)

with open(BACKWARD_RESULTS) as f:
    bk = json.load(f)

survivors = bk["final_set"]
removed = bk["removed"]
print("=" * 70)
print("  NBA v27 BACKWARD ROUND 2 — Full Validation")
print("=" * 70)
print(f"\n  Round 1 survivors: {len(survivors)}")
print(f"  Round 1 removed: {len(removed)}")

# Load data
print("\nLoading training data...")
from nba_v27_eliminate import load_training_data, build_features, prepare_targets

df = load_training_data()
X, feature_names = build_features(df)
targets = prepare_targets(df)
spreads = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values

dates = pd.to_datetime(df["game_date"])
sort_idx = dates.argsort()
X = X.iloc[sort_idx].reset_index(drop=True)
for tname in targets:
    targets[tname] = targets[tname][sort_idx]
spreads = spreads[sort_idx]

y_margin = targets["ML"]
n = len(X)

# Verify survivors exist in X
survivors = [f for f in survivors if f in X.columns]
removed = [f for f in removed if f in X.columns]
print(f"  {n} games, {len(X.columns)} features, available survivors: {len(survivors)}")


# ═══════════════════════════════════════════════════════════
# MULTI-LEARNER ATS EVALUATION (same as backward_elim)
# ═══════════════════════════════════════════════════════════

def _score_ats(preds, actual, spr):
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

    return _score_ats(ensemble_preds, y[common_idx], spr[common_idx])


# ═══════════════════════════════════════════════════════════
# PHASE 1: FULL RE-TEST (no auto-keep cutoff)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 1: Full re-test of all {len(survivors)} survivors")
print(f"  (No auto-keep cutoff — every feature gets tested)")
print(f"{'='*70}")

t0 = time.time()

# Baseline
baseline = ats_eval(X[survivors].values, y_margin, spreads)
print(f"\n  Baseline ({len(survivors)}f): composite={baseline['composite']:.1f}, "
      f"ATS7={baseline['ats_7']:.1%}, MAE={baseline['mae']:.3f}")

current_composite = baseline["composite"]

# Score every feature
removal_scores = []
for i, feat in enumerate(survivors):
    test = [f for f in survivors if f != feat]
    r = ats_eval(X[test].values, y_margin, spreads)
    delta = r["composite"] - current_composite
    removal_scores.append({"feature": feat, "delta": delta, "composite_without": r["composite"],
                            "mae_without": r["mae"]})
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(survivors) - i - 1)
        print(f"  [{i+1}/{len(survivors)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

removal_scores.sort(key=lambda x: -x["delta"])

print(f"\n  {'Rank':>4} {'Feature':<40s} {'ΔComp':>8s} {'Action'}")
print("  " + "-" * 60)
for i, r in enumerate(removal_scores):
    delta = r["delta"]
    if delta > 0.5: action = "REMOVE ✗"
    elif delta > 0.0: action = "maybe rm"
    elif delta > -0.3: action = "weak keep"
    elif delta > -1.0: action = "KEEP"
    else: action = "CRITICAL ✓"
    print(f"  {i+1:4d} {r['feature']:<40s} {delta:>+7.2f}  {action}")

# Iterative removal (threshold -0.3 same as Round 1)
print(f"\n  Iterative removal...")
current_set = list(survivors)
round2_removed = []

remove_candidates = [r for r in removal_scores if r["delta"] > -0.3]
remove_candidates.sort(key=lambda x: -x["delta"])

for r in remove_candidates:
    feat = r["feature"]
    if feat not in current_set:
        continue
    test = [f for f in current_set if f != feat]
    result = ats_eval(X[test].values, y_margin, spreads)
    delta = result["composite"] - current_composite
    if delta > -0.3:
        current_set.remove(feat)
        current_composite = result["composite"]
        round2_removed.append(feat)
        tag = "HELPS" if delta > 0.1 else "neutral"
        print(f"  REMOVE {feat:<40s} Δ={delta:+.2f} ({tag}) → {len(current_set)}f")
    else:
        print(f"  KEEP   {feat:<40s} Δ={delta:+.2f}")

print(f"\n  Round 2: removed {len(round2_removed)} more → {len(current_set)} remaining")
print(f"  Composite: {current_composite:.1f}")

round2_survivors = list(current_set)


# ═══════════════════════════════════════════════════════════
# PHASE 2: GROUP ADD-BACK TEST
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 2: Group Add-Back Test")
print(f"{'='*70}")

# Collect ALL removed features from both rounds (Δ -0.10 to -0.29)
all_removed = removed + round2_removed
borderline = []

# Re-score each removed feature against the current set
for feat in all_removed:
    if feat not in X.columns:
        continue
    test = round2_survivors + [feat]
    r = ats_eval(X[test].values, y_margin, spreads)
    delta = r["composite"] - current_composite
    borderline.append({"feature": feat, "delta_individual": delta})

borderline.sort(key=lambda x: -x["delta_individual"])

# Features where individual add-back shows any positive signal
positive_addbacks = [b["feature"] for b in borderline if b["delta_individual"] > -0.05]

print(f"\n  Individual add-back scores ({len(all_removed)} removed features):")
for b in borderline[:20]:
    print(f"    {b['feature']:<40s} Δ={b['delta_individual']:+.2f}")

# Now test ALL borderline features added back at once
if positive_addbacks:
    test_group = round2_survivors + positive_addbacks
    r_group = ats_eval(X[test_group].values, y_margin, spreads)
    group_delta = r_group["composite"] - current_composite
    print(f"\n  Group add-back ({len(positive_addbacks)} features):")
    print(f"    Features: {positive_addbacks}")
    print(f"    Composite: {r_group['composite']:.1f} (current {current_composite:.1f}, Δ={group_delta:+.1f})")
    print(f"    ATS 7+: {r_group['ats_7']:.1%}")

    if group_delta > 1.0:
        print(f"\n    ✓ Group add-back improves by {group_delta:+.1f} — these features carry collective signal")
        print(f"    Adding them to final set")
        round2_survivors.extend(positive_addbacks)
        current_composite = r_group["composite"]
    else:
        print(f"\n    ✗ Group add-back only {group_delta:+.1f} — not enough collective signal")
else:
    print(f"\n  No positive add-back candidates found")


# ═══════════════════════════════════════════════════════════
# PHASE 3: NUCLEAR — Shuffle removal order 5×
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 3: Nuclear Validation — 5 Random Shuffle Orders")
print(f"{'='*70}")

# Start from the original Round 1 survivors (before any iterative removal)
original_survivors = list(survivors)

shuffle_results = []
for run in range(5):
    seed = 42 + run * 17
    random.seed(seed)

    current = list(original_survivors)
    
    # Get baseline
    base_r = ats_eval(X[current].values, y_margin, spreads)
    comp = base_r["composite"]

    # Shuffle the testing order
    test_order = list(current)
    random.shuffle(test_order)

    run_removed = []
    for feat in test_order:
        if feat not in current:
            continue
        test = [f for f in current if f != feat]
        r = ats_eval(X[test].values, y_margin, spreads)
        delta = r["composite"] - comp
        if delta > -0.3:
            current.remove(feat)
            comp = r["composite"]
            run_removed.append(feat)

    shuffle_results.append({
        "seed": seed,
        "n_final": len(current),
        "composite": comp,
        "survivors": set(current),
        "removed": set(run_removed),
    })
    print(f"  Run {run+1} (seed={seed}): {len(current)} survivors, composite={comp:.1f}, "
          f"removed {len(run_removed)}")

# Intersection: features that survived ALL 5 runs
intersection = shuffle_results[0]["survivors"]
for sr in shuffle_results[1:]:
    intersection = intersection & sr["survivors"]

# Union of removed: features removed in ANY run
union_removed = set()
for sr in shuffle_results:
    union_removed = union_removed | sr["removed"]

# Contested: survived some runs, removed in others
all_features_tested = set(original_survivors)
always_kept = intersection
always_removed = all_features_tested - set().union(*[sr["survivors"] for sr in shuffle_results])
contested = all_features_tested - always_kept - always_removed

print(f"\n  Results across 5 shuffles:")
print(f"    Always kept (intersection):  {len(always_kept)} features")
print(f"    Always removed:              {len(always_removed)} features")
print(f"    Contested (order-dependent): {len(contested)} features")

# Evaluate the intersection set
inter_list = sorted(always_kept)
r_inter = ats_eval(X[inter_list].values, y_margin, spreads, n_folds=20)
print(f"\n  Intersection set ({len(inter_list)}f):")
print(f"    Composite: {r_inter['composite']:.1f}")
print(f"    ATS 7+: {r_inter['ats_7']:.1%}")
print(f"    MAE: {r_inter['mae']:.3f}")

# Evaluate intersection + contested
inter_plus_contested = sorted(always_kept | contested)
r_ipc = ats_eval(X[inter_plus_contested].values, y_margin, spreads, n_folds=20)
print(f"\n  Intersection + Contested ({len(inter_plus_contested)}f):")
print(f"    Composite: {r_ipc['composite']:.1f}")
print(f"    ATS 7+: {r_ipc['ats_7']:.1%}")
print(f"    MAE: {r_ipc['mae']:.3f}")


# ═══════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  FINAL COMPARISON — All Approaches (20-fold)")
print(f"{'='*70}")

configs = {
    f"Round 2 survivors ({len(round2_survivors)}f)": round2_survivors,
    f"Nuclear intersection ({len(inter_list)}f)": inter_list,
    f"Intersection + contested ({len(inter_plus_contested)}f)": inter_plus_contested,
    f"Forward-selected (10f)": [
        "market_spread", "streak_diff", "efg_diff", "implied_prob_home",
        "lineup_value_diff", "roll_three_fg_rate_diff", "margin_skew_diff",
        "is_early_season", "opp_suppression_diff", "scoring_hhi_diff"
    ],
    f"Round 1 survivors ({len(survivors)}f)": survivors,
    "All 105 features": feature_names,
}

print(f"\n  {'Config':<50s} {'MAE':>7s} {'ATS':>7s} {'ATS4+':>7s} {'ATS7+':>7s} {'ATS10+':>7s} {'Comp':>7s}")
print("  " + "-" * 100)

best_config = None
best_composite = -999

for label, feats in configs.items():
    available = [f for f in feats if f in X.columns]
    if not available:
        continue
    r = ats_eval(X[available].values, y_margin, spreads, n_folds=20)
    print(f"  {label:<50s} {r['mae']:>7.3f} {r['ats_0']:>6.1%} {r['ats_4']:>6.1%} "
          f"{r['ats_7']:>6.1%} {r['ats_10']:>6.1%} {r['composite']:>7.1f}")
    if r["composite"] > best_composite:
        best_composite = r["composite"]
        best_config = label

print(f"\n  BEST: {best_config} (composite={best_composite:.1f})")


# ═══════════════════════════════════════════════════════════
# OUTPUT: RECOMMENDED FEATURE SET
# ═══════════════════════════════════════════════════════════

# Pick the best performing set
if "Nuclear intersection" in best_config:
    final = inter_list
elif "Intersection + contested" in best_config:
    final = inter_plus_contested
elif "Round 2" in best_config:
    final = round2_survivors
else:
    final = round2_survivors  # default

print(f"\n{'='*70}")
print(f"  RECOMMENDED FEATURE SET ({len(final)} features)")
print(f"{'='*70}")

for i, f in enumerate(sorted(final)):
    cov = (X[f] != 0).sum() / len(X) * 100
    print(f"  {i+1:3d}. {f:45s} coverage={cov:.0f}%")

print(f"\n  Contested features (order-dependent — worth monitoring):")
for f in sorted(contested):
    cov = (X[f] != 0).sum() / len(X) * 100
    print(f"    ? {f:45s} coverage={cov:.0f}%")

# Save
output = {
    "recommended_set": sorted(final),
    "round2_survivors": sorted(round2_survivors),
    "nuclear_intersection": sorted(always_kept),
    "contested": sorted(contested),
    "always_removed": sorted(always_removed),
    "group_addback_tested": positive_addbacks if positive_addbacks else [],
    "best_composite": best_composite,
}
with open("nba_v27_round2_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Saved to nba_v27_round2_results.json")

total = time.time() - t0
print(f"\n{'='*70}")
print(f"  DONE — {total/3600:.1f} hours total")
print(f"{'='*70}")
