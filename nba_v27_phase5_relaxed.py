#!/usr/bin/env python3
"""
nba_v27_phase5_relaxed.py — Run Phase 4 (relaxed) + Phase 5 validation.

The original Phase 4 used composite threshold 0.2 and stalled at 5 features.
This re-runs with threshold 0.05 to find the full ATS-optimal feature set,
then validates it in Phase 5 against the original 5-feature set and all 102.

Run from ~/Desktop/sports-predictor-api/:
    python3 nba_v27_phase5_relaxed.py
"""
import sys, os, json, time, copy, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("WARNING: CatBoost not found, using Lasso (install catboost for best results)")


# ═══════════════════════════════════════════════════════════
# LOAD PRIOR RESULTS
# ═══════════════════════════════════════════════════════════

STATE_FILE = "nba_v27_elimination_results.json"
if not os.path.exists(STATE_FILE):
    print(f"ERROR: {STATE_FILE} not found. Run nba_v27_eliminate.py --phase4 first.")
    sys.exit(1)

with open(STATE_FILE) as f:
    state = json.load(f)

if "phase4" not in state or "phase3" not in state or "phase1" not in state:
    print("ERROR: Need phases 1-4 complete. Run nba_v27_eliminate.py --phase1 --phase2 --phase3 --phase4")
    sys.exit(1)

original_5 = state["phase4"]["selected"]
print(f"  Prior Phase 4 result: {len(original_5)} features: {original_5}")


# ═══════════════════════════════════════════════════════════
# LOAD TRAINING DATA (same as nba_v27_eliminate.py)
# ═══════════════════════════════════════════════════════════

print("\nLoading training data...")

# Try loading via the eliminate script's function
try:
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
    df = df.iloc[sort_idx].reset_index(drop=True)
except Exception as e:
    print(f"  Direct import failed ({e}), loading from parquet + Supabase...")

    # Fallback: load parquet for features, Supabase for targets
    from db import sb_get
    from sports.nba import _nba_merge_historical

    rows = sb_get("nba_predictions", "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df, _, _ = _nba_merge_historical(current_df)

    X_parquet = pd.read_parquet("nba_v27_features.parquet")
    feature_names = sorted(X_parquet.columns.tolist())

    y_margin = (df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)).values
    spreads = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce").fillna(0).values

    # Align lengths
    n = min(len(X_parquet), len(y_margin))
    X = X_parquet.iloc[:n]
    targets = {"ML": y_margin[:n]}
    spreads = spreads[:n]

n = len(X)
y_margin = targets["ML"]
has_spread = np.abs(spreads) > 0.1
print(f"  {n} games, {len(feature_names)} features, {has_spread.sum()} with spreads")


# ═══════════════════════════════════════════════════════════
# ATS EVALUATION — MULTI-LEARNER CONSENSUS
# ═══════════════════════════════════════════════════════════

try:
    from sklearn.linear_model import Ridge, ElasticNet
    HAS_RIDGE = True
except ImportError:
    HAS_RIDGE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_GBM = True
except ImportError:
    HAS_GBM = False


def _score_ats(preds, actual, spr):
    """Compute ATS metrics from prediction arrays."""
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


def ats_eval(X_sub, y, spr, n_folds=15, model_type="consensus"):
    """
    Multi-learner consensus ATS evaluation.
    Runs CatBoost + Lasso + Ridge, averages their predictions,
    then scores ATS on the ensemble. This prevents feature selection
    from being biased toward any single model architecture.
    """
    if isinstance(X_sub, pd.DataFrame):
        X_sub = X_sub.values
    if X_sub.ndim == 1:
        X_sub = X_sub.reshape(-1, 1)

    n_f = X_sub.shape[1] if len(X_sub.shape) > 1 else 1

    # Build model factories
    def make_lasso(nf):
        return Lasso(alpha=0.1, max_iter=5000)
    def make_ridge(nf):
        return Ridge(alpha=1.0)
    def make_catboost(nf):
        return CatBoostRegressor(depth=min(4, nf), iterations=500, learning_rate=0.05,
                                  l2_leaf_reg=3, random_seed=42, verbose=0)
    def make_gbm(nf):
        return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, random_state=42)

    learners = {"Lasso": make_lasso, "Ridge": make_ridge}
    if HAS_CAT and n_f >= 2:
        learners["CatBoost"] = make_catboost
    if HAS_GBM and n_f >= 2:
        learners["GBM"] = make_gbm

    # Run walk-forward for each learner
    all_learner_preds = {}
    common_idx = None
    for name, builder in learners.items():
        preds, idx = _wf_predict(X_sub, y, n_folds, builder)
        all_learner_preds[name] = (preds, idx)
        if common_idx is None:
            common_idx = set(idx)
        else:
            common_idx = common_idx & set(idx)

    if not common_idx:
        return {"composite": -100, "mae": 99, "ats_0": 0.5}

    # Average predictions across learners (ensemble)
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
# PHASE 4 RELAXED: Forward selection with lower threshold
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 4 RELAXED: Forward Selection (threshold=0.05, stall=40)")
print(f"{'='*70}")

# Get Phase 2 results for ranking
p2_results = state["phase2"]["results"]
survivors = state["phase3"]["survivors"]
survivors.sort(key=lambda f: -p2_results.get(f, {}).get("avg_delta", 0))

# Correlation groups
corr_groups = state["phase1"]["groups"]
group_map = {}
for group in corr_groups:
    for f in group:
        group_map[f] = set(group)

# Start with market_spread
selected = ["market_spread"]
if "market_spread" not in survivors:
    survivors = ["market_spread"] + survivors

X_sel = X[selected].values
base = ats_eval(X_sel, y_margin, spreads)
current_composite = base["composite"]
print(f"  Baseline (spread only): composite={current_composite:.1f}, MAE={base['mae']}")
print(f"  ATS 0+: {base['ats_0']:.1%}  4+: {base['ats_4']:.1%}  7+: {base['ats_7']:.1%}  10+: {base['ats_10']:.1%}")

candidates = [f for f in survivors if f != "market_spread" and f in X.columns]
log = [{"step": 0, "feature": "market_spread", "composite": current_composite}]
stall = 0
THRESHOLD = 0.05  # Relaxed from 0.2
MAX_STALL = 40    # Let it search deep — 102 candidates is small enough

t0 = time.time()
for i, feat in enumerate(candidates):
    if stall >= MAX_STALL:
        print(f"\n  {MAX_STALL} consecutive non-improvements, stopping")
        break

    # Skip correlated partners
    if feat in group_map:
        already = group_map[feat] & set(selected)
        if already:
            continue

    test = selected + [feat]
    X_test = X[test].values
    r = ats_eval(X_test, y_margin, spreads)
    delta = r["composite"] - current_composite

    if delta > THRESHOLD:
        selected.append(feat)
        current_composite = r["composite"]
        stall = 0
        log.append({"step": len(selected), "feature": feat,
                     "composite": round(current_composite, 2), "delta": round(delta, 2),
                     "ats_4": r.get("ats_4", 0), "ats_7": r.get("ats_7", 0),
                     "ats_10": r.get("ats_10", 0)})
        print(f"  [{len(selected):3d}] ADD  {feat:40s} comp={current_composite:.1f} "
              f"Δ={delta:+.1f}  ATS7={r.get('ats_7',0):.1%}  MAE={r['mae']}")
    else:
        stall += 1
        if stall <= 3 or stall % 5 == 0:
            print(f"  [   ] skip {feat:40s} Δ={delta:+.2f}")

elapsed = time.time() - t0
print(f"\n  Forward selection pass 1: {len(selected)} features in {elapsed:.0f}s")
print(f"  Composite: {current_composite:.1f}")

# Second pass: retry ALL non-selected features (early skips may help now that base is larger)
print(f"\n  Second pass: retrying all {len(feature_names) - len(selected)} remaining features...")
all_remaining = [f for f in feature_names if f not in selected and f in X.columns]
added_pass2 = 0
for feat in all_remaining:
    # Skip correlated partners
    if feat in group_map:
        already = group_map[feat] & set(selected)
        if already:
            continue
    test = selected + [feat]
    X_test = X[test].values
    r = ats_eval(X_test, y_margin, spreads)
    delta = r["composite"] - current_composite
    if delta > THRESHOLD:
        selected.append(feat)
        current_composite = r["composite"]
        added_pass2 += 1
        print(f"  [{len(selected):3d}] ADD2 {feat:40s} comp={current_composite:.1f} "
              f"Δ={delta:+.1f}  ATS7={r.get('ats_7',0):.1%}")
print(f"  Pass 2 added {added_pass2} features → {len(selected)} total")

# Backward prune
print(f"\n  Backward pruning...")
for feat in list(selected):
    if feat == "market_spread":
        continue
    test = [f for f in selected if f != feat]
    r = ats_eval(X[test].values, y_margin, spreads)
    delta = r["composite"] - current_composite
    if delta >= -0.05:  # removing doesn't hurt (relaxed)
        selected.remove(feat)
        current_composite = r["composite"]
        print(f"  PRUNE {feat:40s} Δ={delta:+.2f} → {len(selected)}f")
    else:
        print(f"  KEEP  {feat:40s} Δ={delta:+.2f}")

relaxed_set = list(selected)
print(f"\n  Relaxed optimal: {len(relaxed_set)} features")


# ═══════════════════════════════════════════════════════════
# PHASE 5: Compare all configurations
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PHASE 5: Final Validation — Head-to-Head Comparison")
print(f"{'='*70}")

# Current production v27 (38 features)
V27_PROD = [
    "lineup_value_diff", "win_pct_diff", "scoring_hhi_diff",
    "espn_pregame_wp", "ceiling_diff", "matchup_efg",
    "ml_implied_spread", "sharp_spread_signal", "efg_diff",
    "opp_suppression_diff", "net_rtg_diff", "steals_to_diff",
    "threepct_diff", "b2b_diff", "ftpct_diff", "ou_gap",
    "roll_dreb_diff", "ts_regression_diff", "roll_paint_pts_diff",
    "ref_home_whistle", "opp_ppg_diff", "roll_max_run_avg",
    "away_is_public_team", "away_after_loss", "games_last_14_diff",
    "h2h_total_games", "three_pt_regression_diff", "games_diff",
    "ref_foul_proxy", "roll_fast_break_diff", "crowd_pct",
    "matchup_to", "overround", "spread_juice_imbalance",
    "vig_uncertainty", "roll_ft_trip_rate_diff", "home_after_loss",
    "rest_diff",
]
v27_available = [f for f in V27_PROD if f in X.columns]

configs = {
    "Original Phase 4 (5 features)": original_5,
    f"Relaxed Phase 4 ({len(relaxed_set)} features)": relaxed_set,
    f"Current v27 prod ({len(v27_available)} features)": v27_available,
    f"All 102 features": feature_names,
}

print(f"\n  {'Config':<45s} {'MAE':>7s} {'ATS':>7s} {'ATS4+':>7s} {'ATS7+':>7s} {'ATS10+':>7s} {'Comp':>7s} {'ROI7':>7s}")
print("  " + "-" * 100)

best_config = None
best_composite = -999

for label, feats in configs.items():
    available = [f for f in feats if f in X.columns]
    if not available:
        print(f"  {label:<45s} NO FEATURES AVAILABLE")
        continue

    r = ats_eval(X[available].values, y_margin, spreads, n_folds=20)
    print(f"  {label:<45s} {r['mae']:>7.3f} {r['ats_0']:>6.1%} {r['ats_4']:>6.1%} "
          f"{r['ats_7']:>6.1%} {r['ats_10']:>6.1%} {r['composite']:>7.1f} {r['roi_7']:>+6.1f}%")

    if r["composite"] > best_composite:
        best_composite = r["composite"]
        best_config = label

print(f"\n  BEST: {best_config} (composite={best_composite:.1f})")

# Per-learner breakdown for the best config
print(f"\n  Per-learner breakdown for best config:")
best_feats = configs[best_config]
best_available = [f for f in best_feats if f in X.columns]
X_best = np.array(X[best_available].values, dtype=float)

def _single_learner_eval(X_sub, y, spr, n_folds, builder):
    """Walk-forward for a single learner."""
    preds, idx = _wf_predict(X_sub, y, n_folds, builder)
    if len(preds) == 0:
        return None
    return _score_ats(preds, y[idx], spr[idx])

def make_lasso_fn(nf): return Lasso(alpha=0.1, max_iter=5000)
def make_ridge_fn(nf): return Ridge(alpha=1.0)
def make_cat_fn(nf): return CatBoostRegressor(depth=min(4, nf), iterations=500,
    learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0)
def make_gbm_fn(nf): return GradientBoostingRegressor(n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8, random_state=42)

single_learners = {"Lasso": make_lasso_fn, "Ridge": make_ridge_fn}
if HAS_CAT: single_learners["CatBoost"] = make_cat_fn
if HAS_GBM: single_learners["GBM"] = make_gbm_fn

print(f"  {'Learner':<15s} {'MAE':>7s} {'ATS':>7s} {'ATS4+':>7s} {'ATS7+':>7s} {'ATS10+':>7s} {'Comp':>7s}")
print("  " + "-" * 65)
for name, builder in single_learners.items():
    r = _single_learner_eval(X_best, y_margin, spreads, 20, builder)
    if r:
        print(f"  {name:<15s} {r['mae']:>7.3f} {r['ats_0']:>6.1%} {r['ats_4']:>6.1%} "
              f"{r['ats_7']:>6.1%} {r['ats_10']:>6.1%} {r['composite']:>7.1f}")
print(f"  {'CONSENSUS':<15s} — see main table above")


# ═══════════════════════════════════════════════════════════
# PRINT OPTIMAL FEATURE SET
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  RELAXED OPTIMAL FEATURE SET ({len(relaxed_set)} features)")
print(f"{'='*70}")
for i, f in enumerate(relaxed_set):
    cov = (X[f] != 0).sum() / len(X) * 100
    print(f"  {i+1:3d}. {f:45s} coverage={cov:.0f}%")

# Save
output = {
    "relaxed_set": relaxed_set,
    "relaxed_composite": current_composite,
    "original_5": original_5,
    "log": log,
}
with open("nba_v27_relaxed_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved to nba_v27_relaxed_results.json")

print(f"\n{'='*70}")
print(f"  DONE — {time.time()-t0:.0f}s total")
print(f"{'='*70}")
PYEOF
