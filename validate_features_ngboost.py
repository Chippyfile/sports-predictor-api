#!/usr/bin/env python3
"""
Feature Pruning + NGBoost Validation (v2)
==========================================
Tier 1: All seasons, 55K clean games (all core stats filled), ref features removed
Tier 2: Test adding NGBoost as 4th learner
Tier 3: Test aggressive pruning of low-importance features

Run: python3 -u validate_features_ngboost.py
Prereq: pip install ngboost --break-system-packages
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sports.ncaa import ncaa_build_features
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, brier_score_loss

PARQUET = "ncaa_training_data.parquet"
N_FOLDS = 10  # Quick validation (use 50 for production)

# ═══════════════════════════════════════════════════════════════
# LOAD + FILTER
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  FEATURE VALIDATION v2 — Clean Data + NGBoost")
print("=" * 70)

print("\n  Loading data...")
df_raw = pd.read_parquet(PARQUET)
print(f"  Raw: {len(df_raw)} rows × {df_raw.shape[1]} cols")

# ESPN odds fallback
if "dk_spread_close" in df_raw.columns and "home_spread" in df_raw.columns:
    m1 = df_raw["home_spread"].isna() & df_raw["dk_spread_close"].notna()
    df_raw.loc[m1, "home_spread"] = df_raw.loc[m1, "dk_spread_close"]
    if "odds_api_spread_close" in df_raw.columns:
        m2 = df_raw["home_spread"].isna() & df_raw["odds_api_spread_close"].notna()
        df_raw.loc[m2, "home_spread"] = df_raw.loc[m2, "odds_api_spread_close"]
    print(f"  ESPN odds fallback: {m1.sum() + m2.sum() if 'm2' in dir() else m1.sum()} spreads filled")

# Require actual scores
df_raw = df_raw[df_raw["actual_home_score"].notna() & df_raw["actual_away_score"].notna()].copy()

# Core quality filter: all key stats must be non-null and non-zero
core_cols = [
    "home_ppg", "away_ppg", "home_fgpct", "away_fgpct",
    "home_adj_em", "away_adj_em", "home_threepct", "away_threepct",
    "home_tempo", "away_tempo", "home_orb_pct", "away_orb_pct",
    "home_turnovers", "away_turnovers", "home_assists", "away_assists",
    "home_ftpct", "away_ftpct", "home_steals", "away_steals",
    "home_blocks", "away_blocks", "home_wins", "away_wins",
    "home_opp_ppg", "away_opp_ppg", "home_elo", "away_elo",
]
existing_core = [c for c in core_cols if c in df_raw.columns]
core_mask = df_raw[existing_core].notna().all(axis=1) & (df_raw[existing_core] != 0).all(axis=1)
df_clean = df_raw[core_mask].copy()
print(f"  Clean data (all core filled): {len(df_clean)} games (dropped {len(df_raw) - len(df_clean)})")

# Build features
print("  Building features...")
# CRITICAL: Load referee profiles (97% of games have ref data, but features
# are 100% zero without this — ref_home_whistle is SHAP #4 at 2.7%)
import json as _json
try:
    with open("referee_profiles.json") as _rf:
        ncaa_build_features._ref_profiles = _json.load(_rf)
    print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
except FileNotFoundError:
    print("  ⚠ referee_profiles.json not found — ref features will be zero")
X_full = ncaa_build_features(df_clean)
all_features = list(X_full.columns)
y = (df_clean["actual_home_score"] - df_clean["actual_away_score"]).values
y_win = (y > 0).astype(int)
print(f"  {len(df_clean)} games × {len(all_features)} features")

# ═══════════════════════════════════════════════════════════════
# IDENTIFY DEAD FEATURES
# ═══════════════════════════════════════════════════════════════

# Features that are 100% zero in training — these are pure noise
zero_rate = (X_full == 0).mean()
always_zero = zero_rate[zero_rate >= 0.999].index.tolist()

# Ref features (0% data in ALL seasons)
ref_features = [c for c in all_features if c.startswith("ref_")]

# Features that are legitimately binary (don't remove just because they're mostly 0)
legit_binary = [
    "is_conf_tourney", "is_sandwich", "is_lookahead", "is_revenge_game",
    "is_midweek", "neutral", "is_early", "has_mkt", "has_ats_data",
    "has_ref_data", "is_postseason",
]

# True dead features: always zero AND not a legitimate binary flag
dead_features = [f for f in always_zero if f not in legit_binary]
# Also include ref features even if not in always_zero
dead_features = list(set(dead_features + ref_features))

print(f"\n  === DEAD FEATURES (removing from all tests) ===")
print(f"  Always zero (≥99.9%): {len(always_zero)}")
print(f"  Ref features:         {len(ref_features)}")
print(f"  Total dead:           {len(dead_features)}")
for f in sorted(dead_features):
    print(f"    {f:35s} zero_rate={zero_rate.get(f, 0)*100:.1f}%")

# Clean feature set = all features minus dead
clean_features = [f for f in all_features if f not in dead_features]
print(f"\n  Clean feature set: {len(clean_features)} (removed {len(dead_features)})")

# ═══════════════════════════════════════════════════════════════
# QUICK IMPORTANCE SCAN (to identify pruning candidates)
# ═══════════════════════════════════════════════════════════════

print("\n  Running quick XGBoost for feature importance...")
from xgboost import XGBRegressor
xgb_q = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1,
                      tree_method="hist", random_state=42, verbosity=0)
xgb_q.fit(X_full[clean_features], y)
imp = pd.Series(xgb_q.feature_importances_, index=clean_features).sort_values(ascending=False)

# Bottom 20 features by importance
bottom20 = imp.tail(20).index.tolist()
# Ultra-low importance (< 0.001)
ultra_low = imp[imp < 0.001].index.tolist()

print(f"  Bottom 20 features (candidates for aggressive pruning):")
for f in bottom20:
    print(f"    {f:35s} imp={imp[f]:.6f}  zero={zero_rate.get(f,0)*100:.0f}%")

# Pruned sets
pruned_low = [f for f in clean_features if f not in ultra_low]
pruned_aggressive = [f for f in clean_features if f not in bottom20]

print(f"\n  Feature sets:")
print(f"    Full clean:         {len(clean_features)}")
print(f"    Pruned (<0.001):    {len(pruned_low)} (removed {len(ultra_low)})")
print(f"    Pruned bottom 20:   {len(pruned_aggressive)} (removed {len(bottom20)})")

# ═══════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════

def make_xgb():
    return XGBRegressor(
        n_estimators=175, max_depth=7, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, verbosity=0)

def make_cat():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=500, depth=7, learning_rate=0.08,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        early_stopping_rounds=50)

def make_mlp():
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=300,
        learning_rate_init=0.001, early_stopping=True,
        validation_fraction=0.1, random_state=42)

def make_ngb():
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor
    return NGBRegressor(
        n_estimators=300, learning_rate=0.04,
        Base=DecisionTreeRegressor(max_depth=4, max_features=0.8),
        Dist=Normal, verbose=False, random_state=42)

# ═══════════════════════════════════════════════════════════════
# CV EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate(X_data, y_data, y_win_data, learner_map, label):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = {n: np.zeros(len(y_data)) for n in learner_map}
    oof_stack = np.zeros(len(y_data))

    for fold, (tr, va) in enumerate(kf.split(X_data)):
        Xtr, Xva = X_data.iloc[tr], X_data.iloc[va]
        ytr, yva = y_data[tr], y_data[va]
        preds_tr, preds_va = {}, {}

        for name, build in learner_map.items():
            m = build()
            if name == "cat":
                m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
            elif name == "ngb":
                m.fit(Xtr.values, ytr, X_val=Xva.values, Y_val=yva)
            else:
                m.fit(Xtr, ytr)
            p_tr = m.predict(Xtr.values if name == "ngb" else Xtr)
            p_va = m.predict(Xva.values if name == "ngb" else Xva)
            preds_tr[name] = p_tr
            preds_va[name] = p_va
            oof[name][va] = p_va

        S_tr = np.column_stack([preds_tr[n] for n in learner_map])
        S_va = np.column_stack([preds_va[n] for n in learner_map])
        meta = Ridge(alpha=1.0, fit_intercept=False)
        meta.fit(S_tr, ytr)
        oof_stack[va] = meta.predict(S_va)

    mae = mean_absolute_error(y_data, oof_stack)
    sigma = np.std(y_data - oof_stack)
    probs = np.clip(1 / (1 + 10 ** (-oof_stack / sigma)), 0.01, 0.99)
    brier = brier_score_loss(y_win_data, probs)
    per = {n: mean_absolute_error(y_data, oof[n]) for n in learner_map}
    weights = meta.coef_

    return {"mae": mae, "brier": brier, "sigma": sigma,
            "weights": dict(zip(learner_map.keys(), weights)),
            "per_model": per, "n_features": X_data.shape[1]}

# ═══════════════════════════════════════════════════════════════
# RUN CONFIGS
# ═══════════════════════════════════════════════════════════════

three_learners = {"xgb": make_xgb, "cat": make_cat, "mlp": make_mlp}
four_learners = {"xgb": make_xgb, "cat": make_cat, "mlp": make_mlp, "ngb": make_ngb}

configs = [
    ("A: Clean baseline (no dead features)",       clean_features,      three_learners),
    ("B: Prune ultra-low (<0.001 importance)",      pruned_low,          three_learners),
    ("C: Prune bottom 20 features",                 pruned_aggressive,   three_learners),
    ("D: Clean + NGBoost (4 learners)",             clean_features,      four_learners),
    ("E: Pruned ultra-low + NGBoost",               pruned_low,          four_learners),
    ("F: Pruned bottom 20 + NGBoost",               pruned_aggressive,   four_learners),
]

results = []

for label, feats, learners in configs:
    n_f = len(feats)
    n_l = len(learners)
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  {n_f} features | {n_l} learners: {', '.join(learners.keys())}")
    print(f"  {len(y)} games | {N_FOLDS}-fold CV")
    print(f"{'─'*70}")

    t0 = time.time()
    try:
        r = evaluate(X_full[feats], y, y_win, learners, label)
        elapsed = time.time() - t0
        print(f"  ✅ Stacked MAE: {r['mae']:.4f}  |  Brier: {r['brier']:.4f}  |  σ: {r['sigma']:.2f}")
        print(f"     Weights: {', '.join(f'{k}={v:.3f}' for k,v in r['weights'].items())}")
        print(f"     Per-model: {', '.join(f'{k}={v:.3f}' for k,v in r['per_model'].items())}")
        print(f"     Time: {elapsed:.0f}s")
        results.append({"label": label, **r, "elapsed": elapsed})
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()
        results.append({"label": label, "mae": None, "error": str(e)})

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)

valid = [r for r in results if r.get("mae")]
if not valid:
    print("  No valid results.")
else:
    best = min(valid, key=lambda r: r["mae"])
    base = valid[0]["mae"]

    print(f"\n  {'Config':<45} {'Feats':>5} {'MAE':>8} {'Brier':>8} {'Δ MAE':>8} {'Time':>6}")
    print(f"  {'─'*45} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    for r in valid:
        d = r["mae"] - base
        star = " ★" if r["mae"] == best["mae"] else ""
        print(f"  {r['label']:<45} {r['n_features']:>5} {r['mae']:>8.4f} {r['brier']:>8.4f} {d:>+8.4f} {r['elapsed']:>5.0f}s{star}")

    print(f"\n  BEST: {best['label']}")
    print(f"  MAE:  {best['mae']:.4f} (Δ {best['mae'] - base:+.4f} vs baseline)")
    print(f"  Weights: {', '.join(f'{k}={v:.3f}' for k,v in best['weights'].items())}")

    # Feature pruning verdict
    three_only = [r for r in valid if "NGBoost" not in r["label"] and "4 learner" not in r["label"]]
    if len(three_only) >= 2:
        best_3 = min(three_only, key=lambda r: r["mae"])
        print(f"\n  PRUNING VERDICT (3-learner):")
        print(f"    Best 3-learner: {best_3['label']} — MAE {best_3['mae']:.4f}")
        if best_3["n_features"] < valid[0]["n_features"]:
            print(f"    ✅ Fewer features is better — drop {valid[0]['n_features'] - best_3['n_features']} features")
        else:
            print(f"    ⚪ Baseline features are optimal")

    # NGBoost verdict
    ngb = [r for r in valid if "NGBoost" in r["label"] or "4 learner" in r["label"]]
    non_ngb = [r for r in valid if r not in ngb]
    if ngb and non_ngb:
        best_ngb = min(ngb, key=lambda r: r["mae"])
        best_no = min(non_ngb, key=lambda r: r["mae"])
        delta = best_ngb["mae"] - best_no["mae"]
        print(f"\n  NGBOOST VERDICT:")
        print(f"    Best without: {best_no['label']} — MAE {best_no['mae']:.4f}")
        print(f"    Best with:    {best_ngb['label']} — MAE {best_ngb['mae']:.4f}")
        print(f"    Delta:        {delta:+.4f}")
        if delta < -0.005:
            print(f"    ✅ NGBoost helps — add as 4th learner")
            print(f"       Weight: {best_ngb['weights'].get('ngb', 0):.3f}")
        elif delta < 0:
            print(f"    🟡 Marginal improvement — test at 50-fold before committing")
        else:
            print(f"    ⚪ NGBoost doesn't help — keep 3-learner stack")

    # Final recommendation
    print(f"\n  {'═'*50}")
    print(f"  RECOMMENDATION:")
    if best["mae"] < base - 0.01:
        print(f"  ✅ Switch to: {best['label']}")
        print(f"     Features: {best['n_features']}")
        print(f"     Improvement: {base - best['mae']:.4f} MAE")
    elif best["mae"] < base:
        print(f"  🟡 Slight improvement with {best['label']}")
        print(f"     Confirm at 50-fold before production deploy")
    else:
        print(f"  ⚪ Current config is optimal")
    print(f"  {'═'*50}")

print("\n  Done.")
