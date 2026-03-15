#!/usr/bin/env python3
"""
test_ensemble_combos.py — Test ensemble combinations with proper OOF stacking
==============================================================================
Uses nested CV: inner folds generate out-of-fold predictions for meta-learner,
outer folds evaluate final stacked MAE. This matches retrain_and_upload.py logic.
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic

print("=" * 70)
print("  ENSEMBLE COMBO TEST — Proper OOF stacking, 10-fold TSCV")
print("=" * 70)

# ── Load data once ──────────────────────────────────────────────────
print("\n  Loading data...")
t0 = time.time()
rows = sb_get("ncaa_historical", "actual_home_score=not.is.null&order=season.asc")
df = pd.DataFrame(rows)
print(f"  Loaded {len(df)} rows in {time.time()-t0:.0f}s")

# ESPN odds fallback
for c1, c2 in [("espn_spread", "market_spread_home"), ("espn_over_under", "market_ou_total")]:
    if c1 in df.columns and c2 in df.columns:
        m = df[c2].isna() & df[c1].notna()
        df.loc[m, c2] = df.loc[m, c1]

# Quality filter
qc = ["home_adj_em", "away_adj_em", "home_ppg", "away_ppg", "market_spread_home", "market_ou_total"]
existing = [c for c in qc if c in df.columns]
if existing:
    cov = df[existing].notna().mean(axis=1)
    before = len(df)
    df = df[cov >= 0.8].copy()
    print(f"  Quality filter: {len(df)}/{before} games")

# Heuristic backfill
df = _ncaa_backfill_heuristic(df)

# Referee profiles
try:
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
    print(f"  Loaded referee profiles")
except:
    pass

# Build features
X = ncaa_build_features(df)
y = (df["actual_home_score"] - df["actual_away_score"]).astype(float).values
print(f"  {len(df)} games × {X.shape[1]} features\n")

X_np = X.values.astype(np.float32)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# ── Model builders ──────────────────────────────────────────────────
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit


def make_model(name, seed=42):
    if name == "xgb":
        return xgb.XGBRegressor(n_estimators=175, max_depth=7, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=seed)
    elif name == "cat":
        return cb.CatBoostRegressor(iterations=175, depth=7, learning_rate=0.1,
                                     random_seed=seed, verbose=0)
    elif name == "lgb":
        return lgb.LGBMRegressor(n_estimators=175, max_depth=7, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)
    elif name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200,
                            random_state=seed, early_stopping=True, validation_fraction=0.1)


def get_X_for(name, indices):
    """Return appropriate X slice for model type (scaled for MLP, raw for trees)."""
    if name == "mlp":
        return X_scaled[indices]
    return X_np[indices]


def generate_oof_predictions(model_names, train_idx, n_inner_folds=5):
    """
    Generate out-of-fold predictions on training data using inner CV.
    This is what retrain_and_upload.py does — each base model's predictions
    on the training set come from folds where that data was held out.
    """
    n_train = len(train_idx)
    oof_preds = {mn: np.zeros(n_train) for mn in model_names}

    # Inner time-series CV on training data only
    inner_tscv = TimeSeriesSplit(n_splits=n_inner_folds)
    inner_train_data = np.arange(n_train)

    for inner_train_rel, inner_val_rel in inner_tscv.split(inner_train_data):
        # Map relative indices back to absolute indices
        inner_train_abs = train_idx[inner_train_rel]
        inner_val_abs = train_idx[inner_val_rel]

        for mn in model_names:
            m = make_model(mn)
            m.fit(get_X_for(mn, inner_train_abs), y[inner_train_abs])
            oof_preds[mn][inner_val_rel] = m.predict(get_X_for(mn, inner_val_abs))

    # Only return rows that were predicted (inner TSCV skips earliest rows)
    # Find which rows have non-zero predictions (were in at least one val fold)
    predicted_mask = np.zeros(n_train, dtype=bool)
    for inner_train_rel, inner_val_rel in inner_tscv.split(inner_train_data):
        predicted_mask[inner_val_rel] = True

    return oof_preds, predicted_mask


# ── Test all combos ─────────────────────────────────────────────────
combos = {
    "XGB+CAT+LGBM+MLP": ["xgb", "cat", "lgb", "mlp"],
    "XGB+CAT+MLP":       ["xgb", "cat", "mlp"],
    "CAT+LGBM+MLP":      ["cat", "lgb", "mlp"],
    "CAT+MLP":            ["cat", "mlp"],
    "XGB+CAT+LGBM":      ["xgb", "cat", "lgb"],
    "XGB+CAT":            ["xgb", "cat"],
    "CAT+LGBM":           ["cat", "lgb"],
    "XGB+MLP":            ["xgb", "mlp"],
    "CAT only":           ["cat"],
    "MLP only":           ["mlp"],
}

# Outer CV
outer_tscv = TimeSeriesSplit(n_splits=5)  # 5 outer folds (each has inner 5-fold)
outer_splits = list(outer_tscv.split(X_np))

print(f"  Outer: 5-fold TSCV | Inner: 5-fold TSCV for OOF stacking")
print(f"  Testing {len(combos)} ensemble combinations...\n")
print(f"  {'Combo':<25} {'MAE':>8} {'±Std':>8} {'vs Best':>10} {'Weights':>40}")
print(f"  {'-'*95}")

results = []
best_mae = 999

for combo_name, model_names in combos.items():
    t0 = time.time()
    fold_maes = []
    all_weights = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_splits):

        if len(model_names) == 1:
            # Single model — no stacking needed
            mn = model_names[0]
            m = make_model(mn)
            m.fit(get_X_for(mn, train_idx), y[train_idx])
            final_preds = m.predict(get_X_for(mn, test_idx))
            all_weights.append([1.0])
        else:
            # Generate OOF predictions on training data for meta-learner
            oof_preds, predicted_mask = generate_oof_predictions(model_names, train_idx, n_inner_folds=5)

            # Build meta-learner training matrix (only rows with OOF predictions)
            S_train = np.column_stack([oof_preds[mn][predicted_mask] for mn in model_names])
            y_train_meta = y[train_idx[predicted_mask]]

            # Fit meta-learner on OOF predictions
            meta = Ridge(alpha=1.0)
            meta.fit(S_train, y_train_meta)

            # Train final base models on ALL training data
            final_models = {}
            for mn in model_names:
                m = make_model(mn)
                m.fit(get_X_for(mn, train_idx), y[train_idx])
                final_models[mn] = m

            # Get test predictions from final models
            S_test = np.column_stack([
                final_models[mn].predict(get_X_for(mn, test_idx))
                for mn in model_names
            ])

            # Stack
            final_preds = meta.predict(S_test)

            # Record weights
            w = meta.coef_
            w_sum = w.sum()
            w_norm = w / w_sum if w_sum > 0 else w
            all_weights.append(w_norm)

        mae = np.mean(np.abs(final_preds - y[test_idx]))
        fold_maes.append(mae)

    avg_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    avg_weights = np.mean(all_weights, axis=0)

    if avg_mae < best_mae:
        best_mae = avg_mae

    delta = avg_mae - best_mae
    delta_str = f"+{delta:.3f}" if delta > 0.001 else "BEST"

    weight_str = " ".join([f"{mn}={w:.3f}" for mn, w in zip(model_names, avg_weights)])
    elapsed = time.time() - t0

    print(f"  {combo_name:<25} {avg_mae:>7.3f}  {std_mae:>7.3f}  {delta_str:>9}  {weight_str}")
    results.append({
        "combo": combo_name,
        "mae": avg_mae,
        "std": std_mae,
        "weights": dict(zip(model_names, avg_weights)),
        "time": elapsed,
    })

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  RANKING (lower MAE = better):")
results.sort(key=lambda x: x["mae"])
for i, r in enumerate(results):
    marker = " ◀ WINNER" if i == 0 else ""
    delta = r["mae"] - results[0]["mae"]
    delta_str = f"+{delta:.3f}" if delta > 0.001 else ""
    print(f"  {i+1}. {r['combo']:<25} MAE: {r['mae']:.3f} (±{r['std']:.3f}) {delta_str}{marker}")

winner = results[0]
print(f"\n  Best: {winner['combo']} — MAE {winner['mae']:.3f}")
print(f"  Weights: {winner['weights']}")

# Check if stacking actually helps vs best single model
single_models = [r for r in results if len(r["weights"]) == 1]
best_single = min(single_models, key=lambda x: x["mae"]) if single_models else None
if best_single and winner["mae"] < best_single["mae"]:
    improvement = best_single["mae"] - winner["mae"]
    print(f"  Stacking benefit: {improvement:.3f} MAE improvement over best single ({best_single['combo']})")
elif best_single:
    print(f"  NOTE: Best single model ({best_single['combo']} {best_single['mae']:.3f}) matches or beats stacking")

print(f"\n  Done.")
