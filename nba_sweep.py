"""
NBA Full Sweep: Model combinations × estimator counts.
Tests all meaningful combos and finds the optimal configuration.
Run from sports-predictor-api root:
    python3 nba_sweep.py
"""
import time, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Import project modules
from sports.nba import nba_build_features, _nba_merge_historical
from ml_utils import HAS_XGB, _time_series_oof
from db import sb_get

if HAS_XGB:
    from xgboost import XGBRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Try loading dynamic league averages
try:
    from dynamic_constants import compute_nba_league_averages
    _nba_lg = compute_nba_league_averages()
    if _nba_lg:
        nba_build_features._league_averages = _nba_lg
        print(f"Using dynamic NBA averages ({len(_nba_lg)} stats)")
except:
    print("Using static NBA averages")

print("=" * 80)
print("NBA FULL SWEEP: Model Combinations × Estimator Counts")
print("=" * 80)

# ── Load data once ──
rows = sb_get("nba_predictions",
              "result_entered=eq.true&actual_home_score=not.is.null&select=*")
current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
df, sample_weights, n_historical = _nba_merge_historical(current_df)

X = nba_build_features(df)
y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n = len(df)
cv_folds = 10
fit_weights = sample_weights if sample_weights is not None else np.ones(n)

print(f"\nDataset: {n} rows, {len(X.columns)} features, {cv_folds}-fold CV")

# ── Define model builders ──
def make_models(combo, n_est):
    """Return dict of {name: model} for the given combo and estimator count."""
    models = {}
    if "GBM" in combo:
        models["GBM"] = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=20, random_state=42,
        )
    if "RF" in combo:
        models["RF"] = RandomForestRegressor(
            n_estimators=n_est, max_depth=6, min_samples_leaf=15,
            max_features=0.7, random_state=42, n_jobs=-1,
        )
    if "XGB" in combo and HAS_XGB:
        models["XGB"] = XGBRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=42, tree_method="hist", verbosity=0,
        )
    if "CAT" in combo and HAS_CAT:
        models["CAT"] = CatBoostRegressor(
            iterations=n_est, depth=4, learning_rate=0.06,
            subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0,
        )
    if "LGB" in combo and HAS_LGB:
        models["LGB"] = LGBMRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, verbosity=-1,
        )
    return models


def run_config(combo, n_est):
    """Train stacking ensemble and return OOF stacked MAE."""
    t0 = time.time()
    models = make_models(combo, n_est)
    model_names = list(models.keys())

    if len(models) == 1:
        # Single model — no stacking, just OOF MAE
        name = model_names[0]
        oof = _time_series_oof(
            {name.lower(): models[name]}, X_scaled, y_margin, df,
            n_splits=cv_folds, weights=fit_weights
        )
        mae = float(np.mean(np.abs(oof[name.lower()] - y_margin.values)))
        elapsed = time.time() - t0
        return mae, {name: 1.0}, elapsed

    # Multi-model stacking
    reg_dict = {k.lower(): v for k, v in models.items()}
    oof = _time_series_oof(reg_dict, X_scaled, y_margin, df,
                           n_splits=cv_folds, weights=fit_weights)

    # Fit all models on full data (not needed for MAE, but needed for weights)
    for name, model in models.items():
        if name == "GBM":
            model.fit(X_scaled, y_margin, sample_weight=fit_weights)
        elif name == "RF":
            model.fit(X_scaled, y_margin, sample_weight=fit_weights)
        elif name == "XGB":
            model.fit(X_scaled, y_margin, sample_weight=fit_weights)
        elif name == "CAT":
            model.fit(X_scaled, y_margin, sample_weight=fit_weights)
        elif name == "LGB":
            model.fit(X_scaled, y_margin, sample_weight=fit_weights)

    oof_cols = [oof[k.lower()] for k in model_names]
    meta_X = np.column_stack(oof_cols)

    meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta_reg.fit(meta_X, y_margin)

    oof_pred = meta_reg.predict(meta_X)
    mae = float(np.mean(np.abs(oof_pred - y_margin.values)))

    weights = {name: round(float(w), 4) for name, w in zip(model_names, meta_reg.coef_)}
    elapsed = time.time() - t0
    return mae, weights, elapsed


# ── Phase 1: Model combination search at 100 estimators ──
print("\n" + "=" * 80)
print("PHASE 1: Model Combinations @ 100 estimators")
print("=" * 80)

combos = [
    # Singles
    ["RF"],
    ["XGB"],
    ["CAT"],
    ["GBM"],
    # Pairs
    ["XGB", "RF"],
    ["CAT", "RF"],
    ["XGB", "CAT"],
    ["GBM", "RF"],
    # Triples
    ["XGB", "CAT", "RF"],
    ["XGB", "GBM", "RF"],
    ["CAT", "GBM", "RF"],
    ["XGB", "CAT", "GBM"],
    # Quads
    ["XGB", "CAT", "GBM", "RF"],
    # Full
    ["XGB", "LGB", "CAT", "GBM", "RF"],
]

print(f"\n{'Combo':<30} | {'MAE':>8} | {'Weights':<50} | {'Time':>6}")
print("-" * 105)

phase1_results = []
for combo in combos:
    label = "+".join(combo)
    mae, weights, elapsed = run_config(combo, 100)
    w_str = ", ".join(f"{k}:{v:+.3f}" for k, v in weights.items())
    print(f"{label:<30} | {mae:>8.3f} | {w_str:<50} | {elapsed:>5.1f}s")
    phase1_results.append({"combo": combo, "label": label, "mae": mae, "weights": weights})

phase1_results.sort(key=lambda r: r["mae"])
print("-" * 105)
print(f"\nTop 3 combos:")
for i, r in enumerate(phase1_results[:3]):
    w_str = ", ".join(f"{k}:{v:+.3f}" for k, v in r["weights"].items())
    print(f"  {i+1}. {r['label']:<30} MAE={r['mae']:.3f}  [{w_str}]")

# ── Phase 2: Estimator sweep on top 3 combos ──
top_combos = [r["combo"] for r in phase1_results[:3]]

print("\n" + "=" * 80)
print("PHASE 2: Estimator Sweep on Top 3 Combos (50 → 200, step 25)")
print("=" * 80)

est_range = list(range(50, 225, 25))
phase2_results = []

for combo in top_combos:
    label = "+".join(combo)
    print(f"\n--- {label} ---")
    print(f"  {'Est':>5} | {'MAE':>8} | {'Time':>6}")
    print(f"  " + "-" * 30)
    best_mae = 999
    best_est = 0
    for n_est in est_range:
        mae, weights, elapsed = run_config(combo, n_est)
        marker = " ★" if mae < best_mae else ""
        if mae < best_mae:
            best_mae = mae
            best_est = n_est
        print(f"  {n_est:>5} | {mae:>8.3f} | {elapsed:>5.1f}s{marker}")
        phase2_results.append({"combo": label, "n_est": n_est, "mae": mae, "weights": weights})
    print(f"  Best: {best_est} est → MAE {best_mae:.3f}")

# ── Final summary ──
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
overall_best = min(phase2_results, key=lambda r: r["mae"])
print(f"\n★ OVERALL BEST: {overall_best['combo']} @ {overall_best['n_est']} estimators")
print(f"  MAE: {overall_best['mae']:.3f}")
w_str = ", ".join(f"{k}:{v:+.4f}" for k, v in overall_best["weights"].items())
print(f"  Weights: {w_str}")
