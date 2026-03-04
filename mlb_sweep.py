"""
MLB Sweep: Verify RF-only vs stacking combos with true OOF MAE.
Run from sports-predictor-api root:
    python3 mlb_sweep.py
"""
import time
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from sports.mlb import mlb_build_features, _mlb_merge_historical
from ml_utils import HAS_XGB, _time_series_oof
from db import sb_get

if HAS_XGB:
    from xgboost import XGBRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

print("=" * 80)
print("MLB SWEEP: RF-only vs Stacking Combos (true OOF MAE)")
print("=" * 80)

# ── Load data once ──
rows = sb_get("mlb_predictions",
              "result_entered=eq.true&actual_home_runs=not.is.null&game_type=eq.R&select=*")
current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
df, sample_weights = _mlb_merge_historical(current_df)

X = mlb_build_features(df)
y_margin = df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n = len(df)
cv_folds = 10
fit_weights = sample_weights if sample_weights is not None else np.ones(n)

print(f"\nDataset: {n} rows, {len(X.columns)} features, {cv_folds}-fold CV")


def run_config(combo, n_est):
    t0 = time.time()
    models = {}
    if "RF" in combo:
        models["rf"] = RandomForestRegressor(
            n_estimators=n_est, max_depth=6, min_samples_leaf=15,
            max_features=0.7, random_state=42, n_jobs=-1)
    if "XGB" in combo and HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=42, tree_method="hist", verbosity=0)
    if "CAT" in combo and HAS_CAT:
        models["cat"] = CatBoostRegressor(
            iterations=n_est, depth=4, learning_rate=0.06,
            subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0)
    if "GBM" in combo:
        models["gbm"] = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=20, random_state=42)

    oof = _time_series_oof(models, X_scaled, y_margin, df,
                           n_splits=cv_folds, weights=fit_weights)

    if len(models) == 1:
        name = list(models.keys())[0]
        mae = float(np.mean(np.abs(oof[name] - y_margin.values)))
        return mae, {name: 1.0}, time.time() - t0

    # Stack
    for name, model in models.items():
        model.fit(X_scaled, y_margin, sample_weight=fit_weights)

    oof_cols = [oof[k] for k in models.keys()]
    meta_X = np.column_stack(oof_cols)
    meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta_reg.fit(meta_X, y_margin)

    oof_pred = meta_reg.predict(meta_X)
    mae = float(np.mean(np.abs(oof_pred - y_margin.values)))
    weights = {k: round(float(w), 4) for k, w in zip(models.keys(), meta_reg.coef_)}
    return mae, weights, time.time() - t0


# ── Phase 1: Key combos at 85 est (MLB's current setting) ──
print(f"\n{'Combo':<25} | {'MAE':>8} | {'Weights':<45} | {'Time':>6}")
print("-" * 95)

combos = [
    ["RF"],
    ["CAT"],
    ["XGB"],
    ["CAT", "RF"],
    ["XGB", "RF"],
    ["XGB", "CAT"],
    ["XGB", "CAT", "RF"],
    ["XGB", "CAT", "GBM", "RF"],
]

phase1 = []
for combo in combos:
    label = "+".join(combo)
    mae, weights, elapsed = run_config(combo, 85)
    w_str = ", ".join(f"{k}:{v:+.3f}" for k, v in weights.items())
    print(f"{label:<25} | {mae:>8.3f} | {w_str:<45} | {elapsed:>5.1f}s")
    phase1.append({"combo": combo, "label": label, "mae": mae})

phase1.sort(key=lambda r: r["mae"])
print(f"\nTop 3:")
for i, r in enumerate(phase1[:3]):
    print(f"  {i+1}. {r['label']:<25} MAE={r['mae']:.3f}")

# ── Phase 2: Sweep top 3 from 50 to 150 step 25 ──
top = [r["combo"] for r in phase1[:3]]
print("\n" + "=" * 80)
print("PHASE 2: Estimator Sweep (50 → 150, step 25)")
print("=" * 80)

for combo in top:
    label = "+".join(combo)
    print(f"\n--- {label} ---")
    print(f"  {'Est':>5} | {'MAE':>8} | {'Time':>6}")
    print(f"  " + "-" * 30)
    best_mae, best_est = 999, 0
    for n_est in range(50, 175, 25):
        mae, weights, elapsed = run_config(combo, n_est)
        marker = " ★" if mae < best_mae else ""
        if mae < best_mae:
            best_mae, best_est = mae, n_est
        print(f"  {n_est:>5} | {mae:>8.3f} | {elapsed:>5.1f}s{marker}")
    print(f"  Best: {best_est} est → MAE {best_mae:.3f}")
