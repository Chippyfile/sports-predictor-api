"""
NCAA estimator sweep: XGB+CAT+GBM+RF from 100 to 175 in steps of 5.
Run from sports-predictor-api root:
    python3 ncaa_sweep.py
"""
import importlib, time, sys
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Import project modules
import sports.ncaa as ncaa_mod
from sports.ncaa import ncaa_build_features, _ncaa_merge_historical, _ncaa_backfill_heuristic
from ml_utils import HAS_XGB, _time_series_oof, StackedRegressor
from db import sb_get

if HAS_XGB:
    from xgboost import XGBRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

print("=" * 70)
print("NCAA ESTIMATOR SWEEP: XGB+CAT+GBM+RF, 100 → 175 (step 5)")
print("=" * 70)

# ── Load data once ──
rows = sb_get("ncaa_predictions",
              "result_entered=eq.true&actual_home_score=not.is.null&select=*")
current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
df, sample_weights, n_historical = _ncaa_merge_historical(current_df)

X = ncaa_build_features(df)
y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n = len(df)
cv_folds = 10
fit_weights = sample_weights if sample_weights is not None else np.ones(n)

print(f"\nDataset: {n} rows, {len(X.columns)} features, {cv_folds}-fold CV")
print(f"{'Est':>5} | {'MAE':>8} | {'GBM':>7} {'RF':>7} {'XGB':>7} {'CAT':>7} | {'Time':>6}")
print("-" * 70)

results = []

for n_est in range(100, 180, 5):
    t0 = time.time()

    gbm = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=4,
        learning_rate=0.06, subsample=0.8,
        min_samples_leaf=20, random_state=42,
    )
    rf_reg = RandomForestRegressor(
        n_estimators=n_est, max_depth=6,
        min_samples_leaf=15, max_features=0.7,
        random_state=42, n_jobs=-1,
    )

    reg_models = {"gbm": gbm, "rf": rf_reg}

    if HAS_XGB:
        xgb_reg = XGBRegressor(
            n_estimators=n_est, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=42, tree_method="hist", verbosity=0,
        )
        reg_models["xgb"] = xgb_reg

    if HAS_CAT:
        cat_reg = CatBoostRegressor(
            iterations=n_est, depth=4, learning_rate=0.06,
            subsample=0.8, min_data_in_leaf=20,
            random_seed=42, verbose=0,
        )
        reg_models["cat"] = cat_reg

    # OOF predictions
    oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=fit_weights)

    # Full fits
    gbm.fit(X_scaled, y_margin, sample_weight=fit_weights)
    rf_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
    if HAS_XGB:
        xgb_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
    if HAS_CAT:
        cat_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)

    # Stack
    oof_cols = [oof["gbm"], oof["rf"]]
    if HAS_XGB: oof_cols.append(oof["xgb"])
    if HAS_CAT: oof_cols.append(oof["cat"])
    meta_X = np.column_stack(oof_cols)

    meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta_reg.fit(meta_X, y_margin)

    # MAE from OOF stacked predictions
    oof_pred = meta_reg.predict(meta_X)
    mae = float(np.mean(np.abs(oof_pred - y_margin.values)))

    weights = meta_reg.coef_.round(4)
    elapsed = time.time() - t0

    # weights order: GBM, RF, XGB, CAT
    w_gbm, w_rf = weights[0], weights[1]
    w_xgb = weights[2] if HAS_XGB else 0
    w_cat = weights[3] if HAS_CAT else (weights[2] if not HAS_XGB else 0)

    print(f"{n_est:>5} | {mae:>8.3f} | {w_gbm:>+7.3f} {w_rf:>+7.3f} {w_xgb:>+7.3f} {w_cat:>+7.3f} | {elapsed:>5.1f}s")

    results.append({
        "n_est": n_est, "mae": mae,
        "w_gbm": float(w_gbm), "w_rf": float(w_rf),
        "w_xgb": float(w_xgb), "w_cat": float(w_cat),
        "duration": elapsed,
    })

print("-" * 70)
best = min(results, key=lambda r: r["mae"])
print(f"\n★ BEST: {best['n_est']} estimators → MAE {best['mae']:.3f}")
print(f"  Weights: GBM={best['w_gbm']:+.4f}, RF={best['w_rf']:+.4f}, "
      f"XGB={best['w_xgb']:+.4f}, CAT={best['w_cat']:+.4f}")
