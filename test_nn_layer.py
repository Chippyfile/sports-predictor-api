#!/usr/bin/env python3
"""Test adding MLP (Neural Network) to XGB+CAT+LGBM stack."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.')

from sports.ncaa import ncaa_build_features, _ncaa_merge_historical
from db import sb_get
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from ml_utils import _time_series_oof

# Load ref profiles
try:
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
    print(f"Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
except FileNotFoundError:
    print("No referee profiles")

# Load data
print("Loading data...")
rows = sb_get("ncaa_predictions", "result_entered=eq.true&actual_home_score=not.is.null&select=*")
current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
df, weights, n_hist = _ncaa_merge_historical(current_df)
X = ncaa_build_features(df)
y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)

# Quality filter (strict — match 8.905 run)
quality_cols = ["home_adj_em", "away_adj_em", "home_ppg", "away_ppg",
                "market_spread_home", "market_ou_total"]
qcols = [c for c in quality_cols if c in df.columns]
qmat = pd.DataFrame({c: df[c].notna() & (pd.to_numeric(df[c], errors="coerce") != 0) for c in qcols})
keep = qmat.mean(axis=1) >= 0.8
X = X.loc[keep].reset_index(drop=True)
y_margin = y_margin.loc[keep].reset_index(drop=True)
df = df.loc[keep].reset_index(drop=True)
w = weights[keep.values] if weights is not None else np.ones(int(keep.sum()))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n = len(X)
print(f"Dataset: {n} games, {X.shape[1]} features")

# Base learners (winning config)
xgb = XGBRegressor(n_estimators=175, max_depth=7, learning_rate=0.10,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    random_state=42, tree_method="hist", verbosity=0)
cat = CatBoostRegressor(iterations=175, depth=7, learning_rate=0.10,
                        subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0)
lgbm = LGBMRegressor(n_estimators=175, max_depth=7, learning_rate=0.10,
                      subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                      random_state=42, verbosity=-1)

# MLP configs to test
mlp_configs = [
    ("MLP-64", MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True,
                             validation_fraction=0.1, random_state=42)),
    ("MLP-128-64", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                                 validation_fraction=0.1, random_state=42)),
    ("MLP-256-128-64", MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True,
                                     validation_fraction=0.1, random_state=42)),
]

# First: baseline without NN
print("\n=== BASELINE: XGB+CAT+LGBM (no NN) ===")
reg_models = {"xgb": xgb, "cat": cat, "lgbm": lgbm}
oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=50, weights=w)

for name, model in reg_models.items():
    model.fit(X_scaled, y_margin, sample_weight=w)

oof_cols = [oof["xgb"], oof["cat"], oof["lgbm"]]
meta_X = np.column_stack(oof_cols)
meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
meta_reg.fit(meta_X, y_margin)
baseline_mae = mean_absolute_error(y_margin, meta_reg.predict(meta_X))
print(f"  Stacked MAE: {baseline_mae:.3f}")
print(f"  Weights: XGB={meta_reg.coef_[0]:.3f} CAT={meta_reg.coef_[1]:.3f} LGBM={meta_reg.coef_[2]:.3f}")

# Test each MLP config added to the stack
print("\n=== WITH NN ===")
print(f"{'Config':<20} {'MAE':>8} {'Delta':>8} {'XGB':>6} {'CAT':>6} {'LGBM':>6} {'MLP':>6}")
print("-" * 62)

for mlp_name, mlp_model in mlp_configs:
    # OOF for MLP (manual time-series split since _time_series_oof expects dict)
    oof_mlp = np.zeros(n)
    fold_size = n // 51
    for i in range(50):
        tr_end = fold_size * (i + 2)
        val_start = tr_end
        val_end = min(tr_end + fold_size, n)
        if val_start >= n:
            break
        import copy
        m = copy.deepcopy(mlp_model)
        m.fit(X_scaled[:tr_end], y_margin.values[:tr_end])
        oof_mlp[val_start:val_end] = m.predict(X_scaled[val_start:val_end])

    # Fit on full data
    mlp_full = copy.deepcopy(mlp_model)
    mlp_full.fit(X_scaled, y_margin.values)

    # Stack with MLP
    meta_X_nn = np.column_stack([oof["xgb"], oof["cat"], oof["lgbm"], oof_mlp])
    meta_nn = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta_nn.fit(meta_X_nn, y_margin)
    nn_mae = mean_absolute_error(y_margin, meta_nn.predict(meta_X_nn))
    delta = nn_mae - baseline_mae
    w_xgb, w_cat, w_lgbm, w_mlp = meta_nn.coef_

    print(f"  {mlp_name:<18} {nn_mae:>8.3f} {delta:>+8.3f} {w_xgb:>6.3f} {w_cat:>6.3f} {w_lgbm:>6.3f} {w_mlp:>6.3f}")

print(f"\n  Baseline (no NN): {baseline_mae:.3f}")
