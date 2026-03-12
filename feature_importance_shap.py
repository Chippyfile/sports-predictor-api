#!/usr/bin/env python3
"""Feature importance via SHAP — train once, rank all 132 features."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')

from ultimate_sweep import load_sport_data
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap

print("Loading data...")
df, X, y_margin, market_spread, weights, feature_groups, sigma = load_sport_data('ncaa', min_quality=0.8)
print(f"Dataset: {len(df)} games, {len(X.columns)} features\n")

# Train XGB at winning config (single model, fast)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training XGB (e=175, d=7, lr=0.1)...")
model = XGBRegressor(n_estimators=175, max_depth=7, learning_rate=0.10,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                     random_state=42, tree_method="hist", verbosity=0)
model.fit(X_scaled, y_margin)

# SHAP values (sample 5000 for speed)
print("Computing SHAP values (5000 sample)...")
sample_idx = np.random.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled[sample_idx])

# Mean absolute SHAP per feature
mean_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_shap,
    'std': X.values.std(axis=0),
}).sort_values('mean_abs_shap', ascending=False)

print(f"\n{'Rank':>4}  {'Feature':<35} {'SHAP':>8}  {'Std':>8}  {'Cumulative':>10}")
print("-" * 72)

total_shap = importance['mean_abs_shap'].sum()
cumulative = 0
for i, (_, row) in enumerate(importance.iterrows()):
    cumulative += row['mean_abs_shap']
    pct = cumulative / total_shap * 100
    bar = "█" * int(pct / 5)
    print(f"  {i+1:>3}  {row['feature']:<35} {row['mean_abs_shap']:>8.3f}  {row['std']:>8.3f}  {pct:>8.1f}% {bar}")

# Summary: how many features for 90%, 95%, 99%
print(f"\n  COVERAGE:")
cumsum = importance['mean_abs_shap'].cumsum() / total_shap
for threshold in [0.50, 0.75, 0.90, 0.95, 0.99]:
    n = int((cumsum <= threshold).sum()) + 1
    print(f"    {threshold*100:.0f}% of signal: top {n} features")

# Bottom features (candidates to drop)
print(f"\n  BOTTOM 20 (candidates to drop):")
bottom = importance.tail(20)
for _, row in bottom.iterrows():
    print(f"    {row['feature']:<35} SHAP={row['mean_abs_shap']:.4f}")
