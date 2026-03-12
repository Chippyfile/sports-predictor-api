#!/usr/bin/env python3
"""Test enhanced features (4 legit, no player leakage) vs pruned at winning config."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from ultimate_sweep import load_sport_data, run_config, add_enhanced_features

print("Loading data...")
df, X, y_margin, market_spread, weights, feature_groups, sigma = load_sport_data('ncaa', min_quality=0.8)

# Pruned: drop zero-var/weak
keep = [c for c in X.columns if X[c].std() > 1e-10]
X_pruned = X[keep]

# Enhanced (safe only): add 4 legit features, skip 7 player-derived ones
X_enhanced_full = add_enhanced_features(df, X, 'ncaa')

# Identify the 7 leaking player columns
leaking_cols = [
    'star1_dep_diff', 'top3_dep_diff', 'bench_depth_diff',
    'bench_pts_diff', 'rotation_diff', 'minutes_hhi_diff', 'star_x_spread'
]
safe_enhanced_cols = [c for c in X_enhanced_full.columns if c not in leaking_cols]
X_enhanced_safe = X_enhanced_full[safe_enhanced_cols]

# Enhanced safe + pruned (drop zero-var/weak from the safe enhanced set)
keep_enhanced = [c for c in X_enhanced_safe.columns if X_enhanced_safe[c].std() > 1e-10]
X_enhanced_pruned = X_enhanced_safe[keep_enhanced]

print(f"Baseline:        {len(X.columns)} features")
print(f"Pruned:          {len(X_pruned.columns)} features")
print(f"Enhanced (safe): {len(X_enhanced_safe.columns)} features (+{len(X_enhanced_safe.columns) - len(X.columns)} new)")
print(f"Enhanced+Pruned: {len(X_enhanced_pruned.columns)} features")
print()

configs = [
    ("Pruned", X_pruned),
    ("Enhanced (safe)", X_enhanced_safe),
    ("Enhanced+Pruned", X_enhanced_pruned),
    ("Baseline", X),
]

combo = ['XGB', 'CAT', 'LGBM']
print(f"XGB+CAT+LGBM, e=175, d=7, lr=0.1, f=50, ridge")
print(f"{'Feature Set':<20} {'Cols':>5} {'MAE':>8} {'ATS':>8}")
print("-" * 45)

for name, Xf in configs:
    r = run_config(Xf, y_margin, y_margin, market_spread, df, weights,
                   combo, 175, 7, 0.1, 50, 'ridge', sigma)
    if r and 'mae' in r:
        ats = f"{r['ats']*100:.1f}%" if r.get('ats') and not np.isnan(r.get('ats', float('nan'))) else 'N/A'
        print(f"  {name:<18} {len(Xf.columns):>5} {r['mae']:>8.3f} {ats:>8}")
