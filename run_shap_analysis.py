"""
run_shap_analysis.py — SHAP feature importance using local model + cached data.
No Supabase hits required.
"""
import sys, os
sys.path.insert(0, '.')

import numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from sklearn.preprocessing import StandardScaler
from dump_training_data import load_cached
import shap

MODEL_PATH = "ncaa_model_local.pkl"

print("=" * 70)
print("  SHAP ANALYSIS")
print("=" * 70)

# ── Load model ──
print(f"\n  Loading model from {MODEL_PATH}...")
bundle = joblib.load(MODEL_PATH)
feature_cols = bundle["feature_cols"]
print(f"  Features: {len(feature_cols)}")
print(f"  Model: {bundle.get('model_type', 'unknown')}")
print(f"  Trained: {bundle.get('trained_at', 'unknown')}")
print(f"  MAE: {bundle.get('mae_cv', 'unknown')}")

# ── Load data from parquet cache ──
print(f"\n  Loading data from parquet cache...")
df = load_cached()
if df is None:
    print("  ❌ No local cache. Run: python3 dump_training_data.py")
    sys.exit(1)

# Filter to scored games
df = df[df["actual_home_score"].notna()].copy()

# ESPN odds fallback (same as retrain)
if "espn_spread" in df.columns:
    espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df.loc[fill, "market_spread_home"] = espn_s[fill]
if "espn_over_under" in df.columns:
    espn_ou = pd.to_numeric(df["espn_over_under"], errors="coerce")
    mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
    fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
    df.loc[fill_ou, "market_ou_total"] = espn_ou[fill_ou]

# Quality filter
_quality_cols = ["home_adj_em", "away_adj_em", "home_ppg", "away_ppg", "market_spread_home", "market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em", "away_adj_em", "market_spread_home", "market_ou_total"] else True) for c in _qcols})
_row_q = _qmat.mean(axis=1)
df = df.loc[_row_q >= 0.8].reset_index(drop=True)

# Column alignment
for col in ["actual_home_score", "actual_away_score", "home_adj_em", "away_adj_em",
            "home_adj_oe", "away_adj_oe", "home_adj_de", "away_adj_de",
            "home_ppg", "away_ppg", "home_opp_ppg", "away_opp_ppg",
            "home_tempo", "away_tempo", "home_rank", "away_rank", "season",
            "home_record_wins", "away_record_wins", "home_record_losses", "away_record_losses"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "home_record_wins" in df.columns and "home_wins" not in df.columns:
    df["home_wins"] = df["home_record_wins"]
if "away_record_wins" in df.columns and "away_wins" not in df.columns:
    df["away_wins"] = df["away_record_wins"]
if "home_record_losses" in df.columns and "home_losses" not in df.columns:
    df["home_losses"] = df["home_record_losses"]
if "away_record_losses" in df.columns and "away_losses" not in df.columns:
    df["away_losses"] = df["away_record_losses"]

# Referee profiles
try:
    import json
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
    print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
except FileNotFoundError:
    print("  referee_profiles.json not found - ref features zero")

# Backfill + build features
print("  Running heuristic backfill...")
df = _ncaa_backfill_heuristic(df)
df = df.dropna(subset=["actual_home_score", "actual_away_score"])

print("  Building features...")
X = ncaa_build_features(df)
print(f"  {len(X)} games × {X.shape[1]} features")

# Ensure feature alignment with model
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

# Scale
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ── SHAP (TreeExplainer on XGBoost base learner) ──
print(f"\n  Computing SHAP values on XGBoost base learner...")
xgb_model = bundle["reg"].base_learners[0]  # First model in stack = XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_s)

# ── Results ──
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
total_shap = mean_abs_shap.sum()
feature_importance = sorted(
    zip(feature_cols, mean_abs_shap),
    key=lambda x: x[1],
    reverse=True
)

print(f"\n{'='*70}")
print(f"  SHAP FEATURE IMPORTANCE — {len(feature_cols)} features")
print(f"{'='*70}")
print(f"  {'Rank':>4}  {'Feature':40s}  {'|SHAP|':>8}  {'%':>6}  {'Cum%':>6}")
print(f"  {'─'*4}  {'─'*40}  {'─'*8}  {'─'*6}  {'─'*6}")

cum_pct = 0
for i, (feat, val) in enumerate(feature_importance):
    pct = val / total_shap * 100
    cum_pct += pct
    marker = " ★" if feat in [
        "eff_vol_diff", "pts_off_to_diff", "games7_diff", "fg_divergence_diff",
        "rhythm_disruption_diff", "def_improvement_diff", "dow_effect_diff",
        "conf_balance_diff", "espn_ml_edge",
        "fouls_diff", "run_vulnerability_diff", "close_win_rate_diff",
        "ft_pressure_diff", "margin_autocorr_diff", "blowout_asym_diff",
        "sos_trajectory_diff", "anti_fragility_diff", "clutch_over_exp_diff",
        "is_revenge_game", "revenge_margin", "is_sandwich", "is_letdown",
        "def_rest_advantage", "luck_x_spread"
    ] else ""
    print(f"  {i+1:4d}  {feat:40s}  {val:8.4f}  {pct:5.1f}%  {cum_pct:5.1f}%{marker}")

# ── Summary of new features ──
new_features = [
    "eff_vol_diff", "pts_off_to_diff", "games7_diff", "fg_divergence_diff",
    "rhythm_disruption_diff", "def_improvement_diff", "dow_effect_diff",
    "conf_balance_diff", "espn_ml_edge",
    "fouls_diff", "run_vulnerability_diff", "close_win_rate_diff",
    "ft_pressure_diff", "margin_autocorr_diff", "blowout_asym_diff",
    "sos_trajectory_diff", "anti_fragility_diff", "clutch_over_exp_diff",
    "is_revenge_game", "revenge_margin", "is_sandwich", "is_letdown",
    "def_rest_advantage", "luck_x_spread"
]

print(f"\n{'='*70}")
print(f"  NEW FEATURES (v22) — ★ marked above")
print(f"{'='*70}")
new_shap = {f: v for f, v in feature_importance if f in new_features}
sorted_new = sorted(new_shap.items(), key=lambda x: x[1], reverse=True)
total_new_pct = sum(v / total_shap * 100 for _, v in sorted_new)
print(f"  Combined SHAP contribution: {total_new_pct:.1f}%\n")
for feat, val in sorted_new:
    pct = val / total_shap * 100
    rank = [i+1 for i, (f, _) in enumerate(feature_importance) if f == feat][0]
    print(f"    #{rank:3d}  {feat:35s}  {pct:5.2f}%")

# ── Save results ──
results = {
    "n_features": len(feature_cols),
    "n_games": len(X),
    "top_20": [(f, round(float(v), 4), round(v/total_shap*100, 2)) for f, v in feature_importance[:20]],
    "new_features": [(f, round(float(v), 4), round(v/total_shap*100, 2)) for f, v in sorted_new],
    "new_features_total_pct": round(total_new_pct, 2),
}

import json
with open("shap_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved to shap_results.json")
