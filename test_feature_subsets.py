#!/usr/bin/env python3
"""Test feature subsets (top 50, 75, 90% signal) with refs properly loaded."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.')

from sports.ncaa import ncaa_build_features, _ncaa_merge_historical
from db import sb_get
from ultimate_sweep import run_config
from sklearn.preprocessing import StandardScaler

# Load ref profiles BEFORE building features
try:
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
    print(f"Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
except FileNotFoundError:
    print("No referee profiles found")

# Load data
print("Loading data...")
rows = sb_get("ncaa_predictions", "result_entered=eq.true&actual_home_score=not.is.null&select=*")
current_df = pd.DataFrame(rows) if rows else pd.DataFrame()
df, weights, n_hist = _ncaa_merge_historical(current_df)

# Build features (refs will be populated now)
X = ncaa_build_features(df)
y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
market_spread = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0)
w = weights if weights is not None else np.ones(len(df))

print(f"Dataset: {len(df)} games, {len(X.columns)} features")

# Check ref features are real
ref_cols = ["ref_ou_bias", "ref_home_whistle", "ref_foul_rate", "ref_pace_impact", "has_ref_data"]
for col in ref_cols:
    if col in X.columns:
        nonzero = (X[col] != 0).sum()
        print(f"  {col}: {nonzero}/{len(X)} non-zero ({nonzero/len(X)*100:.1f}%)")

# Quality filter
quality_cols = ["home_adj_em", "away_adj_em", "home_ppg", "away_ppg",
                "market_spread_home", "market_ou_total"]
qcols = [c for c in quality_cols if c in df.columns]
if qcols:
    qmat = pd.DataFrame({c: df[c].notna() & (pd.to_numeric(df[c], errors="coerce") != 0) for c in qcols})
    row_q = qmat.mean(axis=1)
    keep = row_q >= 0.8
    X = X.loc[keep].reset_index(drop=True)
    y_margin = y_margin.loc[keep].reset_index(drop=True)
    market_spread = market_spread.loc[keep].reset_index(drop=True)
    df = df.loc[keep].reset_index(drop=True)
    if isinstance(w, np.ndarray):
        w = w[keep.values]
    print(f"Quality filter: {int(keep.sum())} games kept")

# SHAP-ranked features (from your results, top to bottom)
shap_ranked = [
    "neutral_em_diff", "elo_diff", "mkt_spread", "pit_sos_diff", "crowd_pct",
    "roll_garbage_diff", "espn_wp_edge", "orb_pct_diff", "ato_diff", "season_phase",
    "hca_pts", "drb_pct_diff", "mkt_total", "blocks_diff", "ppp_diff",
    "paint_pts_diff", "threepct_diff", "mkt_spread_vs_model", "luck_diff", "twopt_diff",
    "matchup_orb", "pace_adj_margin_diff", "tempo_avg", "fatigue_x_quality", "fatigue_diff",
    "ft_dependency_diff", "three_rate_diff", "roll_dominance_diff", "common_opp_diff",
    "scoring_source_entropy_diff", "matchup_efg", "pace_leverage", "recovery_diff",
    "assist_rate_diff", "pace_control_diff", "opp_to_rate_diff", "margin_accel_diff",
    "season_pct_avg", "venue_advantage", "matchup_ft", "eff_vol_diff", "def_stability_diff",
    "to_conversion_diff", "steals_diff", "form_x_familiarity", "rest_x_defense",
    "streak_diff", "opp_suppression_diff", "ppg_diff", "fgpct_diff",
    # 50 features = 75% signal
    "roll_lead_change_avg", "matchup_to", "three_value_diff", "block_foul_diff",
    "steal_foul_diff", "consistency_diff", "opp_adj_form_diff", "roll_top3_share_diff",
    "to_margin_diff", "floor_diff", "roll_run_diff", "margin_skew_diff", "rest_diff",
    "def_fgpct_diff", "ceiling_diff", "roll_ats_margin_gated", "bimodal_diff",
    "roll_drought_diff", "fta_rate_diff", "roll_bench_share_diff", "margin_trend_diff",
    "def_versatility_diff", "rhythm_disruption_diff", "scoring_entropy_diff",
    "opp_ppg_diff", "opp_fta_rate_diff", "regression_diff", "concentration_diff",
    "pts_off_to_diff",
    # 79 features = 90% signal
    "form_diff", "rank_diff", "roll_bench_pts_diff", "info_gain_diff", "opp_orb_pct_diff",
    "three_divergence_diff", "opp_efg_diff", "ppp_divergence_diff", "roll_ats_diff_gated",
    "steals_to_diff", "fastbreak_diff", "roll_star1_share_diff",
    # 91 features = 95% signal
    "transition_dep_diff", "neutral", "wl_momentum_diff", "overreaction_diff",
    "ts_diff", "fg_divergence_diff", "efg_diff", "sos_diff", "def_improvement_diff",
    "roll_top3_dep_diff", "has_ats_data", "win_pct_diff", "is_midweek",
    "roll_star_dep_diff", "roll_bench_diff",
    # Remaining + ref features
    "def_rest_advantage", "has_mkt", "games7_diff", "is_early",
    "roll_clutch_ft_diff", "luck_x_spread", "style_familiarity",
    "consistency_x_spread", "is_revenge_game", "is_conf_game",
    "after_loss_either", "is_sandwich", "either_b2b", "is_top_matchup",
    "is_ranked_game", "is_letdown", "spread_regime",
    # Ref features
    "ref_ou_bias", "ref_home_whistle", "ref_foul_rate",
    "ref_pace_impact", "has_ref_data",
]

# Filter to features that exist in X
shap_ranked = [f for f in shap_ranked if f in X.columns]

# Feature subsets to test
subsets = {
    "Top 17 (50%)": shap_ranked[:17],
    "Top 50 (75%)": shap_ranked[:50],
    "Top 79 (90%)": shap_ranked[:79],
    "Top 91 (95%)": shap_ranked[:91],
    "All 141": list(X.columns),
    "All + refs loaded": list(X.columns),  # same features but refs are now real
}

combo = ['XGB', 'CAT', 'LGBM']
print(f"\nXGB+CAT+LGBM, e=175, d=7, lr=0.1, f=50, ridge")
print(f"{'Subset':<25} {'Cols':>5} {'MAE':>8} {'ATS':>8}")
print("-" * 50)

for name, features in subsets.items():
    X_sub = X[features]
    r = run_config(X_sub, y_margin, y_margin, market_spread, df, w,
                   combo, 175, 7, 0.1, 50, 'ridge', 11.0)
    if r and 'mae' in r:
        ats = f"{r['ats']*100:.1f}%" if r.get('ats') and not np.isnan(r.get('ats', float('nan'))) else 'N/A'
        print(f"  {name:<23} {len(features):>5} {r['mae']:>8.3f} {ats:>8}")
