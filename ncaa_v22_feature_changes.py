"""
═══════════════════════════════════════════════════════════════
  ncaa_build_features — 24 NEW FEATURES (131 → 155)
  3 edits to sports/ncaa.py
═══════════════════════════════════════════════════════════════

EDIT 1: Add revenge_margin to raw_cols (~line 184)
EDIT 2: Add 9 new diff computations (after line 625)
EDIT 3: Add 24 features to feature_cols list (after line 725)

Copy this file to sports-predictor-api for reference.
Apply edits to sports/ncaa.py before retrain.
"""

# ═══════════════════════════════════════════════════════════════
# EDIT 1: Add revenge_margin to raw_cols
# ═══════════════════════════════════════════════════════════════
# FIND this line (~line 184):
#     "is_revenge_game": 0, "is_sandwich": 0, "is_letdown": 0, "is_midweek": 0,
#
# REPLACE WITH:
#     "is_revenge_game": 0, "revenge_margin": 0.0, "is_sandwich": 0, "is_letdown": 0, "is_midweek": 0,


# ═══════════════════════════════════════════════════════════════
# EDIT 2: Add 9 new diff computations
# ═══════════════════════════════════════════════════════════════
# FIND this line (~line 625):
#     df["conf_balance_diff"] = df["home_conf_balance"] - df["away_conf_balance"]
#
# ADD AFTER IT (before the ESPN moneyline edge block):
"""
    # ── v22: Previously orphaned features — well-populated in Supabase ──
    df["fouls_diff"] = df["home_fouls"] - df["away_fouls"]
    df["run_vulnerability_diff"] = df["home_run_vulnerability"] - df["away_run_vulnerability"]
    df["close_win_rate_diff"] = df["home_close_win_rate"] - df["away_close_win_rate"]
    df["ft_pressure_diff"] = df["home_ft_pressure"] - df["away_ft_pressure"]
    df["margin_autocorr_diff"] = df["home_margin_autocorr"] - df["away_margin_autocorr"]
    df["blowout_asym_diff"] = df["home_blowout_asym"] - df["away_blowout_asym"]
    df["sos_trajectory_diff"] = df["home_sos_trajectory"] - df["away_sos_trajectory"]
    df["anti_fragility_diff"] = df["home_anti_fragility"] - df["away_anti_fragility"]
    df["clutch_over_exp_diff"] = df["home_clutch_over_exp"] - df["away_clutch_over_exp"]
"""


# ═══════════════════════════════════════════════════════════════
# EDIT 3: Add 24 features to feature_cols list
# ═══════════════════════════════════════════════════════════════
# FIND these lines (~line 724-725):
#         "spread_movement", "total_movement",
#     ]
#
# REPLACE WITH:
"""
        "spread_movement", "total_movement",
        # ═══ v22: Orphaned features — Category 1 (already computed, 9 features) ═══
        "eff_vol_diff",              # efficiency-volatility ratio diff (59,770 coverage)
        "pts_off_to_diff",           # points off turnovers diff (59,168)
        "games7_diff",               # games in last 7 days diff (52,837)
        "fg_divergence_diff",        # FG% hot/cold divergence (46,583)
        "rhythm_disruption_diff",    # rhythm disruption diff (59,742)
        "def_improvement_diff",      # defensive improvement trend (44,268)
        "dow_effect_diff",           # day-of-week effect diff (49,673)
        "conf_balance_diff",         # conference balance diff (59,771)
        "espn_ml_edge",              # ESPN moneyline implied prob edge
        # ═══ v22: Orphaned features — Category 2 (new diffs, 9 features) ═══
        "fouls_diff",                # personal fouls diff (59,742)
        "run_vulnerability_diff",    # run vulnerability diff (59,739)
        "close_win_rate_diff",       # close game win rate diff (58,077)
        "ft_pressure_diff",          # FT pressure situations diff (54,589)
        "margin_autocorr_diff",      # margin autocorrelation diff (52,359)
        "blowout_asym_diff",         # blowout asymmetry diff (51,834)
        "sos_trajectory_diff",       # SOS trajectory diff (42,714)
        "anti_fragility_diff",       # anti-fragility diff (30,289)
        "clutch_over_exp_diff",      # clutch over-expectation diff (23,562)
        # ═══ v22: Orphaned features — Category 3 (situational flags, 6 features) ═══
        "is_revenge_game",           # rematch flag (17,514 games)
        "revenge_margin",            # prior loss margin in rematch
        "is_sandwich",               # sandwich scheduling spot (682)
        "is_letdown",                # letdown spot after big win (3,222)
        "def_rest_advantage",        # defensive rest edge (19,402)
        "luck_x_spread",             # luck × spread interaction (35,880)
    ]
"""


# ═══════════════════════════════════════════════════════════════
# ALSO UPDATE: predict_ncaa_game row_data (line ~1489)
# ═══════════════════════════════════════════════════════════════
# No changes needed — revenge_margin, is_revenge_game, is_sandwich,
# is_letdown, def_rest_advantage, luck_x_spread, and all the raw
# home/away columns are already in the prediction row_data dict.
# The new diff features will be computed automatically by
# ncaa_build_features when called on the single-row DataFrame.


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
# Edit 1: 1 line changed (add revenge_margin to raw_cols)
# Edit 2: 9 lines added (new diff computations)
# Edit 3: 24 features added to feature_cols list
# Total: 131 → 155 features
# Risk: LOW — all columns have 35-92% coverage, defaults handle rest
