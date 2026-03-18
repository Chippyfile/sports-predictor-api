"""
ADVANCED FEATURES — Add to ncaa_build_features() in sports/ncaa.py
===================================================================
Paste this block BEFORE the feature_cols list definition.
These use pre-computed columns from:
  - compute_player_impact.py (player ratings)
  - compute_advanced_features.py (h2h, conference, pace-adj, recency)
  - compute_lineup_features.py (lineup stability)

New features (up to 13):
  Player impact:
    - player_rating_diff: RAPM sum of starters (home - away)
    - weakest_starter_diff: weakest link comparison
    - starter_balance_diff: star-dependent vs balanced lineup

  Head-to-head:
    - h2h_margin_avg: historical matchup margin
    - h2h_games: number of prior meetings (familiarity signal)

  Conference:
    - conf_strength_diff: aggregate conference quality gap
    - cross_conf_flag: inter-conference game flag

  Pace-adjusted:
    - pace_adj_ppg_diff: PPG normalized to 70 possessions
    - pace_adj_opp_ppg_diff: defensive PPG normalized

  Recency:
    - recent_form_diff: last 5 games win rate diff
    - scoring_trend_diff: scoring trajectory diff

  Lineup (from compute_lineup_features.py):
    - lineup_changes_diff
    - lineup_stability_diff  
    - starter_experience_diff
"""

# ── Player impact ratings (RAPM-lite from compute_player_impact.py) ──
_h_pr = pd.to_numeric(df.get("home_player_rating_sum", 0), errors="coerce").fillna(0)
_a_pr = pd.to_numeric(df.get("away_player_rating_sum", 0), errors="coerce").fillna(0)
df["player_rating_diff"] = _h_pr - _a_pr

_h_ws = pd.to_numeric(df.get("home_weakest_starter", 0), errors="coerce").fillna(0)
_a_ws = pd.to_numeric(df.get("away_weakest_starter", 0), errors="coerce").fillna(0)
df["weakest_starter_diff"] = _h_ws - _a_ws

_h_sv = pd.to_numeric(df.get("home_starter_variance", 0), errors="coerce").fillna(0)
_a_sv = pd.to_numeric(df.get("away_starter_variance", 0), errors="coerce").fillna(0)
df["starter_balance_diff"] = _h_sv - _a_sv

# ── Head-to-head matchup history ──
df["h2h_margin_avg"] = pd.to_numeric(df.get("h2h_margin_avg", 0), errors="coerce").fillna(0)
df["h2h_games"] = pd.to_numeric(df.get("h2h_games", 0), errors="coerce").fillna(0)

# ── Conference strength ──
df["conf_strength_diff"] = pd.to_numeric(df.get("conf_strength_diff", 0), errors="coerce").fillna(0)
df["cross_conf_flag"] = pd.to_numeric(df.get("cross_conf_flag", 0), errors="coerce").fillna(0)

# ── Pace-adjusted stats ──
_h_tempo_raw = pd.to_numeric(df.get("home_tempo", 70), errors="coerce").fillna(70)
_a_tempo_raw = pd.to_numeric(df.get("away_tempo", 70), errors="coerce").fillna(70)
_h_ppg_raw = pd.to_numeric(df.get("home_ppg", 0), errors="coerce").fillna(0)
_a_ppg_raw = pd.to_numeric(df.get("away_ppg", 0), errors="coerce").fillna(0)
_h_opp_raw = pd.to_numeric(df.get("home_opp_ppg", 0), errors="coerce").fillna(0)
_a_opp_raw = pd.to_numeric(df.get("away_opp_ppg", 0), errors="coerce").fillna(0)
_STD_PACE = 70.0
df["pace_adj_ppg_diff"] = np.where(_h_tempo_raw > 0, _h_ppg_raw * _STD_PACE / _h_tempo_raw, _h_ppg_raw) - \
                           np.where(_a_tempo_raw > 0, _a_ppg_raw * _STD_PACE / _a_tempo_raw, _a_ppg_raw)
df["pace_adj_opp_ppg_diff"] = np.where(_h_tempo_raw > 0, _h_opp_raw * _STD_PACE / _h_tempo_raw, _h_opp_raw) - \
                               np.where(_a_tempo_raw > 0, _a_opp_raw * _STD_PACE / _a_tempo_raw, _a_opp_raw)

# ── Recency (last 5 games) ──
df["recent_form_diff"] = pd.to_numeric(df.get("recent_form_diff", 0), errors="coerce").fillna(0)
df["scoring_trend_diff"] = pd.to_numeric(df.get("scoring_trend_diff", 0), errors="coerce").fillna(0)


# ═══ Add to feature_cols list: ═══
# 
# # ═══ v24: Advanced features ═══
# "player_rating_diff",      # RAPM starter quality sum diff
# "weakest_starter_diff",    # weakest link in lineup diff
# "starter_balance_diff",    # star-dependent vs balanced diff
# "h2h_margin_avg",          # historical matchup margin
# "h2h_games",               # number of prior meetings
# "conf_strength_diff",      # conference quality gap (overwrites existing if present)
# "cross_conf_flag",         # inter-conference game
# "pace_adj_ppg_diff",       # pace-normalized offensive diff
# "pace_adj_opp_ppg_diff",   # pace-normalized defensive diff
# "recent_form_diff",        # last 5 games win rate diff
# "scoring_trend_diff",      # scoring trajectory diff
# "lineup_changes_diff",     # starters changed from last game
# "lineup_stability_diff",   # rolling 5-game lineup consistency
# "starter_experience_diff", # games this exact 5 started together
