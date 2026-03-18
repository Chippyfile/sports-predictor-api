"""
NBA Feature Builder v22 — 120+ features (NCAA parity)
Requires enrich_nba_v2.py to have been run on the parquet first.

Features organized in same categories as NCAA:
  1. Core efficiency (5)
  2. Raw stat diffs (15)  
  3. Advanced shooting (10)
  4. Market/betting (5)
  5. Elo (2)
  6. Context (7)
  7. Momentum/form (20)
  8. Scoring distribution (14)
  9. Defense advanced (6)
  10. Rest/fatigue (7)
  11. Interactions (6)
  12. Matchup-specific (11)
  13. Referee (4) — zeros until ESPN scrape
  14. Rolling ATS (3)
  15. ESPN signal (2) — zeros until ESPN scrape
  16. Injury (2) — zeros for historical
  17. NBA-specific (3)
"""

import numpy as np
import pandas as pd

try:
    from dynamic_constants import NBA_DEFAULT_AVERAGES
except ImportError:
    NBA_DEFAULT_AVERAGES = {
        "ppg": 113, "opp_ppg": 113, "fgpct": 0.471, "threepct": 0.365,
        "ftpct": 0.780, "assists": 25, "turnovers": 14, "tempo": 99.5,
        "orb_pct": 0.245, "fta_rate": 0.270, "ato_ratio": 1.8,
        "steals": 7.5, "blocks": 5.0,
    }


def nba_build_features(df):
    df = df.copy()
    _avgs = getattr(nba_build_features, "_league_averages", NBA_DEFAULT_AVERAGES)

    def _col(name, default):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    # ══════════════════════════════════════════════════════════
    # 1. CORE EFFICIENCY
    # ══════════════════════════════════════════════════════════
    df["score_diff_pred"] = _col("pred_home_score", 110) - _col("pred_away_score", 110)
    df["total_pred"] = _col("pred_home_score", 110) + _col("pred_away_score", 110)
    df["home_fav"] = (_col("model_ml_home", 0) < 0).astype(int)
    df["win_pct_home"] = _col("win_pct_home", 0.5)
    df["net_rtg_diff"] = _col("home_net_rtg", 0) - _col("away_net_rtg", 0)

    # Offensive/Defensive efficiency diffs
    h_ppg, a_ppg = _col("home_ppg", 113), _col("away_ppg", 113)
    h_opp, a_opp = _col("home_opp_ppg", 113), _col("away_opp_ppg", 113)
    h_tempo, a_tempo = _col("home_tempo", 99.5), _col("away_tempo", 99.5)
    df["adj_oe_diff"] = (h_ppg / h_tempo.clip(85) * 100) - (a_ppg / a_tempo.clip(85) * 100)
    df["adj_de_diff"] = (h_opp / h_tempo.clip(85) * 100) - (a_opp / a_tempo.clip(85) * 100)

    # ══════════════════════════════════════════════════════════
    # 2. RAW STAT DIFFERENTIALS
    # ══════════════════════════════════════════════════════════
    stat_pairs = [
        ("ppg", 113), ("opp_ppg", 113), ("fgpct", 0.471), ("threepct", 0.365),
        ("ftpct", 0.780), ("assists", 25), ("turnovers", 14), ("tempo", 99.5),
        ("orb_pct", 0.245), ("fta_rate", 0.270), ("ato_ratio", 1.8),
        ("opp_fgpct", 0.471), ("opp_threepct", 0.365), ("steals", 7.5), ("blocks", 5.0),
    ]
    for base, default in stat_pairs:
        h, a = _col(f"home_{base}", default), _col(f"away_{base}", default)
        df[f"home_{base}"] = h
        df[f"away_{base}"] = a
        df[f"{base}_diff"] = h - a

    # Turnover quality
    df["to_margin_diff"] = df["away_turnovers"] - df["home_turnovers"]
    df["steals_to_diff"] = (df["home_steals"] / df["home_turnovers"].clip(0.5)) - \
                           (df["away_steals"] / df["away_turnovers"].clip(0.5))

    # ══════════════════════════════════════════════════════════
    # 3. ADVANCED SHOOTING
    # ══════════════════════════════════════════════════════════
    h_3rate = _col("home_three_att_rate", 0.40)
    a_3rate = _col("away_three_att_rate", 0.40)
    df["efg_diff"] = (df["home_fgpct"] + 0.5*h_3rate*df["home_threepct"]) - \
                     (df["away_fgpct"] + 0.5*a_3rate*df["away_threepct"])
    df["three_rate_diff"] = h_3rate - a_3rate
    # Points per possession
    df["ppp_diff"] = (h_ppg / h_tempo.clip(85)) - (a_ppg / a_tempo.clip(85))
    # FT dependency: FT points as % of total (FTM ≈ PPG × ftpct × fta_rate_approx)
    df["ft_dependency_diff"] = (df["home_ftpct"] * _col("home_fta_rate", 0.27)) - \
                               (df["away_ftpct"] * _col("away_fta_rate", 0.27))
    # 3PT value: 3P points contribution
    df["three_value_diff"] = (df["home_threepct"] * h_3rate * 1.5) - \
                             (df["away_threepct"] * a_3rate * 1.5)
    # DRB%
    h_drb = _col("home_def_reb", 34)
    a_drb = _col("away_def_reb", 34)
    df["drb_pct_diff"] = (h_drb / (h_drb + 10.5).clip(1)) - (a_drb / (a_drb + 10.5).clip(1))
    # Opponent Four Factors
    df["opp_efg_diff"] = (_col("home_opp_fgpct", 0.471) + 0.5*0.36*_col("home_opp_threepct", 0.365)) - \
                         (_col("away_opp_fgpct", 0.471) + 0.5*0.36*_col("away_opp_threepct", 0.365))
    # 3P divergence from league avg
    lg_3 = _avgs.get("threepct", 0.365)
    df["three_divergence_diff"] = (df["home_threepct"] - lg_3) - (df["away_threepct"] - lg_3)
    # PPP divergence
    lg_ppp = _avgs.get("ppg", 113) / _avgs.get("tempo", 99.5)
    df["ppp_divergence_diff"] = ((h_ppg/h_tempo.clip(85)) - lg_ppp) - ((a_ppg/a_tempo.clip(85)) - lg_ppp)

    # ══════════════════════════════════════════════════════════
    # 4. MARKET / BETTING
    # ══════════════════════════════════════════════════════════
    df["market_spread"] = _col("market_spread_home", 0)
    _mkt_ou = _col("market_ou_total", 0)
    _has_ou = (_mkt_ou != 0)
    df["market_total"] = np.where(_has_ou, _mkt_ou, 0.0)
    _has_spread = (df["market_spread"] != 0)
    df["has_market"] = (_has_spread | _has_ou).astype(int)
    df["spread_vs_market"] = np.where(df["has_market"] == 1, df["score_diff_pred"] - df["market_spread"], 0.0)
    df["ou_gap"] = np.where(_has_ou, df["total_pred"] - _mkt_ou, 0.0)
    # Spread regime: bucketed
    df["spread_regime"] = pd.cut(df["market_spread"].clip(-25, 25),
                                  bins=[-25, -10, -4, -1, 1, 4, 10, 25],
                                  labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(3)
    df["spread_x_market"] = df["market_spread"] * df["has_market"]

    # ══════════════════════════════════════════════════════════
    # 5. ELO
    # ══════════════════════════════════════════════════════════
    h_elo, a_elo = _col("home_elo", 1500), _col("away_elo", 1500)
    df["elo_diff"] = h_elo - a_elo
    df["elo_win_prob"] = 1.0 / (1.0 + 10.0 ** (-(df["elo_diff"] + 65) / 400.0))

    # ══════════════════════════════════════════════════════════
    # 6. CONTEXT
    # ══════════════════════════════════════════════════════════
    h_wins = _col("home_wins", 0); h_losses = _col("home_losses", 0)
    a_wins = _col("away_wins", 0); a_losses = _col("away_losses", 0)
    h_wpct = h_wins / (h_wins + h_losses).clip(1)
    a_wpct = a_wins / (a_wins + a_losses).clip(1)
    df["win_pct_diff"] = h_wpct - a_wpct
    df["form_diff"] = _col("home_form", 0) - _col("away_form", 0)
    df["tempo_avg"] = (h_tempo + a_tempo) / 2
    df["season_phase"] = _col("season_phase", 0.5)
    df["is_early_season"] = _col("is_early_season", 0)
    df["is_midweek"] = _col("is_midweek", 0)
    df["is_playoff"] = _col("is_playoff", 0)

    # ══════════════════════════════════════════════════════════
    # 7. MOMENTUM / FORM (20 features)
    # ══════════════════════════════════════════════════════════
    df["margin_trend_diff"] = _col("home_margin_trend", 0) - _col("away_margin_trend", 0)
    df["margin_accel_diff"] = _col("home_margin_accel", 0) - _col("away_margin_accel", 0)
    df["streak_diff"] = _col("home_streak", 0) - _col("away_streak", 0)
    df["days_since_loss_diff"] = _col("home_days_since_loss", 15) - _col("away_days_since_loss", 15)
    df["games_since_blowout_diff"] = _col("home_games_since_blowout", 10) - _col("away_games_since_blowout", 10)
    df["wl_momentum_diff"] = _col("home_wl_momentum", 0.5) - _col("away_wl_momentum", 0.5)
    df["momentum_halflife_diff"] = _col("home_momentum_halflife", 0) - _col("away_momentum_halflife", 0)
    df["win_aging_diff"] = _col("home_win_aging", 0.5) - _col("away_win_aging", 0.5)
    df["recovery_diff"] = _col("home_recovery", 5) - _col("away_recovery", 5)
    df["after_loss_either"] = _col("after_loss_either", 0)
    df["pyth_luck_diff"] = _col("home_pyth_luck", 0) - _col("away_pyth_luck", 0)
    df["pyth_residual_diff"] = _col("home_pyth_residual", 0) - _col("away_pyth_residual", 0)
    df["opp_adj_form_diff"] = _col("home_opp_adj_form", 0) - _col("away_opp_adj_form", 0)
    df["regression_diff"] = df["pyth_residual_diff"] * -1  # Teams with positive luck regress down
    df["games_last_14_diff"] = _col("home_games_last_14", 5) - _col("away_games_last_14", 5)

    # ══════════════════════════════════════════════════════════
    # 8. SCORING DISTRIBUTION (14 features)
    # ══════════════════════════════════════════════════════════
    df["scoring_var_diff"] = _col("home_scoring_var", 10) - _col("away_scoring_var", 10)
    df["score_kurtosis_diff"] = _col("home_score_kurtosis", 0) - _col("away_score_kurtosis", 0)
    df["margin_skew_diff"] = _col("home_margin_skew", 0) - _col("away_margin_skew", 0)
    df["ceiling_diff"] = _col("home_ceiling", 10) - _col("away_ceiling", 10)
    df["floor_diff"] = _col("home_floor", -10) - _col("away_floor", -10)
    df["scoring_entropy_diff"] = _col("home_scoring_entropy", 1.5) - _col("away_scoring_entropy", 1.5)
    df["bimodal_diff"] = _col("home_bimodal", 2) - _col("away_bimodal", 2)
    df["consistency_diff"] = -df["scoring_var_diff"]  # Lower variance = more consistent

    # ══════════════════════════════════════════════════════════
    # 9. DEFENSE ADVANCED (6 features)
    # ══════════════════════════════════════════════════════════
    df["def_stability_diff"] = _col("home_def_stability", 10) - _col("away_def_stability", 10)
    df["opp_suppression_diff"] = _col("home_opp_suppression", 0) - _col("away_opp_suppression", 0)
    lg_steals, lg_blocks = _avgs.get("steals", 7.5), _avgs.get("blocks", 5.0)
    df["steal_foul_diff"] = (df["home_steals"] / 20.0) - (df["away_steals"] / 20.0)
    df["block_foul_diff"] = (df["home_blocks"] / 20.0) - (df["away_blocks"] / 20.0)
    df["def_eff_diff"] = df["opp_ppg_diff"] * -1
    df["block_rate_diff"] = (df["home_blocks"] - lg_blocks) - (df["away_blocks"] - lg_blocks)

    # ══════════════════════════════════════════════════════════
    # 10. REST / FATIGUE (7 features)
    # ══════════════════════════════════════════════════════════
    h_rest = _col("home_days_rest", 2); a_rest = _col("away_days_rest", 2)
    df["rest_diff"] = h_rest - a_rest
    df["away_travel"] = _col("away_travel_dist", 0)
    df["home_b2b"] = (h_rest == 0).astype(int)
    df["away_b2b"] = (a_rest == 0).astype(int)
    df["b2b_diff"] = df["away_b2b"] - df["home_b2b"]
    # Composite fatigue: rest + travel + schedule density
    df["fatigue_diff"] = (a_rest * -1 + df["away_travel"] / 1000 + _col("away_games_last_14", 5) * 0.2) - \
                         (h_rest * -1 + 0 + _col("home_games_last_14", 5) * 0.2)
    df["season_pct_avg"] = (_col("home_wins", 20) + _col("home_losses", 20) + 
                            _col("away_wins", 20) + _col("away_losses", 20)) / 4 / 82

    # ══════════════════════════════════════════════════════════
    # 11. INTERACTIONS (6 features)
    # ══════════════════════════════════════════════════════════
    df["fatigue_x_quality"] = df["fatigue_diff"] * df["net_rtg_diff"].clip(-15, 15) / 15
    df["rest_x_defense"] = df["rest_diff"] * df["adj_de_diff"].clip(-10, 10) / 10
    df["form_x_familiarity"] = df["form_diff"] * (1.0 / (1.0 + (h_tempo - a_tempo).abs() / 5))
    df["consistency_x_spread"] = df["consistency_diff"] * df["spread_regime"] / 6
    # rest × travel
    df["rest_x_travel"] = np.where(a_rest == 0, df["away_travel"].clip(0, 3000) / 1000.0, 0.0)
    df["netrtg_x_sos"] = df["net_rtg_diff"] * (_col("home_sos", 0.5) - _col("away_sos", 0.5)).clip(-0.2, 0.2) * 5

    # ══════════════════════════════════════════════════════════
    # 12. MATCHUP-SPECIFIC (11 features)
    # ══════════════════════════════════════════════════════════
    # Four Factors matchups: team's strength vs opponent's weakness
    df["matchup_efg"] = df["efg_diff"] - df["opp_efg_diff"]
    df["matchup_to"] = df["to_margin_diff"] * df["steals_diff"]  # Steal-happy team vs turnover-prone team
    df["matchup_orb"] = df["orb_pct_diff"] + df["drb_pct_diff"]  # Rebounding advantage
    df["matchup_ft"] = df["fta_rate_diff"] * df["ftpct_diff"]  # FT rate × FT accuracy
    # Style familiarity: how similar are the teams' paces?
    df["style_familiarity"] = 1.0 / (1.0 + (h_tempo - a_tempo).abs() / 3)
    # Pace leverage: faster team benefits from pace mismatch
    pace_diff = h_tempo - a_tempo
    df["pace_leverage"] = pace_diff * df["net_rtg_diff"].clip(-10, 10) / 10
    df["pace_control_diff"] = pace_diff / 5.0
    # Common opponents
    df["common_opp_diff"] = _col("common_opp_diff", 0)
    df["n_common_opps"] = _col("n_common_opps", 0)
    # Venue advantage
    df["venue_advantage"] = _col("altitude_factor", 0)
    df["timezone_diff"] = _col("timezone_diff", 0)

    # ══════════════════════════════════════════════════════════
    # 13. REFEREE (4 features — zeros until ESPN scrape)
    # ══════════════════════════════════════════════════════════
    df["ref_home_whistle"] = _col("ref_home_whistle", 0)
    df["ref_ou_bias"] = _col("ref_ou_bias", 0)
    df["ref_foul_rate"] = _col("ref_foul_rate", 0)
    df["ref_pace_impact"] = _col("ref_pace_impact", 0)

    # ══════════════════════════════════════════════════════════
    # 14. ROLLING ATS (3 features)
    # ══════════════════════════════════════════════════════════
    df["roll_ats_diff_gated"] = np.where(
        df["has_market"] == 1,
        _col("home_ats_record_10", 0.5) - _col("away_ats_record_10", 0.5),
        0.0
    )
    df["roll_ats_margin_gated"] = np.where(
        df["has_market"] == 1,
        _col("home_ats_margin_10", 0) - _col("away_ats_margin_10", 0),
        0.0
    )
    df["has_ats_data"] = ((df["roll_ats_diff_gated"] != 0) | (df["roll_ats_margin_gated"] != 0)).astype(int)

    # ══════════════════════════════════════════════════════════
    # 15. ESPN SIGNAL (2 features — zeros until ESPN scrape)
    # ══════════════════════════════════════════════════════════
    df["espn_wp_edge"] = _col("espn_wp_edge", 0)
    df["crowd_pct"] = _col("crowd_pct", 0)

    # ══════════════════════════════════════════════════════════
    # 16. INJURY (2 features — zeros for historical)
    # ══════════════════════════════════════════════════════════
    df["injury_diff"] = _col("home_injury_penalty", 0) - _col("away_injury_penalty", 0)
    df["missing_starters_diff"] = _col("home_missing_starters", 0) - _col("away_missing_starters", 0)

    # ══════════════════════════════════════════════════════════
    # 17. NBA-SPECIFIC (2 features)
    # ══════════════════════════════════════════════════════════
    h_total = h_wins + h_losses; a_total = a_wins + a_losses
    df["min_games_played"] = np.minimum(h_total, a_total)
    df["games_diff"] = h_total - a_total

    # ══════════════════════════════════════════════════════════
    # ASSEMBLE FEATURE VECTOR
    # ══════════════════════════════════════════════════════════
    feature_cols = [
        # 1. Core efficiency (7)
        "score_diff_pred", "win_pct_home", "home_fav", "net_rtg_diff",
        "adj_oe_diff", "adj_de_diff", "ou_gap",
        # 2. Raw stat diffs (17)
        "ppg_diff", "fgpct_diff", "threepct_diff", "ftpct_diff",
        "orb_pct_diff", "fta_rate_diff", "ato_ratio_diff",
        "opp_ppg_diff", "opp_fgpct_diff", "opp_threepct_diff",
        "steals_diff", "blocks_diff", "assists_diff", "turnovers_diff",
        "to_margin_diff", "steals_to_diff", "tempo_avg",
        # 3. Advanced shooting (10)
        "efg_diff", "three_rate_diff", "ppp_diff",
        "ft_dependency_diff", "three_value_diff", "drb_pct_diff",
        "opp_efg_diff", "three_divergence_diff", "ppp_divergence_diff",
        # 4. Market (7)
        "market_spread", "market_total", "spread_vs_market", "has_market",
        "spread_regime", "spread_x_market",
        # 5. Elo (2)
        "elo_diff", "elo_win_prob",
        # 6. Context (7)
        "win_pct_diff", "form_diff", "season_phase", "is_early_season",
        "is_midweek", "is_playoff", "min_games_played",
        # 7. Momentum/form (15)
        "margin_trend_diff", "margin_accel_diff", "streak_diff",
        "days_since_loss_diff", "games_since_blowout_diff",
        "wl_momentum_diff", "momentum_halflife_diff", "win_aging_diff",
        "recovery_diff", "after_loss_either",
        "pyth_luck_diff", "pyth_residual_diff", "opp_adj_form_diff",
        "regression_diff", "games_last_14_diff",
        # 8. Scoring distribution (8)
        "scoring_var_diff", "score_kurtosis_diff", "margin_skew_diff",
        "ceiling_diff", "floor_diff", "scoring_entropy_diff",
        "bimodal_diff", "consistency_diff",
        # 9. Defense advanced (6)
        "def_stability_diff", "opp_suppression_diff",
        "steal_foul_diff", "block_foul_diff", "def_eff_diff", "block_rate_diff",
        # 10. Rest/fatigue (7)
        "rest_diff", "away_travel", "home_b2b", "away_b2b", "b2b_diff",
        "fatigue_diff", "season_pct_avg",
        # 11. Interactions (6)
        "fatigue_x_quality", "rest_x_defense", "form_x_familiarity",
        "consistency_x_spread", "rest_x_travel", "netrtg_x_sos",
        # 12. Matchup (11)
        "matchup_efg", "matchup_to", "matchup_orb", "matchup_ft",
        "style_familiarity", "pace_leverage", "pace_control_diff",
        "common_opp_diff", "n_common_opps", "venue_advantage", "timezone_diff",
        # 13. Referee (4)
        "ref_home_whistle", "ref_ou_bias", "ref_foul_rate", "ref_pace_impact",
        # 14. ATS (3)
        "roll_ats_diff_gated", "roll_ats_margin_gated", "has_ats_data",
        # 15. ESPN signal (2)
        "espn_wp_edge", "crowd_pct",
        # 16. Injury (2)
        "injury_diff", "missing_starters_diff",
        # 17. NBA-specific (2)
        "games_diff",
    ]

    return df[feature_cols].fillna(0)
