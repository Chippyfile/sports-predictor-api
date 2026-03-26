#!/usr/bin/env python3
"""
nba_build_features_v27.py — Build ~112 candidate features from nba_training_data.parquet

Usage:
    python3 nba_build_features_v27.py                   # Build features + save
    python3 nba_build_features_v27.py --sweep            # Build + run model sweep
    python3 nba_build_features_v27.py --eliminate         # Build + L1 elimination
    python3 nba_build_features_v27.py --ats              # Build + ATS-focused eval

All features must be servable from pre-game data (ESPN summary, rolling tables, enrichment).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

def load_training_data(path="nba_training_data.parquet"):
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} games, {len(df.columns)} columns")
    print(f"Seasons: {sorted(df.season.unique())}")
    
    # Target: home margin
    df["target_margin"] = df["actual_home_score"] - df["actual_away_score"]
    df["target_ats"] = df["target_margin"] + df["market_spread_home"]
    df["target_home_win"] = (df["target_margin"] > 0).astype(int)
    
    # Filter: need market spread
    df = df[df["market_spread_home"] != 0].copy()
    print(f"After market filter: {len(df)} games")
    
    return df


# ═══════════════════════════════════════════════════════════
# FEATURE BUILDER
# ═══════════════════════════════════════════════════════════

def _safe_diff(df, home_col, away_col, default=0):
    """Compute home - away diff with safe defaults."""
    h = pd.to_numeric(df.get(home_col, default), errors="coerce").fillna(default)
    a = pd.to_numeric(df.get(away_col, default), errors="coerce").fillna(default)
    return h - a


def _safe_col(df, col, default=0):
    """Get column with safe default."""
    return pd.to_numeric(df.get(col, default), errors="coerce").fillna(default)


def build_features(df):
    """Build all ~112 candidate features. Returns (X, feature_names)."""
    
    feats = pd.DataFrame(index=df.index)
    
    # ══════════════════════════════════════════════════════
    # GROUP 1: MARKET FEATURES (10)
    # ══════════════════════════════════════════════════════
    
    feats["market_spread"] = _safe_col(df, "market_spread_home")
    
    # Overround from ML odds
    def _ml_to_prob(ml):
        if isinstance(ml, np.ndarray):
            ml = np.where(np.isnan(ml), 0, ml)
        else:
            ml = pd.to_numeric(ml, errors="coerce").fillna(0).values
        prob = np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
        return np.where(ml == 0, 0.5, prob)
    
    h_ml = _safe_col(df, "home_ml_close", 0).values
    a_ml = _safe_col(df, "away_ml_close", 0).values
    # Fallback to open if close not available
    h_ml_open = _safe_col(df, "home_ml_open", 0).values
    a_ml_open = _safe_col(df, "away_ml_open", 0).values
    h_ml = np.where(h_ml == 0, h_ml_open, h_ml)
    a_ml = np.where(a_ml == 0, a_ml_open, a_ml)
    # Fallback to dk
    h_ml_dk = _safe_col(df, "dk_home_ml", 0).values
    a_ml_dk = _safe_col(df, "dk_away_ml", 0).values
    h_ml = np.where(h_ml == 0, h_ml_dk, h_ml)
    a_ml = np.where(a_ml == 0, a_ml_dk, a_ml)
    
    impl_h = _ml_to_prob(h_ml)
    impl_a = _ml_to_prob(a_ml)
    
    feats["overround"] = np.round(impl_h + impl_a - 1, 4)
    feats["implied_prob_home"] = np.round(np.where(impl_h + impl_a > 0,
                                                     impl_h / np.maximum(impl_h + impl_a, 0.01),
                                                     0.5), 4)
    
    spread_prob = 1 / (1 + 10 ** (feats["market_spread"] / 8))
    feats["ml_spread_dislocation"] = np.round(impl_h - spread_prob, 4)
    feats["home_fav"] = (feats["market_spread"] < 0).astype(int)
    
    # Market total
    mkt_total = _safe_col(df, "market_ou_total", 0).values
    mkt_total_ou = _safe_col(df, "ou_total", 0).values
    mkt_total_dk = _safe_col(df, "dk_ou", 0).values
    mkt_total = np.where(mkt_total == 0, mkt_total_ou, mkt_total)
    mkt_total = np.where(mkt_total == 0, mkt_total_dk, mkt_total)
    feats["market_total"] = mkt_total
    feats["ou_gap"] = _safe_col(df, "home_ppg") + _safe_col(df, "away_ppg") - mkt_total
    
    # Line movement
    spread_open = _safe_col(df, "spread_open", 0).values
    spread_close = _safe_col(df, "spread_close", 0).values
    spread_close = np.where(spread_close == 0, feats["market_spread"].values, spread_close)
    feats["sharp_spread_signal"] = np.round(spread_close - spread_open, 2)
    
    ml_open_h = _ml_to_prob(_safe_col(df, "opening_home_ml", 0).values)
    ml_close_h = _ml_to_prob(h_ml)
    feats["sharp_ml_signal"] = np.round(ml_close_h - ml_open_h, 4)
    
    sm = feats["sharp_spread_signal"]
    mm = feats["sharp_ml_signal"]
    feats["reverse_line_movement"] = ((sm != 0) & (mm != 0) & (sm * mm > 0)).astype(int)
    feats["line_reversal"] = np.round(np.abs(mm), 4)
    
    # NEW: public spread pct proxy (DK implied - ESPN predictor)
    espn_wp = _safe_col(df, "espn_pregame_wp", 0.5)
    feats["public_home_spread_pct"] = np.round(feats["implied_prob_home"] - espn_wp, 4)
    feats["public_home_spread_pct"] = np.where(espn_wp > 0, feats["public_home_spread_pct"], 0)
    
    # NEW: spread juice imbalance
    dk_h_odds = _safe_col(df, "dk_home_spread_odds", -110).values
    dk_a_odds = _safe_col(df, "dk_away_spread_odds", -110).values
    h_juice = np.where(dk_h_odds < 0, -dk_h_odds / (-dk_h_odds + 100), 100 / (dk_h_odds + 100))
    a_juice = np.where(dk_a_odds < 0, -dk_a_odds / (-dk_a_odds + 100), 100 / (dk_a_odds + 100))
    feats["spread_juice_imbalance"] = np.round(h_juice - a_juice, 4)
    
    # NEW: vig uncertainty
    feats["vig_uncertainty"] = np.round(feats["overround"] - 0.045, 4)
    
    # NEW: ML implied spread
    feats["ml_implied_spread"] = np.round(
        np.where(feats["implied_prob_home"] > 0.01,
                 -8 * np.log10(feats["implied_prob_home"] / (1 - feats["implied_prob_home"] + 0.001) + 0.001),
                 0), 2)
    
    # ══════════════════════════════════════════════════════
    # GROUP 2: ESPN WIN PROBABILITY (2)
    # ══════════════════════════════════════════════════════
    
    feats["espn_pregame_wp"] = _safe_col(df, "espn_pregame_wp", 0.5)
    feats["espn_pregame_wp_pbp"] = _safe_col(df, "espn_pregame_wp_pbp", 0.5)
    
    # ══════════════════════════════════════════════════════
    # GROUP 3: TEAM STAT DIFFS (10)
    # ══════════════════════════════════════════════════════
    
    feats["net_rtg_diff"] = _safe_diff(df, "home_net_rtg", "away_net_rtg")
    feats["opp_ppg_diff"] = _safe_diff(df, "home_opp_ppg", "away_opp_ppg")
    feats["threepct_diff"] = _safe_diff(df, "home_threepct", "away_threepct")
    feats["turnovers_diff"] = _safe_diff(df, "home_turnovers", "away_turnovers")
    feats["win_pct_diff"] = (
        _safe_col(df, "home_wins") / np.maximum(_safe_col(df, "home_wins") + _safe_col(df, "home_losses"), 1) -
        _safe_col(df, "away_wins") / np.maximum(_safe_col(df, "away_wins") + _safe_col(df, "away_losses"), 1)
    )
    
    # eFG diff: FG% + 0.5 * 3P% (simplified)
    h_fg = _safe_col(df, "home_fgpct", 0.46)
    a_fg = _safe_col(df, "away_fgpct", 0.46)
    h_3p = _safe_col(df, "home_threepct", 0.36)
    a_3p = _safe_col(df, "away_threepct", 0.36)
    feats["efg_diff"] = np.round((h_fg + 0.2 * h_3p) - (a_fg + 0.2 * a_3p), 4)
    
    feats["ftpct_diff"] = _safe_diff(df, "home_ftpct", "away_ftpct")
    feats["drb_pct_diff"] = _safe_diff(df, "home_orb_pct", "away_orb_pct") * -1  # DRB% ≈ 1 - opp ORB%
    
    feats["steals_to_diff"] = (
        (_safe_col(df, "home_steals") / np.maximum(_safe_col(df, "home_turnovers"), 1)) -
        (_safe_col(df, "away_steals") / np.maximum(_safe_col(df, "away_turnovers"), 1))
    )
    feats["ato_ratio_diff"] = _safe_diff(df, "home_ato_ratio", "away_ato_ratio")
    
    # ══════════════════════════════════════════════════════
    # GROUP 4: PLAYER FEATURES (3)
    # ══════════════════════════════════════════════════════
    
    feats["star1_share_diff"] = _safe_diff(df, "home_star1_share", "away_star1_share")
    feats["lineup_value_diff"] = _safe_diff(df, "home_lineup_value", "away_lineup_value")
    
    # Star minutes fatigue: star share × games in last 14
    h_star = _safe_col(df, "home_star1_share", 0.25)
    a_star = _safe_col(df, "away_star1_share", 0.25)
    h_g14 = _safe_col(df, "home_games_last_14", 6)
    a_g14 = _safe_col(df, "away_games_last_14", 6)
    feats["star_minutes_fatigue_diff"] = np.round(h_star * h_g14 - a_star * a_g14, 3)
    
    # ══════════════════════════════════════════════════════
    # GROUP 5: ELO (2)
    # ══════════════════════════════════════════════════════
    
    feats["elo_diff"] = _safe_diff(df, "home_form", "away_form")  # Elo stored as 'form' in training
    spread_prob_elo = 1 / (1 + 10 ** (feats["market_spread"] / 8))
    elo_prob = 1 / (1 + 10 ** (-feats["elo_diff"] / 400))
    feats["elo_market_residual"] = np.round(elo_prob - spread_prob_elo, 4)
    
    # ══════════════════════════════════════════════════════
    # GROUP 6: SCHEDULE (7)
    # ══════════════════════════════════════════════════════
    
    feats["rest_diff"] = _safe_diff(df, "home_days_rest", "away_days_rest")
    feats["b2b_diff"] = (_safe_col(df, "home_days_rest") == 0).astype(int) - \
                         (_safe_col(df, "away_days_rest") == 0).astype(int)
    feats["home_b2b"] = (_safe_col(df, "home_days_rest") == 0).astype(int)
    feats["games_last_14_diff"] = _safe_diff(df, "home_games_last_14", "away_games_last_14")
    feats["streak_diff"] = _safe_diff(df, "home_streak", "away_streak")
    feats["games_diff"] = (_safe_col(df, "home_wins") + _safe_col(df, "home_losses")) - \
                           (_safe_col(df, "away_wins") + _safe_col(df, "away_losses"))
    
    # NEW: days since loss
    feats["days_since_loss_diff"] = _safe_diff(df, "home_days_since_loss", "away_days_since_loss")
    
    # ══════════════════════════════════════════════════════
    # GROUP 7: ROLLING PBP (13 — 8 existing + 5 new)
    # ══════════════════════════════════════════════════════
    
    feats["roll_q4_diff"] = _safe_diff(df, "home_roll_q4", "away_roll_q4")
    feats["roll_paint_pts_diff"] = _safe_diff(df, "home_roll_paint_pts", "away_roll_paint_pts")
    feats["roll_fast_break_diff"] = _safe_diff(df, "home_roll_fast_break_pts", "away_roll_fast_break_pts")
    feats["roll_largest_lead_diff"] = _safe_diff(df, "home_roll_largest_lead", "away_roll_largest_lead")
    feats["roll_max_run_avg"] = (_safe_col(df, "home_roll_max_run") + _safe_col(df, "away_roll_max_run")) / 2
    feats["roll_bench_pts_diff"] = _safe_diff(df, "home_roll_bench_pts", "away_roll_bench_pts")
    feats["roll_ft_trip_rate_diff"] = _safe_diff(df, "home_roll_ft_trip_rate", "away_roll_ft_trip_rate")
    feats["roll_three_fg_rate_diff"] = _safe_diff(df, "home_roll_three_fg_rate", "away_roll_three_fg_rate")
    
    # NEW rolling PBP
    feats["roll_pts_off_to_diff"] = _safe_diff(df, "home_roll_pts_off_turnovers", "away_roll_pts_off_turnovers")
    # Fallback column name
    if feats["roll_pts_off_to_diff"].abs().sum() == 0:
        feats["roll_pts_off_to_diff"] = _safe_diff(df, "home_roll_pts_off_to", "away_roll_pts_off_to")
    feats["roll_oreb_diff"] = _safe_diff(df, "home_roll_oreb", "away_roll_oreb")
    feats["roll_game_pf_diff"] = _safe_diff(df, "home_roll_game_pf", "away_roll_game_pf")
    feats["roll_dreb_diff"] = _safe_diff(df, "home_roll_dreb", "away_roll_dreb")
    feats["roll_paint_fg_rate_diff"] = _safe_diff(df, "home_roll_paint_fg_rate", "away_roll_paint_fg_rate")
    
    # ══════════════════════════════════════════════════════
    # GROUP 8: ATS (3 — 1 existing + 2 new)
    # ══════════════════════════════════════════════════════
    
    feats["roll_ats_margin_gated"] = _safe_diff(df, "home_ats_margin_10", "away_ats_margin_10")
    
    # NEW: ATS records
    feats["ats_record_diff"] = _safe_diff(df, "home_ats_record_10", "away_ats_record_10")
    feats["after_loss_either"] = _safe_col(df, "after_loss_either", 0)
    feats["home_after_loss"] = _safe_col(df, "home_after_loss", 0)
    feats["away_after_loss"] = _safe_col(df, "away_after_loss", 0)
    
    # ══════════════════════════════════════════════════════
    # GROUP 9: ENRICHMENT (22 — 12 existing + 10 new)
    # ══════════════════════════════════════════════════════
    
    # Existing enrichment diffs
    feats["scoring_var_diff"] = _safe_diff(df, "home_scoring_var", "away_scoring_var")
    feats["bimodal_diff"] = _safe_diff(df, "home_bimodal", "away_bimodal")
    feats["scoring_entropy_diff"] = _safe_diff(df, "home_scoring_entropy", "away_scoring_entropy")
    feats["consistency_diff"] = _safe_diff(df, "home_ceiling", "away_ceiling") - \
                                 _safe_diff(df, "home_floor", "away_floor")  # consistency = ceiling - floor range
    feats["ceiling_diff"] = _safe_diff(df, "home_ceiling", "away_ceiling")
    feats["def_stability_diff"] = _safe_diff(df, "home_def_stability", "away_def_stability")
    feats["opp_suppression_diff"] = _safe_diff(df, "home_opp_suppression", "away_opp_suppression")
    feats["three_value_diff"] = (
        _safe_col(df, "home_three_fg_rate") * (_safe_col(df, "home_three_fg_rate") - 0.20) -
        _safe_col(df, "away_three_fg_rate") * (_safe_col(df, "away_three_fg_rate") - 0.20)
    )
    feats["ts_regression_diff"] = _safe_diff(df, "home_ts_regression", "away_ts_regression")
    feats["three_pt_regression_diff"] = _safe_diff(df, "home_three_pt_regression", "away_three_pt_regression")
    
    # pace_leverage: combined (average), not diff
    feats["pace_leverage"] = (
        np.abs(_safe_col(df, "home_opp_suppression")) + np.abs(_safe_col(df, "away_opp_suppression"))
    ) * 0.01
    
    feats["pace_control_diff"] = 1 / np.maximum(_safe_col(df, "home_scoring_var", 10), 1) - \
                                  1 / np.maximum(_safe_col(df, "away_scoring_var", 10), 1)
    
    # NEW enrichment features
    feats["floor_diff"] = _safe_diff(df, "home_floor", "away_floor")
    feats["score_kurtosis_diff"] = _safe_diff(df, "home_score_kurtosis", "away_score_kurtosis")
    feats["margin_skew_diff"] = _safe_diff(df, "home_margin_skew", "away_margin_skew")
    feats["pyth_residual_diff"] = _safe_diff(df, "home_pyth_residual", "away_pyth_residual")
    feats["pyth_luck_diff"] = _safe_diff(df, "home_pyth_luck", "away_pyth_luck")
    feats["margin_accel_diff"] = _safe_diff(df, "home_margin_accel", "away_margin_accel")
    feats["momentum_halflife_diff"] = _safe_diff(df, "home_momentum_halflife", "away_momentum_halflife")
    feats["win_aging_diff"] = _safe_diff(df, "home_win_aging", "away_win_aging")
    feats["opp_adj_form_diff"] = _safe_diff(df, "home_opp_adj_form", "away_opp_adj_form")
    feats["recovery_diff"] = _safe_diff(df, "home_recovery", "away_recovery")
    feats["scoring_hhi_diff"] = _safe_diff(df, "home_scoring_hhi", "away_scoring_hhi")
    
    # ══════════════════════════════════════════════════════
    # GROUP 10: MATCHUP (4)
    # ══════════════════════════════════════════════════════
    
    feats["matchup_efg"] = _safe_diff(df, "home_three_fg_rate", "away_three_fg_rate")
    feats["matchup_to"] = (_safe_col(df, "home_net_rtg") - _safe_col(df, "away_net_rtg")) * 0.02
    
    h_oreb = _safe_col(df, "home_oreb", 5).values
    a_oreb = _safe_col(df, "away_oreb", 5).values
    # Use rolling if available
    h_oreb_r = _safe_col(df, "home_roll_oreb", 0).values
    a_oreb_r = _safe_col(df, "away_roll_oreb", 0).values
    h_oreb = np.where(h_oreb_r > 0, h_oreb_r, h_oreb)
    a_oreb = np.where(a_oreb_r > 0, a_oreb_r, a_oreb)
    feats["matchup_orb"] = np.round(h_oreb - a_oreb, 2)
    
    feats["matchup_ft"] = _safe_diff(df, "home_ft_trip_rate", "away_ft_trip_rate")
    
    # ══════════════════════════════════════════════════════
    # GROUP 11: H2H (3 — computed from prior meetings)
    # ══════════════════════════════════════════════════════
    
    feats["h2h_total_games"] = _safe_col(df, "h2h_total_games", 0)
    
    # Compute h2h_avg_margin and is_revenge_home from actual game results
    # (These columns don't exist in raw parquet — must derive from prior meetings)
    h2h_margin = np.zeros(len(df))
    revenge = np.zeros(len(df))
    
    if "home_team" in df.columns and "away_team" in df.columns and "actual_home_score" in df.columns:
        _ht = df["home_team"].values
        _at = df["away_team"].values
        _season = df["season"].values if "season" in df.columns else np.zeros(len(df))
        _margin = (pd.to_numeric(df["actual_home_score"], errors="coerce").fillna(0) -
                   pd.to_numeric(df["actual_away_score"], errors="coerce").fillna(0)).values
        _gd = df["game_date"].values if "game_date" in df.columns else np.arange(len(df))
        
        # Group by matchup pair + season for efficiency
        _pair_key = {}
        for i in range(len(df)):
            pair = tuple(sorted([str(_ht[i]), str(_at[i])]))
            key = (pair, _season[i])
            if key not in _pair_key:
                _pair_key[key] = []
            _pair_key[key].append(i)
        
        for key, indices in _pair_key.items():
            if len(indices) < 2:
                continue
            for pos in range(1, len(indices)):
                idx = indices[pos]
                home = str(_ht[idx])
                # Get margins from home team's perspective for all prior meetings
                prior_margins = []
                for pi in indices[:pos]:
                    if str(_ht[pi]) == home:
                        prior_margins.append(_margin[pi])
                    else:
                        prior_margins.append(-_margin[pi])
                
                h2h_margin[idx] = round(float(np.mean(prior_margins)), 1)
                revenge[idx] = 1 if prior_margins[-1] < 0 else 0
    
    feats["h2h_avg_margin"] = h2h_margin
    feats["is_revenge_home"] = revenge
    
    # ══════════════════════════════════════════════════════
    # GROUP 12: CONTEXT (12 — 4 existing + 8 new)
    # ══════════════════════════════════════════════════════
    
    feats["is_midweek"] = _safe_col(df, "is_midweek", 0)
    # Recompute if column doesn't exist
    if "is_midweek" not in df.columns or df["is_midweek"].sum() == 0:
        dow = pd.to_datetime(df["game_date"]).dt.dayofweek
        feats["is_midweek"] = ((dow >= 0) & (dow <= 3)).astype(int)  # Mon-Thu
    
    feats["post_trade_deadline"] = _safe_col(df, "post_trade_deadline", 0)
    feats["crowd_pct"] = _safe_col(df, "attendance", 18000) / np.maximum(_safe_col(df, "venue_capacity", 19000), 1)
    feats["crowd_pct"] = np.clip(feats["crowd_pct"], 0, 1.05)
    feats["away_is_public_team"] = _safe_col(df, "away_is_public_team", 0)
    
    # NEW context
    dow = pd.to_datetime(df["game_date"]).dt.dayofweek
    feats["is_friday_sat"] = ((dow == 4) | (dow == 5)).astype(int)
    feats["is_sunday"] = (dow == 6).astype(int)
    
    total_games = _safe_col(df, "home_wins") + _safe_col(df, "home_losses")
    feats["is_early_season"] = (total_games < 15).astype(int)
    feats["season_phase"] = np.where(total_games < 20, 0,
                             np.where(total_games < 50, 1,
                             np.where(total_games < 70, 2, 3)))
    
    feats["post_allstar"] = _safe_col(df, "post_allstar", 0)
    feats["altitude_factor"] = _safe_col(df, "altitude_factor", 0)
    feats["timezone_diff"] = _safe_col(df, "timezone_diff", 0)
    # Compute conference_game from team abbreviations
    NBA_CONF = {
        "ATL":"E","BOS":"E","BKN":"E","CHA":"E","CHI":"E","CLE":"E",
        "DET":"E","IND":"E","MIA":"E","MIL":"E","NYK":"E","ORL":"E",
        "PHI":"E","TOR":"E","WAS":"E",
        "DAL":"W","DEN":"W","GSW":"W","HOU":"W","LAC":"W","LAL":"W",
        "MEM":"W","MIN":"W","NOP":"W","OKC":"W","PHX":"W","POR":"W",
        "SAC":"W","SAS":"W","UTA":"W",
    }
    if "home_team" in df.columns and "away_team" in df.columns:
        feats["conference_game"] = df.apply(
            lambda r: 1 if NBA_CONF.get(str(r.get("home_team",""))) == NBA_CONF.get(str(r.get("away_team",""))) else 0,
            axis=1
        ).values
    else:
        feats["conference_game"] = _safe_col(df, "conference_game", 0)
    
    # is_revenge_home already computed in GROUP 11 above
    
    # NEW: injuries
    feats["injuries_out_diff"] = _safe_diff(df, "home_players_out", "away_players_out")
    
    # ══════════════════════════════════════════════════════
    # GROUP 13: REFEREE (4)
    # ══════════════════════════════════════════════════════
    
    feats["ref_home_whistle"] = _safe_col(df, "ref_home_whistle", 0)
    feats["ref_ou_bias"] = _safe_col(df, "ref_ou_bias", 0)
    feats["ref_pace_impact"] = _safe_col(df, "ref_pace_impact", 0)
    feats["ref_foul_proxy"] = _safe_col(df, "ref_foul_proxy", 0)
    
    # ══════════════════════════════════════════════════════
    # GROUP 14: INTERACTIONS (1)
    # ══════════════════════════════════════════════════════
    
    tight = np.where(np.abs(feats["market_spread"]) <= 5, 1.0,
             np.where(np.abs(feats["market_spread"]) <= 8, 0.5, 0.0))
    feats["clutch_x_tight_spread"] = np.round(feats["roll_q4_diff"] * tight, 3)
    
    # ══════════════════════════════════════════════════════
    # CLEAN UP
    # ══════════════════════════════════════════════════════
    
    # Replace infinities and NaN
    feats = feats.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Drop any columns that are 100% zero (no signal)
    zero_cols = [c for c in feats.columns if (feats[c] == 0).all()]
    if zero_cols:
        print(f"\nDropping {len(zero_cols)} all-zero columns: {zero_cols}")
        feats = feats.drop(columns=zero_cols)
    
    # Coverage report
    nonzero_pct = {}
    for c in feats.columns:
        pct = (feats[c] != 0).mean()
        nonzero_pct[c] = pct
    
    low_coverage = {c: p for c, p in nonzero_pct.items() if p < 0.10}
    if low_coverage:
        print(f"\nWarning: {len(low_coverage)} features with <10% nonzero coverage:")
        for c, p in sorted(low_coverage.items(), key=lambda x: x[1]):
            print(f"  {c:40s} {p:.1%}")
    
    feature_names = list(feats.columns)
    print(f"\nBuilt {len(feature_names)} features")
    
    return feats, feature_names


# ═══════════════════════════════════════════════════════════
# MODEL TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════

def walk_forward_eval(X, y, df, n_folds=30, model_type="catboost", **kwargs):
    """Walk-forward cross-validation with ATS evaluation."""
    from sklearn.preprocessing import StandardScaler
    
    # Sort by date
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    df_sorted = df.iloc[sort_idx].reset_index(drop=True)
    dates = dates.iloc[sort_idx].reset_index(drop=True)
    
    fold_size = len(X) // (n_folds + 1)
    min_train = fold_size * 2
    
    results = []
    
    for fold in range(n_folds):
        test_start = min_train + fold * fold_size
        test_end = min(test_start + fold_size, len(X))
        if test_end > len(X):
            break
        
        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        df_test = df_sorted.iloc[test_start:test_end]
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train
        if model_type == "catboost":
            from catboost import CatBoostRegressor
            depth = kwargs.get("depth", 4)
            iters = kwargs.get("iterations", 650)
            model = CatBoostRegressor(
                depth=depth, iterations=iters, learning_rate=0.05,
                l2_leaf_reg=3, random_seed=42, verbose=0
            )
            model.fit(X_train_s, y_train)
        elif model_type == "lasso":
            from sklearn.linear_model import Lasso
            alpha = kwargs.get("alpha", 0.1)
            model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X_train_s, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        preds = model.predict(X_test_s)
        
        # Metrics
        mae = np.mean(np.abs(preds - y_test))
        spread = _safe_col(df_test, "market_spread_home")
        actual_margin = y_test.values
        
        # ATS: model picks side based on predicted margin vs spread
        ats_edge = preds - (-spread.values)  # positive = model says home covers
        ats_correct = np.sign(ats_edge) == np.sign(actual_margin + spread.values)
        # Exclude pushes
        not_push = (actual_margin + spread.values) != 0
        
        for threshold in [0, 2, 4, 7, 10]:
            mask = (np.abs(ats_edge) >= threshold) & not_push
            n = mask.sum()
            if n > 0:
                acc = ats_correct[mask].mean()
                results.append({
                    "fold": fold, "mae": mae, "threshold": threshold,
                    "n_picks": n, "ats_acc": acc
                })
    
    rdf = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"  Walk-Forward Results ({model_type}, {n_folds} folds)")
    print(f"{'='*70}")
    
    overall_mae = rdf.groupby("fold")["mae"].first().mean()
    print(f"  MAE: {overall_mae:.3f}")
    
    for t in [0, 2, 4, 7, 10]:
        sub = rdf[rdf["threshold"] == t]
        if len(sub) == 0:
            continue
        acc = sub["ats_acc"].mean()
        n_avg = sub["n_picks"].mean()
        roi = (acc * 1.909 - 1) * 100  # -110 juice
        print(f"  ATS {t}+: {acc:.1%} ({n_avg:.0f} picks/fold) → ROI {roi:+.1f}%")
    
    return rdf, overall_mae


def l1_elimination(X, y, feature_names, n_alphas=20):
    """Use Lasso path to rank features by importance."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    alphas = np.logspace(-3, 1, n_alphas)
    
    # Track when each feature drops out
    feature_persistence = {f: 0 for f in feature_names}
    
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_s, y)
        active = np.abs(model.coef_) > 1e-6
        for i, f in enumerate(feature_names):
            if active[i]:
                feature_persistence[f] += 1
    
    ranked = sorted(feature_persistence.items(), key=lambda x: -x[1])
    
    print(f"\n{'='*70}")
    print(f"  L1 Feature Ranking ({n_alphas} alphas)")
    print(f"{'='*70}")
    for i, (feat, count) in enumerate(ranked[:30]):
        bar = "█" * count + "░" * (n_alphas - count)
        print(f"  {i+1:3d}. {feat:40s} {bar} {count}/{n_alphas}")
    
    # Features that survive < 25% of alphas are candidates for removal
    weak = [(f, c) for f, c in ranked if c < n_alphas * 0.25]
    print(f"\n  Weak features (<25% survival): {len(weak)}")
    for f, c in weak:
        print(f"    {f:40s} {c}/{n_alphas}")
    
    return ranked


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run model sweep")
    parser.add_argument("--eliminate", action="store_true", help="L1 feature elimination")
    parser.add_argument("--ats", action="store_true", help="ATS-focused evaluation")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iters", type=int, default=650)
    args = parser.parse_args()
    
    df = load_training_data()
    X, feature_names = build_features(df)
    y = df["target_margin"]
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target: home margin (mean {y.mean():.2f}, std {y.std():.2f})")
    
    if args.eliminate:
        ranked = l1_elimination(X, y, feature_names)
        # Save ranking
        pd.DataFrame(ranked, columns=["feature", "l1_persistence"]).to_csv(
            "nba_v27_feature_ranking.csv", index=False)
        print(f"\nSaved to nba_v27_feature_ranking.csv")
    
    if args.sweep:
        print("\n" + "="*70)
        print("  MODEL SWEEP")
        print("="*70)
        
        for mt, kw in [
            ("catboost", {"depth": 3, "iterations": 650}),
            ("catboost", {"depth": 4, "iterations": 650}),
            ("catboost", {"depth": 5, "iterations": 650}),
            ("lasso", {"alpha": 0.05}),
            ("lasso", {"alpha": 0.1}),
            ("lasso", {"alpha": 0.2}),
        ]:
            label = f"{mt}({', '.join(f'{k}={v}' for k,v in kw.items())})"
            print(f"\n>>> {label}")
            try:
                walk_forward_eval(X, y, df, n_folds=20, model_type=mt, **kw)
            except Exception as e:
                print(f"  ERROR: {e}")
    
    if args.ats:
        print("\n" + "="*70)
        print("  ATS-FOCUSED EVAL (CatBoost d=4)")
        print("="*70)
        walk_forward_eval(X, y, df, n_folds=30, model_type="catboost",
                         depth=args.depth, iterations=args.iters)
    
    if not (args.sweep or args.eliminate or args.ats):
        # Default: just build features and print summary
        print("\nRun with --eliminate, --sweep, or --ats to train models")
        print("Example: python3 nba_build_features_v27.py --eliminate --sweep --ats")
        
        # Save feature matrix for inspection
        X.to_parquet("nba_v27_features.parquet")
        print(f"Saved feature matrix to nba_v27_features.parquet")
