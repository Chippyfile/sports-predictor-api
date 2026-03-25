"""
nba_v27_features_live.py — Compute v27 features from live game data.

Maps ESPN + Supabase + market data → 38 v27 features for prediction.
Used by nba_full_predict.py at inference time.

Each feature documents its data source so missing data produces
a sensible default (0 for diffs, league average for levels).
"""
import numpy as np
import pandas as pd

# Static: "public" teams that attract betting action
PUBLIC_TEAMS = {
    "LAL", "LAC", "GSW", "BOS", "NYK", "BKN", "MIA", "CHI",
    "DAL", "PHX", "PHI", "MIL", "OKC", "DEN", "CLE",
}


def _safe(val, default=0):
    """Safely convert to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (ValueError, TypeError):
        return default


def build_v27_features(game: dict, enrichment: dict = None, ref_profile: dict = None) -> pd.DataFrame:
    """
    Build 38 v27 features from a live game dict.

    Args:
        game: Dict with keys from nba_full_predict.py's game assembly
              (ESPN stats, Supabase rolling stats, market data, etc.)
        enrichment: Dict with nba_team_enrichment data for both teams
                    {"home": {...}, "away": {...}}
        ref_profile: Dict with referee profile data (from nba_ref_profiles)

    Returns:
        Single-row DataFrame with the 38 v27 feature columns.
    """
    if enrichment is None:
        enrichment = {"home": {}, "away": {}}
    if ref_profile is None:
        ref_profile = {}

    he = enrichment.get("home", {})
    ae = enrichment.get("away", {})

    # ── Helpers ──
    def g(key, default=0):
        return _safe(game.get(key), default)

    def h_enr(key, default=0):
        return _safe(he.get(key), default)

    def a_enr(key, default=0):
        return _safe(ae.get(key), default)

    def ref(key, default=0):
        return _safe(ref_profile.get(key), default)

    # ── Basic stats from game dict ──
    h_ppg = g("home_ppg", 110)
    a_ppg = g("away_ppg", 110)
    h_opp_ppg = g("home_opp_ppg", 110)
    a_opp_ppg = g("away_opp_ppg", 110)
    h_fgpct = g("home_fgpct", 0.46)
    a_fgpct = g("away_fgpct", 0.46)
    h_3pct = g("home_threepct", 0.365)
    a_3pct = g("away_threepct", 0.365)
    h_ftpct = g("home_ftpct", 0.77)
    a_ftpct = g("away_ftpct", 0.77)
    h_steals = g("home_steals", 7.5)
    a_steals = g("away_steals", 7.5)
    h_to = g("home_turnovers", 14)
    a_to = g("away_turnovers", 14)
    h_wins = g("home_wins", 20)
    h_losses = g("home_losses", 20)
    a_wins = g("away_wins", 20)
    a_losses = g("away_losses", 20)

    h_net_rtg = g("home_net_rtg", 0)
    a_net_rtg = g("away_net_rtg", 0)

    market_spread = g("market_spread_home", 0)
    market_total = g("market_ou_total", 0)

    h_rest = g("home_days_rest", 2)
    a_rest = g("away_days_rest", 2)
    h_b2b = 1 if h_rest == 0 else 0
    a_b2b = 1 if a_rest == 0 else 0

    # ── eFG% (from enrichment or approximate from FG%/3P%) ──
    h_efg = h_enr("efg_pct", 0)
    a_efg = a_enr("efg_pct", 0)
    if h_efg == 0:
        # Approximate: eFG% ≈ FG% + 0.5 * 3P% * 3PA_rate (rough)
        h_efg = h_fgpct + 0.015  # rough adjustment
    if a_efg == 0:
        a_efg = a_fgpct + 0.015

    # Opponent eFG%
    h_opp_efg = h_enr("opp_efg_pct", 0) or g("home_opp_fgpct", 0.46) + 0.015
    a_opp_efg = a_enr("opp_efg_pct", 0) or g("away_opp_fgpct", 0.46) + 0.015

    # ═══════════════════════════════════════════════════════════
    # FEATURE COMPUTATIONS (38 features)
    # ═══════════════════════════════════════════════════════════

    feats = {}

    # 1. lineup_value_diff — from enrichment (lineup stability × star impact)
    feats["lineup_value_diff"] = h_enr("lineup_value", 0) - a_enr("lineup_value", 0)

    # 2. win_pct_diff
    h_wp = h_wins / max(h_wins + h_losses, 1)
    a_wp = a_wins / max(a_wins + a_losses, 1)
    feats["win_pct_diff"] = h_wp - a_wp

    # 3. scoring_hhi_diff — Herfindahl index of scoring concentration
    feats["scoring_hhi_diff"] = h_enr("scoring_hhi", 0.15) - a_enr("scoring_hhi", 0.15)

    # 4. espn_pregame_wp — ESPN pregame win probability for home team
    feats["espn_pregame_wp"] = g("espn_pregame_wp", 0.5)

    # 5. ceiling_diff — best N-game stretch margin
    feats["ceiling_diff"] = h_enr("ceiling", 0) - a_enr("ceiling", 0)

    # 6. matchup_efg — home 3FG rate vs away 3FG rate (matches training: home_three_fg_rate diff)
    feats["matchup_efg"] = g("home_three_fg_rate", 0) - g("away_three_fg_rate", 0)

    # 7. ml_implied_spread — train formula: -8 * log10(implied_prob / (1-implied_prob))
    #    Priority: pre-computed implied_prob_home → moneyline → 0.5 sentinel
    _precomp_imp = g("implied_prob_home", 0)
    home_odds = g("home_moneyline", 0)
    if _precomp_imp > 0.01:
        implied_prob = _precomp_imp  # passed from nba_full_predict (most accurate)
    elif home_odds != 0:
        if home_odds > 0:
            implied_prob = 100 / (home_odds + 100)
        else:
            implied_prob = abs(home_odds) / (abs(home_odds) + 100)
    else:
        implied_prob = 0.5  # sentinel: matches training default (no-moneyline rows)

    if implied_prob > 0.01:
        feats["ml_implied_spread"] = round(
            -8 * np.log10(implied_prob / (1 - implied_prob + 0.001) + 0.001), 2
        )
    else:
        feats["ml_implied_spread"] = 0

    # 8. sharp_spread_signal — market consensus signal
    #    Derived from: spread × juice direction or public% inverse
    feats["sharp_spread_signal"] = g("sharp_spread_signal", 0)

    # 9. efg_diff — effective field goal percentage differential
    feats["efg_diff"] = h_efg - a_efg

    # 10. opp_suppression_diff — raw column diff to match train formula
    # Train: _safe_diff(df, "home_opp_suppression", "away_opp_suppression") — raw units ~5-13
    feats["opp_suppression_diff"] = g("home_opp_suppression", 0) - g("away_opp_suppression", 0)

    # 11. net_rtg_diff
    feats["net_rtg_diff"] = h_net_rtg - a_net_rtg

    # 12. steals_to_diff — steals-to-turnover ratio differential
    h_sto = h_steals / max(h_to, 1)
    a_sto = a_steals / max(a_to, 1)
    feats["steals_to_diff"] = h_sto - a_sto

    # 13. threepct_diff
    feats["threepct_diff"] = h_3pct - a_3pct

    # 14. b2b_diff
    feats["b2b_diff"] = h_b2b - a_b2b

    # 15. ftpct_diff
    feats["ftpct_diff"] = h_ftpct - a_ftpct

    # 16. ou_gap — PPG-implied total vs market total
    # Train formula: home_ppg + away_ppg - mkt_total (positive = teams score more than market)
    ppg_total = h_ppg + a_ppg
    feats["ou_gap"] = ppg_total - market_total  # 0 when no market line → matches training sentinel

    # 17. roll_dreb_diff — rolling defensive rebounds differential
    feats["roll_dreb_diff"] = g("home_roll_dreb", 0) - g("away_roll_dreb", 0)

    # 18. ts_regression_diff — true shooting regression toward mean
    h_ts = h_enr("ts_pct", 0.56)
    a_ts = a_enr("ts_pct", 0.56)
    feats["ts_regression_diff"] = (h_ts - 0.56) - (a_ts - 0.56)

    # 19. roll_paint_pts_diff — rolling paint points differential
    feats["roll_paint_pts_diff"] = g("home_roll_paint_pts", 0) - g("away_roll_paint_pts", 0)

    # 20. ref_home_whistle — referee home bias tendency
    feats["ref_home_whistle"] = ref("home_whistle", 0)

    # 21. opp_ppg_diff
    feats["opp_ppg_diff"] = h_opp_ppg - a_opp_ppg

    # 22. roll_max_run_avg — average max run in recent games
    # Train: (home_roll_max_run + away_roll_max_run) / 2 — NOT a diff
    feats["roll_max_run_avg"] = (g("home_roll_max_run", 0) + g("away_roll_max_run", 0)) / 2

    # 23. away_is_public_team — check multiple key variants
    _away_ipt = game.get("away_is_public_team")
    if _away_ipt is not None:
        feats["away_is_public_team"] = int(_safe(_away_ipt, 0))
    else:
        away_abbr = (game.get("away_team_abbr") or game.get("away_abbr") or
                     game.get("away_team") or game.get("away_team_name", ""))
        feats["away_is_public_team"] = 1 if str(away_abbr).upper() in PUBLIC_TEAMS else 0

    # 24. away_after_loss — direct column preferred; derive from last_result as fallback
    _away_al = game.get("away_after_loss")
    if _away_al is not None:
        feats["away_after_loss"] = int(_safe(_away_al, 0))
    else:
        feats["away_after_loss"] = 1 if g("away_last_result", 0) == -1 else 0

    # 25. games_last_14_diff — games played in last 14 days
    feats["games_last_14_diff"] = g("home_games_last_14", 6) - g("away_games_last_14", 6)

    # 26. h2h_total_games — season head-to-head games played
    feats["h2h_total_games"] = g("h2h_total_games", 0)

    # 27. three_pt_regression_diff — reads pre-computed column (matches training)
    feats["three_pt_regression_diff"] = g("home_three_pt_regression", 0) - g("away_three_pt_regression", 0)

    # 28. games_diff — total games played differential
    h_games = h_wins + h_losses
    a_games = a_wins + a_losses
    feats["games_diff"] = h_games - a_games

    # 29. ref_foul_proxy — referee foul tendency
    feats["ref_foul_proxy"] = ref("foul_rate", 0) or ref("home_whistle", 0)

    # 30. roll_fast_break_diff — rolling fast break points differential
    # Train uses home_roll_fast_break_pts / away_roll_fast_break_pts
    feats["roll_fast_break_diff"] = g("home_roll_fast_break_pts", 0) - g("away_roll_fast_break_pts", 0)

    # 31. crowd_pct — attendance / venue_capacity (matches training formula)
    attendance = g("attendance", 0)
    venue_cap = g("venue_capacity", 19000)
    if attendance > 0 and venue_cap > 0:
        feats["crowd_pct"] = min(1.05, attendance / venue_cap)
    else:
        feats["crowd_pct"] = g("crowd_pct", 0.9)

    # 32. matchup_to — training formula: (home_net_rtg - away_net_rtg) * 0.02
    feats["matchup_to"] = (h_net_rtg - a_net_rtg) * 0.02

    # 33. overround — bookmaker margin (sum of implied probs - 1)
    home_odds = g("home_moneyline", 0)
    away_odds = g("away_moneyline", 0)

    def ml_to_prob(ml):
        if ml > 0:
            return 100 / (ml + 100)
        elif ml < 0:
            return abs(ml) / (abs(ml) + 100)
        return 0.5

    if home_odds != 0 and away_odds != 0:
        feats["overround"] = ml_to_prob(home_odds) + ml_to_prob(away_odds) - 1
    else:
        feats["overround"] = 0

    # 37. spread_juice_imbalance — h_juice - a_juice (matches train formula exactly)
    #     Source: pickcenter homeTeamOdds.spreadOdds / awayTeamOdds.spreadOdds
    #     e.g. home=-108, away=-112 → imbalance=+4 (home side cheaper = public on away)
    # Train formula: h_juice - a_juice where juice = implied prob from spread odds
    # e.g. -111 → 111/211 = 0.5261, -108 → 108/208 = 0.5192, diff = +0.0069
    h_sp_odds = g("home_spread_odds", 0)
    a_sp_odds = g("away_spread_odds", 0)
    if h_sp_odds != 0 and a_sp_odds != 0:
        def _odds_to_juice(ml):
            if ml < 0:
                return abs(ml) / (abs(ml) + 100)
            elif ml > 0:
                return 100 / (ml + 100)
            return 0.5
        feats["spread_juice_imbalance"] = round(
            _odds_to_juice(h_sp_odds) - _odds_to_juice(a_sp_odds), 4
        )
    else:
        feats["spread_juice_imbalance"] = 0

    # 8. sharp_spread_signal — exact training formula: spread_close - spread_open
    #    spread_close falls back to market_spread when 0 (matches training line 113)
    #    When no open line exists, spread_open=0 → signal = market_spread (closing line)
    #    Live fallback: juice approximation when no open/close data at all
    _spread_open  = g("spread_open", 0)
    _spread_close = g("spread_close", 0)
    if _spread_open != 0 and _spread_close != 0:
        # Real open+close data — compute actual movement
        feats["sharp_spread_signal"] = round(_spread_close - _spread_open, 2)
    elif _spread_open != 0:
        # Have open but close defaulted to market spread
        feats["sharp_spread_signal"] = round(market_spread - _spread_open, 2)
    elif feats["spread_juice_imbalance"] != 0:
        # No line movement data — approximate from juice asymmetry
        feats["sharp_spread_signal"] = round(-feats["spread_juice_imbalance"] * 0.05, 2)
    else:
        # No data — matches training behavior for pre-2022 no-line games
        feats["sharp_spread_signal"] = 0

    # 38. vig_uncertainty — overall market uncertainty (train: overround - 0.045)
    feats["vig_uncertainty"] = round(feats["overround"] - 0.045, 4)

    # 34. roll_ft_trip_rate_diff — FT attempt rate differential (rolling)
    feats["roll_ft_trip_rate_diff"] = g("home_roll_ft_trip_rate", 0) - g("away_roll_ft_trip_rate", 0)

    # 35. home_after_loss — direct column preferred; derive from last_result as fallback
    _home_al = game.get("home_after_loss")
    if _home_al is not None:
        feats["home_after_loss"] = int(_safe(_home_al, 0))
    else:
        feats["home_after_loss"] = 1 if g("home_last_result", 0) == -1 else 0

    # 36. rest_diff
    feats["rest_diff"] = h_rest - a_rest

    return pd.DataFrame([feats])


# Feature documentation for data sourcing in nba_full_predict.py
FEATURE_DATA_SOURCES = {
    # Feature name → (primary source, fallback, game_dict key needed)
    "lineup_value_diff": ("nba_team_enrichment.lineup_value", "0", "enrichment"),
    "win_pct_diff": ("ESPN records", "0.5 each", "home_wins/losses, away_wins/losses"),
    "scoring_hhi_diff": ("nba_team_enrichment.scoring_hhi", "0.15", "enrichment"),
    "espn_pregame_wp": ("ESPN pregame predictor", "0.5", "espn_pregame_wp"),
    "ceiling_diff": ("nba_team_enrichment.ceiling", "0", "enrichment"),
    "matchup_efg": ("enrichment eFG% + opp eFG%", "0", "enrichment"),
    "ml_implied_spread": ("moneyline implied prob → -8*log10(p/(1-p))", "0", "home_moneyline → espn_pregame_wp fallback"),
    "sharp_spread_signal": ("precomputed OR approx: -(h_sp_odds - a_sp_odds)*0.05", "0", "sharp_spread_signal OR home/away_spread_odds"),
    "efg_diff": ("enrichment eFG%", "~FG%", "enrichment or home/away_fgpct"),
    "opp_suppression_diff": ("home/away_opp_suppression raw col", "0", "home_opp_suppression, away_opp_suppression"),
    "net_rtg_diff": ("ESPN or enrichment", "0", "home_net_rtg, away_net_rtg"),
    "steals_to_diff": ("ESPN stats", "0", "home_steals/turnovers"),
    "threepct_diff": ("ESPN stats", "0", "home_threepct, away_threepct"),
    "b2b_diff": ("schedule", "0", "home_days_rest, away_days_rest"),
    "ftpct_diff": ("ESPN stats", "0", "home_ftpct, away_ftpct"),
    "ou_gap": ("PPG total - market total (positive=teams outscore market)", "0", "market_ou_total, home/away_ppg"),
    "roll_dreb_diff": ("nba_game_stats rolling", "0", "home_roll_dreb, away_roll_dreb"),
    "ts_regression_diff": ("enrichment TS%", "0", "enrichment"),
    "roll_paint_pts_diff": ("nba_game_stats rolling", "0", "home_roll_paint_pts"),
    "ref_home_whistle": ("nba_ref_profiles", "0", "ref_profile"),
    "opp_ppg_diff": ("ESPN/Supabase", "0", "home_opp_ppg, away_opp_ppg"),
    "roll_max_run_avg": ("nba_game_stats rolling — AVERAGE of both teams, not diff", "0", "home_roll_max_run, away_roll_max_run"),
    "away_is_public_team": ("static list", "0", "away_team_abbr"),
    "away_after_loss": ("last game result", "0", "away_last_result"),
    "games_last_14_diff": ("schedule data", "0", "home_games_last_14"),
    "h2h_total_games": ("nba_game_stats H2H lookup", "0", "h2h_total_games"),
    "three_pt_regression_diff": ("derived from threepct", "0", "home/away_threepct"),
    "games_diff": ("ESPN records", "0", "home/away wins+losses"),
    "ref_foul_proxy": ("nba_ref_profiles", "0", "ref_profile"),
    "roll_fast_break_diff": ("nba_game_stats rolling", "0", "home_roll_fast_break_pts, away_roll_fast_break_pts"),
    "crowd_pct": ("ESPN attendance/capacity", "0.9", "crowd_pct"),
    "matchup_to": ("derived from TO and PPG", "0", "home/away turnovers/ppg"),
    "overround": ("moneyline odds", "0", "home/away_moneyline"),
    "roll_ft_trip_rate_diff": ("nba_game_stats rolling", "0", "home_roll_ft_trip_rate"),
    "home_after_loss": ("last game result", "0", "home_last_result"),
    "rest_diff": ("schedule data", "0", "home/away_days_rest"),
    "spread_juice_imbalance": ("pickcenter homeTeamOdds.spreadOdds - awayTeamOdds.spreadOdds", "0", "home_spread_odds, away_spread_odds"),
    "vig_uncertainty": ("derived: overround - 0.045", "0", "home/away_moneyline"),
}
