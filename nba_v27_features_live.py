"""
nba_v27_features_live.py — Compute 69 backward-eliminated features from live game data.

Maps ESPN + Supabase + enrichment + referee data -> 69 features for prediction.
Used by nba_full_predict.py at inference time.

Each feature mirrors the EXACT formula from nba_build_features_v27.py (training builder).
"""
import numpy as np
import pandas as pd
from datetime import datetime

PUBLIC_TEAMS = {
    "LAL", "LAC", "GSW", "BOS", "NYK", "BKN", "MIA", "CHI",
    "DAL", "PHX", "PHI", "MIL", "OKC", "DEN", "CLE",
}
NBA_CONFERENCES = {
    "ATL":"E","BOS":"E","BKN":"E","CHA":"E","CHI":"E","CLE":"E",
    "DET":"E","IND":"E","MIA":"E","MIL":"E","NYK":"E","ORL":"E",
    "PHI":"E","TOR":"E","WAS":"E",
    "DAL":"W","DEN":"W","GSW":"W","HOU":"W","LAC":"W","LAL":"W",
    "MEM":"W","MIN":"W","NOP":"W","OKC":"W","PHX":"W","POR":"W",
    "SAC":"W","SAS":"W","UTA":"W",
}

FEATURES_69 = [
    "after_loss_either", "altitude_factor", "ato_ratio_diff",
    "away_is_public_team", "b2b_diff", "bimodal_diff", "ceiling_diff",
    "conference_game", "consistency_diff", "days_since_loss_diff",
    "efg_diff", "elo_diff", "espn_pregame_wp", "espn_pregame_wp_pbp",
    "floor_diff", "ftpct_diff", "games_diff", "games_last_14_diff",
    "h2h_avg_margin", "h2h_total_games", "home_b2b", "home_fav",
    "implied_prob_home", "is_early_season", "is_friday_sat",
    "is_revenge_home", "lineup_value_diff", "margin_accel_diff",
    "market_spread", "matchup_efg", "matchup_ft", "matchup_orb",
    "momentum_halflife_diff", "opp_suppression_diff", "ou_gap",
    "overround", "pace_control_diff", "pace_leverage", "post_allstar",
    "post_trade_deadline", "pyth_luck_diff", "pyth_residual_diff",
    "recovery_diff", "ref_foul_proxy", "ref_home_whistle", "ref_ou_bias",
    "ref_pace_impact", "reverse_line_movement", "roll_bench_pts_diff",
    "roll_ft_trip_rate_diff", "roll_max_run_avg", "roll_paint_fg_rate_diff",
    "roll_paint_pts_diff", "roll_q4_diff", "roll_three_fg_rate_diff",
    "score_kurtosis_diff", "scoring_entropy_diff", "scoring_hhi_diff",
    "sharp_spread_signal", "spread_juice_imbalance", "steals_to_diff",
    "three_pt_regression_diff", "three_value_diff", "threepct_diff",
    "ts_regression_diff", "turnovers_diff", "vig_uncertainty",
    "win_aging_diff", "win_pct_diff",
]


def _safe(val, default=0):
    if val is None: return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (ValueError, TypeError):
        return default


def _ml_to_prob(ml):
    ml = _safe(ml, 0)
    if ml < 0: return abs(ml) / (abs(ml) + 100)
    if ml > 0: return 100 / (ml + 100)
    return 0.5


def build_v27_features(game, enrichment=None, ref_profile=None, league_avg_ts=0.575):
    if enrichment is None: enrichment = {"home": {}, "away": {}}
    if ref_profile is None: ref_profile = {}
    he = enrichment.get("home", {}); ae = enrichment.get("away", {})

    def g(key, default=0): return _safe(game.get(key), default)
    def h_enr(key, default=0): return _safe(he.get(key), default)
    def a_enr(key, default=0): return _safe(ae.get(key), default)
    def ref(key, default=0): return _safe(ref_profile.get(key), default)

    h_ppg=g("home_ppg",110); a_ppg=g("away_ppg",110)
    h_fgpct=g("home_fgpct",0.46); a_fgpct=g("away_fgpct",0.46)
    h_3pct=g("home_threepct",0.365); a_3pct=g("away_threepct",0.365)
    h_ftpct=g("home_ftpct",0.77); a_ftpct=g("away_ftpct",0.77)
    h_steals=g("home_steals",7.5); a_steals=g("away_steals",7.5)
    h_to=g("home_turnovers",14); a_to=g("away_turnovers",14)
    h_wins=g("home_wins",20); h_losses=g("home_losses",20)
    a_wins=g("away_wins",20); a_losses=g("away_losses",20)
    h_rest=g("home_days_rest",2); a_rest=g("away_days_rest",2)
    market_spread=g("market_spread_home",0)
    home_abbr=game.get("home_team",game.get("home_team_abbr",""))
    away_abbr=game.get("away_team",game.get("away_team_abbr",""))
    game_date_str=game.get("game_date","")

    f = {}

    # === MARKET (9) ===
    f["market_spread"] = market_spread
    h_ml=g("home_moneyline",0) or g("home_ml",0) or g("home_ml_close",0)
    a_ml=g("away_moneyline",0) or g("away_ml",0) or g("away_ml_close",0)
    impl_h=_ml_to_prob(h_ml); impl_a=_ml_to_prob(a_ml)
    f["implied_prob_home"] = round(impl_h/max(impl_h+impl_a,0.01),4) if (h_ml and a_ml) else 0.5
    f["overround"] = round(impl_h+impl_a-1,4) if (h_ml and a_ml) else 0
    f["home_fav"] = 1 if market_spread < 0 else 0
    mkt_total=g("market_ou_total",0) or g("ou_total",0)
    f["ou_gap"] = (h_ppg+a_ppg)-mkt_total if mkt_total>0 else 0
    sm=g("_spread_move",0); so=g("spread_open",0); sc=g("spread_close",0)
    f["sharp_spread_signal"] = round(sm,2) if sm else (round((sc or market_spread)-so,2) if so else 0)
    hso=g("home_spread_odds",0); aso=g("away_spread_odds",0)
    f["spread_juice_imbalance"] = round(_ml_to_prob(hso)-_ml_to_prob(aso),4) if (hso and aso) else 0
    f["vig_uncertainty"] = round(f["overround"]-0.045,4)
    mm=g("_ml_move",0)
    f["reverse_line_movement"] = 1 if (sm and mm and sm*mm>0) else 0

    # === ESPN WP (2) ===
    f["espn_pregame_wp"] = g("espn_pregame_wp",0.5)
    f["espn_pregame_wp_pbp"] = g("espn_pregame_wp_pbp",0.5)

    # === TEAM STATS (11) ===
    f["efg_diff"] = round((h_fgpct+0.2*h_3pct)-(a_fgpct+0.2*a_3pct),4)
    f["ftpct_diff"] = round(h_ftpct-a_ftpct,4)
    f["threepct_diff"] = round(h_3pct-a_3pct,4)
    f["turnovers_diff"] = round(h_to-a_to,2)
    f["ato_ratio_diff"] = round(g("home_ato_ratio",1.8)-g("away_ato_ratio",1.8),4)
    f["steals_to_diff"] = round(h_steals/max(h_to,1)-a_steals/max(a_to,1),4)
    h_wp=h_wins/max(h_wins+h_losses,1); a_wp=a_wins/max(a_wins+a_losses,1)
    f["win_pct_diff"] = round(h_wp-a_wp,4)
    f["opp_suppression_diff"] = g("home_opp_suppression",0)-g("away_opp_suppression",0)
    h3r=g("home_three_fg_rate",0) or h_enr("three_fg_rate",0)
    a3r=g("away_three_fg_rate",0) or a_enr("three_fg_rate",0)
    f["three_value_diff"] = round(h3r*(h3r-0.20)-a3r*(a3r-0.20),4)
    h3reg=g("home_three_pt_regression",0) or (h_3pct-0.365)
    a3reg=g("away_three_pt_regression",0) or (a_3pct-0.365)
    f["three_pt_regression_diff"] = round(h3reg-a3reg,4)
    hts=h_enr("ts_pct",0) or g("home_ts_pct",0)
    ats=a_enr("ts_pct",0) or g("away_ts_pct",0)
    f["ts_regression_diff"] = round((hts-league_avg_ts)-(ats-league_avg_ts),4) if (hts>0 and ats>0) else 0

    # === PLAYER/LINEUP (3) ===
    f["lineup_value_diff"] = h_enr("lineup_value",0)-a_enr("lineup_value",0)
    f["scoring_hhi_diff"] = h_enr("scoring_hhi",0.15)-a_enr("scoring_hhi",0.15)
    f["scoring_entropy_diff"] = h_enr("scoring_entropy",0)-a_enr("scoring_entropy",0)

    # === ELO (1) ===
    f["elo_diff"] = g("elo_diff",0) or (g("home_elo",1500)-g("away_elo",1500))

    # === ENRICHMENT (16) ===
    f["ceiling_diff"] = h_enr("ceiling",0)-a_enr("ceiling",0)
    f["floor_diff"] = h_enr("floor",0)-a_enr("floor",0)
    f["consistency_diff"] = f["ceiling_diff"]-f["floor_diff"]
    f["bimodal_diff"] = h_enr("bimodal",0)-a_enr("bimodal",0)
    f["score_kurtosis_diff"] = h_enr("score_kurtosis",0)-a_enr("score_kurtosis",0)
    f["margin_accel_diff"] = h_enr("margin_accel",0)-a_enr("margin_accel",0)
    f["momentum_halflife_diff"] = h_enr("momentum_halflife",0)-a_enr("momentum_halflife",0)
    f["win_aging_diff"] = h_enr("win_aging",0)-a_enr("win_aging",0)
    f["pyth_residual_diff"] = h_enr("pyth_residual",0)-a_enr("pyth_residual",0)
    f["pyth_luck_diff"] = h_enr("pyth_luck",0)-a_enr("pyth_luck",0)
    hsv=h_enr("scoring_var",0) or g("home_scoring_var",0)
    asv=a_enr("scoring_var",0) or g("away_scoring_var",0)
    f["pace_control_diff"] = (1/max(hsv,1))-(1/max(asv,1)) if (hsv>0 or asv>0) else 0
    f["pace_leverage"] = (abs(g("home_opp_suppression",0))+abs(g("away_opp_suppression",0)))*0.01
    f["recovery_diff"] = h_enr("recovery_idx",0)-a_enr("recovery_idx",0)
    f["matchup_efg"] = g("home_three_fg_rate",0)-g("away_three_fg_rate",0)
    hfr=h_enr("ft_trip_rate",0) or g("home_fta_rate",0.28)
    afr=a_enr("ft_trip_rate",0) or g("away_fta_rate",0.28)
    f["matchup_ft"] = round(hfr-afr,4)
    horb=g("home_roll_oreb",0) or h_enr("oreb",0) or g("home_orb_pct",0.25)*40
    aorb=g("away_roll_oreb",0) or a_enr("oreb",0) or g("away_orb_pct",0.25)*40
    f["matchup_orb"] = round(horb-aorb,2)

    # === ROLLING PBP (7) ===
    # Builder reads component keys (home_roll_X, away_roll_X)
    # But get_rolling_diffs() puts pre-computed diffs in row — check both
    def _roll(h_key, a_key, diff_key):
        """Component diff if either side has data, else pre-computed diff."""
        h = g(h_key, 0); a = g(a_key, 0)
        if h != 0 or a != 0: return h - a
        return g(diff_key, 0)
    f["roll_q4_diff"] = _roll("home_roll_q4", "away_roll_q4", "roll_q4_scoring_diff")
    f["roll_paint_pts_diff"] = _roll("home_roll_paint_pts", "away_roll_paint_pts", "roll_paint_pts_diff")
    f["roll_bench_pts_diff"] = _roll("home_roll_bench_pts", "away_roll_bench_pts", "roll_bench_pts_diff")
    f["roll_ft_trip_rate_diff"] = _roll("home_roll_ft_trip_rate", "away_roll_ft_trip_rate", "roll_ft_trip_rate_diff")
    f["roll_three_fg_rate_diff"] = _roll("home_roll_three_fg_rate", "away_roll_three_fg_rate", "roll_three_fg_rate_diff")
    f["roll_paint_fg_rate_diff"] = _roll("home_roll_paint_fg_rate", "away_roll_paint_fg_rate", "roll_paint_fg_rate_diff")
    # Fallback: derive from already-computed roll_paint_pts_diff / avg_ppg
    # (nba_game_stats has no paint_fg_rate column, so derive from paint_pts)
    if f["roll_paint_fg_rate_diff"] == 0 and f.get("roll_paint_pts_diff", 0) != 0:
        f["roll_paint_fg_rate_diff"] = round(f["roll_paint_pts_diff"] / max((h_ppg + a_ppg) / 2, 80), 4)
    _rmr_h = g("home_roll_max_run", 0); _rmr_a = g("away_roll_max_run", 0)
    f["roll_max_run_avg"] = (_rmr_h + _rmr_a) / 2 if (_rmr_h or _rmr_a) else g("roll_max_run_avg", 0)

    # === SCHEDULE (10) ===
    hb2b=1 if h_rest==0 else 0; ab2b=1 if a_rest==0 else 0
    f["b2b_diff"]=hb2b-ab2b; f["home_b2b"]=hb2b

    # games_last_14_diff — ESPN lastFiveGames only returns 5, so games_14d caps at 5
    # Prefer games_14d (from ESPN schedule parser) over games_last_14 (from Supabase, often 0)
    _g14h = g("home_games_14d", 0)
    _g14a = g("away_games_14d", 0)
    if _g14h == 0: _g14h = g("home_games_last_14", 0)
    if _g14a == 0: _g14a = g("away_games_last_14", 0)
    # If still both same (e.g. both 5 from ESPN cap), check overrides
    if _g14h == _g14a and _g14h > 0:
        f["games_last_14_diff"] = g("games_last_14_diff", 0)  # use pre-computed override if available
    else:
        f["games_last_14_diff"] = _g14h - _g14a

    ht=h_wins+h_losses; at=a_wins+a_losses
    f["games_diff"]=ht-at

    # days_since_loss_diff — Supabase often stores defaults (30.0)
    # Derive from streak data which is always fresh from ESPN
    _dsl_h = g("home_days_since_loss", 0)
    _dsl_a = g("away_days_since_loss", 0)
    h_streak = g("home_streak", 0)
    a_streak = g("away_streak", 0)
    # Detect stale defaults: if both are identical non-zero (e.g. 30.0), or if streak contradicts
    _stale = (_dsl_h == _dsl_a and _dsl_h > 0) or \
             (h_streak < 0 and _dsl_h > 5) or \
             (a_streak < 0 and _dsl_a > 5)
    if _stale or (_dsl_h == 0 and _dsl_a == 0):
        # Derive from streak: negative streak = just lost (0 days), positive = streak * ~2.2 days
        _dsl_h = 0 if h_streak < 0 else max(h_streak * 2.2, 0)
        _dsl_a = 0 if a_streak < 0 else max(a_streak * 2.2, 0)
    f["days_since_loss_diff"] = round(_dsl_h - _dsl_a, 1)

    f["is_early_season"]=1 if ht<15 else 0
    try:
        dt=datetime.strptime(game_date_str,"%Y-%m-%d")
        f["is_friday_sat"]=1 if dt.weekday() in [4,5] else 0
        f["post_allstar"]=1 if (dt.month>2 or (dt.month==2 and dt.day>20)) else 0
        f["post_trade_deadline"]=1 if (dt.month>2 or (dt.month==2 and dt.day>6)) else 0
    except (ValueError,TypeError):
        f["is_friday_sat"]=0; f["post_allstar"]=0; f["post_trade_deadline"]=0
    f["altitude_factor"]=1 if str(home_abbr).upper()=="DEN" else 0

    # === H2H (4) ===
    f["h2h_total_games"]=g("h2h_total_games",0) or g("_h2h_n",0)
    f["h2h_avg_margin"]=g("h2h_avg_margin",0)
    f["is_revenge_home"]=g("is_revenge_home",0)
    hc=NBA_CONFERENCES.get(str(home_abbr).upper()); ac=NBA_CONFERENCES.get(str(away_abbr).upper())
    f["conference_game"]=1 if (hc and ac and hc==ac) else g("conference_game",0)

    # === SITUATIONAL (2) ===
    hal=g("home_after_loss",0); aal=g("away_after_loss",0)
    if hal==0:
        hsr=game.get("home_streak_raw","")
        if isinstance(hsr,str) and hsr.startswith("L"): hal=1
        elif g("home_last_result",0)==-1: hal=1
        elif g("home_streak",0)<0: hal=1  # negative streak = on losing streak
    if aal==0:
        asr=game.get("away_streak_raw","")
        if isinstance(asr,str) and asr.startswith("L"): aal=1
        elif g("away_last_result",0)==-1: aal=1
        elif g("away_streak",0)<0: aal=1
    f["after_loss_either"]=1 if (hal or aal) else g("after_loss_either",0)
    f["away_is_public_team"]=1 if str(away_abbr).upper() in PUBLIC_TEAMS else 0

    # === REFEREE (4) ===
    f["ref_home_whistle"]=ref("home_whistle",0)
    f["ref_foul_proxy"]=ref("foul_rate",0) or ref("home_whistle",0)
    # ref_ou_bias: refs with higher home_whistle tend to call more fouls → higher scoring
    # Approximate: ou_bias ≈ home_whistle * 2 (more fouls = more FTs = higher totals)
    f["ref_ou_bias"]=ref("ou_bias",0) or round(ref("home_whistle",0) * 2.0, 4)
    # ref_pace_impact: refs who call more fouls slow the game down slightly
    # Approximate: pace_impact ≈ -home_whistle * 0.5 (more whistles = slower pace)
    f["ref_pace_impact"]=ref("pace_impact",0) or round(ref("home_whistle",0) * -0.5, 4)

    # === VALIDATE ===
    for feat in FEATURES_69:
        if feat not in f: f[feat]=0

    return pd.DataFrame([f])
