import numpy as np, pandas as pd, traceback as _tb, shap, requests, time as _time
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, brier_score_loss
from db import sb_get, save_model, load_model
from config import SUPABASE_URL, SUPABASE_KEY
from ml_utils import HAS_XGB, _time_series_oof, StackedRegressor, StackedClassifier
if HAS_XGB:
    from xgboost import XGBRegressor, XGBClassifier
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Conference HCA mapping — used by ncaa_build_features R1 fix and _ncaa_backfill_heuristic
_NCAA_CONF_HCA = {
    "Big 12": 3.8, "Southeastern Conference": 3.7, "SEC": 3.7,
    "Big Ten": 3.6, "Big Ten Conference": 3.6,
    "Atlantic Coast Conference": 3.4, "ACC": 3.4,
    "Big East": 3.3, "Big East Conference": 3.3,
    "Pac-12": 3.0, "Pac-12 Conference": 3.0,
    "Mountain West Conference": 3.2, "Mountain West": 3.2,
    "American Athletic Conference": 3.0, "AAC": 3.0,
    "West Coast Conference": 2.8, "WCC": 2.8,
    "Atlantic 10 Conference": 2.7, "A-10": 2.7,
    "Missouri Valley Conference": 2.9, "MVC": 2.9,
}


def ncaa_build_features(df):
    df = df.copy()

    # ── F6 FIX: Normalize conference column to support both ESPN IDs and full names ──
    # Historical data may store ESPN numeric IDs ("8", "23"), current season stores full names.
    # Map IDs to names so _NCAA_CONF_HCA lookup works consistently.
    _ESPN_CONF_ID_TO_NAME = {
        "8": "Big 12", "23": "Southeastern Conference", "7": "Big Ten",
        "2": "Atlantic Coast Conference", "4": "Big East", "21": "Pac-12",
        "44": "Mountain West Conference", "62": "American Athletic Conference",
        "26": "West Coast Conference", "3": "Atlantic 10 Conference",
        "18": "Missouri Valley Conference", "40": "Sun Belt", "12": "Mid-American",
        "10": "Colonial Athletic Association", "22": "Ivy League",
        "1": "America East", "46": "ASUN", "5": "Big Sky", "6": "Big South",
        "9": "Big West", "11": "Conference USA", "13": "Horizon League",
        "14": "Metro Atlantic Athletic", "16": "MEAC", "17": "Mountain West",
        "19": "Northeast Conference", "20": "Ohio Valley", "24": "Southland",
        "25": "Southern Conference", "27": "Summit League", "28": "SWAC",
        "29": "WAC", "30": "Patriot League",
    }
    if "home_conference" in df.columns:
        df["home_conference"] = df["home_conference"].fillna("").astype(str).apply(
            lambda x: _ESPN_CONF_ID_TO_NAME.get(x.strip(), x.strip())
        )
    if "away_conference" in df.columns:
        df["away_conference"] = df["away_conference"].fillna("").astype(str).apply(
            lambda x: _ESPN_CONF_ID_TO_NAME.get(x.strip(), x.strip())
        )

    # ── Raw team stats (with defaults for missing data) ──
    raw_cols = {
        # v1 CORE
        "home_ppg": 75.0, "away_ppg": 75.0,
        "home_opp_ppg": 72.0, "away_opp_ppg": 72.0,
        "home_fgpct": 0.455, "away_fgpct": 0.455,
        "home_threepct": 0.340, "away_threepct": 0.340,
        "home_ftpct": 0.720, "away_ftpct": 0.720,
        "home_assists": 14.0, "away_assists": 14.0,
        "home_turnovers": 12.0, "away_turnovers": 12.0,
        "home_tempo": 68.0, "away_tempo": 68.0,
        "home_orb_pct": 0.28, "away_orb_pct": 0.28,
        "home_fta_rate": 0.34, "away_fta_rate": 0.34,
        "home_ato_ratio": 1.2, "away_ato_ratio": 1.2,
        "home_opp_fgpct": 0.430, "away_opp_fgpct": 0.430,
        "home_opp_threepct": 0.330, "away_opp_threepct": 0.330,
        "home_steals": 7.0, "away_steals": 7.0,
        "home_blocks": 3.5, "away_blocks": 3.5,
        "home_wins": 10, "away_wins": 10,
        "home_losses": 5, "away_losses": 5,
        "home_form": 0.0, "away_form": 0.0,
        "home_sos": 0.500, "away_sos": 0.500,
        "home_rank": 200, "away_rank": 200,
        "home_rest_days": 3, "away_rest_days": 3,
        # v18 P1-INJ: Injury columns
        "home_injury_penalty": 0.0, "away_injury_penalty": 0.0,
        "injury_diff": 0.0,
        "home_missing_starters": 0, "away_missing_starters": 0,
        # v18 P1-CTX: Tournament context columns
        "is_conference_tournament": 0, "is_ncaa_tournament": 0,
        "is_bubble_game": 0, "is_early_season": 0,
        "importance_multiplier": 1.0,
        "home_roll_star1_share": 0.25, "away_roll_star1_share": 0.25,
        "home_roll_top3_share": 0.65, "away_roll_top3_share": 0.65,
        "home_roll_bench_share": 0.20, "away_roll_bench_share": 0.20,
        "home_roll_bench_pts": 15.0, "away_roll_bench_pts": 15.0,
        # ── v3 ADVANCED SHOOTING ──
        "home_twopt_pct": 0.48, "away_twopt_pct": 0.48,
        "home_efg_pct": 0.50, "away_efg_pct": 0.50,
        "home_ts_pct": 0.53, "away_ts_pct": 0.53,
        "home_three_rate": 0.35, "away_three_rate": 0.35,
        "home_assist_rate": 0.55, "away_assist_rate": 0.55,
        "home_drb_pct": 0.70, "away_drb_pct": 0.70,
        "home_ppp": 1.0, "away_ppp": 1.0,
        # ── v3 OPPONENT FOUR FACTORS ──
        "home_opp_efg_pct": 0.50, "away_opp_efg_pct": 0.50,
        "home_opp_to_rate": 0.18, "away_opp_to_rate": 0.18,
        "home_opp_fta_rate": 0.30, "away_opp_fta_rate": 0.30,
        "home_opp_orb_pct": 0.28, "away_opp_orb_pct": 0.28,
        # ── v3 ANALYTICS ──
        "home_luck": 0.0, "away_luck": 0.0,
        "home_consistency": 15.0, "away_consistency": 15.0,
        "home_elo": 1500, "away_elo": 1500,
        "home_pyth_residual": 0.0, "away_pyth_residual": 0.0,
        "home_margin_trend": 0.0, "away_margin_trend": 0.0,
        "home_close_win_rate": 0.5, "away_close_win_rate": 0.5,
        # ── v3 NOVEL ──
        "home_eff_vol_ratio": 1.0, "away_eff_vol_ratio": 1.0,
        "home_ceiling": 15.0, "away_ceiling": 15.0,
        "home_floor": -10.0, "away_floor": -10.0,
        "home_recovery_idx": 0.0, "away_recovery_idx": 0.0,
        "home_is_after_loss": 0, "away_is_after_loss": 0,
        "home_opp_suppression": 0.0, "away_opp_suppression": 0.0,
        "home_concentration": 0.0, "away_concentration": 0.0,
        "home_blowout_asym": 0.0, "away_blowout_asym": 0.0,
        "home_margin_accel": 0.0, "away_margin_accel": 0.0,
        "home_wl_momentum": 0.0, "away_wl_momentum": 0.0,
        "home_clutch_over_exp": 0.0, "away_clutch_over_exp": 0.0,
        "home_def_stability": 10.0, "away_def_stability": 10.0,
        "home_fatigue_load": 0.0, "away_fatigue_load": 0.0,
        "home_info_gain": 0.0, "away_info_gain": 0.0,
        "home_regression_pressure": 0.0, "away_regression_pressure": 0.0,
        "home_pace_adj_margin": 0.0, "away_pace_adj_margin": 0.0,
        "home_ft_pressure": 0.0, "away_ft_pressure": 0.0,
        "home_transition_dep": 0.0, "away_transition_dep": 0.0,
        "home_margin_autocorr": 0.0, "away_margin_autocorr": 0.0,
        "home_opp_adj_form": 0.0, "away_opp_adj_form": 0.0,
        "home_scoring_entropy": 1.5, "away_scoring_entropy": 1.5,
        "home_run_vulnerability": 0.0, "away_run_vulnerability": 0.0,
        "home_anti_fragility": 0.0, "away_anti_fragility": 0.0,
        "home_sos_trajectory": 0.0, "away_sos_trajectory": 0.0,
        "home_margin_skew": 0.0, "away_margin_skew": 0.0,
        "home_bimodal": 0.0, "away_bimodal": 0.0,
        "home_pit_sos": 1500, "away_pit_sos": 1500,
        # ── v3 SCHEDULE ──
        "home_games_last_7": 2, "away_games_last_7": 2,
        "home_streak": 0, "away_streak": 0,
        "home_season_pct": 0.5, "away_season_pct": 0.5,
        # ── v3 ESPN EXTRAS ──
        "home_pts_off_to": 12.0, "away_pts_off_to": 12.0,
        "home_fastbreak_pts": 10.0, "away_fastbreak_pts": 10.0,
        "home_paint_pts": 30.0, "away_paint_pts": 30.0,
        "home_fouls": 17.0, "away_fouls": 17.0,
        # ── v3 CENTURY ──
        "home_scoring_source_entropy": 1.5, "away_scoring_source_entropy": 1.5,
        "home_ft_dependency": 0.20, "away_ft_dependency": 0.20,
        "home_three_value": 0.35, "away_three_value": 0.35,
        "home_steal_foul_ratio": 0.40, "away_steal_foul_ratio": 0.40,
        "home_block_foul_ratio": 0.20, "away_block_foul_ratio": 0.20,
        "home_def_versatility": 0.5, "away_def_versatility": 0.5,
        "home_to_conversion": 1.0, "away_to_conversion": 1.0,
        "home_fg_divergence": 0.0, "away_fg_divergence": 0.0,
        "home_three_divergence": 0.0, "away_three_divergence": 0.0,
        "home_ppp_divergence": 0.0, "away_ppp_divergence": 0.0,
        "home_def_improvement": 0.0, "away_def_improvement": 0.0,
        "home_home_margin": 0.0, "home_away_margin": 0.0,
        "away_home_margin": 0.0, "away_away_margin": 0.0,
        "home_rhythm_disruption": 0.0, "away_rhythm_disruption": 0.0,
        "home_overreaction": 0.0, "away_overreaction": 0.0,
        # ── v3 MATCHUP-LEVEL (pre-computed per-game in backfill) ──
        "matchup_efg": 0.0, "matchup_to": 0.0, "matchup_orb": 0.0, "matchup_ft": 0.0,
        "style_familiarity": 0.5, "pace_leverage": 0.0,
        "common_opp_diff": 0.0, "pace_control_diff": 0.0, "def_rest_advantage": 0.0,
        "is_revenge_game": 0, "revenge_margin": 0.0, "is_sandwich": 0, "is_letdown": 0, "is_midweek": 0,
        "spread_regime": 1,
        # ── v3 INTERACTION (pre-computed) ──
        "fatigue_x_quality": 0.0, "luck_x_spread": 0.0,
        "rest_x_defense": 0.0, "form_x_familiarity": 0.0, "consistency_x_spread": 0.0,
        # ═══ v19: ESPN COMPREHENSIVE EXTRACTION COLUMNS ═══
        # ESPN Odds (from ncaa_espn_extract_all.py — 64% coverage)
        "espn_spread": 0.0, "espn_over_under": 0.0,
        "espn_home_win_pct": 0.5, "espn_predictor_home_pct": 0.5,
        # PBP Half Scores
        "home_1h_score": 0.0, "away_1h_score": 0.0,
        "home_2h_score": 0.0, "away_2h_score": 0.0,
        "home_1h_margin": 0.0, "home_2h_margin": 0.0,
        # PBP Momentum
        "home_largest_run": 0.0, "away_largest_run": 0.0,
        "home_runs_8plus": 0, "away_runs_8plus": 0,
        "home_drought_count": 0, "away_drought_count": 0,
        "home_longest_drought_sec": 0.0, "away_longest_drought_sec": 0.0,
        # PBP Game Flow
        "lead_changes": 0, "ties": 0,
        "home_time_with_lead_pct": 0.5,
        "largest_lead_home": 0.0, "largest_lead_away": 0.0,
        # PBP Clutch
        "home_clutch_ftm": 0, "home_clutch_fta": 0,
        "away_clutch_ftm": 0, "away_clutch_fta": 0,
        "garbage_time_seconds": 0, "is_garbage_time_game": 0,
        # Player Features
        "home_star1_pts_share": 0.25, "away_star1_pts_share": 0.25,
        "home_top3_pts_share": 0.55, "away_top3_pts_share": 0.55,
        "home_minutes_hhi": 0.2, "away_minutes_hhi": 0.2,
        "home_bench_pts": 15.0, "away_bench_pts": 15.0,
        "home_bench_pts_share": 0.2, "away_bench_pts_share": 0.2,
        "home_players_used": 8, "away_players_used": 8,
        "home_starter_mins": 150.0, "away_starter_mins": 150.0,
        # Win Probability
        "halftime_home_win_prob": 0.5, "wp_volatility": 0.15, "wp_max_swing": 0.1,
        # ═══ v21: VENUE FEATURES (pre-game, no leakage) ═══
        "venue_capacity": 8000, "venue_indoor": 1, "attendance": 0,
        # ═══ v21: ROLLING TEAM TENDENCY FEATURES (from prior games, no leakage) ═══
        # These are computed by ncaa_rolling_tendencies.py and stored in Supabase.
        # PBP tendencies (rolling avg from prior N games)
        "home_roll_largest_run": 8.0, "away_roll_largest_run": 8.0,
        "home_roll_drought_rate": 1.5, "away_roll_drought_rate": 1.5,
        "home_roll_lead_changes": 8.0, "away_roll_lead_changes": 8.0,
        "home_roll_time_with_lead_pct": 0.5, "away_roll_time_with_lead_pct": 0.5,
        # Player dependency tendencies (rolling avg)
        "home_roll_star1_share": 0.25, "away_roll_star1_share": 0.25,
        "home_roll_top3_share": 0.55, "away_roll_top3_share": 0.55,
        "home_roll_bench_share": 0.20, "away_roll_bench_share": 0.20,
        "home_roll_hhi": 0.20, "away_roll_hhi": 0.20,
        "home_roll_players_used": 8.0, "away_roll_players_used": 8.0,
        # Clutch tendencies (rolling avg)
        "home_roll_clutch_ft_pct": 0.70, "away_roll_clutch_ft_pct": 0.70,
        # Garbage time tendency (rolling avg)
        "home_roll_garbage_pct": 0.15, "away_roll_garbage_pct": 0.15,
        # ATS tendency (rolling record against the spread)
        "home_roll_ats_pct": 0.50, "away_roll_ats_pct": 0.50,
        "home_roll_ats_n": 0, "away_roll_ats_n": 0,
        "home_roll_ats_margin": 0.0, "away_roll_ats_margin": 0.0,
    }
    for col, default in raw_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # ── AUDIT P1: Flag potentially leaked ratings ──
    # If rating_synced_at > 24h after game_date, adj_em may contain post-game data.
    if "rating_synced_at" in df.columns and "game_date" in df.columns:
        try:
            synced = pd.to_datetime(df["rating_synced_at"], errors="coerce")
            game_dt = pd.to_datetime(df["game_date"], errors="coerce")
            df["rating_leak_flag"] = ((synced - game_dt).dt.total_seconds() > 86400).astype(int)
            n_leaked = int(df["rating_leak_flag"].sum())
            if n_leaked > 0:
                print(f"  ⚠️ AUDIT: {n_leaked}/{len(df)} rows have ratings synced >24h after game date")
        except:
            df["rating_leak_flag"] = 0
    else:
        df["rating_leak_flag"] = 0

    # ── R1 FIX: Decompose adj_em_diff into neutral component + HCA component ──
    # The raw adj_em_diff contains HCA baked in (from home PPG). Separate them
    # so the ML can learn their independent weights instead of double-counting.
    raw_em_diff = df["home_adj_em"].fillna(0) - df["away_adj_em"].fillna(0)
    # Estimate HCA component: conference-based HCA / tempo * 100 gives per-100-poss effect
    hca_component = df.apply(
        lambda r: 0 if r.get("neutral_site", False) else _NCAA_CONF_HCA.get(
            r.get("home_conference", ""), 3.0
        ) * 0.5, axis=1  # HCA split across both teams, so ~0.5 on each side
    ) if "home_conference" in df.columns else pd.Series(1.5, index=df.index)
    df["neutral_em_diff"] = raw_em_diff - hca_component  # R1: HCA-stripped efficiency gap
    df["hca_pts"]         = hca_component                  # R1: separate HCA signal
    df["neutral"]         = df["neutral_site"].fillna(False).astype(int)

    # ── R2 FIX: Re-introduce heuristic win probability (capped) ──
    # AMPLIFICATION FIX: Previous cap [0.15, 0.85] was too wide — the ML model
    # learned to amplify the heuristic signal since win_pct_home already encodes
    # the same information as neutral_em_diff, ppg_diff, rank_diff, etc.
    # Tightened to [0.35, 0.65] so it provides only a weak directional nudge
    # without dominating the prediction or double-counting raw features.
    if "win_pct_home" in df.columns:
        df["heur_win_prob_capped"] = df["win_pct_home"].fillna(0.5).clip(0.35, 0.65)
    else:
        df["heur_win_prob_capped"] = 0.5

    # ── Differential features ──
    df["ppg_diff"]       = df["home_ppg"] - df["away_ppg"]
    df["opp_ppg_diff"]   = df["home_opp_ppg"] - df["away_opp_ppg"]
    df["fgpct_diff"]     = df["home_fgpct"] - df["away_fgpct"]
    df["threepct_diff"]  = df["home_threepct"] - df["away_threepct"]
    df["tempo_avg"]      = (df["home_tempo"] + df["away_tempo"]) / 2
    df["orb_pct_diff"]   = df["home_orb_pct"] - df["away_orb_pct"]
    df["fta_rate_diff"]  = df["home_fta_rate"] - df["away_fta_rate"]
    df["ato_diff"]       = df["home_ato_ratio"] - df["away_ato_ratio"]
    df["def_fgpct_diff"] = df["home_opp_fgpct"] - df["away_opp_fgpct"]
    df["steals_diff"]    = df["home_steals"] - df["away_steals"]
    df["blocks_diff"]    = df["home_blocks"] - df["away_blocks"]
    df["sos_diff"]       = df["home_sos"] - df["away_sos"]
    df["form_diff"]      = df["home_form"] - df["away_form"]
    # AMPLIFICATION FIX: Unranked teams default to rank=200, creating extreme
    # rank_diff values (e.g., -187 for #13 vs unranked) that GBM overfits on.
    # Cap at 50 before differencing — beyond rank 50, the marginal predictive
    # value of rank is negligible, but the raw number creates outlier inputs.
    df["home_rank_capped"] = df["home_rank"].clip(upper=50)
    df["away_rank_capped"] = df["away_rank"].clip(upper=50)
    df["rank_diff"]      = df["away_rank_capped"] - df["home_rank_capped"]
    df["win_pct_diff"]   = (df["home_wins"] / (df["home_wins"] + df["home_losses"]).clip(1)) - \
                           (df["away_wins"] / (df["away_wins"] + df["away_losses"]).clip(1))

    # F11: Turnover margin differential
    df["to_margin_diff"]    = df["away_turnovers"] - df["home_turnovers"]
    df["steals_to_ratio_h"] = df["home_steals"] / df["home_turnovers"].clip(0.5)
    df["steals_to_ratio_a"] = df["away_steals"] / df["away_turnovers"].clip(0.5)
    df["steals_to_diff"]    = df["steals_to_ratio_h"] - df["steals_to_ratio_a"]

    # Ranking context (use raw ranks for threshold checks, capped for differentials)
    df["is_ranked_game"] = ((df["home_rank"] <= 25) | (df["away_rank"] <= 25)).astype(int)
    df["is_top_matchup"] = ((df["home_rank"] <= 25) & (df["away_rank"] <= 25)).astype(int)

    # R5: Rest days (will be non-default only after ncaaSync wiring)
    df["rest_diff"]  = df["home_rest_days"] - df["away_rest_days"]
    df["either_b2b"] = ((df["home_rest_days"] <= 1) | (df["away_rest_days"] <= 1)).astype(int)

    # ── R3 FIX: Conference game flag + season phase ──
    if "home_conference" in df.columns and "away_conference" in df.columns:
        df["is_conf_game"] = (df["home_conference"].fillna("") == df["away_conference"].fillna("")).astype(int)
        # Filter out cases where both are empty string (missing data)
        df.loc[(df["home_conference"].fillna("") == "") | (df["away_conference"].fillna("") == ""), "is_conf_game"] = 0
    else:
        df["is_conf_game"] = 0

    if "game_date" in df.columns:
        gd = pd.to_datetime(df["game_date"], errors="coerce")
        # Season runs Nov 1 → early April (~155 days). Map to 0.0→1.0
        # Day of year: Nov 1 ≈ 305, April 7 ≈ 97 (next year)
        day_of_year = gd.dt.dayofyear.fillna(60)
        # Normalize: Nov=0.0, Dec=0.2, Jan=0.4, Feb=0.6, Mar=0.8, Apr=1.0
        df["season_phase"] = day_of_year.apply(
            lambda d: (d - 305) / 155 if d >= 305 else (d + 60) / 155
        ).clip(0.0, 1.0)
    else:
        df["season_phase"] = 0.5

    # ── AUDIT P4: Interaction features REMOVED ──
    # ppg_x_sos, em_x_conf had VIF > 10 with component features.
    # Keeping components only reduces multicollinearity.

    # ── v18 P1-INJ: Injury features ──
    df["home_injury_penalty"] = pd.to_numeric(df["home_injury_penalty"], errors="coerce").fillna(0)
    df["away_injury_penalty"] = pd.to_numeric(df["away_injury_penalty"], errors="coerce").fillna(0)
    df["injury_diff"] = df["home_injury_penalty"] - df["away_injury_penalty"]
    df["home_missing_starters"] = pd.to_numeric(df["home_missing_starters"], errors="coerce").fillna(0)
    df["away_missing_starters"] = pd.to_numeric(df["away_missing_starters"], errors="coerce").fillna(0)
    df["starters_diff"] = df["home_missing_starters"] - df["away_missing_starters"]
    df["any_injury_flag"] = ((df["home_missing_starters"] > 0) | (df["away_missing_starters"] > 0)).astype(int)
    # injury_x_em REMOVED (AUDIT P4) — correlated with injury_diff and neutral_em_diff

    # ── v18 P1-CTX: Tournament context features ──
    for _bc in ["is_conference_tournament", "is_ncaa_tournament", "is_bubble_game", "is_early_season"]:
        if _bc in df.columns:
            df[_bc] = df[_bc].map({True: 1, False: 0, "true": 1, "false": 0, 1: 1, 0: 0}).fillna(0).astype(int)
        else:
            df[_bc] = 0
    df["is_conf_tourney"] = df["is_conference_tournament"]
    df["is_ncaa_tourney"] = df["is_ncaa_tournament"]
    df["is_bubble"] = df["is_bubble_game"]
    df["is_early"] = df["is_early_season"]
    df["importance"] = pd.to_numeric(df["importance_multiplier"], errors="coerce").fillna(1.0)

    # ── Unified Market Features ──────────────────────────────────
    # ESPN odds (DraftKings via ESPN, ~77% historical coverage) is preferred.
    # Odds API (market_spread_home, ~4% historical coverage) is the fallback.
    # These are the SAME signal — real Vegas lines — just different data sources.
    _espn_sp = pd.to_numeric(df["espn_spread"] if "espn_spread" in df.columns else pd.Series(dtype=float), errors="coerce")
    _odds_sp = pd.to_numeric(df["market_spread_home"] if "market_spread_home" in df.columns else pd.Series(dtype=float), errors="coerce")
    _espn_ou = pd.to_numeric(df["espn_over_under"] if "espn_over_under" in df.columns else pd.Series(dtype=float), errors="coerce")
    _odds_ou = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else (df["ou_total"] if "ou_total" in df.columns else pd.Series(dtype=float)), errors="coerce"
    )
    _has_espn = _espn_sp.notna() & (_espn_sp != 0)
    _has_odds = _odds_sp.notna() & (_odds_sp != 0)

    # Prefer ESPN, fall back to Odds API, else 0
    df["mkt_spread"] = np.where(_has_espn, _espn_sp,
                       np.where(_has_odds, _odds_sp, 0)).astype(float)
    df["mkt_total"]  = np.where(_espn_ou.notna() & (_espn_ou != 0), _espn_ou,
                       np.where(_odds_ou.notna() & (_odds_ou != 0), _odds_ou, 0)).astype(float)
    df["has_mkt"]    = (_has_espn | _has_odds).astype(int)

    _ncaa_pred_spread = pd.to_numeric(df["spread_home"] if "spread_home" in df.columns else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    # BUGFIX 1: Sign alignment — model spread is +positive when home wins (e.g. +7.8),
    # but market_spread_home is -negative when home is favored (e.g. -7.5, Vegas convention).
    # Fixed: spread_vs_market = 7.8 - (+7.5) = +0.3 → correct small model disagreement
    # BUGFIX 2: Zero out when no market line (has_mkt=0) to prevent pred_spread leaking in.
    df["mkt_spread_vs_model"] = (_ncaa_pred_spread + df["mkt_spread"]) * df["has_mkt"]
    # tourney_x_em, early_x_form REMOVED (AUDIT P4) — correlated with components

    # ── v25 AUDIT: Compute interaction features inline (were pre-computed, broken) ──
    _abs_spread = df["mkt_spread"].abs()
    _h_consistency = pd.to_numeric(df.get("home_consistency", 15.0), errors="coerce").fillna(15.0).clip(lower=0.1)
    _h_luck = pd.to_numeric(df.get("home_luck", 0), errors="coerce").fillna(0)
    # consistency_x_spread: was only using market_spread_home (12%), now uses unified mkt_spread
    df["consistency_x_spread"] = _abs_spread / _h_consistency
    # luck_x_spread: same fix — home_luck * abs(unified mkt_spread)
    df["luck_x_spread"] = _h_luck * _abs_spread
    # spread_regime: buckets abs(spread) at 3.5/7.5/12.5/18.5
    # 0=no spread or <3.5, 1=3.5-7, 2=7.5-12, 3=12.5-18, 4=18.5+
    df["spread_regime"] = np.where(_abs_spread < 3.5, 0,
                          np.where(_abs_spread < 7.5, 1,
                          np.where(_abs_spread < 12.5, 2,
                          np.where(_abs_spread < 18.5, 3, 4)))).astype(int)

    # ══════════════════════════════════════════════════════════════
    # v3 FEATURE COMPUTATIONS — new differentials from PIT backfill
    # ══════════════════════════════════════════════════════════════

    # Elo
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Advanced shooting differentials
    df["efg_diff"] = df["home_efg_pct"] - df["away_efg_pct"]
    df["ts_diff"] = df["home_ts_pct"] - df["away_ts_pct"]
    df["twopt_diff"] = df["home_twopt_pct"] - df["away_twopt_pct"]
    df["three_rate_diff"] = df["home_three_rate"] - df["away_three_rate"]
    df["assist_rate_diff"] = df["home_assist_rate"] - df["away_assist_rate"]
    df["drb_pct_diff"] = df["home_drb_pct"] - df["away_drb_pct"]
    df["ppp_diff"] = df["home_ppp"] - df["away_ppp"]

    # Opponent Four Factors differentials
    df["opp_efg_diff"] = df["home_opp_efg_pct"] - df["away_opp_efg_pct"]
    df["opp_to_rate_diff"] = df["home_opp_to_rate"] - df["away_opp_to_rate"]
    df["opp_fta_rate_diff"] = df["home_opp_fta_rate"] - df["away_opp_fta_rate"]
    df["opp_orb_pct_diff"] = df["home_opp_orb_pct"] - df["away_opp_orb_pct"]

    # KenPom-style analytics
    df["luck_diff"] = df["home_luck"] - df["away_luck"]
    df["consistency_diff"] = df["home_consistency"] - df["away_consistency"]
    df["pyth_residual_diff"] = df["home_pyth_residual"] - df["away_pyth_residual"]

    # Momentum / Form
    df["margin_trend_diff"] = df["home_margin_trend"] - df["away_margin_trend"]
    df["margin_accel_diff"] = df["home_margin_accel"] - df["away_margin_accel"]
    df["opp_adj_form_diff"] = df["home_opp_adj_form"] - df["away_opp_adj_form"]
    df["wl_momentum_diff"] = df["home_wl_momentum"] - df["away_wl_momentum"]
    df["recovery_diff"] = df["home_recovery_idx"] - df["away_recovery_idx"]
    df["after_loss_either"] = ((df["home_is_after_loss"] == 1) | (df["away_is_after_loss"] == 1)).astype(int)

    # Volatility / Distribution
    df["eff_vol_diff"] = df["home_eff_vol_ratio"] - df["away_eff_vol_ratio"]
    df["ceiling_diff"] = df["home_ceiling"] - df["away_ceiling"]
    df["floor_diff"] = df["home_floor"] - df["away_floor"]
    df["margin_skew_diff"] = df["home_margin_skew"] - df["away_margin_skew"]
    df["scoring_entropy_diff"] = df["home_scoring_entropy"] - df["away_scoring_entropy"]
    df["bimodal_diff"] = df["home_bimodal"] - df["away_bimodal"]

    # Defensive profile
    df["def_stability_diff"] = df["home_def_stability"] - df["away_def_stability"]
    df["opp_suppression_diff"] = df["home_opp_suppression"] - df["away_opp_suppression"]
    df["def_versatility_diff"] = df["home_def_versatility"] - df["away_def_versatility"]
    df["steal_foul_diff"] = df["home_steal_foul_ratio"] - df["away_steal_foul_ratio"]
    df["block_foul_diff"] = df["home_block_foul_ratio"] - df["away_block_foul_ratio"]

    # Transition / Paint
    df["transition_dep_diff"] = df["home_transition_dep"] - df["away_transition_dep"]
    df["paint_pts_diff"] = df["home_paint_pts"] - df["away_paint_pts"]
    df["pts_off_to_diff"] = df["home_pts_off_to"] - df["away_pts_off_to"]
    df["fastbreak_diff"] = df["home_fastbreak_pts"] - df["away_fastbreak_pts"]

    # Schedule / Fatigue
    df["fatigue_diff"] = df["home_fatigue_load"] - df["away_fatigue_load"]
    df["games7_diff"] = df["home_games_last_7"] - df["away_games_last_7"]
    df["streak_diff"] = df["home_streak"] - df["away_streak"]
    df["season_pct_avg"] = (df["home_season_pct"] + df["away_season_pct"]) / 2

    # Regression / Information
    df["regression_diff"] = df["home_regression_pressure"] - df["away_regression_pressure"]
    df["info_gain_diff"] = df["home_info_gain"] - df["away_info_gain"]
    df["overreaction_diff"] = df["home_overreaction"] - df["away_overreaction"]

    # Scoring source
    df["scoring_source_entropy_diff"] = df["home_scoring_source_entropy"] - df["away_scoring_source_entropy"]
    df["ft_dependency_diff"] = df["home_ft_dependency"] - df["away_ft_dependency"]
    df["three_value_diff"] = df["home_three_value"] - df["away_three_value"]
    df["concentration_diff"] = df["home_concentration"] - df["away_concentration"]
    df["to_conversion_diff"] = df["home_to_conversion"] - df["away_to_conversion"]

    # Divergence (hot/cold)
    df["fg_divergence_diff"] = df["home_fg_divergence"] - df["away_fg_divergence"]
    df["three_divergence_diff"] = df["home_three_divergence"] - df["away_three_divergence"]
    df["ppp_divergence_diff"] = df["home_ppp_divergence"] - df["away_ppp_divergence"]

    # Pace-adjusted / ELO SOS / Venue
    df["pace_adj_margin_diff"] = df["home_pace_adj_margin"] - df["away_pace_adj_margin"]
    df["pit_sos_diff"] = df["home_pit_sos"] - df["away_pit_sos"]
    df["venue_advantage"] = df["home_home_margin"] - df["away_away_margin"]
    df["rhythm_disruption_diff"] = df["home_rhythm_disruption"] - df["away_rhythm_disruption"]
    df["def_improvement_diff"] = df["home_def_improvement"] - df["away_def_improvement"]

    # ═══════════════════════════════════════════════════════════════
    # v19: ESPN COMPREHENSIVE EXTRACTION FEATURES
    # ═══════════════════════════════════════════════════════════════

    # ── Market Win Probability Edge (v25 audit fix) ──
    # REPLACED espn_wp_edge: ESPN data was bad (in-game snapshots, not pre-game).
    # Cascade: DK moneyline → OA moneyline → spread-derived probability
    # ML conversion: fav(-): -ML/(-ML+100), dog(+): 100/(ML+100), then remove vig
    # Spread conversion: 1 / (1 + 10^(spread/16))
    _dk_ml_h = pd.to_numeric(df.get("dk_ml_home_close", 0), errors="coerce").fillna(0)
    _dk_ml_a = pd.to_numeric(df.get("dk_ml_away_close", 0), errors="coerce").fillna(0)
    _oa_ml_h = pd.to_numeric(df.get("odds_api_ml_home_close", 0), errors="coerce").fillna(0)
    _oa_ml_a = pd.to_numeric(df.get("odds_api_ml_away_close", 0), errors="coerce").fillna(0)
    # Pick best ML source: DK > OA
    _ml_h = np.where(_dk_ml_h != 0, _dk_ml_h, _oa_ml_h)
    _ml_a = np.where(_dk_ml_a != 0, _dk_ml_a, _oa_ml_a)
    _has_ml = (_ml_h != 0) & (_ml_a != 0)
    # Convert to implied probabilities
    _imp_h = np.where(_ml_h > 0, 100.0 / (_ml_h + 100), np.where(_ml_h < 0, -_ml_h / (-_ml_h + 100), 0.5))
    _imp_a = np.where(_ml_a > 0, 100.0 / (_ml_a + 100), np.where(_ml_a < 0, -_ml_a / (-_ml_a + 100), 0.5))
    _vig = np.clip(_imp_h + _imp_a, 0.01, 3.0)
    _ml_wp = _imp_h / _vig  # vig-removed true probability
    # Spread-derived probability as fallback (SIGMA=16 for college basketball)
    _mkt_sp = df["mkt_spread"].values.astype(float)
    _spread_wp = 1.0 / (1.0 + np.power(10.0, _mkt_sp / 16.0))
    _has_spread = _mkt_sp != 0
    # Cascade: moneyline (best) → spread-derived (fallback)
    _best_wp = np.where(_has_ml, _ml_wp, np.where(_has_spread, _spread_wp, 0.5))
    df["market_wp_edge"] = _best_wp - 0.5

    # ── PBP Half Split Features ──
    df["half_margin_swing"] = df["home_2h_margin"] - df["home_1h_margin"]
    df["largest_lead_diff"] = df["largest_lead_home"] - df["largest_lead_away"]

    # ── PBP Momentum Features ──
    df["run_diff"] = df["home_largest_run"] - df["away_largest_run"]
    df["drought_diff"] = df["home_drought_count"] - df["away_drought_count"]

    # ── PBP Clutch Features ──
    df["home_clutch_ft_pct"] = (df["home_clutch_ftm"] / df["home_clutch_fta"].clip(1)).fillna(0.7)
    df["away_clutch_ft_pct"] = (df["away_clutch_ftm"] / df["away_clutch_fta"].clip(1)).fillna(0.7)
    df["clutch_ft_diff"] = df["home_clutch_ft_pct"] - df["away_clutch_ft_pct"]

    # ── Player Dependency Features ──
    df["star1_share_diff"] = df["home_star1_pts_share"] - df["away_star1_pts_share"]
    df["top3_share_diff"] = df["home_top3_pts_share"] - df["away_top3_pts_share"]
    df["minutes_hhi_diff"] = df["home_minutes_hhi"] - df["away_minutes_hhi"]
    df["bench_pts_share_diff"] = df["home_bench_pts_share"] - df["away_bench_pts_share"]
    df["players_used_diff"] = df["home_players_used"] - df["away_players_used"]
    df["starter_mins_diff"] = df["home_starter_mins"] - df["away_starter_mins"]

    # ── Win Probability Features ──
    df["halftime_wp_edge"] = df["halftime_home_win_prob"] - 0.5

    # ═══════════════════════════════════════════════════════════════
    # v21: VENUE FEATURES (pre-game facts, zero leakage)
    # ═══════════════════════════════════════════════════════════════

    # Crowd factor: attendance as % of venue capacity (proxy for crowd energy)
    # High values (>0.90) = packed arena = stronger HCA; low (<0.50) = weak crowd
    _att = df["attendance"]
    _has_att = (_att > 0).astype(int)
    # Normalize attendance by league median (~2400) since venue_capacity is 0% populated
    # Result: 0.5 = small crowd, 1.0 = typical, 2.0+ = packed major arena
    df["crowd_pct"] = (_att / 2400).clip(0, 5.0) * _has_att
    df["has_crowd_data"] = _has_att

    # Venue capacity tier: small gym (<5K) vs mid (5-10K) vs large (>10K)
    # Normalized so model can learn venue size effect on HCA
    df["venue_size_norm"] = (df["venue_capacity"].clip(lower=1000) / 15000).clip(0, 2.0)

    # ═══════════════════════════════════════════════════════════════
    # v21: ROLLING TEAM TENDENCY FEATURES (from prior games, no leakage)
    # ═══════════════════════════════════════════════════════════════
    # These columns are pre-computed by ncaa_rolling_tendencies.py
    # and stored in Supabase. They represent each team's behavioral
    # profile from their PRIOR games — NOT from the current game.

    # PBP tendencies
    df["roll_run_diff"] = df["home_roll_largest_run"] - df["away_roll_largest_run"]
    df["roll_drought_diff"] = df["home_roll_drought_rate"] - df["away_roll_drought_rate"]
    df["roll_lead_change_avg"] = (df["home_roll_lead_changes"] + df["away_roll_lead_changes"]) / 2
    df["roll_dominance_diff"] = df["home_roll_time_with_lead_pct"] - df["away_roll_time_with_lead_pct"]

    # Player dependency tendencies
    df["roll_star_dep_diff"] = df["home_roll_star1_share"] - df["away_roll_star1_share"]
    df["roll_top3_dep_diff"] = df["home_roll_top3_share"] - df["away_roll_top3_share"]
    df["roll_bench_diff"] = df["home_roll_bench_share"] - df["away_roll_bench_share"]
    df["roll_rotation_diff"] = df["home_roll_players_used"] - df["away_roll_players_used"]
    df["roll_hhi_diff"] = df["home_roll_hhi"] - df["away_roll_hhi"]

    # Clutch tendency
    df["roll_clutch_ft_diff"] = df["home_roll_clutch_ft_pct"] - df["away_roll_clutch_ft_pct"]

    # Garbage time tendency (teams that frequently blow out opponents)
    df["roll_garbage_diff"] = df["home_roll_garbage_pct"] - df["away_roll_garbage_pct"]

    # ATS tendency (market mispricing signal)
    # Teams that consistently cover → market undervalues them
    # Teams that consistently fail to cover → market overvalues them
    df["roll_ats_diff"] = df["home_roll_ats_pct"] - df["away_roll_ats_pct"]
    df["roll_ats_margin_diff"] = df["home_roll_ats_margin"] - df["away_roll_ats_margin"]
    # Flag: both teams have enough ATS data for signal to be meaningful
    df["has_ats_data"] = ((df["home_roll_ats_n"] >= 3) & (df["away_roll_ats_n"] >= 3)).astype(int)
    # Gated ATS: zero out when not enough data
    df["roll_ats_diff_gated"] = df["roll_ats_diff"] * df["has_ats_data"]
    df["roll_ats_margin_gated"] = df["roll_ats_margin_diff"] * df["has_ats_data"]

    # Rolling player features (legitimate pre-game signal)
    df["roll_star1_share_diff"] = df["home_roll_star1_share"] - df["away_roll_star1_share"]
    df["roll_top3_share_diff"] = df["home_roll_top3_share"] - df["away_roll_top3_share"]
    df["roll_bench_share_diff"] = df["home_roll_bench_share"] - df["away_roll_bench_share"]
    df["roll_bench_pts_diff"] = df["home_roll_bench_pts"] - df["away_roll_bench_pts"]

    # Referee crew features (from prebuilt profiles)
    _ref_profiles = getattr(ncaa_build_features, "_ref_profiles", {})
    _ref_ou = []
    _ref_hw = []
    _ref_fr = []
    _ref_pace = []
    for _, _row in df.iterrows():
        _ou, _hw, _fr, _pa = [], [], [], []
        for _rc in ["referee_1", "referee_2", "referee_3"]:
            _name = str(_row.get(_rc, "")).strip()
            if _name and _name in _ref_profiles:
                _p = _ref_profiles[_name]
                _ou.append(_p.get("ou_bias", 0))
                _hw.append(_p.get("home_whistle", 0))
                _fr.append(_p.get("foul_rate", 0))
                _pa.append(_p.get("pace_impact", 145))
        _ref_ou.append(float(np.mean(_ou)) if _ou else 0.0)
        _ref_hw.append(float(np.mean(_hw)) if _hw else 0.0)
        _ref_fr.append(float(np.mean(_fr)) if _fr else 0.0)
        _ref_pace.append(float(np.mean(_pa)) - 145.0 if _pa else 0.0)
    df["ref_ou_bias"] = _ref_ou
    df["ref_home_whistle"] = _ref_hw
    df["ref_foul_rate"] = _ref_fr
    df["ref_pace_impact"] = _ref_pace
    df["has_ref_data"] = (pd.Series(_ref_ou) != 0.0).astype(int).values


    # ── Orphaned features (were in Supabase but not in model) ──
    df["adj_oe_diff"] = df["home_adj_oe"] - df["away_adj_oe"]
    df["adj_de_diff"] = df["away_adj_de"] - df["home_adj_de"]  # flipped: lower D is better
    df["scoring_var_diff"] = df["home_scoring_var"] - df["away_scoring_var"]
    df["score_kurtosis_diff"] = df["home_score_kurtosis"] - df["away_score_kurtosis"]
    df["clutch_ratio_diff"] = df["home_clutch_ratio"] - df["away_clutch_ratio"]
    df["garbage_adj_ppp_diff"] = df["home_garbage_adj_ppp"] - df["away_garbage_adj_ppp"]
    df["days_since_loss_diff"] = df["home_days_since_loss"] - df["away_days_since_loss"]
    df["games_since_blowout_diff"] = df["home_games_since_blowout_loss"] - df["away_games_since_blowout_loss"]
    df["games_last_14_diff"] = df["home_games_last_14"] - df["away_games_last_14"]
    df["rest_effect_diff"] = df["home_rest_effect"] - df["away_rest_effect"]
    df["momentum_halflife_diff"] = df["home_momentum_halflife"] - df["away_momentum_halflife"]
    df["win_aging_diff"] = df["home_win_aging"] - df["away_win_aging"]
    df["centrality_diff"] = df["home_centrality"] - df["away_centrality"]
    df["dow_effect_diff"] = df["home_dow_effect"] - df["away_dow_effect"]
    df["conf_balance_diff"] = df["home_conf_balance"] - df["away_conf_balance"]
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
    # ESPN moneyline edge: convert to implied prob delta vs model
    _espn_ml_h = pd.to_numeric(df["espn_ml_home"], errors="coerce").fillna(0)
    _espn_ml_a = pd.to_numeric(df["espn_ml_away"], errors="coerce").fillna(0)
    _has_espn_ml = (_espn_ml_h != 0).astype(float)
    _espn_imp_h = np.where(_espn_ml_h > 0, 100 / (_espn_ml_h + 100), -_espn_ml_h / (-_espn_ml_h + 100))
    _espn_imp_a = np.where(_espn_ml_a > 0, 100 / (_espn_ml_a + 100), -_espn_ml_a / (-_espn_ml_a + 100))
    _espn_vig_total = _espn_imp_h + _espn_imp_a
    _espn_true_h = np.where(_espn_vig_total > 0, _espn_imp_h / _espn_vig_total, 0.5)
    df["espn_ml_edge"] = np.where(_has_espn_ml, _espn_true_h - 0.5, 0.0)  # deviation from 50%


    # ── Spread movement (sharp money signal) ──
    # Merge Odds API (2024-2025) and DraftKings/ESPN pickcenter (2026) sources
    # v25 AUDIT: Also derive from close-open when pre-computed movement unavailable
    _oa_mvmt = pd.to_numeric(df["odds_api_spread_movement"], errors="coerce").fillna(0)
    _dk_mvmt = pd.to_numeric(df["dk_spread_movement"], errors="coerce").fillna(0)
    # Derive from open/close when movement column is missing
    _dk_sp_open = pd.to_numeric(df.get("dk_spread_open", 0), errors="coerce").fillna(0)
    _dk_sp_close = pd.to_numeric(df.get("dk_spread_close", 0), errors="coerce").fillna(0)
    _oa_sp_open = pd.to_numeric(df.get("odds_api_spread_open", 0), errors="coerce").fillna(0)
    _oa_sp_close = pd.to_numeric(df.get("odds_api_spread_close", 0), errors="coerce").fillna(0)
    _dk_derived = np.where((_dk_sp_open != 0) & (_dk_sp_close != 0), _dk_sp_close - _dk_sp_open, 0)
    _oa_derived = np.where((_oa_sp_open != 0) & (_oa_sp_close != 0), _oa_sp_close - _oa_sp_open, 0)
    # Cascade: pre-computed OA > pre-computed DK > derived OA > derived DK
    df["spread_movement"] = np.where(_oa_mvmt != 0, _oa_mvmt,
                            np.where(_dk_mvmt != 0, _dk_mvmt,
                            np.where(_oa_derived != 0, _oa_derived, _dk_derived)))
    df["has_spread_movement"] = (df["spread_movement"] != 0).astype(int)
    # Total line movement
    _oa_total_mvmt = pd.to_numeric(df["odds_api_total_movement"], errors="coerce").fillna(0)
    _dk_total_mvmt = pd.to_numeric(df["dk_total_movement"], errors="coerce").fillna(0)
    _dk_ou_open = pd.to_numeric(df.get("dk_total_open", 0), errors="coerce").fillna(0)
    _dk_ou_close = pd.to_numeric(df.get("dk_total_close", 0), errors="coerce").fillna(0)
    _oa_ou_open = pd.to_numeric(df.get("odds_api_total_open", 0), errors="coerce").fillna(0)
    _oa_ou_close = pd.to_numeric(df.get("odds_api_total_close", 0), errors="coerce").fillna(0)
    _dk_ou_derived = np.where((_dk_ou_open != 0) & (_dk_ou_close != 0), _dk_ou_close - _dk_ou_open, 0)
    _oa_ou_derived = np.where((_oa_ou_open != 0) & (_oa_ou_close != 0), _oa_ou_close - _oa_ou_open, 0)
    df["total_movement"] = np.where(_oa_total_mvmt != 0, _oa_total_mvmt,
                           np.where(_dk_total_mvmt != 0, _dk_total_mvmt,
                           np.where(_oa_ou_derived != 0, _oa_ou_derived, _dk_ou_derived)))

    # ── Lineup stability features (from pre-computed starter_ids analysis) ──
    # Computed by compute_lineup_features.py, stored in Supabase/parquet.
    # 4 features capturing roster continuity — 99.9% coverage from starter_ids.
    _h_lc = pd.to_numeric(df.get("home_lineup_changes", 0), errors="coerce").fillna(0)
    _a_lc = pd.to_numeric(df.get("away_lineup_changes", 0), errors="coerce").fillna(0)
    df["lineup_changes_diff"] = _h_lc - _a_lc

    _h_ls = pd.to_numeric(df.get("home_lineup_stability_5g", 1.0), errors="coerce").fillna(1.0)
    _a_ls = pd.to_numeric(df.get("away_lineup_stability_5g", 1.0), errors="coerce").fillna(1.0)
    df["lineup_stability_diff"] = _h_ls - _a_ls

    _h_gt = pd.to_numeric(df.get("home_starter_games_together", 0), errors="coerce").fillna(0)
    _a_gt = pd.to_numeric(df.get("away_starter_games_together", 0), errors="coerce").fillna(0)
    df["starter_experience_diff"] = _h_gt - _a_gt

    # ── Player impact ratings (walk-forward RAPM from compute_player_impact.py) ──
    # r=0.379 with margin (3rd strongest feature, no leakage)
    _h_pr = pd.to_numeric(df.get("home_player_rating_sum", 0), errors="coerce").fillna(0)
    _a_pr = pd.to_numeric(df.get("away_player_rating_sum", 0), errors="coerce").fillna(0)
    df["player_rating_diff"] = _h_pr - _a_pr

    _h_ws = pd.to_numeric(df.get("home_weakest_starter", 0), errors="coerce").fillna(0)
    _a_ws = pd.to_numeric(df.get("away_weakest_starter", 0), errors="coerce").fillna(0)
    df["weakest_starter_diff"] = _h_ws - _a_ws

    _h_sv = pd.to_numeric(df.get("home_starter_variance", 0), errors="coerce").fillna(0)
    _a_sv = pd.to_numeric(df.get("away_starter_variance", 0), errors="coerce").fillna(0)
    df["starter_balance_diff"] = _h_sv - _a_sv

    # ── Head-to-head matchup history (from compute_advanced_features.py) ──
    # r=0.474 with margin, 66.7% coverage, low redundancy with existing
    df["h2h_margin_avg"] = pd.to_numeric(df.get("h2h_margin_avg", 0), errors="coerce").fillna(0)
    df["h2h_home_win_rate"] = pd.to_numeric(df.get("h2h_home_win_rate", 0), errors="coerce").fillna(0)

    # ── Conference strength (backfilled to 95%+ coverage) ──
    # r=0.656 with margin on cross-conference games (45.5% nonzero, rest are same-conf = 0)
    df["conf_strength_diff"] = pd.to_numeric(df.get("conf_strength_diff", 0), errors="coerce").fillna(0)
    df["cross_conf_flag"] = pd.to_numeric(df.get("cross_conf_flag", 0), errors="coerce").fillna(0)

    # ── Pace-adjusted stats (computed inline from existing data) ──
    # r=0.476 — beats raw ppg_diff (0.371) by normalizing to 70 possessions
    _STD_PACE = 70.0
    _h_tempo_r = pd.to_numeric(df.get("home_tempo", 70), errors="coerce").fillna(70)
    _a_tempo_r = pd.to_numeric(df.get("away_tempo", 70), errors="coerce").fillna(70)
    _h_ppg_r = pd.to_numeric(df.get("home_ppg", 0), errors="coerce").fillna(0)
    _a_ppg_r = pd.to_numeric(df.get("away_ppg", 0), errors="coerce").fillna(0)
    _h_opp_r = pd.to_numeric(df.get("home_opp_ppg", 0), errors="coerce").fillna(0)
    _a_opp_r = pd.to_numeric(df.get("away_opp_ppg", 0), errors="coerce").fillna(0)
    df["pace_adj_ppg_diff"] = np.where(_h_tempo_r > 0, _h_ppg_r * _STD_PACE / _h_tempo_r, _h_ppg_r) - \
                               np.where(_a_tempo_r > 0, _a_ppg_r * _STD_PACE / _a_tempo_r, _a_ppg_r)
    df["pace_adj_opp_ppg_diff"] = np.where(_h_tempo_r > 0, _h_opp_r * _STD_PACE / _h_tempo_r, _h_opp_r) - \
                                   np.where(_a_tempo_r > 0, _a_opp_r * _STD_PACE / _a_tempo_r, _a_opp_r)

    # ── Recent form (from compute_advanced_features.py) ──
    # r=0.418, 84% coverage — more responsive than season-long form_diff
    df["recent_form_diff"] = pd.to_numeric(df.get("recent_form_diff", 0), errors="coerce").fillna(0)

    feature_cols = [
        # ── EXISTING 38 (unchanged) ──
        "neutral_em_diff", "hca_pts",
        "ppg_diff", "opp_ppg_diff", "fgpct_diff", "threepct_diff",
        "orb_pct_diff", "fta_rate_diff", "ato_diff",
        "def_fgpct_diff", "steals_diff", "blocks_diff",
        "sos_diff", "form_diff", "win_pct_diff",  # rank_diff REMOVED (v25 audit: 11%, redundant with elo/em)
        "to_margin_diff",         "tempo_avg",         "season_phase",
                # Unified market signal (ESPN preferred → Odds API fallback)
        "mkt_spread", "mkt_total", "mkt_spread_vs_model", "has_mkt",
                "is_conf_tourney",
        "importance",
        # ── v3: Elo ──
        "elo_diff",
        # ── v3: Advanced shooting ──
        "efg_diff", "twopt_diff",
        "three_rate_diff", "assist_rate_diff", "drb_pct_diff", "ppp_diff",
        # ── v3: Opponent Four Factors ──
        "opp_efg_diff", "opp_to_rate_diff", "opp_fta_rate_diff", "opp_orb_pct_diff",
        # ── v3: KenPom analytics ──
        "luck_diff", "consistency_diff", "pyth_residual_diff",
        # ── v3: Momentum / Form ──
        "margin_trend_diff", "margin_accel_diff",
        "opp_adj_form_diff", "wl_momentum_diff",
        "recovery_diff", "after_loss_either",
        # ── v3: Volatility / Distribution ──
        "ceiling_diff", "floor_diff",
        "margin_skew_diff", "scoring_entropy_diff", "bimodal_diff",
        # ── v3: Defensive profile ──
        "def_stability_diff", "opp_suppression_diff", "def_versatility_diff",
        "steal_foul_diff", "block_foul_diff",
        # ── v3: Transition / Paint ──
        "transition_dep_diff", "paint_pts_diff", "fastbreak_diff",
        # ── v3: Schedule / Fatigue ──
        "fatigue_diff", "streak_diff", "season_pct_avg",
        # ── v3: Regression / Information ──
        "regression_diff", "info_gain_diff", "overreaction_diff",
        # ── v3: Scoring source ──
        "scoring_source_entropy_diff", "ft_dependency_diff",
        "three_value_diff", "concentration_diff", "to_conversion_diff",
        # ── v3: Hot/cold divergence ──
        "three_divergence_diff", "ppp_divergence_diff",
        # ── v3: Pace-adjusted / SOS / Venue ──
        "pace_adj_margin_diff", "pit_sos_diff", "venue_advantage",
                # ── v3: Matchup-level (pre-computed in backfill) ──
        "matchup_efg", "matchup_to", "matchup_orb", "matchup_ft",
        "style_familiarity", "pace_leverage",
        "common_opp_diff", "pace_control_diff",         # ── v3: Situational ──
        "is_midweek", "spread_regime",
        # ── v3: Interaction features ──
        "roll_star1_share_diff", "roll_top3_share_diff",
        "roll_bench_share_diff", "roll_bench_pts_diff",
        "ref_ou_bias", "ref_home_whistle", "ref_foul_rate",
                "adj_oe_diff", "adj_de_diff",
        "scoring_var_diff", "score_kurtosis_diff",
        "clutch_ratio_diff", "garbage_adj_ppp_diff",
        "days_since_loss_diff", "games_since_blowout_diff",
        "games_last_14_diff", "rest_effect_diff",
        "momentum_halflife_diff", "win_aging_diff",
        "centrality_diff",         "n_common_opps",         "fatigue_x_quality",         "rest_x_defense", "form_x_familiarity", "consistency_x_spread",
        # is_lookahead REMOVED (v25 audit: 5%, too sparse to learn from)
        # v25 AUDIT: Added situational context features
        "is_early",                  # early season flag (35%, stable, real signal)
        "is_ncaa_tourney",           # NCAA tournament flag (2%, meaningful single-elimination context)
        # ═══ v19→v20: ESPN Win Prob edges (unique signal beyond spread) ═══
        "market_wp_edge",           # v25: replaces espn_wp_edge (was bad data)
        # crowd_pct REMOVED (v25 audit: 0% attendance data, restore after backfill)
        # ═══ v21: Rolling team tendency features (prior games, no leakage) ═══
        # PBP tendencies
        "roll_run_diff", "roll_drought_diff",
        "roll_lead_change_avg", "roll_dominance_diff",
        # Player dependency tendencies
        "roll_rotation_diff", "roll_hhi_diff",
        # Clutch + blowout tendencies
        "roll_clutch_ft_diff", "roll_garbage_diff",
        # ATS tendencies (market mispricing signal)
        "roll_ats_diff_gated", "roll_ats_margin_gated", "has_ats_data",
        # NOTE: PBP, clutch, player, and win probability features REMOVED —
        # they are in-game outcomes (data leakage). The columns remain in
        # Supabase for future use (e.g., post-game analysis, team tendency
        # rolling averages) but must NOT be used as training features.,
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
        # ═══ v22: Orphaned features — Category 2 (new diffs, 9 features) ═══
        "fouls_diff",                # personal fouls diff (59,742)
        "run_vulnerability_diff",    # run vulnerability diff (59,739)
        "ft_pressure_diff",          # FT pressure situations diff (54,589)
        "margin_autocorr_diff",      # margin autocorrelation diff (52,359)
        "blowout_asym_diff",         # blowout asymmetry diff (51,834)
        "sos_trajectory_diff",       # SOS trajectory diff (42,714)
        "anti_fragility_diff",       # anti-fragility diff (30,289)
        "clutch_over_exp_diff",      # clutch over-expectation diff (23,562)
        # ═══ v22: Orphaned features — Category 3 (situational flags, 6 features) ═══
        "is_revenge_game",           # rematch flag (17,514 games)
        "revenge_margin",            # prior loss margin in rematch
        # is_sandwich REMOVED (v25 audit: 1%, too rare to learn from)
        # def_rest_advantage REMOVED (v25 audit: degrading 38%→19%, redundant with rest_diff r=0.79)
        "luck_x_spread",             # luck × spread interaction (v25: now computed inline)
        # ═══ v23: Lineup stability features (3 features) ═══
        "lineup_changes_diff",       # starters changed from last game (home-away)
        "lineup_stability_diff",     # rolling 5-game lineup consistency diff
        "starter_experience_diff",   # games this exact 5 started together diff
        # ═══ v24: Player impact + advanced features (11 features) ═══
        "player_rating_diff",        # RAPM walk-forward starter quality (r=0.379)
        "weakest_starter_diff",      # weakest link in lineup diff
        "starter_balance_diff",      # star-dependent vs balanced lineup diff
        "h2h_margin_avg",            # historical matchup margin (r=0.474)
        "h2h_home_win_rate",         # matchup dominance rate (r=0.333)
        "conf_strength_diff",        # conference quality gap (r=0.656 on cross-conf)
        "cross_conf_flag",           # inter-conference game flag
        "pace_adj_ppg_diff",         # pace-normalized offensive diff (r=0.476)
        "pace_adj_opp_ppg_diff",     # pace-normalized defensive diff (r=0.385)
        "recent_form_diff",          # last 5 games win rate diff (r=0.418)
    ]
    return df[feature_cols].fillna(0)


# ── NCAA Historical Corpus Support ────────────────────────────

def _ncaa_season_weight(season):
    """Recency weighting: recent seasons get higher weight for ML training."""
    current_year = datetime.utcnow().year
    age = current_year - season
    if age <= 0: return 1.0
    if age == 1: return 0.9
    if age == 2: return 0.75
    if age == 3: return 0.6
    return 0.5


def _flush_ncaa_batch(rows):
    """Insert a batch of ncaa_historical rows via Supabase UPSERT."""
    if not rows:
        return
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/ncaa_historical",
            headers=headers,
            json=rows,
            timeout=30,
        )
        if not resp.ok:
            print(f"  UPSERT error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"  UPSERT exception: {e}")


def _ncaa_backfill_heuristic(df):
    """
    Replay ncaaUtils.js heuristic on historical rows — AUDIT v18 FIX.

    CRITICAL FIXES (from forensic audit):
      F1: Score formula changed from averaging ((A+B)/2) to KenPom additive (A+B-lgAvg).
          The averaging formula compressed ALL spreads by ~50% — a mathematical identity.
      F2: Dynamic lg_avg now USED in the formula (was computed but ignored).
      F4: Four Factors boost, defensive disruption, ball control, TS%, blowout scaling
          all ported from JS frontend (were completely missing).
      F5: Form signal upgraded from simple winPct to per-team scaled version with
          formWeight proportional to games played (matches JS pattern).
    """
    df = df.copy()

    # Conference ID → HCA mapping (ESPN conference IDs → HCA points)
    CONF_ID_HCA = {
        "8": 3.8,   # Big 12
        "23": 3.7,  # SEC
        "7": 3.6,   # Big Ten
        "2": 3.4,   # ACC
        "4": 3.3,   # Big East
        "21": 3.0,  # Pac-12
        "44": 3.2,  # Mountain West
        "62": 3.0,  # AAC
        "26": 2.8,  # WCC
        "3": 2.7,   # A-10
        "18": 2.9,  # MVC
        "40": 2.6,  # Sun Belt
        "12": 2.8,  # MAC
        "10": 2.5,  # CAA
        "22": 2.3,  # Ivy
    }
    # Also support full conference names (current-season data uses names, not IDs)
    CONF_NAME_HCA = {
        "Big 12": 3.8, "Southeastern Conference": 3.7, "SEC": 3.7,
        "Big Ten": 3.6, "Big Ten Conference": 3.6,
        "Atlantic Coast Conference": 3.4, "ACC": 3.4,
        "Big East": 3.3, "Big East Conference": 3.3,
        "Pac-12": 3.0, "Pac-12 Conference": 3.0,
        "Mountain West Conference": 3.2, "Mountain West": 3.2,
        "American Athletic Conference": 3.0, "AAC": 3.0,
        "West Coast Conference": 2.8, "WCC": 2.8,
        "Atlantic 10 Conference": 2.7, "A-10": 2.7,
        "Missouri Valley Conference": 2.9, "MVC": 2.9,
    }
    DEFAULT_HCA = 3.0
    SIGMA = 16.0  # matches JS ncaaPredictGame calibration

    def _lookup_hca(conf_val):
        """Lookup HCA from either ESPN ID or conference name."""
        s = str(conf_val).strip()
        if s in CONF_ID_HCA:
            return CONF_ID_HCA[s]
        if s in CONF_NAME_HCA:
            return CONF_NAME_HCA[s]
        return DEFAULT_HCA

    h_em = df["home_adj_em"].fillna(0).values
    a_em = df["away_adj_em"].fillna(0).values
    h_ppg = df["home_ppg"].fillna(70).values
    a_ppg = df["away_ppg"].fillna(70).values
    h_opp = df["home_opp_ppg"].fillna(70).values
    a_opp = df["away_opp_ppg"].fillna(70).values
    h_tempo = df["home_tempo"].fillna(68).values
    a_tempo = df["away_tempo"].fillna(68).values
    neutral = df["neutral_site"].fillna(False).values
    h_conf = df["home_conference"].fillna("").astype(str).values
    h_rank = df["home_rank"].fillna(200).values
    a_rank = df["away_rank"].fillna(200).values
    h_wins = df["home_record_wins"].fillna(0).values
    h_losses = df["home_record_losses"].fillna(0).values
    a_wins = df["away_record_wins"].fillna(0).values
    a_losses = df["away_record_losses"].fillna(0).values

    # F4: Additional stat arrays for Four Factors + defensive boost
    h_fgpct = df["home_fgpct"].fillna(0.455).values if "home_fgpct" in df.columns else np.full(len(df), 0.455)
    a_fgpct = df["away_fgpct"].fillna(0.455).values if "away_fgpct" in df.columns else np.full(len(df), 0.455)
    h_threepct = df["home_threepct"].fillna(0.340).values if "home_threepct" in df.columns else np.full(len(df), 0.340)
    a_threepct = df["away_threepct"].fillna(0.340).values if "away_threepct" in df.columns else np.full(len(df), 0.340)
    h_turnovers = df["home_turnovers"].fillna(12.0).values if "home_turnovers" in df.columns else np.full(len(df), 12.0)
    a_turnovers = df["away_turnovers"].fillna(12.0).values if "away_turnovers" in df.columns else np.full(len(df), 12.0)
    h_orb_pct = df["home_orb_pct"].fillna(0.28).values if "home_orb_pct" in df.columns else np.full(len(df), 0.28)
    a_orb_pct = df["away_orb_pct"].fillna(0.28).values if "away_orb_pct" in df.columns else np.full(len(df), 0.28)
    h_fta_rate = df["home_fta_rate"].fillna(0.34).values if "home_fta_rate" in df.columns else np.full(len(df), 0.34)
    a_fta_rate = df["away_fta_rate"].fillna(0.34).values if "away_fta_rate" in df.columns else np.full(len(df), 0.34)
    h_steals = df["home_steals"].fillna(7.0).values if "home_steals" in df.columns else np.full(len(df), 7.0)
    a_steals = df["away_steals"].fillna(7.0).values if "away_steals" in df.columns else np.full(len(df), 7.0)
    h_blocks = df["home_blocks"].fillna(3.5).values if "home_blocks" in df.columns else np.full(len(df), 3.5)
    a_blocks = df["away_blocks"].fillna(3.5).values if "away_blocks" in df.columns else np.full(len(df), 3.5)
    h_ato = df["home_ato_ratio"].fillna(1.2).values if "home_ato_ratio" in df.columns else np.full(len(df), 1.2)
    a_ato = df["away_ato_ratio"].fillna(1.2).values if "away_ato_ratio" in df.columns else np.full(len(df), 1.2)
    # Defensive rebounds for opponent's matchup ORB%
    h_defreb = df["home_def_reb"].fillna(24.5).values if "home_def_reb" in df.columns else np.full(len(df), 24.5)
    a_defreb = df["away_def_reb"].fillna(24.5).values if "away_def_reb" in df.columns else np.full(len(df), 24.5)

    is_post = df["is_postseason"].fillna(0).values if "is_postseason" in df.columns else np.zeros(len(df))

    n = len(df)
    pred_home_score = np.zeros(n)
    pred_away_score = np.zeros(n)
    win_pct_home = np.full(n, 0.5)
    spread_home = np.zeros(n)

    # ── F2 FIX: Compute dynamic league average PPG (winsorized) ONCE outside loop ──
    _all_ppg = np.concatenate([h_ppg[h_ppg > 0], a_ppg[a_ppg > 0]])
    if len(_all_ppg) > 20:
        _lo, _hi = np.percentile(_all_ppg, 5), np.percentile(_all_ppg, 95)
        lg_avg_ppg = float(np.mean(np.clip(_all_ppg, _lo, _hi)))
    else:
        lg_avg_ppg = 72.5  # fallback NCAA average PPG

    # League averages for Four Factors (can be derived dynamically in future)
    LG_EFG = 0.502
    LG_TO_PCT = 18.0
    LG_ORB_PCT = 0.28
    LG_FTA_RATE = 0.34
    LG_AVG_TEMPO = 68.0

    for i in range(n):
        possessions = (h_tempo[i] + a_tempo[i]) / 2
        tempo_ratio = possessions / LG_AVG_TEMPO

        # ── F1 FIX: KenPom ADDITIVE formula ──
        # BEFORE (wrong): hs = ((home_oe + away_de) / 2) * tempo_ratio
        #   This averaging formula compresses ALL spreads by exactly 50%.
        # AFTER (correct): hs = (home_oe + away_de - lg_avg) * tempo_ratio
        home_oe = h_ppg[i] if h_ppg[i] > 0 else lg_avg_ppg
        away_oe = a_ppg[i] if a_ppg[i] > 0 else lg_avg_ppg
        home_de = h_opp[i] if h_opp[i] > 0 else lg_avg_ppg
        away_de = a_opp[i] if a_opp[i] > 0 else lg_avg_ppg

        hs = (home_oe + away_de - lg_avg_ppg) * tempo_ratio
        asc = (away_oe + home_de - lg_avg_ppg) * tempo_ratio

        # ── F4: Four Factors boost (matches JS ncaaUtils.js) ──
        tempo_scale = possessions / LG_AVG_TEMPO
        three_rate = 0.38  # default NCAA 3PA rate

        def _four_factors(fgpct, threepct, turnovers, tempo, orb_pct, fta_rate):
            efg = fgpct + 0.5 * three_rate * threepct
            efg_boost = (efg - LG_EFG) * 10.0       # F9: recalibrated to Dean Oliver 40% target

            to_pct = (turnovers / tempo * 100) if tempo > 0 else LG_TO_PCT
            to_boost = max(-2.5, min(2.5, (LG_TO_PCT - to_pct) * 0.08))  # F9: 0.09→0.08

            orb_boost = (orb_pct - LG_ORB_PCT) * 4.0   # F9: 5.5→4.0
            ftr_boost = (fta_rate - LG_FTA_RATE) * 2.5  # F9: 3.0→2.5

            return (efg_boost + to_boost + orb_boost + ftr_boost) * tempo_scale

        home_ff = _four_factors(h_fgpct[i], h_threepct[i], h_turnovers[i], h_tempo[i],
                                h_orb_pct[i], h_fta_rate[i])
        away_ff = _four_factors(a_fgpct[i], a_threepct[i], a_turnovers[i], a_tempo[i],
                                a_orb_pct[i], a_fta_rate[i])

        # ── F4: Defensive disruption boost (matches JS) ──
        home_def = (h_steals[i] - 7.0) * 0.08 + (h_blocks[i] - 3.5) * 0.06
        away_def = (a_steals[i] - 7.0) * 0.08 + (a_blocks[i] - 3.5) * 0.06

        # ── F4: Ball control (ATO + turnover margin) ──
        ato_boost = ((h_ato[i] - 1.2) - (a_ato[i] - 1.2)) * 0.5
        to_margin_h = h_steals[i] - h_turnovers[i]
        to_margin_a = a_steals[i] - a_turnovers[i]
        to_margin_boost = (to_margin_h - to_margin_a) * 0.08

        # ── F4: Blowout scaling (matches JS — emGap >= 15 ramps to 1.5×) ──
        em_gap = abs(h_em[i] - a_em[i])
        blowout_scale = min(1.5, 1.0 + (em_gap - 15) / 30) if em_gap >= 15 else 1.0

        # Assemble scores with all components (matches JS ncaaPredictGame)
        hs += home_ff * 0.35 * blowout_scale + home_def * 0.20 * blowout_scale + ato_boost * 0.5 + to_margin_boost * 0.5
        asc += away_ff * 0.35 * blowout_scale + away_def * 0.20 * blowout_scale - ato_boost * 0.5 - to_margin_boost * 0.5

        # ── 2. Home court advantage ──
        if not neutral[i]:
            hca = _lookup_hca(h_conf[i])
            hs += hca / 2
            asc -= hca / 2

        # ── 3. Rank boost (exponential, matches JS) ──
        def rank_boost(rank):
            return max(0, 1.2 * np.exp(-rank / 15)) if rank <= 50 else 0
        hs += rank_boost(h_rank[i]) * 0.3
        asc += rank_boost(a_rank[i]) * 0.3

        # ── F5 FIX: Per-team form signal with games-played scaling ──
        # Matches JS pattern: formWeight scales from 0→0.10 based on sqrt(games/30)
        # Applied as: (winPct - 0.5) * formWeight * 40.0 to approximate JS formScore * formWeight * 4.0
        h_games = h_wins[i] + h_losses[i]
        a_games = a_wins[i] + a_losses[i]
        if h_games >= 3:
            h_wp = h_wins[i] / h_games
            h_form_weight = min(0.10, 0.10 * np.sqrt(min(h_games, 30) / 30))
            hs += (h_wp - 0.5) * h_form_weight * 40.0
        if a_games >= 3:
            a_wp = a_wins[i] / a_games
            a_form_weight = min(0.10, 0.10 * np.sqrt(min(a_games, 30) / 30))
            asc += (a_wp - 0.5) * a_form_weight * 40.0

        # ── v20 cross-sport FIX: True Shooting % (ALIGN-7, ported from NBA) ──
        # TS% captures free throw conversion beyond what FTR measures.
        # Weight at 0.05 to avoid overlap with eFG% in Four Factors.
        NCAA_LG_TS = 0.540  # NCAA D1 average TS% (~54.0% vs NBA's ~57.8%)
        if "home_fga" in df.columns and "home_fta" in df.columns:
            _h_fga = float(df["home_fga"].values[i]) if pd.notna(df["home_fga"].values[i]) else 0
            _a_fga = float(df["away_fga"].values[i]) if "away_fga" in df.columns and pd.notna(df["away_fga"].values[i]) else 0
            _h_fta = float(df["home_fta"].values[i]) if pd.notna(df["home_fta"].values[i]) else 0
            _a_fta = float(df["away_fta"].values[i]) if "away_fta" in df.columns and pd.notna(df["away_fta"].values[i]) else 0
            def _ts_boost(ppg_val, fga_val, fta_val):
                if fga_val <= 0 or fta_val <= 0: return 0
                tsa = fga_val + 0.44 * fta_val
                if tsa <= 0: return 0
                ts = ppg_val / (2 * tsa)
                return max(-2.5, min(2.5, (ts - NCAA_LG_TS) * 15))
            hs += _ts_boost(h_ppg[i], _h_fga, _h_fta) * 0.05
            asc += _ts_boost(a_ppg[i], _a_fga, _a_fta) * 0.05

        # ── 5. Postseason compression ──
        if is_post[i]:
            mid = (hs + asc) / 2
            hs = mid + (hs - mid) * 0.90
            asc = mid + (asc - mid) * 0.90

        # ── Total cap (matches JS maxRealisticTotal = 190) ──
        raw_total = hs + asc
        if raw_total > 190:
            current_spread = hs - asc
            capped_mid = 190 / 2
            hs = capped_mid + current_spread / 2
            asc = capped_mid - current_spread / 2

        # ── Safety clamp [35, 130] ──
        hs = max(35, min(130, hs))
        asc = max(35, min(130, asc))

        # ── Spread and win probability ──
        spread = hs - asc
        wp = 1.0 / (1.0 + 10.0 ** (-spread / SIGMA))
        wp = max(0.03, min(0.97, wp))

        pred_home_score[i] = round(hs, 1)
        pred_away_score[i] = round(asc, 1)
        win_pct_home[i] = round(wp, 4)
        spread_home[i] = round(spread, 1)

    df["pred_home_score"] = pred_home_score
    df["pred_away_score"] = pred_away_score
    df["win_pct_home"] = win_pct_home
    df["spread_home"] = spread_home

    # ── v20 cross-sport FIX: O/U uses PPG-based formula with shrink (matches JS ALIGN-8) ──
    # Before: ou_total = pred_home_score + pred_away_score (spread-optimized, inflated by ~35 pts)
    # After: PPG-based additive with 0.975 shrink factor (matches ncaaUtils.js lines 625-632)
    NCAA_TOTAL_SHRINK = 0.975
    _ou_hca = np.where(df["neutral_site"].fillna(False).values, 0, 1.5)
    _all_ppg_vals = np.concatenate([h_ppg[h_ppg > 0], a_ppg[a_ppg > 0]])
    _ou_lg = float(np.mean(np.clip(_all_ppg_vals, np.percentile(_all_ppg_vals, 5), np.percentile(_all_ppg_vals, 95)))) if len(_all_ppg_vals) > 20 else 72.5
    _ou_home = (h_ppg + a_opp - _ou_lg + _ou_hca / 2) * NCAA_TOTAL_SHRINK
    _ou_away = (a_ppg + h_opp - _ou_lg - _ou_hca / 2) * NCAA_TOTAL_SHRINK
    df["ou_total"] = np.maximum(100, np.round(_ou_home + _ou_away, 1))
    df["model_ml_home"] = [
        int(-round((wp / (1 - wp)) * 100)) if wp >= 0.5
        else int(round(((1 - wp) / wp) * 100))
        for wp in win_pct_home
    ]

    # Diagnostics
    wp_std = np.std(win_pct_home)
    sp_std = np.std(spread_home)
    non_neutral = (win_pct_home != 0.5).sum()
    print(f"  NCAA heuristic backfill (v18 audit fix): {n} rows | "
          f"win_pct std={wp_std:.3f}, range=[{np.min(win_pct_home):.3f}, {np.max(win_pct_home):.3f}] | "
          f"spread std={sp_std:.1f}, range=[{np.min(spread_home):.1f}, {np.max(spread_home):.1f}] | "
          f"{non_neutral}/{n} non-neutral | lg_avg_ppg={lg_avg_ppg:.1f}")

    return df


def _ncaa_merge_historical(current_df):
    """
    Fetch ncaa_historical (multi-season) and combine with current season
    ncaa_predictions for ML training. Same pattern as _mlb_merge_historical.
    """
    hist_rows = sb_get(
        "ncaa_historical",
        "actual_home_score=not.is.null&select=*&order=season.desc&limit=100000"
    )
    if not hist_rows:
        print("  WARNING: ncaa_historical empty - training on current season only")
        if current_df is None or len(current_df) == 0:
            return pd.DataFrame(), None, 0
        return current_df, None, 0

    hist_df = pd.DataFrame(hist_rows)

    numeric_cols = [
        "actual_home_score", "actual_away_score", "home_win",
        "home_adj_em", "away_adj_em", "home_adj_oe", "away_adj_oe",
        "home_adj_de", "away_adj_de", "home_ppg", "away_ppg",
        "home_opp_ppg", "away_opp_ppg", "home_tempo", "away_tempo",
        "home_record_wins", "away_record_wins",
        "home_record_losses", "away_record_losses",
        "home_rank", "away_rank", "season_weight",
    ]
    for col in numeric_cols:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors="coerce")

    # ── Heuristic backfill: replicate ncaaUtils.js prediction logic ──
    # Instead of win_pct_home=0.5, compute real pre-game predictions from
    # the enriched columns so the ML model trains on realistic signal.
    hist_df = _ncaa_backfill_heuristic(hist_df)

    # ── Column name alignment ──
    # ncaa_historical uses home_record_wins/losses, feature builder expects home_wins/losses
    if "home_record_wins" in hist_df.columns and "home_wins" not in hist_df.columns:
        hist_df["home_wins"] = hist_df["home_record_wins"]
    if "away_record_wins" in hist_df.columns and "away_wins" not in hist_df.columns:
        hist_df["away_wins"] = hist_df["away_record_wins"]
    if "home_record_losses" in hist_df.columns and "home_losses" not in hist_df.columns:
        hist_df["home_losses"] = hist_df["home_record_losses"]
    if "away_record_losses" in hist_df.columns and "away_losses" not in hist_df.columns:
        hist_df["away_losses"] = hist_df["away_record_losses"]

    # Default missing stat columns to neutral values so fillna(0) works correctly
    # in ncaa_build_features. These columns exist in live predictions but not historical.
    for col, default in [
        ("home_fgpct", 0.44), ("away_fgpct", 0.44),
        ("home_threepct", 0.34), ("away_threepct", 0.34),
        ("home_orb_pct", 0.28), ("away_orb_pct", 0.28),
        ("home_fta_rate", 0.34), ("away_fta_rate", 0.34),
        ("home_ato_ratio", 1.2), ("away_ato_ratio", 1.2),
        ("home_opp_fgpct", 0.44), ("away_opp_fgpct", 0.44),
        ("home_opp_threepct", 0.33), ("away_opp_threepct", 0.33),
        ("home_steals", 7.0), ("away_steals", 7.0),
        ("home_blocks", 3.5), ("away_blocks", 3.5),
        ("home_turnovers", 12.0), ("away_turnovers", 12.0),
        ("home_sos", 0.0), ("away_sos", 0.0),
        ("home_form", 0.0), ("away_form", 0.0),
        ("home_rest_days", 3), ("away_rest_days", 3),
    ]:
        if col not in hist_df.columns:
            hist_df[col] = default

    # ── Tournament context from is_postseason flag ──
    if "is_postseason" in hist_df.columns:
        hist_df["is_ncaa_tournament"] = hist_df["is_postseason"].fillna(0).astype(int)
    if "is_conference_tournament" not in hist_df.columns:
        hist_df["is_conference_tournament"] = 0
    if "is_bubble_game" not in hist_df.columns:
        hist_df["is_bubble_game"] = 0
    if "is_early_season" not in hist_df.columns:
        # Early season = November games
        if "game_date" in hist_df.columns:
            gd = pd.to_datetime(hist_df["game_date"], errors="coerce")
            hist_df["is_early_season"] = (gd.dt.month.isin([11, 12]) & (gd.dt.day <= 15)).astype(int)
        else:
            hist_df["is_early_season"] = 0
    if "importance_multiplier" not in hist_df.columns:
        hist_df["importance_multiplier"] = 1.0
    # Injury columns (not available for historical)
    for inj_col in ["home_missing_starters", "away_missing_starters",
                     "home_injury_penalty", "away_injury_penalty"]:
        if inj_col not in hist_df.columns:
            hist_df[inj_col] = 0

    if "home_team" not in hist_df.columns and "home_team_abbr" in hist_df.columns:
        hist_df["home_team"] = hist_df["home_team_abbr"]
    if "away_team" not in hist_df.columns and "away_team_abbr" in hist_df.columns:
        hist_df["away_team"] = hist_df["away_team_abbr"]

    if "neutral_site" in hist_df.columns:
        hist_df["neutral_site"] = hist_df["neutral_site"].fillna(False)

    if "actual_margin" not in hist_df.columns:
        hist_df["actual_margin"] = (
            hist_df["actual_home_score"] - hist_df["actual_away_score"]
        )

    if current_df is not None and len(current_df) > 0:
        combined = pd.concat([hist_df, current_df], ignore_index=True)
    else:
        combined = hist_df

    if "season_weight" in combined.columns:
        weights = combined["season_weight"].fillna(1.0).astype(float)
    else:
        weights = pd.Series(1.0, index=combined.index)

    n_hist = len(hist_df)
    n_curr = len(current_df) if current_df is not None else 0
    print(f"  NCAA training corpus: {n_hist} historical + {n_curr} current "
          f"= {n_hist + n_curr} total")


    # ── ESPN/DraftKings odds fallback (sweep-validated: 4% → 59% market coverage) ──
    if "espn_spread" in combined.columns:
        espn_s = pd.to_numeric(combined["espn_spread"], errors="coerce")
        if "market_spread_home" not in combined.columns:
            combined["market_spread_home"] = np.nan
        mkt_s = pd.to_numeric(combined["market_spread_home"], errors="coerce")
        fill_mask = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
        combined.loc[fill_mask, "market_spread_home"] = espn_s[fill_mask]
        n_filled = int(fill_mask.sum())
        if n_filled > 0:
            print(f"  ESPN odds fallback: {n_filled} spreads filled from DraftKings")

    if "espn_over_under" in combined.columns:
        espn_ou = pd.to_numeric(combined["espn_over_under"], errors="coerce")
        if "market_ou_total" not in combined.columns:
            combined["market_ou_total"] = np.nan
        mkt_ou = pd.to_numeric(combined["market_ou_total"], errors="coerce")
        fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
        combined.loc[fill_ou, "market_ou_total"] = espn_ou[fill_ou]

    return combined, weights.values, n_hist


def train_ncaa():
    """NCAA model training with multi-season historical corpus."""
    import traceback as _tb
    try:
        rows = sb_get("ncaa_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        # ── NEW: Merge with historical corpus ────────────────
        df, sample_weights, n_historical = _ncaa_merge_historical(current_df)
        n_current = len(current_df) if current_df is not None else 0

        if len(df) < 10:
            return {"error": "Not enough NCAAB data", "n": len(df),
                    "n_current": len(current_df)}

        try:
            import json as _json
            with open("referee_profiles.json") as _rf:
                ncaa_build_features._ref_profiles = _json.load(_rf)
            print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
        except FileNotFoundError:
            print("  referee_profiles.json not found - ref features zero")
            ncaa_build_features._ref_profiles = {}
        X  = ncaa_build_features(df)
        y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
        y_win    = (y_margin > 0).astype(int)

        # Track which rows are current-season (for isotonic calibration)
        # Merge puts historical first [0..n_historical-1], then current [n_historical..]
        is_current = np.zeros(len(df), dtype=bool)
        is_current[n_historical:] = True

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n = len(df)

        # Cap training data for Railway timeout protection
        MAX_TRAIN = 12000  # Railway timeout protection; set 99999 for local
        if n > MAX_TRAIN:
            if "season_weight" in df.columns:
                keep_idx = df["season_weight"].fillna(0.5).nlargest(MAX_TRAIN).index
            else:
                keep_idx = df.index[-MAX_TRAIN:]
            X_scaled = X_scaled[keep_idx]
            y_margin = y_margin.iloc[keep_idx].reset_index(drop=True)
            y_win = y_win.iloc[keep_idx].reset_index(drop=True)
            is_current = is_current[keep_idx.values]
            if sample_weights is not None:
                sample_weights = sample_weights[keep_idx.values]
            n = MAX_TRAIN
            print(f"  NCAA: Capped to {n} rows for Railway timeout protection")

        cv_folds = min(10, n)  # 5 for Railway; set 10 for local
        fit_weights = sample_weights if sample_weights is not None else np.ones(n)

        if n >= 200:
            # ── Stacking ensemble at e=175 d=7 lr=0.1imators (sweep-optimized) ──
            # SWEEP WINNER: LGBM replaces RF (faster + better MAE)
            if HAS_LGBM:
                lgbm_reg = LGBMRegressor(
                    n_estimators=175, max_depth=7, learning_rate=0.10,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                    random_state=42, verbosity=-1,
                )
            if HAS_XGB:
                xgb_reg = XGBRegressor(n_estimators=175, max_depth=7, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, random_state=42, tree_method="hist", verbosity=0)
            if HAS_CAT:
                cat_reg = CatBoostRegressor(iterations=175, depth=7, learning_rate=0.10, subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0)

            ensemble_parts = []
            if HAS_XGB: ensemble_parts.append('XGB')
            if HAS_CAT: ensemble_parts.append('CAT')
            if HAS_LGBM: ensemble_parts.append('LGBM')
            ensemble_label = '+'.join(ensemble_parts)
            print(f"  NCAAB: Training stacking ensemble on {n} rows (ts-cv, {ensemble_label}, e=175 d=7 lr=0.1)...")

            reg_models = {}
            if HAS_LGBM: reg_models["lgbm"] = lgbm_reg
            if HAS_XGB: reg_models["xgb"] = xgb_reg
            if HAS_CAT: reg_models["cat"] = cat_reg

            oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=fit_weights)
            oof_rf = oof["rf"]

            rf_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
            if HAS_XGB: xgb_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
            if HAS_CAT: cat_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)

            oof_cols = [oof_rf]
            if HAS_XGB: oof_cols.append(oof["xgb"])
            if HAS_CAT: oof_cols.append(oof["cat"])
            meta_X = np.column_stack(oof_cols)
            meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            meta_reg.fit(meta_X, y_margin)

            # Bias correction from OOF residuals
            oof_meta = meta_reg.predict(meta_X)
            bias_correction = float(np.mean(oof_meta - y_margin.values))
            print(f"  NCAAB bias correction: {bias_correction:+.3f} pts (will be subtracted from predictions)")

            base_regs = [rf_reg]
            if HAS_XGB: base_regs.append(xgb_reg)
            if HAS_CAT: base_regs.append(cat_reg)
            reg = StackedRegressor(base_regs, meta_reg, scaler)
            reg_cv_mae = float(np.mean(np.abs(oof_meta - y_margin.values)))
            print(f"  NCAAB stacked OOF MAE: {reg_cv_mae:.3f}")
            explainer = shap.TreeExplainer(xgb_reg if HAS_XGB else rf_reg)
            model_type = "StackedEnsemble_v4_TSCV"
            meta_weights = meta_reg.coef_.round(4).tolist()
            print(f"  NCAAB meta weights: {meta_weights}")

            # ── R6 FIX: Compute bias correction from OOF residuals ──
            oof_meta = meta_reg.predict(meta_X)
            bias_correction = float(np.mean(oof_meta - y_margin.values))
            print(f"  NCAAB bias correction: {bias_correction:+.3f} pts (will be subtracted from predictions)")

            # Stacked classifier
            gbm_clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.06, subsample=0.8,
                min_samples_leaf=20, random_state=42,
            )
            rf_clf = RandomForestClassifier(
                n_estimators=100, max_depth=6,
                min_samples_leaf=15, max_features=0.7,
                random_state=42, n_jobs=1,
            )
            lr_clf = LogisticRegression(max_iter=1000, C=1.0)

            oof_gbm_p = cross_val_predict(gbm_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
            oof_rf_p  = cross_val_predict(rf_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
            oof_lr_p  = cross_val_predict(lr_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]

            gbm_clf.fit(X_scaled, y_win, sample_weight=fit_weights)
            rf_clf.fit(X_scaled, y_win, sample_weight=fit_weights)
            lr_clf.fit(X_scaled, y_win, sample_weight=fit_weights)

            meta_clf_X = np.column_stack([oof_gbm_p, oof_rf_p, oof_lr_p])
            meta_lr = LogisticRegression(max_iter=1000, C=1.0)
            meta_lr.fit(meta_clf_X, y_win)
            clf = StackedClassifier([gbm_clf, rf_clf, lr_clf], meta_lr)

            # ── R4 FIX: Isotonic calibration on CURRENT-SEASON OOF only ──
            # Historical rows have simplified features (missing fgpct, threepct, etc.)
            # which makes their OOF probabilities noisier. Fitting isotonic on all rows
            # causes the calibrator to dampen probabilities too aggressively.
            # Solution: fit on current-season rows only (real pipeline predictions).
            oof_stacked_probs = meta_lr.predict_proba(meta_clf_X)[:, 1]
            current_mask = is_current[:len(oof_stacked_probs)]
            n_current_oof = int(current_mask.sum())

            if n_current_oof >= 50:
                # Enough current-season data — fit isotonic on those rows only
                isotonic = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
                isotonic.fit(oof_stacked_probs[current_mask], y_win.values[current_mask])
                print(f"  NCAAB isotonic calibration fitted on {n_current_oof} CURRENT-SEASON OOF samples "
                      f"(skipped {len(oof_stacked_probs) - n_current_oof} historical)")
            else:
                # Fallback: not enough current-season data, use all OOF
                isotonic = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
                isotonic.fit(oof_stacked_probs, y_win.values)
                print(f"  NCAAB isotonic: only {n_current_oof} current-season rows, "
                      f"falling back to ALL {len(oof_stacked_probs)} OOF samples")

        else:
            # Simple models for small data
            reg = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                             learning_rate=0.08, random_state=42)
            reg.fit(X_scaled, y_margin)
            reg_cv = cross_val_score(reg, X_scaled, y_margin,
                                      cv=min(5, len(df)), scoring="neg_mean_absolute_error")
            reg_cv_mae = float(-reg_cv.mean())
            clf = CalibratedClassifierCV(
                LogisticRegression(max_iter=1000), cv=min(5, len(df))
            )
            clf.fit(X_scaled, y_win)
            explainer = shap.TreeExplainer(reg)
            model_type = "GBM"
            bias_correction = 0.0
            isotonic = None
            meta_weights = []
            n_current_oof = 0

        bundle = {
            "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
            "feature_cols": list(X.columns), "n_train": len(df),
            "mae_cv": reg_cv_mae, "model_type": model_type,
            "trained_at": datetime.utcnow().isoformat(),
            # R6: Bias correction
            "bias_correction": bias_correction,
            # R4: Isotonic calibration
            "isotonic": isotonic,
            # R7: Meta diagnostics
            "meta_weights": meta_weights,
        }
        save_model("ncaa", bundle)
        return {"status": "trained", "n_train": len(df), "model_type": model_type,
                "n_historical": n_historical,
                "n_current": n_current,
                "isotonic_source": f"current_season ({n_current_oof} OOF samples)" if n >= 200 and n_current_oof >= 50 else "all_data",
                "mae_cv": round(reg_cv_mae, 3), "features": list(X.columns),
                "bias_correction": round(bias_correction, 3),
                "meta_weights": meta_weights}

    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": _tb.format_exc(),
        }

def predict_ncaa(game: dict):
    # Auto-inject injury data from Covers.com (refreshes every 4h)
    try:
        from injury_cache import inject_injuries
        game = inject_injuries(game)
    except Exception:
        pass  # Graceful fallback — injuries default to 0

    bundle = load_model("ncaa")
    if not bundle:
        return {"error": "NCAAB model not trained — call /train/ncaa first"}

    ph = game.get("pred_home_score", 72)
    pa = game.get("pred_away_score", 72)
    he = game.get("home_adj_em", 0)
    ae = game.get("away_adj_em", 0)

    # Build a single-row DataFrame with all features the model expects
    row_data = {
        "home_adj_em": he, "away_adj_em": ae,
        "neutral_site": game.get("neutral_site", False),
        "pred_home_score": ph, "pred_away_score": pa,
        "model_ml_home": game.get("model_ml_home", 0),
        "spread_home": game.get("spread_home", 0),
        "market_spread_home": game.get("market_spread_home", 0),
        "market_ou_total": game.get("market_ou_total", game.get("ou_total", 145)),
        "ou_total": game.get("ou_total", 145),
        # R2: Heuristic win probability for capped feature
        "win_pct_home": game.get("win_pct_home", 0.5),
        # R3: Conference info
        "home_conference": game.get("home_conference", ""),
        "away_conference": game.get("away_conference", ""),
        "game_date": game.get("game_date", ""),
        # Raw stats
        "home_ppg": game.get("home_ppg", 75), "away_ppg": game.get("away_ppg", 75),
        "home_opp_ppg": game.get("home_opp_ppg", 72), "away_opp_ppg": game.get("away_opp_ppg", 72),
        "home_fgpct": game.get("home_fgpct", 0.455), "away_fgpct": game.get("away_fgpct", 0.455),
        "home_threepct": game.get("home_threepct", 0.340), "away_threepct": game.get("away_threepct", 0.340),
        "home_ftpct": game.get("home_ftpct", 0.720), "away_ftpct": game.get("away_ftpct", 0.720),
        "home_assists": game.get("home_assists", 14), "away_assists": game.get("away_assists", 14),
        "home_turnovers": game.get("home_turnovers", 12), "away_turnovers": game.get("away_turnovers", 12),
        "home_tempo": game.get("home_tempo", 68), "away_tempo": game.get("away_tempo", 68),
        "home_orb_pct": game.get("home_orb_pct", 0.28), "away_orb_pct": game.get("away_orb_pct", 0.28),
        "home_fta_rate": game.get("home_fta_rate", 0.34), "away_fta_rate": game.get("away_fta_rate", 0.34),
        "home_ato_ratio": game.get("home_ato_ratio", 1.2), "away_ato_ratio": game.get("away_ato_ratio", 1.2),
        "home_opp_fgpct": game.get("home_opp_fgpct", 0.430), "away_opp_fgpct": game.get("away_opp_fgpct", 0.430),
        "home_opp_threepct": game.get("home_opp_threepct", 0.330), "away_opp_threepct": game.get("away_opp_threepct", 0.330),
        "home_steals": game.get("home_steals", 7), "away_steals": game.get("away_steals", 7),
        "home_blocks": game.get("home_blocks", 3.5), "away_blocks": game.get("away_blocks", 3.5),
        "home_wins": game.get("home_wins", 10), "away_wins": game.get("away_wins", 10),
        "home_losses": game.get("home_losses", 5), "away_losses": game.get("away_losses", 5),
        "home_form": game.get("home_form", 0), "away_form": game.get("away_form", 0),
        "home_sos": game.get("home_sos", 0.500), "away_sos": game.get("away_sos", 0.500),
        "home_rank": game.get("home_rank", 200), "away_rank": game.get("away_rank", 200),
        "home_rest_days": game.get("home_rest_days", 3), "away_rest_days": game.get("away_rest_days", 3),
        # v18 P1-INJ: Injury features
        "home_injury_penalty": game.get("home_injury_penalty", 0),
        "away_injury_penalty": game.get("away_injury_penalty", 0),
        "injury_diff": game.get(0),
        "home_missing_starters": game.get("home_missing_starters", 0),
        "away_missing_starters": game.get("away_missing_starters", 0),
        # v18 P1-CTX: Tournament context
        "is_conference_tournament": game.get("is_conference_tournament", False),
        "is_ncaa_tournament": game.get("is_ncaa_tournament", False),
        "is_bubble_game": game.get("is_bubble_game", False),
        "is_early_season": game.get("is_early_season", False),
        "importance_multiplier": game.get("importance_multiplier", 1.0),
        # ── Orphaned features ──
        "home_adj_oe": game.get("home_adj_oe", 105.0),
        "away_adj_oe": game.get("away_adj_oe", 105.0),
        "home_adj_de": game.get("home_adj_de", 105.0),
        "away_adj_de": game.get("away_adj_de", 105.0),
        "home_scoring_var": game.get("home_scoring_var", 12.0),
        "away_scoring_var": game.get("away_scoring_var", 12.0),
        "home_score_kurtosis": game.get("home_score_kurtosis", 0.0),
        "away_score_kurtosis": game.get("away_score_kurtosis", 0.0),
        "home_clutch_ratio": game.get("home_clutch_ratio", 0.5),
        "away_clutch_ratio": game.get("away_clutch_ratio", 0.5),
        "home_garbage_adj_ppp": game.get("home_garbage_adj_ppp", 1.0),
        "away_garbage_adj_ppp": game.get("away_garbage_adj_ppp", 1.0),
        "home_days_since_loss": game.get("home_days_since_loss", 5),
        "away_days_since_loss": game.get("away_days_since_loss", 5),
        "home_games_since_blowout_loss": game.get("home_games_since_blowout_loss", 10),
        "away_games_since_blowout_loss": game.get("away_games_since_blowout_loss", 10),
        "home_games_last_14": game.get("home_games_last_14", 4),
        "away_games_last_14": game.get("away_games_last_14", 4),
        "home_rest_effect": game.get("home_rest_effect", 0.0),
        "away_rest_effect": game.get("away_rest_effect", 0.0),
        "home_momentum_halflife": game.get("home_momentum_halflife", 1.0),
        "away_momentum_halflife": game.get("away_momentum_halflife", 1.0),
        "home_win_aging": game.get("home_win_aging", 1.0),
        "away_win_aging": game.get("away_win_aging", 1.0),
        "home_centrality": game.get("home_centrality", 1.0),
        "away_centrality": game.get("away_centrality", 1.0),
        "home_dow_effect": game.get("home_dow_effect", 0.0),
        "away_dow_effect": game.get("away_dow_effect", 0.0),
        "home_conf_balance": game.get("home_conf_balance", 0.0),
        "away_conf_balance": game.get("away_conf_balance", 0.0),
        "n_common_opps": game.get("n_common_opps", 0),
        "revenge_margin": game.get("revenge_margin", 0.0),
        "is_lookahead": game.get("is_lookahead", 0),
        "is_postseason": game.get("is_postseason", 0),
        "espn_ml_home": game.get("espn_ml_home", 0),
        "odds_api_spread_movement": game.get("odds_api_spread_movement", 0),
        "dk_spread_movement": game.get("dk_spread_movement", 0),
        "odds_api_total_movement": game.get("odds_api_total_movement", 0),
        "dk_total_movement": game.get("dk_total_movement", 0),
        "espn_ml_away": game.get("espn_ml_away", 0),
        # ── v3 fields ──
        "home_twopt_pct": game.get("home_twopt_pct", 0.48), "away_twopt_pct": game.get("away_twopt_pct", 0.48),
        "home_efg_pct": game.get("home_efg_pct", 0.50), "away_efg_pct": game.get("away_efg_pct", 0.50),
        "home_ts_pct": game.get("home_ts_pct", 0.53), "away_ts_pct": game.get("away_ts_pct", 0.53),
        "home_three_rate": game.get("home_three_rate", 0.35), "away_three_rate": game.get("away_three_rate", 0.35),
        "home_assist_rate": game.get("home_assist_rate", 0.55), "away_assist_rate": game.get("away_assist_rate", 0.55),
        "home_drb_pct": game.get("home_drb_pct", 0.70), "away_drb_pct": game.get("away_drb_pct", 0.70),
        "home_ppp": game.get("home_ppp", 1.0), "away_ppp": game.get("away_ppp", 1.0),
        "home_opp_efg_pct": game.get("home_opp_efg_pct", 0.50), "away_opp_efg_pct": game.get("away_opp_efg_pct", 0.50),
        "home_opp_to_rate": game.get("home_opp_to_rate", 0.18), "away_opp_to_rate": game.get("away_opp_to_rate", 0.18),
        "home_opp_fta_rate": game.get("home_opp_fta_rate", 0.30), "away_opp_fta_rate": game.get("away_opp_fta_rate", 0.30),
        "home_opp_orb_pct": game.get("home_opp_orb_pct", 0.28), "away_opp_orb_pct": game.get("away_opp_orb_pct", 0.28),
        "home_luck": game.get("home_luck", 0.0), "away_luck": game.get("away_luck", 0.0),
        "home_consistency": game.get("home_consistency", 15.0), "away_consistency": game.get("away_consistency", 15.0),
        "home_elo": game.get("home_elo", 1500), "away_elo": game.get("away_elo", 1500),
        "home_pyth_residual": game.get("home_pyth_residual", 0.0), "away_pyth_residual": game.get("away_pyth_residual", 0.0),
        "home_margin_trend": game.get("home_margin_trend", 0.0), "away_margin_trend": game.get("away_margin_trend", 0.0),
        "home_close_win_rate": game.get("home_close_win_rate", 0.5), "away_close_win_rate": game.get("away_close_win_rate", 0.5),
        "home_eff_vol_ratio": game.get("home_eff_vol_ratio", 1.0), "away_eff_vol_ratio": game.get("away_eff_vol_ratio", 1.0),
        "home_ceiling": game.get("home_ceiling", 15.0), "away_ceiling": game.get("away_ceiling", 15.0),
        "home_floor": game.get("home_floor", -10.0), "away_floor": game.get("away_floor", -10.0),
        "home_recovery_idx": game.get("home_recovery_idx", 0.0), "away_recovery_idx": game.get("away_recovery_idx", 0.0),
        "home_is_after_loss": game.get("home_is_after_loss", 0), "away_is_after_loss": game.get("away_is_after_loss", 0),
        "home_opp_suppression": game.get("home_opp_suppression", 0.0), "away_opp_suppression": game.get("away_opp_suppression", 0.0),
        "home_concentration": game.get("home_concentration", 0.0), "away_concentration": game.get("away_concentration", 0.0),
        "home_blowout_asym": game.get("home_blowout_asym", 0.0), "away_blowout_asym": game.get("away_blowout_asym", 0.0),
        "home_margin_accel": game.get("home_margin_accel", 0.0), "away_margin_accel": game.get("away_margin_accel", 0.0),
        "home_wl_momentum": game.get("home_wl_momentum", 0.0), "away_wl_momentum": game.get("away_wl_momentum", 0.0),
        "home_clutch_over_exp": game.get("home_clutch_over_exp", 0.0), "away_clutch_over_exp": game.get("away_clutch_over_exp", 0.0),
        "home_def_stability": game.get("home_def_stability", 10.0), "away_def_stability": game.get("away_def_stability", 10.0),
        "home_fatigue_load": game.get("home_fatigue_load", 0.0), "away_fatigue_load": game.get("away_fatigue_load", 0.0),
        "home_info_gain": game.get("home_info_gain", 0.0), "away_info_gain": game.get("away_info_gain", 0.0),
        "home_regression_pressure": game.get("home_regression_pressure", 0.0), "away_regression_pressure": game.get("away_regression_pressure", 0.0),
        "home_pace_adj_margin": game.get("home_pace_adj_margin", 0.0), "away_pace_adj_margin": game.get("away_pace_adj_margin", 0.0),
        "home_ft_pressure": game.get("home_ft_pressure", 0.0), "away_ft_pressure": game.get("away_ft_pressure", 0.0),
        "home_transition_dep": game.get("home_transition_dep", 0.0), "away_transition_dep": game.get("away_transition_dep", 0.0),
        "home_margin_autocorr": game.get("home_margin_autocorr", 0.0), "away_margin_autocorr": game.get("away_margin_autocorr", 0.0),
        "home_opp_adj_form": game.get("home_opp_adj_form", 0.0), "away_opp_adj_form": game.get("away_opp_adj_form", 0.0),
        "home_scoring_entropy": game.get("home_scoring_entropy", 1.5), "away_scoring_entropy": game.get("away_scoring_entropy", 1.5),
        "home_run_vulnerability": game.get("home_run_vulnerability", 0.0), "away_run_vulnerability": game.get("away_run_vulnerability", 0.0),
        "home_anti_fragility": game.get("home_anti_fragility", 0.0), "away_anti_fragility": game.get("away_anti_fragility", 0.0),
        "home_sos_trajectory": game.get("home_sos_trajectory", 0.0), "away_sos_trajectory": game.get("away_sos_trajectory", 0.0),
        "home_margin_skew": game.get("home_margin_skew", 0.0), "away_margin_skew": game.get("away_margin_skew", 0.0),
        "home_bimodal": game.get("home_bimodal", 0.0), "away_bimodal": game.get("away_bimodal", 0.0),
        "home_pit_sos": game.get("home_pit_sos", 1500), "away_pit_sos": game.get("away_pit_sos", 1500),
        "home_games_last_7": game.get("home_games_last_7", 2), "away_games_last_7": game.get("away_games_last_7", 2),
        "home_streak": game.get("home_streak", 0), "away_streak": game.get("away_streak", 0),
        "home_season_pct": game.get("home_season_pct", 0.5), "away_season_pct": game.get("away_season_pct", 0.5),
        "home_pts_off_to": game.get("home_pts_off_to", 12.0), "away_pts_off_to": game.get("away_pts_off_to", 12.0),
        "home_fastbreak_pts": game.get("home_fastbreak_pts", 10.0), "away_fastbreak_pts": game.get("away_fastbreak_pts", 10.0),
        "home_paint_pts": game.get("home_paint_pts", 30.0), "away_paint_pts": game.get("away_paint_pts", 30.0),
        "home_fouls": game.get("home_fouls", 17.0), "away_fouls": game.get("away_fouls", 17.0),
        "home_scoring_source_entropy": game.get("home_scoring_source_entropy", 1.5), "away_scoring_source_entropy": game.get("away_scoring_source_entropy", 1.5),
        "home_ft_dependency": game.get("home_ft_dependency", 0.20), "away_ft_dependency": game.get("away_ft_dependency", 0.20),
        "home_three_value": game.get("home_three_value", 0.35), "away_three_value": game.get("away_three_value", 0.35),
        "home_steal_foul_ratio": game.get("home_steal_foul_ratio", 0.40), "away_steal_foul_ratio": game.get("away_steal_foul_ratio", 0.40),
        "home_block_foul_ratio": game.get("home_block_foul_ratio", 0.20), "away_block_foul_ratio": game.get("away_block_foul_ratio", 0.20),
        "home_def_versatility": game.get("home_def_versatility", 0.5), "away_def_versatility": game.get("away_def_versatility", 0.5),
        "home_to_conversion": game.get("home_to_conversion", 1.0), "away_to_conversion": game.get("away_to_conversion", 1.0),
        "home_fg_divergence": game.get("home_fg_divergence", 0.0), "away_fg_divergence": game.get("away_fg_divergence", 0.0),
        "home_three_divergence": game.get("home_three_divergence", 0.0), "away_three_divergence": game.get("away_three_divergence", 0.0),
        "home_ppp_divergence": game.get("home_ppp_divergence", 0.0), "away_ppp_divergence": game.get("away_ppp_divergence", 0.0),
        "home_def_improvement": game.get("home_def_improvement", 0.0), "away_def_improvement": game.get("away_def_improvement", 0.0),
        "home_home_margin": game.get("home_home_margin", 0.0), "home_away_margin": game.get("home_away_margin", 0.0),
        "away_home_margin": game.get("away_home_margin", 0.0), "away_away_margin": game.get("away_away_margin", 0.0),
        "home_rhythm_disruption": game.get("home_rhythm_disruption", 0.0), "away_rhythm_disruption": game.get("away_rhythm_disruption", 0.0),
        "home_overreaction": game.get("home_overreaction", 0.0), "away_overreaction": game.get("away_overreaction", 0.0),
        "matchup_efg": game.get("matchup_efg", 0.0), "matchup_to": game.get("matchup_to", 0.0),
        "matchup_orb": game.get("matchup_orb", 0.0), "matchup_ft": game.get("matchup_ft", 0.0),
        "style_familiarity": game.get("style_familiarity", 0.5), "pace_leverage": game.get("pace_leverage", 0.0),
        "common_opp_diff": game.get("common_opp_diff", 0.0), "pace_control_diff": game.get("pace_control_diff", 0.0),
        "def_rest_advantage": game.get("def_rest_advantage", 0.0),
        "is_revenge_game": game.get("is_revenge_game", 0), "is_sandwich": game.get("is_sandwich", 0),
        "is_letdown": game.get("is_letdown", 0), "is_midweek": game.get("is_midweek", 0),
        "spread_regime": game.get("spread_regime", 1),
        "fatigue_x_quality": game.get("fatigue_x_quality", 0.0), "luck_x_spread": game.get("luck_x_spread", 0.0),
        "rest_x_defense": game.get("rest_x_defense", 0.0), "form_x_familiarity": game.get("form_x_familiarity", 0.0),
        "consistency_x_spread": game.get("consistency_x_spread", 0.0),
        # ═══ v19: ESPN extraction columns for live predictions ═══
        "espn_spread": game.get("espn_spread", 0.0),
        "espn_over_under": game.get("espn_over_under", 0.0),
        "espn_home_win_pct": game.get("espn_home_win_pct", 0.5),
        "espn_predictor_home_pct": game.get("espn_predictor_home_pct", 0.5),
        "home_1h_score": game.get("home_1h_score", 0.0),
        "away_1h_score": game.get("away_1h_score", 0.0),
        "home_2h_score": game.get("home_2h_score", 0.0),
        "away_2h_score": game.get("away_2h_score", 0.0),
        "home_1h_margin": game.get("home_1h_margin", 0.0),
        "home_2h_margin": game.get("home_2h_margin", 0.0),
        "home_largest_run": game.get("home_largest_run", 0.0),
        "away_largest_run": game.get("away_largest_run", 0.0),
        "home_runs_8plus": game.get("home_runs_8plus", 0),
        "away_runs_8plus": game.get("away_runs_8plus", 0),
        "home_drought_count": game.get("home_drought_count", 0),
        "away_drought_count": game.get("away_drought_count", 0),
        "home_longest_drought_sec": game.get("home_longest_drought_sec", 0.0),
        "away_longest_drought_sec": game.get("away_longest_drought_sec", 0.0),
        "lead_changes": game.get("lead_changes", 0),
        "ties": game.get("ties", 0),
        "home_time_with_lead_pct": game.get("home_time_with_lead_pct", 0.5),
        "largest_lead_home": game.get("largest_lead_home", 0.0),
        "largest_lead_away": game.get("largest_lead_away", 0.0),
        "home_clutch_ftm": game.get("home_clutch_ftm", 0),
        "home_clutch_fta": game.get("home_clutch_fta", 0),
        "away_clutch_ftm": game.get("away_clutch_ftm", 0),
        "away_clutch_fta": game.get("away_clutch_fta", 0),
        "garbage_time_seconds": game.get("garbage_time_seconds", 0),
        "is_garbage_time_game": game.get("is_garbage_time_game", 0),
        "home_star1_pts_share": game.get("home_star1_pts_share", 0.25),
        "away_star1_pts_share": game.get("away_star1_pts_share", 0.25),
        "home_top3_pts_share": game.get("home_top3_pts_share", 0.55),
        "away_top3_pts_share": game.get("away_top3_pts_share", 0.55),
        "home_minutes_hhi": game.get("home_minutes_hhi", 0.2),
        "away_minutes_hhi": game.get("away_minutes_hhi", 0.2),
        "home_bench_pts": game.get("home_bench_pts", 15.0),
        "away_bench_pts": game.get("away_bench_pts", 15.0),
        "home_bench_pts_share": game.get("home_bench_pts_share", 0.2),
        "away_bench_pts_share": game.get("away_bench_pts_share", 0.2),
        "home_players_used": game.get("home_players_used", 8),
        "away_players_used": game.get("away_players_used", 8),
        "home_starter_mins": game.get("home_starter_mins", 150.0),
        "away_starter_mins": game.get("away_starter_mins", 150.0),
        "halftime_home_win_prob": game.get("halftime_home_win_prob", 0.5),
        "wp_volatility": game.get("wp_volatility", 0.15),
        "wp_max_swing": game.get("wp_max_swing", 0.1),
        # ═══ v21: Venue features ═══
        "venue_capacity": game.get("venue_capacity", 8000),
        "venue_indoor": game.get("venue_indoor", 1),
        "attendance": game.get("attendance", 0),
        # ═══ v21: Rolling team tendency features (from prior games) ═══
        "home_roll_largest_run": game.get("home_roll_largest_run", 8.0),
        "away_roll_largest_run": game.get("away_roll_largest_run", 8.0),
        "home_roll_drought_rate": game.get("home_roll_drought_rate", 1.5),
        "away_roll_drought_rate": game.get("away_roll_drought_rate", 1.5),
        "home_roll_lead_changes": game.get("home_roll_lead_changes", 8.0),
        "away_roll_lead_changes": game.get("away_roll_lead_changes", 8.0),
        "home_roll_time_with_lead_pct": game.get("home_roll_time_with_lead_pct", 0.5),
        "away_roll_time_with_lead_pct": game.get("away_roll_time_with_lead_pct", 0.5),
        "home_roll_star1_share": game.get("home_roll_star1_share", 0.25),
        "away_roll_star1_share": game.get("away_roll_star1_share", 0.25),
        "home_roll_top3_share": game.get("home_roll_top3_share", 0.55),
        "away_roll_top3_share": game.get("away_roll_top3_share", 0.55),
        "home_roll_bench_share": game.get("home_roll_bench_share", 0.20),
        "away_roll_bench_share": game.get("away_roll_bench_share", 0.20),
        "home_roll_hhi": game.get("home_roll_hhi", 0.20),
        "away_roll_hhi": game.get("away_roll_hhi", 0.20),
        "home_roll_players_used": game.get("home_roll_players_used", 8.0),
        "away_roll_players_used": game.get("away_roll_players_used", 8.0),
        "home_roll_clutch_ft_pct": game.get("home_roll_clutch_ft_pct", 0.70),
        "away_roll_clutch_ft_pct": game.get("away_roll_clutch_ft_pct", 0.70),
        "home_roll_garbage_pct": game.get("home_roll_garbage_pct", 0.15),
        "away_roll_garbage_pct": game.get("away_roll_garbage_pct", 0.15),
        "home_roll_ats_pct": game.get("home_roll_ats_pct", 0.50),
        "away_roll_ats_pct": game.get("away_roll_ats_pct", 0.50),
        "home_roll_ats_n": game.get("home_roll_ats_n", 0),
        "away_roll_ats_n": game.get("away_roll_ats_n", 0),
        "home_roll_ats_margin": game.get("home_roll_ats_margin", 0.0),
        "away_roll_ats_margin": game.get("away_roll_ats_margin", 0.0),
    }
    # ═══ v24: Compute player ratings from starter_ids + player_ratings.json ═══
    _player_ratings = getattr(predict_ncaa, "_player_ratings", None)
    if _player_ratings is None:
        try:
            import json as _json
            with open("player_ratings.json") as _pf:
                predict_ncaa._player_ratings = _json.load(_pf)
            _player_ratings = predict_ncaa._player_ratings
        except FileNotFoundError:
            predict_ncaa._player_ratings = {}
            _player_ratings = {}

    for side in ["home", "away"]:
        ids_str = game.get(f"{side}_starter_ids", "")
        if isinstance(ids_str, str) and ids_str.strip():
            pids = ids_str.strip().split(",")
            ratings = [_player_ratings.get(pid, 0.0) for pid in pids]
            row_data[f"{side}_player_rating_sum"] = sum(ratings)
            row_data[f"{side}_player_rating_avg"] = np.mean(ratings) if ratings else 0.0
            row_data[f"{side}_weakest_starter"] = min(ratings) if ratings else 0.0
            row_data[f"{side}_starter_variance"] = float(np.std(ratings)) if len(ratings) > 1 else 0.0
        else:
            row_data[f"{side}_player_rating_sum"] = 0.0
            row_data[f"{side}_player_rating_avg"] = 0.0
            row_data[f"{side}_weakest_starter"] = 0.0
            row_data[f"{side}_starter_variance"] = 0.0

    # ═══ v24: Lookup h2h, conference, recent form from JSON caches ═══
    import json as _json

    # H2H lookup
    _h2h = getattr(predict_ncaa, "_h2h_lookup", None)
    if _h2h is None:
        try:
            with open("h2h_lookup.json") as _f:
                predict_ncaa._h2h_lookup = _json.load(_f)
            _h2h = predict_ncaa._h2h_lookup
        except FileNotFoundError:
            predict_ncaa._h2h_lookup = {}
            _h2h = {}

    h_tid = str(game.get("home_team_id", ""))
    a_tid = str(game.get("away_team_id", ""))
    h2h_key = f"{h_tid}:{a_tid}"
    h2h_data = _h2h.get(h2h_key, {})
    row_data["h2h_margin_avg"] = h2h_data.get("margin_avg", 0.0)
    row_data["h2h_home_win_rate"] = h2h_data.get("home_win_rate", 0.0)

    # Conference lookup
    _conf = getattr(predict_ncaa, "_conf_lookup", None)
    if _conf is None:
        try:
            with open("conference_lookup.json") as _f:
                predict_ncaa._conf_lookup = _json.load(_f)
            _conf = predict_ncaa._conf_lookup
        except FileNotFoundError:
            predict_ncaa._conf_lookup = {"teams": {}, "strength": {}}
            _conf = predict_ncaa._conf_lookup

    _conf_teams = _conf.get("teams", {})
    _conf_strength = _conf.get("strength", {})
    h_conf = _conf_teams.get(h_tid, "")
    a_conf = _conf_teams.get(a_tid, "")
    if h_conf and a_conf and h_conf != a_conf:
        row_data["conf_strength_diff"] = _conf_strength.get(h_conf, 0) - _conf_strength.get(a_conf, 0)
        row_data["cross_conf_flag"] = 1
    else:
        row_data["conf_strength_diff"] = 0.0
        row_data["cross_conf_flag"] = 0

    # Recent form lookup
    _form = getattr(predict_ncaa, "_form_lookup", None)
    if _form is None:
        try:
            with open("recent_form_lookup.json") as _f:
                predict_ncaa._form_lookup = _json.load(_f)
            _form = predict_ncaa._form_lookup
        except FileNotFoundError:
            predict_ncaa._form_lookup = {}
            _form = {}

    h_form = _form.get(h_tid, 0.5)
    a_form = _form.get(a_tid, 0.5)
    row_data["recent_form_diff"] = h_form - a_form

    # Lineup features (pass-through — computed by frontend/sync if available)
    row_data["home_lineup_changes"] = game.get("home_lineup_changes", 0)
    row_data["away_lineup_changes"] = game.get("away_lineup_changes", 0)
    row_data["home_lineup_stability_5g"] = game.get("home_lineup_stability_5g", 1.0)
    row_data["away_lineup_stability_5g"] = game.get("away_lineup_stability_5g", 1.0)
    row_data["home_starter_games_together"] = game.get("home_starter_games_together", 0)
    row_data["away_starter_games_together"] = game.get("away_starter_games_together", 0)

    row = pd.DataFrame([row_data])
    X_built = ncaa_build_features(row)

    # Ensure feature alignment with trained model
    for col in bundle["feature_cols"]:
        if col not in X_built.columns:
            X_built[col] = 0
    X_built = X_built[bundle["feature_cols"]]

    X_s      = bundle["scaler"].transform(X_built)
    raw_margin = float(bundle["reg"].predict(X_s)[0])
    raw_win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])

    # R6 FIX: Apply bias correction to margin prediction
    bias = bundle.get("bias_correction", 0.0)
    margin = raw_margin - bias

    # R4 FIX: Apply isotonic calibration to win probability
    isotonic = bundle.get("isotonic")
    if isotonic is not None:
        win_prob = float(isotonic.predict([raw_win_prob])[0])
    else:
        win_prob = raw_win_prob

    # WIN PROBABILITY CAP: Clamp to [0.05, 0.95] to allow large spreads.
    # Previous [0.12, 0.88] cap was too tight — it capped effective spreads
    # at ~16 pts (at sigma=16), causing 22+ pt gaps vs Vegas on blowout games
    # and generating false SPREADLEAN signals. NCAA regular season has genuine
    # 95%+ probability games (top-10 vs sub-300 teams). The moneyline display
    # cap (ML_CAP=800 in the frontend) handles extreme ML values separately.
    # 0.95 → ML -1900, 0.05 → ML +1900 (capped at ±800 for display).
    win_prob = max(0.05, min(0.95, win_prob))

    shap_vals = bundle["explainer"].shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_out = [
        {"feature": f, "shap": round(float(v), 4), "value": round(float(X_built[f].iloc[0]), 3)}
        for f, v in zip(bundle["feature_cols"], shap_vals[0])
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "NCAAB",
        "ml_margin": round(margin, 2),
        "ml_margin_raw": round(raw_margin, 2),  # before bias correction
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "ml_win_prob_raw": round(raw_win_prob, 4),  # before isotonic
        "bias_correction_applied": round(bias, 3),
        "shap": shap_out,
        "model_meta": {"n_train": bundle["n_train"], "mae_cv": bundle["mae_cv"],
                       "model_type": bundle.get("model_type", "unknown"),
                       "trained_at": bundle["trained_at"]},
    }

# ═══════════════════════════════════════════════════════════════
# NFL MODEL
# ═══════════════════════════════════════════════════════════════
