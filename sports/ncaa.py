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

    # ── Market line features ──
    df["market_spread"] = pd.to_numeric(df["market_spread_home"] if "market_spread_home" in df.columns else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else (df["ou_total"] if "ou_total" in df.columns else pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)
    df["has_market"] = ((df["market_spread"] != 0) | (df["market_total"] != 0)).astype(int)
    _ncaa_pred_spread = pd.to_numeric(df["spread_home"] if "spread_home" in df.columns else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    # BUGFIX 1: Sign alignment — model spread is +positive when home wins (e.g. +7.8),
    # but market_spread_home is -negative when home is favored (e.g. -7.5, Vegas convention).
    # Negate market so both point the same direction before subtracting.
    # Wrong: spread_vs_market = 7.8 - (-7.5) = +15.3 → massively inflates home win prob
    # Fixed: spread_vs_market = 7.8 - (+7.5) = +0.3 → correct small model disagreement
    # BUGFIX 2: Zero out when no market line (has_market=0) to prevent pred_spread leaking in.
    df["spread_vs_market"] = (_ncaa_pred_spread + df["market_spread"]) * df["has_market"]
    # tourney_x_em, early_x_form REMOVED (AUDIT P4) — correlated with components

    feature_cols = [
        # R1: Decomposed efficiency + HCA
        "neutral_em_diff", "hca_pts", "neutral",
        # AUDIT P4: heur_win_prob_capped REMOVED — redundant with raw stats it derives from
        # Raw stats — differentials
        "ppg_diff", "opp_ppg_diff", "fgpct_diff", "threepct_diff",
        "orb_pct_diff", "fta_rate_diff", "ato_diff",
        "def_fgpct_diff", "steals_diff", "blocks_diff",
        "sos_diff", "form_diff", "rank_diff", "win_pct_diff",
        # Turnover quality
        "to_margin_diff", "steals_to_diff",
        # Context
        "tempo_avg", "is_ranked_game", "is_top_matchup",
        # R3: Conference + season phase
        "is_conf_game", "season_phase",
        # R5: Schedule fatigue
        "rest_diff", "either_b2b",
        # Market line signal
        "market_spread", "market_total", "spread_vs_market", "has_market",
        # AUDIT P4: ppg_x_sos, em_x_conf, injury_x_em, tourney_x_em, early_x_form REMOVED
        # P1-INJ: Injury signal (components only, no interaction)
        "injury_diff", "starters_diff", "any_injury_flag",
        # P1-CTX: Tournament context (components only, no interaction)
        "is_conf_tourney", "is_ncaa_tourney", "is_bubble", "is_early",
        "importance",
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
    for inj_col in ["injury_diff", "home_missing_starters", "away_missing_starters",
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

        cv_folds = min(5, n)  # 5 for Railway; set 10 for local
        fit_weights = sample_weights if sample_weights is not None else np.ones(n)

        if n >= 200:
            # ── Stacking ensemble at 160 estimators (sweep-optimized) ──
            rf_reg = RandomForestRegressor(
                n_estimators=160, max_depth=6,
                min_samples_leaf=15, max_features=0.7,
                random_state=42, n_jobs=1,  # Railway: single core; set -1 for local
            )
            if HAS_XGB:
                xgb_reg = XGBRegressor(n_estimators=160, max_depth=4, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, random_state=42, tree_method="hist", verbosity=0)
            if HAS_CAT:
                cat_reg = CatBoostRegressor(iterations=160, depth=4, learning_rate=0.06, subsample=0.8, min_data_in_leaf=20, random_seed=42, verbose=0)

            ensemble_parts = []
            if HAS_XGB: ensemble_parts.append('XGB')
            if HAS_CAT: ensemble_parts.append('CAT')
            ensemble_parts.append('RF')
            ensemble_label = '+'.join(ensemble_parts)
            print(f"  NCAAB: Training stacking ensemble on {n} rows (ts-cv, {ensemble_label}, 160 est)...")

            reg_models = {"rf": rf_reg}
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
        "injury_diff": game.get("injury_diff", 0),
        "home_missing_starters": game.get("home_missing_starters", 0),
        "away_missing_starters": game.get("away_missing_starters", 0),
        # v18 P1-CTX: Tournament context
        "is_conference_tournament": game.get("is_conference_tournament", False),
        "is_ncaa_tournament": game.get("is_ncaa_tournament", False),
        "is_bubble_game": game.get("is_bubble_game", False),
        "is_early_season": game.get("is_early_season", False),
        "importance_multiplier": game.get("importance_multiplier", 1.0),
    }
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
