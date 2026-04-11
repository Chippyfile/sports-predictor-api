import numpy as _np

class EnsembleRegressor:
    """Averaging ensemble — must be at module level for pickle deserialization on Railway."""
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        return _np.mean([m.predict(X) for m in self.models], axis=0)
    @property
    def coef_(self):
        if hasattr(self.models[0], 'coef_'):
            return self.models[0].coef_
        return None

MLB_NEGBIN_K_DEFAULT = 0.60
SEASON_CONSTANTS = {
    2015: {"lg_woba": 0.313, "woba_scale": 1.24, "lg_rpg": 4.25, "lg_fip": 3.97, "pa_pg": 38.0},
    2016: {"lg_woba": 0.318, "woba_scale": 1.21, "lg_rpg": 4.48, "lg_fip": 4.19, "pa_pg": 38.0},
    2017: {"lg_woba": 0.321, "woba_scale": 1.21, "lg_rpg": 4.65, "lg_fip": 4.36, "pa_pg": 38.1},
    2018: {"lg_woba": 0.315, "woba_scale": 1.23, "lg_rpg": 4.45, "lg_fip": 4.15, "pa_pg": 37.9},
    2019: {"lg_woba": 0.320, "woba_scale": 1.17, "lg_rpg": 4.83, "lg_fip": 4.51, "pa_pg": 38.2},
    2021: {"lg_woba": 0.313, "woba_scale": 1.22, "lg_rpg": 4.53, "lg_fip": 4.26, "pa_pg": 37.9},
    2022: {"lg_woba": 0.310, "woba_scale": 1.24, "lg_rpg": 4.28, "lg_fip": 4.01, "pa_pg": 37.6},
    2023: {"lg_woba": 0.318, "woba_scale": 1.21, "lg_rpg": 4.62, "lg_fip": 4.33, "pa_pg": 37.8},
    2024: {"lg_woba": 0.317, "woba_scale": 1.25, "lg_rpg": 4.38, "lg_fip": 4.17, "pa_pg": 37.8},
    2025: {"lg_woba": 0.315, "woba_scale": 1.24, "lg_rpg": 4.30, "lg_fip": 4.10, "pa_pg": 37.8},
    2026: {"lg_woba": 0.315, "woba_scale": 1.24, "lg_rpg": 4.30, "lg_fip": 4.10, "pa_pg": 37.8},
}
DEFAULT_CONSTANTS = {"lg_woba": 0.315, "woba_scale": 1.24, "lg_rpg": 4.30, "lg_fip": 4.10, "pa_pg": 37.8}

FIP_COEFF = 0.55
HFA_RUNS = 0.16   # DEPRECATED by v3 audit — HFA now in probability space via PARK_HFA dict
# ── Per-park HFA values (mirrors PARK_FACTORS.hfa in mlb.js frontend) ──
# Applied in probability space after Pythagenpat, not in run space.
PARK_HFA = {
    108: 0.033, 109: 0.038, 110: 0.034, 111: 0.037, 112: 0.036,
    113: 0.034, 114: 0.035, 115: 0.048, 116: 0.034, 117: 0.042,
    118: 0.035, 119: 0.038, 120: 0.034, 121: 0.035, 133: 0.032,
    134: 0.034, 135: 0.037, 136: 0.039, 137: 0.038, 138: 0.035,
    139: 0.041, 140: 0.041, 141: 0.036, 142: 0.035, 143: 0.036,
    144: 0.035, 145: 0.033, 146: 0.040, 147: 0.036, 158: 0.035,
}
# H-5 FIX: Dome parks where outdoor weather adjustments should be skipped
_DOME_PARKS = {109, 117, 136, 139, 140, 141, 146, 158}
# Abbr → team ID for dome detection at serve time
_ABBR_TO_ID = {
    "LAA":108,"ARI":109,"BAL":110,"BOS":111,"CHC":112,"CIN":113,"CLE":114,
    "COL":115,"DET":116,"HOU":117,"KC":118,"LAD":119,"WSH":120,"NYM":121,
    "OAK":133,"PIT":134,"SD":135,"SEA":136,"SF":137,"STL":138,"TB":139,
    "TEX":140,"TOR":141,"MIN":142,"PHI":143,"ATL":144,"CWS":145,"MIA":146,
    "NYY":147,"MIL":158,
}
def _is_dome(home_abbr):
    """Check if home team plays in a dome/retractable roof park."""
    return _ABBR_TO_ID.get((home_abbr or "").upper().strip(), 0) in _DOME_PARKS
import math as _math
from datetime import timedelta as _td

# ── AUDIT FIX F-02: Team city coordinates for travel distance (Haversine) ──
_MLB_CITY_COORDS = {
    "ARI": (33.45, -112.07), "ATL": (33.75, -84.39), "BAL": (39.29, -76.61),
    "BOS": (42.36, -71.06), "CHC": (41.88, -87.63), "CIN": (39.10, -84.51),
    "CLE": (41.50, -81.69), "COL": (39.74, -104.99), "CWS": (41.88, -87.63),
    "DET": (42.33, -83.05), "HOU": (29.76, -95.37), "KC": (39.10, -94.58),
    "LAA": (33.80, -117.88), "LAD": (34.05, -118.24), "MIA": (25.76, -80.19),
    "MIL": (43.04, -87.91), "MIN": (44.98, -93.27), "NYM": (40.71, -74.01),
    "NYY": (40.71, -74.01), "OAK": (37.80, -122.27), "PHI": (39.95, -75.17),
    "PIT": (40.44, -79.99), "SD": (32.72, -117.16), "SEA": (47.61, -122.33),
    "SF": (37.77, -122.42), "STL": (38.63, -90.20), "TB": (27.77, -82.64),
    "TEX": (32.74, -97.11), "TOR": (43.65, -79.38), "WSH": (38.91, -77.04),
}

def _haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat, dlon = _math.radians(lat2-lat1), _math.radians(lon2-lon1)
    a = _math.sin(dlat/2)**2 + _math.cos(_math.radians(lat1))*_math.cos(_math.radians(lat2))*_math.sin(dlon/2)**2
    return R * 2 * _math.asin(_math.sqrt(a))

def _compute_travel_miles(team_abbr, game_date_str, current_home_team):
    """Compute travel miles for away team from their previous game city to today's park."""
    try:
        abbr = (team_abbr or "").upper().strip()
        home = (current_home_team or "").upper().strip()
        if abbr not in _MLB_CITY_COORDS or home not in _MLB_CITY_COORDS:
            return 0.0
        prev = sb_get("mlb_predictions",
            f"game_date=lt.{game_date_str}&or=(home_team.eq.{abbr},away_team.eq.{abbr})"
            f"&order=game_date.desc&limit=1&select=home_team")
        if not prev:
            return 0.0
        prev_city = prev[0].get("home_team", "").upper()
        if prev_city not in _MLB_CITY_COORDS:
            return 0.0
        dest = _MLB_CITY_COORDS[home]
        src = _MLB_CITY_COORDS[prev_city]
        return _haversine_miles(src[0], src[1], dest[0], dest[1])
    except Exception:
        return 0.0

def _compute_series_game_num(home_team, away_team, game_date_str):
    """Determine game number in current series (1-4)."""
    try:
        recent = sb_get("mlb_predictions",
            f"home_team=eq.{home_team}&away_team=eq.{away_team}"
            f"&game_date=lte.{game_date_str}&order=game_date.desc&limit=5&select=game_date")
        if not recent:
            return 1
        from datetime import datetime as _dtparse
        target = _dtparse.strptime(game_date_str, "%Y-%m-%d")
        count = 0
        for r in recent:
            gd = _dtparse.strptime(r["game_date"], "%Y-%m-%d")
            if abs((target - gd).days) <= 4:
                count += 1
            else:
                break
        return min(max(count, 1), 4)
    except Exception:
        return 1

import numpy as np, pandas as pd, traceback as _tb, shap
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import nbinom
from db import sb_get, save_model, load_model
from dynamic_constants import compute_mlb_season_constants, MLB_DEFAULT_CONSTANTS
from ml_utils import _time_series_oof, StackedClassifier
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

def _fit_negbin_k(run_series):
    """
    Fit the overdispersion parameter k for a Negative Binomial distribution
    to a series of run totals using method of moments.
    NegBin variance = μ + μ²/k  →  k = μ² / (variance - μ)
    Returns k clamped to [0.30, 1.20] for stability.
    Winsorized at 5th/95th percentile to exclude blowouts.
    """
    s = run_series.dropna()
    if len(s) < 20:
        mu, var = s.mean(), s.var()
    else:
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        s = s.clip(lo, hi)
        mu, var = s.mean(), s.var()
    if var <= mu or mu <= 0:
        return MLB_NEGBIN_K_DEFAULT
    k = (mu ** 2) / (var - mu)
    return float(np.clip(k, 0.30, 1.20))

def calibrate_mlb_dispersion():
    """
    Fetch completed MLB games, fit NegBin k separately for home and away runs,
    store the calibrated parameters, and return a summary.
    """
    rows = sb_get("mlb_predictions",
                  "result_entered=eq.true&actual_home_runs=not.is.null"
                  "&game_type=eq.R&select=actual_home_runs,actual_away_runs")
    if len(rows) < 30:
        return {
            "warning": f"Only {len(rows)} completed games — using default k={MLB_NEGBIN_K_DEFAULT}. "
                       "Need 30+ for reliable dispersion estimate.",
            "k_home": MLB_NEGBIN_K_DEFAULT,
            "k_away": MLB_NEGBIN_K_DEFAULT,
        }

    df = pd.DataFrame(rows)
    k_home = _fit_negbin_k(df["actual_home_runs"].astype(float))
    k_away = _fit_negbin_k(df["actual_away_runs"].astype(float))
    k_avg  = round((k_home + k_away) / 2, 4)

    bundle = {
        "k_home": k_home,
        "k_away": k_away,
        "k_avg":  k_avg,
        "n_games": len(df),
        "mean_home_runs": round(df["actual_home_runs"].astype(float).mean(), 3),
        "mean_away_runs": round(df["actual_away_runs"].astype(float).mean(), 3),
        "calibrated_at": datetime.utcnow().isoformat(),
    }
    save_model("mlb_dispersion", bundle)
    return {"status": "calibrated", **bundle}

def _get_mlb_k():
    """Load calibrated k values, fall back to default if not yet run."""
    disp = load_model("mlb_dispersion")
    if disp:
        return disp.get("k_home", MLB_NEGBIN_K_DEFAULT), disp.get("k_away", MLB_NEGBIN_K_DEFAULT)
    return MLB_NEGBIN_K_DEFAULT, MLB_NEGBIN_K_DEFAULT

# ═══════════════════════════════════════════════════════════════
# STACKING ENSEMBLE WRAPPERS (module-level for pickle/joblib compatibility)
# ═══════════════════════════════════════════════════════════════

def mlb_build_features(df):
    """
    Build feature matrix. Works on both mlb_predictions and mlb_historical rows.
    Uses raw game inputs (wOBA, FIP, park factor) when available — these are
    the real predictive signal. Heuristic outputs (pred_runs) used as fallback only.

    v3 FIXES:
      - Added K/9 and BB/9 as ML features (were only in heuristic, major signal loss)
      - Use sp_fip_known flag instead of fragile != 4.25 comparison for SP FIP fallback
      - Added has_sp_fip flag so model learns to weight starter FIP vs team FIP
      - Added league run environment feature (lg_rpg) so model knows offensive era context
    """
    df = df.copy()

    # ── Raw inputs (present in mlb_historical, optionally in mlb_predictions) ──
    raw_cols = {
        "home_woba":        0.314,
        "away_woba":        0.314,
        "home_sp_fip":      4.25,   # starter FIP (historical table)
        "away_sp_fip":      4.25,
        "home_fip":         4.25,   # fallback if sp_fip missing
        "away_fip":         4.25,
        "home_bullpen_era": 4.10,
        "away_bullpen_era": 4.10,
        "park_factor":      1.00,
        "temp_f":           70.0,
        "wind_mph":         5.0,
        "wind_out_flag":    0.0,
        "home_rest_days":   4.0,
        "away_rest_days":   4.0,
        "home_travel":      0.0,
        "away_travel":      0.0,
        # K/9 and BB/9 — FIX: these were only in heuristic, not ML features
        "home_k9":          8.5,
        "away_k9":          8.5,
        "home_bb9":         3.2,
        "away_bb9":         3.2,
        # SP innings pitched + defensive OAA
        "home_sp_ip":       5.5,
        "away_sp_ip":       5.5,
        "home_def_oaa":     0.0,
        "away_def_oaa":     0.0,
        # Enhancement: Platoon splits (wOBA delta from L/R matchup)
        "home_platoon_delta": 0.0,
        "away_platoon_delta": 0.0,
        # Enhancement: Lineup confirmation flags
        "home_lineup_confirmed": 0.0,
        "away_lineup_confirmed": 0.0,
    }
    for col, default in raw_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # ── SP FIP fallback: use sp_fip_known flag when available (more robust than != 4.25) ──
    if "home_sp_fip_known" in df.columns:
        home_sp_known = pd.to_numeric(df["home_sp_fip_known"], errors="coerce").fillna(0).astype(bool)
        away_sp_known = pd.to_numeric(df["away_sp_fip_known"], errors="coerce").fillna(0).astype(bool)
        df["home_starter_fip"] = np.where(home_sp_known, df["home_sp_fip"], df["home_fip"])
        df["away_starter_fip"] = np.where(away_sp_known, df["away_sp_fip"], df["away_fip"])
        df["has_sp_fip"] = (home_sp_known & away_sp_known).astype(int)
    else:
        df["home_starter_fip"] = df["home_sp_fip"].where(df["home_sp_fip"] != 4.25, df["home_fip"])
        df["away_starter_fip"] = df["away_sp_fip"].where(df["away_sp_fip"] != 4.25, df["away_fip"])
        df["has_sp_fip"] = ((df["home_sp_fip"] != 4.25) & (df["away_sp_fip"] != 4.25)).astype(int)

    # ── Derived features from raw inputs ──
    df["woba_diff"]        = df["home_woba"] - df["away_woba"]
    df["fip_diff"]         = df["home_starter_fip"] - df["away_starter_fip"]
    df["bullpen_era_diff"] = df["home_bullpen_era"] - df["away_bullpen_era"]
    df["rest_diff"]        = df["home_rest_days"] - df["away_rest_days"]
    df["travel_diff"]      = df["home_travel"] - df["away_travel"]

    # AUDIT v4 Finding 4: Neutralize weather for dome parks in training data
    # Matches serve-time dome fix in predict_mlb(). Without this, training sees
    # real outdoor weather for dome parks but serve sees neutral 70/0/0.
    _dome_col = None
    for c in ["home_team_id", "homeTeamId", "park_id"]:
        if c in df.columns:
            _dome_col = c
            break
    if _dome_col:
        _is_dome_mask = pd.to_numeric(df[_dome_col], errors="coerce").isin(_DOME_PARKS)
        df.loc[_is_dome_mask, "temp_f"] = 70.0
        df.loc[_is_dome_mask, "wind_mph"] = 0.0
        df.loc[_is_dome_mask, "wind_out_flag"] = 0.0

    df["is_warm"]          = (df["temp_f"] > 75).astype(int)
    df["is_cold"]          = (df["temp_f"] < 45).astype(int)
    df["wind_out"]         = df["wind_out_flag"].astype(int)

    # K/9 and BB/9 derived features — FIX: strong predictors missing from ML
    df["k9_diff"]    = df["home_k9"] - df["away_k9"]
    df["bb9_diff"]   = df["home_bb9"] - df["away_bb9"]
    df["k_bb_home"]  = df["home_k9"] - df["home_bb9"]
    df["k_bb_away"]  = df["away_k9"] - df["away_bb9"]
    df["k_bb_diff"]  = df["k_bb_home"] - df["k_bb_away"]

    # SP innings & bullpen exposure + defensive OAA
    # AUDIT FIX F1: Changed baseline from 9.0 to 5.5. The 9.0 baseline made this feature
    # near-constant (every starter has 9.0-5.5=3.5 "exposure"), killing the signal.
    # 5.5 baseline means only short starters (< 5.5 IP avg) generate a penalty,
    # matching the intent of the JS heuristic engine which uses 5.0.
    df["sp_ip_diff"]       = df["home_sp_ip"] - df["away_sp_ip"]
    df["home_bp_exposure"] = np.maximum(0, 5.5 - df["home_sp_ip"]) * (df["home_bullpen_era"] / 4.10)
    df["away_bp_exposure"] = np.maximum(0, 5.5 - df["away_sp_ip"]) * (df["away_bullpen_era"] / 4.10)
    df["bp_exposure_diff"] = df["home_bp_exposure"] - df["away_bp_exposure"]
    df["def_oaa_diff"]     = df["home_def_oaa"] - df["away_def_oaa"]

    # ── Enhancement: Platoon advantage differential ──
    # Positive = home team has larger platoon advantage vs opposing starter
    df["platoon_diff"] = df["home_platoon_delta"] - df["away_platoon_delta"]

    # ── Enhancement: Starter FIP spread (absolute gap between starters) ──
    # Ace vs #5 starter creates high confidence regardless of direction
    # This captures matchup lopsidedness that fip_diff's sign obscures
    df["sp_fip_spread"] = (df["home_starter_fip"] - df["away_starter_fip"]).abs()

    # ── Enhancement: Lineup confirmation (both lineups confirmed = more reliable prediction) ──
    df["both_lineups_confirmed"] = (
        (df["home_lineup_confirmed"] == 1) & (df["away_lineup_confirmed"] == 1)
    ).astype(int)

    # ── FIX ML2: Interaction features (capture non-linear relationships) ──
    # starter_quality × bullpen_quality: short-start ace with bad bullpen is different
    df["fip_x_bullpen"] = df["fip_diff"] * df["bullpen_era_diff"]
    # offensive advantage × park factor: wOBA edge compounds in hitter-friendly parks
    df["woba_x_park"] = df["woba_diff"] * df["park_factor"]
    # wind × pitching advantage: wind out compresses pitching quality advantages
    df["wind_x_fip"] = df["wind_out"].astype(float) * df["fip_diff"]

    # ── League run environment context ──
    if "season" in df.columns:
        # Dynamic league averages (derived from historical data at training time)
        _dyn = getattr(mlb_build_features, "_dynamic_constants", None)
        if _dyn:
            df["lg_rpg"] = df["season"].map(
                lambda s: _dyn.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
                if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
            )
        else:
            df["lg_rpg"] = df["season"].map(
                lambda s: SEASON_CONSTANTS.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
                if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
            )
    else:
        df["lg_rpg"] = DEFAULT_CONSTANTS["lg_rpg"]

    # ── Heuristic outputs (backfilled for historical, live for current season) ──
    for col, default in [("pred_home_runs", 0.0), ("pred_away_runs", 0.0),
                         ("win_pct_home", 0.5), ("ou_total", 9.0),
                         ("model_ml_home", 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    df["run_diff_pred"]  = df["pred_home_runs"] - df["pred_away_runs"]
    df["total_pred"]     = df["pred_home_runs"] + df["pred_away_runs"]
    df["home_fav"]       = (df["model_ml_home"] < 0).astype(int)
    df["ou_gap"]         = df["total_pred"] - df["ou_total"]
    df["has_heuristic"]  = (df["total_pred"] > 0).astype(int)

    # ═══════════════════════════════════════════════════════════════
    # FIX S2/ML1: PRUNED FEATURE SET (47 → 25)
    # ═══════════════════════════════════════════════════════════════
    # Rationale: VIF analysis showed 15+ features were linear combinations.
    # Keep DIFFS only (woba_diff, fip_diff, etc.) — drop individual home/away
    # components that the diff already captures. Drop 9 K/BB features → keep
    # k_bb_diff only. Drop 8 heuristic outputs → keep run_diff_pred + has_heuristic.
    # Add 3 interaction features (Finding ML2) for non-linear signal.
    #
    # Expected impact: +1-1.5% accuracy from reduced multicollinearity,
    # more stable stacking ensemble, better generalization.
    # ── Market line features ──
    df["market_spread"] = pd.to_numeric(df["market_spread_home"] if "market_spread_home" in df.columns else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else (df["ou_total"] if "ou_total" in df.columns else pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)
    df["has_market"] = ((df["market_spread"] != 0) | (df["market_total"] != 0)).astype(int)
    df["spread_vs_market"] = df["run_diff_pred"] - df["market_spread"]

    feature_cols = [
        # Offensive differential (primary signal)
        "woba_diff",
        # Pitching differentials
        "fip_diff", "has_sp_fip",
        "bullpen_era_diff",
        # Strikeout/walk composite differential
        "k_bb_diff",
        # SP workload & defense
        "sp_ip_diff", "bp_exposure_diff", "def_oaa_diff",
        # Park & environment
        "park_factor",
        "temp_f", "wind_mph", "wind_out",
        "is_warm", "is_cold",
        # Context
        "rest_diff", "travel_diff",
        "lg_rpg",
        # Interaction features (Finding ML2)
        "fip_x_bullpen", "woba_x_park", "wind_x_fip",
        # Heuristic signal (now backfilled for all rows)
        "run_diff_pred", "has_heuristic",
        # Enhancement: Platoon, starter spread, lineup confirmation
        "platoon_diff", "sp_fip_spread", "both_lineups_confirmed",
        # Market line signal
        "market_spread", "market_total", "spread_vs_market", "has_market",
        # ── Advanced features (v7) ──
        "pyth_residual_diff", "babip_luck_diff", "scoring_entropy_diff",
        "first_inn_rate_diff", "clutch_divergence_diff", "opp_adj_form_diff",
        "ump_run_env", "series_game_num",
        "scoring_entropy_combined", "first_inn_rate_combined",
        "sp_relative_fip_diff", "temp_x_park",
    ]

    # ── Advanced features: compute what we can, default the rest ──
    advanced_defaults = {
        "pyth_residual_diff": 0.0, "babip_luck_diff": 0.0,
        "scoring_entropy_diff": 0.0, "first_inn_rate_diff": 0.0,
        "clutch_divergence_diff": 0.0, "opp_adj_form_diff": 0.0,
        "ump_run_env": 8.5, "series_game_num": 1.0,
        "scoring_entropy_combined": 5.0, "first_inn_rate_combined": 0.8,
    }
    for col, default in advanced_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # sp_relative_fip: starter FIP relative to own team FIP (computable at serve time)
    if "sp_relative_fip_diff" in df.columns:
        df["sp_relative_fip_diff"] = pd.to_numeric(df["sp_relative_fip_diff"], errors="coerce").fillna(0)
    else:
        df["sp_relative_fip_diff"] = (
            (df["home_starter_fip"] - df["home_fip"]) - (df["away_starter_fip"] - df["away_fip"])
        ).fillna(0)

    # temp × park interaction (computable at serve time)
    if "temp_x_park" in df.columns:
        df["temp_x_park"] = pd.to_numeric(df["temp_x_park"], errors="coerce").fillna(0)
    else:
        df["temp_x_park"] = ((df["temp_f"] - 70) / 30.0) * df["park_factor"]

    return df[feature_cols].fillna(0)


def _mlb_merge_historical(current_df):
    """
    Fetch mlb_historical (2015-present) and combine with current season predictions.
    Historical rows use real game features directly — no data leakage.
    Applies season_weight for recency weighting.
    Excludes outlier seasons (COVID 2020, etc.).
    """
    hist_rows = sb_get(
        "mlb_historical",
        "is_outlier_season=eq.0&actual_home_runs=not.is.null&select=*&order=season.desc&limit=100000"
    )
    if not hist_rows:
        print("  WARNING: mlb_historical returned no rows — training on current season only")
        return current_df, None

    hist_df = pd.DataFrame(hist_rows)

    # Ensure numeric types on key columns
    numeric_cols = ["actual_home_runs", "actual_away_runs", "home_win",
                    "home_woba", "away_woba", "home_sp_fip", "away_sp_fip",
                    "home_fip", "away_fip", "home_bullpen_era", "away_bullpen_era",
                    "park_factor", "temp_f", "wind_mph", "wind_out_flag",
                    "home_rest_days", "away_rest_days", "home_travel", "away_travel",
                    "season_weight"]
    for col in numeric_cols:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors="coerce")

    # ── Heuristic backfill: compute real pre-game predictions ──
    # Instead of win_pct_home=0.5, run the Python heuristic engine on each
    # historical row. This gives the ML model realistic heuristic signal
    # (run projections, win probabilities) for training, matching what the
    # live JS engine produces for current-season games.
    # Uses season-specific FanGraphs constants and date-aware leak regression.
    print("  Backfilling heuristic predictions on historical rows...")
    hist_df = backfill_heuristic(hist_df)
    wp_std = hist_df["win_pct_home"].std()
    print(f"  MLB heuristic backfill: {len(hist_df)} rows | "
          f"win_pct std={wp_std:.3f}, range=[{hist_df['win_pct_home'].min():.3f}, "
          f"{hist_df['win_pct_home'].max():.3f}]")

    # Combine
    combined = pd.concat([hist_df, current_df], ignore_index=True)

    # Season weights for sample_weight in model fitting
    if "season_weight" in combined.columns:
        weights = combined["season_weight"].fillna(1.0).astype(float)
    else:
        weights = pd.Series(1.0, index=combined.index)

    # Print debug info
    n_hist = len(hist_df)
    n_curr = len(current_df)
    print(f"  Training corpus: {n_hist} historical + {n_curr} current = {n_hist + n_curr} total")
    if 'season' in hist_df.columns:
        seasons = sorted(hist_df['season'].dropna().astype(int).unique().tolist())
        print(f"  Historical seasons: {seasons}")

    return combined, weights.values


def train_mlb():
    """
    MLB model training with Railway timeout protection.
    Fixes: data cap at 8000, lighter ensemble (60 estimators, skip RF clf),
           try/except wrapper so it never returns 500.
    """
    import traceback as _tb
    try:
        # ── Step 1: Fetch current season data ────────────────────
        rows = sb_get("mlb_predictions",
                      "result_entered=eq.true&actual_home_runs=not.is.null&game_type=eq.R&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        # ── Step 2: Merge with historical corpus ─────────────────
        df, sample_weights = _mlb_merge_historical(current_df)

        if len(df) < 10:
            return {
                "error": "Not enough MLB regular season data to train (need 10+). "
                         "Spring training games (game_type=S) are excluded.",
                "n_current": len(current_df),
                "n_historical": len(df) - len(current_df) if df is not None else 0,
            }

        X = mlb_build_features(df)
        y_margin = df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)
        y_win = (y_margin > 0).astype(int)
        fit_weights = sample_weights if sample_weights is not None else np.ones(len(df))

        # ── FIX 1: Cap training data to prevent Railway timeout ──
        # With 14k+ rows, 6x cross_val_predict exceeds Railway CPU budget.
        MAX_TRAIN = 99999  # RF-only is fast enough for full data even on Railway
        n = len(df)
        if n > MAX_TRAIN:
            if "season_weight" in df.columns:
                keep_idx = df["season_weight"].fillna(0.5).nlargest(MAX_TRAIN).index
            else:
                keep_idx = df.index[-MAX_TRAIN:]
            X = X.loc[keep_idx].reset_index(drop=True)
            y_margin = y_margin.loc[keep_idx].reset_index(drop=True)
            y_win = y_win.loc[keep_idx].reset_index(drop=True)
            fit_weights = fit_weights[keep_idx.values] if hasattr(keep_idx, 'values') else fit_weights[-MAX_TRAIN:]
            df = df.loc[keep_idx].reset_index(drop=True)
            n = len(X)
            print(f"  Capped training data: {len(df)} -> {n} rows (Railway timeout protection)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cv_folds = 5  # 5 for Railway; set 10 for local runs

        if n >= 200:
            # ── CatBoost solo at 50 estimators (sweep-optimized) ──
            if HAS_CAT:
                cat_reg = CatBoostRegressor(
                    iterations=50, depth=4, learning_rate=0.06,
                    subsample=0.8, min_data_in_leaf=20,
                    random_seed=42, verbose=0,
                )
                print(f"  MLB: Training CatBoost solo on {n} rows (ts-cv, 50 est)...")
                reg_models = {"cat": cat_reg}
                oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=fit_weights)
                oof_cat = oof["cat"]

                cat_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)

                # Bias correction from OOF residuals
                bias_correction = float(np.mean(oof_cat - y_margin.values if hasattr(y_margin, 'values') else oof_cat - y_margin))
                print(f"  MLB bias correction: {bias_correction:+.3f} runs")

                reg = cat_reg
                reg_cv_mae = float(np.mean(np.abs(oof_cat - y_margin.values if hasattr(y_margin, 'values') else oof_cat - y_margin)))
                print(f"  MLB OOF MAE: {reg_cv_mae:.3f}")

                explainer = shap.TreeExplainer(cat_reg)
                model_type = "CatBoost_v5"
            else:
                # Fallback: RF if CatBoost not available
                rf_reg = RandomForestRegressor(
                    n_estimators=85, max_depth=6,
                    min_samples_leaf=15, max_features=0.7,
                    random_state=42, n_jobs=1,
                )
                print(f"  MLB: Training RF fallback on {n} rows (ts-cv, 85 est)...")
                reg_models = {"rf": rf_reg}
                oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=fit_weights)
                oof_rf = oof["rf"]

                rf_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)

                bias_correction = float(np.mean(oof_rf - y_margin.values if hasattr(y_margin, 'values') else oof_rf - y_margin))
                print(f"  MLB bias correction: {bias_correction:+.3f} runs")

                reg = rf_reg
                reg_cv_mae = float(np.mean(np.abs(oof_rf - y_margin.values if hasattr(y_margin, 'values') else oof_rf - y_margin)))
                print(f"  MLB OOF MAE: {reg_cv_mae:.3f}")

                explainer = shap.TreeExplainer(rf_reg)
                model_type = "RF_v4_fallback"

            # ── FIX 3: Lighter classifier (GBM + LR, skip RF) ───
            # RF classifier adds ~30% training time with minimal gain
            gbm_clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.06, subsample=0.8,
                min_samples_leaf=20, random_state=42,
            )
            lr_clf = LogisticRegression(max_iter=1000, C=1.0)

            gbm_clf.fit(X_scaled, y_win, sample_weight=fit_weights)
            lr_clf.fit(X_scaled, y_win, sample_weight=fit_weights)

            oof_gbm_p = cross_val_predict(gbm_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
            oof_lr_p  = cross_val_predict(lr_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]

            meta_lr = LogisticRegression(max_iter=1000, C=1.0)
            meta_clf_X = np.column_stack([oof_gbm_p, oof_lr_p])
            meta_lr.fit(meta_clf_X, y_win)

            clf = StackedClassifier([gbm_clf, lr_clf], meta_lr)

            # Isotonic calibration on OOF stacked probs
            oof_stacked_probs = meta_lr.predict_proba(meta_clf_X)[:, 1]
            isotonic = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            isotonic.fit(oof_stacked_probs, y_win.values if hasattr(y_win, 'values') else y_win)
            print(f"  MLB isotonic calibration fitted on {len(oof_stacked_probs)} OOF samples")
            print(f"  MLB model: {model_type}, bias correction: {bias_correction:+.3f}")

        else:
            reg = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=cv_folds)
            reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
            reg_cv = cross_val_score(reg, X_scaled, y_margin,
                                      cv=cv_folds, scoring="neg_mean_absolute_error")
            reg_cv_mae = float(-reg_cv.mean())
            explainer = shap.LinearExplainer(reg, X_scaled, feature_perturbation="interventional")
            model_type = "Ridge"
            clf = CalibratedClassifierCV(
                LogisticRegression(max_iter=1000), cv=cv_folds
            )
            clf.fit(X_scaled, y_win, sample_weight=fit_weights)
            isotonic = None
            bias_correction = 0.0

        bundle = {
            "scaler": scaler,
            "reg": reg,
            "clf": clf,
            "explainer": explainer,
            "feature_cols": list(X.columns),
            "n_train": n,
            "n_historical": len(df) - len(current_df),
            "n_current": len(current_df),
            "mae_cv": reg_cv_mae,
            "trained_at": datetime.utcnow().isoformat(),
            "model_type": model_type,
            "alpha": float(reg.alpha_) if hasattr(reg, "alpha_") else None,
            "isotonic": isotonic,
            "bias_correction": bias_correction,
        }
        save_model("mlb", bundle)

        disp = calibrate_mlb_dispersion()

        return {
            "status": "trained",
            "model_type": model_type,
            "n_train": n,
            "n_historical": len(df) - len(current_df),
            "n_current": len(current_df),
            "mae_cv": round(reg_cv_mae, 3),
            "alpha": float(reg.alpha_) if hasattr(reg, "alpha_") else None,
            "features": list(X.columns),
            "dispersion": disp,
            "bias_correction": round(bias_correction, 4) if n >= 200 else None,
        }

    except Exception as e:
        # ── FIX 4: Never return 500 — always return diagnostic JSON ──
        return {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": _tb.format_exc(),
            "hint": "Check Railway logs for memory/timeout issues. "
                    "If OOM, reduce MAX_TRAIN. If timeout, reduce n_estimators.",
        }


def predict_mlb(game: dict):
    bundle = load_model("mlb")
    if not bundle:
        return {"error": "MLB model not trained — call /train/mlb first"}

    # None-safe float: game.get() returns None when key exists but value is NULL in Supabase
    def _f(v, d=0.0):
        return float(v) if v is not None else d

    # Get heuristic predictions (may be 0 for historical games)
    ph = _f(game.get("pred_home_runs"), 0)
    pa = _f(game.get("pred_away_runs"), 0)
    
    # AUDIT FIX: capture team abbrs and date for travel/series computation
    home_team = (game.get("home_team") or "").upper().strip()
    away_team = (game.get("away_team") or "").upper().strip()
    game_date = game.get("game_date") or datetime.utcnow().strftime("%Y-%m-%d")

    # AUDIT v6: Season-aware defaults (replaces hardcoded 0.315/4.25/4.10)
    try:
        _sy = int(game_date[:4])
    except (ValueError, TypeError):
        _sy = datetime.utcnow().year
    _sc = SEASON_CONSTANTS.get(_sy, DEFAULT_CONSTANTS)
    _lg_woba = _sc["lg_woba"]
    _lg_fip = _sc["lg_fip"]

    # Get raw inputs if provided
    home_woba = _f(game.get("home_woba"), _lg_woba)
    away_woba = _f(game.get("away_woba"), _lg_woba)
    # M-04 FIX: explicit None check (0.0 FIP is falsy but valid in Python `or`)
    home_sp_fip = _f(game.get("home_sp_fip"), None) if game.get("home_sp_fip") is not None else _f(game.get("home_fip"), _lg_fip)
    away_sp_fip = _f(game.get("away_sp_fip"), None) if game.get("away_sp_fip") is not None else _f(game.get("away_fip"), _lg_fip)
    if home_sp_fip is None: home_sp_fip = _lg_fip
    if away_sp_fip is None: away_sp_fip = _lg_fip
    home_fip = _f(game.get("home_fip"), _lg_fip)
    away_fip = _f(game.get("away_fip"), _lg_fip)
    home_bullpen = _f(game.get("home_bullpen_era"), _lg_fip)
    away_bullpen = _f(game.get("away_bullpen_era"), _lg_fip)
    park_factor = _f(game.get("park_factor"), 1.00)
    temp_f = _f(game.get("temp_f"), 70.0)
    wind_mph = _f(game.get("wind_mph"), 5.0)
    wind_out_flag = _f(game.get("wind_out_flag"), 0.0)

    # DOME FIX: Neutralize weather for dome/retractable-roof parks
    # Outdoor weather data is irrelevant inside a dome — prevents false signal
    if _is_dome(home_team):
        temp_f = 70.0
        wind_mph = 0.0
        wind_out_flag = 0.0

    home_rest = _f(game.get("home_rest_days"), 4.0)
    away_rest = _f(game.get("away_rest_days"), 4.0)
    home_travel = _f(game.get("home_travel"), 0.0)
    away_travel = _f(game.get("away_travel"), 0.0)

    # K/9 and BB/9
    home_k9 = _f(game.get("home_k9"), 8.5)
    away_k9 = _f(game.get("away_k9"), 8.5)
    home_bb9 = _f(game.get("home_bb9"), 3.2)
    away_bb9 = _f(game.get("away_bb9"), 3.2)

    # SP innings pitched + defensive OAA
    home_sp_ip = _f(game.get("home_sp_ip"), 5.5)
    away_sp_ip = _f(game.get("away_sp_ip"), 5.5)
    home_def_oaa = _f(game.get("home_def_oaa"), 0.0)
    away_def_oaa = _f(game.get("away_def_oaa"), 0.0)

    # Enhancement: Platoon splits and lineup confirmation
    home_platoon_delta = _f(game.get("home_platoon_delta"), 0.0)
    away_platoon_delta = _f(game.get("away_platoon_delta"), 0.0)
    home_lineup_confirmed = int(game.get("home_lineup_confirmed") or 0)
    away_lineup_confirmed = int(game.get("away_lineup_confirmed") or 0)

    # SP FIP known flag
    has_sp_fip = 1 if (home_sp_fip != 4.25 and away_sp_fip != 4.25) else 0

    # Use starter FIP when known, fall back to team FIP
    home_starter_fip = home_sp_fip if home_sp_fip != 4.25 else home_fip
    away_starter_fip = away_sp_fip if away_sp_fip != 4.25 else away_fip

    # Calculate derived features
    # AUDIT FIX F1: Changed baseline from 9.0 to 5.5 (matches training)
    home_bp_exposure = max(0, 5.5 - home_sp_ip) * (home_bullpen / 4.10)
    away_bp_exposure = max(0, 5.5 - away_sp_ip) * (away_bullpen / 4.10)

    # Diffs
    woba_diff = home_woba - away_woba
    fip_diff = home_starter_fip - away_starter_fip
    bullpen_era_diff = home_bullpen - away_bullpen
    k_bb_diff = (home_k9 - home_bb9) - (away_k9 - away_bb9)
    wind_out = int(wind_out_flag)

    row_data = {
        # ── 29-feature production set (v8 ensemble) ──
        "woba_diff": woba_diff,
        "fip_diff": fip_diff,
        "bullpen_era_diff": bullpen_era_diff,
        "k_bb_diff": k_bb_diff,
        "sp_ip_diff": home_sp_ip - away_sp_ip,
        "bp_exposure_diff": home_bp_exposure - away_bp_exposure,
        "def_oaa_diff": home_def_oaa - away_def_oaa,  # overwritten by rolling stats if available
        "park_factor": park_factor,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_out": wind_out,
        "is_warm": 1 if temp_f > 75 else 0,
        "is_cold": 1 if temp_f < 45 else 0,
        "rest_diff": home_rest - away_rest,
        # Interactions
        "woba_x_park": woba_diff * park_factor,
        # Enhancement: Platoon, starter spread
        "platoon_diff": home_platoon_delta - away_platoon_delta,
        "sp_fip_spread": abs(home_starter_fip - away_starter_fip),
        # Market
        "market_spread": _f(game.get("market_spread_home") or game.get("market_spread"), 0),
    }

    # ── Advanced features (v7) — compute what we can at serve time ──
    # sp_relative_fip: starter quality vs own team (always computable)
    row_data["sp_relative_fip_diff"] = (home_starter_fip - home_fip) - (away_starter_fip - away_fip)
    # temp × park interaction (always computable)
    row_data["temp_x_park"] = ((temp_f - 70) / 30.0) * park_factor
    # Umpire run environment (from frontend if available, else league avg)
    row_data["ump_run_env"] = _f(game.get("ump_run_env"), 8.5)

    # ── v9 new features (proven +8% ML improvement in walk-forward) ──
    # SP form deltas — individual pitcher form (away is 100x more predictive than home)
    row_data["home_sp_form_delta"] = _f(game.get("home_sp_form_delta"), 0)
    row_data["away_sp_form_delta"] = _f(game.get("away_sp_form_delta"), 0)
    # Bullpen fatigue — recent bullpen workload
    row_data["home_bp_fatigue"] = _f(game.get("home_bp_fatigue"), 0)
    row_data["away_bp_fatigue"] = _f(game.get("away_bp_fatigue"), 0)
    # Market moneyline — adds signal beyond run line
    row_data["market_home_ml"] = _f(game.get("market_home_ml") or game.get("home_moneyline") or game.get("market_home_ml"), 0)
    # AUDIT FIX F-08: Compute series game number from schedule
    _sgn = _f(game.get("series_game_num"), 0)
    row_data["series_game_num"] = _sgn if _sgn > 0 else float(
        _compute_series_game_num(home_team, away_team, game_date) if home_team and away_team else 1)
    # Rolling features — read from mlb_team_rolling + mlb_ump_profiles
    _rolling_loaded = False
    try:
        from mlb_rolling_stats import get_rolling_features
        home_abbr = game.get("home_team", "")
        away_abbr = game.get("away_team", "")
        ump = game.get("ump_name", None)
        rolling = get_rolling_features(home_abbr, away_abbr, ump)
        for k, v in rolling.items():
            row_data[k] = v
        _rolling_loaded = True
    except ImportError:
        print("  [predict_mlb] mlb_rolling_stats not deployed — using defaults")
    except Exception as e:
        print(f"  [predict_mlb] rolling features error: {e}")

    if not _rolling_loaded:
        # Fallback: use frontend-provided values or neutral defaults
        row_data.setdefault("pyth_residual_diff", _f(game.get("pyth_residual_diff"), 0))
        row_data.setdefault("scoring_entropy_diff", _f(game.get("scoring_entropy_diff"), 0))
        row_data.setdefault("first_inn_rate_diff", _f(game.get("first_inn_rate_diff"), 0))
        row_data.setdefault("clutch_divergence_diff", _f(game.get("clutch_divergence_diff"), 0))
        row_data.setdefault("opp_adj_form_diff", _f(game.get("opp_adj_form_diff"), 0))
        row_data.setdefault("scoring_entropy_combined", _f(game.get("scoring_entropy_combined"), 5.0))
        row_data.setdefault("first_inn_rate_combined", _f(game.get("first_inn_rate_combined"), 0.8))
        row_data.setdefault("ump_run_env", _f(game.get("ump_run_env"), 8.5))

    # Gracefully handle any feature the model expects but row_data is missing
    for col in bundle.get("feature_cols", []):
        if col not in row_data:
            row_data[col] = 0.0

    # AUDIT FIX F2: Explicit feature set assertion — catch misalignment immediately
    expected = set(bundle["feature_cols"])
    provided = set(row_data.keys())
    missing = expected - provided
    if missing:
        print(f"[predict_mlb] WARNING: Missing features at serve time: {missing}")

    # Create DataFrame with only the features the model expects
    row = pd.DataFrame([{k: row_data[k] for k in bundle["feature_cols"]}])
    
    # Scale and predict
    X_s = bundle["scaler"].transform(row[bundle["feature_cols"]])
    
    # Ensemble: average predictions from all models if available
    ensemble_models = bundle.get("_ensemble_models")
    _model_preds = []
    if ensemble_models and len(ensemble_models) > 1:
        _model_preds = [float(m.predict(X_s)[0]) for m in ensemble_models]
        raw_margin = float(_np.mean(_model_preds))
    else:
        reg = bundle.get("reg", bundle.get("model"))
        raw_margin = float(reg.predict(X_s)[0])
        _model_preds = [raw_margin]

    # 3-model agreement: all predict same direction (all positive or all negative)
    _models_agree = all(p > 0 for p in _model_preds) or all(p < 0 for p in _model_preds) if len(_model_preds) > 1 else True

    # v9: Deep model validator — d8 CatBoost cross-checks d4 prediction
    # When agree: 73.6% ML. When disagree: 20.9% ML. Powerful filter.
    _validator = bundle.get("_validator")
    if _validator is not None:
        try:
            val_margin = float(_validator.predict(X_s)[0])
            _val_agree = (raw_margin > 0) == (val_margin > 0)
            _models_agree = _models_agree and _val_agree
            if not _val_agree:
                print(f"  [predict_mlb] ⚠️ Validator disagrees: d4={raw_margin:+.2f} vs d8={val_margin:+.2f}")
        except Exception as e:
            print(f"  [predict_mlb] Validator error: {e}")

    # AUDIT FIX F7: Use Gaussian CDF instead of Elo-style formula.
    # Elo (10^(-m/σ)) compresses probabilities toward 50% compared to Gaussian,
    # making confidence gates (65%/60%) harder to trigger. At margin=2.0:
    # Elo gives 63.6% vs Gaussian 69.1% — nearly a full run difference.
    from scipy.stats import norm as _norm
    MLB_SIGMA = 4.0
    raw_win_prob = float(_norm.cdf(raw_margin / MLB_SIGMA))
    raw_win_prob = max(0.20, min(0.80, raw_win_prob))

    # FIX S2b: Apply bias correction to margin prediction
    bias = bundle.get("bias_correction", 0.0)
    margin = raw_margin - bias

    # Margin-based probability via Gaussian CDF
    win_prob = float(_norm.cdf(margin / MLB_SIGMA))
    win_prob = max(0.20, min(0.80, win_prob))

    # SHAP explanation (optional — not all model bundles include explainer)
    shap_out = []
    try:
        _explainer = bundle.get("explainer")
        if _explainer is not None:
            shap_vals = _explainer.shap_values(X_s)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            shap_values_row = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
            for f, v in zip(bundle["feature_cols"], shap_values_row):
                shap_out.append({
                    "feature": f,
                    "shap": round(float(v), 4),
                    "value": round(float(row[f].iloc[0]), 3)
                })
            shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)
    except Exception as e:
        print(f"  [predict_mlb] SHAP error (non-fatal): {e}")

    return {
        "sport": "MLB",
        "ml_margin": round(margin, 2),
        "ml_margin_raw": round(raw_margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "ml_win_prob_raw": round(raw_win_prob, 4),
        "bias_correction": round(bias, 3),
        "models_agree": _models_agree,
        "model_preds": [round(p, 3) for p in _model_preds],
        "feature_coverage": f"{sum(1 for c in bundle['feature_cols'] if row_data.get(c,0)!=0)}/{len(bundle['feature_cols'])}",
        "rolling_stats_loaded": _rolling_loaded,
        "shap": shap_out,  # All features for verification panel
        "model_meta": {
            "n_train": bundle.get("n_train", 0),
            "n_historical": bundle.get("n_historical", 0),
            "n_current": bundle.get("n_current", 0),
            "mae_cv": bundle.get("mae_cv", 0),
            "trained_at": bundle.get("trained_at", ""),
            "model_type": bundle.get("model_type", "unknown"),
            "has_isotonic": bundle.get("isotonic") is not None,
        },
        "_features": row_data,  # expose for v9 ATS model
    }


# ═══════════════════════════════════════════════════════════════
# MLB O/U MODEL (separate total-runs predictor)
# ═══════════════════════════════════════════════════════════════

def predict_mlb_ou(game: dict):
    """Predict total runs using the dedicated O/U model (COMBINED features, not diffs)."""
    bundle = load_model("mlb_ou")
    if not bundle:
        return {"error": "MLB O/U model not trained — run mlb_ou_retrain.py --upload"}

    # ── V3: Triple agreement + lineup + ump features ──
    if bundle.get("_v3_lineup"):
        try:
            from mlb_ou_v3_serve import predict_mlb_ou_v3
            return predict_mlb_ou_v3(game, bundle)
        except Exception as e:
            print(f"  [mlb_ou] v3 error: {e}")

    # ── V2: SP Form residual model (detects _v2_sp_form flag in bundle) ──
    if bundle.get("_v2_sp_form"):
        try:
            from mlb_ou_v2_serve import predict_mlb_ou_v2
            return predict_mlb_ou_v2(game, bundle)
        except Exception as e:
            print(f"  [ou_v2] Error: {e} — falling back to v1")

    def _f(v, d=0.0):
        return float(v) if v is not None else d

    # AUDIT v6: Season-aware defaults
    game_date = game.get("game_date") or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        season_year = int(game_date[:4])
    except (ValueError, TypeError):
        season_year = datetime.utcnow().year
    _sc = SEASON_CONSTANTS.get(season_year, DEFAULT_CONSTANTS)
    _lg_woba = _sc["lg_woba"]
    _lg_fip = _sc["lg_fip"]

    home_woba = _f(game.get("home_woba"), _lg_woba)
    away_woba = _f(game.get("away_woba"), _lg_woba)
    # M-04 FIX: explicit None check
    home_sp_fip = _f(game.get("home_sp_fip"), None) if game.get("home_sp_fip") is not None else _f(game.get("home_fip"), _lg_fip)
    away_sp_fip = _f(game.get("away_sp_fip"), None) if game.get("away_sp_fip") is not None else _f(game.get("away_fip"), _lg_fip)
    if home_sp_fip is None: home_sp_fip = _lg_fip
    if away_sp_fip is None: away_sp_fip = _lg_fip
    home_fip = _f(game.get("home_fip"), _lg_fip)
    away_fip = _f(game.get("away_fip"), _lg_fip)
    home_bullpen = _f(game.get("home_bullpen_era"), _lg_fip)
    away_bullpen = _f(game.get("away_bullpen_era"), _lg_fip)
    park_factor = _f(game.get("park_factor"), 1.00)
    temp_f = _f(game.get("temp_f"), 70.0)
    wind_mph = _f(game.get("wind_mph"), 5.0)
    wind_out_flag = _f(game.get("wind_out_flag"), 0.0)

    # DOME FIX: Neutralize weather for dome/retractable-roof parks
    _home_team_ou = (game.get("home_team") or "").upper().strip()
    if _is_dome(_home_team_ou):
        temp_f = 70.0
        wind_mph = 0.0
        wind_out_flag = 0.0

    home_k9 = _f(game.get("home_k9"), 8.5)
    away_k9 = _f(game.get("away_k9"), 8.5)
    home_bb9 = _f(game.get("home_bb9"), 3.2)
    away_bb9 = _f(game.get("away_bb9"), 3.2)
    home_sp_ip = _f(game.get("home_sp_ip"), 5.5)
    away_sp_ip = _f(game.get("away_sp_ip"), 5.5)
    home_rest = _f(game.get("home_rest_days"), 4.0)
    away_rest = _f(game.get("away_rest_days"), 4.0)
    pred_home = _f(game.get("pred_home_runs"), 0)
    pred_away = _f(game.get("pred_away_runs"), 0)

    home_starter_fip = home_sp_fip if home_sp_fip != _lg_fip else home_fip
    away_starter_fip = away_sp_fip if away_sp_fip != _lg_fip else away_fip
    wind_out = int(wind_out_flag)

    # game_date and season_year already parsed above
    lg_rpg = _sc["lg_rpg"]

    raw_market = game.get("market_ou_total") or game.get("market_total")
    market_total = _f(raw_market, 0) or lg_rpg * 2

    row_data = {
        "woba_combined": home_woba + away_woba,
        "woba_diff": home_woba - away_woba,
        "fip_combined": home_starter_fip + away_starter_fip,
        "fip_diff": home_starter_fip - away_starter_fip,
        "sp_fip_spread": abs(home_starter_fip - away_starter_fip),
        "bullpen_combined": home_bullpen + away_bullpen,
        "k_bb_combined": (home_k9 - home_bb9) + (away_k9 - away_bb9),
        "sp_ip_combined": home_sp_ip + away_sp_ip,
        "bp_exposure_combined": (
            max(0, 5.5 - home_sp_ip) * (home_bullpen / 4.10) +
            max(0, 5.5 - away_sp_ip) * (away_bullpen / 4.10)
        ),
        "park_factor": park_factor,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_out": wind_out,
        "is_warm": 1 if temp_f > 75 else 0,
        "is_cold": 1 if temp_f < 45 else 0,
        "temp_x_park": ((temp_f - 70) / 30.0) * park_factor,
        "lg_rpg": lg_rpg,
        "market_total": market_total,
        "has_market": 1 if raw_market and _f(raw_market, 0) > 0 else 0,
        "total_pred": pred_home + pred_away,
        "has_heuristic": 1 if (pred_home + pred_away) > 0 else 0,
        "rest_combined": home_rest + away_rest,
    }

    # Rolling features
    try:
        from mlb_rolling_stats import get_rolling_features
        home_team = (game.get("home_team") or "").upper().strip()
        away_team = (game.get("away_team") or "").upper().strip()
        ump = game.get("ump_name", None)
        rolling = get_rolling_features(home_team, away_team, ump)
        row_data["scoring_entropy_combined"] = rolling.get("scoring_entropy_combined", 5.0)
        row_data["first_inn_rate_combined"] = rolling.get("first_inn_rate_combined", 0.8)
        row_data["ump_run_env"] = rolling.get("ump_run_env", 8.5)
    except Exception:
        row_data.setdefault("scoring_entropy_combined", _f(game.get("scoring_entropy_combined"), 5.0))
        row_data.setdefault("first_inn_rate_combined", _f(game.get("first_inn_rate_combined"), 0.8))
        row_data.setdefault("ump_run_env", _f(game.get("ump_run_env"), 8.5))

    home_team = (game.get("home_team") or "").upper().strip()
    away_team = (game.get("away_team") or "").upper().strip()
    sgn = _f(game.get("series_game_num"), 0)
    row_data["series_game_num"] = sgn if sgn > 0 else float(
        _compute_series_game_num(home_team, away_team, game_date) if home_team and away_team else 1)

    for col in bundle.get("feature_cols", []):
        if col not in row_data:
            row_data[col] = 0.0

    row = pd.DataFrame([{k: row_data[k] for k in bundle["feature_cols"]}])
    X_s = bundle["scaler"].transform(row[bundle["feature_cols"]])

    ensemble_models = bundle.get("_ensemble_models", [])
    if ensemble_models and len(ensemble_models) > 1:
        raw_total = float(_np.mean([m.predict(X_s)[0] for m in ensemble_models]))
    else:
        raw_total = float(bundle.get("reg", bundle.get("model")).predict(X_s)[0])

    bias = bundle.get("bias_correction", 0.0)
    pred_total = max(4.0, min(18.0, raw_total - bias))

    market_line = _f(game.get("market_ou_total") or game.get("market_total"), 0)
    ou_edge = pred_total - market_line if market_line > 0 else 0.0

    # Asymmetric thresholds: UNDER 1.0/1.5/2.0 → 1/2/3u; OVER 2.0+ → 1u
    ou_pick, ou_units = None, 0
    if market_line > 0:
        if ou_edge < -2.0:
            ou_pick, ou_units = "UNDER", 3
        elif ou_edge < -1.5:
            ou_pick, ou_units = "UNDER", 2
        elif ou_edge < -1.0:
            ou_pick, ou_units = "UNDER", 1
        elif ou_edge > 2.0:
            ou_pick, ou_units = "OVER", 1

    shap_out = []
    try:
        shap_vals = bundle["explainer"].shap_values(X_s)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        sv = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
        for f, v in zip(bundle["feature_cols"], sv):
            shap_out.append({"feature": f, "shap": round(float(v), 4), "value": round(float(row[f].iloc[0]), 3)})
        shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)
    except Exception:
        pass

    return {
        "sport": "MLB", "type": "ou",
        "pred_total": round(pred_total, 2),
        "market_total": round(market_line, 1) if market_line > 0 else None,
        "ou_edge": round(ou_edge, 2),
        "ou_pick": ou_pick, "ou_units": ou_units,
        "bias_correction": round(bias, 3),
        "shap": shap_out[:10],
        "model_meta": {
            "n_train": bundle.get("n_train", 0),
            "mae_cv": bundle.get("mae_cv", 0),
            "trained_at": bundle.get("trained_at", ""),
            "model_type": bundle.get("model_type", "unknown"),
        },
    }


# ═══════════════════════════════════════════════════════════════
# NBA MODEL
# ═══════════════════════════════════════════════════════════════
def heuristic_predict_row(row):
    """
    Replay the mlb.js v16 heuristic in Python using AVAILABLE historical columns.
    Produces differentiated pred_home_runs / pred_away_runs per game.

    v16 AUDIT FIXES (round-robin forensic audit alignment):
      H-1:  HFA applied in probability space (already done in v3, retained).
      H-3:  Uses starter FIP (home_sp_fip) when available, marginal to team FIP
            (matches JS engine). Falls back to team FIP vs league FIP when missing.
      M-4:  Added bullpen quality (already in v3), SP workload/bullpen exposure,
            dome-aware weather adjustments, K/9 and BB/9 adjustments (re-added
            to heuristic to match JS engine — ML features still capture them
            independently via k_bb_diff).
      H-5:  Weather adjustments skipped for dome parks.
      F7:   Clamp range aligned to frontend: [1.8, 9.5].

    Gracefully degrades: null columns → defaults (matches prior behavior).
    Historical rows without new columns produce same predictions as v3.
    """
    season = int(row.get("season", 2024))
    sc = SEASON_CONSTANTS.get(season, DEFAULT_CONSTANTS)

    # ── Read available columns ──
    home_woba   = row.get("home_woba")
    away_woba   = row.get("away_woba")
    home_fip    = row.get("home_fip")        # team aggregate FIP (end-of-season)
    away_fip    = row.get("away_fip")
    home_k9     = row.get("home_k9")
    away_k9     = row.get("away_k9")
    home_bb9    = row.get("home_bb9")
    away_bb9    = row.get("away_bb9")
    home_bp_era = row.get("home_bullpen_era")  # may be null in older historical
    away_bp_era = row.get("away_bullpen_era")
    pf          = row.get("park_factor")
    rest_h      = row.get("home_rest_days")
    rest_a      = row.get("away_rest_days")
    travel_h    = row.get("home_travel")
    travel_a    = row.get("away_travel")

    # H-3: Starter-level FIP (available for current season, null for historical)
    home_sp_fip = row.get("home_sp_fip")
    away_sp_fip = row.get("away_sp_fip")

    # M-4: SP average innings per start
    home_sp_ip  = row.get("home_sp_ip")
    away_sp_ip  = row.get("away_sp_ip")

    # M-4: Weather (for non-dome parks)
    temp_f        = row.get("temp_f")
    wind_mph      = row.get("wind_mph")
    wind_out_flag = row.get("wind_out_flag")

    # Park ID for dome detection and per-park HFA
    park_id = row.get("home_team_id") or row.get("homeTeamId") or row.get("park_id")

    # ── Coerce to float with season-specific defaults ──
    lg_woba    = sc["lg_woba"]
    woba_scale = sc["woba_scale"]
    lg_rpg     = sc["lg_rpg"]
    lg_fip     = sc["lg_fip"]
    pa_pg      = sc["pa_pg"]

    home_woba = float(home_woba) if home_woba is not None and not _isnan(home_woba) else lg_woba
    away_woba = float(away_woba) if away_woba is not None and not _isnan(away_woba) else lg_woba
    home_fip  = float(home_fip)  if home_fip  is not None and not _isnan(home_fip)  else lg_fip
    away_fip  = float(away_fip)  if away_fip  is not None and not _isnan(away_fip)  else lg_fip
    home_k9   = float(home_k9)   if home_k9   is not None and not _isnan(home_k9)   else 8.5
    away_k9   = float(away_k9)   if away_k9   is not None and not _isnan(away_k9)   else 8.5
    home_bb9  = float(home_bb9)  if home_bb9  is not None and not _isnan(home_bb9)  else 3.2
    away_bb9  = float(away_bb9)  if away_bb9  is not None and not _isnan(away_bb9)  else 3.2
    pf        = float(pf)        if pf        is not None and not _isnan(pf)        else 1.0
    rest_h    = float(rest_h)    if rest_h    is not None and not _isnan(rest_h)     else 3.0
    rest_a    = float(rest_a)    if rest_a    is not None and not _isnan(rest_a)     else 3.0
    travel_h  = float(travel_h)  if travel_h  is not None and not _isnan(travel_h)  else 0.0
    travel_a  = float(travel_a)  if travel_a  is not None and not _isnan(travel_a)  else 0.0

    # Starter FIP: use if valid and not the sentinel 4.25
    home_sp = None
    if home_sp_fip is not None and not _isnan(home_sp_fip):
        v = float(home_sp_fip)
        if v != 4.25:  # 4.25 = missing sentinel
            home_sp = v
    away_sp = None
    if away_sp_fip is not None and not _isnan(away_sp_fip):
        v = float(away_sp_fip)
        if v != 4.25:
            away_sp = v

    # Bullpen ERA
    has_bp_home = home_bp_era is not None and not _isnan(home_bp_era)
    has_bp_away = away_bp_era is not None and not _isnan(away_bp_era)
    if has_bp_home:
        home_bp_era = float(home_bp_era)
    if has_bp_away:
        away_bp_era = float(away_bp_era)

    # SP innings per start
    h_sp_ip = float(home_sp_ip) if home_sp_ip is not None and not _isnan(home_sp_ip) else 5.5
    a_sp_ip = float(away_sp_ip) if away_sp_ip is not None and not _isnan(away_sp_ip) else 5.5

    # Weather
    temp = float(temp_f) if temp_f is not None and not _isnan(temp_f) else None
    w_mph = float(wind_mph) if wind_mph is not None and not _isnan(wind_mph) else None
    w_out = int(float(wind_out_flag)) if wind_out_flag is not None and not _isnan(wind_out_flag) else 0

    # Dome detection
    is_dome = False
    if park_id is not None:
        try:
            is_dome = int(park_id) in _DOME_PARKS
        except (ValueError, TypeError):
            pass

    # ═══════════════════════════════════════════════════════════════
    # LEAK DISCOUNT — game-date-aware regression to mean
    # ═══════════════════════════════════════════════════════════════
    game_date = row.get("game_date")
    if game_date and isinstance(game_date, str) and len(game_date) >= 7:
        try:
            month = int(game_date[5:7])
            day = int(game_date[8:10]) if len(game_date) >= 10 else 15
            GAMES_BY_MONTH_END = {3: 5, 4: 30, 5: 56, 6: 81, 7: 105, 8: 133, 9: 162, 10: 162}
            games_before = GAMES_BY_MONTH_END.get(month - 1, 0)
            games_in_month = GAMES_BY_MONTH_END.get(month, 162) - games_before
            games_so_far = games_before + games_in_month * (day / 30.0)
            LEAK_DISCOUNT = max(0.15, min(0.85, games_so_far / 162.0))
        except (ValueError, TypeError):
            LEAK_DISCOUNT = 0.50
    else:
        LEAK_DISCOUNT = 0.50

    # ── Regress stats toward league average ──
    home_fip_adj  = lg_fip  + (home_fip  - lg_fip)  * LEAK_DISCOUNT
    away_fip_adj  = lg_fip  + (away_fip  - lg_fip)  * LEAK_DISCOUNT
    home_k9_adj   = 8.5 + (home_k9 - 8.5) * LEAK_DISCOUNT
    away_k9_adj   = 8.5 + (away_k9 - 8.5) * LEAK_DISCOUNT
    home_bb9_adj  = 3.2 + (home_bb9 - 3.2) * LEAK_DISCOUNT
    away_bb9_adj  = 3.2 + (away_bb9 - 3.2) * LEAK_DISCOUNT
    home_woba_adj = lg_woba + (home_woba - lg_woba) * LEAK_DISCOUNT
    away_woba_adj = lg_woba + (away_woba - lg_woba) * LEAK_DISCOUNT

    LG_BP_ERA = 4.10
    if has_bp_home:
        home_bp_adj = LG_BP_ERA + (home_bp_era - LG_BP_ERA) * LEAK_DISCOUNT
    if has_bp_away:
        away_bp_adj = LG_BP_ERA + (away_bp_era - LG_BP_ERA) * LEAK_DISCOUNT

    # ═══════════════════════════════════════════════════════════════
    # wOBA → Runs (FanGraphs method)
    # ═══════════════════════════════════════════════════════════════
    hr = lg_rpg + ((home_woba_adj - lg_woba) / woba_scale) * pa_pg
    ar = lg_rpg + ((away_woba_adj - lg_woba) / woba_scale) * pa_pg

    # ═══════════════════════════════════════════════════════════════
    # H-3 FIX: Starter FIP marginal to team FIP (matches JS engine)
    # ═══════════════════════════════════════════════════════════════
    if home_sp is not None:
        # Starter available: marginal to team FIP (H-3 aligned with JS)
        ar += (home_sp - home_fip_adj) * FIP_COEFF
    else:
        # No starter: team FIP vs league (original behavior)
        ar += (home_fip_adj - lg_fip) * FIP_COEFF

    if away_sp is not None:
        hr += (away_sp - away_fip_adj) * FIP_COEFF
    else:
        hr += (away_fip_adj - lg_fip) * FIP_COEFF

    # ── K/9 and BB/9 adjustments (using leak-adjusted values) ──
    lg_k9 = 8.5
    lg_bb9 = 3.2
    ar -= (home_k9_adj - lg_k9) * 0.04
    ar += (home_bb9_adj - lg_bb9) * 0.06
    hr -= (away_k9_adj - lg_k9) * 0.04
    hr += (away_bb9_adj - lg_bb9) * 0.06

    # ═══════════════════════════════════════════════════════════════
    # Bullpen quality (FIX F5, upgraded M-4)
    # ═══════════════════════════════════════════════════════════════
    BP_IMPACT = 0.40  # M-4: aligned with JS (was 0.30 in v3)
    if has_bp_home:
        bp_q_home = (LG_BP_ERA - home_bp_adj) / LG_BP_ERA
        ar -= bp_q_home * BP_IMPACT  # good home pen → fewer away runs
    if has_bp_away:
        bp_q_away = (LG_BP_ERA - away_bp_adj) / LG_BP_ERA
        hr -= bp_q_away * BP_IMPACT  # good away pen → fewer home runs

    # ═══════════════════════════════════════════════════════════════
    # M-4 FIX: SP workload / bullpen exposure
    # ═══════════════════════════════════════════════════════════════
    if has_bp_home:
        bp_deficit_h = max(0, -((LG_BP_ERA - home_bp_adj) / LG_BP_ERA))
        bp_exp_home = max(0, 5.0 - h_sp_ip) * (1 + bp_deficit_h) * 0.08
        ar += bp_exp_home
    if has_bp_away:
        bp_deficit_a = max(0, -((LG_BP_ERA - away_bp_adj) / LG_BP_ERA))
        bp_exp_away = max(0, 5.0 - a_sp_ip) * (1 + bp_deficit_a) * 0.08
        hr += bp_exp_away

    # ═══════════════════════════════════════════════════════════════
    # M-4/H-5 FIX: Park factor with dome-aware weather
    # ═══════════════════════════════════════════════════════════════
    pf = max(0.86, min(1.28, pf))
    # Weather adjustments only for non-dome parks
    if not is_dome and temp is not None:
        pf += ((temp - 70) / 10) * 0.0035
    if not is_dome and w_out and w_mph is not None and w_mph > 5:
        pf += (w_mph - 5) * 0.0035
    # Re-clamp after weather (matches JS [0.92, 1.18])
    pf = max(0.92, min(1.18, pf))

    hr *= pf
    ar *= pf

    # ── Rest/travel adjustments ──
    if rest_h == 0:
        hr -= 0.15
    if rest_a == 0:
        ar -= 0.15
    if travel_a > 1500:
        ar -= 0.08

    # ═══════════════════════════════════════════════════════════════
    # Clamp aligned to frontend [1.8, 9.5]
    # ═══════════════════════════════════════════════════════════════
    hr = max(1.8, min(9.5, hr))
    ar = max(1.8, min(9.5, ar))

    # ═══════════════════════════════════════════════════════════════
    # Pythagenpat + HFA in probability space (matching frontend)
    # ═══════════════════════════════════════════════════════════════
    league_rpg = 2 * lg_rpg
    exp = max(1.60, min(2.10, league_rpg ** 0.287))
    pyth_wp = (hr ** exp) / (hr ** exp + ar ** exp)

    # Per-park HFA lookup (falls back to 0.035 league average)
    if park_id is not None:
        try:
            park_hfa = PARK_HFA.get(int(park_id), 0.035)
        except (ValueError, TypeError):
            park_hfa = 0.035
    else:
        park_hfa = 0.035

    win_pct = min(0.85, max(0.15, pyth_wp + park_hfa))

    model_ml = int(-round((win_pct / (1 - win_pct)) * 100) if win_pct >= 0.5
                    else round(((1 - win_pct) / win_pct) * 100))

    return {
        "pred_home_runs": round(hr, 3),
        "pred_away_runs": round(ar, 3),
        "win_pct_home": round(win_pct, 4),
        "ou_total": round(hr + ar, 1),
        "model_ml_home": model_ml,
    }


def _isnan(v):
    """Check if value is NaN (works for float and numpy)."""
    try:
        return v != v  # NaN != NaN
    except (TypeError, ValueError):
        return False


def backfill_heuristic(df):
    """Apply heuristic_predict_row to every row, filling pred columns."""
    preds = df.apply(lambda r: pd.Series(heuristic_predict_row(r)), axis=1)
    df = df.copy()
    df["pred_home_runs"] = preds["pred_home_runs"].astype(float)
    df["pred_away_runs"] = preds["pred_away_runs"].astype(float)
    df["win_pct_home"]   = preds["win_pct_home"].astype(float)
    df["ou_total"]       = preds["ou_total"].astype(float)
    df["model_ml_home"]  = preds["model_ml_home"].astype(int)
    return df


# ══════════════════════════════════════════════════════════════════
# MLB BACKTESTING
