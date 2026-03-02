"""
Multi-Sport Predictor API
Scikit-learn + SHAP backend for Railway deployment
Connects to existing Supabase tables

v2 CHANGES (2025):
  - MLB Monte Carlo: Poisson → Negative Binomial (overdispersion k calibrated from data)
  - MLB Monte Carlo: Added correlated run environment (shared latent factor via log-normal)
  - MLB Monte Carlo: Returns run-line & total probabilities, not just win %
  - MLB training: Added dispersion calibration endpoint (/calibrate/mlb)
  - MLB training: mlb_build_features now accepts raw game inputs when available
  - All sports: Monte Carlo returns over/under probabilities at the posted total
  - requirements.txt: Add scipy>=1.13.0
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
import time as _time
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ────────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression, Ridge, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
import shap
import joblib
from scipy.stats import nbinom  # NEW: Negative Binomial distribution

app = Flask(__name__)
CORS(app)  # Allow requests from your React app

# ── Supabase config ───────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")

# Try multiple possible environment variable names
SUPABASE_KEY = (
    os.environ.get("SUPABASE_ANON_KEY") or 
    os.environ.get("SUPABASE_KEY") or 
    os.environ.get("SUPABASE_SERVICE_KEY") or 
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or 
    ""
)

print(f"Supabase URL set: {'Yes' if SUPABASE_URL else 'No'}")
print(f"Supabase Key set: {'Yes' if SUPABASE_KEY else 'No'}")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route("/debug/supabase")
def debug_supabase():
    """Test Supabase connection and data availability"""
    results = {}
    
    # Test MLB predictions table
    try:
        mlb_rows = sb_get("mlb_predictions", "select=count")
        results["mlb_predictions"] = {
            "accessible": True,
            "data": mlb_rows[:5] if mlb_rows else "No rows"
        }
    except Exception as e:
        results["mlb_predictions"] = {"accessible": False, "error": str(e)}
    
    # Test MLB historical table
    try:
        hist_rows = sb_get("mlb_historical", "select=count&limit=1")
        results["mlb_historical"] = {
            "accessible": True,
            "data": hist_rows[:5] if hist_rows else "No rows"
        }
    except Exception as e:
        results["mlb_historical"] = {"accessible": False, "error": str(e)}
    
    # Check environment variables (redacted)
    results["env"] = {
        "SUPABASE_URL": "Set" if os.environ.get("SUPABASE_URL") else "Missing",
        "SUPABASE_ANON_KEY": "Set" if os.environ.get("SUPABASE_ANON_KEY") else "Missing"
    }
    
    return jsonify(results)


# ── Supabase helper ───────────────────────────────────────────────────────────
def sb_get(table, params=""):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    
    all_data = []
    offset = 0
    limit = 1000  # Supabase default max per request
    
    while True:
        # Add range headers for pagination
        headers["Range"] = f"{offset}-{offset + limit - 1}"
        
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}"
        print(f"Fetching from Supabase: {url} (rows {offset}-{offset + limit - 1})")
        
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if not r.ok:
                print(f"Error response: {r.text[:200]}")
                break
                
            data = r.json()
            if not data:  # No more data
                break
                
            all_data.extend(data)
            print(f"Got {len(data)} rows, total so far: {len(all_data)}")
            
            # Check if we got less than the limit (end of data)
            if len(data) < limit:
                break
                
            offset += limit
            
        except Exception as e:
            print(f"Exception in sb_get: {str(e)}")
            break
    
    print(f"Total rows fetched: {len(all_data)}")
    return all_data


# ── Model cache ───────────────────────────────────────────────────────────────
_models = {}

def save_model(name, obj):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(obj, path)
    _models[name] = obj

def load_model(name):
    if name in _models:
        return _models[name]
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        obj = joblib.load(path)
        _models[name] = obj
        return obj
    return None

# ═══════════════════════════════════════════════════════════════
# MLB DISPERSION CALIBRATION
# ═══════════════════════════════════════════════════════════════
# Default MLB overdispersion constant (k). Lower k = more variance / fatter tails.
# Empirically, MLB run scoring fits NegBin with k ≈ 0.55–0.65 per team per game.
# This is calibrated from historical data via /calibrate/mlb and stored separately.
MLB_NEGBIN_K_DEFAULT = 0.60

def _fit_negbin_k(run_series):
    """
    Fit the overdispersion parameter k for a Negative Binomial distribution
    to a series of run totals using method of moments.
    NegBin variance = μ + μ²/k  →  k = μ² / (variance - μ)
    Returns k clamped to [0.30, 1.20] for stability.
    """
    mu  = run_series.mean()
    var = run_series.var()
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

class StackedRegressor:
    """Drop-in replacement: bundle['reg'].predict(X) still works."""
    def __init__(self, base_learners, meta, scaler_ref=None):
        self.base_learners = base_learners  # [gbm, rf, ridge]
        self.meta = meta
    def predict(self, X):
        base_preds = np.column_stack([m.predict(X) for m in self.base_learners])
        return self.meta.predict(base_preds)

class StackedClassifier:
    """Drop-in replacement: bundle['clf'].predict_proba(X) still works."""
    def __init__(self, base_clfs, meta):
        self.base_clfs = base_clfs
        self.meta = meta
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        base_probs = np.column_stack([c.predict_proba(X)[:, 1] for c in self.base_clfs])
        return self.meta.predict_proba(base_probs)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ═══════════════════════════════════════════════════════════════
# MLB SEASON CONSTANTS (FanGraphs Guts! — used by both ML features and heuristic)
# ═══════════════════════════════════════════════════════════════
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
HFA_RUNS = 0.16   # ~0.32 run total home advantage, split between offense/pitching

# ═══════════════════════════════════════════════════════════════
# MLB MODEL
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
    df["sp_ip_diff"]       = df["home_sp_ip"] - df["away_sp_ip"]
    df["home_bp_exposure"] = np.maximum(0, 9.0 - df["home_sp_ip"]) * (df["home_bullpen_era"] / 4.10)
    df["away_bp_exposure"] = np.maximum(0, 9.0 - df["away_sp_ip"]) * (df["away_bullpen_era"] / 4.10)
    df["bp_exposure_diff"] = df["home_bp_exposure"] - df["away_bp_exposure"]
    df["def_oaa_diff"]     = df["home_def_oaa"] - df["away_def_oaa"]

    # ── FIX ML2: Interaction features (capture non-linear relationships) ──
    # starter_quality × bullpen_quality: short-start ace with bad bullpen is different
    df["fip_x_bullpen"] = df["fip_diff"] * df["bullpen_era_diff"]
    # offensive advantage × park factor: wOBA edge compounds in hitter-friendly parks
    df["woba_x_park"] = df["woba_diff"] * df["park_factor"]
    # wind × pitching advantage: wind out compresses pitching quality advantages
    df["wind_x_fip"] = df["wind_out"].astype(float) * df["fip_diff"]

    # ── League run environment context ──
    if "season" in df.columns:
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
    ]

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
        MAX_TRAIN = 8000
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
            n = len(X)
            print(f"  Capped training data: {len(df)} -> {n} rows (Railway timeout protection)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cv_folds = min(3, n)

        if n >= 200:
            # ── FIX 2: Lighter stacking ensemble ─────────────────
            # 60 estimators (was 150/100), shallower trees — saves ~50% time
            gbm = GradientBoostingRegressor(
                n_estimators=60, max_depth=3,
                learning_rate=0.08, subsample=0.8,
                min_samples_leaf=20, random_state=42,
            )
            rf_reg = RandomForestRegressor(
                n_estimators=60, max_depth=5,
                min_samples_leaf=15, max_features=0.7,
                random_state=42, n_jobs=1,
            )
            enet = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=[0.01, 0.1, 1.0],
                cv=cv_folds, random_state=42,
            )

            print(f"  MLB: Training stacking ensemble on {n} rows (lite mode)...")
            oof_gbm = cross_val_predict(gbm, X_scaled, y_margin, cv=cv_folds)
            oof_rf  = cross_val_predict(rf_reg, X_scaled, y_margin, cv=cv_folds)
            oof_enet = cross_val_predict(enet, X_scaled, y_margin, cv=cv_folds)

            gbm.fit(X_scaled, y_margin, sample_weight=fit_weights)
            rf_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
            enet.fit(X_scaled, y_margin)  # ElasticNet: no sample_weight

            meta_X = np.column_stack([oof_gbm, oof_rf, oof_enet])
            meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            meta_reg.fit(meta_X, y_margin)

            # Bias correction from OOF residuals
            oof_meta = meta_reg.predict(meta_X)
            bias_correction = float(np.mean(oof_meta - y_margin.values if hasattr(y_margin, 'values') else oof_meta - y_margin))
            print(f"  MLB bias correction: {bias_correction:+.3f} runs")

            reg = StackedRegressor([gbm, rf_reg, enet], meta_reg, scaler)
            reg_cv = cross_val_score(gbm, X_scaled, y_margin,
                                      cv=cv_folds, scoring="neg_mean_absolute_error")

            explainer = shap.TreeExplainer(gbm)
            model_type = "StackedEnsemble_v2_lite"

            # ── FIX 3: Lighter classifier (GBM + LR, skip RF) ───
            # RF classifier adds ~30% training time with minimal gain
            gbm_clf = GradientBoostingClassifier(
                n_estimators=60, max_depth=3,
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
            print(f"  Stacking meta weights (reg): {meta_reg.coef_.round(3)}")

        else:
            reg = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=cv_folds)
            reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
            reg_cv = cross_val_score(reg, X_scaled, y_margin,
                                      cv=cv_folds, scoring="neg_mean_absolute_error")
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
            "mae_cv": float(-reg_cv.mean()),
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
            "mae_cv": round(float(-reg_cv.mean()), 3),
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

    # Get heuristic predictions (may be 0 for historical games)
    ph = float(game.get("pred_home_runs", 0))
    pa = float(game.get("pred_away_runs", 0))
    
    # Get raw inputs if provided
    home_woba = float(game.get("home_woba", 0.314))
    away_woba = float(game.get("away_woba", 0.314))
    home_sp_fip = float(game.get("home_sp_fip", game.get("home_fip", 4.25)))
    away_sp_fip = float(game.get("away_sp_fip", game.get("away_fip", 4.25)))
    home_fip = float(game.get("home_fip", 4.25))
    away_fip = float(game.get("away_fip", 4.25))
    home_bullpen = float(game.get("home_bullpen_era", 4.10))
    away_bullpen = float(game.get("away_bullpen_era", 4.10))
    park_factor = float(game.get("park_factor", 1.00))
    temp_f = float(game.get("temp_f", 70.0))
    wind_mph = float(game.get("wind_mph", 5.0))
    wind_out_flag = float(game.get("wind_out_flag", 0.0))
    home_rest = float(game.get("home_rest_days", 4.0))
    away_rest = float(game.get("away_rest_days", 4.0))
    home_travel = float(game.get("home_travel", 0.0))
    away_travel = float(game.get("away_travel", 0.0))

    # K/9 and BB/9 — FIX: now included as ML features
    home_k9 = float(game.get("home_k9", 8.5))
    away_k9 = float(game.get("away_k9", 8.5))
    home_bb9 = float(game.get("home_bb9", 3.2))
    away_bb9 = float(game.get("away_bb9", 3.2))

    # SP innings pitched + defensive OAA
    home_sp_ip = float(game.get("home_sp_ip", 5.5))
    away_sp_ip = float(game.get("away_sp_ip", 5.5))
    home_def_oaa = float(game.get("home_def_oaa", 0.0))
    away_def_oaa = float(game.get("away_def_oaa", 0.0))

    # SP FIP known flag
    has_sp_fip = 1 if (home_sp_fip != 4.25 and away_sp_fip != 4.25) else 0

    # Use starter FIP when known, fall back to team FIP
    home_starter_fip = home_sp_fip if home_sp_fip != 4.25 else home_fip
    away_starter_fip = away_sp_fip if away_sp_fip != 4.25 else away_fip

    # Calculate derived features
    home_bp_exposure = max(0, 9.0 - home_sp_ip) * (home_bullpen / 4.10)
    away_bp_exposure = max(0, 9.0 - away_sp_ip) * (away_bullpen / 4.10)

    # Diffs
    woba_diff = home_woba - away_woba
    fip_diff = home_starter_fip - away_starter_fip
    bullpen_era_diff = home_bullpen - away_bullpen
    k_bb_diff = (home_k9 - home_bb9) - (away_k9 - away_bb9)
    wind_out = int(wind_out_flag)

    row_data = {
        # ── Pruned feature set (S2/ML1 fix: 47→22) ──
        "woba_diff": woba_diff,
        "fip_diff": fip_diff,
        "has_sp_fip": has_sp_fip,
        "bullpen_era_diff": bullpen_era_diff,
        "k_bb_diff": k_bb_diff,
        "sp_ip_diff": home_sp_ip - away_sp_ip,
        "bp_exposure_diff": home_bp_exposure - away_bp_exposure,
        "def_oaa_diff": home_def_oaa - away_def_oaa,
        "park_factor": park_factor,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_out": wind_out,
        "is_warm": 1 if temp_f > 75 else 0,
        "is_cold": 1 if temp_f < 45 else 0,
        "rest_diff": home_rest - away_rest,
        "travel_diff": home_travel - away_travel,
        "lg_rpg": DEFAULT_CONSTANTS["lg_rpg"],
        # Interaction features (ML2 fix)
        "fip_x_bullpen": fip_diff * bullpen_era_diff,
        "woba_x_park": woba_diff * park_factor,
        "wind_x_fip": float(wind_out) * fip_diff,
        # Heuristic signal
        "run_diff_pred": ph - pa,
        "has_heuristic": 1 if ph > 0 or pa > 0 else 0,
    }

    # Create DataFrame with only the features the model expects
    row = pd.DataFrame([{k: row_data[k] for k in bundle["feature_cols"]}])
    
    # Scale and predict
    X_s = bundle["scaler"].transform(row[bundle["feature_cols"]])
    raw_margin = float(bundle["reg"].predict(X_s)[0])
    raw_win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])

    # FIX S2b: Apply bias correction to margin prediction
    bias = bundle.get("bias_correction", 0.0)
    margin = raw_margin - bias

    # FIX S3: Apply isotonic calibration to win probability
    isotonic = bundle.get("isotonic")
    if isotonic is not None:
        win_prob = float(isotonic.predict([raw_win_prob])[0])
    else:
        win_prob = raw_win_prob

    # SHAP explanation
    shap_vals = bundle["explainer"].shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    
    shap_out = []
    if len(shap_vals.shape) > 1:
        shap_values_row = shap_vals[0]
    else:
        shap_values_row = shap_vals
        
    for f, v in zip(bundle["feature_cols"], shap_values_row):
        shap_out.append({
            "feature": f,
            "shap": round(float(v), 4),
            "value": round(float(row[f].iloc[0]), 3)
        })
    
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "MLB",
        "ml_margin": round(margin, 2),
        "ml_margin_raw": round(raw_margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "ml_win_prob_raw": round(raw_win_prob, 4),
        "bias_correction": round(bias, 3),
        "shap": shap_out[:10],  # Top 10 SHAP values
        "model_meta": {
            "n_train": bundle["n_train"],
            "n_historical": bundle.get("n_historical", 0),
            "n_current": bundle.get("n_current", 0),
            "mae_cv": bundle["mae_cv"],
            "trained_at": bundle["trained_at"],
            "model_type": bundle["model_type"],
            "has_isotonic": bundle.get("isotonic") is not None,
        },
    }




# ═══════════════════════════════════════════════════════════════
# NBA MODEL
# ═══════════════════════════════════════════════════════════════

def nba_build_features(df):
    df = df.copy()
    df["net_rtg_diff"]    = df["home_net_rtg"].fillna(0) - df["away_net_rtg"].fillna(0)
    df["score_diff_pred"] = df["pred_home_score"].fillna(0) - df["pred_away_score"].fillna(0)
    df["total_pred"]      = df["pred_home_score"].fillna(0) + df["pred_away_score"].fillna(0)
    df["home_fav"]        = (df["model_ml_home"] < 0).astype(int)
    df["spread_diff"]     = df["spread_home"].fillna(0) - df["market_spread_home"].fillna(0)
    df["ou_gap"]          = df["total_pred"] - df["market_ou_total"].fillna(df["ou_total"].fillna(220))
    df["win_pct_home"]    = df["win_pct_home"].fillna(0.5)

    feature_cols = [
        "pred_home_score", "pred_away_score",
        "home_net_rtg", "away_net_rtg",
        "net_rtg_diff", "score_diff_pred",
        "total_pred", "home_fav",
        "win_pct_home", "ou_gap",
    ]
    return df[feature_cols].fillna(0)

def train_nba():
    rows = sb_get("nba_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    if len(rows) < 10:
        return {"error": "Not enough NBA data", "n": len(rows)}

    df = pd.DataFrame(rows)
    X  = nba_build_features(df)
    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    y_win    = (y_margin > 0).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.1, random_state=42)
    reg.fit(X_scaled, y_margin)
    reg_cv = cross_val_score(reg, X_scaled, y_margin,
                              cv=min(5, len(df)), scoring="neg_mean_absolute_error")

    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), cv=min(5, len(df))
    )
    clf.fit(X_scaled, y_win)

    explainer = shap.TreeExplainer(reg)

    bundle = {
        "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
        "feature_cols": list(X.columns), "n_train": len(df),
        "mae_cv": float(-reg_cv.mean()),
        "trained_at": datetime.utcnow().isoformat(),
    }
    save_model("nba", bundle)
    return {"status": "trained", "n_train": len(df),
            "mae_cv": round(float(-reg_cv.mean()), 3), "features": list(X.columns)}

def predict_nba(game: dict):
    bundle = load_model("nba")
    if not bundle:
        return {"error": "NBA model not trained — call /train/nba first"}

    row = pd.DataFrame([{
        "pred_home_score":  game.get("pred_home_score", 110),
        "pred_away_score":  game.get("pred_away_score", 110),
        "home_net_rtg":     game.get("home_net_rtg", 0),
        "away_net_rtg":     game.get("away_net_rtg", 0),
        "net_rtg_diff":     game.get("home_net_rtg", 0) - game.get("away_net_rtg", 0),
        "score_diff_pred":  game.get("pred_home_score", 110) - game.get("pred_away_score", 110),
        "total_pred":       game.get("pred_home_score", 110) + game.get("pred_away_score", 110),
        "home_fav":         1 if game.get("model_ml_home", 0) < 0 else 0,
        "win_pct_home":     game.get("win_pct_home", 0.5),
        "ou_gap":           (game.get("pred_home_score", 110) + game.get("pred_away_score", 110))
                            - game.get("market_ou_total", game.get("ou_total", 220)),
    }])

    X_s      = bundle["scaler"].transform(row[bundle["feature_cols"]])
    margin   = float(bundle["reg"].predict(X_s)[0])
    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])
    shap_vals = bundle["explainer"].shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_out = [
        {"feature": f, "shap": round(float(v), 4), "value": round(float(row[f].iloc[0]), 3)}
        for f, v in zip(bundle["feature_cols"], shap_vals[0])
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "NBA",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out,
        "model_meta": {"n_train": bundle["n_train"], "mae_cv": bundle["mae_cv"],
                       "trained_at": bundle["trained_at"]},
    }

# ═══════════════════════════════════════════════════════════════
# NCAAB MODEL (v17 — Re-audit fixes R1-R10)
#   R1: Home bias correction via neutral_em_diff + bias subtraction
#   R2: Heuristic signal re-introduced as capped feature
#   R3: Conference game flag + season phase features
#   R4: Isotonic calibration on classifier probabilities
#   R5: Rest days wiring (column detection)
#   R6: Post-training bias correction stored in bundle
#   R7: ElasticNet replaces Ridge for diversity; meta weights logged
#   R8: SOS-weighted interaction features
# ═══════════════════════════════════════════════════════════════

# Conference HCA lookup (same as heuristic, used to decompose adj_em_diff)
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
    Replay a simplified ncaaUtils.js heuristic on historical rows.
    Uses columns available from ncaa_historical enrichment:
      home_adj_em, away_adj_em, home_ppg, away_ppg, home_opp_ppg, away_opp_ppg,
      home_tempo, away_tempo, home_record_wins/losses, away_record_wins/losses,
      home_rank, away_rank, neutral_site, home_conference, away_conference, game_date.

    Mirrors the core ncaaPredictGame logic:
      1. adjEM-based projected spread (KenPom additive formula)
      2. Conference-aware HCA (neutral-site detection)
      3. Win pct via logistic(spread / sigma)
      4. Score projection from tempo × efficiency
      5. Rank boost, record-based form, postseason context
    """
    df = df.copy()

    # Conference ID → HCA mapping (ESPN conference IDs → HCA points)
    # These are the conference IDs from ESPN's API
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
    DEFAULT_HCA = 3.0
    SIGMA = 16.0  # matches live system calibration

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

    # Check if postseason column exists
    is_post = df["is_postseason"].fillna(0).values if "is_postseason" in df.columns else np.zeros(len(df))

    n = len(df)
    pred_home_score = np.zeros(n)
    pred_away_score = np.zeros(n)
    win_pct_home = np.full(n, 0.5)
    spread_home = np.zeros(n)

    for i in range(n):
        # ── 1. Efficiency-based projected scores ──
        # KenPom additive: homeOE + awayDE - lgAvg
        # If we don't have adj_oe/adj_de separately, derive from ppg/opp_ppg
        possessions = (h_tempo[i] + a_tempo[i]) / 2
        lg_avg = 70.0  # approximate NCAA scoring avg

        # Simplified KenPom path using available data
        home_oe = h_ppg[i] if h_ppg[i] > 0 else lg_avg
        away_oe = a_ppg[i] if a_ppg[i] > 0 else lg_avg
        home_de = h_opp[i] if h_opp[i] > 0 else lg_avg
        away_de = a_opp[i] if a_opp[i] > 0 else lg_avg

        # Score projection: (teamOE + oppDE) / 2, scaled by tempo
        tempo_ratio = possessions / 68.0  # vs avg tempo
        hs = ((home_oe + away_de) / 2) * tempo_ratio
        asc = ((away_oe + home_de) / 2) * tempo_ratio

        # ── 2. Home court advantage ──
        if not neutral[i]:
            hca = CONF_ID_HCA.get(str(h_conf[i]).strip(), DEFAULT_HCA)
            hs += hca / 2
            asc -= hca / 2

        # ── 3. Rank boost (exponential, matches JS) ──
        def rank_boost(rank):
            return max(0, 1.2 * np.exp(-rank / 15)) if rank <= 50 else 0
        hs += rank_boost(h_rank[i]) * 0.3
        asc += rank_boost(a_rank[i]) * 0.3

        # ── 4. Record-based form signal ──
        h_games = h_wins[i] + h_losses[i]
        a_games = a_wins[i] + a_losses[i]
        if h_games >= 3:
            h_wp = h_wins[i] / h_games
            hs += (h_wp - 0.5) * 2.0  # ~2 pts swing for strong vs weak record
        if a_games >= 3:
            a_wp = a_wins[i] / a_games
            asc += (a_wp - 0.5) * 2.0

        # ── 5. Postseason / NCAA tournament compression ──
        # Tournament games are closer (neutral site + high stakes = less blowouts)
        if is_post[i]:
            mid = (hs + asc) / 2
            hs = mid + (hs - mid) * 0.90
            asc = mid + (asc - mid) * 0.90

        # ── 6. Safety clamp ──
        hs = max(35, min(130, hs))
        asc = max(35, min(130, asc))

        # ── 7. Spread and win probability ──
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
    df["ou_total"] = pred_home_score + pred_away_score
    # Derive moneyline from win probability (matches JS formula)
    df["model_ml_home"] = [
        int(-round((wp / (1 - wp)) * 100)) if wp >= 0.5
        else int(round(((1 - wp) / wp) * 100))
        for wp in win_pct_home
    ]

    # Stats: how much differentiation did we get?
    wp_std = np.std(win_pct_home)
    wp_range = np.max(win_pct_home) - np.min(win_pct_home)
    non_neutral = (win_pct_home != 0.5).sum()
    print(f"  NCAA heuristic backfill: {n} rows | "
          f"win_pct std={wp_std:.3f}, range=[{np.min(win_pct_home):.3f}, {np.max(win_pct_home):.3f}] | "
          f"{non_neutral}/{n} have non-neutral predictions")

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
        MAX_TRAIN = 12000
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

        cv_folds = min(5, n)
        fit_weights = sample_weights if sample_weights is not None else np.ones(n)

        if n >= 200:
            # ── R7: Stacking with ElasticNet replacing Ridge for diversity ──
            gbm = GradientBoostingRegressor(
                n_estimators=150, max_depth=4,
                learning_rate=0.06, subsample=0.8,
                min_samples_leaf=20, random_state=42,
            )
            rf_reg = RandomForestRegressor(
                n_estimators=100, max_depth=6,
                min_samples_leaf=15, max_features=0.7,
                random_state=42, n_jobs=1,
            )
            # R7 FIX: ElasticNet replaces Ridge — L1 component adds feature selection
            enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                alphas=[0.01, 0.1, 1.0, 5.0],
                                cv=cv_folds, random_state=42)

            print("  NCAAB v17: Training stacking ensemble (GBM + RF + ElasticNet)...")
            oof_gbm   = cross_val_predict(gbm, X_scaled, y_margin, cv=cv_folds)
            oof_rf    = cross_val_predict(rf_reg, X_scaled, y_margin, cv=cv_folds)
            oof_enet  = cross_val_predict(enet, X_scaled, y_margin, cv=cv_folds)

            gbm.fit(X_scaled, y_margin, sample_weight=fit_weights)
            rf_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
            enet.fit(X_scaled, y_margin)

            meta_X = np.column_stack([oof_gbm, oof_rf, oof_enet])
            meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            meta_reg.fit(meta_X, y_margin)

            reg = StackedRegressor([gbm, rf_reg, enet], meta_reg, scaler)
            reg_cv = cross_val_score(gbm, X_scaled, y_margin,
                                      cv=cv_folds, scoring="neg_mean_absolute_error")
            explainer = shap.TreeExplainer(gbm)
            model_type = "StackedEnsemble(GBM+RF+ElasticNet)"

            # R7: Log meta weights for diagnostics
            meta_weights = meta_reg.coef_.round(4).tolist()
            print(f"  NCAAB meta weights [GBM, RF, ElasticNet]: {meta_weights}")
            print(f"  ElasticNet selected: l1_ratio={enet.l1_ratio_}, alpha={enet.alpha_:.4f}")

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
            "mae_cv": float(-reg_cv.mean()), "model_type": model_type,
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
                "mae_cv": round(float(-reg_cv.mean()), 3), "features": list(X.columns),
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

    # AMPLIFICATION FIX: Clamp final win probability to [0.12, 0.88].
    # Without this, the prob→moneyline formula (prob/(1-prob)*100) produces
    # extreme moneylines: 0.775 → -344, 0.85 → -567, 0.90 → -900.
    # College basketball rarely produces true win probabilities above ~85%
    # even in massive mismatches (tournament 1 vs 16 seeds win ~1% of the time).
    win_prob = max(0.12, min(0.88, win_prob))

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

def nfl_build_features(df):
    df = df.copy()
    df["epa_diff"]        = df["home_epa"].fillna(0) - df["away_epa"].fillna(0)
    df["score_diff_pred"] = df["pred_home_score"].fillna(0) - df["pred_away_score"].fillna(0)
    df["total_pred"]      = df["pred_home_score"].fillna(0) + df["pred_away_score"].fillna(0)
    df["home_fav"]        = (df["model_ml_home"] < 0).astype(int)
    df["spread_vs_market"]= df["spread_home"].fillna(0) - df["market_spread_home"].fillna(0)
    df["ou_gap"]          = df["total_pred"] - df["market_ou_total"].fillna(df["ou_total"].fillna(47))
    df["win_pct_home"]    = df["win_pct_home"].fillna(0.5)

    feature_cols = [
        "pred_home_score", "pred_away_score",
        "home_epa", "away_epa", "epa_diff",
        "score_diff_pred", "total_pred",
        "home_fav", "win_pct_home",
        "spread_vs_market", "ou_gap",
    ]
    return df[feature_cols].fillna(0)

def train_nfl():
    rows = sb_get("nfl_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    if len(rows) < 10:
        return {"error": "Not enough NFL data", "n": len(rows)}

    df = pd.DataFrame(rows)
    X  = nfl_build_features(df)
    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    y_win    = (y_margin > 0).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.1, random_state=42)
    reg.fit(X_scaled, y_margin)
    reg_cv = cross_val_score(reg, X_scaled, y_margin,
                              cv=min(5, len(df)), scoring="neg_mean_absolute_error")

    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), cv=min(5, len(df))
    )
    clf.fit(X_scaled, y_win)
    explainer = shap.TreeExplainer(reg)

    bundle = {
        "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
        "feature_cols": list(X.columns), "n_train": len(df),
        "mae_cv": float(-reg_cv.mean()),
        "trained_at": datetime.utcnow().isoformat(),
    }
    save_model("nfl", bundle)
    return {"status": "trained", "n_train": len(df),
            "mae_cv": round(float(-reg_cv.mean()), 3), "features": list(X.columns)}

def predict_nfl(game: dict):
    bundle = load_model("nfl")
    if not bundle:
        return {"error": "NFL model not trained — call /train/nfl first"}

    ph = game.get("pred_home_score", 24)
    pa = game.get("pred_away_score", 24)
    he = game.get("home_epa", 0)
    ae = game.get("away_epa", 0)

    row = pd.DataFrame([{
        "pred_home_score":  ph, "pred_away_score":  pa,
        "home_epa": he, "away_epa": ae, "epa_diff": he - ae,
        "score_diff_pred": ph - pa, "total_pred": ph + pa,
        "home_fav": 1 if game.get("model_ml_home", 0) < 0 else 0,
        "win_pct_home": game.get("win_pct_home", 0.5),
        "spread_vs_market": game.get("spread_home", 0) - game.get("market_spread_home", 0),
        "ou_gap": (ph + pa) - game.get("market_ou_total", game.get("ou_total", 47)),
    }])

    X_s      = bundle["scaler"].transform(row[bundle["feature_cols"]])
    margin   = float(bundle["reg"].predict(X_s)[0])
    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])
    shap_vals = bundle["explainer"].shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_out = [
        {"feature": f, "shap": round(float(v), 4), "value": round(float(row[f].iloc[0]), 3)}
        for f, v in zip(bundle["feature_cols"], shap_vals[0])
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "NFL",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out,
        "model_meta": {"n_train": bundle["n_train"], "mae_cv": bundle["mae_cv"],
                       "trained_at": bundle["trained_at"]},
    }

# ═══════════════════════════════════════════════════════════════
# NCAAF MODEL
# ═══════════════════════════════════════════════════════════════

def ncaaf_build_features(df):
    df = df.copy()
    df["adj_em_diff"]     = df["home_adj_em"].fillna(0) - df["away_adj_em"].fillna(0)
    df["score_diff_pred"] = df["pred_home_score"].fillna(0) - df["pred_away_score"].fillna(0)
    df["total_pred"]      = df["pred_home_score"].fillna(0) + df["pred_away_score"].fillna(0)
    df["home_fav"]        = (df["model_ml_home"] < 0).astype(int)
    df["neutral"]         = df["neutral_site"].fillna(False).astype(int)
    df["ranked_game"]     = ((df["home_rank"].notna()) | (df["away_rank"].notna())).astype(int)
    df["home_rank_fill"]  = df["home_rank"].fillna(99)
    df["away_rank_fill"]  = df["away_rank"].fillna(99)
    df["spread_vs_market"]= df["spread_home"].fillna(0) - df["market_spread_home"].fillna(0)
    df["ou_gap"]          = df["total_pred"] - df["market_ou_total"].fillna(df["ou_total"].fillna(50))
    df["win_pct_home"]    = df["win_pct_home"].fillna(0.5)

    feature_cols = [
        "pred_home_score", "pred_away_score",
        "home_adj_em", "away_adj_em", "adj_em_diff",
        "score_diff_pred", "total_pred",
        "home_fav", "win_pct_home",
        "neutral", "ranked_game",
        "home_rank_fill", "away_rank_fill",
        "spread_vs_market", "ou_gap",
    ]
    return df[feature_cols].fillna(0)

def train_ncaaf():
    rows = sb_get("ncaaf_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    if len(rows) < 10:
        return {"error": "Not enough NCAAF data", "n": len(rows)}

    df = pd.DataFrame(rows)
    X  = ncaaf_build_features(df)
    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    y_win    = (y_margin > 0).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                     learning_rate=0.08, random_state=42)
    reg.fit(X_scaled, y_margin)
    reg_cv = cross_val_score(reg, X_scaled, y_margin,
                              cv=min(5, len(df)), scoring="neg_mean_absolute_error")

    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), cv=min(5, len(df))
    )
    clf.fit(X_scaled, y_win)
    explainer = shap.TreeExplainer(reg)

    bundle = {
        "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
        "feature_cols": list(X.columns), "n_train": len(df),
        "mae_cv": float(-reg_cv.mean()),
        "trained_at": datetime.utcnow().isoformat(),
    }
    save_model("ncaaf", bundle)
    return {"status": "trained", "n_train": len(df),
            "mae_cv": round(float(-reg_cv.mean()), 3), "features": list(X.columns)}

def predict_ncaaf(game: dict):
    bundle = load_model("ncaaf")
    if not bundle:
        return {"error": "NCAAF model not trained — call /train/ncaaf first"}

    ph = game.get("pred_home_score", 28)
    pa = game.get("pred_away_score", 28)
    he = game.get("home_adj_em", 0)
    ae = game.get("away_adj_em", 0)

    row = pd.DataFrame([{
        "pred_home_score": ph, "pred_away_score": pa,
        "home_adj_em": he, "away_adj_em": ae, "adj_em_diff": he - ae,
        "score_diff_pred": ph - pa, "total_pred": ph + pa,
        "home_fav": 1 if game.get("model_ml_home", 0) < 0 else 0,
        "win_pct_home": game.get("win_pct_home", 0.5),
        "neutral": int(game.get("neutral_site", False)),
        "ranked_game": int(game.get("home_rank") is not None or game.get("away_rank") is not None),
        "home_rank_fill": game.get("home_rank", 99),
        "away_rank_fill": game.get("away_rank", 99),
        "spread_vs_market": game.get("spread_home", 0) - game.get("market_spread_home", 0),
        "ou_gap": (ph + pa) - game.get("market_ou_total", game.get("ou_total", 50)),
    }])

    X_s      = bundle["scaler"].transform(row[bundle["feature_cols"]])
    margin   = float(bundle["reg"].predict(X_s)[0])
    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])
    shap_vals = bundle["explainer"].shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_out = [
        {"feature": f, "shap": round(float(v), 4), "value": round(float(row[f].iloc[0]), 3)}
        for f, v in zip(bundle["feature_cols"], shap_vals[0])
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "NCAAF",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out,
        "model_meta": {"n_train": bundle["n_train"], "mae_cv": bundle["mae_cv"],
                       "trained_at": bundle["trained_at"]},
    }

# ═══════════════════════════════════════════════════════════════
# MONTE CARLO  (v2 — MLB upgraded to Negative Binomial + correlation)
# ═══════════════════════════════════════════════════════════════

def _negbin_draw(rng, mu, k, n):
    """
    Draw n samples from NegBin(μ, k).
    Parameterization: mean=μ, variance=μ + μ²/k
    scipy.stats.nbinom uses (r=k, p=k/(k+μ))
    """
    mu  = max(float(mu), 0.5)
    p   = k / (k + mu)
    return rng.negative_binomial(int(round(k * 10)) / 10, p, n).astype(float)
    # Note: scipy parameterization for integer r only; use numpy directly:
    # numpy's negative_binomial(n, p) where n=k, p=k/(k+μ)

def _negbin_draw_numpy(rng, mu, k, size):
    """
    Numpy NegBin draw.  numpy uses n=number of successes (k), p=success prob.
    NegBin(μ, k): p = k / (k + μ)
    """
    mu = max(float(mu), 0.5)
    k  = max(float(k), 0.10)
    p  = k / (k + mu)
    # numpy negative_binomial: r (int or float via workaround)
    # Use gamma-Poisson mixture for non-integer k (more accurate):
    #   λ ~ Gamma(k, μ/k)   then   X ~ Poisson(λ)
    lam = rng.gamma(shape=k, scale=mu / k, size=size)
    return rng.poisson(lam).astype(float)

def monte_carlo(sport, home_mean, away_mean, n_sims=10_000, ou_line=None, game_id=None):
    """
    Run score simulations and return outcome distribution.

    MLB (v2):
      - Negative Binomial draws (overdispersed vs Poisson)
      - Correlated run environment via shared log-normal multiplier
      - Returns run-line cover %, over/under % at posted total

    Others:
      - Normal distribution (unchanged, appropriate for high-scoring sports)
      - Returns spread cover %, over/under % at posted total

    FIX (Finding #17): Seed is now derived from game_id so each game gets
    unique random draws. Fixed seed=42 caused identical simulations for
    any two games with the same run means, wasting 10k draws of signal.
    """
    if game_id is not None:
        seed = hash(str(game_id)) % (2**32)
    else:
        # No game_id provided — use time-based seed for uniqueness
        seed = int(datetime.utcnow().timestamp() * 1000) % (2**32)
    rng = np.random.default_rng(seed)

    if sport == "MLB":
        k_home, k_away = _get_mlb_k()

        # ── Shared run environment (correlation) ─────────────────────────
        # σ_env=0.12 means ~±12% game-level run environment shift.
        # This models factors that affect both teams: umpire strike zone,
        # wind, temperature, park conditions on the day.
        # Validated: MLB game total correlation ≈ 0.10–0.15 between teams.
        sigma_env = 0.12
        env_factor = rng.lognormal(mean=0.0, sigma=sigma_env, size=n_sims)

        home_mean_adj = np.maximum(home_mean * env_factor, 0.5)
        away_mean_adj = np.maximum(away_mean * env_factor, 0.5)

        # ── Negative Binomial draws via Gamma-Poisson mixture ────────────
        # FIX (Finding #15): Removed duplicate scalar-mean draw that was
        # immediately overwritten. That dead code wasted ~40k RNG calls,
        # shifting all subsequent draws to different sequence positions.
        # Only the per-sim correlated draw below is correct — it uses each
        # simulation's individually adjusted mean from the shared env_factor,
        # preserving the home/away run correlation within each simulated game.
        home_lam = rng.gamma(shape=k_home, scale=home_mean_adj / k_home, size=n_sims)
        away_lam = rng.gamma(shape=k_away, scale=away_mean_adj / k_away, size=n_sims)
        home_scores = rng.poisson(home_lam).astype(float)
        away_scores = rng.poisson(away_lam).astype(float)

        distribution_note = (
            f"Negative Binomial (k_home={k_home:.3f}, k_away={k_away:.3f}) "
            f"with correlated run environment (σ={sigma_env})"
        )

    else:
        # F5 FIX: Calibrated base_std per empirical D1 data
        base_std = {"NBA": 11.0, "NCAAB": 10.8, "NFL": 10.5, "NCAAF": 14.0}.get(sport, 10.0)

        # Finding 20: Dynamic σ based on expected game tempo
        avg_total = home_mean + away_mean
        tempo_norm = {"NBA": 220, "NCAAB": 140, "NFL": 44, "NCAAF": 52}.get(sport, 140)
        tempo_factor = avg_total / tempo_norm if tempo_norm > 0 else 1.0
        # F5 FIX: Widened tempo bounds from [0.75,1.25] to [0.80,1.30]
        std = base_std * max(0.80, min(1.30, tempo_factor))

        # Finding 21: Shared pace/environment correlation for basketball
        if sport in ("NBA", "NCAAB"):
            sigma_pace = 0.08  # ~±8% shared pace variance
            pace_factor = rng.lognormal(mean=0.0, sigma=sigma_pace, size=n_sims)
            home_scores = rng.normal(home_mean * pace_factor, std, n_sims)
            away_scores = rng.normal(away_mean * pace_factor, std, n_sims)
            distribution_note = f"Normal(σ={std:.1f}) with pace correlation (σ_pace={sigma_pace})"
        else:
            home_scores = rng.normal(home_mean, std, n_sims)
            away_scores = rng.normal(away_mean, std, n_sims)
            distribution_note = f"Normal(σ={std:.1f})"

    margins = home_scores - away_scores
    totals  = home_scores + away_scores

    # ── Run line / Spread cover probabilities ─────────────────────────────
    # Standard run line for MLB is -1.5 / +1.5
    rl_threshold = 1.5 if sport == "MLB" else 0.5
    home_rl_cover = float((margins > rl_threshold).mean())   # home -1.5 cover
    away_rl_cover = float((margins < -rl_threshold).mean())  # away +1.5 cover

    # ── Over/Under probabilities ──────────────────────────────────────────
    if ou_line is not None:
        over_pct  = float((totals > ou_line).mean())
        under_pct = float((totals < ou_line).mean())
        push_ou   = float((totals == ou_line).mean())
    else:
        # Use the simulation mean as a rough posted line if not provided
        sim_total = float(totals.mean())
        over_pct  = float((totals > sim_total).mean())
        under_pct = float((totals < sim_total).mean())
        push_ou   = float((totals == sim_total).mean())
        ou_line   = round(sim_total, 1)

    return {
        "n_sims":           n_sims,
        "distribution":     distribution_note,

        # Moneyline
        "home_win_pct":     round(float((margins > 0).mean()), 4),
        "away_win_pct":     round(float((margins < 0).mean()), 4),
        "push_pct":         round(float((margins == 0).mean()), 4),

        # Run line / ATS
        "home_rl_cover_pct": round(home_rl_cover, 4),
        "away_rl_cover_pct": round(away_rl_cover, 4),
        "rl_threshold":      rl_threshold,

        # Over/Under
        "ou_line":           ou_line,
        "over_pct":          round(over_pct, 4),
        "under_pct":         round(under_pct, 4),
        "push_ou_pct":       round(push_ou, 4),

        # Score distribution
        "avg_margin":        round(float(margins.mean()), 2),
        "avg_total":         round(float(totals.mean()), 2),
        "std_margin":        round(float(margins.std()), 2),
        "std_total":         round(float(totals.std()), 2),

        "margin_percentiles": {
            "p5":  round(float(np.percentile(margins, 5)),  1),
            "p10": round(float(np.percentile(margins, 10)), 1),
            "p25": round(float(np.percentile(margins, 25)), 1),
            "p50": round(float(np.percentile(margins, 50)), 1),
            "p75": round(float(np.percentile(margins, 75)), 1),
            "p90": round(float(np.percentile(margins, 90)), 1),
            "p95": round(float(np.percentile(margins, 95)), 1),
        },
        "total_percentiles": {
            "p10": round(float(np.percentile(totals, 10)), 1),
            "p25": round(float(np.percentile(totals, 25)), 1),
            "p50": round(float(np.percentile(totals, 50)), 1),
            "p75": round(float(np.percentile(totals, 75)), 1),
            "p90": round(float(np.percentile(totals, 90)), 1),
        },
        "histogram": _histogram(margins, bins=20),
    }

def _histogram(arr, bins=20):
    counts, edges = np.histogram(arr, bins=bins)
    return [
        {"bin": round(float((edges[i] + edges[i+1]) / 2), 1), "count": int(counts[i])}
        for i in range(len(counts))
    ]

# ═══════════════════════════════════════════════════════════════
# MODEL ACCURACY REPORT
# ═══════════════════════════════════════════════════════════════

def accuracy_report(sport_table, sport_label):
    rows = sb_get(sport_table,
                  "result_entered=eq.true&ml_correct=not.is.null"
                  "&select=ml_correct,rl_correct,ou_correct,win_pct_home,"
                  "pred_home_runs,pred_away_runs,pred_home_score,pred_away_score,"
                  "actual_home_runs,actual_away_runs,actual_home_score,actual_away_score,"
                  "ou_total,market_ou_total")
    if not rows:
        return {"error": f"No completed {sport_label} games found"}

    df = pd.DataFrame(rows)
    ml_acc = df["ml_correct"].mean() if "ml_correct" in df else None
    rl_acc = df["rl_correct"].mean() if "rl_correct" in df else None

    # ── O/U accuracy: Compare model's predicted total vs actual total vs line ──
    # FIX: Previous code just checked if ou_correct was "OVER"/"UNDER" (= not a push),
    # which always showed ~99%. Now we properly check if model predicted the right direction.
    ou_acc = None
    ou_n = 0
    # Determine actual and predicted score columns based on sport
    act_h_col = "actual_home_runs" if "actual_home_runs" in df.columns else "actual_home_score"
    act_a_col = "actual_away_runs" if "actual_away_runs" in df.columns else "actual_away_score"
    pred_h_col = "pred_home_runs" if "pred_home_runs" in df.columns else "pred_home_score"
    pred_a_col = "pred_away_runs" if "pred_away_runs" in df.columns else "pred_away_score"

    if all(c in df.columns for c in [act_h_col, act_a_col, pred_h_col, pred_a_col]):
        ou_line = df["market_ou_total"].fillna(df.get("ou_total", pd.Series(dtype=float)))
        actual_total = pd.to_numeric(df[act_h_col], errors="coerce") + \
                       pd.to_numeric(df[act_a_col], errors="coerce")
        pred_total = pd.to_numeric(df[pred_h_col], errors="coerce") + \
                     pd.to_numeric(df[pred_a_col], errors="coerce")

        # Valid rows: have a line, actual total isn't exactly equal to line (not a push)
        valid = ou_line.notna() & actual_total.notna() & pred_total.notna() & \
                (actual_total != ou_line)
        if valid.sum() > 0:
            # Model predicted over if pred_total > line, under if pred_total < line
            model_over = pred_total[valid] > ou_line[valid]
            actual_over = actual_total[valid] > ou_line[valid]
            ou_acc = float((model_over == actual_over).mean())
            ou_n = int(valid.sum())

    # Brier score on win probability (calibration quality)
    brier = None
    if "win_pct_home" in df and "ml_correct" in df:
        sub = df[df["win_pct_home"].notna() & df["ml_correct"].notna()]
        if len(sub) > 5:
            brier = round(float(brier_score_loss(
                sub["ml_correct"].astype(int),
                sub["win_pct_home"].astype(float),
            )), 4)

    return {
        "sport":       sport_label,
        "n_games":     len(df),
        "ml_accuracy": round(float(ml_acc), 4) if ml_acc is not None else None,
        "rl_accuracy": round(float(rl_acc), 4) if rl_acc is not None else None,
        "ou_accuracy": round(float(ou_acc), 4) if ou_acc is not None else None,
        "ou_n_games":  ou_n,
        "brier_score": brier,
        "brier_note":  "Lower is better. Random = 0.25. Perfect calibration approaches 0.",
    }

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({
        "status":  "ok",
        "service": "Multi-Sport Predictor API v2",
        "endpoints": [
            "GET  /health",
            "POST /train/<sport>       (mlb|nba|ncaa|nfl|ncaaf)",
            "POST /train/all",
            "POST /train/all-logged    (shadow comparison + Supabase logging)",
            "POST /calibrate/mlb      (fit NegBin dispersion from historical data)",
            "POST /predict/<sport>",
            "POST /monte-carlo         (body: sport, home_mean, away_mean, n_sims, ou_line)",
            "POST /backtest/mlb        (walk-forward MLB backtest)",
            "POST /backtest/ncaa       (walk-forward NCAAB backtest)",
            "GET  /accuracy/<sport>",
            "GET  /accuracy/all",
            "GET  /model-info/<sport>",
            "POST /cron/auto-train     (daily auto-training — Railway cron)",
            "GET  /cron/status         (model freshness & last training run)",
            "POST /compute/ncaa-efficiency  (KenPom-style ratings — ~360 teams)",
            "GET  /ratings/ncaa        (current ratings from Supabase)",
            "GET  /ratings/ncaa/<id>   (single team rating)",
        ],
    })

@app.route("/health")
def health():
    trained = [s for s in ["mlb", "nba", "ncaa", "nfl", "ncaaf"] if load_model(s)]
    disp    = load_model("mlb_dispersion")
    return jsonify({
        "status":          "healthy",
        "trained_models":  trained,
        "mlb_dispersion":  disp if disp else "not calibrated — POST /calibrate/mlb",
        "timestamp":       datetime.utcnow().isoformat(),
    })

# ── Train endpoints ────────────────────────────────────────────
@app.route("/train/mlb",   methods=["POST"])
def route_train_mlb():   return jsonify(train_mlb())

@app.route("/train/nba",   methods=["POST"])
def route_train_nba():   return jsonify(train_nba())

@app.route("/train/ncaa",  methods=["POST"])
def route_train_ncaa():  return jsonify(train_ncaa())

@app.route("/train/nfl",   methods=["POST"])
def route_train_nfl():   return jsonify(train_nfl())

@app.route("/train/ncaaf", methods=["POST"])
def route_train_ncaaf(): return jsonify(train_ncaaf())

@app.route("/train/all", methods=["POST"])
def route_train_all():
    return jsonify({
        "mlb":   train_mlb(),
        "nba":   train_nba(),
        "ncaa":  train_ncaa(),
        "nfl":   train_nfl(),
        "ncaaf": train_ncaaf(),
    })

# ── MLB dispersion calibration ─────────────────────────────────
@app.route("/calibrate/mlb", methods=["POST"])
def route_calibrate_mlb():
    """
    Fit NegBin overdispersion parameter k from historical MLB run data.
    Call this once per season (or after 50+ new games are logged).
    The result is cached and auto-applied to all subsequent Monte Carlo calls.
    """
    return jsonify(calibrate_mlb_dispersion())

# ── Predict endpoints ──────────────────────────────────────────
@app.route("/predict/<sport>", methods=["POST"])
def route_predict(sport):
    game = request.get_json(force=True, silent=True) or {}
    fns  = {
        "mlb":   predict_mlb,
        "nba":   predict_nba,
        "ncaa":  predict_ncaa,
        "nfl":   predict_nfl,
        "ncaaf": predict_ncaaf,
    }
    fn = fns.get(sport.lower())
    if not fn:
        return jsonify({"error": f"Unknown sport: {sport}"}), 400
    return jsonify(fn(game))

# ── Monte Carlo ────────────────────────────────────────────────
@app.route("/monte-carlo", methods=["POST"])
def route_monte_carlo():
    body      = request.get_json(force=True, silent=True) or {}
    sport     = body.get("sport", "NBA").upper()
    home_mean = float(body.get("home_mean", 110))
    away_mean = float(body.get("away_mean", 110))
    n_sims    = min(int(body.get("n_sims", 10000)), 100_000)
    ou_line   = body.get("ou_line", None)  # NEW: pass the posted O/U line
    if ou_line is not None:
        ou_line = float(ou_line)
    game_id   = body.get("game_id", None)
    return jsonify(monte_carlo(sport, home_mean, away_mean, n_sims, ou_line, game_id))

# ── Accuracy reports ───────────────────────────────────────────
SPORT_TABLES = {
    "mlb":   ("mlb_predictions",   "MLB"),
    "nba":   ("nba_predictions",   "NBA"),
    "ncaa":  ("ncaa_predictions",  "NCAAB"),
    "nfl":   ("nfl_predictions",   "NFL"),
    "ncaaf": ("ncaaf_predictions", "NCAAF"),
}

@app.route("/accuracy/<sport>")
def route_accuracy(sport):
    cfg = SPORT_TABLES.get(sport.lower())
    if not cfg:
        return jsonify({"error": f"Unknown sport: {sport}"}), 400
    return jsonify(accuracy_report(*cfg))

@app.route("/accuracy/all")
def route_accuracy_all():
    return jsonify({k: accuracy_report(*v) for k, v in SPORT_TABLES.items()})

# ── Model info ─────────────────────────────────────────────────
@app.route("/model-info/<sport>")
def route_model_info(sport):
    bundle = load_model(sport.lower())
    if not bundle:
        return jsonify({"error": f"{sport} model not trained yet"})
    info = {
        "sport":      sport.upper(),
        "n_train":    bundle.get("n_train"),
        "mae_cv":     bundle.get("mae_cv"),
        "trained_at": bundle.get("trained_at"),
        "features":   bundle.get("feature_cols"),
        "alpha":      bundle.get("alpha"),
    }
    if sport.lower() == "mlb":
        info["dispersion"] = load_model("mlb_dispersion")
    return jsonify(info)


# ═══════════════════════════════════════════════════════════════
# NBA CONFIDENCE & CALIBRATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

@app.route("/backtest/nba-confidence")
def nba_confidence_calibration():
    """NBA confidence/calibration analysis — mirrors NCAA version."""
    rows = sb_get("nba_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&ml_correct=not.is.null"
                  "&select=*&order=game_date.asc")
    if not rows or len(rows) < 50:
        return jsonify({"error": "Need 50+ graded NBA games", "n": len(rows) if rows else 0})

    df = pd.DataFrame(rows)
    df["win_pct_home"] = pd.to_numeric(df["win_pct_home"], errors="coerce").fillna(0.5)
    df["ml_correct"] = df["ml_correct"].astype(bool)
    df["confidence"] = df["confidence"].fillna("MEDIUM")

    # Accuracy by confidence tier
    tier_results = {}
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        subset = df[df["confidence"] == tier]
        if len(subset) > 0:
            tier_results[tier] = {
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "avg_win_pct_margin": round(float(
                    (subset["win_pct_home"].clip(0.5, 1.0) - 0.5).mean()
                ), 4),
            }

    # Accuracy by win probability margin decile
    df["conf_margin"] = (df["win_pct_home"] - 0.5).abs()
    decile_results = []
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for j in range(len(thresholds) - 1):
        lo, hi = thresholds[j], thresholds[j + 1]
        subset = df[(df["conf_margin"] >= lo) & (df["conf_margin"] < hi)]
        if len(subset) >= 5:
            decile_results.append({
                "margin_range": f"{lo:.2f}-{hi:.2f}",
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "expected_accuracy": round(0.5 + (lo + hi) / 2, 4),
            })

    # Cumulative: "If I only bet on games with margin >= X, what accuracy?"
    cumulative = []
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        subset = df[df["conf_margin"] >= threshold]
        if len(subset) >= 5:
            cumulative.append({
                "min_margin": threshold,
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "pct_of_total": round(len(subset) / len(df), 4),
            })

    # Brier score
    brier_overall = round(float(np.mean(
        (df["win_pct_home"].clip(0.5, 0.97) - df["ml_correct"].astype(float)) ** 2
    )), 4)

    # Suggested thresholds
    suggested_medium = suggested_high = None
    for row in cumulative:
        if row["accuracy"] >= 0.55 and suggested_medium is None:
            suggested_medium = row["min_margin"]
        if row["accuracy"] >= 0.65 and suggested_high is None:
            suggested_high = row["min_margin"]

    return jsonify({
        "total_games": len(df),
        "overall_accuracy": round(float(df["ml_correct"].mean()), 4),
        "brier_score": brier_overall,
        "by_tier": tier_results,
        "by_margin_decile": decile_results,
        "cumulative_threshold": cumulative,
        "suggested_thresholds": {
            "MEDIUM_min_margin": suggested_medium,
            "HIGH_min_margin": suggested_high,
        },
    })


@app.route("/debug/nba-calibration")
def nba_calibration_diagnostic():
    """NBA calibration deep dive — isotonic mapping and refit potential."""
    rows = sb_get("nba_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null"
                  "&select=win_pct_home,ml_correct,confidence,actual_home_score,actual_away_score")
    if not rows or len(rows) < 50:
        return jsonify({"error": "Need 50+ graded NBA games", "n": len(rows) if rows else 0})

    df = pd.DataFrame(rows)
    df["win_pct_home"] = pd.to_numeric(df["win_pct_home"], errors="coerce").fillna(0.5)
    df["actual_home_score"] = pd.to_numeric(df["actual_home_score"], errors="coerce")
    df["actual_away_score"] = pd.to_numeric(df["actual_away_score"], errors="coerce")
    df["home_win"] = (df["actual_home_score"] > df["actual_away_score"]).astype(int)
    df["conf_margin"] = (df["win_pct_home"] - 0.5).abs()
    df["ml_correct"] = df["ml_correct"].astype(bool)

    # Calibration by confidence margin decile
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    cal_deciles = []
    for j in range(len(thresholds) - 1):
        lo, hi = thresholds[j], thresholds[j + 1]
        subset = df[(df["conf_margin"] >= lo) & (df["conf_margin"] < hi)]
        if len(subset) >= 5:
            actual_acc = float(subset["ml_correct"].mean())
            expected_acc = 0.5 + (lo + hi) / 2
            gap = actual_acc - expected_acc
            cal_deciles.append({
                "margin_range": f"{lo:.2f}-{hi:.2f}",
                "n_games": len(subset),
                "actual_accuracy": round(actual_acc, 4),
                "expected_accuracy": round(expected_acc, 4),
                "gap": round(gap, 4),
                "miscalibrated": abs(gap) > 0.10,
            })

    # Current isotonic mapping
    bundle = load_model("nba")
    iso_info = {"status": "not_loaded"}
    if bundle:
        isotonic = bundle.get("isotonic")
        if isotonic is not None:
            try:
                test_probs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
                mapped = isotonic.predict(test_probs)
                iso_info = {
                    "status": "fitted",
                    "mapping": {f"{p:.2f}": round(float(m), 4) for p, m in zip(test_probs, mapped)},
                }
            except Exception:
                iso_info = {"status": "fitted_but_error"}
        else:
            iso_info = {"status": "null"}

    # Refit potential
    refit_info = {}
    if len(df) >= 50:
        from sklearn.isotonic import IsotonicRegression as IR
        refit_iso = IR(y_min=0.02, y_max=0.98, out_of_bounds="clip")
        refit_iso.fit(df["win_pct_home"].values, df["home_win"].values)
        test_probs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        mapped = refit_iso.predict(test_probs)
        refit_info = {
            "mapping": {f"{p:.2f}": round(float(m), 4) for p, m in zip(test_probs, mapped)},
            "n_samples": len(df),
        }

    # Brier
    brier = round(float(np.mean(
        (df["win_pct_home"].clip(0.5, 0.97) - df["ml_correct"].astype(float)) ** 2
    )), 4)

    # Recommendations
    recs = []
    for d in cal_deciles:
        if d["miscalibrated"]:
            direction = "OVERCONFIDENT" if d["gap"] < 0 else "UNDERCONFIDENT"
            recs.append(
                f"Margin {d['margin_range']}: Model is {direction} "
                f"(actual {d['actual_accuracy']:.1%} vs expected {d['expected_accuracy']:.1%})."
            )

    return jsonify({
        "n_graded_games": len(df),
        "overall_accuracy": round(float(df["ml_correct"].mean()), 4),
        "brier_score": brier,
        "calibration_by_decile": cal_deciles,
        "current_isotonic": iso_info,
        "refit_potential": refit_info,
        "recommendations": recs,
    })


# ═══════════════════════════════════════════════════════════════
# CLV (CLOSING LINE VALUE) DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

@app.route("/debug/clv")
def clv_diagnostic():
    """
    Check CLV tracking pipeline health across all sports.
    Reports how many games have opening/closing lines captured,
    and basic CLV statistics for validated predictions.
    """
    results = {}
    for sport, (table, label) in SPORT_TABLES.items():
        try:
            # Check for CLV columns
            rows = sb_get(table,
                          "result_entered=eq.true&select=game_id,market_spread_home,"
                          "market_ou_total,opening_spread_home,closing_spread_home,"
                          "opening_ou_total,closing_ou_total,spread_home,ou_total,"
                          "win_pct_home,ml_correct&limit=5000")
            if not rows:
                results[sport] = {"status": "no_graded_games"}
                continue

            df = pd.DataFrame(rows)
            n_total = len(df)

            # Check which CLV columns exist and have data
            clv_cols = {
                "market_spread_home": 0, "opening_spread_home": 0,
                "closing_spread_home": 0, "market_ou_total": 0,
                "opening_ou_total": 0, "closing_ou_total": 0,
            }
            for col in clv_cols:
                if col in df.columns:
                    non_null = df[col].dropna()
                    clv_cols[col] = len(non_null)

            # Compute spread CLV if we have both model and closing lines
            spread_clv = None
            if "closing_spread_home" in df.columns and "spread_home" in df.columns:
                has_both = df.dropna(subset=["closing_spread_home", "spread_home"])
                if len(has_both) >= 10:
                    model_spread = pd.to_numeric(has_both["spread_home"], errors="coerce")
                    closing_spread = pd.to_numeric(has_both["closing_spread_home"], errors="coerce")
                    valid = model_spread.notna() & closing_spread.notna()
                    if valid.sum() >= 10:
                        # CLV = how much the line moved toward our prediction
                        # Positive CLV = market moved our direction (good sign)
                        clv_values = (closing_spread - model_spread).abs() - (has_both.loc[valid, "market_spread_home"].astype(float) - model_spread[valid]).abs()
                        spread_clv = {
                            "n_games": int(valid.sum()),
                            "avg_clv": round(float(clv_values.mean()), 3),
                            "positive_clv_pct": round(float((clv_values > 0).mean()), 3),
                        }

            results[sport] = {
                "n_graded_games": n_total,
                "clv_column_coverage": clv_cols,
                "spread_clv_summary": spread_clv,
                "pipeline_health": "ACTIVE" if clv_cols.get("closing_spread_home", 0) > 0 else
                                   "PARTIAL" if clv_cols.get("market_spread_home", 0) > 0 else
                                   "NOT_WIRED",
            }
        except Exception as e:
            results[sport] = {"error": str(e)}

    return jsonify(results)

# ══════════════════════════════════════════════════════════════════
# PYTHON-SIDE HEURISTIC REPLAY (mirrors mlb.js logic)
# Uses columns that ACTUALLY EXIST in mlb_historical:
#   home_woba, away_woba, home_fip, away_fip, home_k9, away_k9,
#   home_bb9, away_bb9, park_factor, home_rest_days, away_rest_days,
#   home_travel, away_travel
# Does NOT use: home_sp_fip (null), home_bullpen_era (missing), temp_f (null)
# SEASON_CONSTANTS, DEFAULT_CONSTANTS, FIP_COEFF, HFA_RUNS defined above
# ══════════════════════════════════════════════════════════════════


def heuristic_predict_row(row):
    """
    Replay the mlb.js v15 heuristic in Python using AVAILABLE historical columns.
    Produces differentiated pred_home_runs / pred_away_runs per game.

    F-03 FIX: Team FIP/K9/BB9 in mlb_historical are end-of-season aggregates, not
    pre-game values. This creates a forward-looking data leak. We apply a 50%
    regression-to-mean discount on all pitching stats to approximate mid-season
    knowledge level. wOBA has the same issue but is less correlated with single-game
    outcomes, so the leak is smaller.

    Also aligned with JS engine fixes:
      F-01: Pythagenpat uses league-RPG exponent (not per-matchup)
      F-02: FIP applied as marginal (vs league FIP, discounted)
      F-04: Updated wOBA fallback (not used here — historical has direct wOBA)
    """
    season = int(row.get("season", 2024))
    sc = SEASON_CONSTANTS.get(season, DEFAULT_CONSTANTS)

    # Available columns
    home_woba = row.get("home_woba")
    away_woba = row.get("away_woba")
    home_fip  = row.get("home_fip")    # team FIP (not starter) — END OF SEASON
    away_fip  = row.get("away_fip")
    home_k9   = row.get("home_k9")
    away_k9   = row.get("away_k9")
    home_bb9  = row.get("home_bb9")
    away_bb9  = row.get("away_bb9")
    pf        = row.get("park_factor")
    rest_h    = row.get("home_rest_days")
    rest_a    = row.get("away_rest_days")
    travel_h  = row.get("home_travel")
    travel_a  = row.get("away_travel")

    # Coerce to float with defaults
    home_woba = float(home_woba) if home_woba is not None and not _isnan(home_woba) else sc["lg_woba"]
    away_woba = float(away_woba) if away_woba is not None and not _isnan(away_woba) else sc["lg_woba"]
    home_fip  = float(home_fip)  if home_fip  is not None and not _isnan(home_fip)  else sc["lg_fip"]
    away_fip  = float(away_fip)  if away_fip  is not None and not _isnan(away_fip)  else sc["lg_fip"]
    home_k9   = float(home_k9)   if home_k9   is not None and not _isnan(home_k9)   else 8.5
    away_k9   = float(away_k9)   if away_k9   is not None and not _isnan(away_k9)   else 8.5
    home_bb9  = float(home_bb9)  if home_bb9  is not None and not _isnan(home_bb9)  else 3.2
    away_bb9  = float(away_bb9)  if away_bb9  is not None and not _isnan(away_bb9)  else 3.2
    pf        = float(pf)        if pf        is not None and not _isnan(pf)        else 1.0
    rest_h    = float(rest_h)    if rest_h    is not None and not _isnan(rest_h)     else 3.0
    rest_a    = float(rest_a)    if rest_a    is not None and not _isnan(rest_a)     else 3.0
    travel_h  = float(travel_h)  if travel_h  is not None and not _isnan(travel_h)  else 0.0
    travel_a  = float(travel_a)  if travel_a  is not None and not _isnan(travel_a)  else 0.0

    lg_woba = sc["lg_woba"]
    woba_scale = sc["woba_scale"]
    lg_rpg = sc["lg_rpg"]
    lg_fip = sc["lg_fip"]
    pa_pg = sc["pa_pg"]

    # ═══════════════════════════════════════════════════════════════
    # FIX S1: Game-date-aware regression (replaces flat 50% discount)
    # ═══════════════════════════════════════════════════════════════
    # End-of-season stats in mlb_historical contain future information.
    # The leak magnitude varies by game date:
    #   - April game: ~85% of season is future → heavy regression
    #   - June game: ~60% is future → moderate regression
    #   - September game: ~10% is future → light regression
    # Formula: discount = games_before_date / total_season_games
    # We approximate via game_date month if available, else use 50%.
    game_date = row.get("game_date")
    if game_date and isinstance(game_date, str) and len(game_date) >= 7:
        try:
            month = int(game_date[5:7])
            day = int(game_date[8:10]) if len(game_date) >= 10 else 15
            # MLB season: ~March 27 (game 1) → September 29 (game 162)
            # Approximate games played by month using cumulative schedule
            GAMES_BY_MONTH_END = {3: 5, 4: 30, 5: 56, 6: 81, 7: 105, 8: 133, 9: 162, 10: 162}
            games_before = GAMES_BY_MONTH_END.get(month - 1, 0)
            games_in_month = GAMES_BY_MONTH_END.get(month, 162) - games_before
            games_so_far = games_before + games_in_month * (day / 30.0)
            LEAK_DISCOUNT = max(0.15, min(0.85, games_so_far / 162.0))
        except (ValueError, TypeError):
            LEAK_DISCOUNT = 0.50  # fallback
    else:
        LEAK_DISCOUNT = 0.50  # no date info → use original midpoint

    home_fip_adj = lg_fip + (home_fip - lg_fip) * LEAK_DISCOUNT
    away_fip_adj = lg_fip + (away_fip - lg_fip) * LEAK_DISCOUNT
    home_k9_adj  = 8.5 + (home_k9 - 8.5) * LEAK_DISCOUNT
    away_k9_adj  = 8.5 + (away_k9 - 8.5) * LEAK_DISCOUNT
    home_bb9_adj = 3.2 + (home_bb9 - 3.2) * LEAK_DISCOUNT
    away_bb9_adj = 3.2 + (away_bb9 - 3.2) * LEAK_DISCOUNT
    # Also regress wOBA (same leak, previously missed)
    home_woba_adj = lg_woba + (home_woba - lg_woba) * LEAK_DISCOUNT
    away_woba_adj = lg_woba + (away_woba - lg_woba) * LEAK_DISCOUNT

    # ── wOBA → Runs (FanGraphs method) — now using regressed wOBA (S1 fix) ──
    hr = lg_rpg + ((home_woba_adj - lg_woba) / woba_scale) * pa_pg
    ar = lg_rpg + ((away_woba_adj - lg_woba) / woba_scale) * pa_pg

    # ── Team FIP impact (F-02 aligned: marginal, discounted) ──
    # Using discounted FIP (leak-adjusted) and 0.65 coefficient (reduced from 1.0
    # to avoid double-counting with wOBA-based run estimate)
    ar += (home_fip_adj - lg_fip) * FIP_COEFF * 0.65
    hr += (away_fip_adj - lg_fip) * FIP_COEFF * 0.65

    # ── K/9 and BB/9 adjustments (using leak-adjusted values) ──
    lg_k9 = 8.5
    lg_bb9 = 3.2
    ar -= (home_k9_adj - lg_k9) * 0.04
    ar += (home_bb9_adj - lg_bb9) * 0.06
    hr -= (away_k9_adj - lg_k9) * 0.04
    hr += (away_bb9_adj - lg_bb9) * 0.06

    # ── Park factor ──
    pf = max(0.86, min(1.28, pf))
    hr *= pf
    ar *= pf

    # ── Home field advantage ──
    hr += HFA_RUNS
    ar -= HFA_RUNS

    # ── Rest/travel adjustments ──
    if rest_h == 0:
        hr -= 0.15
    if rest_a == 0:
        ar -= 0.15
    if travel_a > 1500:
        ar -= 0.08

    # ── Clamp ──
    hr = max(2.0, min(9.0, hr))
    ar = max(2.0, min(9.0, ar))

    # ── Pythagenpat (F-01 aligned: fixed league-RPG exponent) ──
    league_rpg = 2 * lg_rpg  # league-wide total RPG
    exp = max(1.60, min(2.10, league_rpg ** 0.287))
    win_pct = (hr ** exp) / (hr ** exp + ar ** exp)
    win_pct = max(0.15, min(0.85, win_pct))

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
# ══════════════════════════════════════════════════════════════════

@app.route("/backtest/mlb", methods=["GET", "POST"])
def route_backtest_mlb():
    """
    Walk-forward backtest with heuristic backfill.
    Body params (all optional):
      - test_seasons: list (default: [2019,2021,2022,2023,2024,2025])
      - use_heuristic: bool (default: true)
      - min_train_seasons: int (default: 3)
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        test_seasons = body.get("test_seasons", [2019, 2021, 2022, 2023, 2024, 2025])
        use_heuristic = body.get("use_heuristic", True)
        min_train = body.get("min_train_seasons", 3)

        all_rows = sb_get(
            "mlb_historical",
            "is_outlier_season=eq.0&actual_home_runs=not.is.null&select=*&order=season.asc&limit=100000"
        )
        if not all_rows or len(all_rows) < 100:
            return jsonify({"error": "Not enough historical data"})

        all_df = pd.DataFrame(all_rows)
        for col in ["actual_home_runs", "actual_away_runs", "home_win",
                     "home_woba", "away_woba", "home_sp_fip", "away_sp_fip",
                     "home_fip", "away_fip", "home_k9", "away_k9",
                     "home_bb9", "away_bb9",
                     "park_factor", "temp_f", "wind_mph", "wind_out_flag",
                     "home_rest_days", "away_rest_days", "home_travel", "away_travel",
                     "season_weight", "season",
                     "home_sp_fip_known", "away_sp_fip_known"]:
            if col in all_df.columns:
                all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

        if use_heuristic:
            print("  Backfilling heuristic predictions...")
            all_df = backfill_heuristic(all_df)
            # Verify differentiation
            wp_std = all_df["win_pct_home"].std()
            wp_min = all_df["win_pct_home"].min()
            wp_max = all_df["win_pct_home"].max()
            print(f"  Heuristic win_pct: min={wp_min:.3f} max={wp_max:.3f} std={wp_std:.3f}")
        else:
            all_df["pred_home_runs"] = 0.0
            all_df["pred_away_runs"] = 0.0
            all_df["win_pct_home"]   = 0.5
            all_df["ou_total"]       = 0.0
            all_df["model_ml_home"]  = 0

        available_seasons = sorted(all_df["season"].dropna().astype(int).unique().tolist())
        results_by_season = []
        all_predictions = []

        for test_season in test_seasons:
            if test_season not in available_seasons:
                continue

            train_df = all_df[all_df["season"] < test_season].copy()
            test_df  = all_df[all_df["season"] == test_season].copy()

            train_seasons = sorted(train_df["season"].dropna().astype(int).unique().tolist())
            if len(train_seasons) < min_train or len(test_df) < 10:
                continue

            X_train = mlb_build_features(train_df)
            y_train_margin = (train_df["actual_home_runs"] - train_df["actual_away_runs"]).values
            y_train_win = train_df["home_win"].astype(int).values

            X_test = mlb_build_features(test_df)
            y_test_margin = (test_df["actual_home_runs"] - test_df["actual_away_runs"]).values
            y_test_win = test_df["home_win"].astype(int).values
            y_test_hr = test_df["actual_home_runs"].values
            y_test_ar = test_df["actual_away_runs"].values

            weights = train_df["season_weight"].fillna(1.0).astype(float).values if "season_weight" in train_df.columns else np.ones(len(train_df))

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            # Lighter models for Railway backtest speed (proxy timeout ~120s)
            gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8, min_samples_leaf=25, random_state=42)
            rf_reg = RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=20, max_features=0.7, random_state=42, n_jobs=1)
            ridge = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=3)

            gbm.fit(X_train_s, y_train_margin, sample_weight=weights)
            rf_reg.fit(X_train_s, y_train_margin, sample_weight=weights)
            ridge.fit(X_train_s, y_train_margin, sample_weight=weights)

            meta_X = np.column_stack([gbm.predict(X_train_s), rf_reg.predict(X_train_s), ridge.predict(X_train_s)])
            meta_reg = Ridge(alpha=1.0)
            meta_reg.fit(meta_X, y_train_margin)

            # Lighter classifier: GBM + LR only (RF classifier adds time, minimal backtest accuracy delta)
            gbm_clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8, min_samples_leaf=25, random_state=42)
            lr_clf = LogisticRegression(max_iter=1000)
            gbm_clf.fit(X_train_s, y_train_win, sample_weight=weights)
            lr_clf.fit(X_train_s, y_train_win, sample_weight=weights)

            test_meta = np.column_stack([gbm.predict(X_test_s), rf_reg.predict(X_test_s), ridge.predict(X_test_s)])
            pred_margin = meta_reg.predict(test_meta)
            pred_wp = 0.6 * gbm_clf.predict_proba(X_test_s)[:, 1] + 0.4 * lr_clf.predict_proba(X_test_s)[:, 1]
            pred_pick = (pred_wp >= 0.5).astype(int)

            accuracy = float(np.mean(pred_pick == y_test_win))
            mae_margin = float(mean_absolute_error(y_test_margin, pred_margin))
            brier = float(brier_score_loss(y_test_win, pred_wp))

            # Run-level MAE
            lg_avg = float(y_test_hr.mean() + y_test_ar.mean()) / 2
            pred_hr = lg_avg + pred_margin / 2
            pred_ar = lg_avg - pred_margin / 2
            mae_home_runs = float(mean_absolute_error(y_test_hr, pred_hr))
            mae_away_runs = float(mean_absolute_error(y_test_ar, pred_ar))
            mae_total = float(mean_absolute_error(y_test_hr + y_test_ar, pred_hr + pred_ar))

            # Heuristic-only comparison
            heur_acc = heur_brier = heur_mae_margin = heur_mae_total = None
            if use_heuristic:
                heur_wp = test_df["win_pct_home"].values
                heur_pick = (heur_wp >= 0.5).astype(int)
                heur_acc = float(np.mean(heur_pick == y_test_win))
                heur_brier = float(brier_score_loss(y_test_win, np.clip(heur_wp, 0.01, 0.99)))
                heur_pred_hr = test_df["pred_home_runs"].values
                heur_pred_ar = test_df["pred_away_runs"].values
                heur_mae_margin = float(mean_absolute_error(y_test_margin, heur_pred_hr - heur_pred_ar))
                heur_mae_total = float(mean_absolute_error(y_test_hr + y_test_ar, heur_pred_hr + heur_pred_ar))

            cal_bins = []
            for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
                mask = (pred_wp >= lo) & (pred_wp < hi)
                n_bin = int(mask.sum())
                if n_bin > 0:
                    cal_bins.append({"range": f"{lo:.0%}-{hi:.0%}", "n": n_bin,
                        "predicted": round(float(pred_wp[mask].mean()), 3),
                        "actual": round(float(y_test_win[mask].mean()), 3)})

            conf_results = []
            for t in [0.52, 0.55, 0.58, 0.60, 0.65]:
                strong = (pred_wp >= t) | (pred_wp <= (1 - t))
                ns = int(strong.sum())
                if ns > 0:
                    conf_results.append({"min_confidence": f"{t:.0%}", "n_games": ns,
                        "accuracy": round(float(np.mean(pred_pick[strong] == y_test_win[strong])), 4)})

            results_by_season.append({
                "season": int(test_season), "n_train": len(train_df), "n_test": len(test_df),
                "train_seasons": train_seasons,
                "ml_accuracy": round(accuracy, 4),
                "ml_brier": round(brier, 4),
                "ml_mae_margin": round(mae_margin, 3),
                "ml_mae_home_runs": round(mae_home_runs, 3),
                "ml_mae_away_runs": round(mae_away_runs, 3),
                "ml_mae_total": round(mae_total, 3),
                "heuristic_accuracy": round(heur_acc, 4) if heur_acc is not None else None,
                "heuristic_brier": round(heur_brier, 4) if heur_brier is not None else None,
                "heuristic_mae_margin": round(heur_mae_margin, 3) if heur_mae_margin is not None else None,
                "heuristic_mae_total": round(heur_mae_total, 3) if heur_mae_total is not None else None,
                "home_win_rate": round(float(y_test_win.mean()), 3),
                "calibration": cal_bins, "confidence_tiers": conf_results,
            })

            for i in range(len(test_df)):
                all_predictions.append({
                    "season": int(test_season),
                    "home_team": str(test_df.iloc[i].get("home_team", "")),
                    "away_team": str(test_df.iloc[i].get("away_team", "")),
                    "pred_win_prob": round(float(pred_wp[i]), 4),
                    "heur_win_prob": round(float(test_df.iloc[i].get("win_pct_home", 0.5)), 4) if use_heuristic else None,
                    "heur_pred_hr": round(float(test_df.iloc[i].get("pred_home_runs", 0)), 2) if use_heuristic else None,
                    "heur_pred_ar": round(float(test_df.iloc[i].get("pred_away_runs", 0)), 2) if use_heuristic else None,
                    "pred_margin": round(float(pred_margin[i]), 2),
                    "actual_margin": int(y_test_margin[i]),
                    "actual_home_win": int(y_test_win[i]),
                    "correct": int(pred_pick[i] == y_test_win[i]),
                })

        if results_by_season:
            total = sum(r["n_test"] for r in results_by_season)
            agg = {
                "total_games_tested": total,
                "seasons_tested": len(results_by_season),
                "heuristic_backfill": use_heuristic,
                "ml_overall_accuracy": round(sum(r["ml_accuracy"] * r["n_test"] for r in results_by_season) / total, 4),
                "ml_overall_brier": round(sum(r["ml_brier"] * r["n_test"] for r in results_by_season) / total, 4),
                "ml_overall_mae_margin": round(sum(r["ml_mae_margin"] * r["n_test"] for r in results_by_season) / total, 3),
                "baseline_home_always": round(sum(r["home_win_rate"] * r["n_test"] for r in results_by_season) / total, 4),
            }
            if use_heuristic:
                agg["heur_overall_accuracy"] = round(sum((r["heuristic_accuracy"] or 0) * r["n_test"] for r in results_by_season) / total, 4)
                agg["heur_overall_brier"] = round(sum((r["heuristic_brier"] or 0) * r["n_test"] for r in results_by_season) / total, 4)
                agg["heur_overall_mae_margin"] = round(sum((r["heuristic_mae_margin"] or 0) * r["n_test"] for r in results_by_season) / total, 3)
                agg["heur_overall_mae_total"] = round(sum((r["heuristic_mae_total"] or 0) * r["n_test"] for r in results_by_season) / total, 3)
        else:
            agg = {"error": "No seasons tested"}

        return jsonify({
            "status": "backtest_complete", "aggregate": agg,
            "by_season": results_by_season, "n_predictions": len(all_predictions),
            "sample_predictions": all_predictions[:20],
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


@app.route("/backtest/mlb/current-model", methods=["POST"])
def route_backtest_current_model():
    """Test the CURRENT production model against a season. Body: { "season": 2024, "use_heuristic": true }"""
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        test_season = int(body.get("season", 2024))
        use_heuristic = body.get("use_heuristic", True)

        bundle = load_model("mlb")
        if not bundle:
            return jsonify({"error": "MLB model not trained"})

        test_rows = sb_get("mlb_historical", f"season=eq.{test_season}&is_outlier_season=eq.0&actual_home_runs=not.is.null&select=*")
        if not test_rows or len(test_rows) < 10:
            return jsonify({"error": f"Not enough data for season {test_season}"})

        test_df = pd.DataFrame(test_rows)
        for col in ["actual_home_runs","actual_away_runs","home_win","home_woba","away_woba",
                     "home_sp_fip","away_sp_fip","home_fip","away_fip","home_k9","away_k9",
                     "home_bb9","away_bb9","park_factor","temp_f","wind_mph","wind_out_flag",
                     "home_rest_days","away_rest_days","home_travel","away_travel"]:
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        if use_heuristic:
            test_df = backfill_heuristic(test_df)
        else:
            test_df["pred_home_runs"] = 0.0
            test_df["pred_away_runs"] = 0.0
            test_df["win_pct_home"]   = 0.5
            test_df["ou_total"]       = 0.0
            test_df["model_ml_home"]  = 0

        X_test = mlb_build_features(test_df)
        y_margin = (test_df["actual_home_runs"] - test_df["actual_away_runs"]).values
        y_win = test_df["home_win"].astype(int).values

        X_s = bundle["scaler"].transform(X_test[bundle["feature_cols"]])
        pred_margin = bundle["reg"].predict(X_s)
        pred_wp = bundle["clf"].predict_proba(X_s)[:, 1]
        pred_pick = (pred_wp >= 0.5).astype(int)

        accuracy = round(float(np.mean(pred_pick == y_win)), 4)
        mae = round(float(mean_absolute_error(y_margin, pred_margin)), 3)
        brier = round(float(brier_score_loss(y_win, pred_wp)), 4)

        heur_acc = heur_mae = None
        if use_heuristic:
            heur_wp = test_df["win_pct_home"].values
            heur_pick = (heur_wp >= 0.5).astype(int)
            heur_acc = round(float(np.mean(heur_pick == y_win)), 4)
            heur_mae = round(float(mean_absolute_error(y_margin, test_df["pred_home_runs"].values - test_df["pred_away_runs"].values)), 3)

        monthly = {}
        if "game_date" in test_df.columns:
            for i in range(len(test_df)):
                m = str(test_df.iloc[i].get("game_date", ""))[:7]
                if not m: continue
                if m not in monthly: monthly[m] = {"n": 0, "correct": 0, "errs": []}
                monthly[m]["n"] += 1
                monthly[m]["correct"] += int(pred_pick[i] == y_win[i])
                monthly[m]["errs"].append(abs(float(pred_margin[i] - y_margin[i])))

        monthly_results = [{"month": m, "n": v["n"], "accuracy": round(v["correct"]/v["n"], 4),
                           "mae": round(float(np.mean(v["errs"])), 3)} for m, v in sorted(monthly.items())]

        return jsonify({
            "status": "current_model_backtest", "test_season": test_season,
            "heuristic_backfill": use_heuristic,
            "n_test": len(test_df), "model_trained_on": bundle.get("n_train", 0),
            "ml_accuracy": accuracy, "ml_brier": brier, "ml_mae_margin": mae,
            "heuristic_accuracy": heur_acc, "heuristic_mae_margin": heur_mae,
            "home_win_rate": round(float(y_win.mean()), 3),
            "monthly": monthly_results,
            "note": "In-sample test. Use /backtest/mlb for unbiased walk-forward.",
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# NCAA BACKTEST (v17: R1-R8 fixes — ElasticNet, bias correction, isotonic)
# ═══════════════════════════════════════════════════════════════

@app.route("/backtest/ncaa", methods=["POST"])
def route_backtest_ncaa():
    """
    Walk-forward backtest for NCAAB predictions.
    v17: Uses ElasticNet, bias correction, isotonic calibration.
    Body: { "min_train": 200 }
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        min_train = int(body.get("min_train", 200))

        rows = sb_get("ncaa_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*&order=game_date.asc")
        if not rows or len(rows) < min_train + 50:
            return jsonify({"error": f"Need {min_train + 50}+ graded games, have {len(rows) if rows else 0}"})

        df = pd.DataFrame(rows)
        for col in ["actual_home_score", "actual_away_score", "pred_home_score", "pred_away_score",
                     "home_adj_em", "away_adj_em", "win_pct_home", "spread_home",
                     "market_spread_home", "market_ou_total", "ou_total", "model_ml_home"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["game_date"] = pd.to_datetime(df["game_date"])
        df["month"] = df["game_date"].dt.to_period("M")
        y_margin = (df["actual_home_score"] - df["actual_away_score"]).to_numpy().astype(float)
        y_win = (y_margin > 0).astype(int)

        months = sorted(df["month"].unique())
        results_by_month = []
        all_predictions = []

        for i, test_month in enumerate(months):
            train_mask = df["month"] < test_month
            test_mask = df["month"] == test_month
            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < min_train or len(test_df) < 5:
                continue

            X_train = ncaa_build_features(train_df)
            X_test = ncaa_build_features(test_df)
            y_train_margin = y_margin[train_mask.to_numpy()]
            y_test_margin = y_margin[test_mask.to_numpy()]
            y_train_win = y_win[train_mask.to_numpy()]
            y_test_win = y_win[test_mask.to_numpy()]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            cv_folds = min(3, len(train_df))

            if len(train_df) >= 200:
                # R7: ElasticNet replaces Ridge
                gbm = GradientBoostingRegressor(
                    n_estimators=150, max_depth=4,
                    learning_rate=0.06, subsample=0.8,
                    min_samples_leaf=20, random_state=42,
                )
                rf_r = RandomForestRegressor(
                    n_estimators=100, max_depth=6,
                    min_samples_leaf=15, max_features=0.7,
                    random_state=42, n_jobs=1,
                )
                enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9],
                                    alphas=[0.01, 0.1, 1.0],
                                    cv=cv_folds, random_state=42)

                oof_g = cross_val_predict(gbm, X_tr, y_train_margin, cv=cv_folds)
                oof_r = cross_val_predict(rf_r, X_tr, y_train_margin, cv=cv_folds)
                oof_e = cross_val_predict(enet, X_tr, y_train_margin, cv=cv_folds)

                gbm.fit(X_tr, y_train_margin)
                rf_r.fit(X_tr, y_train_margin)
                enet.fit(X_tr, y_train_margin)

                meta_X = np.column_stack([oof_g, oof_r, oof_e])
                meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                meta.fit(meta_X, y_train_margin)

                # R6: Compute bias correction from OOF residuals
                oof_meta = meta.predict(meta_X)
                bias_correction = float(np.mean(oof_meta - y_train_margin))

                test_meta_X = np.column_stack([
                    gbm.predict(X_te), rf_r.predict(X_te), enet.predict(X_te)
                ])
                pred_margin = meta.predict(test_meta_X) - bias_correction  # R6: apply

                # Stacked classifier
                gbm_c = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.06, subsample=0.8,
                    min_samples_leaf=20, random_state=42,
                )
                rf_c = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    min_samples_leaf=15, max_features=0.7,
                    random_state=42, n_jobs=1,
                )
                lr_c = LogisticRegression(max_iter=1000, C=1.0)

                oof_gc = cross_val_predict(gbm_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]
                oof_rc = cross_val_predict(rf_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]
                oof_lc = cross_val_predict(lr_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]

                gbm_c.fit(X_tr, y_train_win)
                rf_c.fit(X_tr, y_train_win)
                lr_c.fit(X_tr, y_train_win)

                meta_clf = LogisticRegression(max_iter=1000, C=1.0)
                oof_clf_X = np.column_stack([oof_gc, oof_rc, oof_lc])
                meta_clf.fit(oof_clf_X, y_train_win)

                # R4: Isotonic calibration on OOF stacked probs
                oof_stacked_p = meta_clf.predict_proba(oof_clf_X)[:, 1]
                iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
                iso.fit(oof_stacked_p, y_train_win)

                test_clf_X = np.column_stack([
                    gbm_c.predict_proba(X_te)[:, 1],
                    rf_c.predict_proba(X_te)[:, 1],
                    lr_c.predict_proba(X_te)[:, 1],
                ])
                raw_wp = meta_clf.predict_proba(test_clf_X)[:, 1]
                pred_wp = iso.predict(raw_wp)  # R4: isotonic calibrated
            else:
                reg = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.08, subsample=0.8,
                    min_samples_leaf=15, random_state=42,
                )
                reg.fit(X_tr, y_train_margin)
                pred_margin = reg.predict(X_te)

                clf = CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000), cv=cv_folds
                )
                clf.fit(X_tr, y_train_win)
                pred_wp = clf.predict_proba(X_te)[:, 1]

            pred_pick = (pred_wp >= 0.5).astype(int)

            accuracy = float(np.mean(pred_pick == y_test_win))
            mae_margin = float(mean_absolute_error(y_test_margin, pred_margin))
            brier = float(brier_score_loss(y_test_win, pred_wp))

            # Heuristic baseline (the v16 formula predictions stored in win_pct_home)
            heur_wp = test_df["win_pct_home"].fillna(0.5).values
            heur_pick = (heur_wp >= 0.5).astype(int)
            heur_acc = float(np.mean(heur_pick == y_test_win))
            heur_brier = float(brier_score_loss(y_test_win, heur_wp))

            results_by_month.append({
                "month": str(test_month),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "ml_accuracy": round(accuracy, 4),
                "ml_brier": round(brier, 4),
                "ml_mae_margin": round(mae_margin, 3),
                "heuristic_accuracy": round(heur_acc, 4),
                "heuristic_brier": round(heur_brier, 4),
                "home_win_rate": round(float(y_test_win.mean()), 3),
            })

            for j in range(len(test_df)):
                all_predictions.append({
                    "month": str(test_month),
                    "pred_win_prob": round(float(pred_wp[j]), 4),
                    "heur_win_prob": round(float(heur_wp[j]), 4),
                    "pred_margin": round(float(pred_margin[j]), 2),
                    "actual_margin": int(y_test_margin[j]),
                    "actual_home_win": int(y_test_win[j]),
                    "ml_correct": int(pred_pick[j] == y_test_win[j]),
                    "heur_correct": int(heur_pick[j] == y_test_win[j]),
                })

        if not results_by_month:
            return jsonify({"error": f"No months with >= {min_train} training games"})

        total = sum(r["n_test"] for r in results_by_month)
        agg = {
            "total_games_tested": total,
            "months_tested": len(results_by_month),
            "ml_overall_accuracy": round(sum(r["ml_accuracy"] * r["n_test"] for r in results_by_month) / total, 4),
            "ml_overall_brier": round(sum(r["ml_brier"] * r["n_test"] for r in results_by_month) / total, 4),
            "ml_overall_mae_margin": round(sum(r["ml_mae_margin"] * r["n_test"] for r in results_by_month) / total, 3),
            "heur_overall_accuracy": round(sum(r["heuristic_accuracy"] * r["n_test"] for r in results_by_month) / total, 4),
            "heur_overall_brier": round(sum(r["heuristic_brier"] * r["n_test"] for r in results_by_month) / total, 4),
            "baseline_home_always": round(sum(r["home_win_rate"] * r["n_test"] for r in results_by_month) / total, 4),
        }

        # Confidence tier analysis
        conf_results = []
        all_preds_arr = np.array([(p["pred_win_prob"], p["actual_home_win"]) for p in all_predictions])
        if len(all_preds_arr) > 0:
            for t in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
                strong = (all_preds_arr[:, 0] >= t) | (all_preds_arr[:, 0] <= (1 - t))
                ns = int(strong.sum())
                if ns > 0:
                    pred_side = (all_preds_arr[strong, 0] >= 0.5).astype(int)
                    actual = all_preds_arr[strong, 1].astype(int)
                    conf_results.append({
                        "min_confidence": f"{t:.0%}",
                        "n_games": ns,
                        "accuracy": round(float(np.mean(pred_side == actual)), 4),
                    })

        # F9: Empirical sigma calibration
        sigma_calibration = None
        if len(all_predictions) >= 100:
            from scipy.optimize import minimize_scalar
            spreads = np.array([p["pred_margin"] for p in all_predictions])
            actuals = np.array([p["actual_home_win"] for p in all_predictions])
            def brier_for_sigma(sigma):
                probs = 1 / (1 + np.power(10, -spreads / sigma))
                return np.mean((probs - actuals) ** 2)
            result = minimize_scalar(brier_for_sigma, bounds=(7.0, 16.0), method="bounded")
            sigma_calibration = {
                "optimal_sigma": round(result.x, 2),
                "brier_at_optimal": round(result.fun, 5),
                "brier_at_11": round(brier_for_sigma(11.0), 5),
                "n_games": len(all_predictions),
                "recommendation": f"Set SIGMA = {result.x:.1f} in ncaaUtils.js ncaaPredictGame()"
            }

        return jsonify({
            "status": "backtest_complete",
            "aggregate": agg,
            "by_month": results_by_month,
            "confidence_tiers": conf_results,
            "n_predictions": len(all_predictions),
            "sample_predictions": all_predictions[:20],
            "sigma_calibration": sigma_calibration,
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# AUTO-TRAINING SYSTEM
# ═══════════════════════════════════════════════════════════════
# POST /cron/auto-train  — Called daily by Railway cron at 4 AM ET
# GET  /cron/status      — Model freshness & last training run
# POST /train/all-logged — Manual retrain with shadow comparison + logging
# ═══════════════════════════════════════════════════════════════

def _active_sports():
    """Return list of sports currently in-season."""
    month = datetime.utcnow().month
    active = []
    if month in [3, 4, 5, 6, 7, 8, 9, 10]:      active.append("mlb")
    if month in [10, 11, 12, 1, 2, 3, 4, 5, 6]:  active.append("nba")
    if month in [11, 12, 1, 2, 3, 4]:             active.append("ncaa")
    if month in [9, 10, 11, 12, 1, 2]:            active.append("nfl")
    if month in [8, 9, 10, 11, 12, 1]:            active.append("ncaaf")
    return active


def _log_training(sport, status, result=None, error=None, duration=0.0, trigger="cron"):
    """Write a row to the training_log table in Supabase."""
    try:
        row = {
            "trigger": trigger,
            "sport": sport,
            "status": status,
            "n_train": result.get("n_train") if result else None,
            "mae_cv": result.get("mae_cv") if result else None,
            "model_type": result.get("model_type", "") if result else None,
            "promoted": result.get("_promoted", False) if result else False,
            "promote_reason": result.get("_promote_reason", "") if result else "",
            "details": json.dumps(result) if result else "{}",
            "error_message": str(error)[:500] if error else None,
            "duration_sec": round(duration, 2),
        }
        if result and "_mae_previous" in result:
            row["mae_previous"] = result["_mae_previous"]
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        requests.post(
            f"{SUPABASE_URL}/rest/v1/training_log",
            headers=headers, json=row, timeout=10,
        )
    except Exception as e:
        print(f"  [auto-train] Failed to log training: {e}")


def _should_promote(sport, new_result):
    """Compare new model MAE against current production model."""
    current = load_model(sport)
    if not current:
        return True, "no_existing_model", None
    current_mae = current.get("mae_cv")
    new_mae = new_result.get("mae_cv")
    if current_mae is None or new_mae is None:
        return True, "missing_mae_comparison", current_mae
    improvement = current_mae - new_mae
    if improvement >= -0.01:  # promote if equal or better (0.01 noise tolerance)
        return True, f"mae_delta_{improvement:+.4f}", current_mae
    else:
        return False, f"mae_regressed_{improvement:+.4f}", current_mae


@app.route("/cron/auto-train", methods=["POST"])
def cron_auto_train():
    """
    Daily auto-training with shadow model comparison.
    Called by Railway cron at 4 AM ET (8:00 UTC).

    For each in-season sport:
      1. Train a new model on latest Supabase data
      2. Compare OOF MAE against current production model
      3. Promote only if new model is better (or no model exists)
      4. Log everything to training_log table

    Query params:
      ?force=true       — Train all 5 sports regardless of season
      ?sports=mlb,nba   — Override which sports to train
      ?trigger=manual   — Tag the log entry (default: cron)
    """
    import traceback
    start = _time.time()
    trigger = request.args.get("trigger", "cron")
    force = request.args.get("force", "").lower() == "true"

    if request.args.get("sports"):
        sports = [s.strip().lower() for s in request.args["sports"].split(",")]
    elif force:
        sports = ["mlb", "nba", "ncaa", "nfl", "ncaaf"]
    else:
        sports = _active_sports()

    train_fns = {
        "mlb": train_mlb, "nba": train_nba, "ncaa": train_ncaa,
        "nfl": train_nfl, "ncaaf": train_ncaaf,
    }

    results = {}
    promotions = []
    errors = []

    for sport in sports:
        fn = train_fns.get(sport)
        if not fn:
            results[sport] = {"status": "unknown_sport"}
            continue

        sport_start = _time.time()
        try:
            print(f"\n[auto-train] Training {sport.upper()}...")
            new_result = fn()
            duration = _time.time() - sport_start

            if "error" in new_result:
                results[sport] = {
                    "status": "skipped", "reason": new_result["error"],
                    "duration_sec": round(duration, 2),
                }
                _log_training(sport, "skipped", new_result, duration=duration, trigger=trigger)
                continue

            should_promote, reason, prev_mae = _should_promote(sport, new_result)
            new_result["_promoted"] = should_promote
            new_result["_promote_reason"] = reason
            if prev_mae is not None:
                new_result["_mae_previous"] = prev_mae

            if should_promote:
                promotions.append(sport)
                status = "promoted"
                print(f"  [auto-train] {sport.upper()}: PROMOTED ({reason})")
            else:
                status = "trained_not_promoted"
                print(f"  [auto-train] {sport.upper()}: NOT promoted ({reason})")

            results[sport] = {
                "status": status,
                "n_train": new_result.get("n_train"),
                "mae_cv": new_result.get("mae_cv"),
                "mae_previous": prev_mae,
                "model_type": new_result.get("model_type", ""),
                "promote_reason": reason,
                "duration_sec": round(duration, 2),
            }
            _log_training(sport, status, new_result, duration=duration, trigger=trigger)

        except Exception as e:
            duration = _time.time() - sport_start
            tb = traceback.format_exc()
            print(f"  [auto-train] {sport.upper()} ERROR: {e}")
            results[sport] = {
                "status": "error", "error": str(e),
                "traceback": tb, "duration_sec": round(duration, 2),
            }
            errors.append(sport)
            _log_training(sport, "error", error=e, duration=duration, trigger=trigger)

    # MLB dispersion recalibration
    if "mlb" in sports and "mlb" not in errors:
        try:
            print("\n[auto-train] Calibrating MLB dispersion...")
            results["mlb_dispersion"] = calibrate_mlb_dispersion()
        except Exception as e:
            results["mlb_dispersion"] = {"error": str(e)}

    # NCAA KenPom-style efficiency ratings (nightly)
    if "ncaa" in sports:
        try:
            print("\n[auto-train] Computing NCAA efficiency ratings...")
            eff_result = run_ncaa_efficiency_computation()
            results["ncaa_efficiency"] = {
                "status": "ok",
                "teams_rated": eff_result.get("teams_rated", 0),
                "iterations": eff_result.get("iterations", 0),
                "elapsed_sec": eff_result.get("elapsed_sec", 0),
            }
            print(f"[auto-train] NCAA ratings: {eff_result.get('teams_rated', 0)} teams rated")
        except Exception as e:
            results["ncaa_efficiency"] = {"status": "error", "error": str(e)}
            print(f"[auto-train] NCAA ratings error: {e}")

    total_duration = _time.time() - start
    summary = {
        "status": "complete",
        "timestamp": datetime.utcnow().isoformat(),
        "trigger": trigger,
        "total_duration_sec": round(total_duration, 2),
        "sports_attempted": sports,
        "sports_promoted": promotions,
        "sports_errored": errors,
        "results": results,
    }

    _log_training("all", "complete", {
        "n_train": sum(r.get("n_train", 0) or 0 for r in results.values() if isinstance(r, dict)),
        "_promoted": len(promotions) > 0,
        "_promote_reason": f"promoted:{','.join(promotions)}" if promotions else "none",
    }, duration=total_duration, trigger=trigger)

    print(f"\n[auto-train] Done in {total_duration:.1f}s. Promoted: {promotions or 'none'}")
    return jsonify(summary)


@app.route("/cron/status")
def cron_status():
    """Model freshness, last run info, and in-season detection."""
    status = {}
    for sport in ["mlb", "nba", "ncaa", "nfl", "ncaaf"]:
        model = load_model(sport)
        if model:
            trained_at = model.get("trained_at", "unknown")
            try:
                trained_dt = datetime.fromisoformat(trained_at)
                age_hours = (datetime.utcnow() - trained_dt).total_seconds() / 3600
                freshness = "fresh" if age_hours < 26 else "stale" if age_hours < 72 else "very_stale"
            except Exception:
                age_hours = None
                freshness = "unknown"
            status[sport] = {
                "trained": True, "trained_at": trained_at,
                "age_hours": round(age_hours, 1) if age_hours else None,
                "freshness": freshness,
                "n_train": model.get("n_train"),
                "mae_cv": model.get("mae_cv"),
                "model_type": model.get("model_type", ""),
            }
        else:
            status[sport] = {"trained": False, "freshness": "no_model"}

    last_log = None
    try:
        rows = sb_get("training_log", "sport=eq.all&order=run_at.desc&limit=1")
        if rows:
            last_log = {
                "run_at": rows[0].get("run_at"),
                "status": rows[0].get("status"),
                "duration_sec": rows[0].get("duration_sec"),
            }
    except Exception:
        pass

    return jsonify({
        "active_sports": _active_sports(),
        "models": status,
        "last_cron_run": last_log,
        "next_run": "Daily at 08:00 UTC (4 AM ET)",
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.route("/train/all-logged", methods=["POST"])
def route_train_all_logged():
    """Same as /train/all but with shadow comparison + Supabase logging."""
    with app.test_request_context(
        "/cron/auto-train?force=true&trigger=manual", method="POST"
    ):
        return cron_auto_train()


# ── Debug endpoint ─────────────────────────────────────────────
@app.route("/debug/train-mlb", methods=["POST"])
def debug_train_mlb():
    import traceback
    try:
        result = train_mlb()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500


# ═══════════════════════════════════════════════════════════════
# NCAA KENPOM-STYLE EFFICIENCY RATINGS
# ═══════════════════════════════════════════════════════════════
# Iterative opponent-adjusted efficiency computation for all D1 teams.
# POST /compute/ncaa-efficiency — triggers computation (~3-5 min)
# GET  /ratings/ncaa — returns current ratings from Supabase
#
# Supabase table: ncaa_team_ratings
#   team_id TEXT PRIMARY KEY, team_name TEXT, team_abbr TEXT,
#   conference TEXT, adj_oe REAL, adj_de REAL, adj_em REAL,
#   adj_ppg REAL, adj_opp_ppg REAL, adj_tempo REAL,
#   raw_oe REAL, raw_de REAL, raw_ppg REAL, raw_opp_ppg REAL,
#   sos REAL, wins INT, losses INT, games_used INT,
#   iterations INT, rank_adj_em INT,
#   updated_at TIMESTAMPTZ DEFAULT now()

ESPN_CBB_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

def _espn_cbb_get(path, retries=2):
    """Fetch from ESPN CBB API with retries."""
    url = f"{ESPN_CBB_BASE}/{path}"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                return r.json()
            if r.status_code == 429:
                _time.sleep(2 ** attempt)
                continue
        except Exception as e:
            if attempt == retries:
                print(f"  ESPN CBB fetch failed: {path} — {e}")
    return None


def _fetch_all_d1_teams():
    """Fetch all D1 basketball team IDs from ESPN using pagination."""
    team_ids = set()

    # Method 1: Paginate through groups=50 (all D1)
    # ESPN returns max 50 per page, ~363 D1 teams total = ~8 pages
    print("  Fetching D1 teams (paginated)...")
    page = 1
    while page <= 10:
        data = _espn_cbb_get(f"teams?limit=50&groups=50&page={page}")
        if not data:
            break
        teams_list = []
        if "sports" in data:
            for sport in data["sports"]:
                for league in sport.get("leagues", []):
                    teams_list.extend(league.get("teams", []))
        elif "teams" in data:
            teams_list = data.get("teams", [])

        if not teams_list:
            break

        for team_obj in teams_list:
            t = team_obj.get("team", team_obj)
            tid = t.get("id")
            if tid:
                team_ids.add(str(tid))

        print(f"    Page {page}: +{len(teams_list)} teams (total: {len(team_ids)})")
        if len(teams_list) < 50:
            break  # Last page
        page += 1
        _time.sleep(0.3)

    print(f"  Total from pagination: {len(team_ids)} teams")

    # Method 2: Scoreboard fallback if pagination didn't get enough
    if len(team_ids) < 300:
        print("  Scoreboard fallback...")
        for days_back in range(0, 60, 2):
            dt = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
            try:
                data = _espn_cbb_get(f"scoreboard?dates={dt}&limit=200")
                if data and "events" in data:
                    for ev in data["events"]:
                        for comp in ev.get("competitions", []):
                            for c in comp.get("competitors", []):
                                tid = c.get("team", {}).get("id")
                                if tid:
                                    team_ids.add(str(tid))
            except Exception:
                pass
            if days_back % 20 == 0 and days_back > 0:
                _time.sleep(0.5)
        print(f"  After scoreboard: {len(team_ids)} teams")

    return list(team_ids)


def _fetch_team_data_for_ratings(team_id):
    """Fetch team info, season stats, schedule, and record for ratings."""
    team_data = _espn_cbb_get(f"teams/{team_id}")
    if not team_data:
        return None

    stats_data = _espn_cbb_get(f"teams/{team_id}/statistics")
    sched_data = _espn_cbb_get(f"teams/{team_id}/schedule")
    record_data = _espn_cbb_get(f"teams/{team_id}/record")

    # Handle both {"team": {...}} and direct {...} response formats
    team = team_data.get("team", team_data)
    cats = (stats_data or {}).get("results", {}).get("stats", {}).get("categories", [])
    # Fallback: some ESPN endpoints nest stats differently
    if not cats:
        cats = (stats_data or {}).get("statistics", {}).get("categories", [])
        if not cats:
            # Try splits format
            splits = (stats_data or {}).get("results", {}).get("splits", {}).get("categories", [])
            if splits:
                cats = splits
    def get_stat(name):
        for cat in cats:
            for s in cat.get("stats", []):
                if s.get("name") == name or s.get("displayName") == name:
                    try:
                        return float(s["value"])
                    except (ValueError, TypeError):
                        return None
        return None

    ppg = get_stat("avgPoints") or get_stat("pointsPerGame") or 75.0
    opp_ppg = get_stat("avgPointsAllowed") or get_stat("opponentPointsPerGame") or 75.0
    fga = get_stat("avgFieldGoalsAttempted") or get_stat("fieldGoalsAttempted") or 55.0
    fta = get_stat("avgFreeThrowsAttempted") or get_stat("freeThrowsAttempted") or 18.0
    off_reb = get_stat("avgOffensiveRebounds") or get_stat("offensiveReboundsPerGame") or 10.0
    turnovers = get_stat("avgTurnovers") or 12.0

    # Detect season totals vs per-game
    wins, losses = 0, 0
    # Method 1: record endpoint items
    if record_data and record_data.get("items"):
        for item in record_data["items"]:
            if item.get("type", "") == "total" or item.get("description", "") == "Overall":
                for st in item.get("stats", []):
                    if st.get("name") == "wins":
                        wins = int(st.get("value", 0))
                    if st.get("name") == "losses":
                        losses = int(st.get("value", 0))
    # Method 2: direct wins/losses in record
    if wins == 0 and record_data:
        for item in (record_data.get("items") or []):
            summary = item.get("summary", "")
            if "-" in summary and item.get("type") in ("total", None, ""):
                parts = summary.split("-")
                if len(parts) == 2 and parts[0].strip().isdigit():
                    wins = int(parts[0].strip())
                    losses = int(parts[1].strip())
                    break
    # Method 3: count from game log
    if wins == 0 and sched_data and sched_data.get("events"):
        for ev in sched_data["events"]:
            comp = (ev.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            for c in comp.get("competitors", []):
                if str(c.get("team", {}).get("id")) == str(team_id):
                    if c.get("winner"):
                        wins += 1
                    else:
                        losses += 1

    # Conference: try multiple paths
    conference = ""
    conf_obj = team.get("conference") or team.get("groups", {})
    if isinstance(conf_obj, dict):
        conference = conf_obj.get("name", conf_obj.get("shortName", ""))
    if not conference and team.get("groups", {}).get("parent"):
        conference = team["groups"]["parent"].get("name", "")

    gp = wins + losses or 30
    if fga > 200:
        fga /= gp
        fta /= gp
        off_reb /= gp

    # Tempo (Dean Oliver)
    off_poss = fga - off_reb + turnovers + 0.475 * fta
    tempo = max(55, min(80, off_poss))

    # SOS
    sos = None
    if record_data and record_data.get("items"):
        sos_item = next((i for i in record_data["items"] if i.get("type") == "sos"), None)
        if sos_item:
            sos_stat = next((s for s in sos_item.get("stats", []) if s.get("name") == "opponentWinPercent"), None)
            if sos_stat:
                try:
                    sos = float(sos_stat["value"])
                except (ValueError, TypeError):
                    pass

    # Game log
    game_log = []
    if sched_data and sched_data.get("events"):
        for ev in sched_data["events"]:
            comp = (ev.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            team_comp, opp_comp = None, None
            for c in comp.get("competitors", []):
                if str(c.get("team", {}).get("id")) == str(team_id):
                    team_comp = c
                else:
                    opp_comp = c
            if team_comp and opp_comp:
                try:
                    # ESPN score can be int, str, or dict {"displayValue":"75","value":75.0}
                    def _parse_score(s):
                        if isinstance(s, dict):
                            return int(float(s.get("value", s.get("displayValue", 0))))
                        return int(s)
                    game_log.append({
                        "opp_id": str(opp_comp["team"]["id"]),
                        "my_score": _parse_score(team_comp.get("score", 0)),
                        "opp_score": _parse_score(opp_comp.get("score", 0)),
                        "is_home": team_comp.get("homeAway") == "home",
                        "is_neutral": comp.get("neutralSite", False),
                    })
                except (ValueError, TypeError):
                    continue

    raw_oe = (ppg / tempo * 100) if tempo > 0 else 107.0
    raw_de = (opp_ppg / tempo * 100) if tempo > 0 else 107.0

    return {
        "team_id": str(team_id),
        "name": team.get("displayName", ""),
        "abbr": team.get("abbreviation", ""),
        "conference": conference,
        "ppg": ppg, "opp_ppg": opp_ppg, "tempo": tempo,
        "raw_oe": raw_oe, "raw_de": raw_de,
        "sos": sos, "wins": wins, "losses": losses,
        "game_log": game_log,
    }


def compute_kenpom_ratings(teams_data, max_iterations=8, convergence_threshold=0.01):
    """
    Iterative KenPom-style efficiency computation with 5 enhancements:
    1. Game recency weighting (recent games weighted more)
    2. Opponent rank weighting (top-100 games count more)
    3. Conference strength prior (WCC penalized, B10/SEC boosted)
    4. Margin capping (blowouts capped at ±2 std devs)
    5. Home/away efficiency splits (for spread prediction)
    Plus: post-iteration normalization + dynamic Bayesian shrinkage
    """
    import math

    lookup = {t["team_id"]: t for t in teams_data}
    team_ids = list(lookup.keys())
    n_teams = len(team_ids)

    # ── Conference strength mapping ──
    # Historical conference quality tiers (adj_em shift applied post-convergence)
    # Positive = boost, negative = penalty
    # These are mild priors — they shift EM by 0.5-1.5 pts max
    CONF_PRIOR = {
        # Power conferences (slight boost for depth)
        "SEC": 0.4, "Big Ten": 0.4, "Big 12": 0.3, "ACC": 0.2,
        "Big East": 0.1,
        # Strong mid-majors (no adjustment)
        "American Athletic": 0.0, "Mountain West": 0.0, "Atlantic 10": 0.0,
        "Missouri Valley": 0.0,
        # Weaker conferences (penalty for inflated stats)
        "West Coast": -0.8, "Colonial": -0.5, "Conference USA": -0.5,
        "Sun Belt": -0.5, "Mid-American": -0.6, "Ohio Valley": -0.7,
        "Big South": -0.8, "Southland": -0.8, "MEAC": -1.0, "SWAC": -1.0,
        "Northeast": -0.9, "Patriot": -0.6, "Ivy": -0.4,
        "Horizon": -0.5, "Summit": -0.7, "Big Sky": -0.6,
        "WAC": -0.7, "ASUN": -0.5, "Big West": -0.6,
        "CAA": -0.4, "Southern": -0.6, "America East": -0.7,
        "Atlantic Sun": -0.5, "Metro Atlantic": -0.7, "MAAC": -0.7,
    }

    # ── Compute per-game efficiency std dev for margin capping ──
    all_game_oes = []
    for t in teams_data:
        for g in t["game_log"]:
            if g["opp_id"] in lookup:
                opp = lookup[g["opp_id"]]
                poss = (t["tempo"] + opp["tempo"]) / 2
                if poss > 0:
                    all_game_oes.append(g["my_score"] / poss * 100)
    if all_game_oes:
        oe_global_mean = sum(all_game_oes) / len(all_game_oes)
        oe_global_std = (sum((x - oe_global_mean)**2 for x in all_game_oes) / len(all_game_oes)) ** 0.5
    else:
        oe_global_mean, oe_global_std = 109.7, 15.0
    OE_CAP_LOW = oe_global_mean - 2.0 * oe_global_std
    OE_CAP_HIGH = oe_global_mean + 2.0 * oe_global_std
    print(f"  Margin cap: OE range [{OE_CAP_LOW:.1f}, {OE_CAP_HIGH:.1f}] (mean={oe_global_mean:.1f}, std={oe_global_std:.1f})")

    # ── Recency weights: exponential decay ──
    # Most recent game = 1.0, oldest game ≈ 0.7
    # decay = 0.7^(1/n_games) per game from newest
    RECENCY_FLOOR = 0.70  # oldest game gets 70% weight of newest

    # Iteration 0: raw values
    adj_oe = {tid: lookup[tid]["raw_oe"] for tid in team_ids}
    adj_de = {tid: lookup[tid]["raw_de"] for tid in team_ids}
    adj_ppg = {tid: lookup[tid]["ppg"] for tid in team_ids}
    adj_opp_ppg = {tid: lookup[tid]["opp_ppg"] for tid in team_ids}

    # Home/away tracking (Fix #5)
    home_oes = {tid: [] for tid in team_ids}
    away_oes = {tid: [] for tid in team_ids}
    home_des = {tid: [] for tid in team_ids}
    away_des = {tid: [] for tid in team_ids}

    n_iters = 0
    for iteration in range(max_iterations):
        n_iters = iteration + 1

        all_oe_v = [adj_oe[t] for t in team_ids if adj_oe[t] is not None]
        all_de_v = [adj_de[t] for t in team_ids if adj_de[t] is not None]
        lg_oe = sum(all_oe_v) / len(all_oe_v) if all_oe_v else 109.7
        lg_de = sum(all_de_v) / len(all_de_v) if all_de_v else 109.7
        lg_ppg = sum(lookup[t]["ppg"] for t in team_ids) / len(team_ids)

        # Build current rankings for opponent-rank weighting
        cur_em = {t: adj_oe[t] - adj_de[t] for t in team_ids}
        sorted_by_em = sorted(team_ids, key=lambda t: cur_em[t], reverse=True)
        rank_map = {t: i + 1 for i, t in enumerate(sorted_by_em)}

        new_oe, new_de, new_ppg, new_opp_ppg = {}, {}, {}, {}
        max_delta = 0.0

        # Reset home/away tracking on last iteration
        if iteration == max_iterations - 1 or iteration >= 6:
            home_oes = {tid: [] for tid in team_ids}
            away_oes = {tid: [] for tid in team_ids}
            home_des = {tid: [] for tid in team_ids}
            away_des = {tid: [] for tid in team_ids}

        for tid in team_ids:
            team = lookup[tid]
            if not team["game_log"]:
                new_oe[tid] = adj_oe[tid]
                new_de[tid] = adj_de[tid]
                new_ppg[tid] = adj_ppg[tid]
                new_opp_ppg[tid] = adj_opp_ppg[tid]
                continue

            g_oes, g_des, g_ppgs, g_opps, g_weights = [], [], [], [], []
            n_games = len(team["game_log"])

            for game_idx, game in enumerate(team["game_log"]):
                opp_id = game["opp_id"]
                if opp_id not in lookup:
                    continue

                opp = lookup[opp_id]
                opp_tempo = opp["tempo"]
                opp_de_r = adj_de.get(opp_id, lg_de)
                opp_oe_r = adj_oe.get(opp_id, lg_oe)
                opp_def_ppg = adj_opp_ppg.get(opp_id, opp["opp_ppg"])
                opp_off_ppg = adj_ppg.get(opp_id, opp["ppg"])

                my_score = game["my_score"]
                opp_score = game["opp_score"]
                game_poss = (team["tempo"] + opp_tempo) / 2
                if game_poss <= 0:
                    continue

                # Per-game raw efficiency
                game_oe_raw = my_score / game_poss * 100
                game_de_raw = opp_score / game_poss * 100

                # ── Fix #4: Margin capping ──
                # Cap extreme efficiencies at ±2 std devs
                game_oe_raw = max(OE_CAP_LOW, min(OE_CAP_HIGH, game_oe_raw))
                game_de_raw = max(OE_CAP_LOW, min(OE_CAP_HIGH, game_de_raw))

                # Opponent-adjust
                adj_game_oe = game_oe_raw * (lg_de / opp_de_r) if opp_de_r > 0 else game_oe_raw
                adj_game_de = game_de_raw * (lg_oe / opp_oe_r) if opp_oe_r > 0 else game_de_raw

                # ── Fix #1: Recency weighting ──
                # Games are in chronological order; later index = more recent
                if n_games > 1:
                    recency = RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * (game_idx / (n_games - 1))
                else:
                    recency = 1.0

                # ── Fix #2: Opponent rank weighting ──
                # Top-50 opponents: weight 1.3x, 50-100: 1.1x, 100-200: 1.0x, 200+: 0.85x
                opp_rank = rank_map.get(opp_id, n_teams)
                if opp_rank <= 50:
                    rank_weight = 1.30
                elif opp_rank <= 100:
                    rank_weight = 1.15
                elif opp_rank <= 200:
                    rank_weight = 1.00
                else:
                    rank_weight = 0.85

                # Combined weight
                weight = recency * rank_weight

                g_oes.append(adj_game_oe)
                g_des.append(adj_game_de)
                g_weights.append(weight)

                # PPG-level adjustment
                g_ppgs.append(my_score * (lg_ppg / opp_def_ppg) if opp_def_ppg > 0 else my_score)
                g_opps.append(opp_score * (lg_ppg / opp_off_ppg) if opp_off_ppg > 0 else opp_score)

                # ── Fix #5: Home/away tracking (last iteration only) ──
                if iteration == max_iterations - 1 or iteration >= 6:
                    is_home = game.get("is_home", False)
                    if is_home:
                        home_oes[tid].append(adj_game_oe)
                        home_des[tid].append(adj_game_de)
                    else:
                        away_oes[tid].append(adj_game_oe)
                        away_des[tid].append(adj_game_de)

            if g_oes and sum(g_weights) > 0:
                total_w = sum(g_weights)
                nv_oe = sum(o * w for o, w in zip(g_oes, g_weights)) / total_w
                nv_de = sum(d * w for d, w in zip(g_des, g_weights)) / total_w
                max_delta = max(max_delta, abs(nv_oe - adj_oe[tid]), abs(nv_de - adj_de[tid]))
                new_oe[tid] = nv_oe
                new_de[tid] = nv_de
                # PPG uses simple average (weights less critical for totals)
                new_ppg[tid] = sum(g_ppgs) / len(g_ppgs)
                new_opp_ppg[tid] = sum(g_opps) / len(g_opps)
            else:
                new_oe[tid] = adj_oe[tid]
                new_de[tid] = adj_de[tid]
                new_ppg[tid] = adj_ppg[tid]
                new_opp_ppg[tid] = adj_opp_ppg[tid]

        adj_oe, adj_de = new_oe, new_de
        adj_ppg, adj_opp_ppg = new_ppg, new_opp_ppg

        # ── NORMALIZATION ──
        target_avg = 109.7
        cur_oe_vals = [adj_oe[t] for t in team_ids]
        cur_de_vals = [adj_de[t] for t in team_ids]
        cur_ppg_vals = [adj_ppg[t] for t in team_ids]
        cur_opp_vals = [adj_opp_ppg[t] for t in team_ids]
        oe_mean = sum(cur_oe_vals) / len(cur_oe_vals)
        de_mean = sum(cur_de_vals) / len(cur_de_vals)
        ppg_mean = sum(cur_ppg_vals) / len(cur_ppg_vals)
        opp_mean = sum(cur_opp_vals) / len(cur_opp_vals)
        oe_shift = target_avg - oe_mean
        de_shift = target_avg - de_mean
        ppg_shift = lg_ppg - ppg_mean
        opp_shift = lg_ppg - opp_mean
        for tid in team_ids:
            adj_oe[tid] += oe_shift
            adj_de[tid] += de_shift
            adj_ppg[tid] += ppg_shift
            adj_opp_ppg[tid] += opp_shift

        print(f"  Iteration {n_iters}: max_delta={max_delta:.4f}, oe_mean={oe_mean:.1f}→{target_avg:.1f}")
        if max_delta < convergence_threshold:
            print(f"  Converged after {n_iters} iterations")
            break

    # ── POST-CONVERGENCE: Bayesian shrinkage ──
    avg_games = sum(len(lookup[t]["game_log"]) for t in team_ids) / len(team_ids)
    # Base 0.63 (recalibrated for recency + opponent-rank weighting which
    # compresses distribution ~3 pts vs unweighted iteration)
    SHRINK = 0.63 + 0.35 * avg_games / (avg_games + 8)
    SHRINK = max(0.70, min(0.95, SHRINK))
    final_oe_vals = [adj_oe[t] for t in team_ids]
    final_de_vals = [adj_de[t] for t in team_ids]
    oe_avg = sum(final_oe_vals) / len(final_oe_vals)
    de_avg = sum(final_de_vals) / len(final_de_vals)
    ppg_vals = [adj_ppg[t] for t in team_ids]
    opp_vals = [adj_opp_ppg[t] for t in team_ids]
    ppg_avg = sum(ppg_vals) / len(ppg_vals)
    opp_avg = sum(opp_vals) / len(opp_vals)
    for tid in team_ids:
        adj_oe[tid] = oe_avg + (adj_oe[tid] - oe_avg) * SHRINK
        adj_de[tid] = de_avg + (adj_de[tid] - de_avg) * SHRINK
        adj_ppg[tid] = ppg_avg + (adj_ppg[tid] - ppg_avg) * SHRINK
        adj_opp_ppg[tid] = opp_avg + (adj_opp_ppg[tid] - opp_avg) * SHRINK
    print(f"  Bayesian shrinkage applied (factor={SHRINK:.4f}, avg_games={avg_games:.1f})")

    # ── Fix #3: Conference strength prior ──
    # Apply AFTER shrinkage so it doesn't get compressed.
    # Mild shift: only affects EM by adjusting OE up and DE down (or vice versa)
    conf_applied = 0
    for tid in team_ids:
        conf = lookup[tid].get("conference", "")
        prior = CONF_PRIOR.get(conf, 0.0)
        if prior != 0.0:
            adj_oe[tid] += prior / 2   # half to offense boost
            adj_de[tid] -= prior / 2   # half to defense boost (lower = better)
            adj_ppg[tid] += prior / 4
            adj_opp_ppg[tid] -= prior / 4
            conf_applied += 1
    print(f"  Conference priors applied to {conf_applied}/{n_teams} teams")

    # Build results with rankings and home/away splits
    results = []
    for tid in team_ids:
        t = lookup[tid]
        em = adj_oe[tid] - adj_de[tid]

        # Home/away splits (Fix #5)
        h_oe = sum(home_oes[tid]) / len(home_oes[tid]) if home_oes[tid] else None
        a_oe = sum(away_oes[tid]) / len(away_oes[tid]) if away_oes[tid] else None
        h_de = sum(home_des[tid]) / len(home_des[tid]) if home_des[tid] else None
        a_de = sum(away_des[tid]) / len(away_des[tid]) if away_des[tid] else None

        result = {
            "team_id": tid, "team_name": t["name"], "team_abbr": t["abbr"],
            "conference": t["conference"],
            "adj_oe": round(adj_oe[tid], 2), "adj_de": round(adj_de[tid], 2),
            "adj_em": round(em, 2),
            "adj_ppg": round(adj_ppg[tid], 2), "adj_opp_ppg": round(adj_opp_ppg[tid], 2),
            "adj_tempo": round(t["tempo"], 1),
            "raw_oe": round(t["raw_oe"], 2), "raw_de": round(t["raw_de"], 2),
            "raw_ppg": round(t["ppg"], 2), "raw_opp_ppg": round(t["opp_ppg"], 2),
            "sos": round(t["sos"], 4) if t["sos"] is not None else None,
            "wins": t["wins"], "losses": t["losses"],
            "games_used": len(t["game_log"]), "iterations": n_iters,
        }
        # Add home/away splits if available
        if h_oe is not None:
            result["home_oe"] = round(h_oe, 2)
            result["home_de"] = round(h_de, 2) if h_de else None
        if a_oe is not None:
            result["away_oe"] = round(a_oe, 2)
            result["away_de"] = round(a_de, 2) if a_de else None

        results.append(result)

    results.sort(key=lambda x: x["adj_em"], reverse=True)
    for i, r in enumerate(results):
        r["rank_adj_em"] = i + 1

    return results


def run_ncaa_efficiency_computation():
    """Full pipeline: fetch teams → iterate → store in Supabase."""
    start = _time.time()
    print("\n" + "=" * 60)
    print("NCAA EFFICIENCY RATINGS — KenPom Replication")
    print("=" * 60)

    print("\n[1/4] Fetching D1 team IDs...")
    team_ids = _fetch_all_d1_teams()
    print(f"  Found {len(team_ids)} teams")
    if len(team_ids) < 100:
        return {"error": f"Only found {len(team_ids)} teams", "teams_found": len(team_ids)}

    print(f"\n[2/4] Fetching team data ({len(team_ids)} teams)...")
    teams_data = []
    failed = 0
    for i, tid in enumerate(team_ids):
        if i > 0 and i % 50 == 0:
            print(f"  ... {i}/{len(team_ids)} teams ({failed} failed)")
            _time.sleep(1)
        if i > 0 and i % 10 == 0:
            _time.sleep(0.2)

        data = _fetch_team_data_for_ratings(tid)
        if data and data["game_log"]:
            teams_data.append(data)
        else:
            failed += 1

    print(f"  Loaded {len(teams_data)} teams ({failed} failed)")
    if len(teams_data) < 100:
        return {"error": f"Only {len(teams_data)} teams with data", "teams_loaded": len(teams_data)}

    print(f"\n[3/4] Computing ratings (iterative)...")
    ratings = compute_kenpom_ratings(teams_data)

    print(f"\n[4/4] Storing {len(ratings)} ratings in Supabase...")
    stored = _store_ncaa_ratings(ratings)

    elapsed = _time.time() - start
    top5 = ratings[:5]
    print(f"\nCOMPLETE: {len(ratings)} teams in {elapsed:.1f}s")
    for r in top5:
        print(f"  #{r['rank_adj_em']} {r['team_abbr']:6s} EM={r['adj_em']:+.1f} OE={r['adj_oe']:.1f} DE={r['adj_de']:.1f}")

    return {
        "status": "ok",
        "teams_rated": len(ratings),
        "teams_fetched": len(team_ids),
        "iterations": ratings[0]["iterations"] if ratings else 0,
        "elapsed_sec": round(elapsed, 1),
        "stored_to_supabase": stored,
        "top_10": ratings[:10],
    }


def _store_ncaa_ratings(ratings):
    """Upsert ratings to ncaa_team_ratings in Supabase."""
    if not SUPABASE_KEY:
        print("  ⚠️ No Supabase key — skipping storage")
        return False

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    success = 0
    batch_size = 50
    for i in range(0, len(ratings), batch_size):
        batch = ratings[i:i + batch_size]
        for r in batch:
            r["updated_at"] = datetime.utcnow().isoformat()
        try:
            resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/ncaa_team_ratings",
                headers=headers, json=batch, timeout=15,
            )
            if resp.ok:
                success += len(batch)
            else:
                print(f"  Upsert error (batch {i}): {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"  Upsert exception (batch {i}): {e}")

    print(f"  Stored {success}/{len(ratings)} ratings")
    return success == len(ratings)


@app.route("/compute/ncaa-efficiency", methods=["POST"])
def route_compute_ncaa_efficiency():
    try:
        result = run_ncaa_efficiency_computation()
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/debug/mlb-xwoba")
def route_debug_mlb_xwoba():
    """
    Diagnostic: Check whether xwOBA (Statcast) data is flowing into predictions.
    
    Examines recent MLB prediction rows to determine:
    1. How many rows have home_woba populated
    2. Distribution of wOBA values (real xwOBA clusters ~0.280-0.380, 
       OBP/SLG approximation clusters tighter around 0.310-0.330)
    3. Whether values show real team differentiation or are all ~0.314 (default)
    """
    rows = sb_get(
        "mlb_predictions",
        "home_woba=not.is.null&select=game_date,home_team,away_team,"
        "home_woba,away_woba,home_sp_fip,away_sp_fip,"
        "park_factor,temp_f,wind_mph"
        "&order=game_date.desc&limit=100"
    )
    if not rows:
        return jsonify({"error": "No MLB predictions with wOBA data found"})

    df = pd.DataFrame(rows)
    hw = pd.to_numeric(df["home_woba"], errors="coerce").dropna()
    aw = pd.to_numeric(df["away_woba"], errors="coerce").dropna()
    all_woba = pd.concat([hw, aw])

    # Check for signs of real Statcast data vs approximation
    n_unique = all_woba.round(3).nunique()
    at_default = ((all_woba - 0.314).abs() < 0.002).sum()
    spread = float(all_woba.max() - all_woba.min())

    # Real xwOBA: range ~0.060+, many unique values, few at exact default
    # Approximation: range ~0.030, clusters around 0.310-0.330
    # All defaults: range ~0, nearly all at 0.314
    if spread < 0.005:
        diagnosis = "ALL_DEFAULTS — Statcast pipeline is NOT flowing. All values ~0.314."
    elif spread < 0.035 and n_unique < 15:
        diagnosis = "LIKELY_APPROXIMATION — Values vary but range is narrow. Probably using OBP/SLG formula."
    elif spread >= 0.035 and n_unique >= 15:
        diagnosis = "LIKELY_REAL_XWOBA — Good spread and differentiation. Statcast pipeline appears active."
    else:
        diagnosis = "UNCERTAIN — Some variation but inconclusive."

    # Also check if Statcast-specific features are present
    fip_populated = pd.to_numeric(df.get("home_sp_fip", pd.Series(dtype=float)), errors="coerce").notna().sum()
    park_populated = pd.to_numeric(df.get("park_factor", pd.Series(dtype=float)), errors="coerce").notna().sum()
    temp_populated = pd.to_numeric(df.get("temp_f", pd.Series(dtype=float)), errors="coerce").notna().sum()

    return jsonify({
        "diagnosis": diagnosis,
        "n_rows_checked": len(df),
        "woba_stats": {
            "mean": round(float(all_woba.mean()), 4),
            "std": round(float(all_woba.std()), 4),
            "min": round(float(all_woba.min()), 3),
            "max": round(float(all_woba.max()), 3),
            "spread": round(spread, 4),
            "n_unique_values": int(n_unique),
            "n_at_default_0.314": int(at_default),
        },
        "feature_coverage": {
            "home_woba": int(hw.notna().sum()),
            "away_woba": int(aw.notna().sum()),
            "home_sp_fip": int(fip_populated),
            "park_factor": int(park_populated),
            "temp_f": int(temp_populated),
        },
        "sample_values": df[["game_date", "home_team", "away_team", "home_woba", "away_woba"]].head(10).to_dict(orient="records"),
    })


@app.route("/debug/ncaa-teams")
def route_debug_ncaa_teams():
    """Debug: test team discovery and single team data fetch."""
    results = {}

    # Test 1: Try fetching a known team (Duke = 150)
    duke_raw = _espn_cbb_get("teams/150")
    results["duke_raw_keys"] = list(duke_raw.keys()) if duke_raw else "FAILED"
    if duke_raw:
        team = duke_raw.get("team", duke_raw)
        results["duke_name"] = team.get("displayName", team.get("name", "???"))
        results["duke_id"] = team.get("id", "???")

    # Test 2: Duke schedule — inspect competitor structure
    duke_sched = _espn_cbb_get("teams/150/schedule")
    if duke_sched:
        events = duke_sched.get("events", [])
        completed = [e for e in events if e.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed")]
        results["duke_total_events"] = len(events)
        results["duke_completed"] = len(completed)

        # Inspect first completed game's structure
        if completed:
            first_comp = completed[0].get("competitions", [{}])[0]
            competitors = first_comp.get("competitors", [])
            results["first_game_n_competitors"] = len(competitors)
            comp_details = []
            for c in competitors:
                detail = {
                    "keys": list(c.keys()),
                    "score": c.get("score"),
                    "homeAway": c.get("homeAway"),
                }
                t = c.get("team", {})
                if isinstance(t, dict):
                    detail["team_id"] = t.get("id")
                    detail["team_name"] = t.get("displayName", t.get("shortDisplayName", "???"))
                    detail["team_keys"] = list(t.keys())[:8]
                else:
                    detail["team_raw"] = str(t)[:100]
                comp_details.append(detail)
            results["first_game_competitors"] = comp_details

            # Try parsing with our logic
            team_comp, opp_comp = None, None
            for c in competitors:
                cid = str(c.get("team", {}).get("id", ""))
                if cid == "150":
                    team_comp = c
                else:
                    opp_comp = c
            results["first_game_duke_found"] = team_comp is not None
            results["first_game_opp_found"] = opp_comp is not None
            if team_comp:
                results["first_game_duke_score"] = team_comp.get("score")
                results["first_game_duke_score_type"] = type(team_comp.get("score")).__name__
            if opp_comp:
                results["first_game_opp_score"] = opp_comp.get("score")
                results["first_game_opp_id"] = str(opp_comp.get("team", {}).get("id", ""))
    else:
        results["duke_schedule"] = "FAILED"

    # Test 3: Conference group
    data = _espn_cbb_get("teams?limit=50&groups=50")
    if data:
        results["group50_keys"] = list(data.keys())
        if "sports" in data:
            for sport in data["sports"]:
                for league in sport.get("leagues", []):
                    teams = league.get("teams", [])
                    results["group50_count"] = len(teams)
        elif "teams" in data:
            results["group50_count"] = len(data["teams"])

    # Test 4: Try page param
    data2 = _espn_cbb_get("teams?limit=50&groups=50&page=1")
    if data2:
        if "sports" in data2:
            for sport in data2["sports"]:
                for league in sport.get("leagues", []):
                    results["group50_page1_count"] = len(league.get("teams", []))

    # Test 5: Full data fetch
    duke_full = _fetch_team_data_for_ratings("150")
    if duke_full:
        results["duke_full_ppg"] = duke_full["ppg"]
        results["duke_full_games"] = len(duke_full["game_log"])
        if duke_full["game_log"]:
            results["duke_first_game"] = duke_full["game_log"][0]
    else:
        results["duke_full"] = "FAILED"

    return jsonify(results)


@app.route("/ratings/ncaa")
def route_get_ncaa_ratings():
    ratings = sb_get("ncaa_team_ratings", "select=*&order=rank_adj_em.asc")
    return jsonify({
        "count": len(ratings) if ratings else 0,
        "updated_at": ratings[0].get("updated_at") if ratings else None,
        "ratings": ratings or [],
    })


@app.route("/ratings/ncaa/<team_id>")
def route_get_ncaa_team_rating(team_id):
    rows = sb_get("ncaa_team_ratings", f"select=*&team_id=eq.{team_id}")
    if rows:
        return jsonify(rows[0])
    return jsonify({"error": f"No rating for team {team_id}"}), 404


# ═══════════════════════════════════════════════════════════════
# AUDIT FIX #1: Walk-Forward Validation — NCAA
# Time-based expanding window: never trains on future data.
# This is the HONEST accuracy measurement.
# ═══════════════════════════════════════════════════════════════

@app.route("/debug/ncaa-calibration")
def ncaa_calibration_diagnostic():
    """
    Deep calibration diagnostic for NCAA predictions.
    Shows raw vs calibrated probabilities, identifies systematic gaps,
    and optionally refits isotonic calibration on current season data.
    """
    rows = sb_get("ncaa_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&ml_correct=not.is.null"
                  "&select=*&order=game_date.asc")
    if len(rows) < 50:
        return jsonify({"error": "Need 50+ graded games", "n": len(rows)})

    df = pd.DataFrame(rows)
    df["win_pct_home"] = pd.to_numeric(df["win_pct_home"], errors="coerce").fillna(0.5)
    df["ml_correct"] = df["ml_correct"].astype(bool).astype(int)
    df["conf_margin"] = (df["win_pct_home"] - 0.5).abs()

    # ── 1. Calibration by decile ──────────────────────────────
    deciles = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    calibration = []
    for i in range(len(deciles) - 1):
        lo, hi = deciles[i], deciles[i + 1]
        subset = df[(df["conf_margin"] >= lo) & (df["conf_margin"] < hi)]
        if len(subset) > 0:
            actual_acc = float(subset["ml_correct"].mean())
            expected_acc = 0.5 + (lo + hi) / 2
            gap = actual_acc - expected_acc
            calibration.append({
                "margin_range": f"{lo:.2f}-{hi:.2f}",
                "n_games": len(subset),
                "actual_accuracy": round(actual_acc, 4),
                "expected_accuracy": round(expected_acc, 4),
                "gap": round(gap, 4),
                "miscalibrated": abs(gap) > 0.08,
            })

    # ── 2. Check isotonic calibration quality ─────────────────
    bundle = load_model("ncaa")
    iso_info = None
    if bundle:
        iso = bundle.get("isotonic")
        if iso is not None:
            # Test isotonic mapping at key points
            test_probs = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
            mapped = iso.predict(test_probs)
            iso_info = {
                "status": "fitted",
                "mapping": {f"{p:.2f}": round(float(m), 4) for p, m in zip(test_probs, mapped)},
            }
        else:
            iso_info = {"status": "not_fitted"}

    # ── 3. Recalibration potential ────────────────────────────
    # If we refit isotonic on the current graded data, what would it look like?
    refit_potential = None
    if len(df) > 100:
        try:
            probs = df["win_pct_home"].values
            actuals = df["ml_correct"].values
            refit_iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            refit_iso.fit(probs, actuals)
            test_probs = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
            mapped = refit_iso.predict(test_probs)
            refit_potential = {
                "n_samples": len(df),
                "mapping": {f"{p:.2f}": round(float(m), 4) for p, m in zip(test_probs, mapped)},
                "note": "This shows what isotonic would map if refitted on current graded data",
            }
        except Exception as e:
            refit_potential = {"error": str(e)}

    # ── 4. Recommendations ────────────────────────────────────
    worst_gaps = sorted(calibration, key=lambda x: abs(x["gap"]), reverse=True)[:3]
    recommendations = []
    for g in worst_gaps:
        if g["miscalibrated"]:
            if g["gap"] < 0:
                recommendations.append(
                    f"Margin {g['margin_range']}: Model is OVERCONFIDENT "
                    f"(actual {g['actual_accuracy']:.1%} vs expected {g['expected_accuracy']:.1%}). "
                    f"Isotonic calibration should dampen probabilities in this range."
                )
            else:
                recommendations.append(
                    f"Margin {g['margin_range']}: Model is UNDERCONFIDENT "
                    f"(actual {g['actual_accuracy']:.1%} vs expected {g['expected_accuracy']:.1%}). "
                    f"Isotonic calibration should boost probabilities in this range."
                )

    if not recommendations:
        recommendations.append("Calibration is within acceptable bounds across all deciles.")

    return jsonify({
        "n_graded_games": len(df),
        "overall_accuracy": round(float(df["ml_correct"].mean()), 4),
        "brier_score": round(float(brier_score_loss(
            df["ml_correct"], df["win_pct_home"]
        )), 4),
        "calibration_by_decile": calibration,
        "current_isotonic": iso_info,
        "refit_potential": refit_potential,
        "recommendations": recommendations,
    })


@app.route("/backtest/ncaa-walkforward")
def ncaa_walk_forward():
    rows = sb_get("ncaa_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*&order=game_date.asc")
    if len(rows) < 100:
        return jsonify({"error": "Need 100+ graded games", "n": len(rows)})

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["actual_margin"] = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    df["actual_win"] = (df["actual_margin"] > 0).astype(int)
    df = df.sort_values("game_date").reset_index(drop=True)

    dates = sorted(df["game_date"].unique())
    results = []
    min_train = 200

    cumulative = df.groupby("game_date").size().cumsum()
    start_dates = cumulative[cumulative >= min_train].index
    if len(start_dates) == 0:
        return jsonify({"error": f"Need {min_train}+ games before first test window", "n": len(df)})

    train_cutoff_idx = list(dates).index(start_dates[0])
    window_days = 7
    i = train_cutoff_idx

    while i < len(dates):
        test_start = dates[i]
        test_end = test_start + pd.Timedelta(days=window_days)
        train_df = df[df["game_date"] < test_start]
        test_df = df[(df["game_date"] >= test_start) & (df["game_date"] < test_end)]

        if len(test_df) == 0 or len(train_df) < min_train:
            i += 1
            continue
        try:
            X_train = ncaa_build_features(train_df)
            y_train_m = train_df["actual_margin"]
            y_train_w = train_df["actual_win"]
            X_test = ncaa_build_features(test_df)
            y_test_m = test_df["actual_margin"]
            y_test_w = test_df["actual_win"]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            reg = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                learning_rate=0.06, subsample=0.8, min_samples_leaf=20, random_state=42)
            clf = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                learning_rate=0.06, subsample=0.8, min_samples_leaf=20, random_state=42)
            reg.fit(X_tr_s, y_train_m)
            clf.fit(X_tr_s, y_train_w)

            pred_prob = clf.predict_proba(X_te_s)[:, 1]
            pred_win = (pred_prob >= 0.5).astype(int)
            pred_margin = reg.predict(X_te_s)

            accuracy = float((pred_win == y_test_w.values).mean())
            mae_val = float(np.abs(pred_margin - y_test_m.values).mean())
            brier = float(np.mean((pred_prob - y_test_w.values) ** 2))

            results.append({
                "period": f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}",
                "n_train": len(train_df), "n_test": len(test_df),
                "accuracy": round(accuracy, 4), "mae": round(mae_val, 2), "brier": round(brier, 4),
            })
        except Exception as e:
            results.append({"period": str(test_start.date()), "error": str(e)})

        next_dates = [d for d in dates if d >= test_end]
        if not next_dates:
            break
        i = list(dates).index(next_dates[0])

    valid = [r for r in results if "accuracy" in r]
    if valid:
        total_test = sum(r["n_test"] for r in valid)
        agg = {
            "accuracy": round(sum(r["accuracy"] * r["n_test"] for r in valid) / total_test, 4),
            "mae": round(sum(r["mae"] * r["n_test"] for r in valid) / total_test, 2),
            "brier": round(sum(r["brier"] * r["n_test"] for r in valid) / total_test, 4),
        }
    else:
        agg = {}

    return jsonify({
        "method": "walk-forward (7-day expanding window)",
        "min_train_size": min_train,
        "total_periods": len(results),
        "total_test_games": sum(r.get("n_test", 0) for r in results),
        "aggregate": agg,
        "by_period": results,
    })


# ═══════════════════════════════════════════════════════════════
# AUDIT FIX #5: Confidence Score Calibration — NCAA
# Validates whether HIGH/MEDIUM/LOW tiers actually predict accuracy.
# ═══════════════════════════════════════════════════════════════

@app.route("/backtest/ncaa-confidence")
def ncaa_confidence_calibration():
    rows = sb_get("ncaa_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&ml_correct=not.is.null"
                  "&select=*&order=game_date.asc")
    if len(rows) < 100:
        return jsonify({"error": "Need 100+ graded games", "n": len(rows)})

    df = pd.DataFrame(rows)
    df["win_pct_home"] = pd.to_numeric(df["win_pct_home"], errors="coerce").fillna(0.5)
    df["ml_correct"] = df["ml_correct"].astype(bool)
    df["confidence"] = df["confidence"].fillna("MEDIUM")

    # Accuracy by confidence tier
    tier_results = {}
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        subset = df[df["confidence"] == tier]
        if len(subset) > 0:
            tier_results[tier] = {
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "avg_win_pct_margin": round(float(
                    (subset["win_pct_home"].clip(0.5, 1.0) - 0.5).mean()
                ), 4),
            }

    # Accuracy by win probability margin decile
    df["conf_margin"] = (df["win_pct_home"] - 0.5).abs()
    decile_results = []
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for j in range(len(thresholds) - 1):
        lo, hi = thresholds[j], thresholds[j + 1]
        subset = df[(df["conf_margin"] >= lo) & (df["conf_margin"] < hi)]
        if len(subset) >= 10:
            decile_results.append({
                "margin_range": f"{lo:.2f}-{hi:.2f}",
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "expected_accuracy": round(0.5 + (lo + hi) / 2, 4),
            })

    # Cumulative: "If I only bet on games with margin >= X, what accuracy?"
    cumulative = []
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        subset = df[df["conf_margin"] >= threshold]
        if len(subset) >= 10:
            cumulative.append({
                "min_margin": threshold,
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "pct_of_total": round(len(subset) / len(df), 4),
            })

    # Brier score overall
    brier_overall = round(float(np.mean(
        (df["win_pct_home"].clip(0.5, 0.97) - df["ml_correct"].astype(float)) ** 2
    )), 4)

    # Suggested tier thresholds based on accuracy jumps
    suggested_medium = suggested_high = None
    for row in cumulative:
        if row["accuracy"] >= 0.55 and suggested_medium is None:
            suggested_medium = row["min_margin"]
        if row["accuracy"] >= 0.65 and suggested_high is None:
            suggested_high = row["min_margin"]

    return jsonify({
        "total_games": len(df),
        "overall_accuracy": round(float(df["ml_correct"].mean()), 4),
        "brier_score": brier_overall,
        "by_tier": tier_results,
        "by_margin_decile": decile_results,
        "cumulative_threshold": cumulative,
        "current_thresholds": {"LOW": "<35", "MEDIUM": "35-62", "HIGH": ">=62"},
        "suggested_thresholds": {
            "MEDIUM_min_margin": suggested_medium,
            "HIGH_min_margin": suggested_high,
            "note": "Based on where cumulative accuracy exceeds 55% / 65%"
        },
    })


# ── NCAA Historical Backfill & Enrichment ─────────────────────

@app.route("/backfill/ncaa-historical", methods=["POST"])
def backfill_ncaa_historical():
    """
    Scrape NCAA game results from ESPN for historical seasons.
    Body: { "seasons": [2022, 2023, 2024, 2025] }
    Safe to call multiple times (uses UPSERT on game_id).
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        seasons = body.get("seasons", [2022, 2023, 2024, 2025])
        total_inserted = 0
        season_counts = {}
        errors = []

        for season in seasons:
            start_date = datetime(season - 1, 11, 1)
            end_date = datetime(season, 4, 15)
            current = start_date
            season_count = 0
            batch = []

            print(f"\n[ncaa-backfill] Season {season}: "
                  f"{start_date.date()} -> {end_date.date()}")

            while current <= end_date:
                date_str = current.strftime("%Y%m%d")
                api_date = current.strftime("%Y-%m-%d")
                try:
                    url = (
                        "https://site.api.espn.com/apis/site/v2/sports/"
                        "basketball/mens-college-basketball/scoreboard"
                        f"?dates={date_str}&limit=200"
                    )
                    resp = requests.get(url, timeout=15)
                    if not resp.ok:
                        current += timedelta(days=1)
                        _time.sleep(0.2)
                        continue
                    data = resp.json()
                    for event in data.get("events", []):
                        comp = event.get("competitions", [{}])[0]
                        if not comp.get("status", {}).get("type", {}).get("completed", False):
                            continue
                        competitors = comp.get("competitors", [])
                        if len(competitors) != 2:
                            continue
                        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                        if not home or not away:
                            continue
                        home_score = int(home.get("score", 0))
                        away_score = int(away.get("score", 0))
                        home_team = home.get("team", {})
                        away_team = away.get("team", {})
                        if not home_team.get("id") or not away_team.get("id"):
                            continue
                        neutral = comp.get("neutralSite", False)
                        home_rank = None
                        away_rank = None
                        try:
                            hrv = home.get("curatedRank", {}).get("current")
                            if hrv and hrv <= 25: home_rank = hrv
                            arv = away.get("curatedRank", {}).get("current")
                            if arv and arv <= 25: away_rank = arv
                        except Exception:
                            pass
                        is_post = 0
                        if event.get("season", {}).get("type", 2) == 3:
                            is_post = 1
                        elif api_date >= f"{season}-03-15":
                            is_post = 1
                        batch.append({
                            "game_id": event.get("id", ""),
                            "game_date": api_date,
                            "season": season,
                            "home_team_id": str(home_team.get("id")),
                            "away_team_id": str(away_team.get("id")),
                            "home_team_name": home_team.get("displayName", ""),
                            "away_team_name": away_team.get("displayName", ""),
                            "home_team_abbr": home_team.get("abbreviation", ""),
                            "away_team_abbr": away_team.get("abbreviation", ""),
                            "actual_home_score": home_score,
                            "actual_away_score": away_score,
                            "actual_margin": home_score - away_score,
                            "home_win": 1 if home_score > away_score else 0,
                            "neutral_site": neutral,
                            "home_conference": str(home_team.get("conferenceId", "")),
                            "away_conference": str(away_team.get("conferenceId", "")),
                            "is_postseason": is_post,
                            "home_rank": home_rank,
                            "away_rank": away_rank,
                            "season_weight": _ncaa_season_weight(season),
                        })
                    if len(batch) >= 200:
                        _flush_ncaa_batch(batch)
                        season_count += len(batch)
                        batch = []
                except Exception as e:
                    errors.append(f"{api_date}: {str(e)[:100]}")
                current += timedelta(days=1)
                _time.sleep(0.35)

            if batch:
                _flush_ncaa_batch(batch)
                season_count += len(batch)
            total_inserted += season_count
            season_counts[season] = season_count
            print(f"  Season {season}: {season_count} games inserted")

        return jsonify({
            "status": "complete",
            "total_inserted": total_inserted,
            "by_season": season_counts,
            "errors_count": len(errors),
            "errors_sample": errors[:10],
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/enrich/ncaa-historical", methods=["POST"])
def enrich_ncaa_historical():
    """
    Compute team efficiency ratings from game results and attach to ncaa_historical.
    Body: { "season": 2024 }
    Processes games chronologically (no data leakage).
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        season = body.get("season", 2024)

        rows = sb_get(
            "ncaa_historical",
            f"season=eq.{season}&actual_home_score=not.is.null&select=*&order=game_date.asc"
        )
        if not rows:
            return jsonify({"error": f"No games found for season {season}"})

        df = pd.DataFrame(rows)
        print(f"  Enriching {len(df)} games for season {season}")

        team_stats = {}
        updates = []
        for _, row in df.iterrows():
            h_id = str(row["home_team_id"])
            a_id = str(row["away_team_id"])
            h_score = float(row["actual_home_score"])
            a_score = float(row["actual_away_score"])

            h_stats = team_stats.get(h_id, {"ppg_sum": 0, "opp_sum": 0, "games": 0, "wins": 0, "losses": 0})
            a_stats = team_stats.get(a_id, {"ppg_sum": 0, "opp_sum": 0, "games": 0, "wins": 0, "losses": 0})

            h_ppg = h_stats["ppg_sum"] / max(1, h_stats["games"])
            h_opp = h_stats["opp_sum"] / max(1, h_stats["games"])
            a_ppg = a_stats["ppg_sum"] / max(1, a_stats["games"])
            a_opp = a_stats["opp_sum"] / max(1, a_stats["games"])

            update = {
                "home_ppg": round(h_ppg, 2) if h_stats["games"] > 0 else None,
                "away_ppg": round(a_ppg, 2) if a_stats["games"] > 0 else None,
                "home_opp_ppg": round(h_opp, 2) if h_stats["games"] > 0 else None,
                "away_opp_ppg": round(a_opp, 2) if a_stats["games"] > 0 else None,
                "home_adj_em": round(h_ppg - h_opp, 2) if h_stats["games"] >= 3 else None,
                "away_adj_em": round(a_ppg - a_opp, 2) if a_stats["games"] >= 3 else None,
                "home_adj_oe": round(h_ppg, 2) if h_stats["games"] >= 3 else None,
                "away_adj_oe": round(a_ppg, 2) if a_stats["games"] >= 3 else None,
                "home_adj_de": round(h_opp, 2) if h_stats["games"] >= 3 else None,
                "away_adj_de": round(a_opp, 2) if a_stats["games"] >= 3 else None,
                "home_tempo": round((h_ppg + h_opp) / 2, 1) if h_stats["games"] > 0 else 70.0,
                "away_tempo": round((a_ppg + a_opp) / 2, 1) if a_stats["games"] > 0 else 70.0,
                "home_record_wins": h_stats["wins"],
                "away_record_wins": a_stats["wins"],
                "home_record_losses": h_stats["losses"],
                "away_record_losses": a_stats["losses"],
            }
            updates.append((row["id"], update))

            h_stats["ppg_sum"] += h_score
            h_stats["opp_sum"] += a_score
            h_stats["games"] += 1
            h_stats["wins" if h_score > a_score else "losses"] += 1
            team_stats[h_id] = h_stats

            a_stats["ppg_sum"] += a_score
            a_stats["opp_sum"] += h_score
            a_stats["games"] += 1
            a_stats["wins" if a_score > h_score else "losses"] += 1
            team_stats[a_id] = a_stats

        updated = 0
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        for row_id, update in updates:
            try:
                resp = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/ncaa_historical?id=eq.{row_id}",
                    headers=headers, json=update, timeout=10,
                )
                if resp.ok: updated += 1
            except Exception:
                pass
            if updated % 100 == 0:
                _time.sleep(0.1)

        return jsonify({
            "status": "enriched", "season": season,
            "games_processed": len(updates), "games_updated": updated,
            "teams_tracked": len(team_stats),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── Startup ────────────────────────────────────────────────────
@app.before_request
def _once():
    """Models load lazily via load_model() on first use."""
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
