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
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ────────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
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
    2025: {"lg_woba": 0.317, "woba_scale": 1.25, "lg_rpg": 4.38, "lg_fip": 4.17, "pa_pg": 37.8},
}
DEFAULT_CONSTANTS = {"lg_woba": 0.317, "woba_scale": 1.25, "lg_rpg": 4.38, "lg_fip": 4.17, "pa_pg": 37.8}

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

    # ── League run environment context ──
    if "season" in df.columns:
        df["lg_rpg"] = df["season"].map(
            lambda s: SEASON_CONSTANTS.get(int(s), DEFAULT_CONSTANTS)["lg_rpg"]
            if pd.notna(s) else DEFAULT_CONSTANTS["lg_rpg"]
        )
    else:
        df["lg_rpg"] = DEFAULT_CONSTANTS["lg_rpg"]

    # ── Heuristic outputs (only from mlb_predictions rows, 0 for historical) ──
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

    feature_cols = [
        # Raw inputs — primary signal
        "home_woba", "away_woba", "woba_diff",
        "home_starter_fip", "away_starter_fip", "fip_diff",
        "has_sp_fip",
        "home_bullpen_era", "away_bullpen_era", "bullpen_era_diff",
        "park_factor",
        "temp_f", "wind_mph", "wind_out",
        "is_warm", "is_cold",
        "rest_diff", "travel_diff",
        # K/9 and BB/9 features
        "home_k9", "away_k9", "k9_diff",
        "home_bb9", "away_bb9", "bb9_diff",
        "k_bb_home", "k_bb_away", "k_bb_diff",
        # SP workload & defensive quality
        "home_sp_ip", "away_sp_ip", "sp_ip_diff",
        "home_bp_exposure", "away_bp_exposure", "bp_exposure_diff",
        "home_def_oaa", "away_def_oaa", "def_oaa_diff",
        # League era context
        "lg_rpg",
        # Heuristic outputs — secondary signal (0 for historical rows)
        "pred_home_runs", "pred_away_runs",
        "run_diff_pred", "total_pred",
        "win_pct_home", "home_fav", "ou_gap",
        "has_heuristic",
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

    # Historical rows have no heuristic predictions — zero them out explicitly
    # so has_heuristic=0 correctly flags them in the feature matrix.
    #
    # CRITICAL FIX (Finding #10): win_pct_home was previously set to home_win
    # (1.0 if home won, 0.0 if lost) — this leaked the target variable directly
    # into the feature matrix. Historical rows had no pre-game prediction, so
    # the correct neutral value is 0.5 (no information).
    hist_df["pred_home_runs"] = 0.0
    hist_df["pred_away_runs"] = 0.0
    hist_df["win_pct_home"]   = 0.5   # was: hist_df["home_win"].fillna(0.5) — DATA LEAK
    hist_df["ou_total"]       = 0.0
    hist_df["model_ml_home"]  = 0

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
    # Current season predictions with results (may be empty early in season)
    rows = sb_get("mlb_predictions",
                  "result_entered=eq.true&actual_home_runs=not.is.null&game_type=eq.R&select=*")
    current_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Merge with historical corpus — this is the primary training data
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

    # Use sample weights if available (recency weighting)
    fit_weights = sample_weights if sample_weights is not None else np.ones(len(df))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose model based on data volume
    n = len(df)
    cv_folds = min(3, n)  # 3-fold CV (was 5) — faster, still robust at 5k+ rows

    if n >= 200:
        # ── STACKING ENSEMBLE ──────────────────────────────────────────
        # Reduced complexity for Railway deployment constraints.
        # 3-fold CV, fewer estimators — minimal accuracy loss vs 5-fold/300 trees.

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
        ridge = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=cv_folds)

        # Generate out-of-fold predictions for stacking
        print("  Training stacking ensemble (GBM + RF + Ridge)...")
        oof_gbm = cross_val_predict(gbm, X_scaled, y_margin, cv=cv_folds)
        oof_rf  = cross_val_predict(rf_reg, X_scaled, y_margin, cv=cv_folds)
        oof_ridge = cross_val_predict(ridge, X_scaled, y_margin, cv=cv_folds)

        # Fit all base learners on full data
        gbm.fit(X_scaled, y_margin, sample_weight=fit_weights)
        rf_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
        ridge.fit(X_scaled, y_margin, sample_weight=fit_weights)

        # Level 1: meta-learner on stacked OOF predictions
        meta_X = np.column_stack([oof_gbm, oof_rf, oof_ridge])
        meta_reg = Ridge(alpha=1.0)
        meta_reg.fit(meta_X, y_margin)

        # Use module-level StackedRegressor for pickle compatibility
        reg = StackedRegressor([gbm, rf_reg, ridge], meta_reg, scaler)
        reg_cv = cross_val_score(gbm, X_scaled, y_margin,
                                  cv=cv_folds, scoring="neg_mean_absolute_error")

        # SHAP: use GBM component (most interpretable tree-based learner)
        explainer = shap.TreeExplainer(gbm)
        model_type = "StackedEnsemble(GBM+RF+Ridge)"

        # ── Stacked classifier for win probability ───────────────────────
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
        lr_clf = LogisticRegression(max_iter=1000)

        # OOF probabilities for classifier stacking
        oof_gbm_p = cross_val_predict(gbm_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
        oof_rf_p  = cross_val_predict(rf_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
        oof_lr_p  = cross_val_predict(lr_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]

        # Fit base classifiers on full data
        gbm_clf.fit(X_scaled, y_win, sample_weight=fit_weights)
        rf_clf.fit(X_scaled, y_win, sample_weight=fit_weights)
        lr_clf.fit(X_scaled, y_win, sample_weight=fit_weights)

        # Meta-classifier on stacked OOF probabilities
        meta_clf_X = np.column_stack([oof_gbm_p, oof_rf_p, oof_lr_p])
        meta_lr = LogisticRegression(max_iter=1000)
        meta_lr.fit(meta_clf_X, y_win)

        # Use module-level StackedClassifier for pickle compatibility
        # The stacked meta-learner (LogisticRegression) already produces well-calibrated
        # probabilities via Platt scaling. No additional CalibratedClassifierCV needed.
        # (cv="prefit" was removed in sklearn 1.4+)
        clf = StackedClassifier([gbm_clf, rf_clf, lr_clf], meta_lr)

        print(f"  Stacking meta weights (reg): {meta_reg.coef_.round(3)}")

    else:
        reg = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=cv_folds)
        reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
        reg_cv = cross_val_score(reg, X_scaled, y_margin,
                                  cv=cv_folds, scoring="neg_mean_absolute_error")
        explainer = shap.LinearExplainer(reg, X_scaled, feature_perturbation="interventional")
        model_type = "Ridge"

        # Simple classifier for small data
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000),
            cv=cv_folds
        )
        clf.fit(X_scaled, y_win, sample_weight=fit_weights)

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
    }
    save_model("mlb", bundle)

    # Auto-calibrate dispersion whenever we retrain
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
        "upgrade_note": f"Using {'StackedEnsemble (GBM+RF+Ridge)' if 'Stacked' in model_type else 'Ridge (linear) — will auto-upgrade to stacking ensemble at 200+ training rows'}",
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

    row_data = {
        # Raw inputs
        "home_woba": home_woba,
        "away_woba": away_woba,
        "woba_diff": home_woba - away_woba,
        
        "home_starter_fip": home_starter_fip,
        "away_starter_fip": away_starter_fip,
        "fip_diff": home_starter_fip - away_starter_fip,
        "has_sp_fip": has_sp_fip,
        
        "home_bullpen_era": home_bullpen,
        "away_bullpen_era": away_bullpen,
        "bullpen_era_diff": home_bullpen - away_bullpen,
        
        "park_factor": park_factor,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_out": int(wind_out_flag),
        "is_warm": 1 if temp_f > 75 else 0,
        "is_cold": 1 if temp_f < 45 else 0,
        
        "rest_diff": home_rest - away_rest,
        "travel_diff": home_travel - away_travel,

        # K/9 and BB/9 features
        "home_k9": home_k9,
        "away_k9": away_k9,
        "k9_diff": home_k9 - away_k9,
        "home_bb9": home_bb9,
        "away_bb9": away_bb9,
        "bb9_diff": home_bb9 - away_bb9,
        "k_bb_home": home_k9 - home_bb9,
        "k_bb_away": away_k9 - away_bb9,
        "k_bb_diff": (home_k9 - home_bb9) - (away_k9 - away_bb9),

        # SP workload & defensive quality
        "home_sp_ip": home_sp_ip,
        "away_sp_ip": away_sp_ip,
        "sp_ip_diff": home_sp_ip - away_sp_ip,
        "home_bp_exposure": home_bp_exposure,
        "away_bp_exposure": away_bp_exposure,
        "bp_exposure_diff": home_bp_exposure - away_bp_exposure,
        "home_def_oaa": home_def_oaa,
        "away_def_oaa": away_def_oaa,
        "def_oaa_diff": home_def_oaa - away_def_oaa,

        # League run environment context
        "lg_rpg": DEFAULT_CONSTANTS["lg_rpg"],
        # Heuristic outputs
        "pred_home_runs": ph,
        "pred_away_runs": pa,
        "run_diff_pred": ph - pa,
        "total_pred": ph + pa,
        "win_pct_home": float(game.get("win_pct_home", 0.5)),
        "home_fav": 1 if game.get("model_ml_home", 0) < 0 else 0,
        "ou_gap": (ph + pa) - float(game.get("ou_total", 9.0)),
        "has_heuristic": 1 if ph > 0 or pa > 0 else 0,
    }

    # Create DataFrame with only the features the model expects
    row = pd.DataFrame([{k: row_data[k] for k in bundle["feature_cols"]}])
    
    # Scale and predict
    X_s = bundle["scaler"].transform(row[bundle["feature_cols"]])
    margin = float(bundle["reg"].predict(X_s)[0])
    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])

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
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out[:10],  # Top 10 SHAP values
        "model_meta": {
            "n_train": bundle["n_train"],
            "n_historical": bundle.get("n_historical", 0),
            "n_current": bundle.get("n_current", 0),
            "mae_cv": bundle["mae_cv"],
            "trained_at": bundle["trained_at"],
            "model_type": bundle["model_type"],
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
# NCAAB MODEL (v16 — Audit fixes F3, F4, F11, F17, F18, F19, F23, F24)
# ═══════════════════════════════════════════════════════════════

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
        # F23: Rest days (if available in data)
        "home_rest_days": 3, "away_rest_days": 3,
    }
    for col, default in raw_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # ── F4 FIX: Heuristic-derived features kept SEPARATE from raw ──
    # Only include adj_em_diff and neutral as minimal heuristic context;
    # REMOVED: pred_home_score, pred_away_score, score_diff_pred, total_pred,
    #          home_fav, win_pct_home (F3), spread_vs_market, ou_gap
    df["adj_em_diff"]     = df["home_adj_em"].fillna(0) - df["away_adj_em"].fillna(0)
    df["neutral"]         = df["neutral_site"].fillna(False).astype(int)

    # ── F17 FIX: Differential features ONLY (removed redundant absolute stats) ──
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
    df["rank_diff"]      = df["away_rank"] - df["home_rank"]  # positive = home ranked higher
    df["win_pct_diff"]   = (df["home_wins"] / (df["home_wins"] + df["home_losses"]).clip(1)) - \
                           (df["away_wins"] / (df["away_wins"] + df["away_losses"]).clip(1))

    # F11: Turnover margin differential — one of strongest NCAAB predictors
    df["to_margin_diff"]    = df["away_turnovers"] - df["home_turnovers"]
    df["steals_to_ratio_h"] = df["home_steals"] / df["home_turnovers"].clip(0.5)
    df["steals_to_ratio_a"] = df["away_steals"] / df["away_turnovers"].clip(0.5)
    df["steals_to_diff"]    = df["steals_to_ratio_h"] - df["steals_to_ratio_a"]

    # F24: Conference game flag and ranking context
    df["is_ranked_game"] = ((df["home_rank"] <= 25) | (df["away_rank"] <= 25)).astype(int)
    df["is_top_matchup"] = ((df["home_rank"] <= 25) & (df["away_rank"] <= 25)).astype(int)

    # F23: Rest days differential (if available)
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["either_b2b"] = ((df["home_rest_days"] <= 1) | (df["away_rest_days"] <= 1)).astype(int)

    feature_cols = [
        # Minimal heuristic context (F4: limited to avoid circularity)
        "adj_em_diff", "neutral",
        # Raw stats — differentials ONLY (F17: removed absolute stats)
        "ppg_diff", "opp_ppg_diff", "fgpct_diff", "threepct_diff",
        "orb_pct_diff", "fta_rate_diff", "ato_diff",
        "def_fgpct_diff", "steals_diff", "blocks_diff",
        "sos_diff", "form_diff", "rank_diff", "win_pct_diff",
        # F11: Turnover quality features
        "to_margin_diff", "steals_to_diff",
        # Context features
        "tempo_avg", "is_ranked_game", "is_top_matchup",
        # F23: Schedule fatigue
        "rest_diff", "either_b2b",
    ]
    return df[feature_cols].fillna(0)

def train_ncaa():
    rows = sb_get("ncaa_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    if len(rows) < 10:
        return {"error": "Not enough NCAAB data", "n": len(rows)}

    df = pd.DataFrame(rows)
    X  = ncaa_build_features(df)
    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    y_win    = (y_margin > 0).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(df)
    cv_folds = min(5, n)

    if n >= 200:
        # ── STACKING ENSEMBLE (F18: tuned meta-learner) ──────────
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
        # F18 FIX: Use RidgeCV with broader alpha range for meta-learner
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 5.0, 10.0, 50.0], cv=cv_folds)

        print("  NCAAB: Training stacking ensemble (GBM + RF + Ridge)...")
        oof_gbm = cross_val_predict(gbm, X_scaled, y_margin, cv=cv_folds)
        oof_rf  = cross_val_predict(rf_reg, X_scaled, y_margin, cv=cv_folds)
        oof_ridge = cross_val_predict(ridge, X_scaled, y_margin, cv=cv_folds)

        gbm.fit(X_scaled, y_margin)
        rf_reg.fit(X_scaled, y_margin)
        ridge.fit(X_scaled, y_margin)

        meta_X = np.column_stack([oof_gbm, oof_rf, oof_ridge])
        # F18 FIX: Use RidgeCV for meta-learner instead of fixed alpha
        meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        meta_reg.fit(meta_X, y_margin)

        reg = StackedRegressor([gbm, rf_reg, ridge], meta_reg, scaler)
        reg_cv = cross_val_score(gbm, X_scaled, y_margin,
                                  cv=cv_folds, scoring="neg_mean_absolute_error")
        explainer = shap.TreeExplainer(gbm)
        model_type = "StackedEnsemble(GBM+RF+Ridge)"

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
        # F19 FIX: Use LogisticRegression without balanced class_weight
        # (home win rate ~60% is mild, balanced weights hurt calibration)
        lr_clf = LogisticRegression(max_iter=1000, C=1.0)

        oof_gbm_p = cross_val_predict(gbm_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
        oof_rf_p  = cross_val_predict(rf_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]
        oof_lr_p  = cross_val_predict(lr_clf, X_scaled, y_win, cv=cv_folds, method="predict_proba")[:, 1]

        gbm_clf.fit(X_scaled, y_win)
        rf_clf.fit(X_scaled, y_win)
        lr_clf.fit(X_scaled, y_win)

        meta_clf_X = np.column_stack([oof_gbm_p, oof_rf_p, oof_lr_p])
        meta_lr = LogisticRegression(max_iter=1000, C=1.0)
        meta_lr.fit(meta_clf_X, y_win)
        clf = StackedClassifier([gbm_clf, rf_clf, lr_clf], meta_lr)

        print(f"  NCAAB stacking meta weights: {meta_reg.coef_.round(3)}")
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

    bundle = {
        "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
        "feature_cols": list(X.columns), "n_train": len(df),
        "mae_cv": float(-reg_cv.mean()), "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(),
    }
    save_model("ncaa", bundle)
    return {"status": "trained", "n_train": len(df), "model_type": model_type,
            "mae_cv": round(float(-reg_cv.mean()), 3), "features": list(X.columns)}

def predict_ncaa(game: dict):
    bundle = load_model("ncaa")
    if not bundle:
        return {"error": "NCAAB model not trained — call /train/ncaa first"}

    ph = game.get("pred_home_score", 72)
    pa = game.get("pred_away_score", 72)
    he = game.get("home_adj_em", 0)
    ae = game.get("away_adj_em", 0)

    # Build a single-row DataFrame with all features the model expects
    # F3/F4 FIX: Removed win_pct_home and most heuristic outputs
    row_data = {
        "home_adj_em": he, "away_adj_em": ae,
        "neutral_site": game.get("neutral_site", False),
        # F4: Keep pred scores for backward compat but they're no longer features
        "pred_home_score": ph, "pred_away_score": pa,
        "model_ml_home": game.get("model_ml_home", 0),
        "spread_home": game.get("spread_home", 0),
        "market_spread_home": game.get("market_spread_home", 0),
        "market_ou_total": game.get("market_ou_total", game.get("ou_total", 145)),
        "ou_total": game.get("ou_total", 145),
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
        # F23: Rest days
        "home_rest_days": game.get("home_rest_days", 3), "away_rest_days": game.get("away_rest_days", 3),
    }
    row = pd.DataFrame([row_data])
    X_built = ncaa_build_features(row)

    # Ensure feature alignment with trained model
    for col in bundle["feature_cols"]:
        if col not in X_built.columns:
            X_built[col] = 0
    X_built = X_built[bundle["feature_cols"]]

    X_s      = bundle["scaler"].transform(X_built)
    margin   = float(bundle["reg"].predict(X_s)[0])
    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])
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
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
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
                  "&select=ml_correct,rl_correct,ou_correct,win_pct_home")
    if not rows:
        return {"error": f"No completed {sport_label} games found"}

    df = pd.DataFrame(rows)
    ml_acc = df["ml_correct"].mean() if "ml_correct" in df else None
    rl_acc = df["rl_correct"].mean() if "rl_correct" in df else None
    ou_df  = df[df["ou_correct"].notna()] if "ou_correct" in df else pd.DataFrame()
    ou_acc = (ou_df["ou_correct"].isin(["OVER", "UNDER"])).mean() if len(ou_df) > 0 else None

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
            "POST /calibrate/mlb      (fit NegBin dispersion from historical data)",
            "POST /predict/<sport>",
            "POST /monte-carlo         (body: sport, home_mean, away_mean, n_sims, ou_line)",
            "POST /backtest/mlb        (walk-forward MLB backtest)",
            "POST /backtest/ncaa       (walk-forward NCAAB backtest)",
            "GET  /accuracy/<sport>",
            "GET  /accuracy/all",
            "GET  /model-info/<sport>",
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

    # F-03: Regress pitching stats 50% toward league mean to limit forward-looking leak.
    # End-of-season FIP contains information from games that haven't happened yet at
    # prediction time. 50% regression approximates mid-season knowledge.
    LEAK_DISCOUNT = 0.50
    home_fip_adj = lg_fip + (home_fip - lg_fip) * LEAK_DISCOUNT
    away_fip_adj = lg_fip + (away_fip - lg_fip) * LEAK_DISCOUNT
    home_k9_adj  = 8.5 + (home_k9 - 8.5) * LEAK_DISCOUNT
    away_k9_adj  = 8.5 + (away_k9 - 8.5) * LEAK_DISCOUNT
    home_bb9_adj = 3.2 + (home_bb9 - 3.2) * LEAK_DISCOUNT
    away_bb9_adj = 3.2 + (away_bb9 - 3.2) * LEAK_DISCOUNT

    # ── wOBA → Runs (FanGraphs method) ──
    hr = lg_rpg + ((home_woba - lg_woba) / woba_scale) * pa_pg
    ar = lg_rpg + ((away_woba - lg_woba) / woba_scale) * pa_pg

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

@app.route("/backtest/mlb", methods=["POST"])
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
# NCAA BACKTEST (F12 FIX: uses stacking ensemble, adds sigma calibration)
# ═══════════════════════════════════════════════════════════════

@app.route("/backtest/ncaa", methods=["POST"])
def route_backtest_ncaa():
    """
    Walk-forward backtest for NCAAB predictions.
    Trains on games before each month, tests on that month.
    F12 FIX: Uses stacking ensemble when n >= 200 (matches production).
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

            # F12 FIX: Use stacking ensemble when enough training data
            if len(train_df) >= 200:
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
                rdg = RidgeCV(alphas=[0.01, 0.1, 1.0, 5.0, 10.0], cv=cv_folds)

                oof_g = cross_val_predict(gbm, X_tr, y_train_margin, cv=cv_folds)
                oof_r = cross_val_predict(rf_r, X_tr, y_train_margin, cv=cv_folds)
                oof_d = cross_val_predict(rdg, X_tr, y_train_margin, cv=cv_folds)

                gbm.fit(X_tr, y_train_margin)
                rf_r.fit(X_tr, y_train_margin)
                rdg.fit(X_tr, y_train_margin)

                meta_X = np.column_stack([oof_g, oof_r, oof_d])
                meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                meta.fit(meta_X, y_train_margin)

                test_meta_X = np.column_stack([
                    gbm.predict(X_te), rf_r.predict(X_te), rdg.predict(X_te)
                ])
                pred_margin = meta.predict(test_meta_X)

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

                test_clf_X = np.column_stack([
                    gbm_c.predict_proba(X_te)[:, 1],
                    rf_c.predict_proba(X_te)[:, 1],
                    lr_c.predict_proba(X_te)[:, 1],
                ])
                meta_clf = LogisticRegression(max_iter=1000, C=1.0)
                meta_clf.fit(np.column_stack([oof_gc, oof_rc, oof_lc]), y_train_win)
                pred_wp = meta_clf.predict_proba(test_clf_X)[:, 1]
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


# ── Startup ────────────────────────────────────────────────────
@app.before_request
def _once():
    """Models load lazily via load_model() on first use."""
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

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
