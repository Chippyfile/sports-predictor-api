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
# MLB MODEL
# ═══════════════════════════════════════════════════════════════

def mlb_build_features(df):
    """
    Build feature matrix. Works on both mlb_predictions and mlb_historical rows.
    Uses raw game inputs (wOBA, FIP, park factor) when available — these are
    the real predictive signal. Heuristic outputs (pred_runs) used as fallback only.
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
        # Step 13: SP innings pitched + defensive OAA
        "home_sp_ip":       5.5,    # starter avg IP (league avg ~5.5 in 2024)
        "away_sp_ip":       5.5,
        "home_def_oaa":     0.0,    # team defensive Outs Above Average (Statcast)
        "away_def_oaa":     0.0,
    }
    for col, default in raw_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    # Use home_sp_fip preferentially, fall back to home_fip
    df["home_starter_fip"] = df["home_sp_fip"].where(df["home_sp_fip"] != 4.25, df["home_fip"])
    df["away_starter_fip"] = df["away_sp_fip"].where(df["away_sp_fip"] != 4.25, df["away_fip"])

    # ── Derived features from raw inputs ──
    df["woba_diff"]       = df["home_woba"] - df["away_woba"]
    df["fip_diff"]        = df["home_starter_fip"] - df["away_starter_fip"]  # negative = home ace advantage
    df["bullpen_era_diff"] = df["home_bullpen_era"] - df["away_bullpen_era"]
    df["rest_diff"]       = df["home_rest_days"] - df["away_rest_days"]
    df["travel_diff"]     = df["home_travel"] - df["away_travel"]  # negative = home advantage
    df["is_warm"]         = (df["temp_f"] > 75).astype(int)
    df["is_cold"]         = (df["temp_f"] < 45).astype(int)
    df["wind_out"]        = df["wind_out_flag"].astype(int)

    # Step 13: SP innings & bullpen exposure + defensive OAA
    df["sp_ip_diff"]      = df["home_sp_ip"] - df["away_sp_ip"]
    # Bullpen exposure: low SP IP = more bullpen innings needed.
    # Interaction with bullpen ERA: bad bullpen + short starter = compounding problem.
    df["home_bp_exposure"] = np.maximum(0, 9.0 - df["home_sp_ip"]) * (df["home_bullpen_era"] / 4.10)
    df["away_bp_exposure"] = np.maximum(0, 9.0 - df["away_sp_ip"]) * (df["away_bullpen_era"] / 4.10)
    df["bp_exposure_diff"] = df["home_bp_exposure"] - df["away_bp_exposure"]
    df["def_oaa_diff"]     = df["home_def_oaa"] - df["away_def_oaa"]

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
    df["has_heuristic"]  = (df["total_pred"] > 0).astype(int)  # flag: 0 for historical rows

    feature_cols = [
        # Raw inputs — primary signal
        "home_woba", "away_woba", "woba_diff",
        "home_starter_fip", "away_starter_fip", "fip_diff",
        "home_bullpen_era", "away_bullpen_era", "bullpen_era_diff",
        "park_factor",
        "temp_f", "wind_mph", "wind_out",
        "is_warm", "is_cold",
        "rest_diff", "travel_diff",
        # Step 13: SP workload & defensive quality
        "home_sp_ip", "away_sp_ip", "sp_ip_diff",
        "home_bp_exposure", "away_bp_exposure", "bp_exposure_diff",
        "home_def_oaa", "away_def_oaa", "def_oaa_diff",
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
    cv_folds = min(5, n)

    if n >= 200:
        # ── STACKING ENSEMBLE (Finding #15) ──────────────────────────────
        # Level 0: three diverse base learners for margin regression
        # Level 1: Ridge meta-learner trained on out-of-fold predictions
        # Diversity: GBM (boosted trees) + RF (bagged trees) + Ridge (linear)
        # This captures non-linear interactions (GBM/RF) while Ridge stabilizes.

        gbm = GradientBoostingRegressor(
            n_estimators=300, max_depth=4,
            learning_rate=0.04, subsample=0.8,
            min_samples_leaf=20, random_state=42,
        )
        rf_reg = RandomForestRegressor(
            n_estimators=200, max_depth=6,
            min_samples_leaf=15, max_features=0.7,
            random_state=42, n_jobs=-1,
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

        # Wrap into a single object that has .predict() for compatibility
        class StackedRegressor:
            """Drop-in replacement: bundle['reg'].predict(X) still works."""
            def __init__(self, base_learners, meta, scaler_ref):
                self.base_learners = base_learners  # [gbm, rf, ridge]
                self.meta = meta
            def predict(self, X):
                base_preds = np.column_stack([m.predict(X) for m in self.base_learners])
                return self.meta.predict(base_preds)

        reg = StackedRegressor([gbm, rf_reg, ridge], meta_reg, scaler)
        reg_cv = cross_val_score(gbm, X_scaled, y_margin,
                                  cv=cv_folds, scoring="neg_mean_absolute_error")

        # SHAP: use GBM component (most interpretable tree-based learner)
        explainer = shap.TreeExplainer(gbm)
        model_type = "StackedEnsemble(GBM+RF+Ridge)"

        # ── Stacked classifier for win probability ───────────────────────
        gbm_clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=20, random_state=42,
        )
        rf_clf = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=15, max_features=0.7,
            random_state=42, n_jobs=-1,
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

    # Step 13: SP innings pitched + defensive OAA
    home_sp_ip = float(game.get("home_sp_ip", 5.5))
    away_sp_ip = float(game.get("away_sp_ip", 5.5))
    home_def_oaa = float(game.get("home_def_oaa", 0.0))
    away_def_oaa = float(game.get("away_def_oaa", 0.0))

    # Calculate derived features
    home_bp_exposure = max(0, 9.0 - home_sp_ip) * (home_bullpen / 4.10)
    away_bp_exposure = max(0, 9.0 - away_sp_ip) * (away_bullpen / 4.10)

    row_data = {
        # Raw inputs
        "home_woba": home_woba,
        "away_woba": away_woba,
        "woba_diff": home_woba - away_woba,
        
        "home_starter_fip": home_sp_fip,
        "away_starter_fip": away_sp_fip,
        "fip_diff": home_sp_fip - away_sp_fip,
        
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

        # Step 13: SP workload & defensive quality
        "home_sp_ip": home_sp_ip,
        "away_sp_ip": away_sp_ip,
        "sp_ip_diff": home_sp_ip - away_sp_ip,
        "home_bp_exposure": home_bp_exposure,
        "away_bp_exposure": away_bp_exposure,
        "bp_exposure_diff": home_bp_exposure - away_bp_exposure,
        "home_def_oaa": home_def_oaa,
        "away_def_oaa": away_def_oaa,
        "def_oaa_diff": home_def_oaa - away_def_oaa,
        
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
# NCAAB MODEL
# ═══════════════════════════════════════════════════════════════

def ncaa_build_features(df):
    df = df.copy()
    df["adj_em_diff"]     = df["home_adj_em"].fillna(0) - df["away_adj_em"].fillna(0)
    df["score_diff_pred"] = df["pred_home_score"].fillna(0) - df["pred_away_score"].fillna(0)
    df["total_pred"]      = df["pred_home_score"].fillna(0) + df["pred_away_score"].fillna(0)
    df["home_fav"]        = (df["model_ml_home"] < 0).astype(int)
    df["neutral"]         = df["neutral_site"].fillna(False).astype(int)
    df["spread_vs_market"]= df["spread_home"].fillna(0) - df["market_spread_home"].fillna(0)
    df["ou_gap"]          = df["total_pred"] - df["market_ou_total"].fillna(df["ou_total"].fillna(145))
    df["win_pct_home"]    = df["win_pct_home"].fillna(0.5)

    feature_cols = [
        "pred_home_score", "pred_away_score",
        "home_adj_em", "away_adj_em",
        "adj_em_diff", "score_diff_pred",
        "total_pred", "home_fav",
        "win_pct_home", "neutral",
        "ou_gap", "spread_vs_market",
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
    save_model("ncaa", bundle)
    return {"status": "trained", "n_train": len(df),
            "mae_cv": round(float(-reg_cv.mean()), 3), "features": list(X.columns)}

def predict_ncaa(game: dict):
    bundle = load_model("ncaa")
    if not bundle:
        return {"error": "NCAAB model not trained — call /train/ncaa first"}

    ph = game.get("pred_home_score", 72)
    pa = game.get("pred_away_score", 72)
    he = game.get("home_adj_em", 0)
    ae = game.get("away_adj_em", 0)

    row = pd.DataFrame([{
        "pred_home_score":  ph, "pred_away_score":  pa,
        "home_adj_em":      he, "away_adj_em":      ae,
        "adj_em_diff":      he - ae,
        "score_diff_pred":  ph - pa, "total_pred": ph + pa,
        "home_fav":         1 if game.get("model_ml_home", 0) < 0 else 0,
        "win_pct_home":     game.get("win_pct_home", 0.5),
        "neutral":          int(game.get("neutral_site", False)),
        "ou_gap":           (ph + pa) - game.get("market_ou_total", game.get("ou_total", 145)),
        "spread_vs_market": game.get("spread_home", 0) - game.get("market_spread_home", 0),
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
        "sport": "NCAAB",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out,
        "model_meta": {"n_train": bundle["n_train"], "mae_cv": bundle["mae_cv"],
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
        std = {"NBA": 11.0, "NCAAB": 9.0, "NFL": 10.5, "NCAAF": 14.0}.get(sport, 10.0)
        home_scores = rng.normal(home_mean, std, n_sims)
        away_scores = rng.normal(away_mean, std, n_sims)
        distribution_note = f"Normal(σ={std})"

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
    game = request.get_json() or {}
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
    body      = request.get_json() or {}
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

# ── Startup ────────────────────────────────────────────────────
@app.before_request
def _once():
    """Models load lazily via load_model() on first use."""
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
