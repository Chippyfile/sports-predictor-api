"""
Multi-Sport Predictor API
Scikit-learn + SHAP backend for Railway deployment
Connects to existing Supabase tables
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
import shap
import joblib

app = Flask(__name__)
CORS(app)  # Allow requests from your React app

# ── Supabase config ───────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Supabase helper ───────────────────────────────────────────────────────────
def sb_get(table, params=""):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    url = f"{SUPABASE_URL}/rest/v1/{table}?{params}"
    r = requests.get(url, headers=headers, timeout=15)
    return r.json() if r.ok else []

# ── Mongo-style model cache ───────────────────────────────────────────────────
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
# MLB MODEL
# ═══════════════════════════════════════════════════════════════

def mlb_build_features(df):
    """
    Extract feature matrix from mlb_predictions rows that have actuals.
    Features we can derive from what's already in the table:
      - pred_home_runs, pred_away_runs  (model's own run estimates)
      - win_pct_home                    (model's win probability)
      - run_line_home                   (always -1.5 in your data)
      - ou_total                        (over/under line)
      - run_diff_pred                   (pred_home - pred_away)
      - total_pred                      (pred_home + pred_away)
      - home_fav                        (1 if home is favorite)
      - model_ml_home                   (moneyline)
    Target: actual_home_runs - actual_away_runs (margin)
    """
    df = df.copy()
    df["run_diff_pred"] = df["pred_home_runs"] - df["pred_away_runs"]
    df["total_pred"]    = df["pred_home_runs"] + df["pred_away_runs"]
    df["home_fav"]      = (df["model_ml_home"] < 0).astype(int)
    df["ou_gap"]        = df["total_pred"] - df["ou_total"]
    df["win_pct_home"]  = df["win_pct_home"].fillna(0.5)

    feature_cols = [
        "pred_home_runs", "pred_away_runs",
        "win_pct_home", "ou_total",
        "run_diff_pred", "total_pred",
        "home_fav", "ou_gap",
    ]
    return df[feature_cols].fillna(0)

def train_mlb():
    rows = sb_get("mlb_predictions",
                  "result_entered=eq.true&actual_home_runs=not.is.null&select=*")
    if len(rows) < 10:
        return {"error": "Not enough MLB data to train (need 10+ completed games)", "n": len(rows)}

    df = pd.DataFrame(rows)
    X  = mlb_build_features(df)
    y_margin = df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)
    y_win    = (y_margin > 0).astype(int)

    # Margin regressor (Ridge with cross-validated alpha)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=min(5, len(df)))
    reg.fit(X_scaled, y_margin)
    reg_cv  = cross_val_score(reg, X_scaled, y_margin, cv=min(5, len(df)),
                               scoring="neg_mean_absolute_error")

    # Win probability classifier (calibrated logistic)
    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), cv=min(5, len(df))
    )
    clf.fit(X_scaled, y_win)

    # SHAP explainer (linear — fast, exact)
    explainer = shap.LinearExplainer(reg, X_scaled, feature_perturbation="interventional")

    bundle = {
        "scaler": scaler,
        "reg": reg,
        "clf": clf,
        "explainer": explainer,
        "feature_cols": list(X.columns),
        "n_train": len(df),
        "mae_cv": float(-reg_cv.mean()),
        "trained_at": datetime.utcnow().isoformat(),
        "alpha": float(reg.alpha_),
    }
    save_model("mlb", bundle)
    return {
        "status": "trained",
        "n_train": len(df),
        "mae_cv": round(float(-reg_cv.mean()), 3),
        "alpha": float(reg.alpha_),
        "features": list(X.columns),
    }

def predict_mlb(game: dict):
    bundle = load_model("mlb")
    if not bundle:
        return {"error": "MLB model not trained — call /train/mlb first"}

    row = pd.DataFrame([{
        "pred_home_runs": game.get("pred_home_runs", 4.5),
        "pred_away_runs": game.get("pred_away_runs", 4.5),
        "win_pct_home":   game.get("win_pct_home", 0.5),
        "ou_total":       game.get("ou_total", 9.0),
        "run_diff_pred":  game.get("pred_home_runs", 4.5) - game.get("pred_away_runs", 4.5),
        "total_pred":     game.get("pred_home_runs", 4.5) + game.get("pred_away_runs", 4.5),
        "home_fav":       1 if game.get("model_ml_home", 0) < 0 else 0,
        "ou_gap":         (game.get("pred_home_runs", 4.5) + game.get("pred_away_runs", 4.5)) - game.get("ou_total", 9.0),
    }])

    X_s = bundle["scaler"].transform(row[bundle["feature_cols"]])
    margin = float(bundle["reg"].predict(X_s)[0])
    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])

    # SHAP explanation
    shap_vals = bundle["explainer"].shap_values(X_s)[0]
    shap_out  = [
        {"feature": f, "shap": round(float(v), 4), "value": round(float(row[f].iloc[0]), 3)}
        for f, v in zip(bundle["feature_cols"], shap_vals)
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "MLB",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out,
        "model_meta": {
            "n_train": bundle["n_train"],
            "mae_cv": bundle["mae_cv"],
            "trained_at": bundle["trained_at"],
        }
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

    # Gradient Boosting captures non-linear rest/travel interactions better
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.1, random_state=42)
    reg.fit(X_scaled, y_margin)
    reg_cv = cross_val_score(reg, X_scaled, y_margin,
                              cv=min(5, len(df)), scoring="neg_mean_absolute_error")

    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), cv=min(5, len(df))
    )
    clf.fit(X_scaled, y_win)

    # SHAP tree explainer for GBM
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

    X_s    = bundle["scaler"].transform(row[bundle["feature_cols"]])
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
                       "trained_at": bundle["trained_at"]}
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

    X_s    = bundle["scaler"].transform(row[bundle["feature_cols"]])
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
                       "trained_at": bundle["trained_at"]}
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

    X_s    = bundle["scaler"].transform(row[bundle["feature_cols"]])
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
                       "trained_at": bundle["trained_at"]}
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

    X_s    = bundle["scaler"].transform(row[bundle["feature_cols"]])
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
                       "trained_at": bundle["trained_at"]}
    }

# ═══════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════

def monte_carlo(sport, home_mean, away_mean, n_sims=10_000):
    """
    Run score simulations and return outcome distribution.
    MLB: Poisson (goal-count distribution)
    Others: Normal (continuous score)
    """
    rng = np.random.default_rng(42)

    if sport == "MLB":
        home_scores = rng.poisson(max(home_mean, 0.5), n_sims).astype(float)
        away_scores = rng.poisson(max(away_mean, 0.5), n_sims).astype(float)
    else:
        std = {"NBA": 11.0, "NCAAB": 9.0, "NFL": 10.5, "NCAAF": 14.0}.get(sport, 10.0)
        home_scores = rng.normal(home_mean, std, n_sims)
        away_scores = rng.normal(away_mean, std, n_sims)

    margins = home_scores - away_scores
    totals  = home_scores + away_scores

    return {
        "n_sims": n_sims,
        "home_win_pct": round(float((margins > 0).mean()), 4),
        "away_win_pct": round(float((margins < 0).mean()), 4),
        "push_pct":     round(float((margins == 0).mean()), 4),
        "avg_margin":   round(float(margins.mean()), 2),
        "avg_total":    round(float(totals.mean()), 2),
        "margin_percentiles": {
            "p10": round(float(np.percentile(margins, 10)), 1),
            "p25": round(float(np.percentile(margins, 25)), 1),
            "p50": round(float(np.percentile(margins, 50)), 1),
            "p75": round(float(np.percentile(margins, 75)), 1),
            "p90": round(float(np.percentile(margins, 90)), 1),
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
                  "result_entered=eq.true&ml_correct=not.is.null&select=ml_correct,rl_correct,ou_correct,win_pct_home")
    if not rows:
        return {"error": f"No completed {sport_label} games found"}

    df = pd.DataFrame(rows)
    ml_acc  = df["ml_correct"].mean() if "ml_correct" in df else None
    rl_acc  = df["rl_correct"].mean() if "rl_correct" in df else None
    ou_df   = df[df["ou_correct"].notna()] if "ou_correct" in df else pd.DataFrame()
    ou_acc  = (ou_df["ou_correct"].isin(["OVER","UNDER"])).mean() if len(ou_df) > 0 else None

    # Brier score on win probability (calibration)
    brier = None
    if "win_pct_home" in df and "ml_correct" in df:
        sub = df[df["win_pct_home"].notna() & df["ml_correct"].notna()]
        if len(sub) > 5:
            brier = round(float(brier_score_loss(
                sub["ml_correct"].astype(int),
                sub["win_pct_home"].astype(float)
            )), 4)

    return {
        "sport": sport_label,
        "n_games": len(df),
        "ml_accuracy": round(float(ml_acc), 4) if ml_acc is not None else None,
        "rl_accuracy": round(float(rl_acc), 4) if rl_acc is not None else None,
        "ou_accuracy": round(float(ou_acc), 4) if ou_acc is not None else None,
        "brier_score": brier,
        "brier_note": "Lower is better. Perfect calibration = 0.25 for 50/50 games.",
    }

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({"status": "ok", "service": "Multi-Sport Predictor API",
                    "endpoints": [
                        "GET  /health",
                        "POST /train/<sport>     (mlb|nba|ncaa|nfl|ncaaf)",
                        "POST /train/all",
                        "POST /predict/<sport>",
                        "POST /monte-carlo",
                        "GET  /accuracy/<sport>",
                        "GET  /accuracy/all",
                        "GET  /model-info/<sport>",
                    ]})

@app.route("/health")
def health():
    trained = [s for s in ["mlb","nba","ncaa","nfl","ncaaf"] if load_model(s)]
    return jsonify({"status": "healthy", "trained_models": trained,
                    "timestamp": datetime.utcnow().isoformat()})

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

# ── Predict endpoints ──────────────────────────────────────────
@app.route("/predict/<sport>", methods=["POST"])
def route_predict(sport):
    game = request.get_json() or {}
    fns  = {"mlb": predict_mlb, "nba": predict_nba, "ncaa": predict_ncaa,
            "nfl": predict_nfl, "ncaaf": predict_ncaaf}
    fn = fns.get(sport.lower())
    if not fn:
        return jsonify({"error": f"Unknown sport: {sport}"}), 400
    return jsonify(fn(game))

# ── Monte Carlo ────────────────────────────────────────────────
@app.route("/monte-carlo", methods=["POST"])
def route_monte_carlo():
    body = request.get_json() or {}
    sport      = body.get("sport", "NBA").upper()
    home_mean  = float(body.get("home_mean", 110))
    away_mean  = float(body.get("away_mean", 110))
    n_sims     = min(int(body.get("n_sims", 10000)), 100_000)
    return jsonify(monte_carlo(sport, home_mean, away_mean, n_sims))

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
    return jsonify({
        "sport": sport.upper(),
        "n_train": bundle.get("n_train"),
        "mae_cv": bundle.get("mae_cv"),
        "trained_at": bundle.get("trained_at"),
        "features": bundle.get("feature_cols"),
        "alpha": bundle.get("alpha"),
    })

# ── Startup auto-train ─────────────────────────────────────────
@app.before_request
def _once():
    """Auto-load any previously saved models on first request."""
    pass  # joblib load happens lazily via load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
