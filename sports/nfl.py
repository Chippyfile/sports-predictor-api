import numpy as np, pandas as pd, shap
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from db import sb_get, save_model, load_model

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
