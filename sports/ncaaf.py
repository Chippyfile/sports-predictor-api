import numpy as np, pandas as pd, shap
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from db import sb_get, save_model, load_model

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
