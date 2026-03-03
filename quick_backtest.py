"""
Lightweight backtests — uses already-trained models for inference only.
No retraining, runs in seconds.
"""
import numpy as np
import pandas as pd
from db import sb_get, load_model
from sklearn.metrics import mean_absolute_error, brier_score_loss


def quick_backtest_nba():
    """Test current NBA model against graded predictions."""
    bundle = load_model("nba")
    if not bundle:
        return {"error": "NBA model not trained"}

    rows = sb_get("nba_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    if not rows or len(rows) < 20:
        return {"error": "Not enough graded NBA games", "n": len(rows) if rows else 0}

    df = pd.DataFrame(rows)
    from sports.nba import nba_build_features
    X = nba_build_features(df)

    scaler = bundle.get("scaler")
    reg = bundle.get("reg")
    clf = bundle.get("clf")
    isotonic = bundle.get("isotonic")

    if not scaler or not reg:
        return {"error": "Model bundle incomplete"}

    X_scaled = scaler.transform(X)
    pred_margin = reg.predict(X_scaled)

    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    y_win = (y_margin > 0).astype(int)

    # Win predictions
    if clf:
        pred_wp = clf.predict_proba(X_scaled)[:, 1]
        if isotonic:
            pred_wp = isotonic.predict(pred_wp)
        pred_wp = np.clip(pred_wp, 0.05, 0.95)
    else:
        pred_wp = 1 / (1 + np.exp(-pred_margin / 6.0))

    pred_pick = (pred_wp >= 0.5).astype(int)
    accuracy = float((pred_pick == y_win).mean())
    mae = float(mean_absolute_error(y_margin, pred_margin))
    brier = float(brier_score_loss(y_win, pred_wp))

    # Confidence tiers
    tiers = []
    for thresh in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        strong = (pred_wp >= thresh) | (pred_wp <= (1 - thresh))
        n_strong = int(strong.sum())
        if n_strong >= 10:
            tier_acc = float((pred_pick[strong] == y_win.values[strong]).mean())
            tiers.append({"threshold": thresh, "n": n_strong, "accuracy": round(tier_acc, 4)})

    return {
        "sport": "NBA",
        "model_type": bundle.get("model_type", ""),
        "n_test": len(df),
        "ml_accuracy": round(accuracy, 4),
        "mae_margin": round(mae, 3),
        "brier_score": round(brier, 4),
        "home_win_rate": round(float(y_win.mean()), 3),
        "confidence_tiers": tiers,
    }


def quick_backtest_ncaa():
    """Test current NCAA model against graded predictions."""
    bundle = load_model("ncaa")
    if not bundle:
        return {"error": "NCAA model not trained"}

    rows = sb_get("ncaa_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")
    if not rows or len(rows) < 20:
        return {"error": "Not enough graded NCAA games", "n": len(rows) if rows else 0}

    df = pd.DataFrame(rows)
    from sports.ncaa import ncaa_build_features
    X = ncaa_build_features(df)

    scaler = bundle.get("scaler")
    reg = bundle.get("reg")
    clf = bundle.get("clf")
    isotonic = bundle.get("isotonic")

    if not scaler or not reg:
        return {"error": "Model bundle incomplete"}

    X_scaled = scaler.transform(X)
    pred_margin = reg.predict(X_scaled)

    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    y_win = (y_margin > 0).astype(int)

    if clf:
        pred_wp = clf.predict_proba(X_scaled)[:, 1]
        if isotonic:
            pred_wp = isotonic.predict(pred_wp)
        pred_wp = np.clip(pred_wp, 0.05, 0.95)
    else:
        pred_wp = 1 / (1 + np.exp(-pred_margin / 8.0))

    pred_pick = (pred_wp >= 0.5).astype(int)
    accuracy = float((pred_pick == y_win).mean())
    mae = float(mean_absolute_error(y_margin, pred_margin))
    brier = float(brier_score_loss(y_win, pred_wp))

    tiers = []
    for thresh in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        strong = (pred_wp >= thresh) | (pred_wp <= (1 - thresh))
        n_strong = int(strong.sum())
        if n_strong >= 10:
            tier_acc = float((pred_pick[strong] == y_win.values[strong]).mean())
            tiers.append({"threshold": thresh, "n": n_strong, "accuracy": round(tier_acc, 4)})

    return {
        "sport": "NCAAB",
        "model_type": bundle.get("model_type", ""),
        "n_test": len(df),
        "ml_accuracy": round(accuracy, 4),
        "mae_margin": round(mae, 3),
        "brier_score": round(brier, 4),
        "home_win_rate": round(float(y_win.mean()), 3),
        "confidence_tiers": tiers,
    }


def quick_backtest_mlb():
    """Test current MLB model against graded predictions."""
    bundle = load_model("mlb")
    if not bundle:
        return {"error": "MLB model not trained"}

    rows = sb_get("mlb_predictions",
                  "result_entered=eq.true&actual_home_runs=not.is.null&select=*")
    if not rows or len(rows) < 20:
        return {"error": "Not enough graded MLB games", "n": len(rows) if rows else 0}

    df = pd.DataFrame(rows)
    from sports.mlb import mlb_build_features
    X = mlb_build_features(df)

    scaler = bundle.get("scaler")
    reg = bundle.get("reg")
    clf = bundle.get("clf")
    isotonic = bundle.get("isotonic")

    if not scaler or not reg:
        return {"error": "Model bundle incomplete"}

    feature_cols = bundle.get("feature_cols", [])
    # Align features
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    X_scaled = scaler.transform(X)
    pred_margin = reg.predict(X_scaled)

    y_margin = df["actual_home_runs"].astype(float) - df["actual_away_runs"].astype(float)
    y_win = (y_margin > 0).astype(int)

    if clf:
        pred_wp = clf.predict_proba(X_scaled)[:, 1]
        if isotonic:
            pred_wp = isotonic.predict(pred_wp)
        pred_wp = np.clip(pred_wp, 0.15, 0.85)
    else:
        pred_wp = 1 / (1 + np.exp(-pred_margin / 3.0))

    pred_pick = (pred_wp >= 0.5).astype(int)
    accuracy = float((pred_pick == y_win).mean())
    mae = float(mean_absolute_error(y_margin, pred_margin))
    brier = float(brier_score_loss(y_win, pred_wp))

    tiers = []
    for thresh in [0.52, 0.55, 0.58, 0.60, 0.65]:
        strong = (pred_wp >= thresh) | (pred_wp <= (1 - thresh))
        n_strong = int(strong.sum())
        if n_strong >= 10:
            tier_acc = float((pred_pick[strong] == y_win.values[strong]).mean())
            tiers.append({"threshold": thresh, "n": n_strong, "accuracy": round(tier_acc, 4)})

    return {
        "sport": "MLB",
        "model_type": bundle.get("model_type", ""),
        "n_test": len(df),
        "ml_accuracy": round(accuracy, 4),
        "mae_margin": round(mae, 3),
        "brier_score": round(brier, 4),
        "home_win_rate": round(float(y_win.mean()), 3),
        "confidence_tiers": tiers,
    }
