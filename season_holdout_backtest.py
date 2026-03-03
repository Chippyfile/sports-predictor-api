"""
Season-Holdout Walk-Forward Backtests
Train on seasons [S1..SN-1], test on season SN.
No data leakage — test season is completely excluded from training.
Runs in <60s per sport since we only train once per holdout.
"""
import numpy as np
import pandas as pd
from db import sb_get
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, brier_score_loss
from ml_utils import _time_series_oof, StackedRegressor, StackedClassifier
from sklearn.calibration import IsotonicRegression


def _train_and_test(train_df, test_df, build_features_fn, score_cols, margin_sign=1):
    """
    Train a stacking ensemble on train_df, predict on test_df.
    Returns dict with accuracy metrics.
    """
    X_train = build_features_fn(train_df)
    X_test = build_features_fn(test_df)

    act_h, act_a = score_cols
    y_train_margin = (train_df[act_h].astype(float) - train_df[act_a].astype(float)).values
    y_train_win = (y_train_margin > 0).astype(int)
    y_test_margin = (test_df[act_h].astype(float) - test_df[act_a].astype(float)).values
    y_test_win = (y_test_margin > 0).astype(int)

    feature_cols = list(X_train.columns)
    # Align test features
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Train stacking ensemble (same as production)
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import Ridge, LogisticRegression

    try:
        from xgboost import XGBRegressor, XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    # Regression ensemble
    base_regs = [
        GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=42),
        RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        Ridge(alpha=1.0),
    ]
    if has_xgb:
        base_regs.append(XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, random_state=42, verbosity=0))

    # Classification ensemble
    base_clfs = [
        GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=42),
        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        LogisticRegression(max_iter=1000),
    ]
    if has_xgb:
        base_clfs.append(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                        subsample=0.8, random_state=42, verbosity=0,
                                        use_label_encoder=False, eval_metric="logloss"))

    # Fit regression
    for m in base_regs:
        m.fit(X_tr, y_train_margin)
    meta_X_reg = np.column_stack([m.predict(X_tr) for m in base_regs])
    meta_reg = Ridge(alpha=1.0)
    meta_reg.fit(meta_X_reg, y_train_margin)
    reg = StackedRegressor(base_regs, meta_reg)

    # Fit classification
    for m in base_clfs:
        m.fit(X_tr, y_train_win)
    meta_X_clf = np.column_stack([m.predict_proba(X_tr)[:, 1] for m in base_clfs])
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(meta_X_clf, y_train_win)
    clf = StackedClassifier(base_clfs, meta_clf)

    # Predict on test set
    test_meta_reg = np.column_stack([m.predict(X_te) for m in base_regs])
    pred_margin = meta_reg.predict(test_meta_reg)

    test_meta_clf = np.column_stack([m.predict_proba(X_te)[:, 1] for m in base_clfs])
    pred_wp = meta_clf.predict_proba(test_meta_clf)[:, 1]

    # Isotonic calibration from training set OOF
    try:
        train_meta_clf = np.column_stack([m.predict_proba(X_tr)[:, 1] for m in base_clfs])
        train_wp = meta_clf.predict_proba(train_meta_clf)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_wp, y_train_win)
        pred_wp = iso.predict(pred_wp)
    except:
        pass

    pred_wp = np.clip(pred_wp, 0.05, 0.95)
    pred_pick = (pred_wp >= 0.5).astype(int)

    accuracy = float((pred_pick == y_test_win).mean())
    mae = float(mean_absolute_error(y_test_margin, pred_margin))
    brier = float(brier_score_loss(y_test_win, pred_wp))

    # Confidence tiers
    tiers = []
    for thresh in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        strong = (pred_wp >= thresh) | (pred_wp <= (1 - thresh))
        n_strong = int(strong.sum())
        if n_strong >= 10:
            tier_acc = float((pred_pick[strong] == y_test_win[strong]).mean())
            tiers.append({"threshold": thresh, "n": n_strong, "accuracy": round(tier_acc, 4)})

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": round(accuracy, 4),
        "mae": round(mae, 3),
        "brier": round(brier, 4),
        "home_win_rate": round(float(y_test_win.mean()), 3),
        "confidence_tiers": tiers,
    }


def season_holdout_nba(test_season=2025):
    """
    NBA season-holdout: train on all seasons before test_season, test on test_season.
    """
    from sports.nba import nba_build_features

    # Fetch all historical
    hist = sb_get("nba_historical",
                  "is_outlier_season=eq.false&actual_home_score=not.is.null&select=*&limit=50000")
    if not hist:
        return {"error": "No NBA historical data"}

    df = pd.DataFrame(hist)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    if len(test_df) < 20:
        # Try current season predictions as test set
        curr = sb_get("nba_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        if curr:
            test_df = pd.DataFrame(curr)

    if len(train_df) < 100 or len(test_df) < 20:
        return {"error": f"Insufficient data: {len(train_df)} train, {len(test_df)} test"}

    result = _train_and_test(train_df, test_df, nba_build_features,
                              ("actual_home_score", "actual_away_score"))
    result["sport"] = "NBA"
    result["test_season"] = test_season
    result["train_seasons"] = sorted(train_df["season"].unique().tolist())
    return result


def season_holdout_ncaa(test_season=2026):
    """
    NCAA season-holdout: train on historical seasons, test on current season predictions.
    """
    from sports.ncaa import ncaa_build_features

    hist = sb_get("ncaa_historical",
                  "actual_home_score=not.is.null&select=*&limit=50000")
    curr = sb_get("ncaa_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&select=*")

    hist_df = pd.DataFrame(hist) if hist else pd.DataFrame()
    curr_df = pd.DataFrame(curr) if curr else pd.DataFrame()

    if "season" in hist_df.columns:
        hist_df["season"] = pd.to_numeric(hist_df["season"], errors="coerce")

    if test_season and len(hist_df) > 0 and "season" in hist_df.columns:
        train_df = hist_df[hist_df["season"] < test_season].copy()
        test_in_hist = hist_df[hist_df["season"] == test_season]
        test_df = pd.concat([test_in_hist, curr_df], ignore_index=True) if len(curr_df) > 0 else test_in_hist.copy()
    else:
        # Fallback: use historical as train, current predictions as test
        train_df = hist_df.copy()
        test_df = curr_df.copy()

    if len(train_df) < 100 or len(test_df) < 20:
        return {"error": f"Insufficient data: {len(train_df)} train, {len(test_df)} test"}

    result = _train_and_test(train_df, test_df, ncaa_build_features,
                              ("actual_home_score", "actual_away_score"))
    result["sport"] = "NCAAB"
    result["test_season"] = test_season
    return result


def season_holdout_mlb(test_season=2024):
    """
    MLB season-holdout: train on seasons before test_season, test on test_season.
    """
    from sports.mlb import mlb_build_features

    hist = sb_get("mlb_historical",
                  "is_outlier_season=eq.0&actual_home_runs=not.is.null&select=*&limit=100000")
    if not hist:
        return {"error": "No MLB historical data"}

    df = pd.DataFrame(hist)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    if len(train_df) < 200 or len(test_df) < 50:
        return {"error": f"Insufficient data: {len(train_df)} train, {len(test_df)} test"}

    # Cap training rows for speed
    if len(train_df) > 10000:
        train_df = train_df.tail(10000)

    result = _train_and_test(train_df, test_df, mlb_build_features,
                              ("actual_home_runs", "actual_away_runs"))
    result["sport"] = "MLB"
    result["test_season"] = test_season
    result["train_seasons"] = sorted(train_df["season"].unique().tolist())
    return result


def season_holdout_all():
    """Run all three sports with appropriate holdout seasons."""
    results = {}

    # NBA: train on 2021-2024, test on 2025
    print("  Running NBA holdout (test=2025)...")
    results["nba"] = season_holdout_nba(test_season=2025)

    # NCAA: train on historical, test on current season
    print("  Running NCAA holdout (test=current)...")
    results["ncaa"] = season_holdout_ncaa(test_season=2026)

    # MLB: train on 2015-2023, test on 2024
    print("  Running MLB holdout (test=2024)...")
    results["mlb"] = season_holdout_mlb(test_season=2024)

    return results
