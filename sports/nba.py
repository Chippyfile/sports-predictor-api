import numpy as np, pandas as pd, traceback as _tb, shap
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from db import sb_get, save_model, load_model
from dynamic_constants import compute_nba_league_averages, NBA_DEFAULT_AVERAGES
from ml_utils import HAS_XGB, _time_series_oof, _time_series_oof_proba, StackedRegressor, StackedClassifier
if HAS_XGB:
    from xgboost import XGBRegressor, XGBClassifier

def nba_build_features(df):
    """
    NBA feature builder — expanded to use 30+ raw stat columns from nbaSync.
    Mirrors the NCAA pattern: differential features + context + heuristic signal.
    """
    df = df.copy()

    # ── Core heuristic outputs ──
    df["score_diff_pred"] = df["pred_home_score"].fillna(0) - df["pred_away_score"].fillna(0)
    df["total_pred"]      = df["pred_home_score"].fillna(0) + df["pred_away_score"].fillna(0)
    df["home_fav"]        = (pd.to_numeric(df.get("model_ml_home", 0), errors="coerce").fillna(0) < 0).astype(int)
    df["win_pct_home"]    = pd.to_numeric(df.get("win_pct_home", 0.5), errors="coerce").fillna(0.5)
    df["ou_gap"]          = df["total_pred"] - pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else (df["ou_total"] if "ou_total" in df.columns else pd.Series(220, index=df.index)), errors="coerce"
    ).fillna(220)

    # ── Net rating ──
    df["home_net_rtg"] = pd.to_numeric(df.get("home_net_rtg", 0), errors="coerce").fillna(0)
    df["away_net_rtg"] = pd.to_numeric(df.get("away_net_rtg", 0), errors="coerce").fillna(0)
    df["net_rtg_diff"] = df["home_net_rtg"] - df["away_net_rtg"]

    # ── Raw stat differentials (from nbaSync 30+ columns) ──
    # Dynamic league averages (derived from nba_historical, falls back to static)
    _nba_avgs = getattr(nba_build_features, "_league_averages", NBA_DEFAULT_AVERAGES)
    for col_base, default in [
        ("ppg", _nba_avgs.get("ppg", 110)),
        ("opp_ppg", _nba_avgs.get("opp_ppg", 110)),
        ("fgpct", _nba_avgs.get("fgpct", 0.46)),
        ("threepct", _nba_avgs.get("threepct", 0.36)),
        ("ftpct", _nba_avgs.get("ftpct", 0.77)),
        ("assists", _nba_avgs.get("assists", 25)),
        ("turnovers", _nba_avgs.get("turnovers", 14)),
        ("tempo", _nba_avgs.get("tempo", 100)),
        ("orb_pct", _nba_avgs.get("orb_pct", 0.25)),
        ("fta_rate", _nba_avgs.get("fta_rate", 0.28)),
        ("ato_ratio", _nba_avgs.get("ato_ratio", 1.7)),
        ("opp_fgpct", _nba_avgs.get("fgpct", 0.46)),
        ("opp_threepct", _nba_avgs.get("threepct", 0.35)),
        ("steals", _nba_avgs.get("steals", 7.5)),
        ("blocks", _nba_avgs.get("blocks", 5.0)),
    ]:
        h_col = f"home_{col_base}"
        a_col = f"away_{col_base}"
        df[h_col] = pd.to_numeric(df.get(h_col, default), errors="coerce").fillna(default)
        df[a_col] = pd.to_numeric(df.get(a_col, default), errors="coerce").fillna(default)
        df[f"{col_base}_diff"] = df[h_col] - df[a_col]

    # ── Win % differential ──
    h_wins = pd.to_numeric(df.get("home_wins", 0), errors="coerce").fillna(0)
    h_losses = pd.to_numeric(df.get("home_losses", 0), errors="coerce").fillna(0)
    a_wins = pd.to_numeric(df.get("away_wins", 0), errors="coerce").fillna(0)
    a_losses = pd.to_numeric(df.get("away_losses", 0), errors="coerce").fillna(0)
    df["win_pct_diff"] = (h_wins / (h_wins + h_losses).clip(1)) - (a_wins / (a_wins + a_losses).clip(1))

    # ── Form differential ──
    df["form_diff"] = (
        pd.to_numeric(df.get("home_form", 0), errors="coerce").fillna(0) -
        pd.to_numeric(df.get("away_form", 0), errors="coerce").fillna(0)
    )

    # ── Tempo average ──
    df["tempo_avg"] = (df["home_tempo"] + df["away_tempo"]) / 2

    # ── Rest & travel ──
    df["rest_diff"] = (
        pd.to_numeric(df.get("home_days_rest", 2), errors="coerce").fillna(2) -
        pd.to_numeric(df.get("away_days_rest", 2), errors="coerce").fillna(2)
    )
    df["away_travel"] = pd.to_numeric(df.get("away_travel_dist", 0), errors="coerce").fillna(0)

    # ── Turnover quality ──
    df["to_margin_diff"] = df["away_turnovers"] - df["home_turnovers"]
    df["steals_to_h"] = df["home_steals"] / df["home_turnovers"].clip(0.5)
    df["steals_to_a"] = df["away_steals"] / df["away_turnovers"].clip(0.5)
    df["steals_to_diff"] = df["steals_to_h"] - df["steals_to_a"]

    # ── Market line features (strongest public predictor) ──
    df["market_spread"] = pd.to_numeric(df["market_spread_home"] if "market_spread_home" in df.columns else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    df["market_total"] = pd.to_numeric(
        df["market_ou_total"] if "market_ou_total" in df.columns else (df["ou_total"] if "ou_total" in df.columns else pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)
    df["has_market"] = ((df["market_spread"] != 0) | (df["market_total"] != 0)).astype(int)
    # Spread difference: model prediction vs market line (positive = model more bullish on home)
    df["spread_vs_market"] = df["score_diff_pred"] - df["market_spread"]

    feature_cols = [
        # Heuristic signal
        "score_diff_pred", "win_pct_home", "ou_gap",
        # Net rating (primary signal)
        "net_rtg_diff",
        # Offensive differentials
        "ppg_diff", "fgpct_diff", "threepct_diff", "ftpct_diff",
        # Four factors
        "orb_pct_diff", "fta_rate_diff", "ato_ratio_diff",
        # Defensive differentials
        "opp_ppg_diff", "opp_fgpct_diff", "opp_threepct_diff",
        "steals_diff", "blocks_diff",
        # Turnover quality
        "to_margin_diff", "steals_to_diff",
        # Context
        "win_pct_diff", "form_diff", "tempo_avg",
        "rest_diff", "away_travel",
        # Market line signal (Vegas spread is strongest public predictor)
        "market_spread", "market_total", "spread_vs_market", "has_market",
    ]

    return df[feature_cols].fillna(0)


def _nba_season_weight(season):
    current = 2026
    age = current - season
    if age <= 0: return 1.0
    if age == 1: return 1.0
    if age == 2: return 0.9
    if age == 3: return 0.8
    if age == 4: return 0.7
    if age == 5: return 0.6
    return 0.5


def _nba_merge_historical(current_df):
    hist_rows = sb_get(
        "nba_historical",
        "is_outlier_season=eq.false&actual_home_score=not.is.null&select=*&order=season.desc&limit=50000"
    )
    if not hist_rows:
        print("  WARNING: nba_historical empty -- current season only")
        if current_df is not None and len(current_df) > 0:
            return current_df, None, 0
        return pd.DataFrame(), None, 0
    hist_df = pd.DataFrame(hist_rows)
    for col in hist_df.columns:
        if col not in ['home_team', 'away_team', 'game_date', 'id', 'season', 'is_outlier_season', 'home_win', 'result_entered']:
            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
    if "season" in hist_df.columns:
        hist_df["season_weight"] = hist_df["season"].apply(
            lambda s: _nba_season_weight(int(s)) if pd.notna(s) else 0.5
        )
    if current_df is not None and len(current_df) > 0:
        combined = pd.concat([hist_df, current_df], ignore_index=True)
    else:
        combined = hist_df
    weights = combined["season_weight"].fillna(1.0).astype(float).values if "season_weight" in combined.columns else None
    n_hist = len(hist_df)
    n_curr = len(current_df) if current_df is not None else 0
    print(f"  NBA corpus: {n_hist} historical + {n_curr} current = {n_hist + n_curr}")
    return combined, weights, n_hist

def train_nba():
    """NBA model training — stacking ensemble with isotonic calibration."""
    import traceback as _tb
    try:
        rows = sb_get("nba_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Merge with historical corpus (2021-2025)
        df, sample_weights, n_historical = _nba_merge_historical(current_df)
        if len(df) < 10:
            return {"error": "Not enough NBA data", "n": len(df), "n_current": len(current_df)}
        # Derive league averages from historical data
        try:
            _nba_lg = compute_nba_league_averages()
            if _nba_lg:
                nba_build_features._league_averages = _nba_lg
                print(f"  Using dynamic NBA averages ({len(_nba_lg)} stats)")
        except Exception as e:
            print(f"  Dynamic NBA averages failed ({e}), using static")

        # Derive league averages from historical data
        try:
            _nba_lg = compute_nba_league_averages()
            if _nba_lg:
                nba_build_features._league_averages = _nba_lg
                print(f"  Using dynamic NBA averages ({len(_nba_lg)} stats)")
        except Exception as e:
            print(f"  Dynamic NBA averages failed ({e}), using static")

        # Derive league averages from historical data
        try:
            _nba_lg = compute_nba_league_averages()
            if _nba_lg:
                nba_build_features._league_averages = _nba_lg
                print(f"  Using dynamic NBA averages ({len(_nba_lg)} stats)")
        except Exception as e:
            print(f"  Dynamic NBA averages failed ({e}), using static")

        X  = nba_build_features(df)
        y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
        y_win    = (y_margin > 0).astype(int)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n = len(df)
        cv_folds = min(5, n)

        if n >= 200:
            # ── Stacking Ensemble (matches MLB/NCAA architecture) ──
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

            if HAS_XGB:
                xgb_reg = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, random_state=42, tree_method="hist", verbosity=0)
            print(f"  NBA: Training stacking ensemble (ts-cv, {'XGB+' if HAS_XGB else ''}GBM+RF+Ridge)...")
            reg_models = {"gbm": gbm, "rf": rf_reg, "ridge": ridge}
            if HAS_XGB:
                reg_models["xgb"] = xgb_reg
            oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=sample_weights)
            oof_gbm, oof_rf, oof_ridge = oof["gbm"], oof["rf"], oof["ridge"]
            gbm.fit(X_scaled, y_margin)
            rf_reg.fit(X_scaled, y_margin)
            ridge.fit(X_scaled, y_margin)
            if HAS_XGB:
                xgb_reg.fit(X_scaled, y_margin)
            if HAS_XGB:
                meta_X = np.column_stack([oof_gbm, oof_rf, oof_ridge, oof["xgb"]])
            else:
                meta_X = np.column_stack([oof_gbm, oof_rf, oof_ridge])
            meta_reg = Ridge(alpha=1.0)
            meta_reg.fit(meta_X, y_margin)
            reg = StackedRegressor([gbm, rf_reg, ridge] + ([xgb_reg] if HAS_XGB else []), meta_reg, scaler)
            reg_cv = cross_val_score(gbm, X_scaled, y_margin, cv=cv_folds, scoring="neg_mean_absolute_error")
            explainer = shap.TreeExplainer(xgb_reg if HAS_XGB else gbm)
            model_type = "StackedEnsemble_v3_TSCV" + ("_XGB" if HAS_XGB else "")
            meta_weights = meta_reg.coef_.round(4).tolist()
            print(f"  NBA meta weights: {meta_weights}")

            # ── Bias correction ──
            oof_meta = meta_reg.predict(meta_X)
            bias_correction = float(np.mean(oof_meta - y_margin.values))
            print(f"  NBA bias correction: {bias_correction:+.3f} pts")

            # ── Stacked classifier ──
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

            gbm_clf.fit(X_scaled, y_win)
            rf_clf.fit(X_scaled, y_win)
            lr_clf.fit(X_scaled, y_win)

            meta_clf_X = np.column_stack([oof_gbm_p, oof_rf_p, oof_lr_p])
            meta_lr = LogisticRegression(max_iter=1000, C=1.0)
            meta_lr.fit(meta_clf_X, y_win)
            clf = StackedClassifier([gbm_clf, rf_clf, lr_clf], meta_lr)

            # ── Isotonic calibration on OOF classifier probs ──
            oof_stacked_probs = meta_lr.predict_proba(meta_clf_X)[:, 1]
            isotonic = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            isotonic.fit(oof_stacked_probs, y_win.values)
            print(f"  NBA isotonic calibration fitted on {len(oof_stacked_probs)} OOF samples")

        else:
            # Simple models for small data
            reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                             learning_rate=0.1, random_state=42)
            reg.fit(X_scaled, y_margin)
            reg_cv = cross_val_score(reg, X_scaled, y_margin,
                                      cv=min(5, n), scoring="neg_mean_absolute_error")
            clf = CalibratedClassifierCV(
                LogisticRegression(max_iter=1000), cv=min(5, n)
            )
            clf.fit(X_scaled, y_win)
            explainer = shap.TreeExplainer(reg)
            model_type = "GBM"
            bias_correction = 0.0
            isotonic = None
            meta_weights = []

        bundle = {
            "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
            "feature_cols": list(X.columns), "n_train": n,
            "mae_cv": float(-reg_cv.mean()), "model_type": model_type,
            "trained_at": datetime.utcnow().isoformat(),
            "bias_correction": bias_correction,
            "isotonic": isotonic,
            "meta_weights": meta_weights if n >= 200 else [],
        }
        save_model("nba", bundle)
        return {"status": "trained", "n_train": n, "model_type": model_type,
                "mae_cv": round(float(-reg_cv.mean()), 3), "features": list(X.columns),
                "bias_correction": round(bias_correction, 3),
                "meta_weights": meta_weights if n >= 200 else []}

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__,
                "traceback": _tb.format_exc()}

def predict_nba(game: dict):
    bundle = load_model("nba")
    if not bundle:
        return {"error": "NBA model not trained — call /train/nba first"}

    # Build a single-row DataFrame with all features the model might need.
    # FIX (v22): nba_build_features uses df.get("home_ppg", default) etc.
    # When a column is missing, df.get() returns the scalar default instead
    # of a Series, and the subsequent .fillna() crashes with AttributeError.
    # Solution: pre-populate ALL raw stat columns that the feature builder
    # reads so df.get() always returns a Series, not a scalar.
    _RAW_DEFAULTS = {
        "pred_home_score": 110, "pred_away_score": 110,
        "home_net_rtg": 0, "away_net_rtg": 0,
        "win_pct_home": 0.5, "ou_total": 220,
        "model_ml_home": 0, "market_ou_total": 220,
        "market_spread_home": 0,
        "home_ppg": 110, "away_ppg": 110,
        "home_opp_ppg": 110, "away_opp_ppg": 110,
        "home_fgpct": 0.46, "away_fgpct": 0.46,
        "home_threepct": 0.36, "away_threepct": 0.36,
        "home_ftpct": 0.77, "away_ftpct": 0.77,
        "home_assists": 25, "away_assists": 25,
        "home_turnovers": 14, "away_turnovers": 14,
        "home_tempo": 100, "away_tempo": 100,
        "home_orb_pct": 0.25, "away_orb_pct": 0.25,
        "home_fta_rate": 0.28, "away_fta_rate": 0.28,
        "home_ato_ratio": 1.7, "away_ato_ratio": 1.7,
        "home_opp_fgpct": 0.46, "away_opp_fgpct": 0.46,
        "home_opp_threepct": 0.35, "away_opp_threepct": 0.35,
        "home_steals": 7.5, "away_steals": 7.5,
        "home_blocks": 5.0, "away_blocks": 5.0,
        "home_wins": 20, "away_wins": 20,
        "home_losses": 20, "away_losses": 20,
        "home_form": 0, "away_form": 0,
        "home_days_rest": 2, "away_days_rest": 2,
        "away_travel_dist": 0,
    }
    # Merge: game values override defaults
    merged = {**_RAW_DEFAULTS, **game}
    row = pd.DataFrame([merged])

    X = nba_build_features(row)
    X_s = bundle["scaler"].transform(X[bundle["feature_cols"]])

    margin   = float(bundle["reg"].predict(X_s)[0])

    # Apply bias correction if available
    bias = bundle.get("bias_correction", 0.0)
    if bias:
        margin -= bias

    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])

    # Apply isotonic calibration if available
    isotonic = bundle.get("isotonic")
    if isotonic is not None:
        try:
            win_prob = float(isotonic.predict([win_prob])[0])
        except Exception:
            pass

    shap_vals = bundle["explainer"].shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    # Handle both 1D (single sample) and 2D arrays
    sv = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
    shap_out = [
        {"feature": f, "shap": round(float(v), 4), "value": round(float(X[f].iloc[0]), 3)}
        for f, v in zip(bundle["feature_cols"], sv)
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "NBA",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "shap": shap_out,
        "model_meta": {"n_train": bundle.get("n_train"), "mae_cv": bundle.get("mae_cv"),
                       "trained_at": bundle.get("trained_at"),
                       "model_type": bundle.get("model_type", "unknown"),
                       "has_isotonic": isotonic is not None},
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
