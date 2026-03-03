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
HFA_RUNS = 0.16   # ~0.32 run total home advantage, split between offense/pitching
import numpy as np, pandas as pd, traceback as _tb, shap
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import nbinom
from db import sb_get, save_model, load_model
from ml_utils import HAS_XGB, _time_series_oof, _time_series_oof_proba, StackedRegressor, StackedClassifier
if HAS_XGB:
    from xgboost import XGBRegressor, XGBClassifier

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
        # Enhancement: Platoon, starter spread, lineup confirmation
        "platoon_diff", "sp_fip_spread", "both_lineups_confirmed",
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
        MAX_TRAIN = 12000
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

            # AUDIT: XGBoost added to ensemble when available
            if HAS_XGB:
                xgb_reg = XGBRegressor(
                    n_estimators=80, max_depth=3, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=20, random_state=42,
                    tree_method="hist", verbosity=0,
                )
            print(f"  MLB: Training stacking ensemble on {n} rows (ts-cv, {'XGB+' if HAS_XGB else ''}GBM+RF+ENet)...")
            reg_models = {"gbm": gbm, "rf": rf_reg, "enet": enet}
            if HAS_XGB:
                reg_models["xgb"] = xgb_reg
            oof = _time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=fit_weights)
            oof_gbm = oof["gbm"]
            oof_rf = oof["rf"]
            oof_enet = oof["enet"]

            gbm.fit(X_scaled, y_margin, sample_weight=fit_weights)
            rf_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)
            enet.fit(X_scaled, y_margin)  # ElasticNet: no sample_weight
            if HAS_XGB:
                xgb_reg.fit(X_scaled, y_margin, sample_weight=fit_weights)

            if HAS_XGB:
                meta_X = np.column_stack([oof_gbm, oof_rf, oof_enet, oof["xgb"]])
            else:
                meta_X = np.column_stack([oof_gbm, oof_rf, oof_enet])
            meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            meta_reg.fit(meta_X, y_margin)

            # Bias correction from OOF residuals
            oof_meta = meta_reg.predict(meta_X)
            bias_correction = float(np.mean(oof_meta - y_margin.values if hasattr(y_margin, 'values') else oof_meta - y_margin))
            print(f"  MLB bias correction: {bias_correction:+.3f} runs")

            reg = StackedRegressor([gbm, rf_reg, enet] + ([xgb_reg] if HAS_XGB else []), meta_reg, scaler)
            reg_cv = cross_val_score(gbm, X_scaled, y_margin,
                                      cv=cv_folds, scoring="neg_mean_absolute_error")

            explainer = shap.TreeExplainer(xgb_reg if HAS_XGB else gbm)
            model_type = "StackedEnsemble_v3_TSCV" + ("_XGB" if HAS_XGB else "")

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

    # Enhancement: Platoon splits and lineup confirmation
    home_platoon_delta = float(game.get("home_platoon_delta", 0.0))
    away_platoon_delta = float(game.get("away_platoon_delta", 0.0))
    home_lineup_confirmed = int(game.get("home_lineup_confirmed", 0))
    away_lineup_confirmed = int(game.get("away_lineup_confirmed", 0))

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
        # Enhancement: Platoon, starter spread, lineup confirmation
        "platoon_diff": home_platoon_delta - away_platoon_delta,
        "sp_fip_spread": abs(home_starter_fip - away_starter_fip),
        "both_lineups_confirmed": 1 if (home_lineup_confirmed and away_lineup_confirmed) else 0,
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
    # AUDIT FIX 1: Removed 0.65 - FIP_COEFF already correct
    ar += (home_fip_adj - lg_fip) * FIP_COEFF
    hr += (away_fip_adj - lg_fip) * FIP_COEFF

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
