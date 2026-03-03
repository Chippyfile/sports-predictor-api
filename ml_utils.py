import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, brier_score_loss
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from db import sb_get

def _time_series_oof(models, X, y, df=None, n_splits=3, weights=None):
    n = len(X)
    oof = {name: np.full(n, np.nan) for name in models}
    X_arr = X if isinstance(X, np.ndarray) else X.values
    y_arr = y if isinstance(y, np.ndarray) else y.values
    if df is not None and "game_date" in df.columns:
        sort_idx = pd.to_datetime(df["game_date"], errors="coerce").argsort()
        X_arr, y_arr = X_arr[sort_idx], y_arr[sort_idx]
        w_arr = weights[sort_idx] if weights is not None else None
    else:
        sort_idx = np.arange(n)
        w_arr = weights
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, w_tr = y_arr[train_idx], w_arr[train_idx] if w_arr is not None else None
        for name, model in models.items():
            try:
                if w_tr is not None and hasattr(model, 'fit'):
                    import inspect
                    if 'sample_weight' in inspect.signature(model.fit).parameters:
                        model.fit(X_tr, y_tr, sample_weight=w_tr)
                    else:
                        model.fit(X_tr, y_tr)
                else:
                    model.fit(X_tr, y_tr)
                oof[name][sort_idx[val_idx]] = model.predict(X_val)
            except Exception as e:
                print(f"  [ts-cv] {name} fold error: {e}")
                oof[name][sort_idx[val_idx]] = 0.0
    for name in oof:
        mask = np.isnan(oof[name])
        if mask.any():
            oof[name][mask] = np.nanmean(oof[name][~mask]) if (~mask).any() else 0.0
    return oof

def _time_series_oof_proba(models, X, y, df=None, n_splits=3, weights=None):
    n = len(X)
    oof = {name: np.full(n, np.nan) for name in models}
    X_arr = X if isinstance(X, np.ndarray) else X.values
    y_arr = y if isinstance(y, np.ndarray) else y.values
    if df is not None and "game_date" in df.columns:
        sort_idx = pd.to_datetime(df["game_date"], errors="coerce").argsort()
        X_arr, y_arr = X_arr[sort_idx], y_arr[sort_idx]
        w_arr = weights[sort_idx] if weights is not None else None
    else:
        sort_idx = np.arange(n)
        w_arr = weights
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, w_tr = y_arr[train_idx], w_arr[train_idx] if w_arr is not None else None
        for name, model in models.items():
            try:
                if w_tr is not None and hasattr(model, 'fit'):
                    import inspect
                    if 'sample_weight' in inspect.signature(model.fit).parameters:
                        model.fit(X_tr, y_tr, sample_weight=w_tr)
                    else:
                        model.fit(X_tr, y_tr)
                else:
                    model.fit(X_tr, y_tr)
                oof[name][sort_idx[val_idx]] = model.predict_proba(X_val)[:, 1]
            except Exception as e:
                print(f"  [ts-cv-proba] {name} fold error: {e}")
                oof[name][sort_idx[val_idx]] = 0.5
    for name in oof:
        mask = np.isnan(oof[name])
        if mask.any():
            oof[name][mask] = 0.5
    return oof


# # MLB DISPERSION CALIBRATION
# ═══════════════════════════════════════════════════════════════
# Default MLB overdispersion constant (k). Lower k = more variance / fatter tails.
# Empirically, MLB run scoring fits NegBin with k ≈ 0.55–0.65 per team per game.
# This is calibrated from historical data via /calibrate/mlb and stored separately.
MLB_NEGBIN_K_DEFAULT = 0.60

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

